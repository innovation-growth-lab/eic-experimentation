"""This module contains the OAIValidationFlow class for the OAI validation flow.

    The OAIValidationFlow class is a flow for topic distribution analysis. 
    It performs topic distribution analysis by matching DBPedia tags to a 
    comprehensive list of topics generated by Leiden University. It ensures 
    the highest level of precision in topic relevance.

    The flow consists of the following steps:
        1. start: Loads data from S3 and prepares the vectors.
        2. prepare_vectors: Prepares the vectors for analysis.
        3. create_vectorstore: Creates a vector store for the topics.
        4. invoke_chain: Invokes the topic distribution analysis chain.
        5. end: Prints a message indicating the flow has finished.

    Example:
        $ python -m eic_case_studies.pipeline.cs2.oai_validation_flow 
            --environment pypi run --save_to_s3 True

"""

# pylint: skip-file
import os, ast
import pandas as pd
from metaflow import FlowSpec, step, Parameter, pypi_base
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings


@pypi_base(
    packages={
        "boto3": "1.34.1",
        "pandas": "2.1.3",
        "pyarrow": "14.0.1",
        "openpyxl": "3.1.2",
        "langchain": "0.1.12",
        "langchain-openai": "0.0.8",
        "faiss-cpu": "1.8.0",
    },
    python="3.12.0",
)
class OAIValidationFlow(FlowSpec):
    """
    A flow for topic distribution analysis.

    This flow performs topic distribution analysis by matching DBPedia tags to a comprehensive list of topics generated by Leiden University. It ensures the highest level of precision in topic relevance.

    Steps:
    1. start: Loads data from S3 and prepares the vectors.
    2. prepare_vectors: Prepares the vectors for analysis.
    3. create_vectorstore: Creates a vector store for the topics.
    4. invoke_chain: Invokes the topic distribution analysis chain.
    5. end: Prints a message indicating the flow has finished.
    """

    save_to_s3 = Parameter(
        "save_to_s3",
        help="Whether to save the data to S3.",
        default=False,
    )

    are_assigned = Parameter(
        "are_assigned",
        help="Whether the proposals are assigned a topic.",
        default=True,
    )

    join_topic_tags = Parameter(
        "join_topic_tags",
        help="Whether to join the topic tags.",
        default=False,
    )

    @step
    def start(self):
        """
        Load data.
        """
        from getters.s3io import (
            S3DataManager,
        )

        s3dm = S3DataManager()
        if self.are_assigned:
            self.annotated_proposals = s3dm.load_s3_data(
                "data/03_primary/he_2020/pathfinder/proposals/main_dbp_assigned.parquet"
            )
        else:
            self.annotated_proposals = s3dm.load_s3_data(
                "data/02_intermediate/he_2020/pathfinder/proposals/main_dbp_embeddings.parquet"
            )

        # sort by proposal number
        self.annotated_proposals = self.annotated_proposals.sort_values(
            by="proposal_number"
        )

        self.cwts_taxonomy = s3dm.load_s3_data("data/01_raw/cwts/cwts_oa_topics.xlsx")
        self.cwts_taxonomy["keywords"] = self.cwts_taxonomy["keywords"].str.split(";")
        self.next(self.prepare_filter_keywords)

    @step
    def prepare_filter_keywords(self):
        """
        Filters the DBPedia keywords.

        """
        self.dbpedia_inputs = list(set(self.annotated_proposals["keywords"].to_list()))
        self.dbpedia_inputs = [
            self.dbpedia_inputs[i : i + 4]
            for i in range(0, len(self.dbpedia_inputs), 4)
        ]

        self.next(self.invoke_filter_chain)

    @step
    def invoke_filter_chain(self):
        """
        Invokes the filter chain.

        """

        template = """
        Given a DBPedia tag extracted from a research proposal, your task is to assess whether this tag is likely to denote a relevant discipline, especially within the context of deep technology research. Deep technology (deep tech) encompasses advanced and highly specialized areas of technology that are at the forefront of innovation, including but not limited to artificial intelligence, biotechnology, robotics, quantum computing, and advanced materials science. 

        For each DBPedia tag:

        1. Disciplinary Relevance: Consider if the tag directly refers to, or is strongly associated with, specific fields of study, technological areas, or domains of scientific research. Evaluate the tag's relevance to deep tech disciplines or any other academic fields.
        2. Exclusion Criteria: Identify tags that primarily reflect societal claims, narrative elements, organizational types, or general descriptors lacking in disciplinary specificity. Tags that do not directly contribute to understanding the disciplinary focus of the proposal should be flagged for exclusion.
        3. Response Format: For each DBPedia tag, provide a judgment of "Relevant" if the tag likely denotes a discipline or field of study pertinent to deep tech or other academic areas; otherwise, respond with "Exclude" for tags that do not meet this criterion.

        Example Input:

        DBPedia Tags: "Machine Learning", "Revolutionary", "Consortium", "Quantum Computing"

        Example Output (as a JSON file):

        "Machine Learning": "Relevant", "Revolutionary": "Exclude", "Consortium": "Exclude", "Quantum Computing": "Relevant"

        Do the same with the following example:
        DBPedia Tag: {question}

        Output:
        """

        prompt = ChatPromptTemplate.from_template(template)
        model = ChatOpenAI()

        chain = {"question": RunnablePassthrough()} | prompt | model | StrOutputParser()

        output = []
        for i, dbpedia_input in enumerate(self.dbpedia_inputs):
            print(f"Processing {i+1} of {len(self.dbpedia_inputs)}")
            while True:
                try:
                    result = chain.invoke(dbpedia_input)
                    json_result = ast.literal_eval(result)
                    output.append(json_result)
                    break
                except (ValueError, KeyError, SyntaxError):
                    continue
        # flatten output as a single dictioanry
        output = {k: v for o in output for k, v in o.items()}

        # print count of relevant and irrelevant keywords
        print(f"Relevant: {len([k for k, v in output.items() if v == 'Relevant'])}")
        print(f"Exclude: {len([k for k, v in output.items() if v == 'Exclude'])}")

        self.relevant_keywords = [k for k, v in output.items() if v == "Relevant"]

        self.next(self.subset_keywords)

    @step
    def subset_keywords(self):
        """
        Subsets the DBPedia keywords.

        """
        # loc filter dbpedia keywords that are not Relevant
        self.annotated_proposals = self.annotated_proposals[
            self.annotated_proposals["keywords"].isin(self.relevant_keywords)
        ]

        self.next(self.prepare_vectors)

    @step
    def prepare_vectors(self):
        """
        Prepares the vectors for analysis.

        - Sets the OpenAI API key.
        - Prepares the DBPedia vectors.
        - Prepares the topic vectors.

        """

        self.vector_topics = (
            self.cwts_taxonomy[
                [
                    "topic_id",
                    "topic_name",
                    "subfield_name",
                    "field_name",
                    "domain_name",
                    "keywords",
                ]
            ]
            .apply(
                lambda x: f"Topic ID: {x[0]} - Topic Name: {x[1]} - Subfield Name: {x[2]} - Field Name: {x[3]} - Domain Name: {x[4]} - Keywords: {', '.join(x[5])}",
                axis=1,
            )
            .to_list()
        )
        self.next(self.invoke_topic_chain)


    @step
    def invoke_topic_chain(self):
        """
        Invokes the topic distribution analysis chain.

        - Defines the template for the analysis.
        - Sets up the chain for analysis.
        - Invokes the chain.

        """
        if not self.join_topic_tags:
            template = """
            Given a list of concepts extracted from DBPedia relevant to a research proposal, each concept (tag) is to be analyzed in the context of a comprehensive list of topics generated by Leiden University. These topics are represented by a set of 10 keywords each, encompassing a total of 4516 distinct topics. Your task is to match each DBPedia tag to the most appropriate topic ID from the provided list, ensuring the highest level of precision in topic relevance.

            For each DBPedia tag provided:

            1. Contextual Understanding: Briefly describe the core essence of the DBPedia tag, considering its potential relevance to research domains.
            2. Topic Analysis: Examine the list of keywords associated with each of the 4516 topics. Identify the topic whose keywords resonate most closely with the essence and domain relevance of the DBPedia tag.
            3. Precision Priority: If a highly relevant match is not evident, it is imperative to choose \"null\" rather than suggesting a close but incorrect match. The goal is to maintain the utmost precision, favoring no assignment over an inaccurate one.
            4. Response Format: For each DBPedia tag analyzed, provide the most relevant topic ID or \"null\" if no satisfactory match is found. Ensure your analysis is rooted in the provided keywords' context and relevance to the DBPedia tag.
            Make sure you are as conservative as possible, high precision is the goal here. If nno topic is a particularly good match, simply return "topic_id": \"null\". 
            
            Input:

            DBPedia Tag: "Machine Learning"
            Context: [Topic ID 1 - Topic Name: Geochronological Evolution - Subfield_name: Geophysics - Field_name: Earth and Planetary Sciences - Domain_name: Physical sciences - Keywords: Zircon, Geeochronology, Tectonics, Granitic Rocks, etc., ..., Topic ID 4516: ...]

            Output as a JSON:

            "topic_id": 1234 

            Input:

            DBPedia Tag: "Revolutionary"
            Context: [Topic ID: 10003 - Topic Name: Knowledge Management and Organizational Innovation - Subfield Name: Strategy and Management - Field Name: Business, Management and Accounting - Domain Name: Social Sciences - Keywords: Dynamic Capabilities,  Knowledge Transfer,, ..., Topic ID 4516: ...]

            Example Output as a JSON:

            "topic_id": "null" 


            Let's do it with the following tag and relevant context. Please do not output anything other than the JSON "topic_id" and ID pair.
            
            Input:

            DBPedia Tag: {question}
            Context: {context}

            Output:
            """
        else:
            template = """
            Given a list of concepts extracted from DBPedia relevant to a research proposal, each concept (tag) is to be analyzed in the context of a comprehensive list of topics generated by Leiden University. These topics are represented by a set of 10 keywords each, encompassing a total of 4516 distinct topics. Your task is to match the joined list of DBPedia tags to the most appropriate topic IDs from the provided list, ensuring the highest level of precision in topic relevance. Include between 3 and 10 topics.

            For each DBPedia tag provided:

            1. Contextual Understanding: Briefly describe the core essence of the series of DBPedia tags, considering its potential relevance to research domains.
            2. Topic Analysis: Examine the list of keywords associated with each of the 4516 topics. Identify the topic whose keywords resonate most closely with the essence and domain relevance of the DBPedia tags.
            3. Precision Priority: If a highly relevant match is not evident, it is imperative to choose \"null\" rather than suggesting a close but incorrect match. The goal is to maintain the utmost precision, favoring no assignment over an inaccurate one.
            4. Response Format: For each set of DBPedia tags analyzed, provide the list of most relevant topic IDs or \"null\" if no satisfactory match is found. Ensure your analysis is rooted in the provided keywords' context and relevance to the DBPedia tag.
            Make sure you are as conservative as possible, high precision is the goal here. If nno topic is a particularly good match, simply return "topic_id": \"null\". 
            
            Input:

            DBPedia Tag: "Machine Learning, Artificial Intelligence, Robotics, Quantum Computing"
            Context: [Topic ID 1 - Topic Name: Geochronological Evolution - Subfield_name: Geophysics - Field_name: Earth and Planetary Sciences - Domain_name: Physical sciences - Keywords: Zircon, Geeochronology, Tectonics, Granitic Rocks, etc., ..., Topic ID 4516: ...]

            Output as a JSON:

            "topic_id": [1234, 3893, 30293] 

            Input:

            DBPedia Tag: "Revolutionary, Plants, Footwear, Quantum Computing"
            Context: [Topic ID: 10003 - Topic Name: Knowledge Management and Organizational Innovation - Subfield Name: Strategy and Management - Field Name: Business, Management and Accounting - Domain Name: Social Sciences - Keywords: Dynamic Capabilities,  Knowledge Transfer,, ..., Topic ID 4516: ...]

            Example Output as a JSON:

            "topic_id": "null" 


            Let's do it with the following tag and relevant context. Please do not output anything other than the JSON "topic_id" and list of IDs.
            
            Input:

            DBPedia Tag: {question}
            Context: {context}

            Output:
            """

        vectorstore = FAISS.from_texts(
            self.vector_topics, embedding=OpenAIEmbeddings()
        )
        retriever = vectorstore.as_retriever()

        if not self.join_topic_tags:
            self.dbpedia_inputs = list(set(self.annotated_proposals["keywords"].to_list()))

        else:
            # groupby proposal number and create a list of keywords
            self.dbpedia_inputs = (
                self.annotated_proposals.groupby("proposal_number")["keywords"]
                .apply(list)
                .reset_index()
            )

            # join the list of keywords into a single string
            self.dbpedia_inputs["keywords"] = self.dbpedia_inputs["keywords"].apply(
                lambda x: ", ".join(x)
            )

        prompt = ChatPromptTemplate.from_template(template)
        model = ChatOpenAI()

        chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | model
            | StrOutputParser()
        )

        output = []
        if not self.join_topic_tags:
            for i, dbpedia_input in enumerate(self.dbpedia_inputs):
                print(f"Processing {i+1} of {len(self.dbpedia_inputs)}")
                while True:
                    try:
                        result = chain.invoke(dbpedia_input)
                        json_result = ast.literal_eval(result)
                        output.append({dbpedia_input: json_result["topic_id"]})
                        break
                    except (ValueError, KeyError, SyntaxError):
                        continue
            # flatten output as a single dictioanry
            output = {k: v for o in output for k, v in o.items()}

            self.annotated_proposals["oa_topic_id"] = self.annotated_proposals[
                "keywords"
            ].map(output)

            # clean the oa_topic_id column
            self.annotated_proposals["oa_topic_id"] = self.annotated_proposals[
                "oa_topic_id"
            ].astype(str)

        else:
            # iterrows to get proposal number and keywords
            for i, dbpedia_input in self.dbpedia_inputs.iterrows():
                print(f"Processing {i+1} of {len(self.dbpedia_inputs)}")
                while True:
                    try:
                        result = chain.invoke(dbpedia_input["keywords"])
                        try:
                            json_result = ast.literal_eval(result)
                        except:
                            json_result = ast.literal_eval("{" + result + "}")
                        output.append({dbpedia_input["proposal_number"]: json_result["topic_id"]})
                        break
                    except (ValueError, KeyError, SyntaxError):
                        continue
            # create a single flattened dictionary
            output = {k: v for o in output for k, v in o.items()}

            # create pandas dataframe keeping the value list as a single column
            self.annotated_proposals = pd.DataFrame(output.items(), columns=["proposal_number", "oa_topic_id_joined"])

        # trasnform all items in oa_topic_id_joined lists to strings
        self.annotated_proposals["oa_topic_id_joined"] = self.annotated_proposals["oa_topic_id_joined"].apply(lambda x: [str(i) for i in x])
            

        self.next(self.save_results)

    @step
    def save_results(self):
        """
        Save the annotated dataset to S3 or local file system.
        """
        from getters.s3io import (
            S3DataManager,
        )

        if self.save_to_s3:
            s3dm = S3DataManager()
            if not self.join_topic_tags:
                s3dm.save_to_s3(
                    self.annotated_proposals,
                    "data/03_primary/he_2020/pathfinder/proposals/main_dbp_oa_validated.parquet",
                )
            else:
                s3dm.save_to_s3(
                    self.annotated_proposals,
                    "data/03_primary/he_2020/pathfinder/proposals/main_dbp_oa_joined.parquet",
                )

        self.next(self.end)

    @step
    def end(self):
        """
        Prints a message indicating the flow has finished.

        """
        print("Flow finished.")


if __name__ == "__main__":
    OAIValidationFlow()
