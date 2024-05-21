"""
This file contains a Metaflow flow specification for generating synthetic data 
based on the provided input file.

The flow consists of several steps:
1. Load annotated proposals and taxonomy data.
2. Process the loaded data, transform dates, and generate synthetic IDs.
3. Generate synthetic project inputs using a pre-trained model.
4. Generate JSON-like objects describing cutting-edge tech projects.

To use this flow, follow these steps:
1. Create an instance of the SyntheticDataGenerator class.
2. Set the `files_to_load` parameter to specify the input file.
3. Run the flow using the `run()` method.

Example usage:
"""

# pylint: skip-file
import logging
import re
import ast
import random
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from metaflow import FlowSpec, step, Parameter, pypi, pypi_base
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

logger = logging.getLogger(__name__)


@pypi_base(
    packages={
        "numpy": "1.26.2",
        "scipy": "1.12.0",
        "scikit-learn": "1.4.1.post1",
        "boto3": "1.34.1",
        "pandas": "2.1.3",
        "pyarrow": "14.0.1",
        "openpyxl": "3.1.2",
        "torch": "2.2.1",
        "transformers": "4.38.2",
        "sentencepiece": "0.2.0",
        "langchain": "0.1.12",
        "langchain-openai": "0.0.8",
    },
    python="3.12.0",
)
class SyntheticDataGenerator(FlowSpec):
    """
    A class that generates synthetic data based on the provided input file.

    Parameters:
    - files_to_load (str): The file to load and process. Default is None.

    Steps:
    - start: Load annotated proposals and taxonomy data.
    - data_processing: Process the loaded data, transform dates, and generate synthetic IDs.
    - synthetic_project_inputs: Generate synthetic project inputs using a pre-trained model.
    - get_synthetic_projects: Generate JSON-like objects describing cutting-edge tech projects.

    Usage:
    1. Create an instance of SyntheticDataGenerator.
    2. Set the `files_to_load` parameter to specify the input file.
    3. Run the flow using the `run()` method.

    Example:
    ```
    generator = SyntheticDataGenerator(file_to_load="data.xlsx")
    generator.run()
    ```
    """

    files_to_load = Parameter(
        "files_to_load",
        help="The file to load and process",
        default=None,
        separator=",",
    )

    @step
    def start(self):
        """
        Load annotated proposals and taxonomy data.
        """
        import os, random
        from getters.s3io import (
            S3DataManager,
        )

        self.id_mapping = {}

        # # fix seeds [these are for the public repo, not the actual ones used]
        os.environ["PYTHONHASHSEED"] = "123"
        random.seed(123)
        np.random.seed(123)

        s3dm = S3DataManager()

        if self.files_to_load is None:
            # list all files in s3 folder eic-case-studies/eic/datathon
            files = s3dm.get_s3_data_paths("datathon", "*.xlsx")

            # load these
            self.df_real = []
            for file in files:
                print(f"Loading {file} ...")
                self.df_real.append([file, s3dm.load_s3_data(file)])
        else:
            # load these
            self.df_real = []
            for file in self.files_to_load:
                print(f"Loading {file} ...")
                self.df_real.append([file, s3dm.load_s3_data(file)])

        self.next(self.data_processing, foreach="df_real")

    @step
    def data_processing(self):
        """
        Process the data by performing various transformations and generating synthetic
            IDs.

        This method performs the following steps:
        1. Drops any columns with "Unnamed" in the name.
        2. Deletes columns that have "CEO" in the name, except for those containing
            "ID" or "Gender" and "Website".
        3. Transforms date columns from five-digit format to datetime.
        4. Generates synthetic IDs for specified columns and stores them in self.id_mapping.
        5. Replaces the original IDs in the dataframe with the synthetic IDs.
        """
        self.file, self.dataframe = self.input[0], self.input[1]
        print(self.file, "- Preprocessing -", f"Processing {self.input[0]} ...")

        # drop any Unnamed column
        self.dataframe = self.dataframe.loc[
            :, ~self.dataframe.columns.str.contains("^Unnamed")
        ]

        # delete any columns that have "CEO" except ID and Gender + Website
        cols_to_drop = [
            col
            for col in self.dataframe.columns
            if "CEO" in col and not any(x in col for x in ["ID", "Gender"])
        ] + [col for col in self.dataframe.columns if "website" in col.lower()]
        self.dataframe.drop(columns=cols_to_drop, inplace=True)

        # transform dates from five digit to datetime
        for col in self.dataframe.columns:
            if "date" in col.lower() or "deadline" in col.lower():
                if (
                    self.dataframe[col]
                    .apply(
                        lambda x: (isinstance(x, float) or isinstance(x, int))
                        and 10000 <= x < 100000
                    )
                    .any()
                ):
                    self.dataframe[col] = pd.to_datetime(
                        self.dataframe[col].apply(
                            lambda x: x if np.isnan(x) else int(x)
                        ),
                        unit="D",
                        origin="1900-01-01",
                    )

            # Generate synthetic IDs for the specified columns and store them in self.id_mapping
            if any(
                keyword in col.lower() for keyword in ["id", "pic", "number"]
            ) and not any(
                keyword in col.lower() for keyword in ["topic", "call", "context"]
            ):
                print(col)
                unique_ids = self.dataframe[col].unique()
                for id in unique_ids:
                    if id not in self.id_mapping:
                        self.id_mapping[id] = self.generate_synthetic_id(str(id))
                        # drop duplicate keys
                reversed_dict = {v: k for k, v in self.id_mapping.items()}
                self.id_mapping = {v: k for k, v in reversed_dict.items()}

                # replace the IDs in the dataframe with the synthetic IDs
                self.dataframe[col] = self.dataframe[col].map(self.id_mapping)

        self.next(self.synthetic_project_inputs)

    @step
    def synthetic_project_inputs(self):
        """
        Generates synthetic project inputs by paraphrasing project abstracts.

        This method uses the T5 model for conditional generation to paraphrase project
        abstracts. It takes each unique project abstract from the dataframe, replaces
        named entities with "ENTITY", and generates a paraphrased version of the abstract
        using the T5 model.
        """
        from transformers import T5ForConditionalGeneration, T5Tokenizer

        tokenizer = T5Tokenizer.from_pretrained("t5-small")
        model = T5ForConditionalGeneration.from_pretrained("t5-small")

        def _process_abstract(abstract):
            anonymised_abstract = re.sub(r"\b[A-Z]\w*\b", "ENTITY", str(abstract))
            inputs = tokenizer.encode("paraphrase: " + anonymised_abstract, return_tensors="pt")
            outputs = model.generate(inputs, max_length=50, num_return_sequences=1, temperature=1.5)
            paraphrase = tokenizer.decode(outputs[0], skip_special_tokens=True)
            return abstract, paraphrase
        print(
            self.file,
            "- Abstract Input Creation - Initialising",
        )
        # get all unique project abstract, date pairs
        abstract_column = next(
            (col for col in self.dataframe.columns if "abstract" in col.lower()), None
        )

        if abstract_column:
            unique_projects = self.dataframe[[abstract_column]].drop_duplicates()

        # self.outputs = []
        # for i, abstract in enumerate(unique_projects[abstract_column]):
        #     print(
        #         self.file,
        #         "- Abstract Input Creation -",
        #         f"Processing {i+1} of {len(unique_projects)}",
        #     )
        #     anonymised_abstract = re.sub(r"\b[A-Z]\w*\b", "ENTITY", str(abstract))
        #     inputs = tokenizer.encode(
        #         "paraphrase: " + anonymised_abstract, return_tensors="pt"
        #     )
        #     outputs = model.generate(
        #         inputs, max_length=50, num_return_sequences=1, temperature=1.5
        #     )

        #     # decode the output tensor
        #     paraphrase = tokenizer.decode(outputs[0], skip_special_tokens=True)
        #     self.outputs.append((abstract, paraphrase))

            # Parallel processing of abstracts
        self.outputs = Parallel(n_jobs=8, verbose=10)(
            delayed(_process_abstract)(abstract) for abstract in unique_projects[abstract_column]
        )

        # now 'outputs' is a list of the generated text for each abstract
        self.next(self.get_synthetic_projects)

    @step
    def get_synthetic_projects(self):
        """
        Generate synthetic projects based on a given prompt using cutting-edge technology.
        The generated projects will follow a specific structure and include fields such as
        Acronym, Title, and Abstract.
        """
        abstract_column = next(
            (col for col in self.dataframe.columns if "abstract" in col.lower()), None
        )
        title_column = next(
            (col for col in self.dataframe.columns if "title" in col.lower()), None
        )
        acronym_column = next(
            (col for col in self.dataframe.columns if "acronym" in col.lower()), None
        )

        template = """

        Generate a JSON-like object describing a cutting-edge tech project in a similar 
        style to the examples below from a short prompt. The examples are given in JSON format, and the output 
        should also follow the same structure. The generated project should embody innovation, 
        ambition, and professionalism. Use clear and concise language, provide specific 
        details about the technology, partnerships, and market potential, and highlight 
        the project's unique impact, funding status, and strategic partnerships. Please remove
        any references to a specific company or product name.

        Examples:
        
        - input: "ENTITY is an AI-based autonomous flight control for the aircraft of today and electric VTOLs"
        - output: {{
            "Acronym": "Machbel",
            "Title": "AI-based autonomous flight control for the aircraft of today and electric VTOLs of tomorrow",
            "Abstract": "We are developing Machbel, a novel AI autonomous flight control system. Machbel is an airframe-agnostic solution targeted at fly-by-wire aircraft of today (fixed-wing airplanes & helicopters) & tomorrow (eVTOLs & others). It is a combination of custom-designed neural networks, core avionics software, computer vision algorithms & special-purpose aviation-grade hardware. We are developing a special kind of neural networks that are deterministic by design, built specifically for the purposes of certification in aviation. They learn while they are getting trained, they don’t learn when they are piloting aircraft. They run in the dedicated environment & on certifiable hardware that we are creating. This is a major innovation over the generic state-of-the-art in AI.Airbus forecasts that the potential demand for eVTOLs in the urban air mobility market will be 100X bigger its current yearly production of helicopters. This market can only grow with full flight autonomy provided by Raven.We have raised €4m+ of funding & already generated €110,000 in early revenues. Our current deal pipeline is worth €2m. As of May 2019 we have signed agreements with 14 most prominent eVTOL OEMs (out of 85+) & have been approached by dozens of others. We are already working with high-profile eVTOL manufacturers such as Volocopter, Lilium Aviation, Kitty Hawk, Dufour Aerospace & multiple others. We also have closed commercial agreements with the avionics vendors, drone OEMs & traditional aircraft OEMs such Euroavionics, AirTractor & Airbus Helicopters. We have also entered a strategic commercial partnership with a major German car manufacturer that wants to enter the eVTOL market & sees Raven as the key enabling solution.Starting with 2020 we forecast steady cashflows with the drones & existing general aviation market segments. To reach a unicorn valuation we are betting on the urban air mobility market segment."
        }}

        - input: "ENTITY is a company committed to reducing the Textile Industry defective production (from 5%) to 0%"
        - output: {{
            "Acronym": "PUTLA",
            "Title": "Detection of defective textile production",
            "Abstract": "PUTLA is a company committed to reducing the Textile Industry defective production (from 5%) to 0%. Our dedicated and highly skilled team has made this possible by developing unique devices, which combine hardware and software built in-house to achieve such a goal. SMARTEX is able to detect defects during the production stage, offering an online software as a service (SaaS) based on Computer Vision, Machine Learning and Artificial Intelligence with the clear mission of becoming the main solution for automation of the inspection processes in the Knitting Textile Industry. Currently, SMARTEX devices are in a prototype stage (TRL 6), since real short-time environment tests have been successfully performed and the market-fit is also validated by multiple LOIs and prices agreed with potential clients, including Decathlon group, PVH (Calvin Klein, Tommy Hilfiger, etc.), Kering (Gucci, YSL), in different continents. SMARTEX’s disruptive hardware & software technology has been recognized with several grants and awards and has recently participated in the biggest hardware startup acceleration programme HAX (Shenzhen, China), backed by Smartex’s first investor SOSV (Sean O’Sullivan Ventures)."
        }}

        - input: "Entity is an innovative autonomous solar drone dedicated to earth observation."
        - output: {{
            "Acronym": "Massimmo",
            "Title": "Earth Observation by Autonomous Solar UAV",
            "Abstract": "MAsla develops an innovative autonomous solar drone dedicated to earth observation. Inspired from satellite earthobservation, MAsla aims to offer affordable earth data acquisition performed by unmmaned autonomous vehicules to arange of end users which have all express their support to the SolarXOne project : Linear infrastructure observation (such asrailway, pipeline or electrical grid), environmental & security surveillance issues (forest fire detection, traffic surveillance),Maritime observation (traffic surveillance, fishing surveillance), precision agriculture (monitoring of the health of crops andlivestock). A first autonomous prototype is ready (TRL6). Thanks to a patented double-wing innovative design, theperformances of SolarXOne are disruptive compared to existing solutions: large payload capacity (7kg) enabling to carry awide range of data acquisition sensors, very stable flight enabling precise data acquisition, long flight (>600 km / Day),cheaper price for end user compared to competitors. A world record of autonomous solar flight will be tempted in the nextmonths. SolarXOne project objectives are to industrialise the drone production with enhance performances. Adaptability toHydrogen energy source will be added as well as vertical take-off capability. Market demonstration will be done during theproject for linear infrastructure, fire detection, maritime surveillance and precision agriculture. Two operating centres will beopened in France and Germany. XSun is based on 2 complementary business models: earth data service commercializationoperated from the control centres and complete system commercialization. Market analysis have been performed for eachsegment and the business plan shows promising revenue reaching 30 M€ in 2025. MAsla has been created in 2016, its team (12 people) is composed of experienced managers and business developers in the aerospace industry as well as young passionate engineers."
        }}

        Make sure your generated project follows the same tone and structure, as JSON files. Include these fields:
        - Acronym
        - Title
        - Abstract

        Go ahead, produce the output for the following example:
        - input: {sentence}
        - output:
        """


        # output = {}
        # for i, (abstract, paraphrase) in enumerate(self.outputs):
        #     print(
        #         self.file,
        #         "- Abstract Generation -",
        #         f"Processing {i+1} of {len(self.outputs)}",
        #     )
        #     while True:
        #         try:
        #             result = chain.invoke(paraphrase)
        #             json_result = ast.literal_eval(result)
        #             output[abstract] = json_result
        #             break
        #         except Exception:
        #             continue

        results = Parallel(n_jobs=8, verbose=10)(
            delayed(self._process_item)(abstract, paraphrase, template) for abstract, paraphrase in self.outputs
        )

        output = {abstract: json_result for abstract, json_result in results}

        # iterate over the DataFrame rows
        for i, row in self.dataframe.iterrows():
            abstract = row[abstract_column]

            # match the abstract to the specific rows in output
            if abstract in output:
                result = output[abstract]
                for key, col in [
                    ["Acronym", acronym_column],
                    ["Title", title_column],
                    ["Abstract", abstract_column],
                ]:
                    try:
                        # replace the values in the acronym, title, and abstract columns
                        self.dataframe.at[i, col] = result.get(
                            key, self.dataframe.at[i, col]
                        )
                    except:
                        pass

        self.next(self.organisation_randomiser)

    @step
    def organisation_randomiser(self):
        """
        Randomizes the names and addresses of organizations based on existing ones in
        the dataframe.

        This method generates random names and addresses based only vaguely on existing
        ones in the dataframe. It takes into account the columns containing the legal
        names and addresses of organizations. If both name and address columns are present,
        it generates random names and addresses together. If only the name column is
        present, it generates random names only.

        Examples:
        - input: ["DAEDALEAN AG", "Wattstr. 11"]
            output: {"name": "Schwarz GK", "address": "WILHELMINA VAN PRUISENWEG 35"}

        - input: ["FEOPS NV", "WILHELMINA VAN PRUISENWEG 35"]
            output: {"name": "Klein AG", "address": "Frederik Strasse 1"}
        """
        name_cols = [
            col for col in self.dataframe.columns if "legal" and "name" in col.lower()
        ]
        address_cols = [
            col for col in self.dataframe.columns if "address" in col.lower()
        ]

        if name_cols and address_cols:
            unique_inputs = (
                self.dataframe[name_cols + address_cols]
                .drop_duplicates()
                .values.tolist()
            )

            template = """

            Generate random names and addresses based only vaguely on existing ones.
        

            Examples:

            - input: ["DAEDALEAN AG", "Wattstr. 11"]
            - output: {{"name":"Schwarz GK", "address": "WILHELMINA VAN PRUISENWEG 35"}}

            - input: ["FEOPS NV", "WILHELMINA VAN PRUISENWEG 35"]
            - output: {{"name":"Klein AG", "address": "Frederik Strasse 1"}}

            Go ahead, produce the output for the following example:
            - input: {prompt_input}
            - output:
            """
        else:
            unique_inputs = self.dataframe[name_cols].drop_duplicates().values.tolist()

            template = """

            Generate random names and addresses based only vaguely on existing ones.
        

            Examples:
            
            - input: "DAEDALEAN AG"
            - output: {{"name":"Schwarz GK"}}

            - input: "FEOPS NV"
            - output: {{"name":"Klein AG"}}

            Go ahead, produce the output for the following example:
            - input: {prompt_input}
            - output:
            """

        prompt = ChatPromptTemplate.from_template(template)
        model = ChatOpenAI()

        chain = (
            {"prompt_input": RunnablePassthrough()} | prompt | model | StrOutputParser()
        )

        output_names = {}
        output_addresses = {}
        for i, prompt_input in enumerate(unique_inputs):
            print(self.file, "- Orgs -", f"Processing {i+1} of {len(unique_inputs)}")
            while True:
                try:
                    result = chain.invoke(prompt_input)
                    json_result = ast.literal_eval(result)
                    output_names[prompt_input[0]] = json_result.get("name", "")
                    if len(prompt_input) > 1:
                        output_addresses[prompt_input[1]] = json_result.get(
                            "address", ""
                        )
                    break
                except (ValueError, KeyError, SyntaxError):
                    continue

        # try to map cols to results
        for col in name_cols:
            self.dataframe[col] = self.dataframe[col].map(output_names)
        if address_cols:
            for col in address_cols:
                self.dataframe[col] = self.dataframe[col].apply(lambda x: output_addresses.get(x, ""))

        self.next(self.zip_randomiser)

    @step
    def zip_randomiser(self):
        """
        Randomizes the digits in a column containing zipcodes.

        This method identifies a column in the dataframe that includes a zipcode and
        randomizes the digits in that column. It searches for a column with the word
        "zipcode" (case-insensitive) in its name. If such a column is found, it applies
        the `randomize_zipcode` function to each value in that column.

        The `randomize_zipcode` function randomly selects a digit in the zipcode and
        replaces it with a random digit from 0 to 9. The modified zipcode is then
        assigned back to the dataframe.

        After randomizing the zipcodes, the method proceeds to the `shuffle_amounts`
        step.
        """
        # identify a possible column that includes a zipcode
        zipcode_col = next(
            (col for col in self.dataframe.columns if "zipcode" in col.lower()), None
        )

        if zipcode_col:
            # identify any numerics and change one random number
            def randomize_zipcode(zipcode):
                if re.search(r"\d", zipcode):
                    random_index = random.choice(
                        [m.start() for m in re.finditer(r"\d", zipcode)]
                    )
                    new_digit = str(random.randint(0, 9))
                    return (
                        zipcode[:random_index] + new_digit + zipcode[random_index + 1 :]
                    )
                return zipcode

            self.dataframe[zipcode_col] = self.dataframe[zipcode_col].apply(
                randomize_zipcode
            )

        self.next(self.shuffle_amounts)

    @step
    def shuffle_amounts(self):
        """
        Shuffles the values in the amount-related columns of the dataframe.

        This method shuffles the values in the columns that contain the words "amount",
        "grant", or "costs" in their names. It uses numpy's random permutation function
        to shuffle the unique values in each column and then maps the old values to the
        new shuffled values in the dataframe.

        """
        amount_cols = [
            col
            for col in self.dataframe.columns
            if "amount" in col.lower()
            or "grant" in col.lower()
            or "costs" in col.lower()
        ]
        # block reassignment
        for col in amount_cols:
            unique_values = self.dataframe[col].unique()
            shuffled_values = np.random.permutation(unique_values)
            mapping_dict = dict(zip(unique_values, shuffled_values))

            # Map the old values to the new shuffled values
            self.dataframe[col] = self.dataframe[col].map(mapping_dict)

        self.next(self.shuffle_status)

    @step
    def shuffle_status(self):
        """
        Shuffles the values in columns containing the word "status" in their name.

        This method shuffles the unique values in each column that contains the word "status"
        in its name. It uses numpy's random permutation function to generate a new order for
        the unique values,and then maps the old values to the new shuffled values in the
        dataframe.

        After shuffling the status columns, the method proceeds to the next step, which is
        saving the dataframe to S3.

        """
        status_cols = [col for col in self.dataframe.columns if "status" in col.lower()]

        # block reassignment
        for col in status_cols:
            unique_values = self.dataframe[col].unique()
            shuffled_values = np.random.permutation(unique_values)
            mapping_dict = dict(zip(unique_values, shuffled_values))

            # Map the old values to the new shuffled values
            self.dataframe[col] = self.dataframe[col].map(mapping_dict)

        self.next(self.save_to_s3)

    @step
    def save_to_s3(self):
        """Saves the dataframe and id_mapping to S3."""
        import os
        from getters.s3io import (
            S3DataManager,
        )

        s3dm = S3DataManager()

        # remove datathon/synthetic from filename
        self.file = self.file.split("/")[-1]

        s3dm.save_to_s3(
            self.dataframe,
            f"datathon/synthetic/{self.file}",
        )

        self.json_file = os.path.splitext(self.file)[0] + ".json"
        self.id_mapping = {str(k): v for k, v in self.id_mapping.items()}
        s3dm.save_to_s3(
            self.id_mapping,
            f"datathon/map/{self.json_file}",
        )

        self.next(self.join)

    @step
    def join(self, inputs):
        self.next(self.end)

    @step
    def end(self):
        """
        End of the flow.
        """
        pass

    @staticmethod
    def generate_synthetic_id(input_string):
        """
        Generates a synthetic ID by replacing sequences of digits in the input string
        with random digits.

        Args:
            input_string (str): The input string containing sequences of digits.

        Returns:
            str: The input string with sequences of digits replaced by random digits.

        Example:
            >>> generate_synthetic_id("ABC123XYZ789")
            'ABC456XYZ123'
        """

        digit_sequences = re.findall(r"\d+", input_string)

        for sequence in digit_sequences:
            new_digits = "".join(
                str(random.randint(0, 9)) for _ in range(len(sequence))
            )
            new_sequence = sequence.replace(sequence, new_digits, 1)
            input_string = input_string.replace(sequence, new_sequence, 1)

        return input_string
    
    @staticmethod
    def _process_item(abstract, paraphrase, template):

        prompt = ChatPromptTemplate.from_template(template)
        model = ChatOpenAI()

        chain = {"sentence": RunnablePassthrough()} | prompt | model | StrOutputParser()
        while True:
            try:
                result = chain.invoke(paraphrase)
                json_result = ast.literal_eval(result)
                return abstract, json_result
            except Exception:
                continue
    @staticmethod
    def _process_input(prompt_input, template):

        prompt = ChatPromptTemplate.from_template(template)
        model = ChatOpenAI()

        chain = (
            {"prompt_input": RunnablePassthrough()} | prompt | model | StrOutputParser()
        )
        while True:
            try:
                result = chain.invoke(prompt_input)
                json_result = ast.literal_eval(result)
                name = json_result.get("name", "")
                address = json_result.get("address", "") if len(prompt_input) > 1 else ""
                return prompt_input, name, address
            except Exception:
                continue



if __name__ == "__main__":
    SyntheticDataGenerator()
