"""This module contains the flow to encode the annotated keywords in the 
    Pathfinder proposals and CWTS taxonomy datasets using the SentenceTransformer 
    model.

The flow consists of the following steps:
    1. Load data and initialize NLP model.
    2. Annotate the "Proposal Title" column.
    3. Encode the annotated keywords.
    4. Explode the keywords in all three lists and map embeddings to the 
        corresponding keywords.
    5. Save the annotated dataset to S3 or local file system.
    6. End the flow.

    Example:
        $ python -m eic_case_studies.pipeline.cs2.specter_embedding_flow 
            --environment pypi run --save_to_s3 True

"""

# pylint: skip-file
from metaflow import FlowSpec, step, Parameter, pypi_base
import pandas as pd
from requests.adapters import HTTPAdapter, Retry
from functools import partial
from itertools import chain
from toolz import pipe


@pypi_base(
    packages={
        "boto3": "1.34.1",
        "pandas": "2.1.3",
        "openpyxl": "3.1.2",
        "pyarrow": "14.0.1",
        "sentence-transformers": "2.5.1",
        "toolz": "0.12.0",
    },
    python="3.12.0",
)
class EmbeddingEncoder(FlowSpec):
    """
    A flow to encode the annotated keywords in the Pathfinder proposals
        and CWTS taxonomy datasets.
    """

    save_to_s3 = Parameter(
        "save_to_s3",
        help="Whether to save the data to S3.",
        default=False,
    )

    @step
    def start(self):
        """
        Load data and initialize NLP model.
        """
        from sentence_transformers import SentenceTransformer
        from getters.s3io import (
            S3DataManager,
        )

        # create an instance of the S3DataManager class and load data
        s3dm = S3DataManager()
        self.annotated_proposals = s3dm.load_s3_data(
            "data/02_intermediate/he_2020/pathfinder/proposals/main_dbp_annotated.parquet"
        )[["proposal_number", "title_annotations", "abstract_annotations"]]

        self.cwts_taxonomy = s3dm.load_s3_data("data/01_raw/cwts/cwts_oa_topics.xlsx")

        # instantiate the SentenceTransformer model
        self.encoder = SentenceTransformer("sentence-transformers/allenai-specter")

        self.next(self.prepare_data)

    @step
    def prepare_data(self):
        """
        Annotate the "Proposal Title" column.
        """
        # proposal keywords
        annotated_titles = self._get_annotation_list(
            self.annotated_proposals, "title_annotations"
        )
        annotated_abstracts = self._get_annotation_list(
            self.annotated_proposals, "abstract_annotations"
        )
        self.annotated_keywords = annotated_titles.union(annotated_abstracts)

        # cwts keywords
        self.cwts_taxonomy["keywords"] = self.cwts_taxonomy["keywords"].str.split(";")
        self.cwts_taxonomy["keywords"] = self.cwts_taxonomy["keywords"].apply(
            lambda x: [keyword.strip().lower() for keyword in x]
        )
        self.cwts_keywords = self._get_annotation_list(self.cwts_taxonomy, "keywords")

        self.next(self.encode_keywords)

    @step
    def encode_keywords(self):
        """
        Encode the annotated keywords.
        """
        self.annotated_keyword_embeddings = {
            keyword: self.encoder.encode(keyword, convert_to_tensor=False)
            for keyword in self.annotated_keywords
        }
        self.cwts_keyword_embeddings = {
            keyword: self.encoder.encode(keyword, convert_to_tensor=False)
            for keyword in self.cwts_keywords
        }
        self.next(self.merge_back_and_explode)

    @step
    def merge_back_and_explode(self):
        """
        Explode the keywords in all three lists and map embeddings to the
        corresponding keywords.
        """

        print("here")

        # add the two columns as a single set after chaining sublists
        self.annotated_proposals["keywords"] = self.annotated_proposals[
            ["title_annotations", "abstract_annotations"]
        ].apply(
            lambda x: list(
                set(list(x["title_annotations"]) + list(x["abstract_annotations"]))
            ),
            axis=1,
        )

        self.annotated_proposals = self.annotated_proposals[
            ["proposal_number", "keywords"]
        ]

        # explode the keywords
        self.annotated_proposals = self.annotated_proposals.explode("keywords")
        self.cwts_taxonomy = self.cwts_taxonomy.explode("keywords")

        # map the embeddings to each keyword in both datasets
        self.annotated_proposals["keyword_embedding"] = self.annotated_proposals[
            "keywords"
        ].map(self.annotated_keyword_embeddings)

        self.cwts_taxonomy["keyword_embedding"] = self.cwts_taxonomy["keywords"].map(
            self.cwts_keyword_embeddings
        )

        self.next(self.save_results)

    @step
    def save_results(self):
        """
        Save the annotated dataset to S3 or local file system.
        """
        from getters.s3io import (
            S3DataManager,
        )

        self.annotated_proposals = self.annotated_proposals[
            ~self.annotated_proposals["keyword_embedding"].isna()
        ]
        

        if self.save_to_s3:
            s3dm = S3DataManager()
            s3dm.save_to_s3(
                self.annotated_proposals,
                "data/02_intermediate/he_2020/pathfinder/proposals/main_dbp_embeddings.parquet",
            )
            s3dm.save_to_s3(
                self.cwts_taxonomy,
                "data/02_intermediate/cwts/main_dbp_embeddings.parquet",
            )

        self.next(self.end)

    @step
    def end(self):
        """
        End the flow.
        """
        pass

    def _get_annotation_list(self, dataframe: pd.DataFrame, column: str) -> list:
        """
        Get a list of unique annotations from a specified column in a dataframe.

        Args:
            dataframe (pandas.DataFrame): The dataframe containing the annotations.
            column (str): The name of the column containing the annotations.

        Returns:
            list: A list of unique annotations.

        """
        return pipe(dataframe[column].to_list(), chain.from_iterable, set)


if __name__ == "__main__":
    EmbeddingEncoder()
