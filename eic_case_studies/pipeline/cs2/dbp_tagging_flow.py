"""This module contains the DBpediaAnnotationFlow class, which is 
    a Metaflow flow to tag the Pathfinder proposals with DBpedia 
    annotations. The DBpedia Spotlight API is used to annotate the
    text data in the "Proposal Title" and "Proposal Abstract" columns
    of the Pathfinder proposals dataset. The annotations are retrieved
    in JSON format and stored in new columns in the dataset.

Example:
    To run the DBpediaAnnotationFlow flow, use the following command:

        $ python -m eic_case_studies.pipeline.cs2.dbp_tagging_flow 
            --environment pypi run --save_to_s3 True
"""

# pylint: skip-file
from metaflow import FlowSpec, step, Parameter, pypi_base
import spacy
import requests
from requests.adapters import HTTPAdapter, Retry
from functools import partial


@pypi_base(
    packages={
        "requests": "2.31.0",
        "spacy": "3.7.4",
        "boto3": "1.34.1",
        "pandas": "2.1.3",
        "openpyxl": "3.1.2",
        "pyarrow": "14.0.1",
    },
    python="3.12.0",
)
class DBpediaAnnotationFlow(FlowSpec):
    """
    A flow to tag Pathfinder proposals with DBpedia annotations.
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
        import pandas as pd
        from getters.s3io import (
            S3DataManager,
        )

        # download and load Spacy English model
        spacy.cli.download("en_core_web_md")
        self.nlp = spacy.load("en_core_web_md")

        # create an instance of the S3DataManager class and load data
        s3dm = S3DataManager()
        self.pathfinder_he_proposals = s3dm.load_s3_data(
            "data/01_raw/he_2020/pathfinder/proposals/main_he.xlsx",
            skiprows=3,
            usecols=lambda x: "Unnamed" not in x,
        )
        self.pathfinder_h2020_proposals = s3dm.load_s3_data(
            "data/01_raw/he_2020/pathfinder/proposals/main_h2020.xlsx",
            skiprows=3,
            usecols=lambda x: "Unnamed" not in x,
        )

        # concatenate the two datasets
        self.pathfinder_proposals = pd.concat(
            [self.pathfinder_he_proposals, self.pathfinder_h2020_proposals]
        )

        # make sure title and abstract are strings
        self.pathfinder_proposals["Proposal Title"] = self.pathfinder_proposals[
            "Proposal Title"
        ].astype(str)
        self.pathfinder_proposals["Proposal Abstract"] = self.pathfinder_proposals[
            "Proposal Abstract"
        ].astype(str)

        # drop duplicate rows based on the "Proposal Number" column
        self.pathfinder_proposals.drop_duplicates(
            subset="Proposal Number", inplace=True
        )

        # clean names, transform them to snake case
        self.pathfinder_proposals.columns = (
            self.pathfinder_proposals.columns.str.strip()
            .str.lower()
            .str.replace(" ", "_")
        )

        self.next(self.annotate_titles)

    @step
    def annotate_titles(self):
        """
        Annotate the "Proposal Title" column.
        """
        self.pathfinder_proposals["title_annotations"] = self.pathfinder_proposals[
            "proposal_title"
        ].apply(partial(self.get_annotation, confidence=0.25, support=1000))
        self.next(self.annotate_abstracts)

    @step
    def annotate_abstracts(self):
        """
        Annotate the "Proposal Abstract" column.
        """
        self.pathfinder_proposals["abstract_annotations"] = self.pathfinder_proposals[
            "proposal_abstract"
        ].apply(partial(self.get_annotation, confidence=0.25, support=1000))
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
            s3dm.save_to_s3(
                self.pathfinder_proposals,
                "data/02_intermediate/he_2020/pathfinder/proposals/main_dbp_annotated.parquet",
            )
        else:
            self.pathfinder_proposals.to_csv(
                "data/02_intermediate/main_dbp_annotated.csv",
                index=False,
            )
        self.next(self.end)

    @step
    def end(self):
        """
        End the flow.
        """
        pass

    def preprocess_text(self, text: str) -> str:
        """
        Preprocesses the given text by removing stop words, punctuation,
            and keeping only nouns, proper nouns, and adjectives.

        Args:
            text (str): The input text to be preprocessed.

        Returns:
            str: The preprocessed text.

        """
        doc = self.nlp(text)

        # extract tokens and remove stop words, punctuation, and non-nouns
        tokens = [
            token.lemma_.lower()
            for token in doc
            if not token.is_stop
            and not token.is_punct
            and token.pos_ in {"NOUN", "PROPN", "ADJ"}
        ]

        # prune tokens that are GPE (location), PERSON, LOC
        bad_ents = [
            str(ent).lower() for ent in doc.ents if ent in {"GPE", "LOC", "PERSON"}
        ]
        tokens = [token for token in tokens if token not in bad_ents]

        return " ".join(tokens)

    def get_annotation(
        self, text: str, confidence: float = 0.25, support: int = 200
    ) -> list:
        """
        Retrieves DBpedia annotations for the given text.

        Args:
            text (str): The input text to be annotated.
            confidence (float, optional): The confidence score threshold for
                DBpedia annotations. Defaults to 0.25.
            support (int, optional): The support threshold for DBpedia annotations.
                Defaults to 200.

        Returns:
            list: A list of unique annotations extracted from the DBpedia resources.

        """
        text = self.preprocess_text(text)
        if not text:
            return []

        session = requests.Session()
        retry = Retry(
            total=5, backoff_factor=0.1, status_forcelist=[500, 502, 503, 504]
        )
        session.mount("http://", HTTPAdapter(max_retries=retry))
        session.mount("https://", HTTPAdapter(max_retries=retry))

        base_url = "https://api.dbpedia-spotlight.org/en/annotate"
        headers = {"Accept": "application/json"}
        data = {
            "text": text,
            "confidence": confidence,
            "support": support,
            "policy": "whitelist",
        }

        response = session.post(base_url, data=data, headers=headers)
        response_json = response.json()
        resources = response_json.get("Resources", [])
        annotations = [
            resource.get("@URI")
            .replace("http://dbpedia.org/resource/", "")
            .replace("_", " ")
            for resource in resources
        ]

        return list(set(annotations))


if __name__ == "__main__":
    DBpediaAnnotationFlow()
