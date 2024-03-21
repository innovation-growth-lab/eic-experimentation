"""This module contains the DBpediaAnnotationFlow class, which is a Metaflow flow
    to tag OpenAlex works with DBpedia annotations. The flow consists of four steps:
    1. start: Load data and initialize NLP model.
    2. annotate_titles: Annotate the "title" column.
    3. annotate_abstracts: Annotate the "abstract" column.
    4. save_results: Save the annotated dataset to S3 or local file system.

    Example:
        $ python -m eic_case_studies.pipeline.cs2.researchers.dbp_tagging_flow --environment pypi run --save_to_s3 True --counterfactual False


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

    counterfactual = Parameter(
        "counterfactual",
        help="Whether to run on counterfactual researchers.",
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
        if not self.counterfactual:
            self.researcher_outputs = s3dm.load_s3_data(
                "data/03_primary/he_2020/pathfinder/roles/main_works.parquet"
            )
        else:
            self.researcher_outputs = s3dm.load_s3_data(
                "data/03_primary/he_2020/pathfinder/roles/main_cf_works.parquet"
            )
        # make sure title and abstract are strings
        self.researcher_outputs["title"] = self.researcher_outputs["title"].astype(str)
        self.researcher_outputs["abstract"] = self.researcher_outputs[
            "abstract"
        ].astype(str)

        self.next(self.annotate_titles)

    @step
    def annotate_titles(self):
        """
        Annotate the "Proposal Title" column.
        """
        self.researcher_outputs["title_annotations"] = self.researcher_outputs[
            "title"
        ].apply(partial(self.get_annotation, confidence=0.25, support=1000))
        self.next(self.annotate_abstracts)

    @step
    def annotate_abstracts(self):
        """
        Annotate the "Proposal Abstract" column.
        """
        self.researcher_outputs["abstract_annotations"] = self.researcher_outputs[
            "abstract"
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
                self.researcher_outputs,
                "data/03_primary/he_2020/pathfinder/roles/outputs/main_dbp_annotated.parquet",
            )
        else:
            self.researcher_outputs.to_csv(
                "data/03_primary/main_dbp_annotated.csv",
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
