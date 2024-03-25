"""
python -m eic_case_studies.pipeline.cs2.researchers.oa_outputs --environment pypi run --save_to_s3 True --max-workers 6 --max-num-splits 50000
"""

# pylint: skip-file
import pandas as pd
from metaflow import FlowSpec, step, Parameter, pypi_base  # pylint: disable=E0611


@pypi_base(
    packages={
        "pandas": "2.1.3",
        "requests": "2.31.0",
        "toolz": "0.12.0",
        "boto3": "1.33.5",
        "pyyaml": "6.0.1",
        "pyarrow": "14.0.1",
        "openpyxl": "3.1.2",
    },
    python="3.12.0",
)
class OaResearchersFlow(FlowSpec):
    """
    A flow to retrieve Open Access works for researchers in the HE dataset.

    Parameters:
    - email (str): The email of the user. Default is "david.ampudia@nesta.org.uk"
    - save_to_s3 (bool): Whether to save the data to S3. Default is False.

    """

    email = Parameter(
        "email",
        help="The email of the user.",
        default="david.ampudia@nesta.org.uk",
    )

    save_to_s3 = Parameter(
        "save_to_s3",
        help="Whether to save the data to S3.",
        default=False,
    )

    @step
    def start(self):
        """
        Start the flow.
        """
        from getters.s3io import (
            S3DataManager,
        )

        # create an instance of the S3DataManager class and load data
        s3dm = S3DataManager()

        self.researchers_he = s3dm.load_s3_data(
            "data/01_raw/he_2020/pathfinder/roles/roles_he.xlsx",
            skiprows=5,
            usecols=lambda x: "Unnamed" not in x,
        )
        self.researchers_h2020 = s3dm.load_s3_data(
            "data/01_raw/he_2020/pathfinder/roles/roles_h2020.xlsx"
        )

        self.researchers_df = pd.concat(
            [self.researchers_he, self.researchers_h2020], ignore_index=True
        )

        # rename to snake case
        self.researchers_df = self.researchers_df.rename(
            columns={
                "Proposal Context Code": "proposal_context_code",
                "Proposal Topic Code": "proposal_topic_code",
                "Proposal Number": "proposal_number",
                "Applicant Role": "applicant_role",
                "Appl Researcher Google Scholar Id": "appl_researcher_google_scholar_id",
                "Appl Researcher ORCID Id": "appl_researcher_orcid",
                "Appl Researcher Reference Id": "appl_researcher_reference_id",
                "Proposal Submission Date": "submission_date",
            },
        )[
            [
                "proposal_context_code",
                "proposal_topic_code",
                "proposal_number",
                "applicant_role",
                "appl_researcher_google_scholar_id",
                "appl_researcher_orcid",
                "appl_researcher_reference_id",
                "submission_date",
            ]
        ]

        self.unique_researchers = list(
            self.researchers_df["appl_researcher_orcid"].unique()
        )[1:]

        # create batches of 50 authors
        self.next(self.get_researcher_works, foreach="unique_researchers")

    @step
    def get_researcher_works(self):
        """
        Retrieve works for the specified researcher.
        """
        from getters.oa_works import get_oa_researcher_works, revert_abstract_index

        print(f"Retrieving works for researcher: {self.input}")

        self.oa_works = get_oa_researcher_works(
            researcher_id=self.input, email=self.email, pubdate="2013-01-01"
        )

        print(f"Retrieved {self.oa_works.shape[0]} works for researcher: {self.input}.")

        # if dataframe is empty, skip
        if not self.oa_works.empty:

            print(f"Reverting abstract index for researcher: {self.input}.")

            # keep title, abstract_inverted_index, cited_by_count, publication_year, topics
            self.oa_works = self.oa_works[
                [
                    "doi",
                    "title",
                    "abstract_inverted_index",
                    "cited_by_count",
                    "publication_year",
                    "topics",
                ]
            ]

            self.oa_works["researcher"] = self.input

            # apply revert abstract index function where not None
            self.oa_works["abstract"] = self.oa_works["abstract_inverted_index"].apply(
                lambda x: revert_abstract_index(x) if x is not None else None
            )

            # remove abstract_inverted_index column
            self.oa_works = self.oa_works.drop(columns=["abstract_inverted_index"])

            # if topics is a list of dictionaries, extract the id
            self.oa_works["topics"] = self.oa_works["topics"].apply(
                lambda x: [
                    topic["id"].replace("https://openalex.org/T", "") for topic in x
                ] if isinstance(x, list) else []
            )

            

        self.next(self.researcher_join)

    @step
    def researcher_join(self, inputs):
        """Join the retrieved works for all researchers."""

        self.researcher_works = pd.concat([input.oa_works for input in inputs])

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
                self.researcher_works,
                "data/03_primary/he_2020/pathfinder/roles/main_works.parquet",
            )

        self.next(self.end)

    @step
    def end(self):
        """
        End the flow.
        """
        pass 


if __name__ == "__main__":
    OaResearchersFlow()
