"""
python -m eic_case_studies.pipeline.cs2.researchers.researcher_consortia_flow --environment pypi run --save_to_s3 True
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

        self.researcher_works = s3dm.load_s3_data(
            "data/03_primary/he_2020/pathfinder/roles/main_works.parquet",
        )

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

        # create batches of 50 authors
        self.next(self.group_by_consortia)

    @step
    def group_by_consortia(self):
        """
        Group the researcher works by unique proposal ID (consortia).
        """

        # map researcher (appl_researcher_orcid in researchers_df) to proposal_number
        self.researcher_works = self.researcher_works.merge(
            self.researchers_df[["appl_researcher_orcid", "proposal_number", "submission_date"]],
            left_on="researcher",
            right_on="appl_researcher_orcid",
            how="left",
        )

        self.researcher_works_grouped = self.researcher_works.groupby(['proposal_number', 'publication_year']).agg({
            'submission_date': 'first',
            'doi': 'count',  
            'cited_by_count': 'sum', 
            'topics': lambda x: [item for sublist in x for item in sublist],
            'researcher': lambda x: list(x.unique()),
        }).reset_index()

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
                self.researcher_works_grouped,
                "data/03_primary/he_2020/pathfinder/roles/main_consortia_works.parquet",
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
