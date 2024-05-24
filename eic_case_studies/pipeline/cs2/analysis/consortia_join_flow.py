"""This module

    To run the JoinResearchersConsortia flow, use the following command:

        $ python -m eic_case_studies.pipeline.cs2.analysis.consortia_join_flow --environment pypi run --save_to_s3 True

    Plenty of things missing here: no NO_EIC, hapharzard computation of citation std.. NO_EIC is messed up, done on actual EIC data, which gets superseeded in plots flow.
"""

# pylint: skip-file
import pandas as pd
import numpy as np
from toolz import pipe
from sklearn.preprocessing import StandardScaler
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
        "scikit-learn": "1.4.1.post1",
    },
    python="3.12.0",
)
class JoinResearchersConsortia(FlowSpec):

    save_to_s3 = Parameter(
        "save_to_s3",
        help="Whether to save the data to S3.",
        default=False,
    )

    @step
    def start(self):

        self.next(self.load_data)

    @step
    def load_data(self):

        from getters.s3io import S3DataManager

        s3dm = S3DataManager()

        self.topic_disparity = s3dm.load_s3_data(
            "data/03_primary/cwts/topic_disparity.parquet"
        )
        self.topic_disparity.index = self.topic_disparity.columns

        self.researcher_consortia_outputs = s3dm.load_s3_data(
            "data/03_primary/he_2020/pathfinder/roles/main_consortia_works.parquet",
        )

        self.researchers_cf_outputs = s3dm.load_s3_data(
            "data/03_primary/he_2020/pathfinder/roles/main_cf_works.parquet"
        )[["researcher", "cited_by_count", "publication_year", "topics"]]

        self.researchers_in_proposals = s3dm.load_s3_data(
            "data/01_raw/he_2020/pathfinder/roles/roles_he.xlsx",
            skiprows=5,
            usecols=lambda x: "Unnamed" not in x,
        ).rename(
            columns={
                "Proposal Last Evaluation Status": "status",
                "Proposal Number": "proposal_number",
                "Proposal Topic Code": "proposal_call_id",
            }
        )

        self.next(self.filter_data)

    @step
    def filter_data(self):
        self.researcher_consortia_outputs = self.researcher_consortia_outputs[
            self.researcher_consortia_outputs["publication_year"] >= 2018
        ]
        self.researchers_cf_outputs = self.researchers_cf_outputs[
            self.researchers_cf_outputs["publication_year"] >= 2018
        ]

        self.next(self.standardize_data)

    @step
    def standardize_data(self):
        self.scaler = StandardScaler()

        self.combined = pd.concat(
            [self.researcher_consortia_outputs, self.researchers_cf_outputs]
        )
        # num researchers
        self.combined['num_researchers'] = self.combined['researcher'].apply(len)
        # divide cited_by_count by doi when doi is not null
        self.combined["cited_by_count"] = np.where(
            self.combined["doi"].notnull(),
            self.combined["cited_by_count"] / self.combined["num_researchers"],
            self.combined["cited_by_count"],
        )

        self.combined["cited_by_count_std"] = self.combined.groupby("publication_year")[
            "cited_by_count"
        ].transform(
            lambda x: self.scaler.fit_transform(x.values.reshape(-1, 1)).ravel()
        )

        # go back to the two separate dataframes
        self.researcher_consortia_outputs = self.combined[
            self.combined["doi"].notnull()
        ]
        self.researchers_cf_outputs = self.combined[self.combined["doi"].isnull()].drop(
            columns="doi"
        )

        self.next(self.label_datetimes)

    @step
    def label_datetimes(self):

        self.researcher_consortia_outputs["binary_datetime"] = np.where(
            self.researcher_consortia_outputs["publication_year"]
            < self.researcher_consortia_outputs["submission_date"].dt.year,
            "before",
            np.where(
                self.researcher_consortia_outputs["publication_year"]
                >= self.researcher_consortia_outputs[
                    "submission_date"
                ].dt.year,  # note I use only first observed
                "after",
                self.researcher_consortia_outputs["publication_year"],
            ),
        )

        # Calculate the difference between publication year and submission year
        self.researcher_consortia_outputs["year_difference"] = (
            self.researcher_consortia_outputs["publication_year"]
            - self.researcher_consortia_outputs["submission_date"].dt.year
        )

        year_difference_labels = {
            -3: "t-3",
            -2: "t-2",
            -1: "t-1",
            0: "t",
            1: "t+1",
            2: "t+2",
        }

        self.researcher_consortia_outputs["relative_publication_year"] = (
            self.researcher_consortia_outputs["year_difference"].map(
                year_difference_labels
            )
        )

        self.researcher_consortia_outputs["relative_publication_year"].fillna(
            "other", inplace=True
        )

        self.next(self.aggregate_data)

    @step
    def aggregate_data(self):

        # Group by researcher to get the list of all topics and citation counts for each researcher
        self.researchers_outputs_all = (
            self.researcher_consortia_outputs.groupby("proposal_number")
            .agg({"topics": list, "cited_by_count_std": list})
            .reset_index()
        )
        self.researchers_outputs_all["publication_year"] = "all"

        # Group by before/after to get the list of all topics and citation counts for each researcher
        self.researchers_outputs_before_after = (
            self.researcher_consortia_outputs.groupby(
                ["proposal_number", "binary_datetime"]
            )
            .agg({"topics": list, "cited_by_count_std": list})
            .reset_index()
        ).rename(columns={"binary_datetime": "publication_year"})

        # Append the rows with "all" as the year
        self.researchers_outputs_agg = pd.concat(
            [
                self.researcher_consortia_outputs,
                self.researchers_outputs_all,
                self.researchers_outputs_before_after,
            ],
            ignore_index=True,
        )

        # add the status by creating the dictionary and mapping to researcher
        self.researchers_outputs_agg["status"] = self.researchers_outputs_agg[
            "proposal_number"
        ].map(
            self.researchers_in_proposals.set_index("proposal_number")[
                "status"
            ].to_dict()
        )

        # add the proposal call id by creating the dictionary and mapping to researcher
        self.researchers_outputs_agg["proposal_call_id"] = self.researchers_outputs_agg[
            "proposal_number"
        ].map(
            self.researchers_in_proposals.set_index("proposal_number")[
                "proposal_call_id"
            ].to_dict()
        )

        # fill missing values with NO_DATA
        self.researchers_outputs_agg["status"].fillna("NO_DATA", inplace=True)

        # create counts of publications
        self.researchers_outputs_agg["publications_count"] = (
            self.researchers_outputs_agg["doi"]
        )

        # Create 'publications_per_researcher' column
        self.researchers_outputs_agg['publications_per_researcher'] = np.where(
            self.researchers_outputs_agg['num_researchers'].isnull(),
            np.nan,
            self.researchers_outputs_agg['publications_count'] / self.researchers_outputs_agg['num_researchers']
        )

        # standardise by year and take the standard publication count
        self.researchers_outputs_agg[
            "publications_count_std"
        ] = self.researchers_outputs_agg.groupby("publication_year")[
            "publications_per_researcher"
        ].transform(
            lambda x: self.scaler.fit_transform(x.values.reshape(-1, 1)).ravel()
        )

        # flatten topic list if its not already flat
        self.researchers_outputs_agg["topics"] = self.researchers_outputs_agg[
            "topics"
        ].apply(
            lambda x: (
                [item for sublist in x for item in sublist]
                if isinstance(x, list)
                else x
            )
        )

        self.next(self.save_data)

    @step
    def save_data(self):
        from getters.s3io import S3DataManager

        s3dm = S3DataManager()
        if self.save_to_s3:
            # change column publication_year to string
            self.researchers_outputs_agg["publication_year"] = (
                self.researchers_outputs_agg["publication_year"].astype(str)
            )

            self.researchers_outputs_agg["cited_by_count_std"] = self.researchers_outputs_agg["cited_by_count_std"].apply(
                lambda x: (
                    [x]
                    if not isinstance(x, list) else x
                )
            )


            s3dm.save_to_s3(
                self.researchers_outputs_agg,
                "data/04_model_input/he_2020/pathfinder/roles/consortia_outputs_agg.parquet",
            )
        self.next(self.end)

    @step
    def end(self):
        pass


if __name__ == "__main__":
    JoinResearchersConsortia()
