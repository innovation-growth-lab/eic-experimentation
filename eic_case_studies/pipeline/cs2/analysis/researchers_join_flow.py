"""This module contains the Metaflow flow for joining the researchers data.
The flow loads the data from S3, filters the data, standardizes the data,
processes the proposals, merges the data, aggregates the data, and saves the
data to S3.

    To run the JoinResearchers flow, use the following command:

        $ python -m eic_case_studies.pipeline.cs2.analysis.researchers_join_flow --environment pypi run --save_to_s3 True
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
class JoinResearchers(FlowSpec):

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

        self.researchers_outputs = s3dm.load_s3_data(
            "data/03_primary/he_2020/pathfinder/roles/main_works.parquet"
        )[["researcher", "cited_by_count", "publication_year", "topics"]]

        self.researchers_cf_outputs = s3dm.load_s3_data(
            "data/03_primary/he_2020/pathfinder/roles/main_cf_works.parquet"
        )[["researcher", "cited_by_count", "publication_year", "topics"]]

        self.pathfinder_he_proposals = s3dm.load_s3_data(
            "data/01_raw/he_2020/pathfinder/proposals/main_he.xlsx",
            skiprows=3,
            usecols=lambda x: "Unnamed" not in x,
        )

        self.researchers_in_proposals = s3dm.load_s3_data(
            "data/01_raw/he_2020/pathfinder/roles/roles_he.xlsx",
            skiprows=5,
            usecols=lambda x: "Unnamed" not in x,
        )

        self.next(self.filter_data)

    @step
    def filter_data(self):
        self.researchers_outputs = self.researchers_outputs[
            self.researchers_outputs["publication_year"] >= 2018
        ]
        self.researchers_cf_outputs = self.researchers_cf_outputs[
            self.researchers_cf_outputs["publication_year"] >= 2018
        ]

        self.next(self.standardize_data)

    @step
    def standardize_data(self):
        self.scaler = StandardScaler()

        self.combined = pd.concat(
            [self.researchers_outputs, self.researchers_cf_outputs]
        )

        self.combined["cited_by_count_std"] = self.combined.groupby("publication_year")[
            "cited_by_count"
        ].transform(
            lambda x: self.scaler.fit_transform(x.values.reshape(-1, 1)).ravel()
        )

        self.researchers_outputs = self.combined.loc[
            self.combined.index.isin(self.researchers_outputs.index)
        ]
        self.researchers_cf_outputs = self.combined.loc[
            self.combined.index.isin(self.researchers_cf_outputs.index)
        ]

        self.next(self.process_proposals)

    @step
    def process_proposals(self):
        # Assuming 'year' is the column containing the year of the proposal and 'researcher' is the column containing the researcher names
        self.researchers_in_proposals = self.researchers_in_proposals.sort_values(
            "Proposal Submission Date"
        )
        self.first_last_proposals = (
            self.researchers_in_proposals.groupby("Appl Researcher ORCID Id")
            .agg(
                {
                    "Proposal Submission Date": ["min", "max"],
                    "Proposal Last Evaluation Status": "last",
                }
            )
            .reset_index()
        )

        # Flatten the MultiIndex in the columns
        self.first_last_proposals.columns = [
            "_".join(col).strip() for col in self.first_last_proposals.columns.values
        ]

        # Rename the columns
        self.first_last_proposals = self.first_last_proposals.rename(
            columns={
                "Appl Researcher ORCID Id_": "researcher",
                "Proposal Submission Date_min": "first_observed",
                "Proposal Submission Date_max": "last_observed",
                "Proposal Last Evaluation Status_last": "status",
            }
        )

        self.next(self.merge_data)

    @step
    def merge_data(self):
        self.researchers_merged = pd.merge(
            self.researchers_outputs, self.first_last_proposals, on="researcher"
        )

        self.researchers_merged["publication_date"] = np.where(
            self.researchers_merged["publication_year"]
            < self.researchers_merged["first_observed"].dt.year,
            "before",
            np.where(
                self.researchers_merged["publication_year"]
                >= self.researchers_merged[
                    "first_observed"
                ].dt.year,  # note i use only first observed
                "after",
                self.researchers_merged["publication_year"],
            ),
        )

        self.next(self.aggregate_data)

    @step
    def aggregate_data(self):
        # Group by researcher and year to get the list of topics and citation counts for each researcher per year
        self.researchers_outputs_yearly = (
            self.researchers_outputs.groupby(["researcher", "publication_year"])
            .agg({"topics": list, "cited_by_count_std": list})
            .reset_index()
        )

        # Group by researcher to get the list of all topics and citation counts for each researcher
        self.researchers_outputs_all = (
            self.researchers_outputs.groupby("researcher")
            .agg({"topics": list, "cited_by_count_std": list})
            .reset_index()
        )
        self.researchers_outputs_all["publication_year"] = "all"

        # Group by before/after to get the list of all topics and citation counts for each researcher
        self.researchers_outputs_before_after = (
            self.researchers_merged.groupby(["researcher", "publication_date"])
            .agg({"topics": list, "cited_by_count_std": list})
            .reset_index()
        ).rename(columns={"publication_date": "publication_year"})

        # Append the rows with "all" as the year
        self.researchers_outputs_agg = pd.concat(
            [
                self.researchers_outputs_yearly,
                self.researchers_outputs_all,
                self.researchers_outputs_before_after,
            ],
            ignore_index=True,
        )

        # add the status by creating the dictionary and mapping to researcher
        self.researchers_outputs_agg["status"] = self.researchers_outputs_agg[
            "researcher"
        ].map(self.researchers_merged.set_index("researcher")["status"].to_dict())

        # assign for every nan status the value "not_eic"
        self.researchers_outputs_agg["status"] = self.researchers_outputs_agg[
            "status"
        ].fillna("NOT_EIC")

        # create counts of publications
        self.researchers_outputs_agg["publications_count"] = (
            self.researchers_outputs_agg["cited_by_count_std"].apply(len)
        )

        # standardise by year and take the standard publication count
        self.researchers_outputs_agg[
            "publications_count_std"
        ] = self.researchers_outputs_agg.groupby("publication_year")[
            "publications_count"
        ].transform(
            lambda x: self.scaler.fit_transform(x.values.reshape(-1, 1)).ravel()
        )

        # flatten topics lists
        self.researchers_outputs_agg["topics"] = self.researchers_outputs_agg[
            "topics"
        ].apply(lambda x: [item for sublist in x for item in sublist])

        self.next(self.save_data)

    @step
    def save_data(self):
        from getters.s3io import S3DataManager
        s3dm = S3DataManager()
        if self.save_to_s3:
            # change column publication_year to string
            self.researchers_outputs_agg["publication_year"] = self.researchers_outputs_agg[
                "publication_year"
            ].astype(str)
            
            s3dm.save_to_s3(
                self.researchers_outputs_agg,
                "data/04_model_input/he_2020/pathfinder/roles/researchers_outputs_agg.parquet",
            )
        self.next(self.end)

    @step
    def end(self):
        pass


if __name__ == "__main__":
    JoinResearchers()
