"""This module contains the JoinProposals flow, which joins the proposals data with the topics data.

Example:
    To run the JoinProposals flow, use the following command:

        $ python -m eic_case_studies.pipeline.cs2.analysis.proposals_join_flow --environment pypi run --save_to_s3 True

"""

# pylint: skip-file
import yaml, random
import pandas as pd
import numpy as np
from functools import partial
from toolz import pipe
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
class JoinProposals(FlowSpec):

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

        s3dm = S3DataManager()
        self.proposals_joined_text = s3dm.load_s3_data(
            "data/03_primary/he_2020/pathfinder/proposals/main_dbp_oa_joined.parquet"
        ).rename(columns={"oa_topic_id_joined": "oa_topic_id"})

        self.proposals_keywords = s3dm.load_s3_data(
            "data/03_primary/he_2020/pathfinder/proposals/main_dbp_oa_validated.parquet"
        )[["proposal_number", "cs_topic_id", "oa_topic_id"]]

        self.proposals_data = s3dm.load_s3_data(
            "data/02_intermediate/he_2020/pathfinder/proposals/main_dbp_annotated.parquet",
        )[
            [
                "proposal_number",
                "proposal_call_deadline_date",
                "proposal_call_id",
                "proposal_topic_code",
                "proposal_last_evaluation_status",
            ]
        ]

        self.next(self.calculate_gini_coefficient)

    @step
    def calculate_gini_coefficient(self):
        self.unique_topics = self.proposals_keywords["oa_topic_id"].nunique()

        self.proposals_gb = (
            self.proposals_keywords.groupby("proposal_number")["oa_topic_id"]
            .apply(list)
            .reset_index()
            .assign(
                topic_counts=lambda x: (
                    x["oa_topic_id"]
                    .apply(lambda y: pd.Series(y).value_counts().to_dict())
                    .apply(lambda y: list(y.items()))
                )
            )
            .assign(
                topic_list=lambda x: x["topic_counts"].apply(
                    lambda y: [
                        item for sublist in [[t] * c for t, c in y] for item in sublist
                    ]
                )
            )
            .drop(columns="topic_counts")
        )

        self.proposals_gb_matched = (
            self.proposals_keywords[
                self.proposals_keywords["oa_topic_id"]
                == self.proposals_keywords["cs_topic_id"]
            ]
            .groupby("proposal_number")["oa_topic_id"]
            .apply(list)
            .reset_index()
            .assign(
                topic_counts=lambda x: (
                    x["oa_topic_id"]
                    .apply(lambda y: pd.Series(y).value_counts().to_dict())
                    .apply(lambda y: list(y.items()))
                )
            )
            .assign(
                topic_list_matched=lambda x: x["topic_counts"].apply(
                    lambda y: [
                        item for sublist in [[t] * c for t, c in y] for item in sublist
                    ]
                )
            )
            .drop(columns="topic_counts")
        )

        self.proposals_gb = self.proposals_gb.merge(
            self.proposals_gb_matched, on="proposal_number", how="left"
        )
        self.proposals_gb = self.proposals_gb[
            ["proposal_number", "topic_list", "topic_list_matched"]
        ]

        # merge proposals_gb with proposals_joined_text
        self.proposals_gb = self.proposals_gb.merge(
            self.proposals_joined_text, on="proposal_number", how="left"
        )
        self.proposals_gb = self.proposals_gb.rename(
            columns={"oa_topic_id": "oa_topic_id_joined"}
        )

        # merge proposals_gb with proposals_data
        self.proposals_gb = self.proposals_gb.merge(
            self.proposals_data, on="proposal_number", how="left"
        )
        # rename oa_topic_id_joined to topic_list_joined
        self.proposals_gb = self.proposals_gb.rename(
            columns={"oa_topic_id_joined": "topic_list_joined"}
        )

        self.next(self.save_to_s3)

    @step
    def save_to_s3(self):
        from getters.s3io import S3DataManager

        s3dm = S3DataManager()

        if self.save_to_s3:
            s3dm.save_to_s3(
                self.proposals_gb,
                "data/04_model_input/he_2020/pathfinder/proposals/main_dbp_oa_div_components.parquet",
            )

        self.next(self.end)

    @step
    def end(self):
        """
        End the flow.
        """
        pass


if __name__ == "__main__":
    JoinProposals()
