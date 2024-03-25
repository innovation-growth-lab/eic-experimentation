"""This module contains the Metaflow flow for calculating diversity metrics for researchers.
    The flow loads the data from S3, calculates the diversity metrics, and saves the data to S3.

    To run the DiversityFlow flow, use the following command:

        $ python -m eic_case_studies.pipeline.cs2.analysis.researchers_compute_div --environment pypi run --save_to_s3 True
"""

# pylint: skip-file
import pandas as pd
import numpy as np
from toolz import pipe
from sklearn.preprocessing import StandardScaler
from metaflow import FlowSpec, step, Parameter, pypi_base  # pylint: disable=E0611
from collections import Counter


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
class ResearcherDiversityFlow(FlowSpec):

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

        self.researchers_outputs_agg = s3dm.load_s3_data(
            "data/04_model_input/he_2020/pathfinder/roles/researchers_outputs_agg.parquet",
        )
        self.next(self.calculate_diversity)

    @step
    def calculate_diversity(self):
        self.researchers_outputs_agg[["average_disparity", "variety", "balance"]] = pd.DataFrame(
            self.researchers_outputs_agg["topics"]
            .apply(
                lambda x: calculate_div_components(
                    x, len(self.topic_disparity), self.topic_disparity
                )
            )
            .tolist(),
            index=self.researchers_outputs_agg.index,
        )

        # compute div from the three components
        self.researchers_outputs_agg["div"] = (
            self.researchers_outputs_agg["average_disparity"]
            * self.researchers_outputs_agg["variety"]
            * self.researchers_outputs_agg["balance"]
        )

        self.next(self.save_data)

    @step
    def save_data(self):
        from getters.s3io import S3DataManager

        s3dm = S3DataManager()
        if self.save_to_s3:
            s3dm.save_to_s3(
                self.researchers_outputs_agg,
                "data/05_model_output/he_2020/pathfinder/roles/researchers_outputs_agg.parquet",
            )
        self.next(self.end)

    @step
    def end(self):
        pass


def calculate_gini_coefficient(topic_array):
    """Calculate the Gini coefficient for a distribution of topic IDs."""
    counts = list(Counter(topic_array).values())
    array = np.sort(np.sort(np.array(counts)))  # values must be sorted
    index = np.arange(1, array.shape[0] + 1)  # index per array element
    n = array.shape[0]  # number of array elements
    return (np.sum((2 * index - n - 1) * array)) / (n * np.sum(array))


def calculate_individual_disparity_sum(topic_ids, disparity_matrix):
    """Sum the individual disparities for each unique pair of topics in a researcher's list."""
    unique_pairs = {
        (min(i, j), max(i, j)) for i in topic_ids for j in topic_ids if i != j
    }
    disparity_sum = sum(disparity_matrix.loc[i, j] for i, j in unique_pairs)

    return disparity_sum


def calculate_div_components(topic_ids, N, disparity_matrix):
    """Calculate the components of the Diversity (DIV) index using topic IDs and a disparity matrix."""
    # check all topics are in the disparity matrix
    topic_ids = [i for i in topic_ids if i in disparity_matrix.columns]
    unique_topics = list(set(topic_ids))
    nc = len(unique_topics)  # Number of unique classes used
    gini_coefficient = calculate_gini_coefficient(topic_ids)

    variety = nc / N
    balance = 1 - gini_coefficient
    disparity_sum = calculate_individual_disparity_sum(
        unique_topics, disparity_matrix
    )
    # Assuming nc > 1, adjust denominator to avoid division by zero for a single topic
    denominator = nc * (nc - 1) if nc > 1 else 1
    average_disparity = disparity_sum / denominator

    return average_disparity, variety, balance


if __name__ == "__main__":
    ResearcherDiversityFlow()
