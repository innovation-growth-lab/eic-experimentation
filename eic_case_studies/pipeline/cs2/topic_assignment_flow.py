"""This module contains the Metaflow flow to assign topics to Pathfinder 
    proposals based on keyword embeddings and similarity measures.

Example:
    To run the TopicAssignmentFlow flow, use the following command:

        $ python -m eic_case_studies.pipeline.cs2.topic_assignment_flow 
            --environment pypi run --save_to_s3 True
"""

# pylint: skip-file
from metaflow import FlowSpec, step, Parameter, pypi_base
import numpy as np
import pandas as pd
from scipy.special import softmax
from sklearn.metrics.pairwise import cosine_similarity


@pypi_base(
    packages={
        "numpy": "1.26.2",
        "scipy": "1.12.0",
        "scikit-learn": "1.4.1.post1",
        "boto3": "1.34.1",
        "pandas": "2.1.3",
        "pyarrow": "14.0.1",
    },
    python="3.12.0",
)
class TopicAssignmentFlow(FlowSpec):
    """
    A flow to assign topics to Pathfinder proposals based on keyword
        embeddings and similarity measures.
    """

    save_to_s3 = Parameter(
        "save_to_s3",
        help="Whether to save the data to S3.",
        default=False,
    )

    @step
    def start(self):
        """
        Load annotated proposals and taxonomy data.
        """
        from getters.s3io import (
            S3DataManager,
        )  # Assuming this is a custom module for S3 interaction

        s3dm = S3DataManager()
        self.annotated_proposals = s3dm.load_s3_data(
            "data/02_intermediate/he_2020/pathfinder/proposals/main_dbp_embeddings.parquet"
        )

        self.cwts_taxonomy = s3dm.load_s3_data(
            "data/02_intermediate/cwts/main_dbp_embeddings.parquet"
        )

        # topic_id as string
        self.cwts_taxonomy["topic_id"] = self.cwts_taxonomy["topic_id"].astype(str)

        self.next(self.process_embeddings)

    @step
    def process_embeddings(self):
        """
        Process and reshape keyword embeddings.
        """
        # Processing as described in the provided script
        topic_embeddings_df = self.cwts_taxonomy.groupby(["topic_id"])[
            "keyword_embedding"
        ].apply(np.stack)
        topic_embeddings_df = topic_embeddings_df[
            topic_embeddings_df.apply(lambda x: x.shape[0] == 10)
        ]
        self.topic_embeddings = np.stack(topic_embeddings_df)

        # Drop duplicate keywords
        self.unique_keywords = self.annotated_proposals.drop_duplicates(
            subset="keywords"
        )

        # Stack keyword embeddings
        self.proposal_embeddings = np.stack(
            self.unique_keywords["keyword_embedding"].apply(np.stack)
        )

        # Create a dictionary to map index to topic id
        self.index_to_topic_id = dict(
            zip(
                range(len(topic_embeddings_df)),
                topic_embeddings_df.index.values.tolist(),
            )
        )
        self.next(self.compute_similarity)

    @step
    def compute_similarity(self):
        """
        Compute cosine similarity between proposal and topic embeddings.
        """
        topic_embeddings_2d = self.topic_embeddings.reshape(-1, 768)
        cos_sim_2d = cosine_similarity(self.proposal_embeddings, topic_embeddings_2d)
        self.cos_sim = cos_sim_2d.reshape(self.proposal_embeddings.shape[0], 4514, 10)
        self.next(self.filter_similarity)

    @step
    def filter_similarity(self):
        """
        Filter and sum the similarity scores.
        """
        quantiles = np.quantile(self.cos_sim, [0.25, 0.5, 0.95])
        self.cos_sim[self.cos_sim < quantiles[2]] = 0
        self.similarity_scores = self.cos_sim.sum(axis=2)
        self.next(self.assign_topics)

    @step
    def assign_topics(self):
        """
        Assign topics based on similarity scores and softmax probabilities.
        """
        self.softmax_probabilities = softmax(self.similarity_scores, axis=1)
        ratio_threshold = 1.25
        topic_assignment = np.argmax(self.softmax_probabilities, axis=1)
        highest_probs = np.sort(self.softmax_probabilities, axis=1)[:, -1]
        second_highest_probs = np.sort(self.softmax_probabilities, axis=1)[:, -2]
        topic_assignment[highest_probs / second_highest_probs < ratio_threshold] = -1
        self.topic_assignment = np.vectorize(self.index_to_topic_id.get)(
            topic_assignment
        )
        self.next(self.merge_data)

    @step
    def merge_data(self):
        """
        Merge topic assignments back into proposals dataset.
        """
        self.unique_keywords["cs_topic_id"] = self.topic_assignment
        
        # create a dictionary to map keyword to topic id
        self.keyword_to_topic_id = dict(
            zip(
                self.unique_keywords["keywords"].values,
                self.unique_keywords["cs_topic_id"].values.astype(str),
            )
        )

        # map the topic id to the proposals
        self.annotated_proposals["cs_topic_id"] = self.annotated_proposals["keywords"].map(
            self.keyword_to_topic_id
        )

        self.annotated_proposals = self.annotated_proposals.merge(
            self.cwts_taxonomy[
                ["topic_id", "topic_name", "subfield_name", "field_name", "domain_name"]
            ].drop_duplicates(),
            left_on="cs_topic_id",
            right_on="topic_id",
            how="left",
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

        if self.save_to_s3:
            s3dm = S3DataManager()
            s3dm.save_to_s3(
                self.annotated_proposals,
                "data/03_primary/he_2020/pathfinder/proposals/main_dbp_assigned.parquet",
            )

        self.next(self.end)

    @step
    def end(self):
        """
        End of the flow.
        """
        pass


if __name__ == "__main__":
    TopicAssignmentFlow()
