"""This module contains the TopicSimilarityFlow class, which is a Metaflow 
    flow to compute similarity between each CWTS topic and every other topic.

    Example:
        To run the flow, use the following command:
            $ python -m eic_case_studies.pipeline.cs2.analysis.topic_disparity_flow --environment pypi run --save_to_s3 True
"""

# pylint: skip-file
from metaflow import FlowSpec, step, Parameter, pypi_base
import numpy as np
import pandas as pd
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
class TopicSimilarityFlow(FlowSpec):
    """
    A flow to compute similarity between each CWTS topic and every other topic.
    """

    save_to_s3 = Parameter(
        "save_to_s3",
        help="Whether to save the data to S3.",
        default=False,
    )

    use_topic_means = Parameter(
        "normalise",
        help="Whether to normalise the embeddings.",
        default=False,
    )

    @step
    def start(self):
        """
        Load CWTS taxonomy data.
        """
        from getters.s3io import S3DataManager

        s3dm = S3DataManager()
        self.cwts_taxonomy = s3dm.load_s3_data(
            "data/02_intermediate/cwts/main_dbp_embeddings.parquet"
        )

        # topic_id as string
        self.cwts_taxonomy["topic_id"] = self.cwts_taxonomy["topic_id"].astype(str)

        self.next(self.process_embeddings)

    @step
    def process_embeddings(self):
        """
        Process and reshape topic embeddings.
        """
        topic_embeddings_df = self.cwts_taxonomy.groupby(["topic_id"])[
            "keyword_embedding"
        ].apply(np.stack)
        self.topic_embeddings_df = topic_embeddings_df[
            topic_embeddings_df.apply(lambda x: x.shape[0] == 10)
        ]
        self.topic_embeddings = np.stack(self.topic_embeddings_df)

        if self.use_topic_means:
            self.topic_embeddings = np.mean(self.topic_embeddings, axis=1)

        self.next(self.compute_similarity)

    @step
    def compute_similarity(self):
        """
        Compute cosine similarity between each topic and every other topic.
        """
        if self.use_topic_means:
            self.disparity = 1 - cosine_similarity(self.topic_embeddings)

            # transform to a dataframe with topic ids as index and columns
            self.disparity_df = self.topic_embeddings_df.index.to_frame().set_index(
                "topic_id"
            )

            self.disparity_df = self.disparity_df.assign(
                **{
                    topic_id: self.disparity[i]
                    for i, topic_id in enumerate(self.disparity_df.index)
                }
            )
        else:
            num_topics = len(self.topic_embeddings)

            # Initialize an empty array to store the mean similarities for each topic
            mean_dissimilarities = np.empty((num_topics, num_topics))

            # Loop over each topic
            for i in range(num_topics):
                print(f"Computing similarities for topic {i}")
                # Loop over each other topic
                for j in range(num_topics):
                    # Compute the cosine similarity for each word in topic i against each word in topic j
                    dissimilarities = 1 - cosine_similarity(
                        self.topic_embeddings[i], self.topic_embeddings[j]
                    )
                    # Take the mean of these similarities
                    mean_dissimilarities[i, j] = np.mean(dissimilarities)

            self.disparity_df = pd.DataFrame(
                mean_dissimilarities, columns=self.topic_embeddings_df.index
            ).set_index(self.topic_embeddings_df.index)

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
                self.disparity_df,
                "data/03_primary/cwts/topic_disparity.parquet",
            )

        self.next(self.end)

    @step
    def end(self):
        """
        Prints a message indicating the flow has finished.

        """
        print("Flow finished.")


if __name__ == "__main__":
    TopicSimilarityFlow()
