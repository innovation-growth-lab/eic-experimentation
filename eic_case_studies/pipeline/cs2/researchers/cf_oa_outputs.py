"""
python -m eic_case_studies.pipeline.cs2.researchers.cf_oa_outputs --environment pypi run --save_to_s3 True --skip_researchers True --max-workers 6 --max-num-splits 50000

"""

# pylint: skip-file
import yaml, random
import pandas as pd
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
class OaResearchersCounterfactualFlow(FlowSpec):

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

    skip_researchers = Parameter(
        "skip_researchers",
        help="Whether to skip the researchers step.",
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

        with open("eic_case_studies/config/oa.yaml", "rt", encoding="utf-8") as f:
            config = yaml.load(f.read(), Loader=yaml.FullLoader)
        self.iso_codes = config["COUNTRY_ISO"]  # pylint: disable=W0201

        # create an instance of the S3DataManager class and load data
        s3dm = S3DataManager()
        self.baseline_researchers = s3dm.load_s3_data(
                "data/03_primary/he_2020/pathfinder/roles/main_works.parquet",
            )
        
        # get the unique baseline researchers
        self.unique_bl_researchers = list(set(self.baseline_researchers['researcher'].tolist()))

        topics_df = s3dm.load_s3_data(
            "data/03_primary/he_2020/pathfinder/proposals/main_dbp_oa_validated.parquet",
        )
        self.topics = list(topics_df.oa_topic_id.unique())

        self.next(self.get_researchers, foreach="topics")

    @step
    def get_researchers(self):
        """
        Get researchers.
        """
        from getters.oa_works import (
            get_oa_researcher_counterfactual_works,
        )

        if not self.skip_researchers:
            self.oa_topic_works = get_oa_researcher_counterfactual_works(
                topic_id="T" + self.input, email=self.email, pubdate="2018-01-01"
            )

            print(f"Retrieved {len(self.oa_topic_works)} works for topic {self.input}")

            matching_researchers = []

            for _, work in self.oa_topic_works.iterrows():
                if work["authorships"]:
                    for authorship in work["authorships"]:
                        if "countries" in authorship and "orcid" in authorship["author"]:
                            # Check if at least one country ISO in 'countries' matches one ISO in self.iso_codes
                            if any(
                                iso in self.iso_codes for iso in authorship["countries"]
                            ):
                                # If so, append the 'orcid' for that researcher to matching_researchers
                                matching_researchers.append(
                                    authorship["author"]["orcid"].replace(
                                        "https://orcid.org/", ""
                                    )
                                    if authorship["author"]["orcid"]
                                    else None
                                )

            # keep non-None
            matching_researchers = list(filter(None, matching_researchers))

            # remove duplicates
            matching_researchers = list(set(matching_researchers))

            print(f"Retrieved {len(matching_researchers)} matching researchers for topic {self.input}.")

            # randomly select 10 researchers that are not in the baseline
            matching_researchers = list(
                set(matching_researchers).difference(set(self.unique_bl_researchers))
            )
            sample_size = (
                10 if len(matching_researchers) > 10 else len(matching_researchers)
            )
            self.researchers = random.sample(matching_researchers, sample_size)

            # save topic_id and researchers
            from getters.s3io import (
                S3DataManager,
            )
            s3dm = S3DataManager()
            s3dm.save_to_s3(
                pd.DataFrame(
                    {
                        "topic_id": ["T" + self.input] * len(self.researchers),
                        "researchers": self.researchers,
                    }
                ),
                f"data/03_primary/he_2020/pathfinder/roles/topics/{str(self.input)}_cf_researchers.parquet",
            )

        else:
            self.researchers = []

            # for each topic, try to load the researchers
            from getters.s3io import (
                S3DataManager,
            )

            s3dm = S3DataManager()

            try:
                self.researchers = s3dm.load_s3_data(
                    f"data/03_primary/he_2020/pathfinder/roles/topics/{str(self.input)}_cf_researchers.parquet"
                )
                self.researchers = self.researchers.researchers.tolist()
            except:
                pass

        self.next(self.researcher_join)

    @step
    def researcher_join(self, inputs):
        """Join the retrieved reseacher IDs."""

        data = [(item.input, researcher) for item in inputs for researcher in item.researchers]

        # Create a pandas DataFrame with both columns
        unique_researchers_df = pd.DataFrame(data, columns=['topic_id', 'researcher'])

        if self.save_to_s3:
            from getters.s3io import (
                S3DataManager,
            )
            s3dm = S3DataManager()
            s3dm.save_to_s3(
                unique_researchers_df,
                "data/03_primary/he_2020/pathfinder/roles/main_cf_researchers.parquet",
            )

        # Convert the 'researcher' column to a list to pass downstream
        self.unique_researchers = list(set(unique_researchers_df['researcher'].tolist()))

        self.next(self.split_researchers)
    @step
    def split_researchers(self):
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
                lambda x: (
                    [topic["id"].replace("https://openalex.org/T", "") for topic in x]
                    if isinstance(x, list)
                    else []
                )
            )

        self.next(self.researcher_works_join)

    @step
    def researcher_works_join(self, inputs):
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
                "data/03_primary/he_2020/pathfinder/roles/main_cf_works.parquet",
            )

        self.next(self.end)

    @step
    def end(self):
        """
        End the flow.
        """
        pass


if __name__ == "__main__":
    OaResearchersCounterfactualFlow()
