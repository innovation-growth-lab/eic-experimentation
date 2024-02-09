"""
This module exsecutes a Flow to 

Example:
    To retrieve OpenAlex

        $ python -m eic_case_studies.pipeline.openalex.oa_works_api --environment pypi run 
        $ --max-workers 1 --max-num-splits 100000 --iso_code EE --email david.ampudia@nesta.org.uk 
        $ --save_to_s3 True

    This will iteratively 
"""

from metaflow import FlowSpec, step, Parameter, pypi_base  # pylint: disable=E0611


@pypi_base(
    packages={
        "pandas": "2.1.3",
        "requests": "2.31.0",
        "toolz": "0.12.0",
        "boto3": "1.33.12",
        "pyyaml": "6.0.1",
        "pyarrow": "14.0.1",
        "fsspec": "2023.12.2",
        # "s3fs": "2023.12.2",
    },
    python="3.12.0",
)
class OaWorksFlow(FlowSpec):
    """
    A flow to retrieve OpenAlex institutions for a given ISO code.

    Parameters:
    - iso_code (str): The ISO code of the country. Default is "EE".
    - save_to_s3 (bool): Whether to save the data to S3. Default is False.
    """

    iso_code = Parameter(
        "iso_code",
        help="The ISO code of the country.",
        default="EE",
    )

    email = Parameter(
        "email",
        help="The email of the user.",
        default="",
    )

    save_to_s3 = Parameter(
        "save_to_s3",
        help="Whether to save the data to S3.",
        default=False,
    )

    @step
    def start(self):
        """Start the flow."""
        import yaml  # pylint: disable=C0415

        if self.iso_code == "all":
            with open("eic_case_studies/config/oa.yaml", "rt", encoding="utf-8") as f:
                config = yaml.load(f.read(), Loader=yaml.FullLoader)
            self.iso_codes = config["COUNTRY_ISO"]  # pylint: disable=W0201
        else:
            self.iso_codes = [self.iso_code]  # pylint: disable=W0201
        self.next(self.load_institutions, foreach="iso_codes")

    @step
    def load_institutions(self):
        """Retrieves OpenAlex institutions for a given ISO code."""
        # from getters.oa_institutions import get_oa_institutions  # pylint: disable=C0415
        from getters.s3io import S3DataManager  # pylint: disable=C0415
        from toolz import pipe  # pylint: disable=C0415
        from functools import partial  # pylint: disable=C0415

        s3dm = S3DataManager()
        try:
            # [TODO] This will break tomorrow as it won't read right.
            self.parsed_institutions = pipe(  # pylint: disable=W0201
                "data/oa/organisations/scrape_status.csv",
                partial(s3dm.load_s3_data),
                lambda df: df.set_index("Unnamed: 0").to_dict()["0"],
            )
            self.parsed_institutions = (  # pylint: disable=W0201
                self.parsed_institutions.to_dict()
            )
        except:  # pylint: disable=W0702
            self.parsed_institutions = {}  # pylint: disable=W0201

        oa_institutions = pipe(
            f"data/oa/organisations/{self.input}.parquet",
            partial(s3dm.load_s3_data),
            lambda df: df["id"].str.split("/").str[-1],
        )

        # anti-set of oa_institutions not in parsed_institutions
        self.oa_institutions = oa_institutions[  # pylint: disable=W0201
            ~oa_institutions.isin(self.parsed_institutions.keys())
        ]

        self.next(self.get_works, foreach="oa_institutions")

    @step
    def get_works(self):
        """Retrieves works for the specified institution."""

        print(f"Retrieving works for institution: {self.input}")

        from getters.oa_works import get_oa_works  # pylint: disable=C0415

        self.oa_works = get_oa_works(  # pylint: disable=W0201
            institution_id=self.input, email=self.email, pubdate="2015-01-01"
        )

        if self.save_to_s3:
            from getters.s3io import S3DataManager  # pylint: disable=C0415

            s3dm = S3DataManager()
            s3dm.save_to_s3(
                self.oa_works, f"data/oa/organisations/works/{self.input}.parquet"
            )

            # append to parsed_institutions
            self.parsed_institutions[self.input] = True
            s3dm.save_to_s3(
                self.parsed_institutions, "data/oa/organisations/scrape_status.csv"
            )

        self.next(self.institutions_join)

    @step
    def institutions_join(self, inputs):
        """Join the retrieved works for all institutions."""
        import pandas as pd  # pylint: disable=C0415

        self.country_works = pd.concat(  # pylint: disable=W0201
            [input.oa_works for input in inputs]
        )

        self.next(self.countries_join)

    @step
    def countries_join(self, inputs):
        """Join the retrieved works for all countries."""

        self.all_works = [  # pylint: disable=W0201
            [input.country_works for input in inputs]
        ]

        from getters.s3io import S3DataManager  # pylint: disable=C0415

        s3dm = S3DataManager()
        s3dm.save_to_s3(self.all_works, "data/oa/organisations/dump.json")

        self.next(self.end)

    @step
    def end(self):
        """
        End the flow.
        """
        pass  # pylint: disable=W0107


if __name__ == "__main__":
    OaWorksFlow()
