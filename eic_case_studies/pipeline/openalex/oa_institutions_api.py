"""
This module exsecutes a Flow to obtain OpenAlex institutions for a given country.

Example:
    To retrieve OpenAlex institutions for all countries, run the following command:

        $ python -m eic_case_studies.pipeline.openalex.oa_institutions_api --environment pypi run --max-workers 1 --iso_code all --save_to_s3 True

    This will iteratively save S3 files for each country in the data/oa/organisations folder.
"""

from metaflow import FlowSpec, step, Parameter, pypi_base  # pylint: disable=E0611


@pypi_base(
    packages={
        "pandas": "2.1.3",
        "requests": "2.31.0",
        "toolz": "0.12.0",
        "boto3": "1.33.5",
        "pyyaml": "6.0.1",
        "pyarrow": "14.0.1",
    },
    python="3.12.0",
)
class OaFlow(FlowSpec):
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
        import yaml  # pylint: disable=C0415

        if self.iso_code == "all":
            with open("eic_case_studies/config/oa.yaml", "rt", encoding="utf-8") as f:
                config = yaml.load(f.read(), Loader=yaml.FullLoader)
            self.iso_codes = config["COUNTRY_ISO"]  # pylint: disable=W0201
        else:
            self.iso_codes = [self.iso_code]  # pylint: disable=W0201
        self.next(self.get_institutions, foreach="iso_codes")

    @step
    def get_institutions(self):
        """
        Retrieves OpenAlex institutions for a given ISO code.
        """
        from getters.oa_institutions import get_oa_institutions  # pylint: disable=C0415
        from getters.s3io import S3DataManager  # pylint: disable=C0415
        import random  # pylint: disable=C0415

        self.oa_institutions = get_oa_institutions(  # pylint: disable=W0201
            self.input, timesleep=random.randint(0, 3)
        )

        if self.save_to_s3:
            s3dm = S3DataManager()
            s3dm.save_to_s3(
                self.oa_institutions, f"data/01_raw/oa/organisations/{self.input}.parquet"
            )

        self.next(self.join)

    @step
    def join(self, inputs):
        """
        Join the flows.

        Parameters:
        - inputs (List[OaFlow]): List of input flows.

        Returns:
        - oa_institutions (pd.DataFrame): Joined dataframe of OpenAlex institutions.
        """
        import pandas as pd  # pylint: disable=C0415

        self.oa_institutions = pd.concat(  # pylint: disable=W0201
            [input.oa_institutions for input in inputs]
        )
        self.next(self.end)

    @step
    def end(self):
        """
        End the flow.
        """
        pass  # pylint: disable=W0107


if __name__ == "__main__":
    OaFlow()
