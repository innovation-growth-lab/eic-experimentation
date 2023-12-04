"""
python -m eic_case_studies.pipeline.openalex.oa_api --environment pypi run --iso_code EE --save_to_s3 True
"""

from metaflow import FlowSpec, step, batch, Parameter, pypi_base


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
        import yaml

        if self.iso_code == "all":
            with open("config/oa.yaml", "rt", encoding="utf-8") as f:
                config = yaml.load(f.read(), Loader=yaml.FullLoader)
            self.iso_codes = config["ISO_CODES"]
        else:
            self.iso_codes = [self.iso_code]
        self.next(self.get_institutions, foreach="iso_codes")

    @step
    def get_institutions(self):
        """
        Retrieves OpenAlex institutions for a given ISO code.
        """
        from getters.oa import get_oa_institutions
        from getters.s3io import S3DataManager
        import random

        self.oa_institutions = get_oa_institutions(
            self.iso_code, timesleep=random.randint(0, 3)
        )

        if self.save_to_s3:
            s3dm = S3DataManager()
            s3dm.save_to_s3(
                self.oa_institutions, f"data/oa/organisations/{self.iso_code}.parquet"
            )

        self.next(self.join)

    @step
    def join(self, inputs):
        """
        Join the flows.
        """
        import pandas as pd
        
        self.oa_institutions = pd.concat([input.oa_institutions for input in inputs])
        self.next(self.end)

    @step
    def end(self):
        """
        End the flow.
        """
        pass


if __name__ == "__main__":
    OaFlow()
