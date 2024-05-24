"""
This is a boilerplate pipeline 'generate_synthetic'
generated using Kedro 0.19.1
"""

from kedro.pipeline import Pipeline, pipeline, node
from .nodes import (
    data_processing,
    synthetic_project_inputs,
    get_synthetic_projects,
    organisation_randomiser,
    keyword_randomiser,
    zip_randomiser,
    shuffle_status,
    synthetise_data,
)
from ...settings import SOURCES


def create_pipeline(**kwargs) -> Pipeline:
    template_pipeline = Pipeline(
        [
            node(
                data_processing,
                inputs={"data": "raw"},
                outputs=["intermediate", "mapping"],
                tags=["get_synthetic_projects", "data_processing"]
            ),
            node(
                synthetic_project_inputs,
                inputs={"data": "intermediate"},
                outputs="synthetic_outputs",
                tags=["get_synthetic_projects", "synthetic_project_inputs", "abstracts"]
            ),
            node(
                get_synthetic_projects,
                inputs={"outputs": "synthetic_outputs"},
                outputs="abstracts",
                tags=["get_synthetic_projects", "abstracts"]
            ),
            node(
                organisation_randomiser,
                inputs={"data": "intermediate"},
                outputs="orgs",
                tags=["organisation_randomiser"]
            ),
            node(
                keyword_randomiser,
                inputs={"data": "intermediate"},
                outputs="keywords",
                tags=["keyword_randomiser"]
            ),
            node(
                zip_randomiser,
                inputs={"data": "intermediate"},
                outputs="zip_randomised",
                tags=["zip_randomiser", "synthetise_data"]
            ),
            node(
                shuffle_status,
                inputs={"data": "zip_randomised"},
                outputs="shuffled",
                tags=["shuffle_status", "synthetise_data"]
            ),
            node(
                synthetise_data,
                inputs={"data": "shuffled", "orgs_synthetics": "orgs",
                        "keywords_synthetics": "keywords", "content_synthetics": "abstracts"},
                outputs="synthetic",
                tags=["synthetise_data"]
            ),
        ]
    )

    pipelines = [
        pipeline(
            template_pipeline,
            namespace=source,
            tags=[source]
        )
        for source in SOURCES
    ]

    return sum(pipelines)
