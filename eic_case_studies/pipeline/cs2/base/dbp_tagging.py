"""This module contains the code for tagging the Pathfinder proposals 
    with DBpedia annotations.

The DBpedia Spotlight API is used to annotate the text data in the 
    "Proposal Title" and "Proposal Abstract" columns of the Pathfinder 
    proposals dataset. The annotations are retrieved in JSON format and 
    stored in new columns in the dataset.

Example:
    To run the DBpediaAnnotationFlow flow, use the following command:

        $ python -m eic_case_studies.pipeline.cs2.base.dbp_tagging
            --save_to_local True
"""

import argparse
from functools import partial
import requests
from requests.adapters import HTTPAdapter, Retry
from tqdm import tqdm
import spacy
from getters.s3io import S3DataManager

# Load the English language model for spaCy
nlp = spacy.load("en_core_web_lg")

# Enable progress bars for pandas operations
tqdm.pandas()


def preprocess_text(text: str) -> str:
    """Preprocesses the text. It tokenizes the text, lemmatises it,
        removes stop words, punctuation, and non-alphabetic tokens,
        and keeps only nouns and adjectives.

    Args:
        text (str): The text to preprocess.

    Returns:
        str: The preprocessed text.
    """

    # Tokenize the text
    doc = nlp(text)

    tokens = [
        token.lemma_.lower()
        for token in doc
        # if not token.is_stop
        if not token.is_punct
        and token.is_alpha
        and token.pos_ in {"NOUN", "PROPN", "ADJ"}
    ]

    # additionally remove token that are GPE (location), PERSON, LOC
    bad_ents = [str(ent).lower() for ent in doc.ents if ent in {"GPE", "LOC", "PERSON"}]
    tokens = [token for token in tokens if token not in bad_ents]

    # Return the preprocessed text
    return " ".join(tokens)


def get_annotation(text: str, confidence: float = 0.25, support: int = 200):
    """
    Retrieves DBpedia annotations for the given text.

    Args:
        text (str): The input text to be annotated.
        confidence (float, optional): The confidence threshold for the annotations.
            Defaults to 0.25.
        support (int, optional): The support threshold for the annotations.
            Defaults to 200.

    Returns:
        list: A list of unique annotations extracted from the DBpedia resources.
    """
    # Preprocess the text
    text = preprocess_text(text)

    # If the preprocessed text is empty, return an empty list
    if not text:
        return []

    # Create a session with retry mechanism for HTTP requests
    session = requests.Session()
    retry = Retry(total=5, backoff_factor=0.1, status_forcelist=[500, 502, 503, 504])
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)

    # Set the base URL and headers for the DBpedia Spotlight API
    base_url = "https://api.dbpedia-spotlight.org/en/annotate"
    headers = {"Accept": "application/json"}

    # Set the data payload for the API request
    data = {
        "text": text,
        "confidence": confidence,
        "support": support,
        "policy": "whitelist",
    }

    # Send a POST request to the API and get the response
    response = session.post(base_url, data=data, headers=headers)
    response_json = response.json()

    # Extract the DBpedia annotations from the response
    resources = response_json.get("Resources", [])
    annotations = [
        resource.get("@URI")
        .replace("http://dbpedia.org/resource/", "")
        .replace("_", " ")
        for resource in resources
    ]

    # Return the list of unique annotations
    return list(set(annotations))


if __name__ == "__main__":
    # Create the argparse parser
    parser = argparse.ArgumentParser(
        description="Tag the Pathfinder proposals with DBpedia annotations."
    )

    # save arg
    parser.add_argument(
        "--save_to_local",
        type=bool,
        default=False,
        help="Whether to save the data locally.",
    )

    # Parse the arguments
    args = parser.parse_args()

    # Create an instance of the S3DataManager class
    s3dm = S3DataManager()

    # Load data from an S3 bucket
    pathfinder_proposals = s3dm.load_s3_data(
        "data/01_raw/he_2020/pathfinder/proposals/main.xlsx"
    )

    # Drop duplicate rows based on the "Proposal Number" column
    pathfinder_proposals = pathfinder_proposals.drop_duplicates(
        subset="Proposal Number"
    )

    # Apply the get_annotation function to the "Proposal Title" column
    pathfinder_proposals.loc[:, "title_annotations"] = pathfinder_proposals[
        "Proposal Title"
    ].progress_apply(partial(get_annotation, confidence=0.1, support=1000))

    # Apply the get_annotation function to the "Proposal Abstract" column
    pathfinder_proposals.loc[:, "abstract_annotations"] = pathfinder_proposals[
        "Proposal Abstract"
    ].progress_apply(partial(get_annotation, confidence=0.1, support=1000))

    # Save the results if specified
    if args.save_to_local:
        pathfinder_proposals.to_csv(
            "pathfinder_proposals_dbp_annotations.csv", index=False
        )
