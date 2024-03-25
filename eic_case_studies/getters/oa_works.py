"""
This module contains functions to retrieve OpenAlex works for a given institution.

Example:
    To retrieve OpenAlex works from Comenius University Bratislava, run the following command:

        $ python -m eic_case_studies.getters.oa_works --key "I56085075" --email <email>

    This will create a file called oa_institutions_I56085075.csv in the current directory.
"""

from typing import Dict, Sequence, Generator, Union
from time import sleep
import logging
import argparse
import requests
from requests.adapters import HTTPAdapter, Retry
import pandas as pd

from getters.s3io import S3DataManager


def get_cursor_url(
    filter_label: str,
    email: str,
    pubdate: str,
    filter_europe: bool = False,
    filter_condition: str = "institutions.id",
) -> str:
    """
    Creates a cursor URL for the OpenAlex API.

    Args:
        institution_id (str): The ID of the institution.
        email (str): The email of the user.
        pubdate (str): The earliest publication date to retrieve works from.
            Format: "YYYY-MM-DD".

    Returns:
        str: A cursor URL for the OpenAlex API.
    """
    base_url = "https://api.openalex.org/works"
    filter_param = ",".join(
        [
            f"filter={filter_condition}:{filter_label}",
            f"from_publication_date:{pubdate}",
        ]
    )
    if filter_europe:
        filter_param += ",authorships.institutions.continent:europe"
    cursor_param = "cursor={}"
    polite_param = f"mailto={email}"
    return f"{base_url}?{filter_param}&per-page=200&{cursor_param}&{polite_param}"


def _process_institution_object(
    institution_id: Union[str, Sequence[str]], chunk: bool = True
) -> Sequence[str]:
    """
    Processes the institution ID to make it suitable for the OpenAlex API.

    Args:
        institution_id (Union[str, Sequence[str]]): The ID of the institution.

    Returns:
        Sequence[str]: A sequence of institution IDs.
    """
    if isinstance(institution_id, str):
        return [institution_id]
    if chunk:
        return list(_chunk_institution_id(institution_id))
    return institution_id


def _chunk_institution_id(
    institution_id: Sequence[str],
) -> Generator[Sequence[str], None, None]:
    """Chunks the institution ID into groups of 50."""
    for i in range(0, len(institution_id), 50):
        yield "|".join(institution_id[i : i + 50])


def works_generator(
    filter_label: str,
    email: str,
    pubdate: str,
    filter_europe: bool = False,
    filter_condition: str = "institutions.id",
) -> Generator[Sequence[Dict[str, str]], None, None]:
    """
    Creates a generator for OpenAlex works for a given institution.

    Args:
        institution_id (str): The ID of the institution.
        email (str): The email of the user.

    Yields:
        Sequence[Dict[str, str]]: A sequence of OpenAlex works.
    """
    session = requests.Session()
    retries = Retry(
        total=5, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504]
    )
    session.mount("https://", HTTPAdapter(max_retries=retries))
    cursor = "*"

    # make a first call to get the total number of works
    cursor_url = get_cursor_url(
        filter_label=filter_label,
        email=email,
        pubdate=pubdate,
        filter_europe=filter_europe,
        filter_condition=filter_condition,
    )

    try:
        # make a first call to get the total number of works
        r = session.get(cursor_url.format("*"), timeout=60)
        data = r.json()
        logging.info("fetching data for institution %s", filter_label)
        total = data["meta"]["count"]
        num_calls = total // 200
        logging.info("Total results: %s, number of calls: %s", total, num_calls)

        while cursor:
            # collect the data
            cursor_url = get_cursor_url(
                filter_label=filter_label,
                email=email,
                pubdate=pubdate,
                filter_condition=filter_condition,
            )
            r = session.get(cursor_url.format(cursor), timeout=30)
            if r.status_code == 429:
                logging.warning(
                    "Too many requests (limit 5M daily). Sleeping for 1 hours."
                )
                sleep(3600)
            else:
                data = r.json()
                results = data.get("results")
                cursor = data["meta"].get("next_cursor", False)
                yield results
    except Exception as e:  # pylint: disable=W0703
        logging.error(
            "Error fetching data for %s - %s: %s", filter_condition, filter_label, e
        )
        return []


def get_oa_institution_works(
    institution_id: str, email: str, pubdate: str
) -> pd.DataFrame:
    """
    Retrieves Open Access works from a given institution using the provided
        institution ID and email.

    Args:
        institution_id (str): The ID of the institution to retrieve works from.
        email (str): The email associated with the institution.
        pubdate (str): The earliest publication date to retrieve works from.
            Format: "YYYY-MM-DD".
        timesleep (int, optional): The time to sleep between API requests in seconds.
            Defaults to 0.

    Returns:
        pd.DataFrame: A DataFrame containing the retrieved Open Access works.
    """
    oa_works = []  # pylint: disable=W0621
    institution_id = _process_institution_object(institution_id)
    for inst_ in institution_id:
        generator = works_generator(filter_label=inst_, email=email, pubdate=pubdate)
        for works in generator:

            logging.info(
                "Retrieved %d works from institution/s %s. Total: %d.",
                len(works),
                institution_id,
                len(oa_works),
            )
            if not works:
                continue
            oa_works.extend(works)
        logging.info(
            "Retrieved %d works from institution/s %s.", len(oa_works), institution_id
        )

        return pd.DataFrame(oa_works)


def get_oa_researcher_works(
    researcher_id: str, email: str, pubdate: str
) -> pd.DataFrame:
    """
    Retrieves Open Access works from a given researcher using the provided
        researcher ID and email.

    Args:
        researcher_id (str): The ID of the researcher to retrieve works from.
        email (str): The email associated with the researcher.
        pubdate (str): The earliest publication date to retrieve works from.
            Format: "YYYY-MM-DD".
        timesleep (int, optional): The time to sleep between API requests in seconds.
            Defaults to 0.

    Returns:
        pd.DataFrame: A DataFrame containing the retrieved Open Access works.
    """

    generator = works_generator(
        filter_label=researcher_id,
        email=email,
        pubdate=pubdate,
        filter_condition="authorships.author.orcid",
    )

    oa_works = []
    for works in generator:

        logging.info(
            "Retrieved %d works from researcher/s %s. Total: %d.",
            len(works),
            researcher_id,
            len(oa_works),
        )
        if not works:
            continue
        oa_works.extend(works)

    logging.info(
        "Retrieved %d works from researcher/s %s.", len(oa_works), researcher_id
    )

    return pd.DataFrame(oa_works)

def get_oa_researcher_counterfactual_works(
    topic_id: str, email: str, pubdate: str
) -> pd.DataFrame:
    """
    Retrieves OpenAlex works for a given topic, email, and publication date.

    Args:
        topic_id (str): The ID of the topic to filter the works.
        email (str): The email address associated with the researcher.
        pubdate (str): The publication date to filter the works.

    Returns:
        pd.DataFrame: A DataFrame containing the retrieved OpenAlex works.
    """

    generator = works_generator(
        filter_label=topic_id,
        email=email,
        pubdate=pubdate,
        filter_condition="primary_topic.id",
    )

    oa_works = []
    for works in generator:

        logging.info(
            "Retrieved %d works from topic/s %s. Total: %d.",
            len(works),
            topic_id,
            len(oa_works),
        )

        if not works:
            continue
        oa_works.extend(works)

    logging.info(
        "Retrieved %d works from topic/s %s.", len(oa_works), topic_id
    )

    return pd.DataFrame(oa_works)


def revert_abstract_index(abstract_inverted_index: Dict[str, Sequence[int]]) -> str:
    """Reverts the abstract inverted index to the original text.

    Args:
        abstract_inverted_index (Dict[str, Sequence[int]]): The abstract inverted index. 

    Returns:
        str: The original text.
    """    
    try:
        length_of_text = max([index for sublist in abstract_inverted_index.values() for index in sublist]) + 1
        recreated_text = [""] * length_of_text

        for word, indices in abstract_inverted_index.items():
            for index in indices:
                recreated_text[index] = word

        return " ".join(recreated_text)
    except Exception as e:  # pylint: disable=W0703
        return ""


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ids", nargs="+", required=True, help="An institution ID.")
    parser.add_argument(
        "--email",
        type=str,
        default="david.ampudia@nesta.org.uk",
        help="An email for politeness.",
    )
    args = parser.parse_args()
    oa_works = get_oa_institution_works(
        institution_id=args.key, email=args.email, pubdate="2010-01-01"
    )

    logging.info("Saving works from institution %s.", args.key)
    s3dm = S3DataManager()
    s3dm.save_to_s3(oa_works, f"data/01_raw/oa/organisations/works/{args.key}.parquet")

    # use io module to save local copy, avoid "Killed" error
    from io import BytesIO

    with BytesIO() as f:
        oa_works.to_parquet(f)
        f.seek(0)
        with open(f"eic_case_studies/data/{args.key}.parquet", "wb") as f2:
            f2.write(f.read())
