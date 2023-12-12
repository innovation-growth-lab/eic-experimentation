"""
This module contains functions to retrieve OpenAlex works for a given institution.

Example:
    To retrieve OpenAlex works from Comenius University Bratislava, run the following command:

        $ python -m getters/oa_works.py --key "I74788687" --email <email>

    This will create a file called oa_institutions_I74788687.csv in the current directory.
"""

from typing import Dict, Sequence
from time import sleep
import logging
import argparse
import requests
import pandas as pd


def get_cursor_url(institution_id: str, email: str) -> str:
    """
    Creates a cursor URL for the OpenAlex API.

    Args:
        institution_id (str): The ID of the institution.
        email (str): The email of the user.

    Returns:
        str: A cursor URL for the OpenAlex API.
    """
    base_url = "https://api.openalex.org/works"
    filter_param = f"filter=institutions.id:{institution_id}"
    cursor_param = "cursor={}"
    polite_param = f"mailto={email}"
    return f"{base_url}?{filter_param}&{cursor_param}&{polite_param}"


def works_generator(institution_id: str, email: str) -> Sequence[Dict[str, str]]:
    """
    Creates a generator for OpenAlex works for a given institution.

    Args:
        institution_id (str): The ID of the institution.
        email (str): The email of the user.

    Yields:
        Sequence[Dict[str, str]]: A sequence of OpenAlex works.
    """
    cursor = "*"
    while cursor:
        cursor_url = get_cursor_url(institution_id=institution_id, email=email)
        r = requests.get(cursor_url.format(cursor), timeout=3_600)
        if r.status_code == 200:
            data = r.json()
            results = data.get("results")
            cursor = data["meta"].get("next_cursor", False)
            yield results
        elif r.status_code == 429:
            logging.warning("Too many requests (limit 100k daily). Sleeping for 12 hours.")
            sleep(12 * 60 * 60)
        else:
            raise ValueError(f"Error {r.status_code}.")



def get_oa_works(institution_id: str, email: str, timesleep: int = 0) -> pd.DataFrame:
    """
    Retrieves Open Access works from a given institution using the provided institution ID and email.

    Args:
        institution_id (str): The ID of the institution to retrieve works from.
        email (str): The email associated with the institution.
        timesleep (int, optional): The time to sleep between API requests in seconds. Defaults to 0.

    Returns:
        pd.DataFrame: A DataFrame containing the retrieved Open Access works.
    """
    oa_works = []
    generator = works_generator(institution_id=institution_id, email=email)
    for works in generator:
        logging.info(
            "Retrieved %d works from institution %s. Total: %d.",
            len(works),
            institution_id,
            len(oa_works),
        )
        sleep(timesleep)
        if not works:
            continue
        oa_works.extend(works)
    return pd.DataFrame(oa_works)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--key", type=str, default="I74788687", help="An institution ID. See https://docs.openalex.org/api-entities/institutions for more information.")
    parser.add_argument("--email", type=str, default=None, help="An email associated with the institution.")
    args = parser.parse_args()
    oa_works = get_oa_works(institution_id=args.key, email=args.email, timesleep=1)
    oa_works.to_csv(f"oa_works_{args.key}.csv", index=False)
