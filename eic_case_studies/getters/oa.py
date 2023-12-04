"""
This module contains functions to retrieve OpenAlex institutions for a given country.

Example:
    To retrieve OpenAlex institutions for Estonia, run the following command:

        $ python -m eic_case_studies.getters.oa --key EE

    This will create a file called oa_institutions_EE.csv in the current directory.
"""

from typing import Dict, Sequence
from time import sleep
import argparse
import requests
import pandas as pd


def get_cursor_url(iso_code: str) -> str:
    """Creates a cursor url for the OpenAlex API.

    Args:
        iso_code (str): The ISO code of the country.

    Returns:
        str: A cursor url for the OpenAlex API.
    """
    base_url = "https://api.openalex.org/institutions"
    filter_param = f"filter=country_code:{iso_code}"
    cursor_param = "cursor={}"
    select_param = "select=id,country_code"
    return f"{base_url}?{filter_param}&{cursor_param}&{select_param}"


def institutions_generator(iso_code: str) -> Sequence[Dict[str, str]]:
    """
    Creates a generator that yields a list of institutions from the OpenAlex API.
    It uses cursor pagination to get all the institutions.

    Args:
        iso_code (str): The ISO code of the country.

    Yields:
        Iterator[list]: A generator that yields a list of institutions from the OpenAlex API.
    """
    cursor = "*"
    while cursor:
        cursor_url = get_cursor_url(iso_code)
        r = requests.get(cursor_url.format(cursor), timeout=3_600)
        data = r.json()
        results = data.get("results")
        cursor = data["meta"].get("next_cursor", False)
        yield results


def get_oa_institutions(iso_code: str, timesleep: int = 0) -> pd.DataFrame:
    """
    Retrieves OpenAlex institutions for a given ISO code.

    Args:
        iso_code (str): The ISO code of the country.

    Returns:
        pd.DataFrame: A DataFrame containing the OpenAlex institutions.
    """
    oa_inst = []
    generator = institutions_generator(iso_code)
    for institutions in generator:
        sleep(timesleep)
        if not institutions:
            continue
        oa_inst.extend(institutions)
    return pd.DataFrame(oa_inst)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--key", type=str, help="ISO code of the country")
    args = parser.parse_args()
    oa_institutions = get_oa_institutions(iso_code=args.key, timesleep=0)
    oa_institutions.to_csv(f"oa_institutions_{args.key}.csv", index=False)
