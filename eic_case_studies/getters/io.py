
"""
Module for local input/output operations.
"""

from fnmatch import fnmatch
from decimal import Decimal
import json
import pandas as pd
import numpy as np

from eic_case_studies import logger

class LocalDataManager:
    """
    Class to manage data storage and retrieval from local file system.
    """

    class CustomJE(json.JSONEncoder):
        """
        Custom JSON encoder to handle special data types like Decimal and numpy objects.
        """
        def default(self, o):
            if isinstance(o, Decimal):
                return float(o)
            if isinstance(o, np.integer):
                return int(o)
            if isinstance(o, np.floating):
                return float(o)
            if isinstance(o, np.ndarray):
                return o.tolist()
            return super().default(o)

    def load_data(self, file_name: str) -> pd.DataFrame:
        """
        Loads data from a local file path.

        Args:
            file_name (str): Local path to data.

        Returns:
            pd.DataFrame: Loaded data.
        """
        if fnmatch(file_name, "*.csv"):
            return pd.read_csv(file_name)
        elif fnmatch(file_name, "*.xlsx"):
            return pd.read_excel(file_name, engine='openpyxl')
        else:
            logger.error('%s extension must be "*.csv" or "*.xlsx"', file_name)

    def load_json_dict(self, file_name: str) -> dict:
        """
        Loads a dictionary stored in a local JSON file.

        Args:
            file_name (str): Local path to JSON file.

        Returns:
            dict: Loaded dictionary.
        """
        if fnmatch(file_name, "*.json"):
            with open(file_name, "r", encoding="utf-8") as file:
                return json.load(file)
        else:
            logger.error('%s extension must be "*.json"', file_name)

    def save_json_dict(self, dictionary: dict, file_name: str):
        """
        Saves a dictionary to a local JSON file.

        Args:
            dictionary (dict): The dictionary to be saved.
            file_name (str): Local path to JSON file.
        """
        if fnmatch(file_name, "*.json"):
            logger.error('%s extension must be "*.json"', file_name)
            with open(file_name, "w", encoding="utf-8") as file:
                json.dump(dictionary, file, cls=self.CustomJE)
        else:
            logger.error('%s extension must be "*.json"', file_name)

    def load_txt_lines(self, file_name: str) -> list:
        """
        Loads lines from a local text file.

        Args:
            file_name (str): Local path to text file.

        Returns:
            list: List of lines from the file.
        """
        txt_list = []
        if fnmatch(file_name, "*.txt"):
            with open(file_name, encoding="utf-8") as file:
                txt_list = [line.rstrip() for line in file]
        else:
            logger.error('%s extension must be "*.txt"', file_name)

        return txt_list
