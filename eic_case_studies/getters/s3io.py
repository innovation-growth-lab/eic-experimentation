
"""
Module for S3 input/output operations.
"""

from fnmatch import fnmatch
from decimal import Decimal
import io
import logging
import json
import pickle
import gzip
import boto3
import numpy as np
import pandas as pd


class S3DataManager:
    """
    Class to manage data storage and retrieval from S3.
    """

    def __init__(self):
        self.s3 = boto3.resource("s3")
        self.bucket_name = "eic-case-studies"

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

    def save_to_s3(self, output_var, output_file_dir):
        """
        Save data to an S3 bucket.

        output_var: Data to be saved.
        output_file_dir: S3 path (including file name) where data will be saved.
        """
        obj = self.s3.Object(self.bucket_name, output_file_dir)

        if fnmatch(output_file_dir, "*.csv"):
            csv_buffer = io.StringIO()
            output_var.to_csv(csv_buffer, index=False)
            # Move to the beginning of the buffer
            csv_buffer.seek(0)
            # Upload the buffer content to S3
            obj.put(Body=csv_buffer.getvalue())
        elif fnmatch(output_file_dir, "*.parquet"):
            output_var.to_parquet(f"s3://{self.bucket_name}/{output_file_dir}", index=False)
        elif fnmatch(output_file_dir, "*.pkl") or fnmatch(output_file_dir, "*.pickle"):
            obj.put(Body=pickle.dumps(output_var))
        elif fnmatch(output_file_dir, "*.gz"):
            obj.put(Body=gzip.compress(json.dumps(output_var).encode()))
        elif fnmatch(output_file_dir, "*.txt"):
            obj.put(Body=output_var)
        else:
            obj.put(Body=json.dumps(output_var, cls=self.CustomJE))

        logging.info("Saved to s3://%s/%s ...", self.bucket_name, output_file_dir)

    def load_s3_json(self, file_name):
        """
        Load a JSON file from S3.

        file_name: S3 key to load.
        """
        obj = self.s3.Object(self.bucket_name, file_name)
        file = obj.get()["Body"].read().decode()
        return json.loads(file)

    def load_prodigy_jsonl_s3_data(self, file_name):
        """
        Load Prodigy jsonl formatted data from S3.

        file_name: S3 key to load.
        """
        obj = self.s3.Object(self.bucket_name, file_name)
        if fnmatch(file_name, "*.jsonl"):
            file = obj.get()["Body"].read().decode()
            return [json.loads(item) for item in file.strip().split("\n")]

    def load_s3_data(self, file_name):
        """
        Load data from S3 based on file extension.

        file_name: S3 key to load.
        """
        obj = self.s3.Object(self.bucket_name, file_name)
        if fnmatch(file_name, "*.jsonl.gz"):
            with gzip.GzipFile(fileobj=obj.get()["Body"]) as file:
                return [json.loads(line) for line in file]
        elif fnmatch(file_name, "*.jsonl"):
            file = obj.get()["Body"].read().decode()
            return [json.loads(line) for line in file]
        elif fnmatch(file_name, "*.json.gz"):
            with gzip.GzipFile(fileobj=obj.get()["Body"]) as file:
                return json.load(file)
        elif fnmatch(file_name, "*.json"):
            file = obj.get()["Body"].read().decode()
            return json.loads(file)
        elif fnmatch(file_name, "*.csv"):
            data = io.BytesIO(obj.get()['Body'].read())
            return pd.read_csv(data)
        elif fnmatch(file_name, "*.xlsx"):
            data = io.BytesIO(obj.get()['Body'].read())
            return pd.read_excel(data, engine="openpyxl")
        elif fnmatch(file_name, "*.parquet"):
            return pd.read_parquet(f"s3://{self.bucket_name}/{file_name}")
        elif fnmatch(file_name, "*.pkl") or fnmatch(file_name, "*.pickle"):
            file = obj.get()["Body"].read().decode()
            return pickle.loads(file)
        else:
            logging.error("Unsupported file type for S3 data loading.")

    def get_s3_data_paths(self, root, file_types):
        """
        Get all paths to specific file types in a S3 root location.

        root: The root folder to look for files in.
        file_types: List of file types to look for.
        """
        if isinstance(file_types, str):
            file_types = [file_types]

        bucket = self.s3.Bucket(self.bucket_name)

        s3_keys = []
        for files in bucket.objects.filter(Prefix=root):
            key = files.key
            if any(fnmatch(key, pattern) for pattern in file_types):
                s3_keys.append(key)

        return s3_keys

    def load_s3_file(self, file_path):
        """
        Load a file either from the S3 bucket or locally.

        file_path: Path of the file to be loaded.
        """
        return self.load_s3_data(file_path)
