# eic-case-studies init file

# Standard library imports
import logging
import logging.config
from pathlib import Path
from typing import Optional

# Third-party imports
import yaml
import warnings

# Suppress all warnings globally
# Note: Use this with caution as it might hide important warnings
warnings.filterwarnings("ignore")

# Constants
BUCKET_NAME = "eic-case-studies"
PUBLIC_DATA_FOLDER_NAME = "data"
PROJECT_DIR = Path(__file__).resolve().parents[1] / "eic_case_studies"
INFO_LOG_PATH = str(PROJECT_DIR / "info.log")
ERROR_LOG_PATH = str(PROJECT_DIR / "errors.log")

def get_yaml_config(file_path: Path) -> Optional[dict]:
    """
    Fetch yaml config and return as dict if it exists.
    Args:
        file_path (Path): The path to the YAML configuration file.
    Returns:
        Optional[dict]: The configuration dictionary if the file exists, otherwise None.
    """
    if isinstance(file_path, str):
        try:
            file_path = PROJECT_DIR / "config" / file_path
        except TypeError:
            raise TypeError("file_path must be a Path object or a string that completes a path.")
    if file_path.exists():
        with open(file_path, "rt", encoding="utf-8") as f:
            return yaml.load(f.read(), Loader=yaml.FullLoader)

# Read log configuration from the YAML file
_log_config_path = PROJECT_DIR / "config/logging.yaml"
_logging_config = get_yaml_config(_log_config_path)

# Configure logging
if _logging_config:
    logging.config.dictConfig(_logging_config)

# Define module-level logger
logger = logging.getLogger(__name__)