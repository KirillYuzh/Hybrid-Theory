from pathlib import Path

import pandas as pd

from kyt_engine.data.validators import validate_columns, validate_file_exists

REQUIRED_COLUMNS = ["transaction_id", "timestamp", "amount"]

FILE_NAME = "openaml.parquet"


def load_openaml(data_dir: str | Path) -> pd.DataFrame:
    path = Path(data_dir) / FILE_NAME
    validate_file_exists(path)
    df = pd.read_parquet(path)
    validate_columns(df, REQUIRED_COLUMNS, FILE_NAME)
    return df
