from pathlib import Path

import pandas as pd

from kyt_engine.data.validators import validate_file_exists

REQUIRED_COLUMNS = ["txId", "timestamp", "from_address", "to_address", "value", "label"]
ETH_FILE = "ethereum_txs.parquet"


def load_ethereum_data(data_dir: str | Path) -> pd.DataFrame:
    path = Path(data_dir) / ETH_FILE
    validate_file_exists(path)
    df = pd.read_parquet(path)
    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Ethereum data is missing required columns: {sorted(missing)}")
    return df


def load_ethereum_dataset(data_dir: str | Path) -> dict[str, pd.DataFrame]:
    df = load_ethereum_data(data_dir)
    return {"transactions": df}