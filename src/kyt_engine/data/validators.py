from pathlib import Path

import pandas as pd


def validate_file_exists(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")


def validate_columns(df: pd.DataFrame, required: list[str], source: str) -> None:
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(
            f"Source '{source}' is missing required columns: {sorted(missing)}"
        )


def map_aml_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Map AML label strings to integers

    Maps: illicit -> 1, licit -> 0.
    Leaves already-integer labels unchanged.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing a 'label' column with AML labels.
    """
    df = df.copy()
    if df["label"].dtype == object or str(df["label"].dtype) == "string":
        df["label"] = df["label"].map({"illicit": 1, "licit": 0}).fillna(0).astype(int)
    else:
        df["label"] = df["label"].astype(int)
    return df