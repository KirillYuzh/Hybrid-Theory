from pathlib import Path

import pandas as pd


def validate_file_exists(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Файл не найден: {path}")


def validate_columns(df: pd.DataFrame, required: list[str], source: str) -> None:
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(
            f"Источник «{source}» не содержит обязательные колонки: {sorted(missing)}"
        )
