import pandas as pd
import pytest

from kyt_engine.data.validators import validate_columns


def test_validate_columns_pass():
    df = pd.DataFrame({"a": [1], "b": [2], "c": [3]})
    validate_columns(df, ["a", "b"], "test")


def test_validate_columns_fail():
    df = pd.DataFrame({"a": [1]})
    with pytest.raises(ValueError, match="не содержит обязательные колонки"):
        validate_columns(df, ["a", "b", "c"], "test_source")


def test_validate_columns_empty_df():
    df = pd.DataFrame()
    with pytest.raises(ValueError, match="не содержит обязательные колонки"):
        validate_columns(df, ["col1"], "test")


def test_validate_columns_extra_columns_ok():
    df = pd.DataFrame({"a": [1], "b": [2], "extra": [3]})
    validate_columns(df, ["a", "b"], "test")
