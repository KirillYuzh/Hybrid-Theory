import numpy as np
import pandas as pd
import pytest

from kyt_engine.features.base import extract_base_features
from kyt_engine.features.behavioral import extract_behavioral_features
from kyt_engine.features.engine import FeatureEngineer


def _make_tx_df(n: int = 20, address: str = "0xABC") -> pd.DataFrame:
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "address": [address] * n,
        "from_address": [f"0x{rng.integers(0, 100):03X}" for _ in range(n)],
        "to_address": [f"0x{rng.integers(0, 100):03X}" for _ in range(n)],
        "value": rng.uniform(0, 100, n),
        "gas_price": rng.uniform(10, 200, n),
        "gas_used": rng.uniform(21000, 100000, n),
        "timestamp": np.linspace(1_000_000, 1_000_000 + n * 60, n),
        "block_number": np.arange(1000, 1000 + n),
    })


def test_extract_base_features_shape():
    df = _make_tx_df(30)
    result = extract_base_features(df)
    assert result.shape[0] == 1
    assert result.shape[1] == 165


def test_extract_base_features_empty_df():
    df = pd.DataFrame()
    result = extract_base_features(df)
    assert result.empty


def test_extract_base_features_two_addresses():
    df1 = _make_tx_df(20, "0xAAA")
    df2 = _make_tx_df(20, "0xBBB")
    df = pd.concat([df1, df2], ignore_index=True)
    result = extract_base_features(df)
    assert result.shape[0] == 2
    assert result.shape[1] == 165


def test_extract_behavioral_features_shape():
    df = _make_tx_df(30)
    result = extract_behavioral_features(df, global_gas_median=50.0)
    assert result.shape[0] == 1
    assert result.shape[1] == 26


def test_extract_behavioral_features_empty_df():
    df = pd.DataFrame()
    result = extract_behavioral_features(df)
    assert result.empty


def test_feature_engineer_fit_transform():
    df = _make_tx_df(30)
    fe = FeatureEngineer()
    result = fe.fit_transform(df)
    assert result.shape[0] == 1
    assert result.shape[1] == 191
    assert fe.n_features == 191
    assert len(fe.feature_names) == 191


def test_feature_engineer_transform_before_fit():
    df = _make_tx_df(30)
    fe = FeatureEngineer()
    with pytest.raises(RuntimeError, match="must be fitted"):
        fe.transform(df)


def test_feature_engineer_fit_then_transform():
    df = _make_tx_df(30)
    fe = FeatureEngineer()
    fe.fit(df)
    result = fe.transform(df)
    assert result.shape[0] == 1
    assert result.shape[1] == 191
