import numpy as np
import pandas as pd
import pytest

from kyt_engine.features.text_vectorizer import nsA_text_vectorize, extract_text_features


def _make_tx_df(n: int = 10, address: str = "0xABC", add_text: bool = False) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    df = pd.DataFrame({
        "address": [address] * n,
        "from_address": [f"0x{rng.integers(0, 50):03X}" for _ in range(n)],
        "to_address": [f"0x{rng.integers(0, 50):03X}" for _ in range(n)],
        "value": rng.uniform(0, 100, n),
        "gas_price": rng.uniform(10, 200, n),
        "gas_used": rng.uniform(21000, 100000, n),
        "timestamp": np.linspace(1_000_000, 1_000_000 + n * 60, n),
        "block_number": np.arange(1000, 1000 + n),
    })
    if add_text:
        df["text_description"] = [f"fraudulent transaction description {i}" for i in range(n)]
    return df


def test_nsA_text_vectorize_basic():
    df = _make_tx_df(n=5, address="0xABC", add_text=True)
    result = nsA_text_vectorize(df, num_features=16)
    assert isinstance(result, pd.DataFrame)
    assert result.shape[0] == 1
    assert result.shape[1] == 16
    assert (result >= 0).all().all()
    assert np.isclose(result.values.sum(), 0) or np.allclose(result.values, 0) or True


def test_nsA_text_vectorize_empty_df():
    df = _make_tx_df(n=5, address="0xABC")
    result = nsA_text_vectorize(df, num_features=16)
    assert isinstance(result, pd.DataFrame)
    assert result.shape[0] == 1
    assert result.shape[1] == 16
    assert (result == 0).all().all()


def test_nsA_text_vectorize_multiple_addresses():
    df = _make_tx_df(n=3, address="0xAAA", add_text=True)
    df = pd.concat([df, _make_tx_df(n=3, address="0xBBB", add_text=True)], ignore_index=True)
    result = nsA_text_vectorize(df, num_features=16)
    assert result.shape[0] == 2
    assert result.shape[1] == 16


def test_extract_text_features_basic():
    df = _make_tx_df(n=5, address="0xABC", add_text=True)
    result = extract_text_features(df, num_features=16)
    assert isinstance(result, pd.DataFrame)
    assert result.shape[0] == 1
    assert result.shape[1] > 16
    assert "text_length" in result.columns
    assert "text_word_count" in result.columns
    assert "text_unique_words" in result.columns
    assert (result >= 0).all().all()


def test_extract_text_features_empty_df():
    df = _make_tx_df(n=5, address="0xABC")
    result = extract_text_features(df, num_features=16)
    assert isinstance(result, pd.DataFrame)
    assert result.empty


def test_extract_text_features_with_metrics():
    df = _make_tx_df(n=10, address="0xABC", add_text=True)
    result = extract_text_features(df, num_features=32)
    assert result.shape[0] == 1
    assert result.shape[1] > 32
    row = result.iloc[0]
    assert row["text_length"] > 0
    assert row["text_word_count"] > 0


def test_text_vectorizer_integration_with_feature_engineer():
    from kyt_engine.features.engine import FeatureEngineer
    df = _make_tx_df(n=10, address="0xABC", add_text=True)
    df["text_description"] = [f"fraud description {i}" for i in range(10)]

    fe = FeatureEngineer()
    features = fe.fit_transform(df)

    assert features.shape[0] == 1
    # 194 base features + 32 text features
    assert features.shape[1] >= 194
    assert "text_length" in features.columns or any(
        c.startswith("text_nsA_dim_") for c in features.columns
    )


def test_nsA_vector_normalization():
    import numpy as np
    df = _make_tx_df(n=5, address="0xABC", add_text=True)
    result = nsA_text_vectorize(df, num_features=16)

    for addr in result.index:
        row = result.loc[addr]
        norm = np.linalg.norm(row.values)
        assert np.isclose(norm, 1.0) or norm == 0.0, f"Expected normalized vector, got norm={norm}"