import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from kyt_engine.api.app import app, get_models, load_model, set_feature_engineer
from kyt_engine.features.engine import FeatureEngineer
from kyt_engine.models.lightgbm_model import LightGBMClassifier


def _make_tx_df(n_per_addr: int = 50, n_addrs: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    frames = []
    base_ts = 1_000_000
    for i in range(n_addrs):
        addr = f"0x{i:03X}"
        frames.append(pd.DataFrame({
            "address": [addr] * n_per_addr,
            "from_address": [f"0x{rng.integers(0, 20):03X}" for _ in range(n_per_addr)],
            "to_address": [f"0x{rng.integers(0, 20):03X}" for _ in range(n_per_addr)],
            "value": rng.uniform(0, 100, n_per_addr),
            "gas_price": rng.uniform(10, 200, n_per_addr),
            "gas_used": rng.uniform(21000, 100000, n_per_addr),
            "timestamp": np.linspace(base_ts, base_ts + n_per_addr * 60, n_per_addr),
            "block_number": np.arange(1000 + i * n_per_addr, 1000 + (i + 1) * n_per_addr),
        }))
    return pd.concat(frames, ignore_index=True)


def _setup_models():
    get_models().clear()
    df = _make_tx_df()
    fe = FeatureEngineer()
    features = fe.fit_transform(df)
    labels = pd.Series(np.random.default_rng(42).integers(0, 2, size=len(features)), index=features.index)
    model = LightGBMClassifier(n_estimators=10, num_leaves=8, min_child_samples=5)
    model.fit(features, labels)
    load_model("lgbm", model)
    set_feature_engineer(fe)


client = TestClient(app)


def test_health_no_models():
    get_models().clear()
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["models_loaded"] is False


def test_health_with_models():
    _setup_models()
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["models_loaded"] is True
    assert "lgbm" in data["model_names"]


def test_predict_endpoint():
    _setup_models()
    tx = {
        "address": "0x999",
        "from_address": "0x001",
        "to_address": "0x002",
        "value": 50.0,
        "gas_price": 100.0,
        "gas_used": 21000.0,
        "timestamp": 1_001_000.0,
        "block_number": 2000,
    }
    resp = client.post("/predict", json=tx)
    assert resp.status_code == 200
    data = resp.json()
    assert data["address"] == "0x999"
    assert 0.0 <= data["risk_score"] <= 1.0
    assert len(data["reasons"]) == 3


def test_predict_no_models():
    get_models().clear()
    tx = {
        "address": "0x999",
        "from_address": "0x001",
        "to_address": "0x002",
        "value": 50.0,
        "gas_price": 100.0,
        "gas_used": 21000.0,
        "timestamp": 1_001_000.0,
        "block_number": 2000,
    }
    resp = client.post("/predict", json=tx)
    assert resp.status_code == 503


def test_batch_predict_endpoint():
    _setup_models()
    txs = [
        {
            "address": f"0x{i:03X}",
            "from_address": "0x001",
            "to_address": "0x002",
            "value": float(i * 10),
            "gas_price": 100.0,
            "gas_used": 21000.0,
            "timestamp": 1_001_000.0 + i * 60,
            "block_number": 2000 + i,
        }
        for i in range(3)
    ]
    resp = client.post("/batch-predict", json={"transactions": txs})
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["results"]) == 3
    for r in data["results"]:
        assert 0.0 <= r["risk_score"] <= 1.0
        assert len(r["reasons"]) == 3
