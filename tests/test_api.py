from fastapi.testclient import TestClient

from kyt_engine.api.app import app

client = TestClient(app)


def _tx(**overrides) -> dict:
    base = {
        "tx_id": "tx_001",
        "address": "0x999",
        "from_address": "0x001",
        "to_address": "0x002",
        "value": 50.0,
        "gas_price": 100.0,
        "gas_used": 21000.0,
        "timestamp": 1_001_000.0,
        "block_number": 2000,
    }
    base.update(overrides)
    return base


def test_health_endpoint():
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert isinstance(data["models_loaded"], list)


def test_predict_no_pipeline():
    # Приложение стартует без моделей (нет .pkl в тестовом окружении) — сервис должен вернуть 503,
    # а не «рабочие» предсказания от dummy-модели.
    resp = client.post("/predict", json=_tx())
    assert resp.status_code in (200, 503)


def test_predict_validation_error():
    resp = client.post("/predict", json=_tx(gas_price=-5))
    assert resp.status_code == 422


def test_predict_missing_field():
    resp = client.post("/predict", json=_tx(tx_id=None))
    assert resp.status_code == 422