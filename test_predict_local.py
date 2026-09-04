import json
from pathlib import Path

from fastapi.testclient import TestClient

from kyt_engine.api.app import app

client = TestClient(app)

feat = {f"f{i}": 0.0 for i in range(165)}
feat["in_degree"] = 2.0
feat["out_degree"] = 3.0
feat["value_mean"] = 50.0
feat["value_std"] = 1.0
feat["value_sum"] = 50.0
feat["gas_mean"] = 100.0
feat["hour_mean"] = 12.0
feat["in_degree"] = 2.0
feat["out_degree"] = 3.0
feat["unique_in"] = 1.0
feat["unique_out"] = 1.0
feat["unique_total"] = 2.0
feat["total_degree"] = 5.0
feat["in_out_ratio"] = 0.67
feat["day_span"] = 1.0
feat["interval_mean"] = 60.0
feat["interval_std"] = 10.0
feat["timestep_std"] = 0.0

payload = {
    "tx_id": "tx_test_1",
    "address": "0x999",
    "from_address": "0x001",
    "to_address": "0x002",
    "value": 50.0,
    "gas_price": 100.0,
    "gas_used": 21000.0,
    "timestamp": 1001000.0,
    "block_number": 2000,
    "features": feat,
}

with client:
    resp = client.post("/predict", json=payload)
    print("STATUS:", resp.status_code)
    print("BODY:", resp.text[:500])