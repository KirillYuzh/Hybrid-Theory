"""
Minimal Iceberg store — deferred to "Should Have" phase.

This module is intentionally sparse. Iceberg's MemoryCatalog requires
namespace setup that complicates local development. Use pickle-based
persistence until Iceberg namespace handling is resolved.

When to add it:
    - Iceberg is required for production data pipelines
    - Namespace management is configured (SQL catalog or resolved memory catalog)
    - Then: uncomment the _init_tables code and add iceberg as a hard dep.
"""

from typing import Optional
from datetime import datetime, timezone
import pickle
import pathlib


class IcebergStore:
    """Placeholder — replace with real Iceberg when needed.

    Priority 7 (Should Have): Add Iceberg read/write for features/predictions/models.
    Skip until production Iceberg deployment.
    """

    def __init__(self) -> None:
        self._data_path = pathlib.Path("data/iceberg_cache")
        self._data_path.mkdir(parents=True, exist_ok=True)
        self._cache: dict[str, dict] = {}

    # ---- Features ----

    def save_features(self, address: str, features_dict: dict[str, float]) -> None:
        """Save feature vector (pickle cache for now)."""
        cache_file = self._data_path / f"features_{address}.pkl"
        with open(cache_file, "wb") as f:
            pickle.dump(features_dict, f)
        self._cache[address] = features_dict

    def load_features(self, address: str) -> dict[str, float]:
        """Load feature vector."""
        if address in self._cache:
            return self._cache[address]
        cache_file = self._data_path / f"features_{address}.pkl"
        if cache_file.exists():
            with open(cache_file, "rb") as f:
                data = pickle.load(f)
            self._cache[address] = data
            return data
        return {}

    # ---- Predictions ----

    def save_prediction(
        self,
        tx_id: str,
        address: str,
        risk_score: float,
        risk_zone: str,
        triage_level: str,
        lgbm_proba: float,
        k_score: float,
        vae_anomaly: float,
        external_risk: float,
    ) -> None:
        """Save prediction result (pickle cache for now)."""
        cache_file = self._data_path / f"predictions_{tx_id}.pkl"
        record = {
            "tx_id": tx_id,
            "address": address,
            "risk_score": risk_score,
            "risk_zone": risk_zone,
            "triage_level": triage_level,
            "lgbm_proba": lgbm_proba,
            "k_score": k_score,
            "vae_anomaly": vae_anomaly,
            "external_risk": external_risk,
        }
        with open(cache_file, "wb") as f:
            pickle.dump(record, f)
        self._cache[f"pred_{tx_id}"] = record

    def load_predictions(self, tx_id: str) -> Optional[dict]:
        """Load prediction result."""
        if f"pred_{tx_id}" in self._cache:
            return self._cache[f"pred_{tx_id}"]
        cache_file = self._data_path / f"predictions_{tx_id}.pkl"
        if cache_file.exists():
            with open(cache_file, "rb") as f:
                data = pickle.load(f)
            self._cache[f"pred_{tx_id}"] = data
            return data
        return None

    # ---- Models ----

    def save_model(self, model_name: str, version: str, metrics: dict[str, float]) -> None:
        """Save model metadata (pickle cache for now)."""
        cache_file = self._data_path / f"model_{model_name}_{version}.pkl"
        record = {"model_name": model_name, "version": version, "metrics": metrics}
        with open(cache_file, "wb") as f:
            pickle.dump(record, f)
        self._cache[f"model_{model_name}_{version}"] = record

    def load_models(self) -> list[dict]:
        """List all registered models."""
        result: list[dict] = []
        for key, value in self._cache.items():
            if key.startswith("model_"):
                result.append(value)
        return result


__all__ = ["IcebergStore"]