from __future__ import annotations

import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from fastapi import FastAPI, HTTPException

from kyt_engine.api.schemas import HealthResponse, PredictResponse, ReasonItem, TxRequest
from kyt_engine.core.audit import AuditLog
from kyt_engine.core.contracts import TxRecord, ScoreResult, AuditRecord
from kyt_engine.core.features import FeatureEngineer
from kyt_engine.core.pipeline import Pipeline
from kyt_engine.core.scorers import ExternalScorer, KScoreScorer, LightGBMScorer
from kyt_engine.core.sinks import ConsoleSink
from kyt_engine.core.triage import TriagePolicy
from kyt_engine.models.kscore import KScoreCalculator


AUDIT_PATH = Path(os.environ.get("KYT_AUDIT_PATH", "data/audit/decisions.jsonl"))

_pipeline: Pipeline | None = None


def _build_pipeline() -> Pipeline:
    lgbm_model = _load_lightgbm_model()
    kscore_calc = KScoreCalculator()

    fe = FeatureEngineer()
    try:
        fe.fit(pd.DataFrame({
            "address": ["X"],
            "timestamp": [0],
            "value": [0.0],
            "gas_price": [0.1],
            "gas_used": [21000],
            "block_number": [1],
            "from_address": ["X"],
            "to_address": ["Y"],
        }))
    except Exception:
        pass

    lgbm_scorer = LightGBMScorer(lgbm_model, feature_names=list(fe.feature_names) if fe.feature_names else None)
    kscore_scorer = KScoreScorer(kscore_calc, feature_names=list(fe.feature_names) if fe.feature_names else [])

    audit = AuditLog(AUDIT_PATH)
    scorers: list = [s for s in [lgbm_scorer, kscore_scorer] if s is not None]

    global _pipeline
    _pipeline = Pipeline(
        features=fe,
        scorers=scorers,
        triage=TriagePolicy(),
        sinks=[ConsoleSink()],
        weights={"lightgbm": 0.5, "kscore": 0.2, "vae": 0.0, "external": 0.15},
        audit_log=audit,
    )
    return _pipeline


def _load_lightgbm_model() -> object:
    """Load LightGBM model from disk. Raises if not found."""
    model_dir = Path(os.environ.get("KYT_MODEL_DIR", "models"))
    for name in ("lightgbm_real.pkl", "lightgbm.pkl", "lightgbm_updated_1788251370.pkl"):
        path = model_dir / name
        if path.exists():
            return joblib.load(path)
    raise RuntimeError(f"LightGBM model not found in {model_dir}")


# Build pipeline at import time; if model not found, start in degraded mode
try:
    _build_pipeline()
except RuntimeError:
    _pipeline = None

app = FastAPI(title="KYT Engine API", version="0.1.0")


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    models_loaded = []
    if _pipeline is not None:
        models_loaded = [s.name for s in _pipeline._scorers if hasattr(s, "name")]
    status = "ok" if _pipeline is not None else "degraded"
    return HealthResponse(status=status, models_loaded=models_loaded)


@app.post("/predict", response_model=PredictResponse)
def predict(tx: TxRequest) -> PredictResponse:
    if _pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized. Ensure LightGBM model is in KYT_MODEL_DIR.")

    record = TxRecord(
        tx_id=tx.tx_id,
        address=tx.address,
        from_address=tx.from_address,
        to_address=tx.to_address,
        value=tx.value,
        gas_price=tx.gas_price,
        gas_used=tx.gas_used,
        timestamp=tx.timestamp,
        block_number=tx.block_number,
        features=tx.features,
    )
    result = _pipeline.score_tx(record)
    return PredictResponse(
        tx_id=result.tx_id,
        risk_score=round(result.risk_score, 6),
        risk_zone=result.risk_zone,
        triage_level=result.triage_level,
        lgbm_proba=round(result.lgbm_proba, 6),
        k_score=round(result.k_score, 6),
        vae_anomaly=round(result.vae_anomaly, 6),
        external_risk=round(result.external_risk, 6),
        reasons=[ReasonItem(feature=str(r["feature"]), value=float(r["value"]), contribution=float(r["contribution"])) for r in result.reasons],
    )