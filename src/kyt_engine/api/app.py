import os
from pathlib import Path

import joblib
import pandas as pd

from fastapi import FastAPI, HTTPException

from kyt_engine.api.schemas import HealthResponse, PredictResponse, ReasonItem, TxRequest
from kyt_engine.core.audit import AuditLog
from kyt_engine.core.contracts import TxRecord
from kyt_engine.core.features import FeatureEngineer
from kyt_engine.core.pipeline import Pipeline
from kyt_engine.core.scorers import KScoreScorer, LightGBMScorer
from kyt_engine.core.sinks import ConsoleSink
from kyt_engine.core.triage import TriagePolicy, TriageConfig
from kyt_engine.models.kscore import KScoreCalculator
from kyt_engine.config import (
    DEFAULT_RISK_ZONE_THRESHOLDS,
    LIGHTGBM_WEIGHT,
    KSCORE_WEIGHT,
    VAE_WEIGHT,
    EXTERNAL_WEIGHT,
)


AUDIT_PATH = Path(os.environ.get("KYT_AUDIT_PATH", "data/audit/decisions.jsonl"))


def _build_pipeline() -> Pipeline:
    lgbm_model = _load_lightgbm_model()
    kscore_calc = KScoreCalculator()

    fe = FeatureEngineer()
    dummy_df = pd.DataFrame({
        "address": ["X"],
        "timestamp": [0],
        "value": [0.0],
        "gas_price": [0.1],
        "gas_used": [21000],
        "block_number": [1],
        "from_address": ["X"],
        "to_address": ["Y"],
    })
    fe.fit(dummy_df)

    lgbm_scorer = LightGBMScorer(lgbm_model, feature_names=list(fe.feature_names) if fe.feature_names else None)
    kscore_scorer = KScoreScorer(kscore_calc, feature_names=list(fe.feature_names) if fe.feature_names else [])

    audit = AuditLog(AUDIT_PATH)
    scorers: list = [s for s in [lgbm_scorer, kscore_scorer] if s is not None]

    return Pipeline(
        features=fe,
        scorers=scorers,
        triage=TriagePolicy(
            TriageConfig(
                close_threshold=DEFAULT_RISK_ZONE_THRESHOLDS[0],
                escalate_threshold=DEFAULT_RISK_ZONE_THRESHOLDS[1],
                confidence_high=0.9,
                confidence_low=0.7,
            )
        ),
        sinks=[ConsoleSink()],
        weights={
            "lightgbm": LIGHTGBM_WEIGHT,
            "kscore": KSCORE_WEIGHT,
            "vae": VAE_WEIGHT,
            "external": EXTERNAL_WEIGHT,
        },
        audit_log=audit,
    )


def _load_lightgbm_model() -> object:
    """Load LightGBM model from disk. Raises if not found."""
    model_dir = Path(os.environ.get("KYT_MODEL_DIR", "models"))
    for name in ("lightgbm_real.pkl", "lightgbm.pkl", "lightgbm_updated_1788251370.pkl"):
        path = model_dir / name
        if path.exists():
            return joblib.load(path)
    raise RuntimeError(f"LightGBM model not found in {model_dir}")


_pipeline: Pipeline | None = None


def get_pipeline() -> Pipeline:
    """Get or create the pipeline. Raises if model not found."""
    global _pipeline
    if _pipeline is None:
        _pipeline = _build_pipeline()
    return _pipeline


app = FastAPI(title="KYT Engine API", version="0.1.0")


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    try:
        pipeline = get_pipeline()
        models_loaded = [s.name for s in pipeline._scorers if hasattr(s, "name")]
        status = "ok"
    except RuntimeError:
        models_loaded = []
        status = "degraded"
    return HealthResponse(status=status, models_loaded=models_loaded)


@app.post("/predict", response_model=PredictResponse)
def predict(tx: TxRequest) -> PredictResponse:
    try:
        pipeline = get_pipeline()
    except RuntimeError:
        raise HTTPException(status_code=503, detail="Model pipeline not available")

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
    try:
        result = pipeline.score_tx(record)
    except RuntimeError:
        raise HTTPException(status_code=503, detail="Scoring failed")
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