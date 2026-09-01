from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from kyt_engine.features.engine import FeatureEngineer
from kyt_engine.models.ensemble import StackingEnsemble

app = FastAPI(title="KYT Engine API", version="0.1.0")

_models: dict[str, Any] = {}
_feature_engineer: FeatureEngineer | None = None


class Transaction(BaseModel):
    address: str
    from_address: str
    to_address: str
    value: float
    gas_price: float
    gas_used: float
    timestamp: float
    block_number: int


class PredictRequest(BaseModel):
    transactions: list[dict[str, Any]]

    @property
    def is_too_large(self) -> bool:
        return len(self.transactions) > 1000


class Reason(BaseModel):
    feature: str
    value: float
    contribution: float


class PredictResponse(BaseModel):
    address: str
    risk_score: float
    reasons: list[Reason]


class BatchPredictResponse(BaseModel):
    results: list[PredictResponse]


class HealthResponse(BaseModel):
    status: str
    models_loaded: bool
    model_names: list[str]


def load_model(name: str, model: Any) -> None:
    _models[name] = model


def set_feature_engineer(fe: FeatureEngineer) -> None:
    global _feature_engineer
    _feature_engineer = fe


def get_models() -> dict[str, Any]:
    return _models


def _prepare_df(transactions: list[dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(transactions)
    required = ["address", "from_address", "to_address", "value", "gas_price", "gas_used", "timestamp", "block_number"]
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    return df


def _compute_reasons(feature_names: list[str], feature_values: np.ndarray, importances: np.ndarray) -> list[Reason]:
    top_idx = np.argsort(importances)[::-1][:3]
    reasons = []
    for idx in top_idx:
        reasons.append(Reason(
            feature=feature_names[idx],
            value=float(feature_values[idx]),
            contribution=float(importances[idx]),
        ))
    return reasons


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    names = list(_models.keys())
    return HealthResponse(
        status="ok",
        models_loaded=len(_models) > 0,
        model_names=names,
    )


@app.post("/predict", response_model=PredictResponse)
def predict(transaction: Transaction) -> PredictResponse:
    if not _models:
        raise HTTPException(status_code=503, detail="No models loaded")

    df = _prepare_df([transaction.model_dump()])

    if _feature_engineer is None:
        raise HTTPException(status_code=503, detail="FeatureEngineer not configured")

    features = _feature_engineer.transform(df)
    feature_names = list(features.columns)
    feature_values = features.values[0]

    ensemble: StackingEnsemble = _models.get("ensemble")
    if ensemble is None:
        first_model = next(iter(_models.values()))
        proba = first_model.predict_proba(features)
        risk_score = float(proba[0, 1])
        importances = np.abs(feature_values)
    else:
        proba = ensemble.predict_proba(features)
        risk_score = float(proba[0, 1])
        importances = np.abs(feature_values)

    reasons = _compute_reasons(feature_names, feature_values, importances)

    return PredictResponse(
        address=transaction.address,
        risk_score=round(risk_score, 6),
        reasons=reasons,
    )


@app.post("/batch-predict", response_model=BatchPredictResponse)
def batch_predict(request: PredictRequest) -> BatchPredictResponse:
    if not _models:
        raise HTTPException(status_code=503, detail="No models loaded")

    if request.is_too_large:
        raise HTTPException(status_code=413, detail="Batch size limited to 1000 transactions")

    if _feature_engineer is None:
        raise HTTPException(status_code=503, detail="FeatureEngineer not configured")

    df = _prepare_df(request.transactions)
    features = _feature_engineer.transform(df)
    feature_names = list(features.columns)

    ensemble: StackingEnsemble = _models.get("ensemble")
    if ensemble is None:
        first_model = next(iter(_models.values()))
        proba = first_model.predict_proba(features)
    else:
        proba = ensemble.predict_proba(features)

    results = []
    for i, tx in enumerate(request.transactions):
        feature_values = features.values[i]
        risk_score = float(proba[i, 1])
        importances = np.abs(feature_values)
        reasons = _compute_reasons(feature_names, feature_values, importances)
        results.append(PredictResponse(
            address=tx["address"],
            risk_score=round(risk_score, 6),
            reasons=reasons,
        ))

    return BatchPredictResponse(results=results)
