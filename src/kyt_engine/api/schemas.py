from __future__ import annotations

from pydantic import BaseModel, Field


class TxRequest(BaseModel):
    tx_id: str
    address: str
    from_address: str
    to_address: str
    value: float
    gas_price: float = Field(..., gt=0)
    gas_used: float = Field(..., ge=0)
    timestamp: float
    block_number: int = Field(..., ge=0)
    features: dict[str, float] | None = None


class ReasonItem(BaseModel):
    feature: str
    value: float
    contribution: float


class PredictResponse(BaseModel):
    tx_id: str
    risk_score: float
    risk_zone: str
    triage_level: str
    lgbm_proba: float
    k_score: float
    vae_anomaly: float
    external_risk: float
    reasons: list[ReasonItem]


class HealthResponse(BaseModel):
    status: str
    models_loaded: list[str]