from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import pandas as pd
import numpy as np


class TransactionInput(BaseModel):
    tx_id: int
    from_address: str
    to_address: str
    value: float
    gas_price: float
    gas_used: int
    timestamp: int
    block_number: int
    features: Optional[dict] = None


class BatchInput(BaseModel):
    transactions: List[TransactionInput]


class RiskOutput(BaseModel):
    tx_id: int
    risk_score: float
    risk_zone: str
    triage_level: str
    lgbm_proba: float
    k_score: float
    vae_anomaly: float
    external_risk: float
    top_reasons: List[dict]


app = FastAPI(title="KYT Engine Inference API", version="0.2.0")

model_registry = {}


@app.on_event("startup")
async def load_models():
    import joblib
    from kyt_engine.models.lightgbm_model import LightGBMClassifier
    from kyt_engine.models.autoencoder import AutoencoderDetector
    from kyt_engine.models.kscore import KScoreCalculator
    from kyt_engine.data.scraper import ExternalLabelStore
    from kyt_engine.models.unified_scorer import UnifiedScorer

    model_registry['lgbm'] = joblib.load('models/lightgbm_real.pkl')

    model_registry['kscore'] = KScoreCalculator()

    try:
        store = ExternalLabelStore()
        df = store.load_latest()
        model_registry['external'] = store
    except:
        model_registry['external'] = None

    model_registry['scorer'] = UnifiedScorer(
        lgbm_model=model_registry['lgbm'],
        k_score_calculator=model_registry['kscore'],
        external_label_store=model_registry['external'],
    )
    print("Models loaded successfully")


@app.post("/analyze", response_model=RiskOutput)
async def analyze_transaction(tx: TransactionInput):
    if 'scorer' not in model_registry:
        raise HTTPException(status_code=503, detail="Models not loaded")

    feat_dict = {f'f{i}': 0.0 for i in range(165)}
    if tx.features:
        feat_dict.update(tx.features)

    df = pd.DataFrame([{
        'txId': tx.tx_id,
        'from_address': tx.from_address,
        'to_address': tx.to_address,
        'value': tx.value,
        'gas_price': tx.gas_price,
        'gas_used': tx.gas_used,
        **feat_dict
    }])

    df['in_degree'] = 0
    df['out_degree'] = 0

    result = model_registry['scorer'].score(df, df['txId'])[0]

    return RiskOutput(
        tx_id=result.tx_id,
        risk_score=result.risk_score,
        risk_zone=result.risk_zone,
        triage_level=result.triage_level,
        lgbm_proba=result.lgbm_proba,
        k_score=result.k_score,
        vae_anomaly=result.vae_anomaly,
        external_risk=result.external_risk,
        top_reasons=result.top_features
    )


@app.post("/batch-analyze", response_model=List[RiskOutput])
async def batch_analyze(batch: BatchInput):
    if 'scorer' not in model_registry:
        raise HTTPException(status_code=503, detail="Models not loaded")

    rows = []
    for tx in batch.transactions:
        feat_dict = {f'f{i}': 0.0 for i in range(165)}
        if tx.features:
            feat_dict.update(tx.features)
        rows.append({
            'txId': tx.tx_id,
            'from_address': tx.from_address,
            'to_address': tx.to_address,
            'value': tx.value,
            'gas_price': tx.gas_price,
            'gas_used': tx.gas_used,
            **feat_dict
        })
    df = pd.DataFrame(rows)
    df['in_degree'] = 0
    df['out_degree'] = 0

    results = model_registry['scorer'].score(df, df['txId'])

    return [
        RiskOutput(
            tx_id=r.tx_id,
            risk_score=r.risk_score,
            risk_zone=r.risk_zone,
            triage_level=r.triage_level,
            lgbm_proba=r.lgbm_proba,
            k_score=r.k_score,
            vae_anomaly=r.vae_anomaly,
            external_risk=r.external_risk,
            top_reasons=r.top_features
        )
        for r in results
    ]


@app.get("/health")
async def health():
    return {"status": "ok", "models_loaded": list(model_registry.keys())}


@app.post("/label")
async def receive_label(tx_id: int, label: int, source: str = "analyst"):
    return {"status": "accepted", "tx_id": tx_id, "label": label}