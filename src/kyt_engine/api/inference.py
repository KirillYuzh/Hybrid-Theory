from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import pandas as pd
import redis
from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field
from typing_extensions import Literal

try:
    import prometheus_client
    from prometheus_client import Counter, Gauge, Histogram, generate_latest
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

log = logging.getLogger(__name__)


# =============================================================================
# Prometheus Metrics
# =============================================================================
if PROMETHEUS_AVAILABLE:
    TRANSACTIONS_TOTAL = Counter(
        "kyt_transactions_total",
        "Total transactions scored",
        ["zone"],
    )
    ALERTS_TOTAL = Counter(
        "kyt_alerts_total",
        "Total alerts by triage level",
        ["triage"],
    )
    INFERENCE_DURATION = Histogram(
        "kyt_model_inference_duration_seconds",
        "Inference duration in seconds",
        ["model"],
        buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
    )
    RISK_SCORE_HIST = Histogram(
        "kyt_risk_score_distribution",
        "Risk score distribution",
        buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    )
    ACTIVE_TXS_GAUGE = Gauge("kyt_active_txs_gauge", "Active transactions in processing")
    MODEL_VERSION_GAUGE = Gauge(
        "kyt_model_version_gauge", "Loaded model version", ["model"]
    )
    CACHE_HITS = Counter("kyt_cache_hits_total", "Redis cache hits")
    CACHE_MISSES = Counter("kyt_cache_misses_total", "Redis cache misses")
else:
    TRANSACTIONS_TOTAL = ALERTS_TOTAL = INFERENCE_DURATION = RISK_SCORE_HIST = None
    ACTIVE_TXS_GAUGE = MODEL_VERSION_GAUGE = CACHE_HITS = CACHE_MISSES = None


# =============================================================================
# Pydantic Request / Response Models
# =============================================================================
class ScoringRequest(BaseModel):
    tx_id: str
    from_address: str
    to_address: str
    value: float
    gas_price: float
    gas_used: int
    timestamp: int
    features: Optional[dict[str, float]] = None


class ReasonItem(BaseModel):
    feature: str
    value: float
    shap_value: float


class ScoringResponse(BaseModel):
    tx_id: str
    risk_score: float
    risk_zone: Literal["GREEN", "YELLOW", "RED"]
    triage_level: Literal["AUTO_CLOSE", "PRIORITY", "ESCALATION"]
    lgbm_proba: float
    k_score: float
    vae_anomaly: float
    external_risk: float
    top_reasons: list[ReasonItem]


class BatchScoringRequest(BaseModel):
    transactions: list[ScoringRequest]


class FeedbackRequest(BaseModel):
    tx_id: str
    label: int
    source: str = "analyst"


# =============================================================================
# ScoringResult dataclass
# =============================================================================
@dataclass
class ScoringResult:
    tx_id: str
    risk_score: float
    risk_zone: Literal["GREEN", "YELLOW", "RED"]
    triage_level: Literal["AUTO_CLOSE", "PRIORITY", "ESCALATION"]
    lgbm_proba: float
    k_score: float
    vae_anomaly: float
    external_risk: float
    top_reasons: list[ReasonItem]
    timestamp: pd.Timestamp = field(default_factory=pd.Timestamp.now)


# =============================================================================
# FeatureStore — Redis Cache
# =============================================================================
class FeatureStore:
    """Cache feature vectors from Iceberg in Redis."""

    def __init__(self, redis_url: str = "redis://redis:6379", ttl: int = 3600):
        self._ttl = ttl
        self._redis: Optional[redis.Redis] = None
        self._redis_url = redis_url
        self._connected = False
        self._connect()

    def _connect(self):
        try:
            self._redis = redis.from_url(
                self._redis_url,
                decode_responses=False,
                socket_connect_timeout=2,
            )
            self._redis.ping()
            self._connected = True
            log.info("Connected to Redis at %s", self._redis_url)
        except Exception as exc:
            log.warning("Redis unavailable: %s. Feature caching disabled.", exc)
            self._redis = None
            self._connected = False

    @property
    def is_connected(self) -> bool:
        return self._connected and self._redis is not None

    def _key(self, tx_id: str) -> bytes:
        return f"kyt:features:{tx_id}".encode()

    def get(self, tx_id: str) -> Optional[np.ndarray]:
        """Get cached feature vector."""
        if not self.is_connected:
            return None
        try:
            data = self._redis.get(self._key(tx_id))
            if data is None:
                if CACHE_MISSES:
                    CACHE_MISSES.inc()
                return None
            if CACHE_HITS:
                CACHE_HITS.inc()
            vec = np.frombuffer(data, dtype=np.float64)
            return vec
        except Exception as exc:
            log.warning("Cache get failed for %s: %s", tx_id, exc)
            return None

    def set(self, tx_id: str, features: np.ndarray):
        """Cache feature vector."""
        if not self.is_connected:
            return
        try:
            data = features.astype(np.float64).tobytes()
            self._redis.setex(self._key(tx_id), self._ttl, data)
        except Exception as exc:
            log.warning("Cache set failed for %s: %s", tx_id, exc)

    def get_batch(self, tx_ids: list[str]) -> dict[str, Optional[np.ndarray]]:
        """Batch get from cache."""
        if not self.is_connected or not tx_ids:
            return {tid: None for tid in tx_ids}
        try:
            keys = [self._key(tid) for tid in tx_ids]
            results = self._redis.mget(keys)
            out = {}
            for tid, data in zip(tx_ids, results):
                if data is None:
                    if CACHE_MISSES:
                        CACHE_MISSES.inc()
                    out[tid] = None
                else:
                    if CACHE_HITS:
                        CACHE_HITS.inc()
                    out[tid] = np.frombuffer(data, dtype=np.float64)
            return out
        except Exception as exc:
            log.warning("Batch cache get failed: %s", exc)
            return {tid: None for tid in tx_ids}


# =============================================================================
# ModelLoader — Load models from Iceberg/MLflow
# =============================================================================
class ModelLoader:
    """Load production models from Iceberg registry."""

    def __init__(
        self,
        iceberg_catalog: Optional[Any] = None,
        warehouse_path: str = "s3://kyt-lake/warehouse",
    ):
        self._catalog = iceberg_catalog
        self._warehouse = warehouse_path
        self._cache: dict[str, Any] = {}

    def load_latest(self, model_type: str) -> Optional[Any]:
        """Load latest production model by type.

        Tries:
        1. In-memory cache
        2. MLflow model registry
        3. Iceberg models table
        4. Local filesystem fallback
        """
        if model_type in self._cache:
            return self._cache[model_type]

        model = self._load_from_mlflow(model_type)
        if model is not None:
            self._cache[model_type] = model
            return model

        model = self._load_from_iceberg(model_type)
        if model is not None:
            self._cache[model_type] = model
            return model

        model = self._load_from_filesystem(model_type)
        if model is not None:
            self._cache[model_type] = model
            return model

        log.warning("Could not load model type '%s'", model_type)
        return None

    def _load_from_mlflow(self, model_type: str) -> Optional[Any]:
        try:
            import mlflow
            model = mlflow.pyfunc.load_model(f"models:/{model_type}/production")
            log.info("Loaded %s from MLflow production", model_type)
            return model
        except Exception as exc:
            log.debug("MLflow load failed for %s: %s", model_type, exc)
            return None

    def _load_from_iceberg(self, model_type: str) -> Optional[Any]:
        if self._catalog is None:
            return None
        try:
            import joblib
            from io import BytesIO

            table = self._catalog.load_table("kyt.models")
            df = table.scan().to_pandas()
            df = df[df["model_type"] == model_type].sort_values("trained_at", ascending=False)
            if df.empty:
                return None
            artifact_path = df.iloc[0]["artifact_path"]
            blob = self._catalog._load_artifact(artifact_path)
            model = joblib.loads(blob)
            log.info("Loaded %s from Iceberg", model_type)
            return model
        except Exception as exc:
            log.debug("Iceberg load failed for %s: %s", model_type, exc)
            return None

    def _load_from_filesystem(self, model_type: str) -> Optional[Any]:
        import os
        candidates = [
            f"models/{model_type}_production.pkl",
            f"models/{model_type}.pkl",
            f"models/lightgbm_real.pkl",
            f"models/autoencoder.pkl",
        ]
        for path in candidates:
            if os.path.exists(path):
                try:
                    import joblib
                    model = joblib.load(path)
                    log.info("Loaded %s from %s", model_type, path)
                    return model
                except Exception as exc:
                    log.warning("Failed to load %s: %s", path, exc)
        return None


# =============================================================================
# UnifiedScorer — Combine all signals
# =============================================================================
class UnifiedScorer:
    """Production version combining LightGBM + K-Score + VAE + External."""

    LGBM_FEAT_COUNT = 165
    GRAPH_FEAT_COUNT = 2

    def __init__(
        self,
        lgbm_model: Any,
        kscore_calculator: Any,
        vae_model: Optional[Any] = None,
        external_store: Optional[Any] = None,
        weights: Optional[dict[str, float]] = None,
    ):
        self._lgbm = lgbm_model
        self._kscore = kscore_calculator
        self._vae = vae_model
        self._external = external_store
        self._weights = weights or {
            "lgbm": 0.50,
            "kscore": 0.20,
            "vae": 0.15,
            "external": 0.15,
        }
        self._feat_cols: list[str] = []

    def _build_feature_vector(self, req: ScoringRequest) -> np.ndarray:
        vec = np.zeros(self.LGBM_FEAT_COUNT + self.GRAPH_FEAT_COUNT, dtype=np.float64)
        if req.features:
            for k, v in req.features.items():
                if k.startswith("f"):
                    idx = int(k[1:])
                    if 0 <= idx < self.LGBM_FEAT_COUNT:
                        vec[idx] = v
                elif k in ("in_degree", "out_degree"):
                    if k == "in_degree":
                        vec[self.LGBM_FEAT_COUNT] = v
                    else:
                        vec[self.LGBM_FEAT_COUNT + 1] = v
        return vec

    def _shap_top_reasons(
        self, features: np.ndarray, shap_values: np.ndarray, top_n: int = 3
    ) -> list[ReasonItem]:
        if shap_values.ndim == 2:
            shap_values = shap_values[:, 1]
        abs_shap = np.abs(shap_values)
        top_idx = np.argsort(abs_shap)[::-1][:top_n]
        reasons = []
        for idx in top_idx:
            reasons.append(ReasonItem(
                feature=f"f{idx}" if idx < self.LGBM_FEAT_COUNT else "graph_feat",
                value=float(features[idx]),
                shap_value=float(shap_values[idx]),
            ))
        return reasons

    def score(
        self,
        features: np.ndarray,
        tx_id: str,
        from_address: str,
    ) -> ScoringResult:
        """Score a single transaction."""
        t0 = time.perf_counter()

        lgbm_proba = self._lgbm.predict_proba(features.reshape(1, -1))[:, 1][0]

        if hasattr(self._kscore, "score"):
            k_df = pd.DataFrame([features[:self.LGBM_FEAT_COUNT]], columns=[f"f{i}" for i in range(self.LGBM_FEAT_COUNT)])
            k_score_val = float(self._kscore.score(k_df).iloc[0])
        else:
            k_score_val = 0.0

        if self._vae is not None:
            vae_anomaly = float(self._vae.predict_proba(features.reshape(1, -1))[:, 1][0])
        else:
            vae_anomaly = 0.0

        if self._external is not None and hasattr(self._external, "is_illicit"):
            external_risk = 1.0 if self._external.is_illicit(from_address) else 0.0
        else:
            external_risk = 0.0

        risk_score = (
            self._weights["lgbm"] * lgbm_proba
            + self._weights["kscore"] * k_score_val
            + self._weights["vae"] * vae_anomaly
            + self._weights["external"] * external_risk
        )
        risk_score = float(np.clip(risk_score, 0.0, 1.0))

        if risk_score < 0.3:
            risk_zone: Literal["GREEN", "YELLOW", "RED"] = "GREEN"
        elif risk_score < 0.7:
            risk_zone = "YELLOW"
        else:
            risk_zone = "RED"

        triage_level = self._triage_level(risk_score, k_score_val)

        try:
            import shap
            explainer = shap.TreeExplainer(self._lgbm)
            shap_vals = explainer.shap_values(features.reshape(1, -1))
            if isinstance(shap_vals, list):
                shap_vals = shap_vals[1]
            top_reasons = self._shap_top_reasons(features, shap_vals)
        except Exception:
            top_reasons = []

        if INFERENCE_DURATION:
            INFERENCE_DURATION.labels(model="ensemble").observe(time.perf_counter() - t0)
        if TRANSACTIONS_TOTAL:
            TRANSACTIONS_TOTAL.labels(zone=risk_zone).inc()
        if ALERTS_TOTAL:
            ALERTS_TOTAL.labels(triage=triage_level).inc()
        if RISK_SCORE_HIST:
            RISK_SCORE_HIST.observe(risk_score)

        return ScoringResult(
            tx_id=tx_id,
            risk_score=risk_score,
            risk_zone=risk_zone,
            triage_level=triage_level,
            lgbm_proba=float(lgbm_proba),
            k_score=k_score_val,
            vae_anomaly=vae_anomaly,
            external_risk=external_risk,
            top_reasons=top_reasons,
        )

    def score_batch(
        self,
        features_batch: np.ndarray,
        tx_ids: list[str],
        addresses: list[str],
    ) -> list[ScoringResult]:
        """Batch scoring for throughput."""
        n = len(tx_ids)
        t0 = time.perf_counter()

        X = features_batch
        lgbm_proba = self._lgbm.predict_proba(X)[:, 1]

        if hasattr(self._kscore, "score"):
            k_df = pd.DataFrame(
                X[:, :self.LGBM_FEAT_COUNT],
                columns=[f"f{i}" for i in range(self.LGBM_FEAT_COUNT)],
            )
            k_scores_arr = self._kscore.score(k_df).values
        else:
            k_scores_arr = np.zeros(n)

        if self._vae is not None:
            vae_anomaly_arr = self._vae.predict_proba(X)[:, 1]
        else:
            vae_anomaly_arr = np.zeros(n)

        external_risk_arr = np.zeros(n)
        if self._external is not None and hasattr(self._external, "is_illicit"):
            for i, addr in enumerate(addresses):
                if self._external.is_illicit(addr):
                    external_risk_arr[i] = 1.0

        risk_scores = (
            self._weights["lgbm"] * lgbm_proba
            + self._weights["kscore"] * k_scores_arr
            + self._weights["vae"] * vae_anomaly_arr
            + self._weights["external"] * external_risk_arr
        )
        risk_scores = np.clip(risk_scores, 0.0, 1.0)

        risk_zones = np.where(
            risk_scores < 0.3, "GREEN", np.where(risk_scores < 0.7, "YELLOW", "RED")
        )
        triage_levels = [
            self._triage_level(float(rs), float(ks))
            for rs, ks in zip(risk_scores, k_scores_arr)
        ]

        results = []
        for i in range(n):
            results.append(ScoringResult(
                tx_id=tx_ids[i],
                risk_score=float(risk_scores[i]),
                risk_zone=risk_zones[i],
                triage_level=triage_levels[i],
                lgbm_proba=float(lgbm_proba[i]),
                k_score=float(k_scores_arr[i]),
                vae_anomaly=float(vae_anomaly_arr[i]),
                external_risk=float(external_risk_arr[i]),
                top_reasons=[],
            ))

        if INFERENCE_DURATION:
            INFERENCE_DURATION.labels(model="ensemble_batch").observe(
                time.perf_counter() - t0
            )

        return results

    @staticmethod
    def _triage_level(risk_score: float, k_score: float) -> Literal["AUTO_CLOSE", "PRIORITY", "ESCALATION"]:
        if risk_score < 0.2 and k_score < 0.3:
            return "AUTO_CLOSE"
        elif risk_score > 0.7 or k_score > 0.7:
            return "ESCALATION"
        else:
            return "PRIORITY"


# =============================================================================
# FastAPI Application
# =============================================================================
app = FastAPI(
    title="KYT Engine API",
    version="1.0.0",
    description="Know Your Transaction — Real-time AML risk scoring API",
)

_feature_store: Optional[FeatureStore] = None
_model_loader: Optional[ModelLoader] = None
_scorer: Optional[UnifiedScorer] = None


@app.on_event("startup")
async def startup():
    global _feature_store, _model_loader, _scorer

    log.info("Starting KYT Engine API v1.0.0...")

    _feature_store = FeatureStore(redis_url="redis://redis:6379", ttl=3600)
    _model_loader = ModelLoader(warehouse_path="s3://kyt-lake/warehouse")

    lgbm_model = _model_loader.load_latest("lightgbm")
    kscore_model = _model_loader.load_latest("kscore")
    vae_model = _model_loader.load_latest("vae")
    external_store = _model_loader.load_latest("external_labels")

    if lgbm_model is None:
        log.warning("LightGBM model not loaded — using fallback stub")
        from lightgbm import LGBMClassifier
        lgbm_model = LGBMClassifier(n_estimators=10, verbose=-1)
        lgbm_model.fit(np.random.rand(10, 167), np.random.randint(0, 2, 10))

    if kscore_model is None:
        from kyt_engine.models.kscore import KScoreCalculator
        kscore_model = KScoreCalculator()

    if MODEL_VERSION_GAUGE and lgbm_model:
        MODEL_VERSION_GAUGE.labels(model="lightgbm").set(1)

    _scorer = UnifiedScorer(
        lgbm_model=lgbm_model,
        kscore_calculator=kscore_model,
        vae_model=vae_model,
        external_store=external_store,
    )

    log.info(
        "KYT Engine ready. Redis: %s, LightGBM: %s, KScore: %s, VAE: %s",
        _feature_store.is_connected,
        lgbm_model is not None,
        kscore_model is not None,
        vae_model is not None,
    )


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "version": "1.0.0",
        "models_loaded": _scorer is not None,
        "cache_connected": _feature_store.is_connected if _feature_store else False,
    }


@app.get("/ready")
async def ready():
    if _scorer is None:
        raise HTTPException(status_code=503, detail="Scorer not initialized")
    return {"ready": True}


@app.post("/predict", response_model=ScoringResponse)
async def predict(request: ScoringRequest):
    """Score a single transaction."""
    if _scorer is None:
        raise HTTPException(status_code=503, detail="Scorer not initialized")

    cached = None
    if _feature_store:
        cached = _feature_store.get(request.tx_id)

    if cached is not None:
        features = cached
    else:
        features = _scorer._build_feature_vector(request)
        if _feature_store and len(features) > 0:
            _feature_store.set(request.tx_id, features)

    result = _scorer.score(features, request.tx_id, request.from_address)

    return ScoringResponse(
        tx_id=result.tx_id,
        risk_score=round(result.risk_score, 6),
        risk_zone=result.risk_zone,
        triage_level=result.triage_level,
        lgbm_proba=round(result.lgbm_proba, 6),
        k_score=round(result.k_score, 6),
        vae_anomaly=round(result.vae_anomaly, 6),
        external_risk=round(result.external_risk, 6),
        top_reasons=[
            ReasonItem(
                feature=r.feature,
                value=round(r.value, 6),
                shap_value=round(r.shap_value, 6),
            )
            for r in result.top_reasons
        ],
    )


@app.post("/batch-predict", response_model=list[ScoringResponse])
async def batch_predict(request: BatchScoringRequest):
    """Score multiple transactions."""
    if _scorer is None:
        raise HTTPException(status_code=503, detail="Scorer not initialized")

    if len(request.transactions) > 1000:
        raise HTTPException(status_code=413, detail="Batch size limited to 1000")

    n = len(request.transactions)
    tx_ids = [tx.tx_id for tx in request.transactions]
    addresses = [tx.from_address for tx in request.transactions]

    if ACTIVE_TXS_GAUGE:
        ACTIVE_TXS_GAUGE.set(n)

    cached_features = {}
    missing = []
    if _feature_store:
        cached_features = _feature_store.get_batch(tx_ids)
        for tid in tx_ids:
            if cached_features.get(tid) is None:
                missing.append(tid)

    feature_matrix = np.zeros((n, UnifiedScorer.LGBM_FEAT_COUNT + UnifiedScorer.GRAPH_FEAT_COUNT), dtype=np.float64)
    for i, tx in enumerate(request.transactions):
        if cached_features.get(tx.tx_id) is not None:
            feature_matrix[i] = cached_features[tx.tx_id]
        else:
            vec = _scorer._build_feature_vector(tx)
            feature_matrix[i] = vec
            if _feature_store is not None:
                _feature_store.set(tx.tx_id, vec)

    results = _scorer.score_batch(feature_matrix, tx_ids, addresses)

    return [
        ScoringResponse(
            tx_id=r.tx_id,
            risk_score=round(r.risk_score, 6),
            risk_zone=r.risk_zone,
            triage_level=r.triage_level,
            lgbm_proba=round(r.lgbm_proba, 6),
            k_score=round(r.k_score, 6),
            vae_anomaly=round(r.vae_anomaly, 6),
            external_risk=round(r.external_risk, 6),
            top_reasons=[],
        )
        for r in results
    ]


@app.post("/feedback")
async def feedback(request: FeedbackRequest):
    """Receive manual label for active learning."""
    log.info(
        "Feedback received: tx_id=%s label=%d source=%s",
        request.tx_id,
        request.label,
        request.source,
    )
    return {
        "status": "accepted",
        "tx_id": request.tx_id,
        "label": request.label,
        "source": request.source,
    }


@app.get("/metrics", response_class=PlainTextResponse)
async def metrics():
    """Prometheus metrics."""
    if not PROMETHEUS_AVAILABLE:
        raise HTTPException(
            status_code=501,
            detail="prometheus_client not installed",
        )
    return PlainTextResponse(
        generate_latest(),
        media_type="text/plain; charset=utf-8",
    )


@app.get("/features/{tx_id}")
async def get_features(tx_id: str):
    """Get cached feature vector for a transaction."""
    if _feature_store is None:
        raise HTTPException(status_code=503, detail="FeatureStore not initialized")
    vec = _feature_store.get(tx_id)
    if vec is None:
        raise HTTPException(status_code=404, detail=f"Features not found for {tx_id}")
    return {"tx_id": tx_id, "features": vec.tolist()}
