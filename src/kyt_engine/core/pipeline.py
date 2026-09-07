import numpy as np
import pandas as pd

from kyt_engine.core.audit import AuditLog
from kyt_engine.core.contracts import FeatureVector, ScoreResult, TxRecord
from kyt_engine.core.features import extract_base_features, extract_behavioral_features
from kyt_engine.core.scorers import Scorer
from kyt_engine.core.sinks import Sink
from kyt_engine.core.triage import TriagePolicy


class Pipeline:
    def __init__(
        self,
        features,
        scorers: list[Scorer],
        triage: TriagePolicy,
        sinks: list[Sink],
        weights: dict[str, float],
        audit_log: object | None = None,
    ) -> None:
        self._features = features
        self._scorers = scorers
        self._triage = triage
        self._sinks = sinks
        self._weights = weights
        self._audit = audit_log

    def _feature_vector(self, tx: TxRecord, history: pd.DataFrame | None = None) -> FeatureVector:
        import pandas as pd

        # Convert TxRecord to DataFrame for feature extraction
        tx_dict = {
            "address": tx.address,
            "from_address": tx.from_address,
            "to_address": tx.to_address,
            "value": tx.value,
            "gas_price": tx.gas_price,
            "gas_used": tx.gas_used,
            "timestamp": tx.timestamp,
            "block_number": tx.block_number,
        }
        df = pd.DataFrame([tx_dict])
        gas_median = getattr(self._features, "_global_gas_median", 0.0) if self._features else 0.0
        base = extract_base_features(df)
        behavioral = extract_behavioral_features(df, gas_median)
        combined = pd.concat([base, behavioral], axis=1)
        values = combined.to_dict(orient="records")[0] if len(combined) > 0 else {}
        return FeatureVector(values=values)

    def _reasons(self, features: FeatureVector, probas: dict[str, float]) -> list[dict[str, float | str]]:
        vals = features.values
        top_feats = sorted(vals.items(), key=lambda kv: abs(kv[1]), reverse=True)[:3]
        return [
            {"feature": str(feat), "value": float(val), "contribution": float(val)}
            for feat, val in top_feats
        ]

    def score_tx(self, tx: TxRecord, history: pd.DataFrame | None = None) -> ScoreResult:
        features = self._feature_vector(tx, history)
        probas = {s.name: s.predict_proba(features, tx) for s in self._scorers}

        total_weight = sum(self._weights.values())
        risk = sum(self._weights.get(name, 0.0) / total_weight * p for name, p in probas.items())
        risk = float(np.clip(risk, 0.0, 1.0))

        if risk < 0.3:
            zone = "GREEN"
        elif risk < 0.7:
            zone = "YELLOW"
        else:
            zone = "RED"

        ks = probas.get("kscore", 0.0)
        level = self._triage.decide(ks, probas.get("lightgbm", 0.0))

        result = ScoreResult(
            tx_id=tx.tx_id,
            risk_score=risk,
            risk_zone=zone,
            triage_level=level,
            lgbm_proba=probas.get("lightgbm", 0.0),
            k_score=ks,
            vae_anomaly=probas.get("vae", 0.0),
            external_risk=probas.get("external", 0.0),
            reasons=self._reasons(features, probas),
        )

        if self._audit is not None and hasattr(self._audit, "append"):
            self._audit.append(tx, result, requested_by="api", model_version="1.0")

        for sink in self._sinks:
            sink.write(result)

        return result

    def score_batch(self, txs: list[TxRecord]) -> list[ScoreResult]:
        return [self.score_tx(tx) for tx in txs]