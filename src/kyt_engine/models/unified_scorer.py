import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional, List


@dataclass
class ScoringConfig:
    weights: tuple[float, float, float, float] = (0.4, 0.3, 0.2, 0.1)
    risk_zone_thresholds: tuple[float, float] = (0.3, 0.7)
    risk_zone_labels: tuple[str, str, str] = ("low", "medium", "high")
    triage_high_k: float = 0.5
    triage_high_lgbm: float = 0.5
    triage_medium_k: float = 0.3
    triage_medium_lgbm: float = 0.3


@dataclass
class ScoringResult:
    tx_id: int
    risk_score: float
    risk_zone: str
    triage_level: str
    lgbm_proba: float
    k_score: float
    vae_anomaly: float
    external_risk: float
    timestamp: pd.Timestamp


class UnifiedScorer:

    def __init__(
        self,
        lgbm_model,
        k_score_calculator,
        vae_model=None,
        external_label_store=None,
        config: Optional[ScoringConfig] = None,
    ):
        self._lgbm = lgbm_model
        self._kscore = k_score_calculator
        self._vae = vae_model
        self._external = external_label_store
        self._config = config or ScoringConfig()

        w = np.array(self._config.weights, dtype=float)
        self._weights = w / w.sum()

    def score(
        self, features: pd.DataFrame, tx_ids: pd.Series
    ) -> List[ScoringResult]:
        lgbm_cols = getattr(self._lgbm, "feature_names", [])
        lgbm_feat_cols = [c for c in features.columns if c in lgbm_cols]
        if not lgbm_feat_cols:
            raise ValueError(f"No valid feature columns found. Available: {list(features.columns)[:10]}...")
        X_lgbm = features[lgbm_feat_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)

        lgbm_proba = self._lgbm.predict_proba(X_lgbm)[:, 1]
        k_scores = self._kscore.score(features)

        if self._vae:
            vae_anomaly = self._vae.predict_proba(X_lgbm)[:, 1]
        else:
            vae_anomaly = np.zeros(len(features))

        if self._external:
            addrs = features["from_address"] if "from_address" in features.columns else tx_ids
            illicit_set = set(self._external.get_illicit_addresses())
            external_risk = np.array([1.0 if a in illicit_set else 0.0 for a in addrs])
        else:
            external_risk = np.zeros(len(features))

        risk_score = (
            self._weights[0] * lgbm_proba
            + self._weights[1] * k_scores
            + self._weights[2] * vae_anomaly
            + self._weights[3] * external_risk
        )

        t1, t2 = self._config.risk_zone_thresholds
        risk_zone_cat = pd.cut(
            risk_score, bins=[0, t1, t2, 1.0], labels=self._config.risk_zone_labels
        )
        risk_zone = np.array([str(v) for v in risk_zone_cat])

        triage_result = self._triage(k_scores, lgbm_proba)

        results = [
            ScoringResult(
                tx_id=int(tx_ids.iloc[i]),
                risk_score=float(risk_score[i]),
                risk_zone=risk_zone[i],
                triage_level=str(triage_result.iloc[i]),
                lgbm_proba=float(lgbm_proba[i]),
                k_score=float(k_scores.iloc[i]),
                vae_anomaly=float(vae_anomaly[i]),
                external_risk=float(external_risk[i]),
                timestamp=pd.Timestamp.now(),
            )
            for i in range(len(features))
        ]
        return results

    def _triage(
        self, k_scores: pd.Series, lgbm_proba: np.ndarray
    ) -> pd.Series:
        c = self._config
        priorities = []
        for k, lgb in zip(k_scores, lgbm_proba):
            if k > c.triage_high_k and lgb > c.triage_high_lgbm:
                priorities.append("escalate")
            elif k > c.triage_medium_k or lgb > c.triage_medium_lgbm:
                priorities.append("priority")
            else:
                priorities.append("auto_close")
        return pd.Series(priorities)