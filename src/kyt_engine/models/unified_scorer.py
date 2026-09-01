from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional


@dataclass
class ScoringResult:
    """
    Complete risk assessment for a transaction
    """
    tx_id: int
    risk_score: float
    risk_zone: str
    triage_level: str

    lgbm_proba: float
    k_score: float
    vae_anomaly: float
    external_risk: float

    top_features: list[dict]

    timestamp: pd.Timestamp


class UnifiedScorer:
    """
    Combines all risk signals into a unified score
    """

    def __init__(
        self,
        lgbm_model,
        k_score_calculator,
        vae_model=None,
        external_label_store=None,
        w_lgbm: float = 0.5,
        w_kscore: float = 0.2,
        w_vae: float = 0.15,
        w_external: float = 0.15,
    ):
        self._lgbm = lgbm_model
        self._kscore = k_score_calculator
        self._vae = vae_model
        self._external = external_label_store
        self._weights = np.array([w_lgbm, w_kscore, w_vae, w_external])
        self._weights = self._weights / self._weights.sum()

    def score(self, features: pd.DataFrame, tx_ids: pd.Series) -> list[ScoringResult]:
        """
        Score a batch of transactions
        """
        results = []

        # Use only the features LightGBM was trained on (165 f's + in_degree + out_degree = 167)
        lgbm_feat_cols = [c for c in features.columns if c.startswith('f') and c[1:].isdigit() and int(c[1:]) < 165]
        lgbm_feat_cols += ['in_degree', 'out_degree']
        lgbm_feat_cols = [c for c in lgbm_feat_cols if c in features.columns]
        X_lgbm = features[lgbm_feat_cols].replace([np.inf, -np.inf], np.nan).fillna(0)

        lgbm_proba = self._lgbm.predict_proba(X_lgbm)[:, 1]

        k_scores = self._kscore.score(features)

        if self._vae:
            vae_anomaly = self._vae.predict_proba(X_lgbm)[:, 1]
        else:
            vae_anomaly = np.zeros(len(features))

        if self._external:
            addrs = features.get('from_address', tx_ids)
            illicit_set = set(self._external.get_illicit_addresses())
            external_risk = np.array([1.0 if a in illicit_set else 0.0 for a in addrs])
        else:
            external_risk = np.zeros(len(features))

        risk_score = (
            self._weights[0] * lgbm_proba +
            self._weights[1] * k_scores +
            self._weights[2] * vae_anomaly +
            self._weights[3] * external_risk
        )

        risk_zone = pd.cut(risk_score, bins=[0, 0.3, 0.7, 1.0], labels=['GREEN', 'YELLOW', 'RED'])
        risk_zone_arr = risk_zone.astype(str).values

        from kyt_engine.models.triage import TriageSystem
        tsys = TriageSystem()
        triage_result = tsys.triage(k_scores, pd.Series(lgbm_proba), pd.Series(np.zeros(len(features))))

        for i in range(len(features)):
            results.append(ScoringResult(
                tx_id=int(tx_ids.iloc[i]),
                risk_score=float(risk_score[i]),
                risk_zone=risk_zone_arr[i],
                triage_level=triage_result['priority'].iloc[i],
                lgbm_proba=float(lgbm_proba[i]),
                k_score=float(k_scores.iloc[i]),
                vae_anomaly=float(vae_anomaly[i]),
                external_risk=float(external_risk[i]),
                top_features=[],
                timestamp=pd.Timestamp.now()
            ))
        return results