from dataclasses import dataclass, field
from typing import Literal, Optional, List, Sequence
import pandas as pd
import numpy as np
from datetime import datetime


TriageLevel = Literal["AUTO_CLOSE", "PRIORITY", "ESCALATION"]
TriageDecision = Literal["APPROVED", "FLAGGED", "BLOCKED"]

LEVEL_TO_DECISION: dict[TriageLevel, TriageDecision] = {
    "AUTO_CLOSE": "APPROVED",
    "PRIORITY": "FLAGGED",
    "ESCALATION": "BLOCKED",
}


@dataclass
class TriageConfig:
    close_threshold: float = 0.3
    escalate_threshold: float = 0.7
    confidence_high: float = 0.9
    confidence_low: float = 0.7
    entropy_high: float = 0.3
    threat_types: list[str] = field(
        default_factory=lambda: ["flash_loan", "reentry", "unknown"]
    )
    time_of_day_enabled: bool = False
    threat_weight_modifier: dict[str, float] = field(default_factory=dict)


class TriagePolicy:
    def __init__(self, config: TriageConfig | None = None):
        self._config = config or TriageConfig()

    def decide(
        self,
        k_score: float,
        lgbm_proba: float,
        entropy: float = 1.0,
        threat_type: Optional[str] = None,
        timestamp: Optional[datetime] = None,
    ) -> TriageDecision:
        c = self._config

        k_adj = k_score
        if threat_type and threat_type in c.threat_weight_modifier:
            k_adj = k_score * c.threat_weight_modifier[threat_type]

        if c.time_of_day_enabled and timestamp is not None:
            hour = timestamp.hour
            if 2 <= hour < 6:
                k_adj *= 1.5

        if k_adj < c.close_threshold and lgbm_proba > c.confidence_high:
            return "APPROVED"

        if k_adj > c.escalate_threshold or entropy < c.entropy_high:
            return "BLOCKED"

        return "FLAGGED"

    def decide_batch(
        self,
        k_scores: list[float],
        lgbm_probas: list[float],
        entropies: Optional[Sequence[float]] = None,
        threat_types: Optional[Sequence[Optional[str]]] = None,
        timestamps: Optional[Sequence[Optional[datetime]]] = None,
    ) -> List[TriageDecision]:
        n = len(k_scores)
        if entropies is None:
            entropies = [1.0] * n
        if threat_types is None:
            threat_types = [None] * n
        if timestamps is None:
            timestamps = [None] * n

        return [
            self.decide(k, lgbm, ent, th, ts)
            for k, lgbm, ent, th, ts in zip(
                k_scores, lgbm_probas, entropies, threat_types, timestamps
            )
        ]

    def apply(
        self,
        k_scores: pd.Series,
        proba: pd.Series,
        entropies: Optional[pd.Series] = None,
        threat_types: Optional[pd.Series] = None,
        timestamps: Optional[pd.Series] = None,
    ) -> pd.Series:
        c = self._config

        ks = np.asarray(k_scores.values, dtype=float)
        pr = np.asarray(proba.values, dtype=float)
        ent = np.asarray(
            entropies.values, dtype=float
        ) if entropies is not None else np.ones_like(ks)
        
        if threat_types is not None:
            th_list = threat_types.tolist()
        else:
            th_list = [None] * len(ks)
            
        if timestamps is not None:
            ts_list = [datetime.utcfromtimestamp(t) for t in timestamps.values]
        else:
            ts_list = [None] * len(ks)

        approved_mask = (ks < c.close_threshold) & (pr > c.confidence_high)
        blocked_mask = (ks > c.escalate_threshold) | (ent < c.entropy_high)

        decisions = np.full(len(ks), "FLAGGED", dtype=object)
        decisions[approved_mask] = "APPROVED"
        decisions[blocked_mask & ~approved_mask] = "BLOCKED"

        return pd.Series(decisions, index=k_scores.index)