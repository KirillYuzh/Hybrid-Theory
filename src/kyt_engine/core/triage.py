from dataclasses import dataclass
from typing import Literal, Optional
import pandas as pd
import numpy as np


TriageLevel = Literal["AUTO_CLOSE", "PRIORITY", "ESCALATION"]


@dataclass
class TriageConfig:
    close_threshold: float = 0.3
    escalate_threshold: float = 0.7
    confidence_high: float = 0.9
    confidence_low: float = 0.7
    entropy_high: float = 0.3


class TriagePolicy:
    def __init__(self, config: TriageConfig | None = None):
        self._config = config or TriageConfig()

    def decide(
        self,
        k_score: float,
        lgbm_proba: float,
        entropy: float = 1.0,
    ) -> TriageLevel:
        c = self._config
        
        # AUTO_CLOSE: low anomaly AND high confidence in licit
        if k_score < c.close_threshold and lgbm_proba > c.confidence_high:
            return "AUTO_CLOSE"
        
        # ESCALATION: high anomaly OR very low entropy (model very confident)
        if k_score > c.escalate_threshold or entropy < c.entropy_high:
            return "ESCALATION"
        
        # Default: PRIORITY
        return "PRIORITY"

    def decide_batch(
        self,
        k_scores: list[float],
        lgbm_probas: list[float],
        entropies: list[float] | None = None,
    ) -> list[TriageLevel]:
        if entropies is None:
            entropies = [1.0] * len(k_scores)
        
        return [
            self.decide(k, lgbm, ent)
            for k, lgbm, ent in zip(k_scores, lgbm_probas, entropies)
        ]

    def apply(
        self,
        k_scores: pd.Series,
        proba: pd.Series,
        entropies: pd.Series | None = None,
    ) -> pd.Series:
        c = self._config
        
        ks = k_scores.values
        pr = proba.values
        ent = entropies.values if entropies is not None else np.ones_like(ks)
        
        auto_close = (ks < c.close_threshold) & (pr > c.confidence_high)
        escalation = (ks > c.escalate_threshold) | (ent < c.entropy_high)
        
        levels = np.full(len(ks), "PRIORITY", dtype=object)
        levels[auto_close] = "AUTO_CLOSE"
        levels[escalation & ~auto_close] = "ESCALATION"
        
        return pd.Series(levels, index=k_scores.index)