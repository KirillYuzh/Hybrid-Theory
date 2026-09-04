from __future__ import annotations

import pandas as pd


class TriagePolicy:
    def __init__(
        self,
        close_threshold: float = 0.3,
        escalate_threshold: float = 0.7,
        confidence_high: float = 0.9,
        confidence_low: float = 0.7,
    ) -> None:
        self._close = close_threshold
        self._escalate = escalate_threshold
        self._conf_high = confidence_high
        self._conf_low = confidence_low

    def apply(self, k_scores: pd.Series, proba: pd.Series) -> pd.Series:
        ks = k_scores.values
        pr = proba.values

        auto_close = (ks < self._close) & (pr > self._conf_high)
        escalation = (ks > self._escalate) | (pr < self._conf_low)

        levels = pd.Series(["priority"] * len(ks), index=k_scores.index)
        levels[auto_close] = "auto_close"
        levels[escalation & ~auto_close] = "escalation"
        return levels