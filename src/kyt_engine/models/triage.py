from __future__ import annotations

import pandas as pd
import numpy as np


class AlertPriority:
    AUTO_CLOSE = "auto_close"
    PRIORITY = "priority"
    ESCALATION = "escalation"


class TriageSystem:
    """Three-level alert triage.

    Level 1 (auto_close): Low K-Score + high model confidence -> archive
    Level 2 (priority): Medium K-Score or medium uncertainty -> analyst review with SHAP
    Level 3 (escalation): High K-Score or max uncertainty -> senior analyst + label request
    """

    def __init__(
        self,
        kscore_close: float = 0.3,
        kscore_escalate: float = 0.7,
        confidence_high: float = 0.9,
        confidence_low: float = 0.7,
    ):
        self._kscore_close = kscore_close
        self._kscore_escalate = kscore_escalate
        self._confidence_high = confidence_high
        self._confidence_low = confidence_low

    def triage(
        self,
        k_scores: pd.Series,
        model_proba: pd.Series,
        model_entropy: pd.Series,
    ) -> pd.DataFrame:
        ks = k_scores.values
        proba = model_proba.values
        ent = model_entropy.values

        is_auto_close = (ks < self._kscore_close) & (proba > self._confidence_high)
        is_escalation = (ks > self._kscore_escalate) | (ent < (1.0 - self._confidence_low))

        priorities = np.full(len(ks), AlertPriority.PRIORITY, dtype=object)
        priorities[is_auto_close] = AlertPriority.AUTO_CLOSE
        priorities[is_escalation & ~is_auto_close] = AlertPriority.ESCALATION

        reasons = np.empty(len(ks), dtype=object)
        reasons[is_auto_close] = np.array([
            f"low_kscore({ks[i]:.3f})+high_confidence({proba[i]:.3f})"
            for i in np.where(is_auto_close)[0]
        ])
        esc_mask = is_escalation & ~is_auto_close
        esc_idx = np.where(esc_mask)[0]
        esc_reasons = []
        for i in esc_idx:
            parts = []
            if ks[i] > self._kscore_escalate:
                parts.append(f"high_kscore({ks[i]:.3f})")
            if ent[i] < (1.0 - self._confidence_low):
                parts.append(f"low_entropy({ent[i]:.3f})")
            esc_reasons.append("+".join(parts))
        reasons[esc_mask] = np.array(esc_reasons)

        pri_mask = ~(is_auto_close | is_escalation)
        pri_idx = np.where(pri_mask)[0]
        reasons[pri_mask] = np.array([
            f"kscore={ks[i]:.3f},proba={proba[i]:.3f},entropy={ent[i]:.3f}"
            for i in pri_idx
        ])

        return pd.DataFrame(
            {
                "kscore": ks,
                "proba": proba,
                "entropy": ent,
                "priority": priorities,
                "reason": reasons,
            }
        )

    def statistics(self, triage_result: pd.DataFrame) -> dict:
        counts = triage_result["priority"].value_counts()
        total = len(triage_result)
        stats: dict = {"total": total, "counts": counts.to_dict()}

        for level in [AlertPriority.AUTO_CLOSE, AlertPriority.PRIORITY, AlertPriority.ESCALATION]:
            mask = triage_result["priority"] == level
            subset = triage_result[mask]
            key = f"{level}_pct"
            stats[key] = float(len(subset) / total * 100) if total > 0 else 0.0
            if len(subset) > 0:
                stats[f"{level}_avg_kscore"] = float(subset["kscore"].mean())
                stats[f"{level}_avg_proba"] = float(subset["proba"].mean())
                stats[f"{level}_avg_entropy"] = float(subset["entropy"].mean())

        return stats
