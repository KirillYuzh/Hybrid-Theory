import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Tuple, Literal, Optional


Priority = Literal["HIGH", "MEDIUM", "LOW"]


@dataclass
class ALConfig:
    entropy_high: float = 0.7
    kscore_high: float = 0.5
    sample_size: int = 500


class UncertaintySampler:
    def __init__(self, config: ALConfig | None = None):
        self._config = config or ALConfig()

    def _entropy(self, probas: np.ndarray) -> np.ndarray:
        p = np.clip(probas, 1e-12, 1 - 1e-12)
        return -p * np.log2(p) - (1 - p) * np.log2(1 - p)

    def select(
        self,
        features: pd.DataFrame,
        lgbm_probas: np.ndarray,
        k_scores: np.ndarray,
        sample_size: int | None = None,
    ) -> pd.DataFrame:
        size = sample_size or self._config.sample_size
        size = min(size, len(features))
        
        if size <= 0:
            return pd.DataFrame()
        
        entropies = self._entropy(lgbm_probas)
        
        # Priority logic
        high_mask = (entropies > self._config.entropy_high) & (k_scores > self._config.kscore_high)
        medium_mask = (entropies > self._config.entropy_high) | (k_scores > self._config.kscore_high)
        low_mask = ~medium_mask
        
        priorities = np.full(len(features), "LOW", dtype=object)
        priorities[high_mask] = "HIGH"
        priorities[medium_mask & ~high_mask] = "MEDIUM"
        
        # Sort by priority (HIGH > MEDIUM > LOW) then by entropy descending
        priority_order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
        sort_keys = [(priority_order[p], -e) for p, e in zip(priorities, entropies)]
        
        sorted_idx = np.argsort(sort_keys)
        selected_idx = sorted_idx[:size]
        
        result = features.iloc[selected_idx].copy()
        result["al_priority"] = priorities[selected_idx]
        result["al_entropy"] = entropies[selected_idx]
        result["al_k_score"] = k_scores[selected_idx]
        result["al_lgbm_proba"] = lgbm_probas[selected_idx]
        
        return result

    def priority_stats(self, priorities: pd.Series) -> dict:
        counts = priorities.value_counts().to_dict()
        total = len(priorities)
        return {
            "HIGH": counts.get("HIGH", 0),
            "MEDIUM": counts.get("MEDIUM", 0),
            "LOW": counts.get("LOW", 0),
            "total": total,
            "high_pct": counts.get("HIGH", 0) / total * 100 if total > 0 else 0,
            "medium_pct": counts.get("MEDIUM", 0) / total * 100 if total > 0 else 0,
            "low_pct": counts.get("LOW", 0) / total * 100 if total > 0 else 0,
        }


class FeedbackLoop:
    def __init__(self, sampler: UncertaintySampler | None = None):
        self._sampler = sampler or UncertaintySampler()

    def incremental_retrain(
        self,
        model,
        new_labels: pd.DataFrame,
        X_all: pd.DataFrame,
        y_all: pd.Series,
    ) -> dict:
        """
        Retrain model with new labeled samples using LightGBM's init_model.
        
        Args:
            model: Trained LightGBMClassifier
            new_labels: DataFrame with 'address' and 'label' columns
            X_all: Full feature matrix
            y_all: Full labels (including newly labeled)
            
        Returns:
            Dict with updated model and metadata
        """
        # Combine old and new labels
        combined_y = y_all.copy()
        for _, row in new_labels.iterrows():
            addr = row["address"]
            label = row["label"]
            mask = X_all.index == addr
            if mask.any():
                combined_y.loc[mask] = label
        
        # Retrain with init_model
        model.fit(X_all, combined_y, X_cal=None, y_cal=None)
        
        return {
            "model": model,
            "n_new_samples": len(new_labels),
            "n_total_samples": len(combined_y),
        }

    def select_for_labeling(
        self,
        features: pd.DataFrame,
        lgbm_probas: np.ndarray,
        k_scores: np.ndarray,
        sample_size: int | None = None,
    ) -> pd.DataFrame:
        return self._sampler.select(features, lgbm_probas, k_scores, sample_size)