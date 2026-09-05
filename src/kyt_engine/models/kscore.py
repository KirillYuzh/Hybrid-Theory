import numpy as np
import pandas as pd
from typing import Dict, Tuple, List, Optional


class KScoreCalculator:
    def __init__(self, baseline_steps: int = 6, norm_percentile: int = 99, epsilon: float = 1e-8):
        self.baseline_steps = baseline_steps
        self.norm_percentile = norm_percentile
        self.epsilon = epsilon
        self.baselines: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        self.norm_factor: float = 1.0
        self.is_fitted: bool = False
        self._feature_names: List[str] = []

    def fit(self, features: pd.DataFrame) -> "KScoreCalculator":
        if "address" not in features.columns:
            raise ValueError("features must contain 'address' column")
        
        feature_cols = [c for c in features.columns if c != "address"]
        if not feature_cols:
            raise ValueError("no feature columns found")
        
        self._feature_names = feature_cols
        
        if "timestamp_step" in features.columns:
            baseline_mask = features["timestamp_step"] <= self.baseline_steps
            if baseline_mask.any():
                baseline_data = features[baseline_mask]
            else:
                baseline_data = features
        else:
            baseline_data = features.groupby("address").head(self.baseline_steps).reset_index(drop=True)
        
        for addr, group in baseline_data.groupby("address"):
            vals = group[feature_cols].to_numpy(dtype=np.float64)
            if len(vals) == 0:
                continue
            mean = np.nanmean(vals, axis=0)
            std = np.nanstd(vals, axis=0, ddof=1)
            std = np.where(std < self.epsilon, self.epsilon, std)
            self.baselines[str(addr)] = (mean, std)
        
        all_scores: List[float] = []
        for addr, group in features.groupby("address"):
            scores = self._score_group(group[feature_cols].to_numpy(dtype=np.float64), str(addr))
            all_scores.extend(scores)
        
        if all_scores:
            self.norm_factor = float(np.percentile(all_scores, self.norm_percentile))
            self.norm_factor = max(self.norm_factor, 1.0)
        
        self.is_fitted = True
        return self

    def _score_group(self, group_features: np.ndarray, address: str) -> List[float]:
        if address not in self.baselines:
            return [0.0] * len(group_features)
        
        mean, std = self.baselines[address]
        z_scores = np.abs((group_features - mean) / std)
        k_scores = np.nanmean(z_scores, axis=1)
        return np.clip(k_scores / self.norm_factor, 0.0, 1.0).tolist()

    def score(self, features: pd.DataFrame) -> pd.Series:
        if not self.is_fitted:
            raise RuntimeError("KScoreCalculator must be fitted before scoring")
        
        feature_cols = [c for c in features.columns if c != "address"]
        if not feature_cols:
            return pd.Series([0.0] * len(features), index=features.index)
        
        scores: List[float] = []
        for addr, group in features.groupby("address"):
            group_scores = self._score_group(group[feature_cols].to_numpy(dtype=np.float64), str(addr))
            scores.extend(group_scores)
        
        return pd.Series(scores, index=features.index)

    def score_row(self, features: dict, address: str) -> float:
        if not self.is_fitted:
            raise RuntimeError("KScoreCalculator must be fitted before scoring")
        
        if address not in self.baselines:
            return 0.0
        
        vals = np.array([features.get(name, 0.0) for name in self._feature_names], dtype=np.float64)
        mean, std = self.baselines[address]
        z_scores = np.abs((vals - mean) / std)
        k_score = float(np.nanmean(z_scores))
        return min(k_score / self.norm_factor, 1.0)

    @property
    def feature_names(self) -> List[str]:
        return list(self._feature_names)