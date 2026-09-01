from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class UserAnomalyDetector:
    def __init__(
        self,
        baseline_window: int = 6,
        threshold_percentile: float = 95,
        entity_col: str | None = None,
    ) -> None:
        self._baseline_window = baseline_window
        self._threshold_percentile = threshold_percentile
        self._entity_col = entity_col
        self._baseline_mean: pd.DataFrame = pd.DataFrame()
        self._baseline_std: pd.DataFrame = pd.DataFrame()
        self._threshold: float = 0.0
        self._feature_cols: list[str] = []

    def _get_entity_ids(self, df: pd.DataFrame) -> pd.Index:
        if self._entity_col is not None and self._entity_col in df.columns:
            return df[self._entity_col]
        return df.index

    def fit(
        self,
        behavioral_features: pd.DataFrame,
        time_steps: pd.Series,
    ) -> UserAnomalyDetector:
        df = behavioral_features.copy()
        self._feature_cols = [c for c in df.columns if c != "time_step"]

        ts = time_steps.values if isinstance(time_steps, pd.Series) else time_steps
        entity_ids = self._get_entity_ids(df)

        baseline_mask = ts <= self._baseline_window
        baseline_df = df.loc[baseline_mask, self._feature_cols].copy()
        baseline_df["_entity"] = entity_ids[baseline_mask]

        self._baseline_mean = baseline_df.groupby("_entity").mean()
        self._baseline_std = baseline_df.groupby("_entity").std(ddof=0)
        self._baseline_std = self._baseline_std.clip(lower=1e-8)

        current_mask = ts > self._baseline_window
        current_df = df.loc[current_mask, self._feature_cols].copy()
        current_df["_entity"] = entity_ids[current_mask]

        if len(current_df) > 0:
            dev = self._compute_deviation(current_df)
            self._threshold = float(np.percentile(dev.values, self._threshold_percentile))
        else:
            self._threshold = 0.0

        return self

    def _compute_deviation(self, features: pd.DataFrame) -> pd.Series:
        entities = features["_entity"]
        feat_cols = [c for c in features.columns if c != "_entity"]
        feat_data = features[feat_cols]

        common_entities = entities.unique()
        common_entities = np.intersect1d(common_entities, self._baseline_mean.index)

        dev = pd.Series(0.0, index=features.index, dtype=np.float64)

        if len(common_entities) == 0:
            logger.warning("No entities found in baseline")
            return dev

        mask = entities.isin(common_entities)
        feat_aligned = feat_data.loc[mask]
        ent_aligned = entities.loc[mask]

        mean_aligned = self._baseline_mean.loc[ent_aligned.values]
        std_aligned = self._baseline_std.loc[ent_aligned.values]
        mean_aligned.index = feat_aligned.index
        std_aligned.index = feat_aligned.index

        z_scores = ((feat_aligned - mean_aligned) / std_aligned).abs()
        dev.loc[mask] = z_scores.mean(axis=1).values

        return dev

    def score(self, behavioral_features: pd.DataFrame) -> pd.Series:
        df = behavioral_features.copy()
        feature_cols = [c for c in df.columns if c != "time_step"]
        df["_entity"] = self._get_entity_ids(behavioral_features)
        raw = self._compute_deviation(df)
        if self._threshold > 0:
            normalized = np.clip(raw / self._threshold, 0.0, 1.0)
        else:
            normalized = raw * 0.0
        return pd.Series(normalized.values, index=behavioral_features.index, dtype=np.float64)

    def predict(self, behavioral_features: pd.DataFrame) -> pd.Series:
        scores = self.score(behavioral_features)
        return (scores >= 0.5).astype(int)
