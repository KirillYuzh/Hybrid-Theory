from __future__ import annotations

import numpy as np
import pandas as pd


class KScoreCalculator:
    """Computes K-Score (normality score) for transactions.

    K-Score = 0: absolutely normal transaction for this address
    K-Score = 1: extremely unusual transaction

    Based on z-score deviation from the address's behavioral baseline.
    """

    def __init__(self, baseline_window: int = 6, default_threshold: float = 0.7):
        self._baseline_window = baseline_window
        self._default_threshold = default_threshold
        self._baselines: pd.DataFrame = pd.DataFrame()
        self._feature_cols: list[str] = []

    def fit(self, features: pd.DataFrame, time_steps: pd.Series) -> KScoreCalculator:
        exclude = {"txId", "time_step", "from_address", "to_address", "label"}
        self._feature_cols = [c for c in features.columns if c not in exclude]

        addr_col = (
            "from_address" if "from_address" in features.columns else "txId"
        )

        grouped = features.groupby(time_steps)
        sorted_steps = sorted(grouped.groups.keys())
        baseline_steps = sorted_steps[: self._baseline_window]

        rows = []
        for step in baseline_steps:
            group = grouped.get_group(step)
            for addr in group[addr_col].unique():
                addr_rows = group[group[addr_col] == addr][self._feature_cols]
                for _, row in addr_rows.iterrows():
                    entry = {"addr": addr}
                    entry.update(row.to_dict())
                    rows.append(entry)

        df = pd.DataFrame(rows)
        stats = df.groupby("addr")[self._feature_cols].agg(["mean", "std"])
        stats.columns = [f"{col}__{stat}" for col, stat in stats.columns]
        stats = stats.fillna(0.0)
        self._baselines = stats
        return self

    def score(self, features: pd.DataFrame) -> pd.Series:
        addr_col = (
            "from_address"
            if "from_address" in features.columns
            else "txId"
        )

        zscores = pd.DataFrame(
            np.zeros((len(features), len(self._feature_cols))),
            columns=self._feature_cols,
        )

        for col in self._feature_cols:
            mean_key = f"{col}__mean"
            std_key = f"{col}__std"
            if mean_key in self._baselines.columns:
                means = features[addr_col].map(self._baselines[mean_key]).fillna(0.0)
                stds = features[addr_col].map(self._baselines[std_key]).fillna(1.0)
                stds = stds.clip(lower=1e-8)
                zscores[col] = (features[col].values - means.values) / stds.values
            else:
                zscores[col] = features[col].values

        abs_z = zscores.abs()
        k_scores = abs_z.mean(axis=1)
        k_scores = k_scores.clip(lower=0.0, upper=3.0) / 3.0
        return k_scores.reset_index(drop=True)

    def classify(self, k_scores: pd.Series) -> pd.Series:
        conditions = [
            k_scores < 0.3,
            k_scores <= 0.7,
            k_scores > 0.7,
        ]
        choices = ["GREEN", "YELLOW", "RED"]
        return pd.Series(np.select(conditions, choices, default="YELLOW"), name="zone")
