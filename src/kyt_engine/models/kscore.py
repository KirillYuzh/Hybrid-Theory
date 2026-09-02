"""K-Score Calculator — нормированный показатель «необычности» транзакции.

K-Score = 0: абсолютно нормальная транзакция для данного адреса
K-Score = 1: крайне необычная транзакция

Основа: z-score отклонение от поведенческого эталона клиента (user-level baseline)
за окна 30/60/90 дней. Не зависит от ground truth — считается от отклонений поведения.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Вычисление K-Score: deviation от behavioral baseline клиента (30/60/90 дней)
# Файл: src/kyt_engine/models/kscore.py


class KScoreCalculator:
    """Computes K-Score (normality score) for transactions.

    For each address, we maintain a behavioral baseline over a rolling window
    (30/60/90 days) and compute the z-score deviation for each feature.
    K-Score aggregates these deviations across all features.
    """

    def __init__(
        self,
        window_days: tuple[int, ...] = (30, 60, 90),
        default_threshold: float = 0.7,
        z_score_max: float = 3.0,
    ):
        self._window_days = window_days
        self._default_threshold = default_threshold
        self._z_score_max = z_score_max
        self._baselines: pd.DataFrame = pd.DataFrame()
        self._feature_cols: list[str] = []

    def fit(self, features: pd.DataFrame, time_steps: pd.Series) -> KScoreCalculator:
        """Обучает baseline на первых 20% временных шагов (эмуляция 30/60/90 дней).

        В production: скользящее окно по реальным датам. Здесь — эвристика
        по времени Elliptic (49 временных шагов).
        """
        exclude = {"txId", "time_step", "from_address", "to_address", "label"}
        self._feature_cols = [c for c in features.columns if c not in exclude]

        addr_col = "from_address" if "from_address" in features.columns else "txId"

        # Определяем baseline-период: первые 20% временных шагов (эмуляция 30 дней)
        n_steps = time_steps.nunique()
        baseline_step_count = max(int(n_steps * 0.2), 1)

        grouped = features.groupby(time_steps)
        sorted_steps = sorted(grouped.groups.keys())
        baseline_steps = sorted_steps[:baseline_step_count]

        rows = []
        for step in baseline_steps:
            group = grouped.get_group(step)
            for addr in group[addr_col].unique():
                addr_rows = group[group[addr_col] == addr][self._feature_cols]
                for _, row in addr_rows.iterrows():
                    entry = {"addr": addr}
                    entry.update(row.to_dict())
                    rows.append(entry)

        if not rows:
            self._baselines = pd.DataFrame()
            return self

        df = pd.DataFrame(rows)
        # Агрегируем по адресу: mean/std по каждой фиче
        stats = df.groupby("addr")[self._feature_cols].agg(["mean", "std"])
        stats.columns = [f"{col}__{stat}" for col, stat in stats.columns]
        stats = stats.fillna(0.0)
        self._baselines = stats
        return self

    def update(self, new_features: pd.DataFrame, time_step: int, window_days: int = 30) -> KScoreCalculator:
        """Инкрементальное обновление baseline с новыми данными (скользящее окно).

        Для continuous learning: каждые 24 часа пересчитываем baseline
        по последним 30/60/90 дням.
        """
        if self._baselines.empty:
            return self.fit(new_features, pd.Series(new_features.get("time_step", [0] * len(new_features))))

        addr_col = "from_address" if "from_address" in new_features.columns else "txId"
        rows = []
        for _, row in new_features[self._feature_cols + [addr_col]].iterrows():
            entry = {"addr": row[addr_col]}
            entry.update({c: row[c] for c in self._feature_cols})
            rows.append(entry)

        df_new = pd.DataFrame(rows)
        if df_new.empty:
            return self

        new_stats = df_new.groupby("addr")[self._feature_cols].agg(["mean", "std"])
        new_stats.columns = [f"{col}__{stat}" for col, stat in new_stats.columns]
        new_stats = new_stats.fillna(0.0)

        # Объединяем с существующим baseline (сглаживание)
        combined = pd.concat([self._baselines, new_stats]).groupby(level=0).mean()
        self._baselines = combined
        return self

    def score(self, features: pd.DataFrame) -> pd.Series:
        """Возвращает K-Score (0-1) для каждой транзакции."""
        if self._baselines.empty:
            return pd.Series(np.zeros(len(features), dtype=float), index=features.index)

        addr_col = "from_address" if "from_address" in features.columns else "txId"

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
        k_scores = k_scores.clip(lower=0.0, upper=self._z_score_max) / self._z_score_max
        return k_scores.reset_index(drop=True)

    def classify(self, k_scores: pd.Series) -> pd.Series:
        """Классифицирует K-Score в зоны GREEN/YELLOW/RED."""
        conditions = [
            k_scores < 0.3,
            k_scores <= self._default_threshold,
            k_scores > self._default_threshold,
        ]
        choices = ["GREEN", "YELLOW", "RED"]
        return pd.Series(np.select(conditions, choices, default="YELLOW"), name="zone")