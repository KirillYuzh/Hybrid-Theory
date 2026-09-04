from __future__ import annotations

import pandas as pd

from kyt_engine.core.contracts import FeatureVector, TxRecord
from kyt_engine.features.base import extract_base_features
from kyt_engine.features.behavioral import extract_behavioral_features


class FeatureEngineer:
    def __init__(self) -> None:
        self._global_gas_median: float = 0.0
        self._feature_names: list[str] = []
        self._is_fitted = False

    def fit(self, df: pd.DataFrame) -> "FeatureEngineer":
        self._global_gas_median = float(df["gas_price"].median())
        self._is_fitted = True
        sample = self.transform(df)
        self._feature_names = list(sample.columns)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self._is_fitted:
            raise RuntimeError("FeatureEngineer must be fitted before transform")
        base = extract_base_features(df)
        behavioral = extract_behavioral_features(df, self._global_gas_median)
        return pd.concat([base, behavioral], axis=1)

    def compute(self, tx: TxRecord, history: pd.DataFrame | None = None) -> FeatureVector:
        if tx.features:
            return FeatureVector(values=dict(tx.features))
        rows = [tx.__dict__]
        if history is not None:
            rows.extend(history.to_dict("records"))
        df = pd.DataFrame(rows)
        feats = self.transform(df)
        return FeatureVector(values=feats.iloc[0].to_dict())

    @property
    def feature_names(self) -> list[str]:
        return list(self._feature_names)
