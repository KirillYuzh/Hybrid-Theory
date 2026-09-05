import pandas as pd

from kyt_engine.features.base import extract_base_features
from kyt_engine.features.behavioral import extract_behavioral_features


class FeatureEngineer:
    def __init__(self) -> None:
        self._is_fitted: bool = False
        self._feature_names: list[str] = []
        self._global_gas_median: float = 0.0

    def fit(self, df: pd.DataFrame):
        self._global_gas_median = float(df["gas_price"].median())
        sample = extract_base_features(df)
        behavioral = extract_behavioral_features(df, self._global_gas_median)
        combined = pd.concat([sample, behavioral], axis=1)
        self._feature_names = list(combined.columns)
        self._is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self._is_fitted:
            raise RuntimeError("FeatureEngineer must be fitted before transform")
        base = extract_base_features(df)
        behavioral = extract_behavioral_features(df, self._global_gas_median)
        return pd.concat([base, behavioral], axis=1)

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)

    @property
    def feature_names(self) -> list[str]:
        return list(self._feature_names)

    @property
    def n_features(self) -> int:
        return len(self._feature_names)