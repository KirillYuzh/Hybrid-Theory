import numpy as np
import pandas as pd
from lightgbm.basic import LightGBMError
from typing import Protocol

from kyt_engine.core.contracts import FeatureVector, TxRecord


class Scorer(Protocol):
    name: str

    def predict_proba(self, features: FeatureVector, tx: TxRecord) -> float: ...


class LightGBMScorer:
    def __init__(self, model, feature_names: list[str] | None = None) -> None:
        self._model = model
        names = feature_names
        if names is None:
            if hasattr(model, "feature_names_in_"):
                names = list(model.feature_names_in_)
            elif hasattr(model, "_feature_names"):
                names = list(model._feature_names)
        self._feature_names = names or []
        self.name = "lightgbm"

    def predict_proba(self, features: FeatureVector, tx: TxRecord) -> float:
        expected_len = len(self._feature_names)
        if expected_len == 0:
            return 0.5

        input_features = features.values
        values = np.array([
            float(input_features.get(c, 0.0)) for c in self._feature_names
        ], dtype=np.float64).reshape(1, -1)

        X_df = pd.DataFrame(values, columns=self._feature_names)

        try:
            proba = self._model.predict_proba(X_df)[0, 1]
        except LightGBMError:
            return 0.5

        return float(np.clip(proba, 0.0, 1.0))


class VAEScorer:
    def __init__(self, model, feature_names: list[str]) -> None:
        self._model = model
        self._feature_names = feature_names
        self.name = "vae"

    def predict_proba(self, features: FeatureVector, tx: TxRecord) -> float:
        vec = np.array([features.values.get(c, 0.0) for c in self._feature_names])[None, :]
        proba = self._model.predict_proba(vec)[0, 1]
        return float(np.clip(proba, 0.0, 1.0))


class KScoreScorer:
    def __init__(self, calculator, feature_names: list[str]) -> None:
        self._calculator = calculator
        self._feature_names = feature_names
        self.name = "kscore"

    def predict_proba(self, features: FeatureVector, tx: TxRecord) -> float:
        df = pd.DataFrame([features.values], columns=self._feature_names)
        score = self._calculator.score(df).iloc[0]
        return float(np.clip(score, 0.0, 1.0))


class ExternalScorer:
    def __init__(self, illicit_addresses: set[str]) -> None:
        self._illicit = illicit_addresses
        self.name = "external"

    def predict_proba(self, features: FeatureVector, tx: TxRecord) -> float:
        return 1.0 if tx.from_address in self._illicit else 0.0