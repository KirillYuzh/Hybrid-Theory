from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.isotonic import IsotonicRegression

from kyt_engine.features._utils import find_best_threshold, prepare_features

LGBM_CONFIG: dict[str, object] = {
    "n_estimators": 500,
    "learning_rate": 0.05,
    "max_depth": -1,
    "num_leaves": 63,
    "min_child_samples": 20,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "class_weight": "balanced",
    "random_state": 42,
    "verbose": -1,
    "n_jobs": 1,
}

# Production-конфиг для полного обучения (используется в training/*)
LGBM_TRAIN_CONFIG: dict[str, object] = {
    "n_estimators": 800,
    "learning_rate": 0.05,
    "num_leaves": 127,
    "max_depth": -1,
    "min_child_samples": 10,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "class_weight": "balanced",
    "random_state": 42,
    "verbose": -1,
    "n_jobs": 1,
}


class LightGBMClassifier:
    def __init__(
        self,
        n_estimators: int = 500,
        learning_rate: float = 0.05,
        max_depth: int = -1,
        num_leaves: int = 63,
        min_child_samples: int = 20,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        reg_alpha: float = 0.1,
        reg_lambda: float = 1.0,
        random_state: int = 42,
    ) -> None:
        self._threshold: float = 0.5
        self._feature_names: list[str] = []

        params: dict[str, object] = {
            "n_estimators": n_estimators,
            "learning_rate": learning_rate,
            "max_depth": max_depth,
            "num_leaves": num_leaves,
            "min_child_samples": min_child_samples,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "reg_alpha": reg_alpha,
            "reg_lambda": reg_lambda,
            "class_weight": "balanced",
            "random_state": random_state,
            "verbose": -1,
            "n_jobs": 1,
        }

        self._model = LGBMClassifier(**params)
        self._calibrator: IsotonicRegression | None = None

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        X_cal: pd.DataFrame | None = None,
        y_cal: pd.Series | None = None,
    ) -> LightGBMClassifier:
        X_train, y_train = prepare_features(X, y)
        self._feature_names = list(X_train.columns)

        self._model.fit(X_train, y_train)
        raw = self._model.predict_proba(X_train)[:, 1]

        if X_cal is not None and y_cal is not None:
            X_c, y_c = prepare_features(X_cal, y_cal)
            cal_raw = self._model.predict_proba(X_c)[:, 1]
        else:
            cal_raw = raw
            y_c = y_train

        self._calibrator = IsotonicRegression(out_of_bounds="clip")
        self._calibrator.fit(cal_raw, y_c.astype(int))

        self._threshold = find_best_threshold(raw, y_train.values)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        X_, _ = prepare_features(X)
        raw = self._model.predict_proba(X_)[:, 1]
        if self._calibrator is not None:
            calibrated = self._calibrator.predict(raw)
        else:
            calibrated = raw
        return np.column_stack([1 - calibrated, calibrated])

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        proba = self.predict_proba(X)[:, 1]
        return (proba >= self._threshold).astype(int)

    @property
    def threshold(self) -> float:
        return self._threshold

    @property
    def feature_names(self) -> list[str]:
        return list(self._feature_names)
