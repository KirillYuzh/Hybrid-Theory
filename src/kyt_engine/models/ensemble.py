import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from kyt_engine.features._utils import find_best_threshold, prepare_features
from kyt_engine.models.lightgbm_model import LightGBMClassifier


def _make_meta_lr(random_state: int = 123):
    return LogisticRegression(
        max_iter=1000,
        random_state=random_state,
        class_weight="balanced",
    )


class StackingEnsemble:
    def __init__(
        self,
        lgbm_params: dict | None = None,
        meta_max_iter: int = 1000,
        random_state: int = 123,
    ) -> None:
        self._lgbm = LightGBMClassifier(**(lgbm_params or {}))
        self._meta = _make_meta_lr(random_state)
        self._threshold: float = 0.5
        self._feature_names: list[str] = []

    def _stack_predictions(
        self,
        lgbm_proba: np.ndarray,
    ) -> np.ndarray:
        lgbm_stack = lgbm_proba[:, 1] if lgbm_proba.ndim == 2 else lgbm_proba
        return lgbm_stack.reshape(-1, 1)

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        X_cal: pd.DataFrame | None = None,
        y_cal: pd.Series | None = None,
    ) -> StackingEnsemble:
        X_df, y_s = prepare_features(X, y)
        self._feature_names = list(X_df.columns)

        self._lgbm.fit(X_df, y_s, X_cal=X_cal, y_cal=y_cal)
        lgbm_proba = self._lgbm.predict_proba(X_df)

        meta_X = self._stack_predictions(lgbm_proba)
        self._meta.fit(meta_X, y_s.astype(int))

        meta_proba = self._meta.predict_proba(meta_X)[:, 1]
        self._threshold = find_best_threshold(meta_proba, y_s.values)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        X_df, _ = prepare_features(X)
        lgbm_proba = self._lgbm.predict_proba(X_df)
        meta_X = self._stack_predictions(lgbm_proba)
        return self._meta.predict_proba(meta_X)[:, 1]

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        proba = self.predict_proba(X)
        return (proba >= self._threshold).astype(int)

    @property
    def threshold(self) -> float:
        return self._threshold

    @property
    def feature_names(self) -> list[str]:
        return list(self._feature_names)