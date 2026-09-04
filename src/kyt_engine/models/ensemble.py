from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from kyt_engine.features._utils import find_best_threshold, prepare_features
from kyt_engine.models.autoencoder import AutoencoderDetector
from kyt_engine.models.lightgbm_model import LightGBMClassifier


class StackingEnsemble:
    def __init__(
        self,
        lgbm_params: dict | None = None,
        ae_params: dict | None = None,
        meta_max_iter: int = 1000,
        random_state: int = 42,
    ) -> None:
        lgbm_kw = lgbm_params or {}
        ae_kw = ae_params or {}
        self._lgbm = LightGBMClassifier(random_state=random_state, **lgbm_kw)
        self._ae = AutoencoderDetector(random_state=random_state, **ae_kw)
        self._meta = LogisticRegression(
            max_iter=meta_max_iter,
            random_state=random_state,
            class_weight="balanced",
        )
        self._threshold: float = 0.5
        self._feature_names: list[str] = []

    @staticmethod
    def _stack_predictions(
        lgbm_proba: np.ndarray,
        ae_proba: np.ndarray,
        beh_proba: np.ndarray | None,
    ) -> np.ndarray:
        # lgbm_proba[:, 1] is already a 1D array of shape (n,), reshape to (n, 1)
        stacks = [lgbm_proba[:, 1:2] if lgbm_proba.ndim == 2 else lgbm_proba[:, None],
                  ae_proba[:, 1:2] if ae_proba.ndim == 2 else ae_proba[:, None]]
        if beh_proba is not None:
            stacks.append(np.asarray(beh_proba).reshape(-1, 1))
        return np.hstack(stacks)

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        X_cal: pd.DataFrame | None = None,
        y_cal: pd.Series | None = None,
        behavioral_proba: np.ndarray | None = None,
    ) -> StackingEnsemble:
        X_df, y_s = prepare_features(X, y)
        self._feature_names = list(X_df.columns)

        self._lgbm.fit(X_df, y_s, X_cal=X_cal, y_cal=y_cal)
        lgbm_proba = self._lgbm.predict_proba(X_df)

        self._ae.fit(X_df, y_s)
        ae_proba = self._ae.predict_proba(X_df)

        meta_X = self._stack_predictions(lgbm_proba, ae_proba, behavioral_proba)
        self._meta.fit(meta_X, y_s.astype(int))

        meta_proba = self._meta.predict_proba(meta_X)[:, 1]
        self._threshold = find_best_threshold(meta_proba, y_s.values)
        return self

    def predict_proba(
        self, X: pd.DataFrame, behavioral_proba: np.ndarray | None = None
    ) -> np.ndarray:
        X_df, _ = prepare_features(X)
        lgbm_proba = self._lgbm.predict_proba(X_df)
        ae_proba = self._ae.predict_proba(X_df)
        meta_X = self._stack_predictions(lgbm_proba, ae_proba, behavioral_proba)
        return self._meta.predict_proba(meta_X)

    def predict(
        self, X: pd.DataFrame, behavioral_proba: np.ndarray | None = None
    ) -> np.ndarray:
        proba = self.predict_proba(X, behavioral_proba)[:, 1]
        return (proba >= self._threshold).astype(int)

    @property
    def threshold(self) -> float:
        return self._threshold

    @property
    def feature_names(self) -> list[str]:
        return list(self._feature_names)
