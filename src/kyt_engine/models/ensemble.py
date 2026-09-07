import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
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
        return lgbm_proba[:, 1].reshape(-1, 1)

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

        if y_cal is not None:
            X_c, y_c = prepare_features(X_cal, y_cal)
            cal_proba = self._lgbm.predict_proba(X_c)[:, 1]
            self._meta.fit(self._stack_predictions(cal_proba), y_c.astype(int))
            meta_proba = self._meta.predict_proba(self._stack_predictions(self._lgbm.predict_proba(X_df)))[:, 1]
        else:
            n_splits = min(5, min(y_s.value_counts()))
            if n_splits < 2:
                self._meta.fit(self._stack_predictions(self._lgbm.predict_proba(X_df)), y_s.astype(int))
            else:
                skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self._lgbm._model.random_state)
                oof_proba = np.zeros(len(y_s))
                for train_idx, val_idx in skf.split(X_df, y_s):
                    self._lgbm._model.fit(X_df.iloc[train_idx], y_s.iloc[train_idx].values)
                    oof_proba[val_idx] = self._lgbm._model.predict_proba(X_df.iloc[val_idx])[:, 1]

                self._meta.fit(oof_proba.reshape(-1, 1), y_s.astype(int))

            meta_proba = self._meta.predict_proba(self._stack_predictions(self._lgbm.predict_proba(X_df)))[:, 1]

        self._threshold = find_best_threshold(meta_proba, y_s.values)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        X_df, _ = prepare_features(X)
        lgbm_proba = self._lgbm.predict_proba(X_df)
        return self._meta.predict_proba(self._stack_predictions(lgbm_proba))[:, 1]

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        proba = self.predict_proba(X)
        return (proba >= self._threshold).astype(int)

    @property
    def threshold(self) -> float:
        return self._threshold

    @property
    def feature_names(self) -> list[str]:
        return list(self._feature_names)