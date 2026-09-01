from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score

logger = logging.getLogger(__name__)


def create_early_warning_labels(
    features: pd.DataFrame,
    label_map: dict[int, int],
    horizon: int = 7,
) -> pd.DataFrame:
    """For each (address, time_step) pair, label=1 if address becomes illicit
    within the next `horizon` time steps.

    Args:
        features: DataFrame with columns [txId, time_step, f0..f164, ...]
        label_map: txId -> {0: licit, 1: illicit}
        horizon: number of future time steps to look ahead

    Returns:
        DataFrame with added column 'future_label' (0 or 1)
    """
    df = features.copy()
    df["label"] = df["txId"].map(label_map)
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)

    df = df.sort_values(["txId", "time_step"]).reset_index(drop=True)

    address_times = (
        df.groupby("txId")["time_step"]
        .apply(lambda s: sorted(s.tolist()))
        .to_dict()
    )

    future_label = np.zeros(len(df), dtype=int)

    for tx_id, group in df.groupby("txId"):
        times = np.array(address_times[tx_id])
        ts_values = group["time_step"].values
        labels = group["label"].values
        idxs = group.index.values

        illicit_mask = labels == 0
        if not np.any(illicit_mask):
            continue

        future_label[idxs[illicit_mask]] = 0

        licit_mask = labels == 1
        if not np.any(licit_mask):
            continue

        licit_ts = ts_values[licit_mask]
        licit_idxs = idxs[licit_mask]

        pos = np.searchsorted(times, licit_ts, side="right")
        valid = (pos < len(times)) & (times[pos] <= licit_ts + horizon)
        future_label[licit_idxs[valid]] = 1

    df["future_label"] = future_label
    return df


class EarlyWarningModel:
    """Predicts if an address will become illicit within `horizon` time steps.

    Uses behavioral features at the current time step to predict future status.
    Trained on addresses that have both current and future labels available.
    """

    def __init__(
        self,
        horizon: int = 7,
        train_end: int = 36,
        val_end: int = 44,
        n_estimators: int = 400,
        random_state: int = 42,
    ):
        self._horizon = horizon
        self._train_end = train_end
        self._val_end = val_end
        self._model = LGBMClassifier(
            n_estimators=n_estimators,
            learning_rate=0.05,
            num_leaves=63,
            class_weight="balanced",
            random_state=random_state,
            verbose=-1,
            n_jobs=1,
        )
        self._feature_names: list[str] = []

    def fit(
        self,
        features: pd.DataFrame,
        label_map: dict[int, int],
        feature_cols: list[str] | None = None,
    ) -> EarlyWarningModel:
        """Train on historical data where future labels are known.

        Uses temporal split: train on steps 1-36, validate on 37-40 (horizon to 44).
        """
        df = create_early_warning_labels(features, label_map, self._horizon)

        if feature_cols is None:
            exclude = {"txId", "label", "future_label", "time_step"}
            feature_cols = [c for c in df.columns if c not in exclude]

        self._feature_names = feature_cols

        train_mask = df["time_step"] <= self._train_end
        val_mask = (df["time_step"] > self._train_end) & (df["time_step"] <= self._val_end)

        X_train = (
            df.loc[train_mask, feature_cols]
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
        )
        y_train = df.loc[train_mask, "future_label"].values.astype(int)

        X_val = (
            df.loc[val_mask, feature_cols]
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
        )
        y_val = df.loc[val_mask, "future_label"].values.astype(int)

        self._model.fit(X_train, y_train)

        if len(X_val) > 0:
            val_proba = self._model.predict_proba(X_val)[:, 1]
            if len(np.unique(y_val)) > 1:
                val_auc = float(roc_auc_score(y_val, val_proba))
                logger.info("EarlyWarning val AUC: %.4f (n=%d)", val_auc, len(y_val))
            else:
                logger.info("EarlyWarning val: single class, skipping AUC (n=%d)", len(y_val))
        else:
            logger.info("EarlyWarning: no validation data in this range, skipping val AUC")

        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Returns P(future illicit) for each row."""
        X_clean = (
            X[self._feature_names]
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
        )
        return self._model.predict_proba(X_clean)[:, 1]

    def evaluate(
        self,
        features: pd.DataFrame,
        label_map: dict[int, int],
        feature_cols: list[str] | None = None,
    ) -> dict[str, float]:
        """Evaluate on test data (steps 45-49). Returns dict with early_warning_auc, etc."""
        df = create_early_warning_labels(features, label_map, self._horizon)

        if feature_cols is None:
            feature_cols = self._feature_names

        test_mask = df["time_step"] > self._val_end
        X_test = (
            df.loc[test_mask, feature_cols]
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
        )
        y_test = df.loc[test_mask, "future_label"].values.astype(int)

        proba = self._model.predict_proba(X_test)[:, 1]

        metrics: dict[str, float] = {
            "n_test": float(len(y_test)),
            "n_positive": float(y_test.sum()),
        }

        if len(np.unique(y_test)) > 1:
            metrics["early_warning_auc"] = float(roc_auc_score(y_test, proba))
        else:
            metrics["early_warning_auc"] = 0.0
            logger.warning("EarlyWarning test: single class, AUC=0")

        return metrics

    @property
    def feature_names(self) -> list[str]:
        return list(self._feature_names)
