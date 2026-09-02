from __future__ import annotations

import logging
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from kyt_engine.models.lightgbm_model import LGBM_TRAIN_CONFIG

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[3]
REPORTS_DIR = ROOT / "reports"
MODELS_DIR = ROOT / "models"


class DriftDetector:
    """Monitors feature distribution shifts using Population Stability Index (PSI)."""

    def __init__(self, threshold_warning: float = 0.10, threshold_retrain: float = 0.25):
        self._threshold_warning = threshold_warning
        self._threshold_retrain = threshold_retrain

    def compute_psi(self, reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
        ref_sorted = np.sort(reference)
        bin_edges = np.percentile(ref_sorted, np.linspace(0, 100, bins + 1))
        bin_edges[0] = -np.inf
        bin_edges[-1] = np.inf

        ref_counts, _ = np.histogram(reference, bins=bin_edges)
        cur_counts, _ = np.histogram(current, bins=bin_edges)

        ref_pct = ref_counts / ref_counts.sum()
        cur_pct = cur_counts / cur_counts.sum()

        ref_pct = np.clip(ref_pct, 1e-6, None)
        cur_pct = np.clip(cur_pct, 1e-6, None)

        psi = float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))
        return psi

    def check_drift(
        self,
        reference_features: pd.DataFrame,
        current_features: pd.DataFrame,
    ) -> dict[str, dict[str, float]]:
        common_cols = [c for c in reference_features.columns if c in current_features.columns]
        report: dict[str, dict[str, float]] = {}
        for col in common_cols:
            ref_vals = reference_features[col].values.astype(float)
            cur_vals = current_features[col].values.astype(float)
            ref_vals = ref_vals[np.isfinite(ref_vals)]
            cur_vals = cur_vals[np.isfinite(cur_vals)]
            if len(ref_vals) < 10 or len(cur_vals) < 10:
                report[col] = {"psi": 0.0, "status": "ok"}
                continue
            psi = self.compute_psi(ref_vals, cur_vals)
            if psi > self._threshold_retrain:
                status = "retrain"
            elif psi > self._threshold_warning:
                status = "warning"
            else:
                status = "ok"
            report[col] = {"psi": psi, "status": status}
        return report

    def should_retrain(self, drift_report: dict[str, dict[str, float]]) -> bool:
        return any(v["status"] == "retrain" for v in drift_report.values())


class IncrementalTrainer:
    """Handles incremental model updates using LightGBM continue_training."""

    def __init__(self, model_path: Path | None = None):
        self._model_path = model_path or MODELS_DIR

    def incremental_update(
        self,
        existing_model_path: Path,
        X_new: pd.DataFrame,
        y_new: np.ndarray,
        n_additional_estimators: int = 100,
    ) -> Path:
        from lightgbm import LGBMClassifier

        with open(existing_model_path, "rb") as f:
            existing_model = joblib.load(f)

        n_old = existing_model.n_estimators
        new_model = LGBMClassifier(
            n_estimators=n_old + n_additional_estimators,
            learning_rate=existing_model.learning_rate,
            num_leaves=existing_model.num_leaves,
            max_depth=existing_model.max_depth,
            min_child_samples=existing_model.min_child_samples,
            subsample=existing_model.subsample,
            colsample_bytree=existing_model.colsample_bytree,
            reg_alpha=existing_model.reg_alpha,
            reg_lambda=existing_model.reg_lambda,
            class_weight="balanced",
            random_state=42,
            verbose=-1,
            n_jobs=1,
        )

        new_model.fit(
            X_new,
            y_new,
            init_model=existing_model.booster_,
        )

        timestamp = int(time.time())
        updated_path = self._model_path / f"lightgbm_updated_{timestamp}.pkl"
        self._model_path.mkdir(parents=True, exist_ok=True)
        with open(updated_path, "wb") as f:
            joblib.dump(new_model, f)

        logger.info("Incremental update saved to %s (%d -> %d trees)",
                     updated_path, n_old, n_old + n_additional_estimators)
        return updated_path

    def full_retrain(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        model_name: str = "lightgbm_retrained",
    ) -> Path:
        from lightgbm import LGBMClassifier

        logger.info("Full retrain on %d samples", len(X_train))
        t0 = time.time()

        model = LGBMClassifier(**LGBM_TRAIN_CONFIG)
        model.fit(X_train, y_train)
        elapsed = time.time() - t0

        self._model_path.mkdir(parents=True, exist_ok=True)
        path = self._model_path / f"{model_name}.pkl"
        with open(path, "wb") as f:
            joblib.dump(model, f)

        logger.info("Full retrain done in %.1fs -> %s", elapsed, path)
        return path


def run_continuous_pipeline(
    features: pd.DataFrame,
    label_map: dict[int, int],
    feature_cols: list[str],
    reference_time_end: int = 36,
) -> dict:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    features = features.copy()
    features["label"] = features["txId"].map(label_map)
    features = features.dropna(subset=["label"])

    ref_mask = features["time_step"] <= reference_time_end
    recent_steps = sorted(features["time_step"].unique())
    recent_steps = [s for s in recent_steps if s > reference_time_end]
    recent_window = recent_steps[-4:] if len(recent_steps) >= 4 else recent_steps
    cur_mask = features["time_step"].isin(recent_window)

    ref_df = features.loc[ref_mask, feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    cur_df = features.loc[cur_mask, feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    if len(ref_df) == 0 or len(cur_df) == 0:
        logger.warning("Insufficient data for drift check (ref=%d, cur=%d)", len(ref_df), len(cur_df))
        return {
            "drift_report": {},
            "should_retrain": False,
            "retrain_method": "none",
            "metrics": {},
        }

    detector = DriftDetector()
    drift_report = detector.check_drift(ref_df, cur_df)
    retrain_needed = detector.should_retrain(drift_report)

    retrain_method = "none"
    updated_model_path = None
    metrics: dict = {}

    if retrain_needed:
        trainer = IncrementalTrainer()
        lgbm_path = MODELS_DIR / "lightgbm_real.pkl"

        y_ref = features.loc[ref_mask, "label"].values.astype(int)
        y_cur = features.loc[cur_mask, "label"].values.astype(int)

        if lgbm_path.exists() and len(np.unique(y_cur)) >= 2:
            try:
                updated_model_path = trainer.incremental_update(lgbm_path, cur_df, y_cur)
                retrain_method = "incremental"

                with open(updated_model_path, "rb") as f:
                    updated_model = joblib.load(f)
                proba = updated_model.predict_proba(cur_df)[:, 1]
                pred = (proba >= 0.5).astype(int)
                from sklearn.metrics import f1_score, roc_auc_score
                metrics = {
                    "f1": float(f1_score(y_cur, pred, zero_division=0)),
                    "auc_roc": float(roc_auc_score(y_cur, proba)),
                }
            except Exception as exc:
                logger.warning("Incremental update failed: %s. Falling back to full retrain.", exc)
                retrain_method = "full"

        if retrain_method == "none":
            X_all = pd.concat([ref_df, cur_df], axis=0)
            y_all = np.concatenate([y_ref, y_cur])
            updated_model_path = trainer.full_retrain(X_all, y_all)
            retrain_method = "full"

    report_path = REPORTS_DIR / "continuous_drift_report.txt"
    with open(report_path, "w") as f:
        f.write("Continuous Learning Drift Report\n")
        f.write("=" * 40 + "\n")
        f.write(f"Reference window: steps <= {reference_time_end}\n")
        f.write(f"Current window: steps {recent_window}\n")
        f.write(f"Retrain needed: {retrain_needed}\n")
        f.write(f"Retrain method: {retrain_method}\n")
        f.write(f"\nFeature drift summary ({len(drift_report)} features):\n")
        for feat, info in sorted(drift_report.items()):
            f.write(f"  {feat}: PSI={info['psi']:.4f} [{info['status']}]\n")
        if metrics:
            f.write(f"\nUpdated model metrics: {metrics}\n")
        if updated_model_path:
            f.write(f"Updated model: {updated_model_path}\n")

    logger.info("Drift report saved to %s", report_path)

    return {
        "drift_report": drift_report,
        "should_retrain": retrain_needed,
        "retrain_method": retrain_method,
        "metrics": metrics,
        "updated_model_path": str(updated_model_path) if updated_model_path else None,
    }
