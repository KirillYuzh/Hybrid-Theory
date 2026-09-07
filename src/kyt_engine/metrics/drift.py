import numpy as np
import pandas as pd
from typing import Optional, Dict, Any


class PSIDriftDetector:
    """Population Stability Index (PSI) drift detector.
    
    PSI < 0.1: no significant drift (green)
    0.1 <= PSI < 0.25: moderate drift (yellow)  
    PSI >= 0.25: significant drift (red) -> trigger retraining
    """

    def __init__(self, psi_threshold: float = 0.1, warning_threshold: float = 0.25):
        self.psi_threshold = psi_threshold
        self.warning_threshold = warning_threshold
        self._reference_data: Optional[pd.DataFrame] = None
        self._last_psi: Optional[float] = None
        self._last_status: str = "green"

    def fit(self, reference_data: pd.DataFrame) -> "PSIDriftDetector":
        self._reference_data = reference_data.copy()
        return self

    def compute_psi(self, current_data: pd.DataFrame, bins: int = 10) -> float:
        if self._reference_data is None:
            raise RuntimeError("Drift detector not fitted. Call .fit() first.")

        if current_data.empty:
            return 0.0

        ref_numeric = self._reference_data.select_dtypes(include=[np.number]).fillna(0.0)
        cur_numeric = current_data.select_dtypes(include=[np.number]).fillna(0.0)

        common_cols = list(set(ref_numeric.columns) & set(cur_numeric.columns))
        if not common_cols:
            return 0.0

        psi_total = 0.0
        for col in common_cols:
            ref_vals = ref_numeric[col].values
            cur_vals = cur_numeric[col].values

            ref_hist, bin_edges = np.histogram(ref_vals, bins=bins, density=True)
            cur_hist, _ = np.histogram(cur_vals, bins=bin_edges, density=True)

            ref_hist = np.clip(ref_hist, 1e-6, 1.0)
            cur_hist = np.clip(cur_hist, 1e-6, 1.0)

            psi_feature = np.sum((cur_hist - ref_hist) * np.log(cur_hist / ref_hist))
            psi_total += psi_feature

        psi_total = psi_total / len(common_cols)
        self._last_psi = psi_total

        if psi_total >= self.warning_threshold:
            self._last_status = "red"
        elif psi_total >= self.psi_threshold:
            self._last_status = "yellow"
        else:
            self._last_status = "green"

        return psi_total

    def detect_drift(self, current_data: pd.DataFrame, bins: int = 10) -> Dict[str, Any]:
        psi = self.compute_psi(current_data, bins=bins)

        return {
            "psi": round(psi, 6),
            "status": self._last_status,
            "psi_threshold": self.psi_threshold,
            "warning_threshold": self.warning_threshold,
            "drift_detected": psi >= self.psi_threshold,
            "significant_drift": psi >= self.warning_threshold,
            "green_zone": psi < self.psi_threshold,
            "yellow_zone": self.psi_threshold <= psi < self.warning_threshold,
            "red_zone": psi >= self.warning_threshold,
        }


class ModelDriftDetector:
    """Monitor model performance drift over time."""

    def __init__(
        self,
        auc_pr_threshold: float = 0.99,
        f1_threshold: float = 0.85,
        metric_window: int = 1000,
        degradation_threshold: float = 0.02,
    ):
        self.auc_pr_threshold = auc_pr_threshold
        self.f1_threshold = f1_threshold
        self.metric_window = metric_window
        self.degradation_threshold = degradation_threshold

        self._auc_pr_history: Optional[pd.Series] = None
        self._f1_history: Optional[pd.Series] = None
        self._timestamp_history: Optional[pd.Series] = None
        self._last_check: Optional[float] = None

    def update(self, auc_pr: float, f1: float, timestamp: float) -> "ModelDriftDetector":
        if self._auc_pr_history is None:
            self._auc_pr_history = pd.Series(dtype=float)
            self._f1_history = pd.Series(dtype=float)
            self._timestamp_history = pd.Series(dtype=float)

        self._auc_pr_history = pd.concat(
            [self._auc_pr_history, pd.Series([auc_pr])], ignore_index=True
        ).tail(self.metric_window)
        self._f1_history = pd.concat(
            [self._f1_history, pd.Series([f1])], ignore_index=True
        ).tail(self.metric_window)
        self._timestamp_history = pd.concat(
            [self._timestamp_history, pd.Series([timestamp])], ignore_index=True
        ).tail(self.metric_window)

        self._last_check = timestamp
        return self

    def check_drift(self) -> Optional[Dict[str, Any]]:
        if self._auc_pr_history is None or len(self._auc_pr_history) < 10:
            return None

        n = len(self._auc_pr_history)
        baseline_auc_pr = self._auc_pr_history.iloc[: max(1, n // 5)].mean()
        recent_auc_pr = self._auc_pr_history.iloc[max(1, 4 * n // 5) :].mean()

        auc_pr_degradation = baseline_auc_pr - recent_auc_pr

        alert: Optional[Dict[str, Any]] = None
        if auc_pr_degradation > self.degradation_threshold:
            alert = {
                "metric_name": "auc_pr",
                "baseline": round(baseline_auc_pr, 4),
                "recent": round(recent_auc_pr, 4),
                "degradation": round(auc_pr_degradation, 4),
                "threshold": self.degradation_threshold,
                "triggered": True,
            }

        if self._f1_history is not None and len(self._f1_history) >= 10:
            n_f1 = len(self._f1_history)
            baseline_f1 = self._f1_history.iloc[: max(1, n_f1 // 5)].mean()
            recent_f1 = self._f1_history.iloc[max(1, 4 * n_f1 // 5) :].mean()

            f1_degradation = baseline_f1 - recent_f1
            if f1_degradation > self.degradation_threshold:
                if alert is None:
                    alert = {}
                alert["metric_name"] = "f1"
                alert["baseline"] = round(baseline_f1, 4)
                alert["recent"] = round(recent_f1, 4)
                alert["degradation"] = round(f1_degradation, 4)
                alert["threshold"] = self.degradation_threshold
                alert["triggered"] = True

        return alert


class DataDriftMonitor:
    """Combines PSI feature drift with target distribution monitoring."""

    def __init__(
        self,
        psi_threshold: float = 0.1,
        psi_warning: float = 0.25,
        monitor_target_dist: bool = True,
    ):
        self.psi_detector = PSIDriftDetector(psi_threshold, psi_warning)
        self.monitor_target_dist = monitor_target_dist
        self._target_dist_ref: Optional[pd.Series] = None

    def fit(self, reference_data: pd.DataFrame, target_column: str = "label") -> "DataDriftMonitor":
        self.psi_detector.fit(reference_data)

        if self.monitor_target_dist and target_column in reference_data.columns:
            self._target_dist_ref = reference_data[target_column].value_counts(normalize=True)

        return self

    def monitor(
        self, current_data: pd.DataFrame, target_column: str = "label"
    ) -> Dict[str, Any]:
        psi_result = self.psi_detector.detect_drift(current_data)

        target_shift: Dict[str, Any] = {"status": "none", "details": {}}
        if self.monitor_target_dist and self._target_dist_ref is not None and target_column in current_data.columns:
            cur_target_dist = current_data[target_column].value_counts(normalize=True)
            common_labels = set(self._target_dist_ref.index) | set(cur_target_dist.index)
            dist_diff = {}
            for label in common_labels:
                ref_p = self._target_dist_ref.get(label, 0.0)
                cur_p = cur_target_dist.get(label, 0.0)
                dist_diff[label] = round(cur_p - ref_p, 6)

            total_var = 0.5 * sum(abs(d) for d in dist_diff.values())

            if total_var > 0.1:
                target_shift = {
                    "status": "drift",
                    "total_variation_distance": round(total_var, 6),
                    "by_label": dist_diff,
                }
            else:
                target_shift = {
                    "status": "stable",
                    "total_variation_distance": round(total_var, 6),
                    "by_label": dist_diff,
                }

        return {
            "psi": psi_result,
            "target_distribution": target_shift,
            "overall_status": (
                "alert" if psi_result["drift_detected"] or target_shift["status"] == "drift" else "stable"
            ),
        }