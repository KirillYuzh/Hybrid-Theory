"""Downstream transfer validation: does a model learn the same things on our
synthetic data as on the real Elliptic?

Shared structural feature space (features_structural) is computed for both the
synthetic run and real Elliptic (requires data/raw). A Random Forest is trained on
the synthetic train split and scored on the REAL held-out test split (transfer),
compared against (a) an RF trained on real train (oracle) and (b) its own synthetic
test split (internal). Report: F1, PR-AUC, ECE per scenario + the transfer gap.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .distribution import load_raw_elliptic
from .features_structural import STRUCTURAL_FEATURES, structural_features
from .stats import LABEL_TO_CLASS

TRAIN_STEP_MAX = 30
TEST_STEP_MIN = 41
_TARGET = "illicit"


def _split_pool(
    features: pd.DataFrame, time_step: pd.Series, labels: pd.Series
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """X/y for train (<=30), test (>=41); labels are 0/1 for the illicit class."""
    ts = time_step.reindex(features.index)
    X = features.to_numpy(dtype=np.float64)
    y = labels.reindex(features.index).fillna(0.0).to_numpy(dtype=np.float64)
    return {
        "train": (X[ts <= TRAIN_STEP_MAX], y[ts <= TRAIN_STEP_MAX]),
        "test": (X[ts >= TEST_STEP_MIN], y[ts >= TEST_STEP_MIN]),
    }


def _fit_rf(X: np.ndarray, y: np.ndarray, seed: int) -> object:
    from sklearn.ensemble import RandomForestClassifier

    model = RandomForestClassifier(n_estimators=200, random_state=seed, class_weight="balanced")
    return model.fit(X, y)


def _ece(y_true: np.ndarray, proba: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    total = len(y_true)
    if total == 0:
        return float("nan")
    acc = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        idx = (proba > lo) & (proba <= hi)
        idx |= (proba == lo) & (lo == 0.0)
        if not idx.any():
            continue
        acc += (np.abs(proba[idx].mean() - y_true[idx].mean()) * idx.sum()) / total
    return float(acc)


def _metrics(y_true: np.ndarray, proba: np.ndarray) -> dict:
    from sklearn.metrics import average_precision_score, f1_score

    pred = (proba >= 0.5).astype(int)
    return {
        "n": int(len(y_true)),
        "positive_rate": float(y_true.mean()),
        "f1": float(f1_score(y_true, pred, pos_label=1, zero_division=0)),
        "pr_auc": float(average_precision_score(y_true, proba)),
        "ece": _ece(y_true, proba),
    }


def _structural_pipeline(
    edgelist: pd.DataFrame, time_step: pd.Series, classes: pd.DataFrame
) -> pd.DataFrame:
    feat = structural_features(edgelist, time_step)
    is_illicit = classes.set_index("txId")["class"].map(LABEL_TO_CLASS) == _TARGET
    feat["_y"] = is_illicit.reindex(feat.index).fillna(False).astype(float)
    return feat


def run_downstream(
    synth_root: Path,
    raw_dir: Path,
    seed: int = 0,
) -> dict:
    """Full downstream report: RF transfer from synthetic -> held-out real Elliptic."""
    raw = load_raw_elliptic(raw_dir)
    synth_feat = pd.read_csv(synth_root / "elliptic_txs_features.csv", header=None)
    synth_feat.columns = ["txId", "time_step"] + [f"x{i}" for i in range(2, synth_feat.shape[1])]
    synth_ts = synth_feat.set_index("txId")["time_step"].astype("int64")
    synth_edg = pd.read_csv(synth_root / "elliptic_txs_edgelist.csv")
    synth_classes = pd.read_csv(synth_root / "elliptic_txs_classes.csv")

    synth = _structural_pipeline(synth_edg, synth_ts, synth_classes).sort_index()
    real = _structural_pipeline(raw["edgelist"], raw["time_step"], raw["classes"]).sort_index()

    pools_synth = _split_pool(synth[STRUCTURAL_FEATURES], synth_ts, synth["_y"])
    pools_real = _split_pool(real[STRUCTURAL_FEATURES], raw["time_step"], real["_y"])

    report: dict = {
        "feature_space": STRUCTURAL_FEATURES,
        "train_steps": f"<= {TRAIN_STEP_MAX}",
        "test_steps": f">= {TEST_STEP_MIN}",
        "n": {
            "synthetic_train": int(len(pools_synth["train"][0])),
            "synthetic_test": int(len(pools_synth["test"][0])),
            "real_train": int(len(pools_real["train"][0])),
            "real_test": int(len(pools_real["test"][0])),
        },
        "synthetic_trained": {
            "on_synthetic_test": None,
            "on_real_test": None,
        },
        "real_trained": {"on_real_test": None},
    }

    clf_synth = _fit_rf(*pools_synth["train"], seed)
    clf_real = _fit_rf(*pools_real["train"], seed)
    X_st, y_st = pools_synth["test"]
    X_rt, y_rt = pools_real["test"]

    p_st = clf_synth.predict_proba(X_st)[:, 1]
    p_sr = clf_synth.predict_proba(X_rt)[:, 1]
    p_rr = clf_real.predict_proba(X_rt)[:, 1]

    report["synthetic_trained"]["on_synthetic_test"] = _metrics(y_st, p_st)
    report["synthetic_trained"]["on_real_test"] = _metrics(y_rt, p_sr)
    report["real_trained"]["on_real_test"] = _metrics(y_rt, p_rr)

    real_f1 = report["real_trained"]["on_real_test"]["f1"]
    transfer_f1 = report["synthetic_trained"]["on_real_test"]["f1"]
    report["transfer_f1_gap"] = round(real_f1 - transfer_f1, 4)
    report["verdict"] = (
        "gap within the generator's own train/test spread" if abs(real_f1 - transfer_f1) <= 0.05
        else "synthetic-trained and real-trained RF diverge on real held-out data; "
        "treat the synthetic as a proxy with caution"
    )
    return report


def write_downstream_report(synth_root: Path, report: dict) -> Path:
    out = synth_root / "downstream_report.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return out
