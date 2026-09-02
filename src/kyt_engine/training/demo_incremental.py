from __future__ import annotations

import logging
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score

from kyt_engine.training.train_real import (
    build_graph_features,
    build_temporal_features,
    load_elliptic_data,
)
from kyt_engine.features._utils import find_best_threshold, prepare_features
from kyt_engine.models.lightgbm_model import LGBM_TRAIN_CONFIG

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[3]
REPORTS_DIR = ROOT / "reports"


def _build_features(
    features: pd.DataFrame, edges: pd.DataFrame
) -> tuple[pd.DataFrame, list[str]]:
    graph_feats = build_graph_features(features, edges)
    temporal_feats = build_temporal_features(features)
    df = features.merge(graph_feats, on="txId", how="left")
    df = df.merge(temporal_feats, on="txId", how="left")
    feature_cols = [c for c in df.columns if c not in ("txId", "time_step")]
    return df, feature_cols


def _temporal_split(
    df: pd.DataFrame,
    label_map: dict[int, int],
    train_steps: tuple[int, int],
    eval_steps: tuple[int, int],
    feature_cols: list[str],
) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray]:
    mask_train = df["time_step"].between(train_steps[0], train_steps[1])
    mask_eval = df["time_step"].between(eval_steps[0], eval_steps[1])

    df_train = df[mask_train].copy()
    df_eval = df[mask_eval].copy()

    df_train["label"] = df_train["txId"].map(label_map)
    df_eval["label"] = df_eval["txId"].map(label_map)

    df_train = df_train.dropna(subset=["label"])
    df_eval = df_eval.dropna(subset=["label"])

    X_train = df_train[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y_train = df_train["label"].values.astype(int)
    X_eval = df_eval[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y_eval = df_eval["label"].values.astype(int)

    return X_train, y_train, X_eval, y_eval


def _eval_metrics(y_true: np.ndarray, proba: np.ndarray, prefix: str) -> dict[str, float]:
    threshold = find_best_threshold(proba, y_true)
    pred = (proba >= threshold).astype(int)
    return {
        f"{prefix}precision": float(precision_score(y_true, pred, zero_division=0)),
        f"{prefix}recall": float(recall_score(y_true, pred, zero_division=0)),
        f"{prefix}f1": float(f1_score(y_true, pred, zero_division=0)),
        f"{prefix}auc_roc": float(roc_auc_score(y_true, proba)),
        f"{prefix}auc_pr": float(average_precision_score(y_true, proba)),
        f"{prefix}threshold": float(threshold),
    }


def run_incremental_demo() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    from lightgbm import LGBMClassifier

    logger.info("Loading data")
    features, label_map, edges = load_elliptic_data()

    df, feature_cols = _build_features(features, edges)
    logger.info("Feature matrix: %d rows, %d features", len(df), len(feature_cols))

    lgbm_params = {**LGBM_TRAIN_CONFIG, "n_jobs": -1}

    X1, y1, X2a, y2a = _temporal_split(df, label_map, (1, 36), (37, 39), feature_cols)
    _, _, X3, y3 = _temporal_split(df, label_map, (1, 42), (40, 42), feature_cols)
    X23, y23, _, _ = _temporal_split(df, label_map, (37, 42), (40, 42), feature_cols)

    logger.info("Phase 1: train on steps 1-36 (%d samples)", len(X1))
    t0 = time.time()
    model_inc = LGBMClassifier(**lgbm_params)
    model_inc.fit(X1, y1)
    t1 = time.time()

    proba_2a = model_inc.predict_proba(X2a)[:, 1]
    m_phase1 = _eval_metrics(y2a, proba_2a, "phase1_")
    m_phase1["train_time_s"] = t1 - t0
    logger.info("Phase 1 eval: %s", {k: v for k, v in m_phase1.items() if k != "phase1_threshold"})

    logger.info("Phase 2: continue training on steps 37-39 (%d samples)", len(X2a))
    t0 = time.time()
    model_inc.fit(X2a, y2a, init_model=model_inc)
    t2 = time.time()

    proba_3 = model_inc.predict_proba(X3)[:, 1]
    m_phase2 = _eval_metrics(y3, proba_3, "phase2_")
    m_phase2["train_time_s"] = t2 - t0
    logger.info("Phase 2 eval: %s", {k: v for k, v in m_phase2.items() if k != "phase2_threshold"})

    logger.info("Full retrain: train on steps 1-42 (%d samples)", len(X1) + len(X2a) + len(X3))
    X_all = pd.concat([X1, X2a, X3], ignore_index=True)
    y_all = np.concatenate([y1, y2a, y3])
    t0 = time.time()
    model_full = LGBMClassifier(**lgbm_params)
    model_full.fit(X_all, y_all)
    t_full = time.time()

    proba_full_3 = model_full.predict_proba(X3)[:, 1]
    m_full = _eval_metrics(y3, proba_full_3, "full_")
    m_full["train_time_s"] = t_full - t0
    logger.info("Full retrain eval: %s", {k: v for k, v in m_full.items() if k != "full_threshold"})

    print("\n" + "=" * 70)
    print("INCREMENTAL vs FULL RETRAIN COMPARISON")
    print("=" * 70)
    header = f"{'Metric':<20} {'Incremental (P2)':<20} {'Full Retrain':<20}"
    print(header)
    print("-" * 70)
    for metric in ("precision", "recall", "f1", "auc_roc", "auc_pr"):
        inc_val = m_phase2.get(f"phase2_{metric}", 0.0)
        full_val = m_full.get(f"full_{metric}", 0.0)
        print(f"{metric:<20} {inc_val:<20.4f} {full_val:<20.4f}")
    print("-" * 70)
    inc_time = m_phase1.get("train_time_s", 0.0) + m_phase2.get("train_time_s", 0.0)
    full_time = m_full.get("train_time_s", 0.0)
    print(f"{'total_train_time':<20} {inc_time:<20.2f} {full_time:<20.2f}")
    print("=" * 70)

    phases = ["Phase 1\n(steps 1-36)", "Phase 2\n(incremental)", "Full Retrain"]
    f1_scores = [m_phase1["phase1_f1"], m_phase2["phase2_f1"], m_full["full_f1"]]
    auc_scores = [m_phase1["phase1_auc_roc"], m_phase2["phase2_auc_roc"], m_full["full_auc_roc"]]
    train_times = [m_phase1.get("train_time_s", 0), m_phase2.get("train_time_s", 0), full_time]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    x = np.arange(len(phases))

    axes[0].bar(x, f1_scores, color=["#4C72B0", "#55A868", "#C44E52"])
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(phases)
    axes[0].set_ylabel("F1 Score")
    axes[0].set_title("F1 Score Comparison")
    axes[0].set_ylim(0, max(f1_scores) * 1.2 + 0.01)
    for i, v in enumerate(f1_scores):
        axes[0].text(i, v + 0.005, f"{v:.3f}", ha="center", fontsize=9)

    axes[1].bar(x, auc_scores, color=["#4C72B0", "#55A868", "#C44E52"])
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(phases)
    axes[1].set_ylabel("AUC-ROC")
    axes[1].set_title("AUC-ROC Comparison")
    axes[1].set_ylim(0, max(auc_scores) * 1.2 + 0.01)
    for i, v in enumerate(auc_scores):
        axes[1].text(i, v + 0.005, f"{v:.3f}", ha="center", fontsize=9)

    axes[2].bar(x, train_times, color=["#4C72B0", "#55A868", "#C44E52"])
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(phases)
    axes[2].set_ylabel("Time (s)")
    axes[2].set_title("Training Time")
    for i, v in enumerate(train_times):
        axes[2].text(i, v + 0.1, f"{v:.1f}s", ha="center", fontsize=9)

    plt.suptitle("LightGBM: Incremental Learning vs Full Retrain on Elliptic Dataset", fontsize=13)
    plt.tight_layout()

    plot_path = REPORTS_DIR / "incremental_comparison.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved comparison plot to %s", plot_path)


if __name__ == "__main__":
    run_incremental_demo()
