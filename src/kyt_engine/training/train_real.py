from __future__ import annotations

import logging
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report as sklearn_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from kyt_engine.features._utils import find_best_threshold, prepare_features
from kyt_engine.features.behavioral_v2 import compute_behavioral_features

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = ROOT / "data" / "raw"
MODELS_DIR = ROOT / "models"


def load_elliptic_data() -> tuple[pd.DataFrame, dict[int, int], pd.DataFrame]:
    features_path = DATA_DIR / "elliptic_txs_features.csv"
    classes_path = DATA_DIR / "elliptic_txs_classes.csv"
    edges_path = DATA_DIR / "elliptic_txs_edgelist.csv"

    logger.info("Loading features from %s", features_path)
    cols = ["txId", "time_step"] + [f"f{i}" for i in range(165)]
    features = pd.read_csv(features_path, header=None, names=cols)
    logger.info("Features: %s", features.shape)

    logger.info("Loading classes from %s", classes_path)
    classes = pd.read_csv(classes_path)
    classes["label"] = classes["class"].map({"1": 0, "2": 1})
    classes = classes.dropna(subset=["label"])
    classes["label"] = classes["label"].astype(int)
    label_map = dict(zip(classes["txId"], classes["label"]))
    logger.info("Labeled: %d (licit=%d, illicit=%d)",
                len(classes), (classes["label"] == 0).sum(), (classes["label"] == 1).sum())

    logger.info("Loading edges from %s", edges_path)
    edges = pd.read_csv(edges_path)
    logger.info("Edges: %d", len(edges))

    return features, label_map, edges


def build_graph_features(features: pd.DataFrame, edges: pd.DataFrame) -> pd.DataFrame:
    logger.info("Building graph features from %d edges", len(edges))
    t0 = time.time()

    tx_ids = features["txId"]
    tx_set = set(tx_ids)

    valid = edges[edges["txId1"].isin(tx_set) & edges["txId2"].isin(tx_set)].copy()

    out_deg = valid.groupby("txId1").size().rename("out_degree").reset_index().rename(columns={"txId1": "txId"})
    in_deg = valid.groupby("txId2").size().rename("in_degree").reset_index().rename(columns={"txId2": "txId"})

    graph_df = tx_ids.to_frame("txId")
    graph_df = graph_df.merge(out_deg, on="txId", how="left")
    graph_df = graph_df.merge(in_deg, on="txId", how="left")
    graph_df["in_degree"] = graph_df["in_degree"].fillna(0).astype(np.int32)
    graph_df["out_degree"] = graph_df["out_degree"].fillna(0).astype(np.int32)
    graph_df["total_degree"] = graph_df["in_degree"] + graph_df["out_degree"]
    graph_df["degree_ratio"] = np.where(
        graph_df["total_degree"] > 0,
        graph_df["in_degree"] / (graph_df["total_degree"] + 1e-9),
        0.0,
    )

    logger.info("Graph features built in %.1fs", time.time() - t0)
    return graph_df


def build_temporal_features(features: pd.DataFrame) -> pd.DataFrame:
    ts = features[["txId", "time_step"]].copy()
    time_counts = ts.groupby("time_step").size().rename("tx_count_in_step").reset_index()
    ts = ts.merge(time_counts, on="time_step", how="left")
    ts["tx_count_in_step"] = ts["tx_count_in_step"].fillna(0)
    return ts[["txId", "tx_count_in_step"]]


def train_lightgbm(
    X_train: pd.DataFrame, y_train: pd.Series,
    X_test: pd.DataFrame, y_test: pd.Series,
    n_estimators: int = 800,
) -> dict:
    from lightgbm import LGBMClassifier

    logger.info("Training LightGBM on %d samples", len(X_train))
    t0 = time.time()

    model = LGBMClassifier(
        n_estimators=n_estimators,
        learning_rate=0.05,
        num_leaves=127,
        max_depth=-1,
        min_child_samples=10,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        class_weight="balanced",
        random_state=42,
        verbose=-1,
        n_jobs=1,
    )

    model.fit(X_train, y_train)
    train_time = time.time() - t0

    proba = model.predict_proba(X_test)[:, 1]
    threshold = find_best_threshold(proba, y_test)
    pred = (proba >= threshold).astype(int)

    metrics = {
        "precision": float(precision_score(y_test, pred, zero_division=0)),
        "recall": float(recall_score(y_test, pred, zero_division=0)),
        "f1": float(f1_score(y_test, pred, zero_division=0)),
        "auc_roc": float(roc_auc_score(y_test, proba)),
        "auc_pr": float(average_precision_score(y_test, proba)),
        "threshold": float(threshold),
        "train_time_s": float(train_time),
    }

    logger.info("LightGBM done in %.1fs. Threshold=%.3f", train_time, threshold)
    logger.info("Classification report:\n%s", sklearn_report(y_test, pred, target_names=["licit", "illicit"]))

    importance = pd.Series(model.feature_importances_, index=X_train.columns)
    importance = importance.sort_values(ascending=False)
    importance.head(30).to_csv(MODELS_DIR / "lightgbm_top30_importance.csv")

    return {"model": model, "metrics": metrics, "proba": proba, "threshold": threshold}


def train_autoencoder(
    X_train: pd.DataFrame, y_train: pd.Series,
    X_test: pd.DataFrame, y_test: pd.Series,
) -> dict:
    from kyt_engine.models.autoencoder import AutoencoderDetector

    logger.info("Training Autoencoder on %d samples", len(X_train))
    t0 = time.time()

    ae = AutoencoderDetector(
        latent_dim=64,
        epochs=30,
        batch_size=512,
        lr=1e-3,
        contamination=0.5,
        device="cpu",
        random_state=42,
    )
    ae.fit(X_train, y_train)
    train_time = time.time() - t0

    proba = ae.predict_proba(X_test)[:, 1]
    pred = ae.predict(X_test)

    metrics = {
        "precision": precision_score(y_test, pred, zero_division=0),
        "recall": recall_score(y_test, pred, zero_division=0),
        "f1": f1_score(y_test, pred, zero_division=0),
        "auc_roc": roc_auc_score(y_test, proba),
        "auc_pr": average_precision_score(y_test, proba),
        "train_time_s": train_time,
    }

    logger.info("Autoencoder done in %.1fs", train_time)
    logger.info("Classification report:\n%s", sklearn_report(y_test, pred, target_names=["licit", "illicit"]))

    return {"model": ae, "metrics": metrics, "proba": proba}


def train_ensemble(
    lgbm_result: dict, ae_result: dict,
    X_val: pd.DataFrame, y_val: np.ndarray,
    X_test: pd.DataFrame, y_test: np.ndarray,
) -> dict:

    logger.info("Training Stacking Ensemble (meta on val, eval on test)")
    t0 = time.time()

    lgbm_proba_val = lgbm_result["model"].predict_proba(X_val)[:, 1]
    ae_proba_val = ae_result["model"].predict_proba(X_val)[:, 1]

    lgbm_proba_test = lgbm_result["model"].predict_proba(X_test)[:, 1]
    ae_proba_test = ae_result["model"].predict_proba(X_test)[:, 1]

    meta_X_val = np.column_stack([lgbm_proba_val, ae_proba_val])
    meta_X_test = np.column_stack([lgbm_proba_test, ae_proba_test])

    meta = LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")
    meta.fit(meta_X_val, y_val)

    proba = meta.predict_proba(meta_X_test)[:, 1]
    threshold = find_best_threshold(proba, y_test)
    pred = (proba >= threshold).astype(int)
    train_time = time.time() - t0

    metrics = {
        "precision": float(precision_score(y_test, pred, zero_division=0)),
        "recall": float(recall_score(y_test, pred, zero_division=0)),
        "f1": float(f1_score(y_test, pred, zero_division=0)),
        "auc_roc": float(roc_auc_score(y_test, proba)),
        "auc_pr": float(average_precision_score(y_test, proba)),
        "threshold": float(threshold),
        "train_time_s": float(train_time),
    }

    logger.info("Ensemble done in %.1fs. Threshold=%.3f", train_time, threshold)
    logger.info("Classification report:\n%s", sklearn_report(y_test, pred, target_names=["licit", "illicit"]))

    return {"model": meta, "metrics": metrics, "proba": proba, "threshold": threshold}


def save_model(model: object, name: str) -> Path:
    path = MODELS_DIR / f"{name}.pkl"
    with open(path, "wb") as f:
        pickle.dump(model, f)
    logger.info("Saved %s -> %s", name, path)
    return path


def temporal_split(
    features: pd.DataFrame, label_map: dict[int, int],
    train_end: int = 36, val_end: int = 44,
) -> dict:
    features = features.copy()
    features["label"] = features["txId"].map(label_map)
    features = features.dropna(subset=["label"])

    train_mask = features["time_step"] <= train_end
    val_mask = (features["time_step"] > train_end) & (features["time_step"] <= val_end)
    test_mask = features["time_step"] > val_end

    feature_cols = [c for c in features.columns if c not in ("txId", "label", "time_step")]

    result = {}
    for split_name, mask in [("train", train_mask), ("val", val_mask), ("test", test_mask)]:
        df = features[mask]
        X = df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        y = df["label"].values.astype(int)
        result[f"X_{split_name}"] = X
        result[f"y_{split_name}"] = y

    result["feature_cols"] = feature_cols
    return result


def drift_analysis(
    lgbm_model, ae_model, meta_model,
    features: pd.DataFrame, label_map: dict[int, int], feature_cols: list[str],
    threshold: float = 0.5,
) -> dict[int, dict[str, float]]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    features = features.copy()
    features["label"] = features["txId"].map(label_map)
    features = features.dropna(subset=["label"])

    results: dict[int, dict[str, float]] = {}
    for step in sorted(features["time_step"].unique()):
        mask = features["time_step"] == step
        df = features[mask]
        if len(df) < 10:
            continue
        X = df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        y = df["label"].values.astype(int)

        if len(np.unique(y)) < 2:
            logger.warning("time_step %d: fewer than 2 classes, skipping", step)
            continue

        lgbm_proba = lgbm_model.predict_proba(X)[:, 1]
        ae_proba = ae_model.predict_proba(X)[:, 1]
        meta_X = np.column_stack([lgbm_proba, ae_proba])
        proba = meta_model.predict_proba(meta_X)[:, 1]
        pred = (proba >= threshold).astype(int)

        results[int(step)] = {
            "f1": float(f1_score(y, pred, zero_division=0)),
            "auc_roc": float(roc_auc_score(y, proba)),
            "precision": float(precision_score(y, pred, zero_division=0)),
            "recall": float(recall_score(y, pred, zero_division=0)),
            "n_samples": int(mask.sum()),
        }

    reports_dir = ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    if results:
        steps = sorted(results.keys())
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle("Drift Analysis per Time Step", fontsize=14)
        for ax, metric in zip(axes.flat, ["f1", "auc_roc", "precision", "recall"]):
            vals = [results[s][metric] for s in steps]
            ax.plot(steps, vals, marker="o", linewidth=1.5)
            ax.set_title(metric)
            ax.set_xlabel("time_step")
            ax.set_ylabel(metric)
            ax.set_ylim(0, 1.05)
            ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(reports_dir / "drift_analysis.png", dpi=150)
        plt.close(fig)
        logger.info("Drift plot saved to %s", reports_dir / "drift_analysis.png")

    return results


def run_training() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    features, label_map, edges = load_elliptic_data()

    graph_feats = build_graph_features(features, edges)
    temporal_feats = build_temporal_features(features)

    features = features.merge(graph_feats, on="txId", how="left")
    features = features.merge(temporal_feats, on="txId", how="left")

    # Add from_address (Elliptic uses txId as proxy for address)
    if "from_address" not in features.columns:
        features["from_address"] = features["txId"].astype(str)
    if "to_address" not in features.columns:
        features["to_address"] = features["txId"].astype(str)

    # Compute behavioral v2 features (multi-window aggregations per address)
    logger.info("Computing behavioral v2 features...")
    bh = compute_behavioral_features(features)

    # Merge behavioral features back (left join on from_address)
    features = features.merge(bh, left_on="from_address", right_index=True, how="left")
    features = features.fillna(0.0)

    split = temporal_split(features, label_map, train_end=36, val_end=44)

    X_train, y_train = split["X_train"], split["y_train"]
    X_val, y_val = split["X_val"], split["y_val"]
    X_test, y_test = split["X_test"], split["y_test"]
    feature_cols = split["feature_cols"]

    lgbm_result = train_lightgbm(X_train, y_train, X_val, y_val)
    save_model(lgbm_result["model"], "lightgbm_real")

    ae_result = train_autoencoder(X_train, y_train, X_val, y_val)
    save_model(ae_result["model"], "autoencoder_real")

    ens_result = train_ensemble(lgbm_result, ae_result, X_val, y_val, X_test, y_test)
    save_model(ens_result["model"], "ensemble_real")

    logger.info("=" * 60)
    logger.info("FINAL RESULTS (test set)")
    logger.info("=" * 60)
    for name, res in [("LightGBM", lgbm_result), ("Autoencoder", ae_result), ("Ensemble", ens_result)]:
        logger.info("%s: %s", name, res["metrics"])

    try:
        drift = drift_analysis(
            lgbm_result["model"], ae_result["model"], ens_result["model"],
            features, label_map, feature_cols, threshold=ens_result["threshold"],
        )
        logger.info("Drift analysis: %d time steps evaluated", len(drift))
    except Exception as exc:
        logger.warning("Drift analysis failed: %s. Continuing without.", exc)

    logger.info("Training pipeline complete.")


if __name__ == "__main__":
    run_training()
