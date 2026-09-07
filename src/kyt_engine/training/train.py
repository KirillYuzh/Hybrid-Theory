import logging
import pickle
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_recall_curve, auc, roc_auc_score, classification_report as sk_report
)

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = ROOT / "data" / "raw"
MODELS_DIR = ROOT / "models"

LOADERS = {
    "bitcoin": lambda d: ("elliptic", "nodes", "classes", "txId"),
    "ethereum": lambda d: ("ethereum", "transactions", None, None),
    "stableml": lambda d: ("stableaml", "transactions", None, None),
    "openml": lambda d: ("openaml", "transactions", None, None),
}


def _load_blockchain(name: str, data_dir: Path) -> tuple[pd.DataFrame, pd.Series]:
    from kyt_engine.data.validators import map_aml_labels

    mod_name, data_key, *extra = LOADERS[name](data_dir)
    module = getattr(__import__(f"kyt_engine.data.{mod_name}", fromlist=[f"load_{mod_name}_dataset"]), f"load_{mod_name}_dataset")
    data = module(data_dir)

    if name == "bitcoin":
        df = data["nodes"].merge(data["classes"], on="txId", how="inner")
    else:
        df = data[data_key]

    df = map_aml_labels(df)
    return df, df["label"].astype(int)


def _load_data(blockchains: list[str] | None = None) -> tuple[pd.DataFrame, pd.Series]:
    if blockchains is None:
        blockchains = ["bitcoin"]

    dfs, labels = [], []
    for bc in blockchains:
        if bc not in LOADERS:
            logger.warning("Unknown blockchain: %s, skipping", bc)
            continue
        df, y = _load_blockchain(bc, DATA_DIR)
        dfs.append(df)
        labels.append(y)
        logger.info("Loaded %s: %d samples", bc, len(df))

    if not dfs:
        raise ValueError("No data loaded - check blockchain names")

    combined_df = pd.concat(dfs, ignore_index=True)
    combined_y = pd.concat(labels, ignore_index=True)
    logger.info("Combined dataset: %d samples from %d blockchains", len(combined_df), len(blockchains))
    return combined_df, combined_y


def _save_model(model: object, name: str) -> Path:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    path = MODELS_DIR / f"{name}.pkl"
    with open(path, "wb") as f:
        pickle.dump(model, f)
    logger.info("Saved %s -> %s", name, path)
    return path


def _prepare_features(X: pd.DataFrame, for_kscore: bool = False) -> pd.DataFrame:
    X = X.copy()
    if "address" in X.columns and not for_kscore:
        X = X.drop(columns=["address"])
    if for_kscore and X.index.name == "address":
        X = X.reset_index(drop=True)
    if for_kscore and "address" not in X.columns:
        X["address"] = [f"addr_{i}" for i in range(len(X))]
    for col in X.columns:
        if X[col].dtype == object:
            X[col] = pd.to_numeric(X[col], errors="coerce").fillna(0.0)
    return X


def _k_score_distribution(y_true: pd.Series, k_scores: pd.Series) -> dict:
    green_mask = (k_scores < 0.3) & (y_true == 0)
    yellow_mask = (k_scores >= 0.3) & (k_scores < 0.7)
    red_mask = k_scores >= 0.7

    total = len(k_scores)
    if total == 0:
        return {"total": 0, "green_pct": 0, "yellow_pct": 0, "red_pct": 0}

    green_pct = round(green_mask.sum() / total * 100, 2)
    yellow_pct = round(yellow_mask.sum() / total * 100, 2)
    red_pct = round(red_mask.sum() / total * 100, 2)

    return {
        "total": total,
        "green_pct": green_pct,
        "yellow_pct": yellow_pct,
        "red_pct": red_pct,
        "green_illicit": int(((k_scores < 0.3) & (y_true == 1)).sum()),
        "yellow_illicit": int(((k_scores >= 0.3) & (k_scores < 0.7) & (y_true == 1)).sum()),
        "red_illicit": int(((k_scores >= 0.7) & (y_true == 1)).sum()),
        "green_target_met": green_pct >= 95.0,
        "yellow_target_met": yellow_pct <= 5.0,
        "red_target_met": red_pct <= 0.5,
    }


def run_training(blockchains: list[str] | None = None) -> dict:
    from kyt_engine.features.engine import FeatureEngineer
    from kyt_engine.models.lightgbm_model import LightGBMClassifier
    from kyt_engine.models.kscore import KScoreCalculator

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    mlflow.set_experiment("kyt_engine_training")

    with mlflow.start_run(run_name="multi_blockchain_pipeline"):
        df, y = _load_data(blockchains=blockchains)
        mlflow.log_param("n_samples", len(df))
        mlflow.log_param("n_blockchains", len(blockchains) if blockchains else 1)
        mlflow.log_param("class_distribution", str(y.value_counts().to_dict()))

        fe = FeatureEngineer()
        X = fe.fit_transform(df)
        mlflow.log_param("n_features_engineered", fe.n_features)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=123,
        )
        mlflow.log_param("train_size", len(X_train))
        mlflow.log_param("test_size", len(X_test))

        # LightGBM
        logger.info("Training LightGBM")
        lgbm = LightGBMClassifier()
        X_train_lgbm = _prepare_features(X_train)
        X_test_lgbm = _prepare_features(X_test)
        lgbm.fit(X_train_lgbm, y_train, X_cal=X_test_lgbm, y_cal=y_test)
        lgbm_pred = lgbm.predict(X_test_lgbm)
        lgbm_proba = lgbm.predict_proba(X_test_lgbm)[:, 1]

        report = sk_report(y_test, lgbm_pred, output_dict=True, zero_division=0)
        lgbm_metrics = {
            "precision": report["1"]["precision"],
            "recall": report["1"]["recall"],
            "f1": report["1"]["f1-score"],
            "auc_roc": roc_auc_score(y_test, lgbm_proba),
            "auc_pr": auc(*precision_recall_curve(y_test, lgbm_proba)[:2]),
        }
        _save_model(lgbm, "lightgbm")
        for k, v in lgbm_metrics.items():
            mlflow.log_metric(f"lgbm_{k}", v)

        # K-Score
        logger.info("Computing K-Score")
        kcalc = KScoreCalculator(baseline_steps=6, norm_percentile=99)
        X_train_ks = _prepare_features(X_train, for_kscore=True)
        X_test_ks = _prepare_features(X_test, for_kscore=True)
        kcalc.fit(X_train_ks)
        k_score_test = kcalc.score(X_test_ks)

        k_dist = _k_score_distribution(y_test, k_score_test)
        mlflow.log_metrics({f"k_{k}": v for k, v in k_dist.items()})
        logger.info("K-Score distribution: %s", k_dist)

    logger.info("Training pipeline complete.")
    return {"lightgbm": lgbm_metrics, "k_score_distribution": k_dist}


if __name__ == "__main__":
    import sys
    blockchains = sys.argv[1].split(",") if len(sys.argv) > 1 else None
    run_training(blockchains=blockchains)