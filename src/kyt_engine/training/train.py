import logging
import pickle
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = ROOT / "data" / "raw"
MODELS_DIR = ROOT / "models"


def _load_data() -> tuple[pd.DataFrame, pd.Series]:
    from kyt_engine.data.elliptic import load_classes, load_nodes

    nodes = load_nodes(DATA_DIR)
    classes = load_classes(DATA_DIR)

    df = nodes.merge(classes, on="txId", how="inner")
    if df["label"].dtype == object:
        df["label"] = df["label"].map({"illicit": 1, "licit": 0}).fillna(0).astype(int)
    else:
        df["label"] = df["label"].astype(int)

    y = df["label"]
    return df, y


def _save_model(model: object, name: str) -> Path:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    path = MODELS_DIR / f"{name}.pkl"
    with open(path, "wb") as f:
        pickle.dump(model, f)
    logger.info("Saved %s -> %s", name, path)
    return path


def run_training() -> None:
    from kyt_engine.features.engine import FeatureEngineer
    from kyt_engine.metrics.classification import classification_report
    from kyt_engine.models.lightgbm_model import LightGBMClassifier

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    mlflow.set_experiment("kyt_engine_training")

    with mlflow.start_run(run_name="pipeline"):
        logger.info("Loading data from %s", DATA_DIR)
        df, y = _load_data()
        mlflow.log_param("n_samples", len(df))
        mlflow.log_param("n_features_raw", df.shape[1])
        mlflow.log_param("illicit_ratio", float(y.mean()))

        logger.info("Feature engineering (%d samples)", len(df))
        fe = FeatureEngineer()
        X = fe.fit_transform(df)
        mlflow.log_param("n_features_engineered", fe.n_features)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=123,
        )
        mlflow.log_param("train_size", len(X_train))
        mlflow.log_param("test_size", len(X_test))

        results: dict[str, dict[str, float]] = {}

        # LightGBM
        logger.info("Training LightGBM")
        lgbm = LightGBMClassifier()
        lgbm.fit(X_train, y_train, X_cal=X_test, y_cal=y_test)
        lgbm_pred = lgbm.predict(X_test)
        lgbm_proba = lgbm.predict_proba(X_test)[:, 1]
        lgbm_metrics = classification_report(y_test, lgbm_pred, lgbm_proba)
        results["lightgbm"] = lgbm_metrics["value"].to_dict()
        _save_model(lgbm, "lightgbm")
        for k, v in results["lightgbm"].items():
            mlflow.log_metric(f"lgbm_{k}", v)
        logger.info("LightGBM metrics:\n%s", lgbm_metrics)

        logger.info("=== Final Results ===")
        for model_name, m in results.items():
            logger.info("%s: %s", model_name, m)

    logger.info("Training pipeline complete.")


if __name__ == "__main__":
    run_training()