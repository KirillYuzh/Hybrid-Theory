from __future__ import annotations

import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Tuple

import mlflow
import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from pyspark.sql import SparkSession

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"


def temporal_split(df: pd.DataFrame, train_end_day: int, val_end_day: int) -> tuple:
    """
    Split by timestamp (not random).
    Input: features DataFrame with timestamp column
    Output: (train_df, val_df, test_df)
    """
    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['day'] = (df['timestamp'] - pd.Timestamp('2024-01-01')).dt.days + 1

    train_df = df[df['day'] <= train_end_day].copy()
    val_df = df[(df['day'] > train_end_day) & (df['day'] <= val_end_day)].copy()
    test_df = df[df['day'] > val_end_day].copy()

    return train_df, val_df, test_df


class LightGBMTrainer:
    def __init__(self, params: dict | None = None, mlflow_experiment: str = "kyt_lightgbm"):
        self._params = params or {
            "objective": "binary",
            "metric": "auc",
            "learning_rate": 0.05,
            "num_leaves": 63,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "verbose": -1,
            "n_jobs": -1,
        }
        self._mlflow_experiment = mlflow_experiment

    def train(self, train_df: pd.DataFrame, val_df: pd.DataFrame) -> dict:
        """
        Train LightGBM with MLflow tracking.
        Returns: {'model': booster, 'metrics': dict, 'feature_importance': pd.DataFrame, 'run_id': str}

        Log to MLflow:
        - Parameters
        - Metrics (AUC-ROC, AUC-PR, Precision, Recall, F1 per class)
        - Feature importance plot
        - Model artifact
        """
        from lightgbm import LGBMClassifier
        from sklearn.metrics import classification_report, roc_auc_score, auc, precision_score, recall_score, f1_score

        X_train = train_df.drop(columns=["tx_id", "timestamp", "label"])
        y_train = train_df["label"]

        X_val = val_df.drop(columns=["tx_id", "timestamp", "label"])
        y_val = val_df["label"]

        mlflow.set_experiment(self._mlflow_experiment)

        with mlflow.start_run() as run:
            run_id = run.info.run_id

            for key, value in self._params.items():
                mlflow.log_param(key, value)

            model = LGBMClassifier(**self._params)
            model.fit(X_train, y_train)

            val_proba = model.predict_proba(X_val)
            if hasattr(val_proba, 'toarray'):
                val_proba = val_proba.toarray()
            val_proba = val_proba[:, 1]
            val_pred = (val_proba >= 0.5).astype(int)

            metrics = classification_report(y_val, val_pred, output_dict=True)

            auc_roc = roc_auc_score(y_val, val_proba)
            auc_pr = auc(val_proba, y_val.values)
            prec = precision_score(y_val, val_pred, zero_division="warn")
            rec = recall_score(y_val, val_pred, zero_division="warn")
            f1 = f1_score(y_val, val_pred, zero_division="warn")

            mlflow.log_metric("auc_roc", auc_roc)
            mlflow.log_metric("auc_pr", auc_pr)
            mlflow.log_metric("precision", prec)
            mlflow.log_metric("recall", rec)
            mlflow.log_metric("f1", f1)

            for class_name, class_metrics in metrics.items():
                if isinstance(class_metrics, dict):
                    for metric_name, metric_value in class_metrics.items():
                        if metric_name not in ("support", "f1-score"):
                            mlflow.log_metric(f"{class_name}_{metric_name}", metric_value)

            feature_importance = pd.DataFrame({
                "feature": X_train.columns,
                "importance": model.feature_importances_
            }).sort_values("importance", ascending=False)

            mlflow.log_param("n_features", X_train.shape[1])
            mlflow.log_param("n_train_samples", len(X_train))
            mlflow.log_param("n_val_samples", len(X_val))

            feature_importance.to_csv(mlflow.get_artifact_uri() + "/feature_importance.csv", index=False)

            mlflow.sklearn.log_model(model, "lightgbm_model")

            logger.info("LightGBM metrics: %s", {
                "auc_roc": auc_roc,
                "auc_pr": auc_pr,
                "precision": prec,
                "recall": rec,
                "f1": f1
            })

            return {
                "model": model,
                "metrics": {
                    "auc_roc": auc_roc,
                    "auc_pr": auc_pr,
                    "precision": prec,
                    "recall": rec,
                    "f1": f1,
                    "classification_report": metrics
                },
                "feature_importance": feature_importance,
                "run_id": run_id
            }


class VAETrainer:
    def __init__(self, latent_dim: int = 32, epochs: int = 50, mlflow_experiment: str = "kyt_vae"):
        self._latent_dim = latent_dim
        self._epochs = epochs
        self._mlflow_experiment = mlflow_experiment

    def train(self, train_df: pd.DataFrame, val_df: pd.DataFrame) -> dict:
        """
        Train VAE on licit-only data (anomaly detection).
        Use only features from licit transactions in train.
        Returns: {'model': VAE, 'metrics': dict, 'run_id': str}
        """
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
        from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score

        X_train = train_df.drop(columns=["tx_id", "timestamp", "label"])
        y_train = train_df["label"]

        X_val = val_df.drop(columns=["tx_id", "timestamp", "label"])
        y_val = val_df["label"]

        mlflow.set_experiment(self._mlflow_experiment)

        with mlflow.start_run() as run:
            run_id = run.info.run_id

            licit_mask = (y_train == 0)
            X_licit = X_train[licit_mask].values.astype(np.float32)

            col_means = X_licit.mean(axis=0)
            col_stds = X_licit.std(axis=0)
            col_stds[col_stds < 1e-8] = 1.0
            X_licit = (X_licit - col_means) / col_stds

            class VAE(nn.Module):
                def __init__(self, input_dim: int, latent_dim: int = 32):
                    super().__init__()
                    enc_dim = max(latent_dim * 2, input_dim // 3)
                    self._encoder = nn.Sequential(
                        nn.Linear(input_dim, enc_dim),
                        nn.ReLU(),
                        nn.Linear(enc_dim, latent_dim),
                        nn.ReLU(),
                    )
                    self._decoder = nn.Sequential(
                        nn.Linear(latent_dim, enc_dim),
                        nn.ReLU(),
                        nn.Linear(enc_dim, input_dim),
                    )

                def forward(self, x: torch.Tensor) -> torch.Tensor:
                    return self._decoder(self._encoder(x))

            model = VAE(X_train.shape[1], self._latent_dim)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            criterion = nn.MSELoss()

            dataset = TensorDataset(torch.tensor(X_licit, dtype=torch.float32))
            loader = DataLoader(dataset, batch_size=256, shuffle=True, num_workers=0)

            model.train()
            for _ in range(self._epochs):
                for (batch,) in loader:
                    optimizer.zero_grad(set_to_none=True)
                    recon = model(batch)
                    loss = criterion(recon, batch)
                    loss.backward()
                    optimizer.step()

            model.eval()
            with torch.no_grad():
                recon = model(torch.tensor(X_licit, dtype=torch.float32))
                errors = torch.mean((recon - torch.tensor(X_licit, dtype=torch.float32)) ** 2, dim=1).numpy()

            mlflow.log_param("latent_dim", self._latent_dim)
            mlflow.log_param("epochs", self._epochs)
            mlflow.log_param("licit_samples", len(X_licit))
            mlflow.log_param("total_features", X_train.shape[1])

            vae_scores = errors
            val_licit_scores = np.zeros(len(X_val))
            all_scores = np.concatenate([vae_scores, val_licit_scores])
            all_labels = np.concatenate([np.ones(len(X_licit)), np.zeros(len(X_val))])

            metrics = {
                "roc_auc": roc_auc_score(all_labels, all_scores),
                "precision": precision_score(all_labels, (all_scores > np.median(all_scores)).astype(int), zero_division="warn"),
                "recall": recall_score(all_labels, (all_scores > np.median(all_scores)).astype(int), zero_division="warn"),
                "f1": f1_score(all_labels, (all_scores > np.median(all_scores)).astype(int), zero_division="warn")
            }

            for k, v in metrics.items():
                mlflow.log_metric(k, v)

            mlflow.pytorch.log_model(model, "vae_model")

            logger.info("VAE metrics: %s", metrics)

            return {
                "model": model,
                "metrics": metrics,
                "run_id": run_id,
                "scaler_mean": col_means,
                "scaler_std": col_stds
            }


class EnsembleTrainer:
    def __init__(self, mlflow_experiment: str = "kyt_ensemble"):
        self._mlflow_experiment = mlflow_experiment

    def train(self, lgbm_train_preds, vae_train_preds, val_preds, val_labels) -> dict:
        """
        Train meta-learner on validation predictions.
        Returns: {'meta_model': sklearn_model, 'weights': np.array, 'metrics': dict, 'run_id': str}
        """
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score

        mlflow.set_experiment(self._mlflow_experiment)

        with mlflow.start_run() as run:
            run_id = run.info.run_id

            meta_X = np.column_stack([lgbm_train_preds, vae_train_preds])
            meta_y = val_labels.values

            meta_model = LogisticRegression(random_state=42)
            meta_model.fit(meta_X, meta_y)

            meta_proba = meta_model.predict_proba(meta_X)[:, 1]
            meta_pred = (meta_proba >= 0.5).astype(int)

            weights = meta_model.coef_[0]

            metrics = {
                "auc_roc": roc_auc_score(val_labels, meta_proba),
                "precision": precision_score(val_labels, meta_pred, zero_division="warn"),
                "recall": recall_score(val_labels, meta_pred, zero_division="warn"),
                "f1": f1_score(val_labels, meta_pred, zero_division="warn")
            }

            for k, v in metrics.items():
                mlflow.log_metric(k, v)

            mlflow.log_param("n_features", meta_X.shape[1])
            mlflow.log_param("n_samples", len(meta_X))

            mlflow.sklearn.log_model(meta_model, "ensemble_model")

            logger.info("Ensemble metrics: %s", metrics)

            return {
                "meta_model": meta_model,
                "weights": weights,
                "metrics": metrics,
                "run_id": run_id
            }


class ModelRegistry:
    """Save models to Iceberg table with versioning."""

    def __init__(self, spark: Any | None = None):
        self._spark = spark
        self._models_table = "kyt.models"

    def register_model(self, model_type: str, model_artifact_path: str,
                       metrics: dict, feature_importance: pd.DataFrame | None,
                       training_snapshot_id: str) -> str:
        """
        Insert into Iceberg "models" table.
        Returns model_id (UUID).
        """
        model_id = str(uuid.uuid4())

        model_data = {
            "model_id": model_id,
            "model_type": model_type,
            "version": "1.0.0",
            "metrics": str(metrics),
            "artifact_path": model_artifact_path,
            "trained_at": datetime.now().isoformat(),
            "training_data_snapshot": training_snapshot_id
        }

        if feature_importance is not None:
            model_data["feature_importance"] = feature_importance.to_json()

        if self._spark is not None:
            df = pd.DataFrame([model_data])
            spark_df = self._spark.createDataFrame(df)
            spark_df.write.format("iceberg").mode("append").save(self._models_table)
        else:
            import json
            registry_path = DATA_DIR / "model_registry.json"
            if registry_path.exists():
                with open(registry_path) as f:
                    registry = json.load(f)
            else:
                registry = []
            registry.append(model_data)
            with open(registry_path, 'w') as f:
                json.dump(registry, f, indent=2)

        logger.info("Registered model %s with ID: %s", model_type, model_id)

        return model_id


def run_training_pipeline(spark: Any | None = None,
                          features_table: str = "kyt.features",
                          output_table: str = "kyt.predictions",
                          features_df: pd.DataFrame | None = None) -> dict:
    """
    Full pipeline:
    1. Read features from Iceberg or use provided DataFrame
    2. Temporal split (train: day 1-36, val: 37-44, test: 45-49)
    3. Train LightGBM -> log to MLflow
    4. Train VAE -> log to MLflow
    5. Train Ensemble -> log to MLflow
    6. Score test set -> write to Iceberg "predictions"
    7. Register all models in Iceberg
    8. Return summary metrics
    """
    logger.info("Starting training pipeline")

    if features_df is None:
        if spark is not None:
            snapshot_id = spark.catalog.currentSnapshot(features_table).snapshotId()
            features_df = spark.table(features_table).toPandas()
        else:
            snapshot_id = "local-run"
            raise ValueError("Either spark or features_df must be provided")
    else:
        snapshot_id = "local-run"

    train_df, val_df, test_df = temporal_split(features_df, train_end_day=36, val_end_day=44)

    logger.info("Data split - Train: %d, Val: %d, Test: %d",
                len(train_df), len(val_df), len(test_df))

    lightgbm_trainer = LightGBMTrainer()
    lgbm_result = lightgbm_trainer.train(train_df, val_df)

    vae_trainer = VAETrainer(latent_dim=32, epochs=50)
    vae_result = vae_trainer.train(train_df, val_df)

    val_features = val_df.drop(columns=["tx_id", "timestamp", "label"])
    lgbm_val_proba = lgbm_result["model"].predict_proba(val_features)
    if hasattr(lgbm_val_proba, 'toarray'):
        lgbm_val_proba = lgbm_val_proba.toarray()
    lgbm_val_proba = lgbm_val_proba[:, 1]

    train_features = train_df.drop(columns=["tx_id", "timestamp", "label"])
    val_vae_input = (val_features.values.astype(np.float32) - vae_result["scaler_mean"]) / vae_result["scaler_std"]
    train_vae_input = (train_features.values.astype(np.float32) - vae_result["scaler_mean"]) / vae_result["scaler_std"]
    vae_val_scores = vae_result["model"](torch.tensor(val_vae_input, dtype=torch.float32))
    import torch
    vae_val_scores = torch.mean((vae_val_scores - torch.tensor(val_vae_input, dtype=torch.float32)) ** 2, dim=1).detach().numpy()
    vae_train_scores = vae_result["model"](torch.tensor(train_vae_input, dtype=torch.float32))
    vae_train_scores = torch.mean((vae_train_scores - torch.tensor(train_vae_input, dtype=torch.float32)) ** 2, dim=1).detach().numpy()

    ensemble_trainer = EnsembleTrainer()
    ensemble_result = ensemble_trainer.train(
        lgbm_train_preds=lgbm_result["model"].predict_proba(train_features),
        vae_train_preds=vae_train_scores,
        val_preds=lgbm_val_proba,
        val_labels=val_df["label"]
    )

    test_features = test_df.drop(columns=["tx_id", "timestamp", "label"])
    lgbm_test_proba = lgbm_result["model"].predict_proba(test_features)
    if hasattr(lgbm_test_proba, 'toarray'):
        lgbm_test_proba = lgbm_test_proba.toarray()
    lgbm_test_proba = lgbm_test_proba[:, 1]

    test_vae_input = (test_features.values.astype(np.float32) - vae_result["scaler_mean"]) / vae_result["scaler_std"]
    vae_test_scores = vae_result["model"](torch.tensor(test_vae_input, dtype=torch.float32))
    vae_test_scores = torch.mean((vae_test_scores - torch.tensor(test_vae_input, dtype=torch.float32)) ** 2, dim=1).detach().numpy()

    ensemble_meta_X = np.column_stack([lgbm_test_proba, vae_test_scores])
    ensemble_test_proba = ensemble_result["meta_model"].predict_proba(ensemble_meta_X)[:, 1]

    predictions_df = test_df.copy()
    predictions_df["lgbm_proba"] = lgbm_test_proba
    predictions_df["vae_anomaly"] = vae_test_scores
    predictions_df["k_score"] = np.random.rand(len(test_df))
    predictions_df["external_risk"] = np.random.rand(len(test_df))

    predictions_df["risk_score"] = 0.4 * lgbm_test_proba + 0.3 * ensemble_test_proba + 0.3 * vae_test_scores
    predictions_df["risk_zone"] = pd.cut(predictions_df["risk_score"], bins=[0, 0.3, 0.7, 1.0], labels=["GREEN", "YELLOW", "RED"], include_lowest=True)
    predictions_df["triage_level"] = pd.cut(predictions_df["risk_score"], bins=[0, 0.4, 0.7, 1.0], labels=["AUTO_CLOSE", "PRIORITY", "ESCALATION"], include_lowest=True)

    if spark is not None:
        spark_predictions_df = spark.createDataFrame(predictions_df)
        spark_predictions_df.write.format("iceberg").mode("append").save(output_table)

    model_registry = ModelRegistry(spark)
    lgbm_model_id = model_registry.register_model(
        "lightgbm",
        f"models/lightgbm/{lgbm_result['run_id']}",
        lgbm_result["metrics"],
        lgbm_result["feature_importance"],
        str(snapshot_id)
    )
    vae_model_id = model_registry.register_model(
        "vae",
        f"models/vae/{vae_result['run_id']}",
        vae_result["metrics"],
        None,
        str(snapshot_id)
    )
    ensemble_model_id = model_registry.register_model(
        "ensemble",
        f"models/ensemble/{ensemble_result['run_id']}",
        ensemble_result["metrics"],
        None,
        str(snapshot_id)
    )

    summary_metrics = {
        "lightgbm": lgbm_result["metrics"],
        "vae": vae_result["metrics"],
        "ensemble": ensemble_result["metrics"],
        "data_split": {
            "train": len(train_df),
            "val": len(val_df),
            "test": len(test_df)
        },
        "model_ids": {
            "lightgbm": lgbm_model_id,
            "vae": vae_model_id,
            "ensemble": ensemble_model_id
        }
    }

    logger.info("Training pipeline complete. Registered models with IDs: %s",
                summary_metrics["model_ids"])

    return summary_metrics
