import numpy as np
import pandas as pd
import shap
from typing import Optional, Dict, Any, List

from kyt_engine.core.contracts import FeatureVector, ScoreResult, TxRecord


class ShapAnalyzer:
    """
    SHAP-based explainability for KYT model decisions
    
    Provides local (per-transaction) and global (feature importance) explanations.

    Parameters
    ----------
    model : Any
        The trained model (e.g., LightGBM, XGBoost) for which SHAP explanations are to be generated.
    feature_names : Optional[List[str]], optional
        List of feature names corresponding to the model's input features. If not provided, attempts to infer from the model.
    model_name : str, optional
        Name of the model, used for reporting purposes. Default is "lightgbm".
    """

    def __init__(
        self,
        model,
        feature_names: Optional[List[str]] = None,
        model_name: str = "lightgbm",
    ):
        self.model = model
        self.model_name = model_name

        if feature_names is not None:
            self.feature_names = feature_names
        elif hasattr(model, "feature_names_in_"):
            self.feature_names = list(model.feature_names_in_)
        elif hasattr(model, "_feature_names"):
            self.feature_names = list(model._feature_names)
        else:
            self.feature_names = []

        self._explainer: Optional[shap.TreeExplainer] = None
        self._background_data: Optional[pd.DataFrame] = None

    def fit(self, df: pd.DataFrame, background_size: int = 100) -> "ShapAnalyzer":
        from kyt_engine.features.engine import FeatureEngineer

        fe = FeatureEngineer()
        X = fe.fit_transform(df)
        X_clean = X.select_dtypes(include=[np.number]).fillna(0.0)

        n_samples = min(background_size, len(X_clean))
        bg_idx = np.random.choice(len(X_clean), n_samples, replace=False)
        self._background_data = X_clean.iloc[bg_idx]

        booster = self.model._model if hasattr(self.model, "_model") else self.model
        self._explainer = shap.TreeExplainer(booster, self._background_data)
        return self

    def explain_tx(self, feature_dict: dict[str, float]) -> dict[str, float]:
        if self._explainer is None:
            raise RuntimeError("SHAP explainer not fitted. Call .fit() first.")

        X_row = pd.DataFrame([feature_dict], columns=self.feature_names).fillna(0.0)
        for col in self.feature_names:
            if col not in X_row.columns:
                X_row[col] = 0.0
        X_row = X_row[self.feature_names]

        shap_values = self._explainer.shap_values(X_row)

        if isinstance(shap_values, list):
            shap_values = shap_values[1] if len(shap_values) >= 2 else shap_values[0]

        if hasattr(shap_values, "shape") and shap_values.ndim > 1:
            shap_values = shap_values[0]

        return {fname: float(shap_values[i]) for i, fname in enumerate(self.feature_names) if i < len(shap_values)}

    def explain_result(self, result: ScoreResult) -> dict[str, Any]:
        explanation: dict[str, Any] = {
            "model": self.model_name,
            "risk_score": result.risk_score,
            "risk_zone": result.risk_zone,
            "triage_level": result.triage_level,
            "top_features": result.reasons[:5] if hasattr(result, "reasons") and result.reasons else [],
            "global_importance": {},
        }

        if self._explainer is not None and self._background_data is not None:
            X_bg = self._background_data[self.feature_names].fillna(0.0)
            shap_values = self._explainer.shap_values(X_bg)

            if isinstance(shap_values, list) and len(shap_values) >= 2:
                shap_values = shap_values[1]

            if hasattr(shap_values, "shape") and shap_values.ndim == 2:
                mean_abs = np.mean(np.abs(shap_values), axis=0)
                for i, fname in enumerate(self.feature_names):
                    if i < len(mean_abs):
                        explanation["global_importance"][fname] = float(mean_abs[i])

        return explanation

    def explain_batch(
        self, feature_dicts: list[dict[str, float]]
    ) -> list[dict[str, float]]:
        if self._explainer is None:
            raise RuntimeError("SHAP explainer not fitted. Call .fit() first.")

        return [self.explain_tx(fd) for fd in feature_dicts]


def feature_importance(
    model,
    X: pd.DataFrame,
    *,
    max_display: int = 20,
) -> pd.DataFrame:
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    mean_abs = np.abs(shap_values).mean(axis=0)
    return pd.DataFrame({"feature": X.columns, "importance": mean_abs}).sort_values("importance", ascending=False).head(max_display)


def dependence_plot(model, X: pd.DataFrame, feature: str) -> None:
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    shap.dependence_plot(feature, shap_values, X)


def summary_plot(model, X: pd.DataFrame, *, max_display: int = 20) -> None:
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    shap.summary_plot(shap_values, X, max_display=max_display)