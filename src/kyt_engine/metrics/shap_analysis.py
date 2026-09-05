
import numpy as np
import pandas as pd
import shap


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
    result = pd.DataFrame(
        {"feature": X.columns, "importance": mean_abs}
    ).sort_values("importance", ascending=False).head(max_display)
    return result


def dependence_plot(
    model,
    X: pd.DataFrame,
    feature: str,
) -> None:
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    shap.dependence_plot(feature, shap_values, X)


def summary_plot(
    model,
    X: pd.DataFrame,
    *,
    max_display: int = 20,
) -> None:
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    shap.summary_plot(shap_values, X, max_display=max_display)
