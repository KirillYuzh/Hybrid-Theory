from __future__ import annotations

from typing import Sequence

import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def precision(
    y_true: Sequence[int],
    y_pred: Sequence[int],
) -> dict[str, float]:
    return {"precision": precision_score(y_true, y_pred, zero_division=0)}


def recall(
    y_true: Sequence[int],
    y_pred: Sequence[int],
) -> dict[str, float]:
    return {"recall": recall_score(y_true, y_pred, zero_division=0)}


def f1(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    *,
    average: str = "binary",
) -> dict[str, float]:
    return {"f1": f1_score(y_true, y_pred, average=average, zero_division=0)}


def auc_roc(
    y_true: Sequence[int],
    y_proba: Sequence[float],
) -> dict[str, float]:
    return {"auc_roc": roc_auc_score(y_true, y_proba)}


def auc_pr(
    y_true: Sequence[int],
    y_proba: Sequence[float],
) -> dict[str, float]:
    return {"auc_pr": average_precision_score(y_true, y_proba)}


def classification_report(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    y_proba: Sequence[float] | None = None,
) -> pd.DataFrame:
    rows = [
        {"metric": "precision", "value": precision_score(y_true, y_pred, zero_division=0)},
        {"metric": "recall", "value": recall_score(y_true, y_pred, zero_division=0)},
        {"metric": "f1", "value": f1_score(y_true, y_pred, zero_division=0)},
    ]
    if y_proba is not None:
        rows.append({"metric": "auc_roc", "value": roc_auc_score(y_true, y_proba)})
        rows.append({"metric": "auc_pr", "value": average_precision_score(y_true, y_proba)})
    return pd.DataFrame(rows).set_index("metric")
