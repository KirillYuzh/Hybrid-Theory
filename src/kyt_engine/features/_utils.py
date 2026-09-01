from __future__ import annotations

import warnings
from collections import Counter

import numpy as np
import pandas as pd
from scipy import stats as sp_stats


def safe_float(val: object, default: float = 0.0) -> float:
    if val is None:
        return default
    try:
        f = float(val)
        if np.isnan(f) or np.isinf(f):
            return default
        return f
    except (TypeError, ValueError):
        return default


def counting_entropy(values: list[str]) -> float:
    if not values:
        return 0.0
    counts = Counter(values)
    total = sum(counts.values())
    probs = np.array([c / total for c in counts.values()])
    return float(-np.sum(probs * np.log2(probs)))


def discretized_entropy(values: np.ndarray, bins: int = 20) -> float:
    if len(values) == 0:
        return 0.0
    p99 = np.percentile(values, 99) + 1e-9
    edges = np.linspace(0, p99, bins)
    digitized = np.digitize(values, bins=edges)
    counts = np.bincount(digitized)
    probs = counts[counts > 0] / counts.sum()
    return float(-np.sum(probs * np.log2(probs)))


def suppress_runtime_warnings(func, *args, **kwargs):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        return func(*args, **kwargs)


def safe_skew(v: np.ndarray) -> float:
    return float(suppress_runtime_warnings(sp_stats.skew, v, bias=False))


def safe_kurtosis(v: np.ndarray) -> float:
    return float(suppress_runtime_warnings(sp_stats.kurtosis, v, bias=False))


def safe_linregress(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    try:
        res = suppress_runtime_warnings(sp_stats.linregress, x, y)
        return float(res.slope), float(res.rvalue) ** 2
    except (ValueError, ZeroDivisionError):
        return 0.0, 0.0


def find_best_threshold(
    proba: np.ndarray, y_true: np.ndarray, step: float = 0.01
) -> float:
    best_t, best_f1 = 0.5, 0.0
    for t in np.arange(0.1, 0.9, step):
        preds = (proba >= t).astype(int)
        tp = float(np.sum((preds == 1) & (y_true == 1)))
        fp = float(np.sum((preds == 1) & (y_true == 0)))
        fn = float(np.sum((preds == 0) & (y_true == 1)))
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        if f1 > best_f1:
            best_f1 = f1
            best_t = t
    return best_t


def prepare_features(
    X: pd.DataFrame, y: pd.Series | None = None
) -> tuple[pd.DataFrame, pd.Series | None]:
    df = X.copy()
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    target = y.copy() if y is not None else None
    return df, target


def extract_counterparties(group: pd.DataFrame, address: str) -> list[str]:
    from_addr = group["from_address"].astype(str)
    to_addr = group["to_address"].astype(str)
    is_out = from_addr == address
    is_in = to_addr == address
    cps = np.where(is_out, to_addr, np.where(is_in, from_addr, "")).tolist()
    return [c for c in cps if c]
