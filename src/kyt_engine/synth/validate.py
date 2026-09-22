from __future__ import annotations

from pathlib import Path

import pandas as pd

from .stats import N_STEPS as TIME_STEP_MAX
from .stats import STEP_MIN as TIME_STEP_MIN

VALID_CLASSES = {"1", "2", "unknown"}


def _read_dataset(root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    feat_path = root / "elliptic_txs_features.csv"
    if not feat_path.exists():
        raise FileNotFoundError(f"features not found: {feat_path}")

    features = pd.read_csv(feat_path, header=None)
    classes = pd.read_csv(root / "elliptic_txs_classes.csv")
    edgelist = pd.read_csv(root / "elliptic_txs_edgelist.csv")

    n_cols = features.shape[1]
    features.columns = ["txId", "time_step"] + [f"feat_{i}" for i in range(2, n_cols)]
    return features, classes, edgelist


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def validate_dataset(
    root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Contract validator mirroring spillety.data.loader expectations.

    - features: 167 columns [txId, time_step, feat_2..feat_166], int time_step, no NaN;
    - classes: [txId, class], class in {"1", "2", "unknown"};
    - edgelist: [txId1, txId2]; tx IDs consistent across files.
    """
    features, classes, edgelist = _read_dataset(root)

    _check(features.shape[1] == 167, f"features has {features.shape[1]} columns, expected 167")
    _check(
        classes.shape == (len(features), 2),
        f"classes shape {classes.shape}, expected ({len(features)}, 2)",
    )
    _check(list(classes.columns) == ["txId", "class"], list(classes.columns))
    _check(edgelist.shape[1] == 2, f"edgelist has {edgelist.shape[1]} columns, expected 2")
    _check(list(edgelist.columns) == ["txId1", "txId2"], list(edgelist.columns))
    _check(features["time_step"].dtype.kind == "i", features["time_step"].dtype)
    _check(
        features["time_step"].between(TIME_STEP_MIN, TIME_STEP_MAX).all(),
        "time_step out of 1..49",
    )
    _check(not features.iloc[:, 2:].isna().any().any(), "NaN in features")
    class_vals = set(classes["class"].astype(str).unique())
    _check(class_vals <= VALID_CLASSES, class_vals)
    _check(classes["txId"].is_unique, "dup txId in classes")
    _check(set(classes["txId"]) == set(features["txId"]), "txId mismatch feat/classes")
    return features, classes, edgelist


def load_elliptic(root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load a dataset the same way spillety.data.loader.load_elliptic does (no merge)."""
    features, classes, edgelist = validate_dataset(root)
    return features, classes, edgelist
