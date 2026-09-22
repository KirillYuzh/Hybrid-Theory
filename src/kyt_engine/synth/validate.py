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
    """Contract validator mirroring the dataset loader expectations.

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
    """Load a dataset the same way the dataset loader does (no merge)."""
    features, classes, edgelist = validate_dataset(root)
    return features, classes, edgelist


def validate_edge_attributes(root: Path) -> pd.DataFrame:
    """Structural checks only (row alignment + ranges).

    Invariant-level checks (mixer fee, peel monotony, wash balance) live in
    tests/test_synth.py::test_edge_attribute_invariants, not here.
    """
    path = root / "elliptic_txs_edge_attributes.csv"
    if not path.exists():
        raise FileNotFoundError(f"edge attributes not found: {path}")
    edgelist = pd.read_csv(root / "elliptic_txs_edgelist.csv")
    attrs = pd.read_csv(path)
    _check(list(attrs.columns) == ["txId1", "txId2", "amount", "timestamp"], list(attrs.columns))
    _check(
        len(attrs) == len(edgelist),
        f"edge attributes rows {len(attrs)}, edgelist {len(edgelist)}",
    )
    _check(attrs["txId1"].astype(int).equals(edgelist["txId1"]), "txId1 mismatch with edgelist")
    _check(attrs["txId2"].astype(int).equals(edgelist["txId2"]), "txId2 mismatch with edgelist")
    _check((attrs["amount"] >= 0).all() and attrs["amount"].notna().all(), "amount must be >= 0")
    _check(attrs["timestamp"].ge(0).all(), "timestamp must be >= 0")
    return attrs


def validate_semantic_features(root: Path) -> pd.DataFrame:
    """Cross-check the semantic feature columns against the edgelist and edge attributes.

    Column plumbing of the fixed layout in features_semantic.SEMANTIC_COLUMNS:
      index 0 -- in_degree  -> feat_2
      index 1 -- out_degree -> feat_3
      index 2 -- value_out  -> feat_4 (log1p of summed outgoing amounts)
      index 3 -- value_in   -> feat_5 (log1p of summed incoming amounts)
    """
    features, _classes, edgelist = validate_dataset(root)
    attrs = validate_edge_attributes(root)
    # Edge attributes are row-aligned with the edgelist (validated above); merging could
    # silently fan out duplicate (u,v) pairs in the p2p background, doubling sums.
    f = features.set_index("txId")

    in_deg = edgelist.groupby("txId2").size().reindex(f.index, fill_value=0).astype(int)
    out_deg = edgelist.groupby("txId1").size().reindex(f.index, fill_value=0).astype(int)
    _check((f["feat_2"] == in_deg).all(), "in_degree feature (feat_2) mismatch with edgelist")
    _check((f["feat_3"] == out_deg).all(), "out_degree feature (feat_3) mismatch with edgelist")

    import numpy as np

    value_in = attrs.groupby("txId2")["amount"].sum().reindex(f.index, fill_value=0.0)
    value_out = attrs.groupby("txId1")["amount"].sum().reindex(f.index, fill_value=0.0)
    log1p = np.vectorize(lambda x: np.log1p(max(x, 0.0)))
    _check(
        np.allclose(f["feat_5"].astype(float).to_numpy(), log1p(value_in.to_numpy()), atol=1e-9),
        "value_in feature (feat_5) mismatch with summed edge amounts",
    )
    _check(
        np.allclose(f["feat_4"].astype(float).to_numpy(), log1p(value_out.to_numpy()), atol=1e-9),
        "value_out feature (feat_4) mismatch with summed edge amounts",
    )
    return features
