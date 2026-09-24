"""Structural feature subspace computable from an edgelist + time_step alone.

Used for cross-dataset validation (downstream transfer, MMD/copula):
no amounts are needed, so the same features are computed for the synthetic dataset
and for real Elliptic from its raw edgelist. Pure pandas/numpy, deterministic, no RNG.
"""

from __future__ import annotations

import pandas as pd

from .stats import N_STEPS

STRUCTURAL_FEATURES = [
    "in_degree",
    "out_degree",
    "in_unique_neighbors",
    "out_unique_neighbors",
    "has_incoming",
    "has_outgoing",
    "step_norm",
]


def structural_features(edgelist: pd.DataFrame, time_step: pd.Series) -> pd.DataFrame:
    """Per-node structural features over a tx universe defined by `time_step`.

    `edgelist` must have columns [txId1, txId2]; `time_step` indexed by txId (all nodes,
    including isolated ones). Column order follows STRUCTURAL_FEATURES; parallel edges are
    allowed and inflate degree but not the unique-neighbor counts.
    """
    all_ids = time_step.index
    in_deg = edgelist.groupby("txId2").size().reindex(all_ids, fill_value=0).astype("int64")
    out_deg = edgelist.groupby("txId1").size().reindex(all_ids, fill_value=0).astype("int64")
    in_uniq = (
        edgelist.drop_duplicates(["txId2", "txId1"])
        .groupby("txId2")
        .size()
        .reindex(all_ids, fill_value=0)
        .astype("int64")
    )
    out_uniq = (
        edgelist.drop_duplicates(["txId1", "txId2"])
        .groupby("txId1")
        .size()
        .reindex(all_ids, fill_value=0)
        .astype("int64")
    )

    out = pd.DataFrame(index=all_ids)
    out["in_degree"] = in_deg
    out["out_degree"] = out_deg
    out["in_unique_neighbors"] = in_uniq
    out["out_unique_neighbors"] = out_uniq
    out["has_incoming"] = (in_deg > 0).astype("int64")
    out["has_outgoing"] = (out_deg > 0).astype("int64")
    out["step_norm"] = time_step.astype("float64") / N_STEPS
    return out[STRUCTURAL_FEATURES]
