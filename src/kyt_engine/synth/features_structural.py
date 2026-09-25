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
    """Compute structural features for the transaction IDs in ``time_step``.

    Parameters
    ----------
    edgelist : pandas.DataFrame
        Directed edges with ``txId1`` and ``txId2`` columns. Parallel edges affect
        degree but not unique-neighbor counts.
    time_step : pandas.Series
        Time step for each transaction, indexed by the node ID to be retained.

    Returns
    -------
    pandas.DataFrame
        Features indexed like ``time_step`` with columns in ``STRUCTURAL_FEATURES`` order.
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
