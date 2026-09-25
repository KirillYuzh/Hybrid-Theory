import math

import numpy as np

from .graph import GeneratedGraph
from .stats import N_FEATURES, N_STEPS

# Column indices are zero-based within the 165 feature columns; CSV names are feat_{index + 2}.
# Amount aggregates, extrema, and standard deviations use log1p; the in/out ratio
# is a difference of log1p totals, while net flow and per-degree values remain dollars.
# Time deltas are steps; edge-hour columns use hours and time_step_norm is time_step / 49.
SEMANTIC_COLUMNS: dict[int, str] = {
    0: "in_degree",
    1: "out_degree",
    2: "value_out",
    3: "value_in",
    4: "net_flow",
    5: "log_ratio_in_out",
    6: "max_in_amount",
    7: "mean_in_amount",
    8: "max_out_amount",
    9: "mean_out_amount",
    10: "min_in_amount",
    11: "in_age_min",
    12: "in_age_mean",
    13: "in_age_max",
    14: "out_age_mean",
    15: "value_per_in",
    16: "value_per_out",
    17: "in_amount_std",
    18: "out_amount_std",
    19: "edge_hours_mean_in",
    20: "edge_hours_mean_out",
    21: "has_incoming",
    22: "has_outgoing",
    23: "time_step_norm",
}

CHAIN_SEMANTIC_COLUMNS: dict[int, str] = {
    24: "chain_bitcoin",
    25: "chain_ethereum",
    26: "chain_tron",
    27: "cross_chain_in",
    28: "cross_chain_out",
}

_N_SEMANTIC = len(SEMANTIC_COLUMNS)


def _log1p(value: float) -> float:
    return math.log1p(max(value, 0.0))


def _derived(S: np.ndarray, col: int) -> np.ndarray:
    """Create a deterministic nonlinear transform for one semantic feature column.

    Returns
    -------
    numpy.ndarray
        One transformed value per row of the base semantic feature matrix.
    """
    k = col - _N_SEMANTIC
    a = k % _N_SEMANTIC
    b = (k // _N_SEMANTIC) % _N_SEMANTIC
    c = (k // (_N_SEMANTIC * _N_SEMANTIC)) % _N_SEMANTIC
    la = np.log1p(np.abs(S[:, a]) + 1e-9)
    lb = np.log1p(np.abs(S[:, b]) + 1e-9)
    return la * lb * (1.0 + 0.01 * np.abs(S[:, c]))


def build_semantic_features(graph: GeneratedGraph, edge_attrs: list[dict | None]) -> np.ndarray:
    """Build the 165-column semantic feature matrix from graph structure.

    Parameters
    ----------
    graph : GeneratedGraph
        Graph providing node time steps, directed edges, and optional chain metadata.
    edge_attrs : list[dict | None]
        Edge attributes aligned with ``graph.edges``. A missing value is ignored,
        while a present record must provide ``amount`` and ``timestamp`` values.

    Returns
    -------
    numpy.ndarray
        Float64 matrix with shape ``(len(graph.nodes), 165)`` in canonical column order.
    """
    n = len(graph.nodes)
    pos = {nd.tx_id: i for i, nd in enumerate(graph.nodes)}
    step = np.array([nd.step for nd in graph.nodes], dtype=np.float64)

    in_amt: list[list[float]] = [[] for _ in range(n)]
    out_amt: list[list[float]] = [[] for _ in range(n)]
    in_age: list[list[float]] = [[] for _ in range(n)]
    out_age: list[list[float]] = [[] for _ in range(n)]
    in_ts: list[list[float]] = [[] for _ in range(n)]
    out_ts: list[list[float]] = [[] for _ in range(n)]

    for e, (u, v) in enumerate(graph.edges):
        att = edge_attrs[e]
        if att is None:
            continue
        amount = float(att["amount"])
        ts = float(att["timestamp"])
        iu, iv = pos[u], pos[v]
        in_amt[iv].append(amount)
        in_age[iv].append(step[iv] - step[iu])
        in_ts[iv].append(ts)
        out_amt[iu].append(amount)
        out_age[iu].append(step[iv] - step[iu])
        out_ts[iu].append(ts)

    S = np.zeros((n, _N_SEMANTIC), dtype=np.float64)
    for i in range(n):
        ins = in_amt[i]
        outs = out_amt[i]
        in_sum = sum(ins)
        out_sum = sum(outs)
        S[i, 0] = len(ins)
        S[i, 1] = len(outs)
        S[i, 2] = _log1p(out_sum)
        S[i, 3] = _log1p(in_sum)
        S[i, 4] = in_sum - out_sum
        S[i, 5] = _log1p(in_sum) - _log1p(out_sum)
        S[i, 6] = _log1p(max(ins)) if ins else 0.0
        S[i, 7] = _log1p(in_sum / len(ins)) if ins else 0.0
        S[i, 8] = _log1p(max(outs)) if outs else 0.0
        S[i, 9] = _log1p(out_sum / len(outs)) if outs else 0.0
        S[i, 10] = _log1p(min(ins)) if ins else 0.0
        ages_in = in_age[i]
        ages_out = out_age[i]
        S[i, 11] = min(ages_in) if ages_in else 0.0
        S[i, 12] = sum(ages_in) / len(ages_in) if ages_in else 0.0
        S[i, 13] = max(ages_in) if ages_in else 0.0
        S[i, 14] = sum(ages_out) / len(ages_out) if ages_out else 0.0
        S[i, 15] = in_sum / len(ins) if ins else 0.0
        S[i, 16] = out_sum / len(outs) if outs else 0.0
        S[i, 17] = _std(ins)
        S[i, 18] = _std(outs)
        S[i, 19] = (sum(in_ts[i]) / len(in_ts[i]) / 3600.0) if in_ts[i] else 0.0
        S[i, 20] = (sum(out_ts[i]) / len(out_ts[i]) / 3600.0) if out_ts[i] else 0.0
        S[i, 21] = 1.0 if ins else 0.0
        S[i, 22] = 1.0 if outs else 0.0
        S[i, 23] = step[i] / N_STEPS

    out = np.empty((n, N_FEATURES), dtype=np.float64)
    for j in range(_N_SEMANTIC):
        out[:, j] = S[:, j]
    for j in range(_N_SEMANTIC, N_FEATURES):
        out[:, j] = _derived(S, j)
    if hasattr(graph, "node_meta"):
        chain_by_tx = {meta.tx_id: meta.chain_id for meta in graph.node_meta}
        chain_features = np.zeros((n, len(CHAIN_SEMANTIC_COLUMNS)), dtype=np.float64)
        chain_index = {
            "bip122:000000000019d6689c085ae165831e93": 0,
            "eip155:1": 1,
            "eip155:728126428": 2,
        }
        for tx_id, chain_id in chain_by_tx.items():
            if chain_id in chain_index:
                chain_features[pos[tx_id], chain_index[chain_id]] = 1.0
        for u, v in graph.edges:
            if chain_by_tx[u] != chain_by_tx[v]:
                chain_features[pos[v], 3] += 1.0
                chain_features[pos[u], 4] += 1.0
        out[:, 24:29] = chain_features
    return out


def _std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    var = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    return _log1p(math.sqrt(var))
