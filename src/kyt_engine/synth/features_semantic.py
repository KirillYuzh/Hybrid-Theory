from __future__ import annotations

import math

import numpy as np

from .graph import GeneratedGraph
from .stats import N_FEATURES, N_STEPS

# Fixed, documented layout of the interpretable feature columns (0-based index within
# the 165 feature columns, i.e. CSV column feat_{index + 2}). Value features are stored
# as log1p(dollars) to keep magnitudes ML-friendly; counts and time deltas stay raw.
SEMANTIC_COLUMNS: dict[int, str] = {
    0: "in_degree",
    1: "out_degree",
    2: "value_out",        # log1p(sum of outgoing amounts)
    3: "value_in",         # log1p(sum of incoming amounts)
    4: "net_flow",         # raw dollars: value_in - value_out
    5: "log_ratio_in_out", # log1p(value_in) - log1p(value_out)
    6: "max_in_amount",    # log1p
    7: "mean_in_amount",   # log1p
    8: "max_out_amount",   # log1p
    9: "mean_out_amount",  # log1p
    10: "min_in_amount",   # log1p
    11: "in_age_min",      # min step delta across in-neighbors
    12: "in_age_mean",
    13: "in_age_max",
    14: "out_age_mean",    # mean step delta across out-neighbors
    15: "value_per_in",    # raw dollars value_in / in_degree
    16: "value_per_out",   # raw dollars value_out / out_degree
    17: "in_amount_std",   # log1p(std of incoming amounts)
    18: "out_amount_std",  # log1p(std of outgoing amounts)
    19: "edge_hours_mean_in",   # mean edge timestamp (hours)
    20: "edge_hours_mean_out",
    21: "has_incoming",    # 0/1
    22: "has_outgoing",    # 0/1
    23: "time_step_norm",  # time_step / 49
}

_N_SEMANTIC = len(SEMANTIC_COLUMNS)
_N_DERIVED = N_FEATURES - _N_SEMANTIC


def _log1p(value: float) -> float:
    return math.log1p(max(value, 0.0))


def _derived(S: np.ndarray, col: int) -> np.ndarray:
    """Deterministic fill for the non-semantic columns (24..164): pairwise log transforms
    of the semantic vector. No RNG, no Elliptic marginals — byte-reproducible by construction."""
    k = col - _N_SEMANTIC
    a = k % _N_SEMANTIC
    b = (k // _N_SEMANTIC) % _N_SEMANTIC
    c = (k // (_N_SEMANTIC * _N_SEMANTIC)) % _N_SEMANTIC
    la = np.log1p(np.abs(S[:, a]) + 1e-9)
    lb = np.log1p(np.abs(S[:, b]) + 1e-9)
    return la * lb * (1.0 + 0.01 * np.abs(S[:, c]))


def build_semantic_features(
    graph: GeneratedGraph, edge_attrs: list[dict | None]
) -> np.ndarray:
    """Interpretable (n, 165) feature matrix computed from topology and edge attributes.

    Structural and amount RNG streams are untouched — everything here is a pure function
    of the graph + already-materialized edge attributes.
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
    return out


def _std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    var = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    return _log1p(math.sqrt(var))
