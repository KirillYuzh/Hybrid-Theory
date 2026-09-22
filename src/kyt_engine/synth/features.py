from __future__ import annotations

import numpy as np

from .graph import GeneratedGraph
from .stats import CLASS_NAMES, N_FEATURES, EllipticStats


def build_features_matrix(
    rng: np.random.Generator, graph: GeneratedGraph, stats: EllipticStats
) -> np.ndarray:
    """Matrix [txId, time_step, feat_2..feat_166]; features sampled by node class."""
    n = len(graph.nodes)
    out = np.empty((n, 2 + N_FEATURES), dtype=float)
    for i, nd in enumerate(graph.nodes):
        out[i, 0] = nd.tx_id
        out[i, 1] = nd.step
    for cls in CLASS_NAMES:
        idx = [i for i, nd in enumerate(graph.nodes) if nd.cls == cls]
        if not idx:
            continue
        feats = stats.sample_features(rng, cls, len(idx))
        out[idx, 2:] = feats
    return out
