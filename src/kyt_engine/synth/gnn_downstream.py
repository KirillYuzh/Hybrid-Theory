import hashlib
import importlib.util
import json
import os
import platform
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

PARTITION_STEPS: dict[str, tuple[int, int]] = {
    "train": (1, 30),
    "validation": (31, 40),
    "test": (41, 49),
}
STRICT_SCHEMA_VERSION = "hybrid-theory.strict-downstream.v1"
MIN_ACCEPTANCE_SEEDS = 10
DEFAULT_SEEDS = tuple(range(10))
REAL_RAW_ROWS = 203_769
PINNED_REAL_RAW_FINGERPRINTS = {
    "elliptic_txs_features.csv": "fd7f83573443c9e302e371d3f110e3b6224160f5d1ed8a287757936127800ff0",
    "elliptic_txs_classes.csv": "93e2e7b2405c735ba752bf6ba06b947561deddd1f5a8fc91e46f6a4c0e439493",
    "elliptic_txs_edgelist.csv": "a35053ba68a98e4382cae2ba65b9d9e36b23b6439e02dff084971b1b72a5156e",
}
PINNED_VOLUME_SHA256 = "99d9eb866a2be6edab5de5490983b0fef027066bf32eda50b42da030d050b1dc"
PINNED_BEHAVIOR_CONFIG_SHA256 = "84298c76dbca964bd534a6e393e1e2da376b240daf922d04d83ff44358073cb7"
PINNED_FULL_VERSIONS = {
    "torch": "2.2.0",
    "torch-geometric": "2.5.0",
    "scikit-learn": "1.3.0",
}


class BackendUnavailable(RuntimeError):
    def __init__(self, message: str, status: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.status = status


def build_edge_index(edges: pd.DataFrame, node_count: int) -> np.ndarray:
    if edges.empty:
        return np.empty((2, 0), dtype=np.int64)
    values = edges[["txId1", "txId2"]].to_numpy(dtype=np.int64)
    if np.any(values < 0) or np.any(values >= node_count):
        raise ValueError("edge endpoint is outside local node range")
    return values.T


def _package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for package in ("torch", "torch-geometric", "scikit-learn", "numpy", "pandas"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def resolve_backend(requested: str) -> dict[str, Any]:
    """Resolve a downstream backend and verify optional PyG availability.

    Parameters
    ----------
    requested : {"auto", "full", "sgc"}
        Requested backend. ``auto`` may resolve to the diagnostic SGC proxy when
        the full backend is unavailable.

    Returns
    -------
    dict[str, Any]
        Resolved backend, availability flags, probe details, package versions, and
        pinned-environment status.

    Raises
    ------
    ValueError
        If ``requested`` is not a supported backend name.
    BackendUnavailable
        If ``full`` is requested but PyTorch, PyG, or the full probe is unusable.
    """
    if requested not in {"auto", "full", "sgc"}:
        raise ValueError("backend must be auto, full or sgc")
    probe_error = None
    try:
        has_torch = importlib.util.find_spec("torch") is not None
        has_pyg = importlib.util.find_spec("torch_geometric") is not None
    except (ImportError, ValueError) as exc:
        has_torch = False
        has_pyg = False
        probe_error = f"{type(exc).__name__}: {exc}"
    available = has_torch and has_pyg
    if requested != "sgc" and has_torch and has_pyg:
        try:
            _probe_pyg()
        except Exception as exc:
            available = False
            probe_error = f"{type(exc).__name__}: {exc}"
    resolved = (
        "sgc_proxy"
        if requested == "sgc"
        else (
            "pyg_sage_cpu" if available else ("unavailable" if requested == "full" else "sgc_proxy")
        )
    )
    package_versions = _package_versions()
    environment_pinned = bool(
        resolved == "pyg_sage_cpu"
        and all(
            package_versions.get(package) == version
            for package, version in PINNED_FULL_VERSIONS.items()
        )
    )
    status = {
        "requested": requested,
        "resolved": resolved,
        "is_full_gnn": resolved == "pyg_sage_cpu",
        "torch_available": has_torch,
        "pyg_available": has_pyg,
        "probe_error": probe_error,
        "device": "cpu",
        "package_versions": package_versions,
        "pinned_versions": PINNED_FULL_VERSIONS,
        "environment_pinned": environment_pinned,
    }
    if requested == "full" and not available:
        detail = f"; probe failed: {probe_error}" if probe_error else ""
        raise BackendUnavailable(f"full GNN requires torch and torch-geometric{detail}", status)
    return status


def partition_for_step(step: int) -> str:
    for name, (lo, hi) in PARTITION_STEPS.items():
        if lo <= step <= hi:
            return name
    raise ValueError(f"step is outside 1..49: {step}")


def build_partition_masks(time_step: np.ndarray) -> dict[str, np.ndarray]:
    values = np.asarray(time_step)
    if values.ndim != 1 or not np.issubdtype(values.dtype, np.integer):
        raise ValueError("time_step must be a one-dimensional integer array")
    if np.any(values < 1) or np.any(values > 49):
        raise ValueError("time_step is outside 1..49")
    masks = {"train": values <= 30}
    masks["validation"] = (values >= 31) & (values <= 40)
    masks["test"] = (values >= 41) & (values <= 49)
    return masks


def build_label_masks(
    classes: pd.DataFrame, time_step: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    if list(classes.columns) != ["txId", "class"]:
        raise ValueError("classes must have txId,class columns")
    if len(classes) != len(time_step):
        raise ValueError("classes and time_step length mismatch")
    values = classes["class"].astype(str).to_numpy()
    y = np.full(len(values), np.nan, dtype=float)
    y[values == "1"] = 1.0
    y[values == "2"] = 0.0
    if np.any(~np.isin(values, ["1", "2", "unknown"])):
        raise ValueError("unsupported class value")
    return y, np.isfinite(y)


def _edge_digest(edges: pd.DataFrame) -> str:
    payload = edges.to_csv(index=False, lineterminator="\n").encode()
    return hashlib.sha256(payload).hexdigest()


def filter_partition_edges(
    edgelist: pd.DataFrame, time_step: pd.Series
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    """Filter self-loops and cross-partition edges into temporal views.

    Parameters
    ----------
    edgelist : pandas.DataFrame
        Directed edges with integer ``txId1`` and ``txId2`` columns.
    time_step : pandas.Series
        Time steps indexed by transaction ID for every edge endpoint.

    Returns
    -------
    tuple[dict[str, pandas.DataFrame], dict[str, Any]]
        Edges grouped by train, validation, and test partition, plus exact input,
        removal, retention, and digest counts.

    Raises
    ------
    ValueError
        If edge columns, transaction references, or time-step values are invalid.
    """
    if list(edgelist.columns) != ["txId1", "txId2"]:
        raise ValueError("edgelist must have txId1,txId2 columns")
    if not isinstance(time_step.index, pd.Index):
        raise ValueError("time_step must be indexed by txId")
    ids = set(time_step.index.tolist())
    edges = edgelist.copy()
    edges["txId1"] = edges["txId1"].astype(int)
    edges["txId2"] = edges["txId2"].astype(int)
    if not edges["txId1"].isin(ids).all() or not edges["txId2"].isin(ids).all():
        raise ValueError("edgelist contains an orphan endpoint")
    source_step = edges["txId1"].map(time_step).astype(int)
    target_step = edges["txId2"].map(time_step).astype(int)
    source_partition = source_step.map(partition_for_step)
    target_partition = target_step.map(partition_for_step)
    cross = source_partition != target_partition
    loops = edges["txId1"] == edges["txId2"]
    retained = edges.loc[~cross & ~loops].copy()
    result: dict[str, pd.DataFrame] = {}
    retained_report: dict[str, Any] = {}
    for name in PARTITION_STEPS:
        selected = retained[
            retained["txId1"].map(time_step).map(partition_for_step).eq(name)
            & retained["txId2"].map(time_step).map(partition_for_step).eq(name)
        ].reset_index(drop=True)
        result[name] = selected
        retained_report[name] = {
            "input_edges": int(len(edges)),
            "retained_edges": int(len(selected)),
            "sha256": _edge_digest(selected),
        }
    final_edges = pd.concat(result.values(), ignore_index=True)
    cross_after = sum(
        partition_for_step(int(time_step.loc[u])) != partition_for_step(int(time_step.loc[v]))
        for u, v in final_edges[["txId1", "txId2"]].itertuples(index=False, name=None)
    )
    report = {
        "input_edges": int(len(edges)),
        "self_loops_dropped": int(loops.sum()),
        "cross_partition_dropped": int(cross.sum()),
        "retained_edges": int(len(retained)),
        "cross_partition_after_filter": int(cross_after),
        "partitions": retained_report,
    }
    return result, report


def choose_f1_threshold(y: np.ndarray, scores: np.ndarray) -> float:
    valid = np.isfinite(y)
    values = np.asarray(scores, dtype=float)[valid]
    labels = np.asarray(y, dtype=float)[valid]
    if len(labels) == 0 or len(np.unique(labels)) < 2:
        raise ValueError("threshold selection requires both labeled classes")
    candidates = np.linspace(0.0, 1.0, 101)
    f1_values = []
    for threshold in candidates:
        pred = values >= threshold
        tp = int(np.sum(pred & (labels == 1)))
        fp = int(np.sum(pred & (labels == 0)))
        fn = int(np.sum(~pred & (labels == 1)))
        denominator = 2 * tp + fp + fn
        f1_values.append(0.0 if denominator == 0 else 2 * tp / denominator)
    return float(candidates[int(np.argmax(f1_values))])


def binary_metrics_at_threshold(
    y: np.ndarray,
    scores: np.ndarray,
    tx_ids: np.ndarray,
    threshold: float,
    k: int = 100,
) -> dict[str, Any]:
    """Compute binary and top-k metrics at a fixed probability threshold.

    Parameters
    ----------
    y : numpy.ndarray
        Binary labels where ``1`` is positive, ``0`` is negative, and non-finite
        values denote unlabeled rows.
    scores : numpy.ndarray
        Finite positive-class probabilities aligned with ``y``.
    tx_ids : numpy.ndarray
        Transaction IDs used as the deterministic top-k tie breaker.
    threshold : float
        Probability threshold in the inclusive range 0..1.
    k : int, default 100
        Positive maximum rank for precision and recall.

    Returns
    -------
    dict[str, Any]
        Counts, class rates, confusion-matrix values, F1, and top-k metrics. Metrics
        that are undefined because the positive class is absent are returned as ``None``.

    Raises
    ------
    ValueError
        If array shapes or lengths differ, values are invalid, or ``k`` is invalid.
    """
    if not isinstance(k, int) or isinstance(k, bool) or k <= 0:
        raise ValueError("k must be a positive integer")
    labels = np.asarray(y, dtype=float)
    probabilities = np.asarray(scores, dtype=float)
    ids = np.asarray(tx_ids)
    if labels.ndim != 1 or probabilities.ndim != 1 or ids.ndim != 1:
        raise ValueError("metric inputs must be one-dimensional")
    if not (len(labels) == len(probabilities) == len(ids)):
        raise ValueError("metric input length mismatch")
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("threshold must be finite and in 0..1")
    if not np.isfinite(probabilities).all():
        raise ValueError("scores must be finite")
    valid = np.isfinite(labels)
    labels = labels[valid]
    probabilities = probabilities[valid]
    ids = ids[valid]
    n = len(labels)
    positives = int(np.sum(labels == 1))
    negatives = int(np.sum(labels == 0))
    if n == 0:
        return {"n": 0, "n_positive": 0, "f1": None, "precision_at_k": None, "recall_at_k": None}
    pred = probabilities >= threshold
    tp = int(np.sum(pred & (labels == 1)))
    fp = int(np.sum(pred & (labels == 0)))
    fn = int(np.sum(~pred & (labels == 1)))
    tn = int(np.sum(~pred & (labels == 0)))
    k_effective = min(k, n)
    order = np.lexsort((ids, -probabilities))[:k_effective]
    top_tp = int(np.sum(labels[order] == 1))
    precision_at_k = top_tp / k_effective if k_effective else None
    recall_at_k = top_tp / positives if positives else None
    f1 = None if positives == 0 or 2 * tp + fp + fn == 0 else 2 * tp / (2 * tp + fp + fn)
    return {
        "n": n,
        "n_positive": positives,
        "n_negative": negatives,
        "positive_rate": positives / n,
        "threshold": float(threshold),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "k": k,
        "k_effective": k_effective,
        "topk_tp": top_tp,
        "f1": f1,
        "precision_at_k": precision_at_k,
        "recall_at_k": recall_at_k,
    }


def per_step_metrics(
    y: np.ndarray,
    scores: np.ndarray,
    tx_ids: np.ndarray,
    time_step: np.ndarray,
    threshold: float,
    k: int = 100,
    steps: list[int] | None = None,
) -> list[dict[str, Any]]:
    rows = []
    for step in range(41, 50) if steps is None else steps:
        mask = np.asarray(time_step) == step
        row = binary_metrics_at_threshold(
            y[mask], scores[mask], np.asarray(tx_ids)[mask], threshold, k
        )
        row["time_step"] = step
        rows.append(row)
    return rows


def aggregate_step_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    f1_values = [row["f1"] for row in rows if row["f1"] is not None]
    precision = [row["precision_at_k"] for row in rows if row["precision_at_k"] is not None]
    recall = [row["recall_at_k"] for row in rows if row["recall_at_k"] is not None]
    expected = len(rows)
    return {
        "macro_f1": (
            float(np.mean(f1_values)) if expected and len(f1_values) == expected else None
        ),
        "macro_precision_at_k": (
            float(np.mean(precision)) if expected and len(precision) == expected else None
        ),
        "macro_recall_at_k": (
            float(np.mean(recall)) if expected and len(recall) == expected else None
        ),
        "steps_evaluated": len(f1_values),
        "precision_steps_evaluated": len(precision),
        "recall_steps_evaluated": len(recall),
    }


def _micro_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    usable = [row for row in rows if row.get("n", 0) > 0]
    if not usable:
        return {
            "n": 0,
            "f1": None,
            "precision_at_k": None,
            "recall_at_k": None,
        }
    tp = sum(int(row["tp"]) for row in usable)
    fp = sum(int(row["fp"]) for row in usable)
    fn = sum(int(row["fn"]) for row in usable)
    positives = sum(int(row["n_positive"]) for row in usable)
    topk_tp = sum(int(row["topk_tp"]) for row in usable)
    k_total = sum(int(row["k_effective"]) for row in usable)
    return {
        "n": sum(int(row["n"]) for row in usable),
        "n_positive": positives,
        "f1": (None if positives == 0 or 2 * tp + fp + fn == 0 else 2 * tp / (2 * tp + fp + fn)),
        "precision_at_k": None if k_total == 0 else topk_tp / k_total,
        "recall_at_k": None if positives == 0 else topk_tp / positives,
    }


def welch_secondary(a: list[float], b: list[float]) -> dict[str, Any]:
    x = np.asarray(a, dtype=float)
    y = np.asarray(b, dtype=float)
    out: dict[str, Any] = {"n_a": len(x), "n_b": len(y)}
    if len(x) < 2 or len(y) < 2 or (len(x) == len(y) and np.allclose(x, y)):
        out.update({"p": None, "cohens_d": None, "significant": False})
        return out
    result = stats.ttest_ind(x, y, equal_var=False)
    pooled = np.sqrt(
        ((len(x) - 1) * x.var(ddof=1) + (len(y) - 1) * y.var(ddof=1)) / (len(x) + len(y) - 2)
    )
    out.update(
        {
            "p": float(result.pvalue),
            "cohens_d": None if pooled == 0 else float((x.mean() - y.mean()) / pooled),
            "significant": bool(result.pvalue < 0.05),
        }
    )
    return out


class PyGClassifier:
    def __init__(self, model: Any, mean: Any, std: Any) -> None:
        self.model = model
        self.mean = mean
        self.std = std

    def predict_proba(self, features: np.ndarray, edge_index: np.ndarray) -> np.ndarray:
        import torch

        self.model.eval()
        with torch.no_grad():
            x = (torch.as_tensor(features, dtype=torch.float32) - self.mean) / self.std
            edge = torch.as_tensor(edge_index, dtype=torch.long)
            logits = self.model(x, edge)
            return torch.sigmoid(logits).cpu().numpy()


def fit_sgc_proxy(
    features: np.ndarray,
    labels: np.ndarray,
    edges: pd.DataFrame,
    node_count: int,
    train_mask: np.ndarray,
    seed: int,
    hops: int = 2,
) -> Any:
    """Fit the SGC proxy on concatenated propagated structural features.

    Parameters
    ----------
    features : numpy.ndarray
        Node-by-feature matrix for the source graph.
    labels : numpy.ndarray
        Binary node labels; unlabeled values are permitted outside ``train_mask``.
    edges : pandas.DataFrame
        Directed local-index edges used to construct row-normalized propagation.
    node_count : int
        Number of nodes in the local graph.
    train_mask : numpy.ndarray
        Boolean mask selecting labeled training rows.
    seed : int
        Random seed passed to logistic regression.
    hops : int, default 2
        Number of propagation steps concatenated with the raw features.

    Returns
    -------
    sklearn.linear_model.LogisticRegression
        Fitted balanced logistic-regression model with the hop count attached for
        use by the matching prediction helper.
    """
    from scipy import sparse
    from sklearn.linear_model import LogisticRegression

    edge_values = edges[["txId1", "txId2"]].to_numpy(dtype=np.int64)
    if len(edge_values):
        rows = edge_values[:, 0]
        cols = edge_values[:, 1]
    else:
        rows = np.empty(0, dtype=np.int64)
        cols = np.empty(0, dtype=np.int64)
    adjacency = sparse.csr_matrix(
        (np.ones(len(rows)), (rows, cols)), shape=(node_count, node_count)
    )
    degree = np.asarray(adjacency.sum(axis=1)).ravel()
    inverse = np.zeros_like(degree)
    np.divide(1.0, degree, out=inverse, where=degree > 0)
    normalized = sparse.diags(inverse) @ adjacency
    blocks = [features]
    propagated = features
    for _ in range(hops):
        propagated = normalized @ propagated
        blocks.append(propagated)
    design = np.hstack(blocks)
    model = LogisticRegression(C=1.0, max_iter=300, class_weight="balanced", random_state=seed)
    model._kyt_hops = hops
    return model.fit(design[train_mask], labels[train_mask])


def predict_sgc_proxy(
    model: Any, features: np.ndarray, edges: pd.DataFrame, node_count: int
) -> np.ndarray:
    from scipy import sparse

    values = edges[["txId1", "txId2"]].to_numpy(dtype=np.int64)
    if len(values):
        rows = values[:, 0]
        cols = values[:, 1]
    else:
        rows = np.empty(0, dtype=np.int64)
        cols = np.empty(0, dtype=np.int64)
    adjacency = sparse.csr_matrix(
        (np.ones(len(rows)), (rows, cols)), shape=(node_count, node_count)
    )
    degree = np.asarray(adjacency.sum(axis=1)).ravel()
    inverse = np.zeros_like(degree)
    np.divide(1.0, degree, out=inverse, where=degree > 0)
    normalized = sparse.diags(inverse) @ adjacency
    blocks = [features]
    propagated = features
    for _ in range(getattr(model, "_kyt_hops", 2)):
        propagated = normalized @ propagated
        blocks.append(propagated)
    return model.predict_proba(np.hstack(blocks))[:, 1]


def _build_pyg_network(kind: str, input_dim: int, hidden: int, layers: int, dropout: float) -> Any:
    """Build a directed PyG network for one full-backend model family.

    GraphSAGE processes original and reversed edge indices in separate branches;
    GAT and GIN use the original edge direction.

    Parameters
    ----------
    kind : {"sage", "gat", "gin"}
        PyG convolution family to construct.
    input_dim : int
        Number of input features per node.
    hidden : int
        Hidden representation width.
    layers : int
        Number of graph-convolution layers.
    dropout : float
        Dropout probability applied between layers.

    Returns
    -------
    torch.nn.Module
        Untrained network that maps node features and edge indices to one logit per node.

    Raises
    ------
    ValueError
        If ``kind`` is unsupported or a GAT hidden width is odd.
    """
    import torch
    from torch import nn
    from torch_geometric.nn import GATConv, GINConv, SAGEConv

    if kind == "gat" and hidden % 2:
        raise ValueError("GAT hidden dimension must be even")

    class DirectedNetwork(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.dropout = nn.Dropout(dropout)
            self.kind = kind
            self.out_convolutions = nn.ModuleList()
            self.in_convolutions = nn.ModuleList()
            self.convolutions = nn.ModuleList()
            current = input_dim
            for _ in range(layers):
                if kind == "sage":
                    self.out_convolutions.append(SAGEConv(current, hidden))
                    self.in_convolutions.append(SAGEConv(current, hidden))
                    current = hidden * 2
                elif kind == "gat":
                    self.convolutions.append(
                        GATConv(current, max(1, hidden // 2), heads=2, dropout=dropout)
                    )
                    current = hidden
                elif kind == "gin":
                    mlp = nn.Sequential(
                        nn.Linear(current, hidden), nn.ReLU(), nn.Linear(hidden, hidden)
                    )
                    self.convolutions.append(GINConv(mlp, train_eps=True))
                    current = hidden
                else:
                    raise ValueError(f"unknown PyG model kind: {kind}")
            self.head = nn.Linear(current, 1)

        def forward(self, x: Any, edge_index: Any) -> Any:
            if self.kind == "sage":
                reverse_edge_index = edge_index.flip(0)
                for out_convolution, in_convolution in zip(
                    self.out_convolutions, self.in_convolutions
                ):
                    out_features = torch.relu(out_convolution(x, edge_index))
                    in_features = torch.relu(in_convolution(x, reverse_edge_index))
                    x = self.dropout(torch.cat([out_features, in_features], dim=-1))
            else:
                for convolution in self.convolutions:
                    x = torch.relu(convolution(x, edge_index))
                    x = self.dropout(x)
            return self.head(x).squeeze(-1)

    return DirectedNetwork()


def _probe_pyg() -> None:
    import torch

    features = torch.randn((3, 2), requires_grad=True)
    edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    for kind in ("sage", "gat", "gin"):
        model = _build_pyg_network(kind, 2, 4, 2, 0.0)
        model.zero_grad(set_to_none=True)
        output = model(features, edge_index).sum()
        output.backward()


def _binary_f1(labels: np.ndarray, predictions: np.ndarray) -> float:
    tp = int(np.sum(predictions & (labels == 1)))
    fp = int(np.sum(predictions & (labels == 0)))
    fn = int(np.sum(~predictions & (labels == 1)))
    return 0.0 if 2 * tp + fp + fn == 0 else 2 * tp / (2 * tp + fp + fn)


def fit_pyg_classifier(
    features: np.ndarray,
    labels: np.ndarray,
    edges: pd.DataFrame,
    node_count: int,
    train_mask: np.ndarray,
    validation_mask: np.ndarray | None,
    kind: str = "sage",
    hidden: int = 64,
    layers: int = 2,
    dropout: float = 0.25,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    epochs: int = 300,
    patience: int = 30,
    threads: int = 1,
    seed: int = 0,
) -> PyGClassifier:
    """Fit a full PyG classifier with train-only normalization and early stopping.

    Parameters
    ----------
    features : numpy.ndarray
        Node-by-feature matrix for the combined training and validation graph.
    labels : numpy.ndarray
        Binary node labels with non-finite values treated as unlabeled.
    edges : pandas.DataFrame
        Directed local-index edges for the combined graph.
    node_count : int
        Number of nodes in the combined graph.
    train_mask : numpy.ndarray
        Boolean mask selecting rows used for loss and normalization statistics.
    validation_mask : numpy.ndarray or None, optional
        Boolean mask used for early stopping. If omitted, the training mask is reused.
    kind : {"sage", "gat", "gin"}, default "sage"
        PyG model family to train.
    hidden : int, default 64
        Hidden representation width.
    layers : int, default 2
        Number of graph-convolution layers.
    dropout : float, default 0.25
        Dropout probability applied between layers.
    learning_rate : float, default 1e-3
        AdamW learning rate.
    weight_decay : float, default 1e-4
        AdamW weight decay.
    epochs : int, default 300
        Maximum number of training epochs.
    patience : int, default 30
        Number of non-improving validation evaluations allowed before stopping.
    threads : int, default 1
        Number of PyTorch CPU threads.
    seed : int, default 0
        Seed applied to PyTorch and NumPy before model construction.

    Returns
    -------
    PyGClassifier
        Best validation-F1 model state together with train-only feature mean and scale.

    Raises
    ------
    ValueError
        If a required class is absent, validation labels are single-class, or model
        parameters are invalid.
    BackendUnavailable
        If the required full PyG backend cannot be resolved.
    """
    resolve_backend("full")
    import torch

    if len(np.unique(labels[train_mask])) < 2:
        raise ValueError("PyG training requires both labeled classes")
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(threads)
    model = _build_pyg_network(kind, features.shape[1], hidden, layers, dropout)
    x = torch.as_tensor(features, dtype=torch.float32)
    edge_index = torch.as_tensor(build_edge_index(edges, node_count), dtype=torch.long)
    y = torch.as_tensor(labels, dtype=torch.float32)
    evaluation_mask = train_mask if validation_mask is None else validation_mask
    if validation_mask is not None and len(np.unique(labels[validation_mask])) < 2:
        raise ValueError("VALIDATION_SINGLE_CLASS")
    mean = x[train_mask].mean(dim=0)
    std = x[train_mask].std(dim=0).clamp_min(1e-6)
    x = (x - mean) / std
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    loss_fn = torch.nn.BCEWithLogitsLoss(reduction="none")
    best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    best_score = -1.0
    stale = 0
    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(x, edge_index)
        loss = loss_fn(logits, y)[train_mask].mean()
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            validation_scores = torch.sigmoid(model(x, edge_index))[evaluation_mask].numpy()
        validation_labels = labels[evaluation_mask]
        score = _binary_f1(validation_labels, validation_scores >= 0.5)
        if score > best_score:
            best_score = score
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    model.load_state_dict(best_state)
    return PyGClassifier(model, mean, std)


def degree_preserving_edge_shuffle(
    edges: pd.DataFrame, seed: int, max_attempts_per_edge: int = 20
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Shuffle directed targets while preserving endpoint degree multisets.

    Parameters
    ----------
    edges : pandas.DataFrame
        Directed edges with ``txId1`` and ``txId2`` columns.
    seed : int
        Seed for the NumPy swap generator.
    max_attempts_per_edge : int, default 20
        Upper bound on swap attempts, scaled by the number of edges.

    Returns
    -------
    tuple[pandas.DataFrame, dict[str, Any]]
        Rewired edge table and a report containing seed, attempts, swaps, changed
        fraction, and whether shuffling was testable.

    Raises
    ------
    ValueError
        If the edge table does not have the required columns.
    """
    if list(edges.columns) != ["txId1", "txId2"]:
        raise ValueError("edges must have txId1,txId2 columns")
    values = [tuple(map(int, row)) for row in edges.itertuples(index=False, name=None)]
    if len(values) < 2:
        return edges.copy().reset_index(drop=True), {
            "seed": seed,
            "attempts": 0,
            "swaps": 0,
            "changed_fraction": 0.0,
            "status": "not_testable",
        }
    rng = np.random.default_rng(seed)
    current = values.copy()
    pair_counts = Counter(current)
    swaps = 0
    attempts = 0
    max_attempts = max(1, len(current) * max_attempts_per_edge)
    while attempts < max_attempts and swaps < len(current):
        attempts += 1
        first, second = rng.choice(len(current), size=2, replace=False)
        first = int(first)
        second = int(second)
        u1, v1 = current[first]
        u2, v2 = current[second]
        new_edges = [(u1, v2), (u2, v1)]
        if any(u == v for u, v in new_edges):
            continue
        if new_edges[0] == new_edges[1]:
            continue
        remaining = pair_counts.copy()
        remaining[current[first]] -= 1
        remaining[current[second]] -= 1
        if remaining[current[first]] == 0:
            del remaining[current[first]]
        if current[second] in remaining:
            remaining[current[second]] -= 1
            if remaining[current[second]] == 0:
                del remaining[current[second]]
        if any(pair in remaining for pair in new_edges):
            continue
        current[first], current[second] = new_edges[0], new_edges[1]
        pair_counts = remaining
        pair_counts.update(new_edges)
        swaps += 1
    shuffled = pd.DataFrame(current, columns=["txId1", "txId2"])
    changed = sum(before != after for before, after in zip(values, current))
    return shuffled, {
        "seed": seed,
        "attempts": attempts,
        "swaps": swaps,
        "changed_fraction": changed / len(current),
        "status": "ok" if changed else "not_testable",
    }


def paired_effect(deltas: list[float]) -> dict[str, Any]:
    values = np.asarray(deltas, dtype=float)
    mean_delta = None if len(values) == 0 else float(np.mean(values))
    if len(values) < 2 or np.isclose(float(np.std(values, ddof=1)), 0.0):
        return {
            "n": len(values),
            "mean_delta": mean_delta,
            "p": None,
            "cohens_dz": None,
            "status": "not_testable",
        }
    result = stats.ttest_1samp(values, 0.0)
    return {
        "n": len(values),
        "mean_delta": float(values.mean()),
        "p": float(result.pvalue),
        "cohens_dz": float(values.mean() / values.std(ddof=1)),
        "status": "ok",
    }


def _integer_values(series: pd.Series, code: str) -> np.ndarray:
    if len(series) == 0:
        return np.empty(0, dtype=np.int64)
    if not np.issubdtype(series.dtype, np.integer):
        raise ValueError(code)
    values = series.to_numpy()
    if values.size == 0 or np.any(values < 0) or np.any(values > np.iinfo(np.int64).max):
        raise ValueError(code)
    return values.astype(np.int64)


def _read_csv_stable(path: Path, error_code: str, **kwargs: Any) -> pd.DataFrame:
    try:
        return pd.read_csv(path, **kwargs)
    except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeError) as exc:
        raise ValueError(error_code) from exc


def load_csv_dataset(
    root: Path, structural_only: bool = False, require_contiguous: bool = False
) -> dict[str, Any]:
    features_path = root / "elliptic_txs_features.csv"
    classes_path = root / "elliptic_txs_classes.csv"
    edges_path = root / "elliptic_txs_edgelist.csv"
    if not features_path.exists() or not classes_path.exists() or not edges_path.exists():
        raise ValueError("RAW_FILES_MISSING")
    if structural_only:
        feature_chunks = []
        try:
            reader = pd.read_csv(features_path, header=None, chunksize=8192)
            for chunk in reader:
                if chunk.shape[1] != 167:
                    raise ValueError("RAW_FEATURE_WIDTH")
                try:
                    values = chunk.iloc[:, 2:].to_numpy(dtype=float)
                except (TypeError, ValueError) as exc:
                    raise ValueError("RAW_NONFINITE_FEATURE") from exc
                if not np.isfinite(values).all():
                    raise ValueError("RAW_NONFINITE_FEATURE")
                feature_chunks.append(chunk.iloc[:, :2].copy())
        except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeError) as exc:
            raise ValueError("RAW_FEATURE_SCHEMA") from exc
        if not feature_chunks:
            raise ValueError("RAW_EMPTY_FEATURE")
        features = pd.concat(feature_chunks, ignore_index=True)
        features.columns = ["txId", "time_step"]
    else:
        features = _read_csv_stable(features_path, "RAW_FEATURE_SCHEMA", header=None)
        if features.shape[1] != 167:
            raise ValueError("RAW_FEATURE_WIDTH")
        features.columns = [f"feature_{index}" for index in range(features.shape[1])]
        features = features.rename(columns={"feature_0": "txId", "feature_1": "time_step"})
    features["txId"] = _integer_values(features["txId"], "RAW_ID_TYPE")
    features["time_step"] = _integer_values(features["time_step"], "RAW_TIME_STEP")
    features = features.set_index("txId")
    classes_raw = _read_csv_stable(classes_path, "RAW_CLASS_SCHEMA")
    if list(classes_raw.columns) != ["txId", "class"]:
        raise ValueError("RAW_CLASS_SCHEMA")
    classes_raw["txId"] = _integer_values(classes_raw["txId"], "RAW_ID_TYPE")
    classes = classes_raw.set_index("txId")
    edges = _read_csv_stable(edges_path, "RAW_EDGE_SCHEMA")
    if list(edges.columns) != ["txId1", "txId2"]:
        raise ValueError("RAW_EDGE_SCHEMA")
    edges["txId1"] = _integer_values(edges["txId1"], "RAW_EDGE_TYPE")
    edges["txId2"] = _integer_values(edges["txId2"], "RAW_EDGE_TYPE")
    if features.index.has_duplicates or classes.index.has_duplicates:
        raise ValueError("RAW_ID_DUPLICATE")
    if require_contiguous and not np.array_equal(
        features.index.to_numpy(), np.arange(len(features))
    ):
        raise ValueError("RAW_ID_SEQUENCE")
    if not features.index.equals(classes.index):
        raise ValueError("RAW_ID_ORDER")
    if (edges[["txId1", "txId2"]] < 0).any().any() or (
        require_contiguous and (edges[["txId1", "txId2"]] >= len(features)).any().any()
    ):
        raise ValueError("RAW_EDGE_RANGE")
    if (
        not edges["txId1"].isin(features.index).all()
        or not edges["txId2"].isin(features.index).all()
    ):
        raise ValueError("RAW_ORPHAN_EDGE")
    return {"features": features, "classes": classes, "edges": edges}


def make_partition_views(
    dataset: dict[str, Any],
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Build local-index temporal partition views and structural features.

    Parameters
    ----------
    dataset : dict[str, Any]
        Dataset mapping with indexed ``features``, ``classes``, and ``edges`` tables
        as returned by :func:`load_csv_dataset`.

    Returns
    -------
    tuple[dict[str, dict[str, Any]], dict[str, Any]]
        Train, validation, and test views plus the edge-filter report. Each view
        contains original IDs, labels, remapped local edges, and recomputed features.

    Raises
    ------
    ValueError
        If temporal masks, edge filtering, or label mapping reject the dataset.
    """
    features = dataset["features"]
    classes = dataset["classes"]
    time_step = features["time_step"].astype("int64")
    retained, edge_report = filter_partition_edges(dataset["edges"], time_step)
    masks = build_partition_masks(time_step.to_numpy())
    y, _ = build_label_masks(classes.reset_index(), time_step.to_numpy())
    views: dict[str, dict[str, Any]] = {}
    for name, mask in masks.items():
        ids = list(time_step.index[mask])
        local = {tx_id: index for index, tx_id in enumerate(ids)}
        edges = retained[name].copy()
        edges["txId1"] = edges["txId1"].map(local)
        edges["txId2"] = edges["txId2"].map(local)
        from .features_structural import structural_features

        structural = structural_features(
            edges,
            pd.Series(
                time_step.loc[ids].to_numpy(),
                index=np.arange(len(ids), dtype=np.int64),
            ),
        )
        views[name] = {
            "ids": np.asarray(ids, dtype=np.int64),
            "time_step": time_step.loc[ids].to_numpy(dtype=np.int64),
            "X": structural.to_numpy(dtype=float),
            "y": y[mask],
            "edges": edges[["txId1", "txId2"]].astype(int).reset_index(drop=True),
            "labeled": np.isfinite(y[mask]),
        }
    return views, edge_report


def _fit_rf_arm(train: dict[str, Any], seed: int) -> Any:
    from sklearn.ensemble import RandomForestClassifier

    if len(np.unique(train["y"][train["labeled"]])) < 2:
        raise ValueError("RF training requires both source classes")
    model = RandomForestClassifier(
        n_estimators=100, random_state=seed, class_weight="balanced", n_jobs=1
    )
    model.fit(train["X"][train["labeled"]], train["y"][train["labeled"]])
    return model


def _rf_scores(model: Any, view: dict[str, Any]) -> np.ndarray:
    return model.predict_proba(view["X"])[:, 1]


def _sgc_arm(train: dict[str, Any], seed: int) -> Any:
    return fit_sgc_proxy(
        train["X"], train["y"], train["edges"], len(train["ids"]), train["labeled"], seed
    )


def _sgc_scores(model: Any, view: dict[str, Any]) -> np.ndarray:
    return predict_sgc_proxy(model, view["X"], view["edges"], len(view["ids"]))


def validate_seed_plan(seeds: list[int], tier: str) -> dict[str, Any]:
    if tier not in {"smoke", "real"}:
        raise ValueError("tier must be smoke or real")
    if len(set(seeds)) != len(seeds):
        raise ValueError("DUPLICATE_SEED")
    if not seeds:
        raise ValueError("EMPTY_SEED_ROSTER")
    if any(isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 for seed in seeds):
        raise ValueError("SEED_VALUE")
    if tier == "real" and len(seeds) < MIN_ACCEPTANCE_SEEDS:
        raise ValueError("N_GATE")
    return {
        "values": list(seeds),
        "n_seeds": len(seeds),
        "minimum": MIN_ACCEPTANCE_SEEDS,
        "acceptance_eligible": tier == "real" and len(seeds) >= MIN_ACCEPTANCE_SEEDS,
    }


def _source_threshold(labels: np.ndarray, scores: np.ndarray) -> float:
    valid = np.isfinite(labels)
    if len(np.unique(np.asarray(labels)[valid])) < 2:
        raise ValueError("VALIDATION_SINGLE_CLASS")
    return choose_f1_threshold(labels, scores)


def _view_metrics(
    view: dict[str, Any], scores: np.ndarray, threshold: float, k: int
) -> dict[str, Any]:
    steps = sorted(set(int(value) for value in view["time_step"]))
    observed = set(steps)
    if observed and min(steps) >= 41:
        metric_steps = list(range(41, 50))
    elif observed and min(steps) >= 31:
        metric_steps = list(range(31, 41))
    elif observed and max(steps) <= 30:
        metric_steps = list(range(1, 31))
    else:
        metric_steps = steps
    rows = per_step_metrics(
        view["y"],
        scores,
        view["ids"],
        view["time_step"],
        threshold,
        k,
        steps=metric_steps,
    )
    return {
        "aggregate": aggregate_step_metrics(rows),
        "micro": _micro_metrics(rows),
        "per_timestep": rows,
    }


def _combine_train_validation(
    train: dict[str, Any], validation: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, np.ndarray, np.ndarray]:
    offset = len(train["ids"])
    train_edges = train["edges"].copy()
    validation_edges = validation["edges"].copy()
    validation_edges["txId1"] += offset
    validation_edges["txId2"] += offset
    edges = pd.concat([train_edges, validation_edges], ignore_index=True)
    features = np.vstack([train["X"], validation["X"]])
    labels = np.concatenate([train["y"], validation["y"]])
    train_mask = np.concatenate([train["labeled"], np.zeros(len(validation["ids"]), dtype=bool)])
    validation_mask = np.concatenate(
        [np.zeros(len(train["ids"]), dtype=bool), validation["labeled"]]
    )
    return features, labels, edges, train_mask, validation_mask


def _fit_full_arm(
    train: dict[str, Any],
    validation: dict[str, Any],
    seed: int,
    kind: str,
    hidden: int,
    layers: int,
    dropout: float,
    learning_rate: float,
    weight_decay: float,
    epochs: int,
    patience: int,
    threads: int,
) -> tuple[Any, float]:
    features, labels, edges, train_mask, validation_mask = _combine_train_validation(
        train, validation
    )
    model = fit_pyg_classifier(
        features,
        labels,
        edges,
        len(train["ids"]) + len(validation["ids"]),
        train_mask,
        validation_mask,
        kind=kind,
        hidden=hidden,
        layers=layers,
        dropout=dropout,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        epochs=epochs,
        patience=patience,
        threads=threads,
        seed=seed,
    )
    scores = model.predict_proba(
        validation["X"], build_edge_index(validation["edges"], len(validation["ids"]))
    )
    threshold = _source_threshold(
        validation["y"][validation["labeled"]], scores[validation["labeled"]]
    )
    return model, threshold


def _rebuild_view(
    view: dict[str, Any], edges: pd.DataFrame, scaler: Any | None = None
) -> dict[str, Any]:
    from .features_structural import structural_features

    time_series = pd.Series(view["time_step"], index=np.arange(len(view["ids"]), dtype=np.int64))
    features = structural_features(edges, time_series).to_numpy(dtype=float)
    if scaler is not None:
        features = scaler.transform(features)
    return {**view, "X": features, "edges": edges.reset_index(drop=True)}


def run_strict_protocol(
    synthetic_dirs: list[Path],
    raw_dir: Path,
    seeds: list[int],
    raw_dataset: dict[str, Any] | None = None,
    backend_status: dict[str, Any] | None = None,
    acceptance_context: bool = False,
    backend: str = "auto",
    tier: str = "smoke",
    k: int = 100,
    gnn_kinds: tuple[str, ...] = ("sage", "gat", "gin"),
    hidden: int = 64,
    layers: int = 2,
    dropout: float = 0.25,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    epochs: int = 300,
    patience: int = 30,
    threads: int = 1,
) -> dict[str, Any]:
    """Run the strict temporal transfer protocol and evaluate acceptance gates.

    Source and target graphs are split into disjoint train, validation, and test
    temporal views. Each source scaler is fit on source training rows only, unknown
    labels are excluded, and raw targets are never used to fit transfer models.

    Parameters
    ----------
    synthetic_dirs : list[pathlib.Path]
        Generated source datasets containing contiguous local transaction IDs.
    raw_dir : pathlib.Path
        Target Elliptic directory, used only when ``raw_dataset`` is not provided.
    seeds : list[int]
        Unique model seeds paired positionally with ``synthetic_dirs``.
    raw_dataset : dict[str, Any] or None, optional
        Prevalidated target dataset that avoids rereading ``raw_dir``.
    backend_status : dict[str, Any] or None, optional
        Pre-resolved backend status. If omitted, ``backend`` is resolved here.
    acceptance_context : bool, default False
        Assertion from the caller that the pinned raw and protocol context was
        independently verified. This function records but does not establish it.
    backend : {"auto", "full", "sgc"}, default "auto"
        Backend policy used when ``backend_status`` is omitted.
    tier : {"smoke", "real"}, default "smoke"
        Benchmark tier used for seed-plan and acceptance rules.
    k : int, default 100
        Positive rank depth for precision and recall metrics.
    gnn_kinds : tuple[str, ...], default ("sage", "gat", "gin")
        Optional PyG model roster. Real acceptance requires all three.
    hidden : int, default 64
        Hidden width for full PyG models.
    layers : int, default 2
        Number of graph-convolution layers.
    dropout : float, default 0.25
        Dropout probability for full PyG models.
    learning_rate : float, default 1e-3
        AdamW learning rate for full PyG models.
    weight_decay : float, default 1e-4
        AdamW weight decay for full PyG models.
    epochs : int, default 300
        Maximum epochs per full PyG fit.
    patience : int, default 30
        Full PyG early-stopping patience.
    threads : int, default 1
        Number of PyTorch CPU threads used by full PyG models.

    Returns
    -------
    dict[str, Any]
        Protocol policy, per-seed internal and transfer results, edge-shuffle
        sensitivity, aggregate metrics, and explicit acceptance-gate status.

    Raises
    ------
    ValueError
        If seed, model, metric, or edge-partition contracts are violated.
    BackendUnavailable
        If a requested full backend is required but cannot be resolved.
    """
    seed_plan = validate_seed_plan(seeds, tier)
    if not isinstance(k, int) or isinstance(k, bool) or k <= 0:
        raise ValueError("K_VALUE")
    required_roster = ("sage", "gat", "gin")
    if (
        len(set(gnn_kinds)) != len(gnn_kinds)
        or not set(gnn_kinds) <= set(required_roster)
        or not gnn_kinds
    ):
        raise ValueError("GNN_ROSTER")
    if len(synthetic_dirs) != len(seeds):
        raise ValueError("synthetic directory and seed roster length mismatch")
    backend_status = backend_status or resolve_backend(backend)
    target_dataset = (
        raw_dataset if raw_dataset is not None else load_csv_dataset(raw_dir, structural_only=True)
    )
    target_base, target_edge_report = make_partition_views(target_dataset)
    oracle_views = {name: {**view, "X": view["X"].copy()} for name, view in target_base.items()}
    from sklearn.preprocessing import StandardScaler

    oracle_scaler = StandardScaler().fit(oracle_views["train"]["X"])
    for view in oracle_views.values():
        view["X"] = oracle_scaler.transform(view["X"])
    oracle_model = _fit_rf_arm(oracle_views["train"], 0)
    oracle_validation_scores = _rf_scores(oracle_model, oracle_views["validation"])
    oracle_threshold = _source_threshold(
        oracle_views["validation"]["y"][oracle_views["validation"]["labeled"]],
        oracle_validation_scores[oracle_views["validation"]["labeled"]],
    )
    oracle_test_scores = _rf_scores(oracle_model, oracle_views["test"])
    oracle_test = _view_metrics(oracle_views["test"], oracle_test_scores, oracle_threshold, k)
    per_seed: list[dict[str, Any]] = []
    for run_dir, seed in zip(synthetic_dirs, seeds):
        target_views = {name: {**view, "X": view["X"].copy()} for name, view in target_base.items()}
        source_views, source_edge_report = make_partition_views(
            load_csv_dataset(run_dir, structural_only=True, require_contiguous=True)
        )
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler().fit(source_views["train"]["X"])
        for view in source_views.values():
            view["X"] = scaler.transform(view["X"])
        for view in target_views.values():
            view["X"] = scaler.transform(view["X"])
        arms: dict[str, tuple[Any, float, Any]] = {}
        rf = _fit_rf_arm(source_views["train"], seed)
        rf_threshold = _source_threshold(
            source_views["validation"]["y"][source_views["validation"]["labeled"]],
            _rf_scores(rf, source_views["validation"])[source_views["validation"]["labeled"]],
        )
        arms["rf"] = (
            rf,
            rf_threshold,
            lambda view, fitted=rf: _rf_scores(fitted, view),
        )
        sgc = _sgc_arm(source_views["train"], seed)
        sgc_threshold = _source_threshold(
            source_views["validation"]["y"][source_views["validation"]["labeled"]],
            _sgc_scores(sgc, source_views["validation"])[source_views["validation"]["labeled"]],
        )
        arms["sgc_proxy"] = (
            sgc,
            sgc_threshold,
            lambda view, fitted=sgc: _sgc_scores(fitted, view),
        )
        if backend_status["is_full_gnn"]:
            for kind in gnn_kinds:
                model, threshold = _fit_full_arm(
                    source_views["train"],
                    source_views["validation"],
                    seed,
                    kind,
                    hidden,
                    layers,
                    dropout,
                    learning_rate,
                    weight_decay,
                    epochs,
                    patience,
                    threads,
                )
                arms[f"pyg_{kind}"] = (
                    model,
                    threshold,
                    lambda view, fitted=model: fitted.predict_proba(
                        view["X"], build_edge_index(view["edges"], len(view["ids"]))
                    ),
                )
        internal: dict[str, Any] = {}
        transfer: dict[str, Any] = {}
        for name, (model, threshold, scorer) in arms.items():
            source_val = scorer(source_views["validation"])
            source_test = scorer(source_views["test"])
            target_val = scorer(target_views["validation"])
            target_test = scorer(target_views["test"])
            internal[name] = {
                "validation": _view_metrics(source_views["validation"], source_val, threshold, k),
                "test": _view_metrics(source_views["test"], source_test, threshold, k),
            }
            transfer[name] = {
                "validation": _view_metrics(target_views["validation"], target_val, threshold, k),
                "test": _view_metrics(target_views["test"], target_test, threshold, k),
            }
        shuffle_seed = int.from_bytes(
            hashlib.sha256(f"{seed}:strict-edge-shuffle".encode()).digest()[:8], "little"
        )
        shuffled_train = {}
        shuffled_views = {}
        for name in ("train", "validation"):
            offset = int(name == "validation")
            shuffled, shuffle_report = degree_preserving_edge_shuffle(
                source_views[name]["edges"], shuffle_seed + offset
            )
            shuffled_train[name] = (shuffled, shuffle_report)
            shuffled_views[name] = _rebuild_view(source_views[name], shuffled, scaler)
        shuffled_sgc = _sgc_arm(shuffled_views["train"], seed)
        target_test_shuffled, target_test_shuffle_report = degree_preserving_edge_shuffle(
            target_views["test"]["edges"], shuffle_seed + 3
        )
        shuffled_target_test = _rebuild_view(target_views["test"], target_test_shuffled, scaler)
        shuffled_scores = _sgc_scores(shuffled_sgc, shuffled_target_test)
        shuffled_validation = shuffled_views["validation"]
        shuffled_threshold = _source_threshold(
            shuffled_validation["y"][shuffled_validation["labeled"]],
            _sgc_scores(shuffled_sgc, shuffled_validation)[shuffled_validation["labeled"]],
        )
        edge_shuffle = {
            "train": shuffled_train["train"][1],
            "validation": shuffled_train["validation"][1],
            "test": target_test_shuffle_report,
            "real_test": _view_metrics(
                shuffled_target_test, shuffled_scores, shuffled_threshold, k
            ),
            "full_models": {},
        }
        if backend_status["is_full_gnn"]:
            full_shuffle_seed = shuffle_seed + 1
            for kind in gnn_kinds:
                shuffled_model, shuffled_model_threshold = _fit_full_arm(
                    shuffled_views["train"],
                    shuffled_views["validation"],
                    seed,
                    kind,
                    hidden,
                    layers,
                    dropout,
                    learning_rate,
                    weight_decay,
                    epochs,
                    patience,
                    threads,
                )
                shuffled_full_scores = shuffled_model.predict_proba(
                    shuffled_target_test["X"],
                    build_edge_index(
                        shuffled_target_test["edges"], len(shuffled_target_test["ids"])
                    ),
                )
                shuffled_full_metrics = _view_metrics(
                    shuffled_target_test,
                    shuffled_full_scores,
                    shuffled_model_threshold,
                    k,
                )
                original_value = transfer[f"pyg_{kind}"]["test"]["aggregate"]["macro_f1"]
                shuffled_value = shuffled_full_metrics["aggregate"]["macro_f1"]
                edge_shuffle["full_models"][f"pyg_{kind}"] = {
                    "model_id": f"pyg_{kind}",
                    "seed": seed,
                    "shuffle_seeds": {
                        "train": shuffle_seed,
                        "validation": full_shuffle_seed,
                        "real_test": shuffle_seed + 3,
                    },
                    "real_test": shuffled_full_metrics,
                    "paired_delta": (
                        original_value - shuffled_value
                        if original_value is not None and shuffled_value is not None
                        else None
                    ),
                }
        per_seed.append(
            {
                "seed": seed,
                "internal": internal,
                "transfer_to_real": transfer,
                "edge_shuffle": edge_shuffle,
                "edge_filters": {"source": source_edge_report, "target": target_edge_report},
            }
        )
    aggregate: dict[str, Any] = {}
    model_names = sorted(per_seed[0]["transfer_to_real"]) if per_seed else []
    metric_paths = {
        "macro_f1": ("aggregate", "macro_f1"),
        "micro_f1": ("micro", "f1"),
        "macro_precision_at_k": ("aggregate", "macro_precision_at_k"),
        "micro_precision_at_k": ("micro", "precision_at_k"),
        "macro_recall_at_k": ("aggregate", "macro_recall_at_k"),
        "micro_recall_at_k": ("micro", "recall_at_k"),
    }
    for name in model_names:
        aggregate[name] = {"n": len(per_seed)}
        for output_name, (section, metric) in metric_paths.items():
            values = [
                entry["transfer_to_real"][name]["test"][section][metric] for entry in per_seed
            ]
            aggregate[name][f"{output_name}_mean"] = (
                None if any(value is None for value in values) else float(np.mean(values))
            )
        aggregate[name]["test_step_coverage_min"] = min(
            entry["transfer_to_real"][name]["test"]["aggregate"]["steps_evaluated"]
            for entry in per_seed
        )
    shuffle_deltas = [
        entry["transfer_to_real"]["sgc_proxy"]["test"]["aggregate"]["macro_f1"]
        - entry["edge_shuffle"]["real_test"]["aggregate"]["macro_f1"]
        for entry in per_seed
        if entry["transfer_to_real"]["sgc_proxy"]["test"]["aggregate"]["macro_f1"] is not None
        and entry["edge_shuffle"]["real_test"]["aggregate"]["macro_f1"] is not None
    ]
    full_shuffle_effects = {}
    full_shuffle_welch = {}
    if backend_status["is_full_gnn"]:
        for kind in gnn_kinds:
            values = [
                entry["edge_shuffle"]["full_models"][f"pyg_{kind}"]["paired_delta"]
                for entry in per_seed
                if entry["edge_shuffle"]["full_models"].get(f"pyg_{kind}", {}).get("paired_delta")
                is not None
            ]
            original_values = [
                entry["transfer_to_real"][f"pyg_{kind}"]["test"]["aggregate"]["macro_f1"]
                for entry in per_seed
                if entry["transfer_to_real"][f"pyg_{kind}"]["test"]["aggregate"]["macro_f1"]
                is not None
            ]
            shuffled_values = [
                entry["edge_shuffle"]["full_models"][f"pyg_{kind}"]["real_test"]["aggregate"][
                    "macro_f1"
                ]
                for entry in per_seed
                if entry["edge_shuffle"]["full_models"].get(f"pyg_{kind}")
                and entry["edge_shuffle"]["full_models"][f"pyg_{kind}"]["real_test"]["aggregate"][
                    "macro_f1"
                ]
                is not None
            ]
            full_shuffle_effects[f"pyg_{kind}"] = paired_effect(values)
            full_shuffle_welch[f"pyg_{kind}"] = welch_secondary(original_values, shuffled_values)
    sgc_original = [
        entry["transfer_to_real"]["sgc_proxy"]["test"]["aggregate"]["macro_f1"]
        for entry in per_seed
        if entry["transfer_to_real"]["sgc_proxy"]["test"]["aggregate"]["macro_f1"] is not None
    ]
    sgc_shuffled = [
        entry["edge_shuffle"]["real_test"]["aggregate"]["macro_f1"]
        for entry in per_seed
        if entry["edge_shuffle"]["real_test"]["aggregate"]["macro_f1"] is not None
    ]
    required_roster_set = {"sage", "gat", "gin"}
    roster_complete = set(gnn_kinds) == required_roster_set
    primary_coverage_complete = aggregate.get("pyg_sage", {}).get("test_step_coverage_min") == 9
    frozen_seed_match = tuple(seeds) == DEFAULT_SEEDS
    frozen_hyperparameters = bool(
        k == 100
        and hidden == 64
        and layers == 2
        and dropout == 0.25
        and learning_rate == 1e-3
        and weight_decay == 1e-4
        and epochs == 300
        and patience == 30
        and threads == 1
    )
    package_versions_recorded = bool(
        backend_status.get("package_versions") and all(backend_status["package_versions"].values())
    )
    eligible = bool(
        tier == "real"
        and acceptance_context
        and backend_status["is_full_gnn"]
        and roster_complete
        and primary_coverage_complete
        and frozen_seed_match
        and frozen_hyperparameters
        and package_versions_recorded
    )
    observed_cross_partition_edges = int(target_edge_report["cross_partition_after_filter"])
    observed_cross_partition_edges += sum(
        int(entry["edge_filters"]["source"]["cross_partition_after_filter"]) for entry in per_seed
    )
    if observed_cross_partition_edges != 0:
        raise ValueError("CROSS_PARTITION_EDGE")
    seed_plan["acceptance_eligible"] = eligible
    target_value = aggregate.get("pyg_sage", {}).get("macro_f1_mean")
    target_met = (
        bool(target_value > 0.5)
        if tier == "real" and eligible and target_value is not None
        else None
    )
    return {
        "schema_version": STRICT_SCHEMA_VERSION,
        "status": "completed",
        "benchmark_tier": tier,
        "acceptance_eligible": eligible,
        "acceptance_context_verified": acceptance_context,
        "split_semantics": "disjoint_temporal_subgraphs",
        "seeds": seed_plan,
        "backend": backend_status,
        "feature_space": "structural_7",
        "feature_space_policy": (
            "common structural topology space; behavior semantic columns are "
            "validated but not transferred to raw"
        ),
        "topology_policy": {
            "edge_index": "directed_original_edges_only",
            "sage": "separate_outgoing_and_reverse_message_branches",
            "gat": "original_edge_direction_only",
            "gin": "original_edge_direction_only",
        },
        "label_policy": {
            "positive_class": "1",
            "negative_class": "2",
            "unknown_policy": "exclude",
            "unknown_in_loss": False,
            "unknown_in_metrics": False,
        },
        "metric_policy": {
            "k": k,
            "threshold_source": "synthetic_validation",
            "tie_break": "txId_ascending",
            "zero_positive": "null",
        },
        "preprocessing": {"primary": "source_only", "scaler_fit": "synthetic_train"},
        "split_contract": {
            "train_steps": [1, 30],
            "validation_steps": [31, 40],
            "test_steps": [41, 49],
            "cross_partition_message_edges": observed_cross_partition_edges,
        },
        "protocol_gates": {
            "full_roster_requested": roster_complete,
            "full_roster_executed": roster_complete and backend_status["is_full_gnn"],
            "primary_step_coverage_complete": primary_coverage_complete,
            "frozen_seed_match": frozen_seed_match,
            "frozen_hyperparameters": frozen_hyperparameters,
            "package_versions_recorded": package_versions_recorded,
        },
        "model_roster": ["rf", "sgc_proxy"]
        + ([f"pyg_{kind}" for kind in gnn_kinds] if backend_status["is_full_gnn"] else []),
        "unavailable_models": [
            f"pyg_{kind}" for kind in ("sage", "gat", "gin") if not backend_status["is_full_gnn"]
        ],
        "oracle_real_trained": oracle_test,
        "per_seed": per_seed,
        "aggregate": aggregate,
        "edge_shuffle_aggregate": {
            "sgc_proxy": paired_effect(shuffle_deltas),
            "full_models": full_shuffle_effects,
        },
        "unpaired_sensitivity": {
            "comparison": "original_vs_shuffled_real_test_macro_f1",
            "sgc_proxy": welch_secondary(sgc_original, sgc_shuffled),
            "full_models": full_shuffle_welch,
        },
        "acceptance": {
            "status": "eligible" if eligible else "diagnostic",
            "target": "primary_macro_test_f1 > 0.5",
            "target_met": target_met,
        },
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_raw_directory(raw_dir: Path) -> tuple[dict[str, Any], dict[str, str]]:
    """Validate the raw Elliptic directory and fingerprint its source files.

    Parameters
    ----------
    raw_dir : pathlib.Path
        Directory containing the three Elliptic CSV files.

    Returns
    -------
    tuple[dict[str, Any], dict[str, str]]
        Validated indexed dataset and SHA-256 digest for each raw CSV.

    Raises
    ------
    ValueError
        If raw files, schemas, class values, IDs, time steps, or edge uniqueness
        violate the pinned input contract.
    """
    dataset = load_csv_dataset(raw_dir, structural_only=True)
    features = dataset["features"]
    classes = dataset["classes"]
    edges = dataset["edges"]
    values = set(classes["class"].astype(str).unique())
    if not values <= {"1", "2", "unknown"}:
        raise ValueError("RAW_CLASS_VALUE")
    if features.index.has_duplicates or classes.index.has_duplicates:
        raise ValueError("RAW_ID_DUPLICATE")
    if not features.index.equals(classes.index):
        raise ValueError("RAW_ID_ORDER")
    time_values = features["time_step"].to_numpy()
    if np.any(time_values < 1) or np.any(time_values > 49):
        raise ValueError("RAW_TIME_STEP")
    edge_values = edges[["txId1", "txId2"]].astype(int)
    if (edge_values["txId1"] == edge_values["txId2"]).any():
        raise ValueError("RAW_SELF_LOOP")
    if edge_values.duplicated().any():
        raise ValueError("RAW_DUPLICATE_EDGE")
    fingerprints = {}
    names = (
        "elliptic_txs_features.csv",
        "elliptic_txs_classes.csv",
        "elliptic_txs_edgelist.csv",
    )
    for name in names:
        path = raw_dir / name
        fingerprints[name] = _sha256_file(path)
    return dataset, fingerprints


def write_report_atomic(
    path: Path, report: dict[str, Any], forbidden_root: Path | None = None
) -> Path:
    """Write a JSON report atomically outside a protected raw-data root.

    Parameters
    ----------
    path : pathlib.Path
        Destination report path. Parent directories are created when needed.
    report : dict[str, Any]
        JSON-serializable report with no non-finite numeric constants.
    forbidden_root : pathlib.Path or None, optional
        Protected raw-data directory that must not contain the destination.

    Returns
    -------
    pathlib.Path
        The destination path after the temporary file has been atomically installed.

    Raises
    ------
    ValueError
        If the destination is inside ``forbidden_root`` or the report is not valid
        finite JSON.
    TypeError
        If the report contains a value that cannot be serialized to JSON.
    OSError
        If directories or files cannot be created, flushed, replaced, or removed.
    """
    if forbidden_root is not None:
        root = forbidden_root.resolve()
        destination = path.resolve()
        if destination == root or root in destination.parents:
            raise ValueError("OUTPUT_UNDER_RAW_DIR")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    payload = json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return path


def run_validate_real(
    config_path: Path,
    raw_dir: Path,
    out_path: Path,
    backend: str | None = None,
    tier: str = "smoke",
    seeds: list[int] | None = None,
    k: int | None = None,
    all_gnn: bool = True,
) -> dict[str, Any]:
    try:
        raw_resolved = raw_dir.resolve()
        out_resolved = out_path.resolve()
        if out_resolved == raw_resolved or raw_resolved in out_resolved.parents:
            return {
                "schema_version": STRICT_SCHEMA_VERSION,
                "status": "error",
                "benchmark_tier": tier,
                "acceptance_eligible": False,
                "raw_fingerprint": {},
                "synthetic_fingerprints": {},
                "config_fingerprint": None,
                "effective_config": None,
                "config_sources": None,
                "report_written": False,
                "error": {
                    "code": "OUTPUT_UNDER_RAW_DIR",
                    "message": "report output must be outside raw data",
                },
            }
    except OSError as exc:
        return {
            "schema_version": STRICT_SCHEMA_VERSION,
            "status": "error",
            "benchmark_tier": tier,
            "acceptance_eligible": False,
            "raw_fingerprint": {},
            "config_fingerprint": None,
            "effective_config": None,
            "config_sources": None,
            "report_written": False,
            "error": {"code": "OUTPUT_PATH", "message": str(exc)},
        }
    seed_values = list(seeds) if seeds is not None else None
    report: dict[str, Any] = {
        "schema_version": STRICT_SCHEMA_VERSION,
        "status": "error",
        "benchmark_tier": tier,
        "acceptance_eligible": False,
        "raw_fingerprint": {},
        "synthetic_fingerprints": {},
        "config_fingerprint": None,
        "environment": None,
        "protocol_config_fingerprint": None,
        "effective_config": None,
        "config_sources": None,
        "report_written": False,
    }
    try:
        from .config import GeneratorConfig
        from .emit import write_dataset
        from .graph import build_graph
        from .stats import VolumeProfile
        from .validate import validate_behavior_dataset, validate_semantic_features

        config = GeneratorConfig.from_yaml(config_path)
        effective_backend = backend if backend is not None else config.gnn.backend
        effective_k = k if k is not None else config.gnn.k
        effective_seeds = seed_values if seed_values is not None else list(config.gnn.seeds)
        report["effective_config"] = {
            "backend": effective_backend,
            "k": effective_k,
            "seeds": effective_seeds,
            "hidden": config.gnn.hidden,
            "layers": config.gnn.layers,
            "dropout": config.gnn.dropout,
            "learning_rate": config.gnn.learning_rate,
            "weight_decay": config.gnn.weight_decay,
            "epochs": config.gnn.epochs,
            "patience": config.gnn.patience,
            "threads": config.gnn.threads,
            "all_gnn": all_gnn,
        }
        report["config_sources"] = {
            "backend": "cli" if backend is not None else "config",
            "k": "cli" if k is not None else "config",
            "seeds": "cli" if seed_values is not None else "config",
            "threads": "config",
            "model_hyperparameters": "config",
            "all_gnn": "cli",
        }
        volume_path = config.stats_dir / "volume.npy"
        if not volume_path.is_file():
            raise ValueError("VOLUME_ARTIFACT_MISSING")
        volume_fingerprint = _sha256_file(volume_path)
        report["environment"] = {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "volume_sha256": volume_fingerprint,
            "expected_volume_sha256": PINNED_VOLUME_SHA256,
            "volume_match": volume_fingerprint == PINNED_VOLUME_SHA256,
        }
        config_payload = json.dumps(
            config.to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode()
        report["config_fingerprint"] = hashlib.sha256(config_payload).hexdigest()
        report["environment"]["expected_config_sha256"] = PINNED_BEHAVIOR_CONFIG_SHA256
        report["environment"]["config_match"] = (
            report["config_fingerprint"] == PINNED_BEHAVIOR_CONFIG_SHA256
        )
        protocol_payload = json.dumps(
            {
                "config_fingerprint": report["config_fingerprint"],
                "effective_config": report["effective_config"],
                "pinned_raw_fingerprints": PINNED_REAL_RAW_FINGERPRINTS,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        report["protocol_config_fingerprint"] = hashlib.sha256(protocol_payload).hexdigest()
        validate_seed_plan(effective_seeds, tier)
        raw_dataset, raw_fingerprint = validate_raw_directory(raw_dir)
        report["raw_fingerprint"] = raw_fingerprint
        raw_row_count = len(raw_dataset["features"])
        raw_fingerprint_match = raw_fingerprint == PINNED_REAL_RAW_FINGERPRINTS
        report["raw_contract"] = {
            "expected_rows": REAL_RAW_ROWS,
            "observed_rows": raw_row_count,
            "rows_match": raw_row_count == REAL_RAW_ROWS,
            "expected_sha256": PINNED_REAL_RAW_FINGERPRINTS,
            "fingerprint_match": raw_fingerprint_match,
            "pinned": tier == "real",
        }
        acceptance_context = bool(
            tier == "real"
            and raw_row_count == REAL_RAW_ROWS
            and raw_fingerprint_match
            and report["environment"]["volume_match"]
            and report["environment"]["config_match"]
            and tuple(effective_seeds) == DEFAULT_SEEDS
        )
        if tier == "real" and not acceptance_context:
            raise ValueError("RAW_ACCEPTANCE_CONTRACT")
        backend_status = resolve_backend(effective_backend)
        report["backend"] = backend_status
        if tier == "real" and not backend_status["is_full_gnn"]:
            raise BackendUnavailable(
                "real tier requires a verified full PyG backend", backend_status
            )
        if tier == "real" and not backend_status["environment_pinned"]:
            raise BackendUnavailable(
                "real tier requires the pinned optional package versions", backend_status
            )
        protocol_payload = json.dumps(
            {
                "config_fingerprint": report["config_fingerprint"],
                "effective_config": report["effective_config"],
                "raw_fingerprint": raw_fingerprint,
                "backend": backend_status,
                "environment": report["environment"],
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        report["protocol_config_fingerprint"] = hashlib.sha256(protocol_payload).hexdigest()
        synthetic_fingerprints: dict[str, dict[str, str]] = {}
        with tempfile.TemporaryDirectory(prefix=".kyt_strict_", dir=Path.cwd()) as temporary:
            run_dirs = []
            for seed in effective_seeds:
                run_config = GeneratorConfig.from_yaml(config_path)
                run_config.seed = seed
                run_config.out_dir = Path(temporary) / f"seed_{seed}"
                stats = VolumeProfile.from_dir(run_config.stats_dir)
                graph = build_graph(run_config, stats)
                write_dataset(run_config, graph, run_config.out_dir)
                validate_behavior_dataset(run_config.out_dir)
                validate_semantic_features(run_config.out_dir)
                synthetic_fingerprints[str(seed)] = {
                    name: _sha256_file(run_config.out_dir / name)
                    for name in (
                        "elliptic_txs_features.csv",
                        "elliptic_txs_classes.csv",
                        "elliptic_txs_edgelist.csv",
                        "elliptic_txs_edge_attributes.csv",
                    )
                }
                run_dirs.append(run_config.out_dir)
            protocol = run_strict_protocol(
                run_dirs,
                raw_dir,
                effective_seeds,
                raw_dataset=raw_dataset,
                backend_status=backend_status,
                acceptance_context=acceptance_context,
                backend=effective_backend,
                tier=tier,
                k=effective_k,
                gnn_kinds=("sage", "gat", "gin") if all_gnn else ("sage",),
                hidden=config.gnn.hidden,
                layers=config.gnn.layers,
                dropout=config.gnn.dropout,
                learning_rate=config.gnn.learning_rate,
                weight_decay=config.gnn.weight_decay,
                epochs=config.gnn.epochs,
                patience=config.gnn.patience,
                threads=config.gnn.threads,
            )
        report.update(protocol)
        report["synthetic_fingerprints"] = synthetic_fingerprints
        report["raw_fingerprint"] = raw_fingerprint
        report["status"] = "completed"
        report["acceptance_eligible"] = bool(protocol["acceptance_eligible"])
    except BackendUnavailable as exc:
        report["status"] = "blocked"
        if exc.status is not None:
            report["backend"] = exc.status
        report["error"] = {"code": "BACKEND_UNAVAILABLE", "message": str(exc)}
    except ValueError as exc:
        report["status"] = "error"
        report["error"] = {"code": str(exc)[:80], "message": str(exc)}
    except (FileNotFoundError, OSError) as exc:
        report["status"] = "error"
        report["error"] = {"code": type(exc).__name__, "message": str(exc)}
    except Exception as exc:
        report["status"] = "error"
        report["error"] = {"code": "VALIDATION_ERROR", "message": str(exc)}
    try:
        report["report_written"] = True
        write_report_atomic(out_path, report, forbidden_root=raw_dir)
    except (OSError, TypeError, ValueError) as exc:
        report["status"] = "error"
        report["report_written"] = False
        report["error"] = {"code": "REPORT_WRITE_ERROR", "message": str(exc)}
    return report


__all__ = [
    "BackendUnavailable",
    "DEFAULT_SEEDS",
    "MIN_ACCEPTANCE_SEEDS",
    "PINNED_BEHAVIOR_CONFIG_SHA256",
    "PINNED_FULL_VERSIONS",
    "PINNED_VOLUME_SHA256",
    "PARTITION_STEPS",
    "STRICT_SCHEMA_VERSION",
    "aggregate_step_metrics",
    "binary_metrics_at_threshold",
    "build_edge_index",
    "build_label_masks",
    "build_partition_masks",
    "choose_f1_threshold",
    "degree_preserving_edge_shuffle",
    "filter_partition_edges",
    "load_csv_dataset",
    "make_partition_views",
    "fit_pyg_classifier",
    "fit_sgc_proxy",
    "predict_sgc_proxy",
    "partition_for_step",
    "paired_effect",
    "PyGClassifier",
    "resolve_backend",
    "run_strict_protocol",
    "run_validate_real",
    "per_step_metrics",
    "validate_raw_directory",
    "validate_seed_plan",
    "welch_secondary",
    "write_report_atomic",
]
