from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .config import DriftConfig, GeneratorConfig
from .schemes import BUILDERS, Scheme
from .stats import N_STEPS, STEP_MIN, EllipticStats

STEP_MAX = N_STEPS
OFFSET_RANGE = 3  # nodes of one scheme spread over birth..birth+2


@dataclass
class TxNode:
    tx_id: int
    step: int
    cls: str  # illicit | licit | unknown
    role: str
    scheme: str


@dataclass
class GeneratedGraph:
    nodes: list[TxNode] = field(default_factory=list)
    edges: list[tuple[int, int]] = field(default_factory=list)


def _offset_steps(rng: np.random.Generator, birth: int, n: int) -> list[int]:
    offs = rng.integers(0, OFFSET_RANGE, size=n)
    return [int(np.clip(birth + o, STEP_MIN, STEP_MAX)) for o in offs]


def _has_illicit(scheme: Scheme) -> bool:
    return any(n["cls"] == "illicit" for n in scheme.nodes)


def _flatten_pool(config: GeneratorConfig) -> list[tuple[str, dict]]:
    pool: list[tuple[str, dict]] = []
    for name, params in config.schemes.items():
        pool.extend((name, params) for _ in range(int(params.get("n_instances", 10))))
    return pool


def build_graph(config: GeneratorConfig, stats: EllipticStats) -> GeneratedGraph:
    """Assemble the graph: composite schemes within illicit budget + background to n_txs."""
    rng = np.random.default_rng(config.seed)
    drift: DriftConfig = config.drift

    n_labeled = int(config.n_txs * config.labeled_ratio)
    n_illicit_budget = int(n_labeled * config.illicit_ratio_in_labeled)
    n_licit_budget = n_labeled - n_illicit_budget

    pool = _flatten_pool(config)
    # The scheme set aside as the drift "novel" pattern is never a regular scheme; it only
    # surfaces in late steps when drift is enabled. It is still shuffled to keep the stream stable.
    novel = [(n, p) for n, p in pool if n == drift.novel_scheme]
    regular = [(n, p) for n, p in pool if n != drift.novel_scheme]
    rng.shuffle(regular)
    rng.shuffle(novel)

    graph = GeneratedGraph()
    next_id = 0
    consumed_illicit = 0

    def _register(scheme: Scheme, birth: int) -> None:
        nonlocal next_id, consumed_illicit
        steps = _offset_steps(rng, birth, len(scheme.nodes))
        local_to_global: dict[int, int] = {}
        for local, node in enumerate(scheme.nodes):
            tx_id = next_id
            next_id += 1
            local_to_global[local] = tx_id
            graph.nodes.append(
                TxNode(
                    tx_id=tx_id,
                    step=steps[local],
                    cls=node["cls"],
                    role=node["role"],
                    scheme=scheme.name,
                )
            )
            if node["cls"] == "illicit":
                consumed_illicit += 1
        for u, v in scheme.edges:
            graph.edges.append((local_to_global[u], local_to_global[v]))

    for name, params in regular:
        scheme = BUILDERS[name](rng, params)
        is_illicit = _has_illicit(scheme)
        birth = int(stats.sample_step(rng, 1)[0])
        if drift.enabled and is_illicit:
            last_step = int(np.clip(birth + OFFSET_RANGE - 1, STEP_MIN, STEP_MAX))
            if last_step >= drift.shutdown_step:
                keep = rng.random() < drift.shutdown_rate_multiplier
                if not keep:
                    continue
        _register(scheme, birth)
        if is_illicit and consumed_illicit >= n_illicit_budget:
            break

    if drift.enabled:
        for name, params in novel:
            scheme = BUILDERS[name](rng, params)
            birth = int(rng.integers(max(STEP_MIN, drift.novel_step), STEP_MAX + 1))
            _register(scheme, birth)

    scheme_n = len(graph.nodes)
    n_bg = config.n_txs - scheme_n
    if n_bg < 0:
        raise ValueError(
            f"schemes produce {scheme_n} nodes, exceeding n_txs={config.n_txs}; reduce n_instances"
        )

    # Background carries the leftover labeled budgets (may stay below them — relaxed ratio).
    n_bg_illicit = max(0, n_illicit_budget - sum(1 for nd in graph.nodes if nd.cls == "illicit"))
    n_bg_licit = max(0, n_licit_budget - sum(1 for nd in graph.nodes if nd.cls == "licit"))
    n_bg_illicit = min(n_bg_illicit, n_bg)
    n_bg_licit = min(n_bg_licit, n_bg - n_bg_illicit)
    n_bg_unknown = n_bg - n_bg_illicit - n_bg_licit

    bg_pool = np.concatenate(
        [
            np.full(n_bg_illicit, "illicit"),
            np.full(n_bg_licit, "licit"),
            np.full(n_bg_unknown, "unknown"),
        ]
    ).astype(object)
    rng.shuffle(bg_pool)

    bg_ids: list[int] = []
    for i in range(n_bg):
        tx_id = next_id
        next_id += 1
        cls = str(bg_pool[i])
        if cls == "illicit" and drift.enabled:
            step = int(rng.integers(STEP_MIN, max(STEP_MIN, drift.shutdown_step - 1) + 1))
        else:
            step = int(stats.sample_step(rng, 1)[0])
        graph.nodes.append(TxNode(tx_id=tx_id, step=step, cls=cls, role="background", scheme="p2p"))
        bg_ids.append(tx_id)

    # Random directed pairs between background txs, no self-loops; not a connectivity guarantee.
    n_bg_edges = int(config.p2p_edges_per_tx * n_bg)
    if bg_ids:
        src = rng.choice(bg_ids, size=n_bg_edges)
        dst = rng.choice(bg_ids, size=n_bg_edges)
        for u, v in zip(src, dst):
            if u != v:
                graph.edges.append((int(u), int(v)))

    return graph
