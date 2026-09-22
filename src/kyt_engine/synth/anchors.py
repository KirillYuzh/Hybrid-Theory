from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import numpy as np

from .config import GeneratorConfig
from .graph import GeneratedGraph
from .stats import STEP_MIN

# New RNG draws (edge amounts) must never touch the structural stream default_rng(seed),
# otherwise AC numbers (10000 tx / 14092 edges) drift. Derive a separate deterministic stream.
ATTR_SALT = ":attrs"


def attr_seed(seed: int) -> int:
    return int.from_bytes(hashlib.sha256(f"{seed}{ATTR_SALT}".encode()).digest()[:8], "little")


def attr_rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(attr_seed(seed))


@dataclass
class PatternInstance:
    """Annotations of one planted subgraph (or decoy p2p block) over the built graph."""

    instance_id: int
    pattern_type: str
    node_ids: list[int]
    edge_ids: list[int]
    anchor_node_id: int
    anchor_role: str
    max_anchor_distance: int
    temporal_window: tuple[int, int]
    edge_attr_recipe: dict[int, str]
    anchored: bool = False
    entity_type: str | None = None
    articulation_points: list[int] = field(default_factory=list)
    holdout: bool = False
    train_excluded: bool = False


def _instance_adjacency(graph: GeneratedGraph, node_ids: list[int], edge_ids: list[int]):
    """Undirected adjacency of the instance subgraph, keyed by global tx_id."""
    members = set(node_ids)
    adj: dict[int, list[tuple[int, int]]] = {n: [] for n in node_ids}
    for e in edge_ids:
        u, v = graph.edges[e]
        if u in members and v in members:
            adj[u].append((v, e))
            adj[v].append((u, e))
    for lst in adj.values():
        lst.sort()
    return adj


def _bfs_layers(adj: dict[int, list[tuple[int, int]]], src: int) -> dict[int, int]:
    """Distances from src within the instance subgraph (values are (neighbor, edge_id))."""
    dist = {src: 0}
    queue = [src]
    for u in queue:
        for v, _ in adj[u]:
            if v not in dist:
                dist[v] = dist[u] + 1
                queue.append(v)
    return dist


def _eccentricity(
    graph: GeneratedGraph, node_ids: list[int], edge_ids: list[int], anchor: int
) -> int:
    adj = _instance_adjacency(graph, node_ids, edge_ids)
    return max(_bfs_layers(adj, anchor).values(), default=0)


def _articulation_points(
    graph: GeneratedGraph, node_ids: list[int], edge_ids: list[int]
) -> list[int]:
    """Tarjan on the instance subgraph (deterministic, no RNG)."""
    adj = _instance_adjacency(graph, node_ids, edge_ids)
    disc: dict[int, int] = {}
    low: dict[int, int] = {}
    visited = set()
    arts: set[int] = set()
    count = 0

    def dfs(u: int, parent: int) -> None:
        nonlocal count
        visited.add(u)
        disc[u] = low[u] = count
        count += 1
        children = 0
        for v, _ in adj[u]:
            if v == parent:
                continue
            if v not in visited:
                children += 1
                dfs(v, u)
                low[u] = min(low[u], low[v])
                if parent == -1 and children > 1:
                    arts.add(u)
                elif parent != -1 and low[v] >= disc[u]:
                    arts.add(u)
            else:
                low[u] = min(low[u], disc[v])

    for n in sorted(adj):
        if n not in visited:
            dfs(n, -1)
    return sorted(arts)


EDGE_RECIPES: dict[str, str] = {
    "peel_chain": "peel",
    "wash": "wash",
    "structuring": "struct_deposit",
    "cycle_round_trip": "round_trip",
    "bridge_hopping": "bridge_swap",
    "amm_swap_chain": "amm_swap",
    "exchange_hub": "exchange",
    "miner_payout": "payout",
    "wallet_provider": "custody",
}


def _edge_recipe(pattern_type: str, edge: tuple[int, int], anchor: int) -> str:
    if pattern_type == "mixer":
        if edge[1] == anchor:
            return "mixer_in"
        return "mixer_out"
    return EDGE_RECIPES.get(pattern_type, "star")  # fanout / hub_spoke / others


def _central_anchor(node_ids: list[int], graph: GeneratedGraph, edge_ids: list[int]) -> int:
    """Node minimizing eccentricity in the instance subgraph (most representative query),
    ties resolved to the smallest node id for determinism."""
    adj = _instance_adjacency(graph, node_ids, edge_ids)
    best, best_ecc = node_ids[0], len(node_ids)
    for n in node_ids:
        ecc = max(_bfs_layers(adj, n).values(), default=0)
        if ecc < best_ecc or (ecc == best_ecc and n < best):
            best, best_ecc = n, ecc
    return best


def build_instances(graph: GeneratedGraph, config: GeneratorConfig) -> list[PatternInstance]:
    """Zero RNG. Instance node/edge ids from contiguous scheme_run blocks."""
    entity_types = config.anchors.entity_types
    instances: list[PatternInstance] = []
    for run in graph.runs:
        node_ids = list(range(run.node_id_start, run.node_id_start + run.node_count))
        edge_ids = list(range(run.edge_id_start, run.edge_id_start + run.edge_count))
        anchor = (
            _central_anchor(node_ids, graph, edge_ids)
            if config.anchors.prefer_central_anchors
            else node_ids[0]
        )
        steps = [graph.nodes[n].step for n in node_ids]
        recipe = {e: _edge_recipe(run.name, graph.edges[e], anchor) for e in edge_ids}
        instances.append(
            PatternInstance(
                instance_id=len(instances),
                pattern_type=run.name,
                node_ids=node_ids,
                edge_ids=edge_ids,
                anchor_node_id=anchor,
                anchor_role=graph.nodes[anchor].role,
                max_anchor_distance=_eccentricity(graph, node_ids, edge_ids, anchor),
                temporal_window=(min(steps), max(steps)),
                edge_attr_recipe=recipe,
                entity_type=entity_types.get(run.name),
                articulation_points=_articulation_points(graph, node_ids, edge_ids),
            )
        )
    return instances


def instance_id_of(
    graph: GeneratedGraph, instances: list[PatternInstance]
) -> dict[int, int | None]:
    """tx_id -> instance_id (None for background). Nodes are registered in tx_id order."""
    mapping: dict[int, int | None] = {}
    for inst in instances:
        for n in inst.node_ids:
            mapping[n] = inst.instance_id
    for nd in graph.nodes:
        mapping.setdefault(nd.tx_id, None)
    return mapping


def _cents(value: float) -> int:
    return int(round(value * 100))


def _to_amount(cents: int) -> float:
    return round(cents / 100.0, 2)


def _uniform_cents(rng: np.random.Generator, lo: float, hi: float) -> int:
    return rng.integers(_cents(lo), _cents(hi) + 1)


def build_edge_attributes(
    graph: GeneratedGraph,
    instances: list[PatternInstance],
    config: GeneratorConfig,
    rng: np.random.Generator,
) -> list[dict | None]:
    """One entry per edge (index == edgelist row). Amounts in cents for exact invariants."""
    amount_conf = config.edge_attributes.amount
    scale = config.edge_attributes.timestamp_scale
    attrs: list[dict | None] = [None] * len(graph.edges)
    node_step = {nd.tx_id: nd.step for nd in graph.nodes}

    def timestamp(u: int, v: int) -> int:
        return scale * (max(node_step[u], node_step[v]) - STEP_MIN)

    def entry(u: int, v: int, cents: int) -> dict:
        return {"txId1": u, "txId2": v, "amount": _to_amount(cents), "timestamp": timestamp(u, v)}

    for inst in instances:
        pattern = inst.pattern_type
        conf = amount_conf.get(pattern, amount_conf["p2p"])
        if pattern == "mixer":
            in_e = [e for e in inst.edge_ids if inst.edge_attr_recipe[e] == "mixer_in"]
            out_e = [e for e in inst.edge_ids if inst.edge_attr_recipe[e] == "mixer_out"]
            a_lo, a_hi = conf["amount_per_in"]
            in_cents = [_uniform_cents(rng, a_lo, a_hi) for _ in in_e]
            total_in = sum(in_cents)
            fee_lo, fee_hi = conf["fee_fraction"]
            fee = int(round(total_in * rng.uniform(fee_lo, fee_hi)))
            out_budget = total_in - fee
            if out_e:
                weights = rng.uniform(0.5, 1.5, size=len(out_e))
                wsum = weights.sum()
                out_cents: list[int] = []
                acc = 0
                for w in weights[:-1]:
                    part = int(out_budget * w / wsum)
                    out_cents.append(part)
                    acc += part
                out_cents.append(out_budget - acc)  # last edge absorbs punctuation
            else:
                out_cents = []
            for e, c in zip(in_e, in_cents):
                u, v = graph.edges[e]
                attrs[e] = entry(u, v, c)
            for e, c in zip(out_e, out_cents):
                u, v = graph.edges[e]
                attrs[e] = entry(u, v, c)
        elif pattern == "peel_chain":
            amount = _uniform_cents(rng, *conf["start_amount"])
            drip_lo, drip_hi = conf["drip"]
            for e in inst.edge_ids:
                u, v = graph.edges[e]
                attrs[e] = entry(u, v, amount)
                if e != inst.edge_ids[-1]:
                    drip = rng.uniform(drip_lo, drip_hi)
                    amount = max(1, int(amount * drip))  # floor at 1 cent; monotone non-increasing
        elif pattern == "wash":
            tol = conf["balance_tolerance"]
            lo, hi = 1.0 / (1.0 + tol), 1.0 + tol
            base = _uniform_cents(rng, *conf["amount"])
            amount = base
            cumulative = 1.0
            for e in inst.edge_ids[:-1]:
                u, v = graph.edges[e]
                attrs[e] = entry(u, v, amount)
                r = rng.uniform(lo, hi)
                new_cum = min(max(cumulative * r, lo), hi)  # bound cumulative drift
                r = new_cum / cumulative
                cumulative = new_cum
                amount = max(1, int(round(amount * r)))
            e = inst.edge_ids[-1]  # closing edge returns to the start -> every node balances
            u, v = graph.edges[e]
            attrs[e] = entry(u, v, base)
        else:  # fanout / hub_spoke star
            for e in inst.edge_ids:
                u, v = graph.edges[e]
                attrs[e] = entry(u, v, _uniform_cents(rng, *conf["amount"]))

    for e, (u, v) in enumerate(graph.edges):
        if attrs[e] is None:
            attrs[e] = entry(u, v, _uniform_cents(rng, *amount_conf["p2p"]["amount"]))
    return attrs


def _bfs_distance(adjacency: dict[int, list[int]], src: int, dst: int, cap: int) -> int | None:
    """Undirected BFS capped at `cap`; None means farther than cap. Nodes visited in id order."""
    if src == dst:
        return 0
    dist = {src: 0}
    queue = [src]
    for u in queue:
        if dist[u] >= cap:
            continue
        for v in adjacency[u]:
            if v not in dist:
                dist[v] = dist[u] + 1
                if v == dst:
                    return dist[v]
                queue.append(v)
    return None


def _full_adjacency(graph: GeneratedGraph) -> dict[int, list[int]]:
    adj: dict[int, list[int]] = {nd.tx_id: [] for nd in graph.nodes}
    for u, v in graph.edges:
        adj[u].append(v)
        adj[v].append(u)
    for lst in adj.values():
        lst.sort()
    return adj


def _too_close(adjacency: dict[int, list[int]], a: int, b: int, min_distance: int) -> bool:
    """True only when a real (measured) distance < min_distance; unreachable-within-cap is OK."""
    d = _bfs_distance(adjacency, a, b, min_distance)
    return d is not None and d < min_distance


def select_anchors(
    instances: list[PatternInstance], graph: GeneratedGraph, config: GeneratorConfig
) -> None:
    """No RNG. Isolation >= min_anchor_distance over the full graph; budget target is a ceiling."""
    cfg = config.anchors
    target = cfg.per_1000_nodes * len(graph.nodes) // 1000
    adjacency = _full_adjacency(graph)
    kept: list[int] = []
    for inst in instances:
        a = inst.anchor_node_id
        if any(_too_close(adjacency, a, k, cfg.min_anchor_distance) for k in kept):
            continue
        inst.anchored = True
        kept.append(a)
        if len(kept) >= target:
            break


def k_hop_neighborhoods(
    inst: PatternInstance, graph: GeneratedGraph, config: GeneratorConfig
) -> dict[str, list[int]]:
    """Nodes at exact hop distance k from the anchor, inside the instance (k in k_hop_range)."""
    range_cfg = config.anchors.k_hop_range or [0]
    max_k = min(inst.max_anchor_distance, max(range_cfg))
    adj = _instance_adjacency(graph, inst.node_ids, inst.edge_ids)
    dist = _bfs_layers(adj, inst.anchor_node_id)
    return {str(k): sorted(n for n, d in dist.items() if d == k) for k in range(1, max_k + 1)}


def detect_decoys(graph: GeneratedGraph, config: GeneratorConfig) -> list[dict]:
    """Find accidental fan-in / cycle motifs among p2p edges and tag them as decoys. No RNG."""
    if not config.background.decoy_detection:
        return []
    fan_in_threshold = config.background.fan_in_threshold
    max_depth = config.background.cycle_max_depth

    is_bg = {nd.tx_id: nd.scheme == "p2p" for nd in graph.nodes}
    bg_edges = [(e, u, v) for e, (u, v) in enumerate(graph.edges) if is_bg[u] and is_bg[v]]

    blocks: list[dict] = []

    in_sources: dict[int, dict[int, int]] = {}  # target -> {source: edge_id}
    for e, u, v in bg_edges:
        in_sources.setdefault(v, {})[u] = e
    for target in sorted(in_sources):
        srcs = in_sources[target]
        if len(srcs) >= fan_in_threshold:
            edge_ids = sorted(srcs.values())
            nodes = [target] + sorted(srcs)
            blocks.append(
                {"nodes": nodes, "edge_ids": edge_ids, "motif": "fan_in", "pattern_type": "p2p"}
            )

    adj: dict[int, list[tuple[int, int]]] = {n: [] for n, _ in is_bg.items() if is_bg[n]}
    for e, u, v in bg_edges:
        adj[u].append((v, e))
        adj[v].append((u, e))
    for lst in adj.values():
        lst.sort()

    edge_of: dict[int, int] = {}
    path: list[int] = []
    path_set: set[int] = set()
    visited: set[int] = set()
    seen_lots: set[tuple[int, ...]] = set()

    def dfs(u: int, parent: int, depth: int) -> None:
        visited.add(u)
        path.append(u)
        path_set.add(u)
        for v, e in adj[u]:
            if v == parent:
                continue
            if v in path_set:
                if depth + 1 > max_depth:
                    continue
                start = path.index(v)
                cycle = path[start:]
                key = tuple(sorted(cycle))
                if key in seen_lots:
                    continue
                seen_lots.add(key)
                cycle_edges = [edge_of[path[i]] for i in range(start + 1, len(path))]
                cycle_edges.append(e)
                blocks.append(
                    {
                        "nodes": cycle,
                        "edge_ids": sorted(cycle_edges),
                        "motif": "cycle",
                        "pattern_type": "p2p",
                    }
                )
            elif v not in visited and depth + 1 <= max_depth:
                edge_of[v] = e
                dfs(v, u, depth + 1)
        path.pop()
        path_set.discard(u)

    for start in sorted(adj):
        if start not in visited:
            dfs(start, -1, 0)
    return blocks


def apply_holdout(instances: list[PatternInstance], config: GeneratorConfig) -> bool:
    """Tag instances whose temporal window overlaps a holdout entry. Annotation only. No RNG."""
    windows_active = bool(config.holdout.entries)
    for inst in instances:
        lo, hi = inst.temporal_window
        for entry in config.holdout.entries:
            if inst.pattern_type != entry.get("pattern"):
                continue
            s_min = entry.get("step_min")
            s_max = entry.get("step_max")
            if s_min is None or s_max is None:
                continue
            if max(lo, s_min) <= min(hi, s_max):
                inst.holdout = True
                inst.train_excluded = bool(entry.get("train_excluded", True))
                break
    return windows_active
