from dataclasses import dataclass, field

from .config import GeneratorConfig
from .graph import GeneratedGraph


@dataclass
class PatternInstance:
    instance_id: int
    pattern_type: str
    node_ids: list[int]
    edge_ids: list[int]
    anchor_node_id: int
    anchor_role: str
    max_anchor_distance: int
    temporal_window: tuple[int, int]
    edge_attr_recipe: dict[int, str] = field(default_factory=dict)
    anchored: bool = False
    entity_type: str | None = None
    articulation_points: list[int] = field(default_factory=list)
    holdout: bool = False
    train_excluded: bool = False


def _instance_adjacency(
    graph: GeneratedGraph, node_ids: list[int], edge_ids: list[int]
) -> dict[int, list[tuple[int, int]]]:
    members = set(node_ids)
    adjacency: dict[int, list[tuple[int, int]]] = {node_id: [] for node_id in node_ids}
    for edge_id in edge_ids:
        source, target = graph.edges[edge_id]
        if source in members and target in members:
            adjacency[source].append((target, edge_id))
            adjacency[target].append((source, edge_id))
    for values in adjacency.values():
        values.sort()
    return adjacency


def _bfs_layers(adjacency: dict[int, list[tuple[int, int]]], source: int) -> dict[int, int]:
    distances = {source: 0}
    queue = [source]
    for node_id in queue:
        for neighbor, _ in adjacency[node_id]:
            if neighbor not in distances:
                distances[neighbor] = distances[node_id] + 1
                queue.append(neighbor)
    return distances


def _eccentricity(
    graph: GeneratedGraph, node_ids: list[int], edge_ids: list[int], anchor: int
) -> int:
    adjacency = _instance_adjacency(graph, node_ids, edge_ids)
    return max(_bfs_layers(adjacency, anchor).values(), default=0)


def _articulation_points(
    graph: GeneratedGraph, node_ids: list[int], edge_ids: list[int]
) -> list[int]:
    adjacency = _instance_adjacency(graph, node_ids, edge_ids)
    discovery: dict[int, int] = {}
    low: dict[int, int] = {}
    visited: set[int] = set()
    result: set[int] = set()
    counter = 0

    def visit(node_id: int, parent: int) -> None:
        nonlocal counter
        visited.add(node_id)
        discovery[node_id] = low[node_id] = counter
        counter += 1
        children = 0
        for neighbor, _ in adjacency[node_id]:
            if neighbor == parent:
                continue
            if neighbor not in visited:
                children += 1
                visit(neighbor, node_id)
                low[node_id] = min(low[node_id], low[neighbor])
                if parent == -1 and children > 1:
                    result.add(node_id)
                elif parent != -1 and low[neighbor] >= discovery[node_id]:
                    result.add(node_id)
            else:
                low[node_id] = min(low[node_id], discovery[neighbor])

    for node_id in sorted(adjacency):
        if node_id not in visited:
            visit(node_id, -1)
    return sorted(result)


def _central_anchor(node_ids: list[int], graph: GeneratedGraph, edge_ids: list[int]) -> int:
    adjacency = _instance_adjacency(graph, node_ids, edge_ids)
    best = node_ids[0]
    best_distance = len(node_ids)
    for node_id in node_ids:
        distance = max(_bfs_layers(adjacency, node_id).values(), default=0)
        if distance < best_distance or (distance == best_distance and node_id < best):
            best = node_id
            best_distance = distance
    return best


def _full_adjacency(graph: GeneratedGraph) -> dict[int, list[int]]:
    adjacency: dict[int, list[int]] = {node.tx_id: [] for node in graph.nodes}
    for source, target in graph.edges:
        adjacency[source].append(target)
        adjacency[target].append(source)
    for values in adjacency.values():
        values.sort()
    return adjacency


def _bfs_distance(
    adjacency: dict[int, list[int]], source: int, target: int, maximum: int
) -> int | None:
    if source == target:
        return 0
    seen = {source}
    queue = [(source, 0)]
    for node_id, distance in queue:
        if distance >= maximum:
            continue
        for neighbor in adjacency[node_id]:
            if neighbor == target:
                return distance + 1
            if neighbor not in seen:
                seen.add(neighbor)
                queue.append((neighbor, distance + 1))
    return None


def _too_close(adjacency: dict[int, list[int]], first: int, second: int, minimum: int) -> bool:
    distance = _bfs_distance(adjacency, first, second, minimum)
    return distance is not None and distance < minimum


def select_anchors(
    instances: list[PatternInstance], graph: GeneratedGraph, config: GeneratorConfig
) -> None:
    target = config.anchors.per_1000_nodes * len(graph.nodes) // 1000
    adjacency = _full_adjacency(graph)
    kept: list[int] = []
    for instance in instances:
        anchor = instance.anchor_node_id
        if any(
            _too_close(adjacency, anchor, other, config.anchors.min_anchor_distance)
            for other in kept
        ):
            continue
        instance.anchored = True
        kept.append(anchor)
        if len(kept) >= target:
            break


def k_hop_neighborhoods(
    instance: PatternInstance, graph: GeneratedGraph, config: GeneratorConfig
) -> dict[str, list[int]]:
    hops = config.anchors.k_hop_range or (0,)
    maximum = min(instance.max_anchor_distance, max(hops))
    adjacency = _instance_adjacency(graph, instance.node_ids, instance.edge_ids)
    distances = _bfs_layers(adjacency, instance.anchor_node_id)
    return {
        str(hop): sorted(node_id for node_id, distance in distances.items() if distance == hop)
        for hop in range(1, maximum + 1)
    }


def instance_id_of(
    graph: GeneratedGraph, instances: list[PatternInstance]
) -> dict[int, int | None]:
    mapping: dict[int, int | None] = {}
    for instance in instances:
        for node_id in instance.node_ids:
            mapping[node_id] = instance.instance_id
    for node in graph.nodes:
        mapping.setdefault(node.tx_id, None)
    return mapping


def apply_holdout(instances: list[PatternInstance], config: GeneratorConfig) -> bool:
    for instance in instances:
        for entry in config.holdout.entries:
            if instance.pattern_type != entry["pattern"]:
                continue
            if max(instance.temporal_window[0], entry["step_min"]) <= min(
                instance.temporal_window[1], entry["step_max"]
            ):
                instance.holdout = True
                instance.train_excluded = entry["train_excluded"]
                break
    return bool(config.holdout.entries)


__all__ = [
    "PatternInstance",
    "_articulation_points",
    "_central_anchor",
    "_eccentricity",
    "apply_holdout",
    "instance_id_of",
    "k_hop_neighborhoods",
    "select_anchors",
]
