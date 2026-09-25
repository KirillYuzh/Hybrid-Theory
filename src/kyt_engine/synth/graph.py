from dataclasses import dataclass, field

from .config import GeneratorConfig
from .stats import VolumeProfile


@dataclass
class TxNode:
    tx_id: int
    step: int
    cls: str
    role: str
    scheme: str


@dataclass
class GeneratedGraph:
    nodes: list[TxNode] = field(default_factory=list)
    edges: list[tuple[int, int]] = field(default_factory=list)


def build_graph(config: GeneratorConfig, stats: VolumeProfile):
    from .behavior import build_behavior_graph

    return build_behavior_graph(config, stats)


__all__ = ["GeneratedGraph", "TxNode", "build_graph"]
