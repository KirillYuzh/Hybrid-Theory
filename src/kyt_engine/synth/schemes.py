from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

LICIT = "licit"
ILLICIT = "illicit"


@dataclass
class Scheme:
    """A planted subgraph pattern; nodes carry role/class, edges are local indices."""

    name: str
    nodes: list[dict[str, str]]
    edges: list[tuple[int, int]]


def _counts(rng: np.random.Generator, lo: int, hi: int) -> int:
    return int(rng.integers(lo, hi + 1))


def _range(params: dict[str, Any], key: str, default: tuple[int, int]) -> tuple[int, int]:
    lo, hi = params.get(key, default)
    return int(lo), int(hi)


def build_mixer(rng: np.random.Generator, params: dict[str, Any]) -> Scheme:
    fan_lo, fan_hi = _range(params, "fan_in_range", (5, 25))
    fan_in = _counts(rng, fan_lo, fan_hi)
    fan_lo, fan_hi = _range(params, "fan_out_range", (5, 25))
    fan_out = _counts(rng, fan_lo, fan_hi)
    nodes: list[dict[str, str]] = [{"role": "mixer_core", "cls": ILLICIT}]
    edges: list[tuple[int, int]] = []
    for i in range(fan_in):
        nodes.append({"role": "mixer_entry", "cls": LICIT})
        edges.append((i + 1, 0))
    for i in range(fan_out):
        nodes.append({"role": "mixer_exit", "cls": LICIT})
        edges.append((0, fan_in + 1 + i))
    return Scheme("mixer", nodes, edges)


def build_peel_chain(rng: np.random.Generator, params: dict[str, Any]) -> Scheme:
    lo, hi = _range(params, "length_range", (4, 10))
    length = _counts(rng, lo, hi)
    nodes = [
        {"role": f"peel_hop_{i}", "cls": ILLICIT if 0 < i < length - 1 else LICIT}
        for i in range(length)
    ]
    edges = [(i, i + 1) for i in range(length - 1)]
    return Scheme("peel_chain", nodes, edges)


def _star(
    rng: np.random.Generator,
    params: dict[str, Any],
    name: str,
    hub_role: str,
    hub_cls: str,
    spoke_role: str,
    spoke_cls: str,
    count_key: str,
    count_range: tuple[int, int],
) -> Scheme:
    lo, hi = _range(params, count_key, count_range)
    spokes = _counts(rng, lo, hi)
    nodes: list[dict[str, str]] = [{"role": hub_role, "cls": hub_cls}]
    nodes += [{"role": spoke_role, "cls": spoke_cls} for _ in range(spokes)]
    edges = [(0, i) for i in range(1, spokes + 1)]
    return Scheme(name, nodes, edges)


def build_fanout(rng: np.random.Generator, params: dict[str, Any]) -> Scheme:
    return _star(rng, params, "fanout", "scam", ILLICIT, "victim", LICIT, "victims_range", (5, 40))


def build_hub_spoke(rng: np.random.Generator, params: dict[str, Any]) -> Scheme:
    return _star(rng, params, "hub_spoke", "hub", LICIT, "spoke", LICIT, "spokes_range", (10, 40))


def build_wash(rng: np.random.Generator, params: dict[str, Any]) -> Scheme:
    lo, hi = _range(params, "cycle_len_range", (4, 8))
    length = _counts(rng, lo, hi)
    nodes = [{"role": f"wash_node_{i}", "cls": ILLICIT} for i in range(length)]
    edges = [(i, (i + 1) % length) for i in range(length)]
    return Scheme("wash", nodes, edges)


BUILDERS: dict[str, Callable[[np.random.Generator, dict[str, Any]], Scheme]] = {
    "mixer": build_mixer,
    "peel_chain": build_peel_chain,
    "fanout": build_fanout,
    "hub_spoke": build_hub_spoke,
    "wash": build_wash,
}
