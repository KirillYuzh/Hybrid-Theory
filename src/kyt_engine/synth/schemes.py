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


def build_structuring(rng: np.random.Generator, params: dict[str, Any]) -> Scheme:
    """Classic smurfing: one structurer fans out micro-deposits (below-threshold)."""
    lo, hi = _range(params, "deposits_range", (5, 30))
    deposits = _counts(rng, lo, hi)
    nodes: list[dict[str, str]] = [{"role": "structurer", "cls": ILLICIT}]
    nodes += [{"role": f"smurf_{i}", "cls": ILLICIT} for i in range(deposits)]
    edges = [(0, i + 1) for i in range(deposits)]
    return Scheme("structuring", nodes, edges)


def build_cycle_round_trip(rng: np.random.Generator, params: dict[str, Any]) -> Scheme:
    """Round-tripping: funds return to the originator through a closed cycle."""
    lo, hi = _range(params, "hops_range", (3, 6))
    hops = _counts(rng, lo, hi)
    nodes = [{"role": f"round_trip_{i}", "cls": ILLICIT} for i in range(hops)]
    edges = [(i, (i + 1) % hops) for i in range(hops)]
    return Scheme("cycle_round_trip", nodes, edges)


def build_bridge_hopping(rng: np.random.Generator, params: dict[str, Any]) -> Scheme:
    """Bridge-hopping: source -> bridge -> target chain, repeated across bridges.

    Topology is a path: entry -> bridge_in -> bridge_out -> ... -> recipient.
    The intermediate bridge hops are the laundering core; entry/exit are ordinary txs.
    """
    lo, hi = _range(params, "bridges_range", (1, 3))
    bridges = _counts(rng, lo, hi)
    nodes: list[dict[str, str]] = [{"role": "bridge_user", "cls": LICIT}]
    for i in range(bridges):
        nodes.append({"role": f"bridge_in_{i}", "cls": ILLICIT})
        nodes.append({"role": f"bridge_out_{i}", "cls": ILLICIT})
    nodes.append({"role": "bridge_recipient", "cls": LICIT})
    edges = [(i, i + 1) for i in range(len(nodes) - 1)]
    return Scheme("bridge_hopping", nodes, edges)


def build_amm_swap_chain(rng: np.random.Generator, params: dict[str, Any]) -> Scheme:
    """Sequential swaps through AMM pools (asset conversion as laundering)."""
    lo, hi = _range(params, "swaps_range", (2, 5))
    swaps = _counts(rng, lo, hi)
    nodes = [{"role": "amm_trader_0", "cls": ILLICIT}]
    for i in range(swaps):
        nodes.append({"role": f"amm_pool_{i}", "cls": ILLICIT})
        nodes.append({"role": f"amm_trader_{i + 1}", "cls": ILLICIT})
    edges = [(i, i + 1) for i in range(len(nodes) - 1)]
    return Scheme("amm_swap_chain", nodes, edges)


def build_exchange_hub(rng: np.random.Generator, params: dict[str, Any]) -> Scheme:
    return _star(
        rng, params, "exchange_hub", "exchange", LICIT, "client", LICIT, "clients_range", (20, 60)
    )


def build_miner_payout(rng: np.random.Generator, params: dict[str, Any]) -> Scheme:
    return _star(
        rng,
        params,
        "miner_payout",
        "miner",
        LICIT,
        "payout_address",
        LICIT,
        "payouts_range",
        (10, 40),
    )


def build_wallet_provider(rng: np.random.Generator, params: dict[str, Any]) -> Scheme:
    return _star(
        rng,
        params,
        "wallet_provider",
        "provider",
        LICIT,
        "custody_address",
        LICIT,
        "addresses_range",
        (10, 50),
    )


def build_stealth_use(rng: np.random.Generator, params: dict[str, Any]) -> Scheme:
    """Stealth-address receivable (modern darknet funds handling).

    One payer sweeps into N one-time stealth addresses; all addresses are controlled
    by the same operator (illicit), so they circle cash past address-based attribution.
    """
    lo, hi = _range(params, "stealth_range", (5, 30))
    addrs = _counts(rng, lo, hi)
    nodes: list[dict[str, str]] = [{"role": "stealth_payer", "cls": ILLICIT}]
    nodes += [{"role": f"stealth_addr_{i}", "cls": ILLICIT} for i in range(addrs)]
    edges = [(0, i + 1) for i in range(addrs)]
    return Scheme("stealth_use", nodes, edges)


def build_lending_laundry(rng: np.random.Generator, params: dict[str, Any]) -> Scheme:
    """Lending-as-laundering: deposit dirty funds as collateral, draw a clean loan,
    repay it, withdraw the collateral. Four directed edges per round between the
    debtor (illicit) and one lending pool (licit): deposit, draw, repay, release.
    The pool's inflows (deposit + repay) match its outflows (draw + release) exactly.
    """
    lo, hi = _range(params, "rounds_range", (1, 3))
    rounds = _counts(rng, lo, hi)
    nodes: list[dict[str, str]] = [{"role": "debtor", "cls": ILLICIT}]
    edges: list[tuple[int, int]] = []
    for r in range(rounds):
        pool = r + 1
        nodes.append({"role": f"lending_pool_{r}", "cls": LICIT})
        edges.extend([(0, pool), (pool, 0), (0, pool), (pool, 0)])
    return Scheme("lending_laundry", nodes, edges)


BUILDERS: dict[str, Callable[[np.random.Generator, dict[str, Any]], Scheme]] = {
    "mixer": build_mixer,
    "peel_chain": build_peel_chain,
    "fanout": build_fanout,
    "hub_spoke": build_hub_spoke,
    "wash": build_wash,
    "structuring": build_structuring,
    "cycle_round_trip": build_cycle_round_trip,
    "bridge_hopping": build_bridge_hopping,
    "amm_swap_chain": build_amm_swap_chain,
    "exchange_hub": build_exchange_hub,
    "miner_payout": build_miner_payout,
    "wallet_provider": build_wallet_provider,
    "stealth_use": build_stealth_use,
    "lending_laundry": build_lending_laundry,
}
