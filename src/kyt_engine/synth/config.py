from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DriftConfig:
    enabled: bool = False
    shutdown_step: int = 40
    shutdown_rate_multiplier: float = 0.0
    novel_step: int = 45
    novel_scheme: str = "wash"


DEFAULT_EDGE_AMOUNTS: dict[str, Any] = {
    "mixer": {"fee_fraction": [0.005, 0.02], "amount_per_in": [10000.0, 100000.0]},
    "peel_chain": {"start_amount": [5000.0, 50000.0], "drip": [0.50, 0.75]},
    "wash": {"amount": [1000.0, 20000.0], "balance_tolerance": 0.05},
    "hub_spoke": {"amount": [50.0, 5000.0]},
    "fanout": {"amount": [50.0, 5000.0]},
    "p2p": {"amount": [1.0, 1000.0]},
}

DEFAULT_ENTITY_TYPES: dict[str, str] = {
    "mixer": "wallet_mixer",
    "peel_chain": "tumbler",
    "fanout": "scam_propagator",
    "hub_spoke": "exchange_hub",
    "wash": "self_cycle",
}


@dataclass
class EdgeAttributeConfig:
    amount: dict[str, Any] = field(default_factory=lambda: copy.deepcopy(DEFAULT_EDGE_AMOUNTS))
    timestamp_scale: int = 1200


@dataclass
class AnchorsConfig:
    per_1000_nodes: int = 2
    min_anchor_distance: int = 3
    entity_types: dict[str, str] = field(
        default_factory=lambda: copy.deepcopy(DEFAULT_ENTITY_TYPES)
    )
    k_hop_range: list[int] = field(default_factory=lambda: [1, 3])


@dataclass
class BackgroundConfig:
    decoy_detection: bool = True
    fan_in_threshold: int = 3
    cycle_max_depth: int = 4


@dataclass
class HoldoutConfig:
    # each entry: {pattern, step_min, step_max, train_excluded}; empty => no windows
    entries: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class GeneratorConfig:
    seed: int = 42
    n_txs: int = 10_000
    labeled_ratio: float = 0.23
    illicit_ratio_in_labeled: float = 0.10
    stats_dir: Path = Path("data/elliptic_stats")
    out_dir: Path = Path("data/synthetic/run")
    schemes: dict[str, dict[str, Any]] = field(default_factory=dict)
    p2p_edges_per_tx: float = 1.5
    drift: DriftConfig = field(default_factory=DriftConfig)
    edge_attributes: EdgeAttributeConfig = field(default_factory=EdgeAttributeConfig)
    anchors: AnchorsConfig = field(default_factory=AnchorsConfig)
    background: BackgroundConfig = field(default_factory=BackgroundConfig)
    holdout: HoldoutConfig = field(default_factory=HoldoutConfig)

    @classmethod
    def from_yaml(cls, path: Path) -> GeneratorConfig:
        raw = yaml.safe_load(Path(path).read_text()) or {}
        defaults = cls()
        drift = raw.get("drift") or {}
        edges = raw.get("edge_attributes") or {}
        anchors = raw.get("anchors") or {}
        background = raw.get("background") or {}
        holdout = raw.get("holdout") or {}
        amount = copy.deepcopy(DEFAULT_EDGE_AMOUNTS)
        for key, value in (edges.get("amount") or {}).items():
            if isinstance(value, dict) and isinstance(amount.get(key), dict):
                amount[key] = {**amount[key], **value}
            else:
                amount[key] = value
        entity_types = copy.deepcopy(DEFAULT_ENTITY_TYPES)
        entity_types.update(anchors.get("entity_types") or {})
        return cls(
            seed=int(raw.get("seed", defaults.seed)),
            n_txs=int(raw.get("n_txs", defaults.n_txs)),
            labeled_ratio=float(raw.get("labeled_ratio", defaults.labeled_ratio)),
            illicit_ratio_in_labeled=float(
                raw.get("illicit_ratio_in_labeled", defaults.illicit_ratio_in_labeled)
            ),
            stats_dir=Path(raw.get("stats_dir", defaults.stats_dir)),
            out_dir=Path(raw.get("out_dir", defaults.out_dir)),
            schemes=raw.get("schemes") or {},
            p2p_edges_per_tx=float(raw.get("p2p_edges_per_tx", defaults.p2p_edges_per_tx)),
            drift=DriftConfig(
                enabled=bool(drift.get("enabled", defaults.drift.enabled)),
                shutdown_step=int(drift.get("shutdown_step", defaults.drift.shutdown_step)),
                shutdown_rate_multiplier=float(
                    drift.get("shutdown_rate_multiplier", defaults.drift.shutdown_rate_multiplier)
                ),
                novel_step=int(drift.get("novel_step", defaults.drift.novel_step)),
                novel_scheme=str(drift.get("novel_scheme", defaults.drift.novel_scheme)),
            ),
            edge_attributes=EdgeAttributeConfig(
                amount=amount,
                timestamp_scale=int(
                    edges.get("timestamp_scale", defaults.edge_attributes.timestamp_scale)
                ),
            ),
            anchors=AnchorsConfig(
                per_1000_nodes=int(anchors.get("per_1000_nodes", defaults.anchors.per_1000_nodes)),
                min_anchor_distance=int(
                    anchors.get("min_anchor_distance", defaults.anchors.min_anchor_distance)
                ),
                entity_types=entity_types,
                k_hop_range=list(anchors.get("k_hop_range", defaults.anchors.k_hop_range)),
            ),
            background=BackgroundConfig(
                decoy_detection=bool(
                    background.get("decoy_detection", defaults.background.decoy_detection)
                ),
                fan_in_threshold=int(
                    background.get("fan_in_threshold", defaults.background.fan_in_threshold)
                ),
                cycle_max_depth=int(
                    background.get("cycle_max_depth", defaults.background.cycle_max_depth)
                ),
            ),
            holdout=HoldoutConfig(entries=list(holdout.get("entries") or [])),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "n_txs": self.n_txs,
            "labeled_ratio": self.labeled_ratio,
            "illicit_ratio_in_labeled": self.illicit_ratio_in_labeled,
            "stats_dir": str(self.stats_dir),
            "schemes": self.schemes,
            "p2p_edges_per_tx": self.p2p_edges_per_tx,
            "drift": {
                "enabled": self.drift.enabled,
                "shutdown_step": self.drift.shutdown_step,
                "shutdown_rate_multiplier": self.drift.shutdown_rate_multiplier,
                "novel_step": self.drift.novel_step,
                "novel_scheme": self.drift.novel_scheme,
            },
            "edge_attributes": {
                "amount": self.edge_attributes.amount,
                "timestamp_scale": self.edge_attributes.timestamp_scale,
            },
            "anchors": {
                "per_1000_nodes": self.anchors.per_1000_nodes,
                "min_anchor_distance": self.anchors.min_anchor_distance,
                "entity_types": self.anchors.entity_types,
                "k_hop_range": self.anchors.k_hop_range,
            },
            "background": {
                "decoy_detection": self.background.decoy_detection,
                "fan_in_threshold": self.background.fan_in_threshold,
                "cycle_max_depth": self.background.cycle_max_depth,
            },
            "holdout": {"entries": self.holdout.entries},
        }
