import copy
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


class _UniqueKeyLoader(yaml.SafeLoader):
    pass


def _construct_unique_mapping(
    loader: _UniqueKeyLoader, node: Any, deep: bool = False
) -> dict[Any, Any]:
    loader.flatten_mapping(node)
    mapping: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            raise ValueError(f"duplicate YAML key: {key}")
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_unique_mapping
)

_KNOWN_PROFILE_IDS = {
    "ordinary_wallet",
    "exchange_hub",
    "miner_payout",
    "wallet_provider",
    "individual",
    "mixer",
    "wash_round_trip",
}


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping")
    return value


def _optional_mapping(data: dict[str, Any], key: str, name: str) -> dict[str, Any]:
    if key in data and data[key] is None:
        raise ValueError(f"{name} must not be null")
    return _mapping(data.get(key), name)


def _strict_bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean")
    return value


def _strict_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    return value


def _strict_float(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _manifest_path(value: Path) -> str:
    return value.name if value.is_absolute() else str(value)


@dataclass(frozen=True)
class ProfileQuota:
    entity_count: int
    tx_quota: int


@dataclass
class BehaviorConfig:
    contract_version: str = "behavior.p0.v1"
    profile_quotas: dict[str, ProfileQuota] = field(
        default_factory=lambda: {
            "ordinary_wallet": ProfileQuota(120, 4_000),
            "exchange_hub": ProfileQuota(12, 1_200),
            "miner_payout": ProfileQuota(20, 800),
            "wallet_provider": ProfileQuota(30, 1_000),
            "individual": ProfileQuota(180, 1_800),
            "mixer": ProfileQuota(8, 600),
            "wash_round_trip": ProfileQuota(6, 600),
        }
    )
    chains: dict[str, dict[str, bool]] = field(
        default_factory=lambda: {
            "bitcoin": {"enabled": True},
            "ethereum": {"enabled": True},
            "tron": {"enabled": True},
        }
    )
    drift: dict[str, Any] = field(
        default_factory=lambda: {
            "phases": [
                {
                    "phase_id": "stable",
                    "step_min": 1,
                    "step_max": 39,
                    "illicit_keep_probability": 1.0,
                    "novel_profile_ids": [],
                },
                {
                    "phase_id": "shutdown",
                    "step_min": 40,
                    "step_max": 44,
                    "illicit_keep_probability": 0.0,
                    "novel_profile_ids": [],
                },
                {
                    "phase_id": "novel",
                    "step_min": 45,
                    "step_max": 49,
                    "illicit_keep_probability": 0.0,
                    "novel_profile_ids": ["wash_round_trip"],
                },
            ]
        }
    )


@dataclass(frozen=True)
class GraphConfig:
    mode: str = "behavior"


@dataclass(frozen=True)
class FeatureConfig:
    mode: str = "semantic"


@dataclass(frozen=True)
class EdgeAttributeConfig:
    timestamp_scale: int = 1200


@dataclass(frozen=True)
class AnchorsConfig:
    per_1000_nodes: int = 2
    min_anchor_distance: int = 3
    k_hop_range: tuple[int, ...] = (1, 3)
    prefer_central_anchors: bool = True


@dataclass(frozen=True)
class HoldoutConfig:
    entries: tuple[dict[str, Any], ...] = ()


@dataclass(frozen=True)
class GNNConfig:
    backend: str = "auto"
    hidden: int = 64
    layers: int = 2
    dropout: float = 0.25
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 300
    patience: int = 30
    threads: int = 1
    k: int = 100
    seeds: tuple[int, ...] = tuple(range(10))


def _parse_behavior(raw: Any, default: BehaviorConfig) -> BehaviorConfig:
    data = _mapping(raw, "behavior")
    allowed = {"contract_version", "profile_quotas", "chains", "drift"}
    unknown = set(data) - allowed
    if unknown:
        raise ValueError(f"unknown behavior fields: {sorted(unknown)}")

    contract_version = data.get("contract_version", default.contract_version)
    if contract_version != "behavior.p0.v1":
        raise ValueError("unsupported behavior contract version")

    quotas_raw = data.get("profile_quotas")
    if quotas_raw is None:
        quotas = copy.deepcopy(default.profile_quotas)
    else:
        quota_data = _mapping(quotas_raw, "behavior.profile_quotas")
        unknown_profiles = set(quota_data) - _KNOWN_PROFILE_IDS
        if unknown_profiles:
            raise ValueError(f"unknown behavior profiles: {sorted(unknown_profiles)}")
        quotas = {}
        for profile_id, values in quota_data.items():
            payload = _mapping(values, f"behavior.profile_quotas.{profile_id}")
            if set(payload) != {"entity_count", "tx_quota"}:
                raise ValueError(f"behavior quota fields are invalid: {profile_id}")
            entity_count = _strict_int(payload["entity_count"], f"{profile_id}.entity_count")
            tx_quota = _strict_int(payload["tx_quota"], f"{profile_id}.tx_quota")
            if entity_count <= 0 or tx_quota <= 0:
                raise ValueError(f"behavior quota for {profile_id} must be positive")
            quotas[profile_id] = ProfileQuota(entity_count, tx_quota)

    chains_raw = data.get("chains")
    if chains_raw is None:
        chains = copy.deepcopy(default.chains)
    else:
        chain_data = _mapping(chains_raw, "behavior.chains")
        if set(chain_data) != {"bitcoin", "ethereum", "tron"}:
            raise ValueError("behavior.chains must contain bitcoin, ethereum and tron")
        chains = {}
        for chain_id, values in chain_data.items():
            payload = _mapping(values, f"behavior.chains.{chain_id}")
            if set(payload) != {"enabled"}:
                raise ValueError(f"invalid behavior chain settings: {chain_id}")
            chains[chain_id] = {"enabled": _strict_bool(payload["enabled"], f"{chain_id}.enabled")}
        if not any(value["enabled"] for value in chains.values()):
            raise ValueError("at least one behavior chain must be enabled")

    drift_raw = data.get("drift")
    if drift_raw is None:
        drift = copy.deepcopy(default.drift)
    else:
        drift = _mapping(drift_raw, "behavior.drift")
        if set(drift) != {"phases"}:
            raise ValueError("behavior.drift must contain phases")
        phases = drift["phases"]
        if not isinstance(phases, list) or not phases:
            raise ValueError("behavior.drift.phases must be a non-empty list")
        for phase in phases:
            required = {
                "phase_id",
                "step_min",
                "step_max",
                "illicit_keep_probability",
                "novel_profile_ids",
            }
            if not isinstance(phase, dict) or set(phase) != required:
                raise ValueError("invalid behavior drift phase fields")
            if not isinstance(phase["phase_id"], str) or not phase["phase_id"]:
                raise ValueError("behavior drift phase_id must be a non-empty string")
            step_min = _strict_int(phase["step_min"], "behavior.drift.step_min")
            step_max = _strict_int(phase["step_max"], "behavior.drift.step_max")
            if not 1 <= step_min <= step_max <= 49:
                raise ValueError("behavior drift phase bounds must be within 1..49")
            probability = _strict_float(
                phase["illicit_keep_probability"], "behavior.drift.illicit_keep_probability"
            )
            if not 0 <= probability <= 1:
                raise ValueError("behavior drift probability must be in [0, 1]")
            novel_ids = phase["novel_profile_ids"]
            if not isinstance(novel_ids, list) or not all(
                isinstance(profile_id, str) and profile_id in _KNOWN_PROFILE_IDS
                for profile_id in novel_ids
            ):
                raise ValueError("behavior drift novel_profile_ids must be known profiles")
        drift = {"phases": copy.deepcopy(phases)}

    return BehaviorConfig(contract_version, quotas, chains, drift)


def _parse_gnn(raw: Any, default: GNNConfig) -> GNNConfig:
    data = _mapping(raw, "gnn")
    allowed = {
        "backend",
        "hidden",
        "layers",
        "dropout",
        "learning_rate",
        "weight_decay",
        "epochs",
        "patience",
        "threads",
        "k",
        "seeds",
    }
    unknown = set(data) - allowed
    if unknown:
        raise ValueError(f"unknown gnn fields: {sorted(unknown)}")
    seeds_raw = data.get("seeds", list(default.seeds))
    if not isinstance(seeds_raw, list):
        raise ValueError("gnn.seeds must be a list")
    parsed = GNNConfig(
        backend=data.get("backend", default.backend),
        hidden=_strict_int(data.get("hidden", default.hidden), "gnn.hidden"),
        layers=_strict_int(data.get("layers", default.layers), "gnn.layers"),
        dropout=_strict_float(data.get("dropout", default.dropout), "gnn.dropout"),
        learning_rate=_strict_float(
            data.get("learning_rate", default.learning_rate), "gnn.learning_rate"
        ),
        weight_decay=_strict_float(
            data.get("weight_decay", default.weight_decay), "gnn.weight_decay"
        ),
        epochs=_strict_int(data.get("epochs", default.epochs), "gnn.epochs"),
        patience=_strict_int(data.get("patience", default.patience), "gnn.patience"),
        threads=_strict_int(data.get("threads", default.threads), "gnn.threads"),
        k=_strict_int(data.get("k", default.k), "gnn.k"),
        seeds=tuple(_strict_int(seed, "gnn.seeds") for seed in seeds_raw),
    )
    if parsed.backend not in {"auto", "full", "sgc"}:
        raise ValueError("gnn.backend must be auto, full or sgc")
    if not parsed.seeds or len(set(parsed.seeds)) != len(parsed.seeds):
        raise ValueError("gnn.seeds must be non-empty and unique")
    if (
        parsed.hidden <= 0
        or parsed.k <= 0
        or parsed.epochs <= 0
        or parsed.patience <= 0
        or parsed.threads <= 0
    ):
        raise ValueError("gnn dimensions, epochs, threads and k must be positive")
    if parsed.layers not in {2, 3} or parsed.hidden % 2:
        raise ValueError("gnn.layers must be 2 or 3 and hidden must be even")
    if not 0 <= parsed.dropout < 1 or parsed.learning_rate <= 0 or parsed.weight_decay < 0:
        raise ValueError("invalid gnn optimization parameters")
    return parsed


def _parse_holdout(raw: Any) -> tuple[dict[str, Any], ...]:
    if raw is None:
        return ()
    if not isinstance(raw, list):
        raise ValueError("holdout.entries must be a list")
    required = {"pattern", "step_min", "step_max", "train_excluded"}
    entries = []
    for index, entry in enumerate(raw):
        if not isinstance(entry, dict) or set(entry) != required:
            raise ValueError(f"holdout.entries[{index}] fields are invalid")
        if entry["pattern"] not in _KNOWN_PROFILE_IDS:
            raise ValueError(f"holdout.entries[{index}].pattern is unknown")
        step_min = _strict_int(entry["step_min"], f"holdout.entries[{index}].step_min")
        step_max = _strict_int(entry["step_max"], f"holdout.entries[{index}].step_max")
        if not 1 <= step_min <= step_max <= 49:
            raise ValueError(f"holdout.entries[{index}] bounds are invalid")
        train_excluded = _strict_bool(
            entry["train_excluded"], f"holdout.entries[{index}].train_excluded"
        )
        entries.append(
            {
                "pattern": entry["pattern"],
                "step_min": step_min,
                "step_max": step_max,
                "train_excluded": train_excluded,
            }
        )
    return tuple(entries)


def _parse_anchors(raw: Any, default: AnchorsConfig) -> AnchorsConfig:
    data = _mapping(raw, "anchors")
    allowed = {"per_1000_nodes", "min_anchor_distance", "k_hop_range", "prefer_central_anchors"}
    unknown = set(data) - allowed
    if unknown:
        raise ValueError(f"unknown anchors fields: {sorted(unknown)}")
    per_1000_nodes = _strict_int(
        data.get("per_1000_nodes", default.per_1000_nodes), "anchors.per_1000_nodes"
    )
    min_anchor_distance = _strict_int(
        data.get("min_anchor_distance", default.min_anchor_distance),
        "anchors.min_anchor_distance",
    )
    if per_1000_nodes < 0 or min_anchor_distance < 0:
        raise ValueError("anchor counts must be non-negative")
    hops_raw = data.get("k_hop_range", list(default.k_hop_range))
    if not isinstance(hops_raw, list) or not all(
        isinstance(value, int) and not isinstance(value, bool) and value >= 0 for value in hops_raw
    ):
        raise ValueError("anchors.k_hop_range must contain non-negative integers")
    prefer_central = _strict_bool(
        data.get("prefer_central_anchors", default.prefer_central_anchors),
        "anchors.prefer_central_anchors",
    )
    return AnchorsConfig(per_1000_nodes, min_anchor_distance, tuple(hops_raw), prefer_central)


@dataclass
class GeneratorConfig:
    seed: int = 72
    n_txs: int = 10_000
    stats_dir: Path = Path("data/volume")
    out_dir: Path = Path("data/synthetic/behavior_run")
    graph: GraphConfig = field(default_factory=GraphConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    behavior: BehaviorConfig = field(default_factory=BehaviorConfig)
    edge_attributes: EdgeAttributeConfig = field(default_factory=EdgeAttributeConfig)
    anchors: AnchorsConfig = field(default_factory=AnchorsConfig)
    holdout: HoldoutConfig = field(default_factory=HoldoutConfig)
    gnn: GNNConfig = field(default_factory=GNNConfig)

    @classmethod
    def from_yaml(cls, path: Path) -> "GeneratorConfig":
        raw = yaml.load(Path(path).read_text(encoding="utf-8"), Loader=_UniqueKeyLoader)
        if raw is None:
            raw = {}
        raw = _mapping(raw, "config")
        defaults = cls()
        allowed = {
            "seed",
            "n_txs",
            "stats_dir",
            "out_dir",
            "graph",
            "features",
            "behavior",
            "edge_attributes",
            "anchors",
            "holdout",
            "gnn",
        }
        unknown = set(raw) - allowed
        if unknown:
            raise ValueError(f"unknown top-level config fields: {sorted(unknown)}")

        seed = _strict_int(raw.get("seed", defaults.seed), "seed")
        n_txs = _strict_int(raw.get("n_txs", defaults.n_txs), "n_txs")
        if n_txs <= 0:
            raise ValueError("n_txs must be positive")

        stats_raw = raw.get("stats_dir", defaults.stats_dir)
        out_raw = raw.get("out_dir", defaults.out_dir)
        if not isinstance(stats_raw, (str, Path)) or not isinstance(out_raw, (str, Path)):
            raise ValueError("stats_dir and out_dir must be paths")

        graph = _optional_mapping(raw, "graph", "graph")
        unknown_graph = set(graph) - {"mode"}
        if unknown_graph:
            raise ValueError(f"unknown graph fields: {sorted(unknown_graph)}")
        graph_mode = graph.get("mode", "behavior")
        if graph_mode != "behavior":
            raise ValueError("this project supports graph.mode=behavior only")

        features = _optional_mapping(raw, "features", "features")
        if set(features) - {"mode"}:
            raise ValueError("unknown features fields")
        if features.get("mode", "semantic") != "semantic":
            raise ValueError("this project supports features.mode=semantic only")

        behavior_raw = raw.get("behavior")
        if behavior_raw is None and "behavior" in raw:
            raise ValueError("behavior must not be null")
        behavior = _parse_behavior(behavior_raw, defaults.behavior)
        if sum(item.tx_quota for item in behavior.profile_quotas.values()) != n_txs:
            raise ValueError("behavior profile quotas must sum to n_txs")

        edge_raw = _optional_mapping(raw, "edge_attributes", "edge_attributes")
        if set(edge_raw) - {"timestamp_scale"}:
            raise ValueError("unknown edge_attributes fields")
        timestamp_scale = _strict_int(
            edge_raw.get("timestamp_scale", defaults.edge_attributes.timestamp_scale),
            "edge_attributes.timestamp_scale",
        )
        if timestamp_scale < 0:
            raise ValueError("edge_attributes.timestamp_scale must be non-negative")

        anchors = _parse_anchors(raw.get("anchors"), defaults.anchors)
        holdout_raw = _optional_mapping(raw, "holdout", "holdout")
        if set(holdout_raw) - {"entries"}:
            raise ValueError("unknown holdout fields")
        holdout = HoldoutConfig(_parse_holdout(holdout_raw.get("entries")))
        gnn = _parse_gnn(raw.get("gnn"), defaults.gnn)

        return cls(
            seed=seed,
            n_txs=n_txs,
            stats_dir=Path(stats_raw),
            out_dir=Path(out_raw),
            graph=GraphConfig("behavior"),
            features=FeatureConfig("semantic"),
            behavior=behavior,
            edge_attributes=EdgeAttributeConfig(timestamp_scale),
            anchors=anchors,
            holdout=holdout,
            gnn=gnn,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "n_txs": self.n_txs,
            "stats_dir": _manifest_path(self.stats_dir),
            "out_dir": _manifest_path(self.out_dir),
            "graph": {"mode": "behavior"},
            "features": {"mode": "semantic"},
            "behavior": {
                "contract_version": self.behavior.contract_version,
                "profile_quotas": {
                    key: {
                        "entity_count": value.entity_count,
                        "tx_quota": value.tx_quota,
                    }
                    for key, value in sorted(self.behavior.profile_quotas.items())
                },
                "chains": {key: dict(value) for key, value in sorted(self.behavior.chains.items())},
                "drift": copy.deepcopy(self.behavior.drift),
            },
            "edge_attributes": {"timestamp_scale": self.edge_attributes.timestamp_scale},
            "anchors": {
                "per_1000_nodes": self.anchors.per_1000_nodes,
                "min_anchor_distance": self.anchors.min_anchor_distance,
                "k_hop_range": list(self.anchors.k_hop_range),
                "prefer_central_anchors": self.anchors.prefer_central_anchors,
            },
            "holdout": {"entries": [dict(entry) for entry in self.holdout.entries]},
            "gnn": {
                "backend": self.gnn.backend,
                "hidden": self.gnn.hidden,
                "layers": self.gnn.layers,
                "dropout": self.gnn.dropout,
                "learning_rate": self.gnn.learning_rate,
                "weight_decay": self.gnn.weight_decay,
                "epochs": self.gnn.epochs,
                "patience": self.gnn.patience,
                "threads": self.gnn.threads,
                "k": self.gnn.k,
                "seeds": list(self.gnn.seeds),
            },
        }


__all__ = [
    "AnchorsConfig",
    "BehaviorConfig",
    "EdgeAttributeConfig",
    "FeatureConfig",
    "GNNConfig",
    "GeneratorConfig",
    "GraphConfig",
    "HoldoutConfig",
    "ProfileQuota",
]
