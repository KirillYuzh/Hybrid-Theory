import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from .anchors import (
    apply_holdout,
    instance_id_of,
    k_hop_neighborhoods,
    select_anchors,
)
from .behavior import (
    BehaviorGraph,
    build_behavior_edge_attributes,
    get_action,
    get_chain,
    get_profile,
    project_behavior_instances,
    registry_sha256,
)
from .config import GeneratorConfig
from .features_semantic import CHAIN_SEMANTIC_COLUMNS, SEMANTIC_COLUMNS, build_semantic_features
from .stats import N_FEATURES


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _instance_entry(instance) -> dict:
    return {
        "instance_id": instance.instance_id,
        "pattern_type": instance.pattern_type,
        "anchor_node_id": instance.anchor_node_id,
        "anchor_role": instance.anchor_role,
        "entity_type": instance.entity_type,
        "max_anchor_distance": instance.max_anchor_distance,
        "temporal_window": list(instance.temporal_window),
        "node_ids": instance.node_ids,
        "edge_ids": instance.edge_ids,
        "articulation_points": instance.articulation_points,
        "anchored": instance.anchored,
        "holdout": instance.holdout,
        "train_excluded": instance.train_excluded,
    }


def _synthetic_tx_id(seed: int, tx_id: int, chain_id: str) -> str:
    digest = hashlib.sha256(f"{seed}:behavior:tx:{tx_id}:{chain_id}".encode()).hexdigest()
    return f"0x{digest}" if chain_id == "eip155:1" else digest


def _behavior_manifest(graph: BehaviorGraph, config: GeneratorConfig) -> dict:
    chain_by_id = {get_chain(key)["chain_id"]: key for key in ("bitcoin", "ethereum", "tron")}
    entity_by_id = {entity.entity_id: entity for entity in graph.entities}
    node_profiles = []
    node_records = []
    for node, meta in zip(graph.nodes, graph.node_meta):
        profile = get_profile(node.scheme)
        action = get_action(node.role)
        chain_key = chain_by_id[meta.chain_id]
        entity_ids = list(meta.entity_ids)
        node_profiles.append(
            {
                "txId": node.tx_id,
                "profile_id": node.scheme,
                "entity_type": profile.entity_type,
                "class_policy": profile.class_policy,
                "behavior_params": {
                    "action_id": node.role,
                    "amount_policy_id": action.amount_policy,
                    "profile_amount_policy_id": profile.amount_policy_id,
                    "channel": action.channel,
                    "temporal_pattern": "scheduler",
                },
                "entity_ids": entity_ids,
                "chain_id": meta.chain_id,
            }
        )
        node_records.append(
            {
                "txId": node.tx_id,
                "chain_id": meta.chain_id,
                "chain_key": chain_key,
                "native_tx_id": _synthetic_tx_id(config.seed, node.tx_id, meta.chain_id),
                "native_entity_id": next(
                    entity_by_id[entity_id].native_entity_id
                    for entity_id in meta.entity_ids
                    if entity_by_id[entity_id].chain_id == meta.chain_id
                ),
                "native_id_is_synthetic": True,
            }
        )

    edge_records = [
        {
            "edge_id": edge.edge_id,
            "event_id": edge.event_id,
            "txId1": edge.txId1,
            "txId2": edge.txId2,
            "source_chain_id": graph.node_meta[edge.txId1].chain_id,
            "target_chain_id": graph.node_meta[edge.txId2].chain_id,
            "kind": edge.kind,
            "role": edge.role,
            "csv_amount": edge.amount_minor / 100.0,
            "timestamp": edge.timestamp,
            "native_amount_minor": None,
            "bridge_id": edge.bridge_id,
        }
        for edge in graph.edge_meta
    ]
    edge_by_event = {edge.event_id: edge for edge in graph.edge_meta}
    bridge_records = [
        {
            "bridge_id": event.bridge_id,
            "event_id": event.event_id,
            "edge_id": edge_by_event[event.event_id].edge_id
            if event.event_id in edge_by_event
            else None,
            "source_chain_id": entity_by_id[event.actor_entity_id].chain_id,
            "target_chain_id": event.chain_id,
        }
        for event in graph.events
        if event.bridge_id is not None
    ]

    realized = []
    for phase in config.behavior.drift["phases"]:
        phase_nodes = [
            node for node in graph.nodes if phase["step_min"] <= node.step <= phase["step_max"]
        ]
        phase_ids = {node.tx_id for node in phase_nodes}
        realized.append(
            {
                "phase_id": phase["phase_id"],
                "step_min": phase["step_min"],
                "step_max": phase["step_max"],
                "node_count": len(phase_nodes),
                "class_counts": {
                    value: sum(node.cls == value for node in phase_nodes)
                    for value in ("illicit", "licit", "unknown")
                },
                "profile_counts": {
                    profile_id: sum(node.scheme == profile_id for node in phase_nodes)
                    for profile_id in sorted(config.behavior.profile_quotas)
                },
                "chain_counts": {
                    chain_id: sum(
                        meta.chain_id == chain_id
                        for node, meta in zip(graph.nodes, graph.node_meta)
                        if node.tx_id in phase_ids
                    )
                    for chain_id in sorted(chain_by_id)
                },
            }
        )

    return {
        "contract_version": config.behavior.contract_version,
        "behavioral_profiles": {
            "registry_version": 1,
            "registry_sha256": registry_sha256(),
            "nodes": node_profiles,
        },
        "entities": [asdict(entity) for entity in graph.entities],
        "events": [asdict(event) for event in graph.events],
        "instances": [asdict(instance) for instance in graph.instances],
        "drift_schedule": {
            "schema_version": "1.0.0",
            "configured": config.behavior.drift,
            "realized": realized,
        },
        "chain_metadata": {
            "schema_version": "1.0.0",
            "identity_scope": "synthetic",
            "chains": [
                {"chain_key": key, **get_chain(key)} for key in ("bitcoin", "ethereum", "tron")
            ],
            "node_records": node_records,
            "edge_records": edge_records,
            "bridge_records": bridge_records,
        },
        "sanctions_anchor_map": {
            "schema_version": "1.0.0",
            "status": "provenance_only",
            "real_address_payload_loaded": False,
            "records": [],
        },
        "calibration": {
            "schema_version": "1.0.0",
            "status": "not_evaluated",
            "positive_class": "1",
            "negative_class": "2",
            "unknown_policy": "exclude",
            "score_semantics": "uncalibrated_model_score",
        },
        "rng_contract": {
            "generator": "PCG64",
            "namespaces": [
                "behavior:population",
                "behavior:schedule",
                "behavior:transition",
                "behavior:amount",
                "behavior:drift",
            ],
        },
    }


def write_dataset(config: GeneratorConfig, graph: BehaviorGraph, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    instances = project_behavior_instances(graph, config)
    edge_attrs = build_behavior_edge_attributes(graph, config.edge_attributes.timestamp_scale)
    select_anchors(instances, graph, config)
    apply_holdout(instances, config)
    id_of = instance_id_of(graph, instances)

    features = np.empty((len(graph.nodes), 2 + N_FEATURES), dtype=np.float64)
    features[:, 0] = [node.tx_id for node in graph.nodes]
    features[:, 1] = [node.step for node in graph.nodes]
    features[:, 2:] = build_semantic_features(graph, edge_attrs)
    feature_semantics = {
        "mode": "semantic",
        "column_layout": {str(index): name for index, name in SEMANTIC_COLUMNS.items()},
        "chain_column_layout": {str(index): name for index, name in CHAIN_SEMANTIC_COLUMNS.items()},
        "csv_columns": "feat_{j + 2} == column j above; remaining columns are derived transforms",
        "note": "computed from topology and amounts",
    }

    columns = ["txId", "time_step"] + [f"feat_{index}" for index in range(2, 2 + N_FEATURES)]
    feature_frame = pd.DataFrame(features, columns=columns)
    feature_frame["txId"] = feature_frame["txId"].astype(np.int64)
    feature_frame["time_step"] = feature_frame["time_step"].astype(np.int64)
    feature_frame.to_csv(out_dir / "elliptic_txs_features.csv", header=False, index=False)
    pd.DataFrame(
        {
            "txId": [node.tx_id for node in graph.nodes],
            "class": [
                {"illicit": "1", "licit": "2"}.get(node.cls, "unknown") for node in graph.nodes
            ],
        }
    ).to_csv(out_dir / "elliptic_txs_classes.csv", index=False)
    pd.DataFrame(graph.edges, columns=["txId1", "txId2"]).to_csv(
        out_dir / "elliptic_txs_edgelist.csv", index=False
    )
    attributes = pd.DataFrame(edge_attrs, columns=["txId1", "txId2", "amount", "timestamp"])
    attributes.to_csv(out_dir / "elliptic_txs_edge_attributes.csv", index=False)

    ground_truth = []
    for node in graph.nodes:
        meta = graph.node_meta[node.tx_id]
        ground_truth.append(
            {
                "txId": node.tx_id,
                "scheme": node.scheme,
                "role": node.role,
                "class": node.cls,
                "instance_id": id_of[node.tx_id],
                "event_id": meta.event_id,
                "entity_ids": list(meta.entity_ids),
                "chain_id": meta.chain_id,
            }
        )

    instance_gt = [_instance_entry(instance) for instance in instances]
    behavior_by_id = {instance.instance_id: instance for instance in graph.instances}
    for entry, instance in zip(instance_gt, instances):
        source = behavior_by_id[instance.instance_id]
        entry.update(
            {
                "profile_id": source.profile_id,
                "profile_ids": list(source.profile_ids),
                "entity_ids": list(source.entity_ids),
                "event_ids": list(source.event_ids),
                "chain_path": list(source.chain_path),
            }
        )

    edge_ground_truth = []
    for instance in instances:
        for edge_id in instance.edge_ids:
            edge = graph.edge_meta[edge_id]
            edge_ground_truth.append(
                {
                    "edge_id": edge_id,
                    "txId1": edge.txId1,
                    "txId2": edge.txId2,
                    "pattern_type": instance.pattern_type,
                    "instance_id": instance.instance_id,
                    "role": instance.edge_attr_recipe[edge_id],
                    "amount": edge_attrs[edge_id]["amount"],
                    "event_id": edge.event_id,
                    "chain_id": edge.chain_id,
                    "edge_kind": edge.kind,
                }
            )

    anchors = [
        {
            "instance_id": instance.instance_id,
            "anchor_node_id": instance.anchor_node_id,
            "anchor_role": instance.anchor_role,
            "entity_type": instance.entity_type,
            "pattern_type": instance.pattern_type,
            "max_anchor_distance": instance.max_anchor_distance,
            "k_hop_neighborhoods": k_hop_neighborhoods(instance, graph, config),
        }
        for instance in instances
        if instance.anchored
    ]
    manifest = {
        "seed": config.seed,
        "config": config.to_dict(),
        "sha256": {
            "features": _sha256(out_dir / "elliptic_txs_features.csv"),
            "classes": _sha256(out_dir / "elliptic_txs_classes.csv"),
            "edgelist": _sha256(out_dir / "elliptic_txs_edgelist.csv"),
            "edge_attributes": _sha256(out_dir / "elliptic_txs_edge_attributes.csv"),
        },
        "ground_truth": ground_truth,
        "instance_ground_truth": instance_gt,
        "edge_attribute_ground_truth": edge_ground_truth,
        "anchor_registry": {
            "per_1000_nodes": config.anchors.per_1000_nodes,
            "min_anchor_distance": config.anchors.min_anchor_distance,
            "anchors": anchors,
        },
        "retrieval_specs": {
            "node_vector_dim": 2 + N_FEATURES,
            "vectors_file": "elliptic_txs_features.csv",
            "node_id_column": "txId",
            "node_vector_lookup": "row = first headerless column value, txId == node id",
            "anchor_key": "anchor_node_id",
            "entity_type_key": "entity_type",
            "neighborhood_field": "k_hop_neighborhoods",
            "default_k": max(config.anchors.k_hop_range or (0,)),
            "feature_mode": "semantic",
        },
        "feature_semantics": feature_semantics,
        "decoys": [],
        "holdout": {
            "windows_active": bool(config.holdout.entries),
            "entries": [dict(entry) for entry in config.holdout.entries],
        },
        "behavior": _behavior_manifest(graph, config),
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8"
    )


__all__ = ["write_dataset"]
