from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .anchors import (
    apply_holdout,
    attr_rng,
    build_edge_attributes,
    build_instances,
    detect_decoys,
    instance_id_of,
    k_hop_neighborhoods,
    select_anchors,
)
from .config import GeneratorConfig
from .graph import GeneratedGraph
from .stats import CLASS_TO_LABEL, N_FEATURES


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_dataset(
    config: GeneratorConfig,
    graph: GeneratedGraph,
    features: np.ndarray,
    out_dir: Path,
) -> None:
    """Emit the CSVs in Elliptic++ format + edge attributes + self-describing manifest.json."""
    out_dir.mkdir(parents=True, exist_ok=True)

    cols = ["txId", "time_step"] + [f"feat_{i}" for i in range(2, 2 + N_FEATURES)]
    feats = pd.DataFrame(features, columns=cols)
    feats["time_step"] = feats["time_step"].astype(int)
    feats["txId"] = feats["txId"].astype(int)
    feats.to_csv(out_dir / "elliptic_txs_features.csv", header=False, index=False)

    classes = pd.DataFrame(
        {
            "txId": [nd.tx_id for nd in graph.nodes],
            "class": [CLASS_TO_LABEL[nd.cls] for nd in graph.nodes],
        }
    )
    classes.to_csv(out_dir / "elliptic_txs_classes.csv", index=False)

    edge_df = pd.DataFrame(graph.edges, columns=["txId1", "txId2"])
    edge_df.to_csv(out_dir / "elliptic_txs_edgelist.csv", index=False)

    instances = build_instances(graph, config)
    select_anchors(instances, graph, config)
    windows_active = apply_holdout(instances, config)
    decoys = detect_decoys(graph, config)
    id_of = instance_id_of(graph, instances)
    edge_attrs = build_edge_attributes(graph, instances, config, attr_rng(config.seed))

    attrs_df = pd.DataFrame(edge_attrs)  # indexes == edgelist row numbers
    attrs_df.to_csv(out_dir / "elliptic_txs_edge_attributes.csv", index=False)

    ground_truth = [
        {
            "txId": nd.tx_id,
            "scheme": nd.scheme,
            "role": nd.role,
            "class": nd.cls,
            "instance_id": id_of[nd.tx_id],
        }
        for nd in graph.nodes
    ]

    config_dict = config.to_dict()
    instance_gt = [_instance_entry(inst) for inst in instances]
    edge_attr_gt = []
    for inst in instances:
        for e in inst.edge_ids:
            entry = edge_attrs[e]
            edge_attr_gt.append(
                {
                    "edge_id": e,
                    "txId1": entry["txId1"],
                    "txId2": entry["txId2"],
                    "pattern_type": inst.pattern_type,
                    "instance_id": inst.instance_id,
                    "role": inst.edge_attr_recipe[e],
                    "amount": entry["amount"],
                }
            )
    anchors = [
        {
            "instance_id": inst.instance_id,
            "anchor_node_id": inst.anchor_node_id,
            "anchor_role": inst.anchor_role,
            "entity_type": inst.entity_type,
            "pattern_type": inst.pattern_type,
            "max_anchor_distance": inst.max_anchor_distance,
            "k_hop_neighborhoods": k_hop_neighborhoods(inst, graph, config),
        }
        for inst in instances
        if inst.anchored
    ]

    manifest = {
        "seed": config.seed,
        "config": config_dict,
        "sha256": {
            "features": _sha256(out_dir / "elliptic_txs_features.csv"),
            "classes": _sha256(out_dir / "elliptic_txs_classes.csv"),
            "edgelist": _sha256(out_dir / "elliptic_txs_edgelist.csv"),
            "edge_attributes": _sha256(out_dir / "elliptic_txs_edge_attributes.csv"),
        },
        "ground_truth": ground_truth,
        "instance_ground_truth": instance_gt,
        "edge_attribute_ground_truth": edge_attr_gt,
        "anchor_registry": {
            "per_1000_nodes": config.anchors.per_1000_nodes,
            "min_anchor_distance": config.anchors.min_anchor_distance,
            "anchors": anchors,
        },
        "retrieval_specs": {
            "node_vector_dim": 2 + N_FEATURES,
            "vectors_file": "elliptic_txs_features.csv",
            "node_id_column": "txId",
            "node_vector_lookup": "row = first (headerless) column value, txId == node id",
            "anchor_key": "anchor_node_id",
            "entity_type_key": "entity_type",
            "neighborhood_field": "k_hop_neighborhoods",
            "default_k": max(config.anchors.k_hop_range or [0]),
        },
        "decoys": decoys,
        "holdout": {"windows_active": windows_active, "entries": config.holdout.entries},
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def _instance_entry(inst) -> dict:
    return {
        "instance_id": inst.instance_id,
        "pattern_type": inst.pattern_type,
        "anchor_node_id": inst.anchor_node_id,
        "anchor_role": inst.anchor_role,
        "entity_type": inst.entity_type,
        "max_anchor_distance": inst.max_anchor_distance,
        "temporal_window": list(inst.temporal_window),
        "node_ids": inst.node_ids,
        "edge_ids": inst.edge_ids,
        "articulation_points": inst.articulation_points,
        "anchored": inst.anchored,
        "holdout": inst.holdout,
        "train_excluded": inst.train_excluded,
    }
