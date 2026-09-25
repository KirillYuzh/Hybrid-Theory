import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from kyt_engine.synth.behavior import build_behavior_graph, get_action
from kyt_engine.synth.config import BehaviorConfig, GeneratorConfig, ProfileQuota
from kyt_engine.synth.emit import write_dataset
from kyt_engine.synth.gnn_downstream import (
    aggregate_step_metrics,
    binary_metrics_at_threshold,
    build_edge_index,
    build_label_masks,
    build_partition_masks,
    filter_partition_edges,
    per_step_metrics,
)
from kyt_engine.synth.graph import build_graph
from kyt_engine.synth.stats import N_STEPS, VolumeProfile
from kyt_engine.synth.validate import (
    validate_behavior_dataset,
    validate_dataset,
    validate_edge_attributes,
    validate_semantic_features,
)


@pytest.fixture()
def volume_dir(tmp_path: Path) -> Path:
    path = tmp_path / "volume"
    path.mkdir()
    np.save(path / "volume.npy", np.ones(N_STEPS, dtype=np.int64) * 100)
    return path


def small_config(volume_dir: Path, out_dir: Path) -> GeneratorConfig:
    config = GeneratorConfig(
        seed=72,
        n_txs=1_000,
        stats_dir=volume_dir,
        out_dir=out_dir,
    )
    config.behavior = BehaviorConfig(
        profile_quotas={
            "ordinary_wallet": ProfileQuota(4, 300),
            "exchange_hub": ProfileQuota(2, 150),
            "miner_payout": ProfileQuota(2, 100),
            "wallet_provider": ProfileQuota(2, 100),
            "individual": ProfileQuota(4, 200),
            "mixer": ProfileQuota(2, 100),
            "wash_round_trip": ProfileQuota(2, 50),
        },
        chains={
            "bitcoin": {"enabled": True},
            "ethereum": {"enabled": True},
            "tron": {"enabled": True},
        },
        drift={
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
        },
    )
    return config


def test_behavior_dataset_contract_and_determinism(volume_dir: Path, tmp_path: Path) -> None:
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    config = small_config(volume_dir, first_dir)
    graph = build_graph(config, VolumeProfile.from_dir(volume_dir))
    write_dataset(config, graph, first_dir)

    second = small_config(volume_dir, second_dir)
    second_graph = build_graph(second, VolumeProfile.from_dir(volume_dir))
    write_dataset(second, second_graph, second_dir)

    validate_dataset(first_dir)
    validate_edge_attributes(first_dir)
    validate_semantic_features(first_dir)
    validate_behavior_dataset(first_dir)
    assert len(graph.nodes) == config.n_txs
    assert graph.edges == second_graph.edges
    for name in (
        "elliptic_txs_features.csv",
        "elliptic_txs_classes.csv",
        "elliptic_txs_edgelist.csv",
        "elliptic_txs_edge_attributes.csv",
    ):
        assert (first_dir / name).read_bytes() == (second_dir / name).read_bytes()
    manifest = json.loads((first_dir / "manifest.json").read_text())
    assert manifest["seed"] == 72
    assert manifest["behavior"]["contract_version"] == "behavior.p0.v1"
    assert len(manifest["ground_truth"]) == config.n_txs


def test_behavior_causality_entities_and_bridges(volume_dir: Path, tmp_path: Path) -> None:
    config = small_config(volume_dir, tmp_path / "run")
    graph = build_behavior_graph(config, VolumeProfile.from_dir(volume_dir))
    entity_by_id = {entity.entity_id: entity for entity in graph.entities}
    for event in graph.events:
        action = get_action(event.action_id)
        if event.source_tx_ids:
            assert event.source_entity_ids
            assert event.target_entity_id in event_entity_ids(event)
        if action.edge_kind == "bridge_link":
            actor_chain = entity_by_id[event.actor_entity_id].chain_id
            assert actor_chain != event.chain_id
            assert event.bridge_id == f"{actor_chain}->{event.chain_id}"
    bridge_edges = [edge for edge in graph.edge_meta if edge.bridge_id is not None]
    assert bridge_edges
    assert all(
        graph.node_meta[edge.txId1].chain_id != graph.node_meta[edge.txId2].chain_id
        for edge in bridge_edges
    )


def event_entity_ids(event) -> set[int]:
    return {
        event.actor_entity_id,
        *event.counterparty_entity_ids,
        *event.source_entity_ids,
        event.target_entity_id,
    }


def test_manifest_rejects_bridge_and_provenance_tampering(volume_dir: Path, tmp_path: Path) -> None:
    config = small_config(volume_dir, tmp_path / "run")
    graph = build_graph(config, VolumeProfile.from_dir(volume_dir))
    write_dataset(config, graph, config.out_dir)
    manifest_path = config.out_dir / "manifest.json"
    original = manifest_path.read_bytes()

    invalid = json.loads(original)
    event = next(item for item in invalid["behavior"]["events"] if item["source_tx_ids"])
    event["source_entity_ids"] = [
        next(
            entity["entity_id"]
            for entity in invalid["behavior"]["entities"]
            if entity["entity_id"] != event["source_entity_ids"][0]
        )
    ]
    manifest_path.write_text(json.dumps(invalid), encoding="utf-8")
    with pytest.raises(ValueError, match="source entity"):
        validate_behavior_dataset(config.out_dir)
    manifest_path.write_bytes(original)

    invalid = json.loads(original)
    bridge = next(item for item in invalid["behavior"]["events"] if item["bridge_id"] is not None)
    bridge["bridge_id"] = None
    manifest_path.write_text(json.dumps(invalid), encoding="utf-8")
    with pytest.raises(ValueError, match="bridge event ID"):
        validate_behavior_dataset(config.out_dir)
    manifest_path.write_bytes(original)


def test_behavior_config_is_seed_72_and_semantic_only(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("seed: 72\nn_txs: 10000\n")
    config = GeneratorConfig.from_yaml(path)
    assert config.seed == 72
    assert config.graph.mode == "behavior"
    assert config.features.mode == "semantic"

    path.write_text("features:\n  mode: cdf\n")
    with pytest.raises(ValueError, match="semantic only"):
        GeneratorConfig.from_yaml(path)
    path.write_text("graph:\n  mode: legacy\n")
    with pytest.raises(ValueError, match="behavior only"):
        GeneratorConfig.from_yaml(path)
    path.write_text("schemes:\n  mixer: {}\n")
    with pytest.raises(ValueError, match="unknown top-level"):
        GeneratorConfig.from_yaml(path)


def test_drift_schedule_is_contiguous(volume_dir: Path, tmp_path: Path) -> None:
    config = small_config(volume_dir, tmp_path / "run")
    config.behavior.drift["phases"][1]["step_min"] = 39
    with pytest.raises(ValueError, match="phase|contiguous"):
        build_behavior_graph(config, VolumeProfile.from_dir(volume_dir))


def test_strict_partition_and_metric_helpers() -> None:
    steps = np.array([1, 2, 30, 31, 40, 41, 42, 49, 45, 46, 47, 48])
    ids = np.arange(len(steps))
    masks = build_partition_masks(steps)
    assert masks["train"].sum() == 3
    assert masks["validation"].sum() == 2
    assert masks["test"].sum() == 7
    y, labeled = build_label_masks(
        pd.DataFrame(
            {
                "txId": ids,
                "class": ["1", "2", "unknown", "1", "2", "1", "2", "1", "2", "1", "2", "1"],
            }
        ),
        steps,
    )
    assert labeled.sum() == 11
    assert np.isnan(y[2])
    edges = pd.DataFrame({"txId1": [0, 0, 3], "txId2": [1, 3, 5]})
    retained, report = filter_partition_edges(edges, pd.Series(steps, index=ids))
    assert report["cross_partition_after_filter"] == 0
    assert len(retained["train"]) == 1
    metrics = binary_metrics_at_threshold(
        np.array([1.0, 0.0, 1.0]), np.array([0.9, 0.2, 0.8]), np.array([1, 2, 3]), 0.5, 2
    )
    assert metrics["f1"] == 1.0
    rows = per_step_metrics(y, np.linspace(0.1, 0.9, len(y)), ids, steps, 0.5, 2)
    assert [row["time_step"] for row in rows] == list(range(41, 50))
    assert aggregate_step_metrics(rows)["steps_evaluated"] >= 1
    assert build_edge_index(pd.DataFrame(columns=["txId1", "txId2"]), 0).shape == (2, 0)
