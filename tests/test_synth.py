from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from kyt_engine.synth.config import GeneratorConfig
from kyt_engine.synth.emit import write_dataset
from kyt_engine.synth.features import build_features_matrix
from kyt_engine.synth.graph import build_graph
from kyt_engine.synth.stats import CDF_GRID, CLASS_NAMES, N_FEATURES, N_STEPS, EllipticStats
from kyt_engine.synth.validate import load_elliptic, validate_dataset, validate_edge_attributes

OUT_FILES = [
    "elliptic_txs_features.csv",
    "elliptic_txs_classes.csv",
    "elliptic_txs_edgelist.csv",
    "elliptic_txs_edge_attributes.csv",
    "manifest.json",
]


@pytest.fixture()
def stats_dir(tmp_path: Path) -> Path:
    root = tmp_path / "stats"
    root.mkdir()
    rng = np.random.default_rng(0)
    for i, cls in enumerate(CLASS_NAMES):
        data = rng.normal(i, 1.0, size=(500, N_FEATURES))
        qs = np.quantile(data, CDF_GRID, axis=0)
        np.save(root / f"cdf_{cls}.npy", qs)
    np.save(root / "volume.npy", np.ones(N_STEPS, dtype=np.int64) * 100)
    return root


def _small_config(stats_dir: Path) -> GeneratorConfig:
    return GeneratorConfig(
        seed=7,
        n_txs=2000,
        labeled_ratio=0.3,
        illicit_ratio_in_labeled=0.2,
        stats_dir=stats_dir,
        out_dir=stats_dir.parent / "run",
        schemes={
            "mixer": {"n_instances": 5, "fan_in_range": (3, 6), "fan_out_range": (3, 6)},
            "peel_chain": {"n_instances": 5, "length_range": (3, 6)},
            "fanout": {"n_instances": 5, "victims_range": (3, 8)},
            "hub_spoke": {"n_instances": 3, "spokes_range": (4, 10)},
            "wash": {"n_instances": 3, "cycle_len_range": (3, 5)},
        },
        p2p_edges_per_tx=1.2,
    )


def _run(cfg: GeneratorConfig) -> None:
    stats = EllipticStats(cfg.stats_dir)
    graph = build_graph(cfg, stats)
    feats = build_features_matrix(np.random.default_rng(cfg.seed), graph, stats)
    write_dataset(cfg, graph, feats, cfg.out_dir)


def test_generate_writes_valid_dataset(stats_dir: Path) -> None:
    cfg = _small_config(stats_dir)
    _run(cfg)

    validate_dataset(cfg.out_dir)
    features, classes, edgelist = load_elliptic(cfg.out_dir)
    assert len(features) == cfg.n_txs
    assert len(classes) == cfg.n_txs
    assert list(classes.columns) == ["txId", "class"]


def test_deterministic(tmp_path: Path, stats_dir: Path) -> None:
    cfg1 = _small_config(stats_dir)
    cfg1.out_dir = tmp_path / "a"
    cfg2 = _small_config(stats_dir)
    cfg2.out_dir = tmp_path / "b"
    _run(cfg1)
    _run(cfg2)
    for name in OUT_FILES:
        assert (cfg1.out_dir / name).read_bytes() == (cfg2.out_dir / name).read_bytes(), name


def test_schemes_restored_from_manifest(stats_dir: Path) -> None:
    cfg = _small_config(stats_dir)
    _run(cfg)
    manifest = json.loads((cfg.out_dir / "manifest.json").read_text())
    assert manifest["seed"] == cfg.seed
    assert manifest["config"]["n_txs"] == cfg.n_txs
    schemes_seen = {gt["scheme"] for gt in manifest["ground_truth"]}
    assert {"mixer", "fanout"} <= schemes_seen
    for gt in manifest["ground_truth"]:
        if gt["role"] in ("mixer_core", "scam"):
            assert gt["class"] == "illicit"
        if gt["role"] in ("victim", "hub", "spoke"):
            assert gt["class"] == "licit"


def test_wash_only_as_drift_scheme(stats_dir: Path) -> None:
    cfg = _small_config(stats_dir)
    graph = build_graph(cfg, EllipticStats(cfg.stats_dir))
    wash_off = [nd for nd in graph.nodes if nd.scheme == "wash"]
    assert not wash_off, "wash is a drift-only scheme and must not appear when drift is disabled"
    cfg.drift.enabled = True
    cfg.drift.novel_step = 45
    cfg.drift.novel_scheme = "wash"
    graph_on = build_graph(cfg, EllipticStats(cfg.stats_dir))
    wash_on = [nd for nd in graph_on.nodes if nd.scheme == "wash"]
    assert wash_on and all(45 <= nd.step <= 49 for nd in wash_on)


def test_drift_reduces_late_illicit(stats_dir: Path) -> None:
    cfg = _small_config(stats_dir)
    cfg.drift.enabled = True
    cfg.drift.shutdown_step = 35
    cfg.drift.shutdown_rate_multiplier = 0.0
    cfg.drift.novel_step = 45
    graph = build_graph(cfg, EllipticStats(cfg.stats_dir))
    late_illicit = [
        nd
        for nd in graph.nodes
        if nd.cls == "illicit" and nd.step >= 35 and nd.scheme != cfg.drift.novel_scheme
    ]
    assert late_illicit == []
    novel = [nd for nd in graph.nodes if nd.scheme == cfg.drift.novel_scheme]
    assert novel and all(nd.step >= 45 for nd in novel)


@pytest.mark.parametrize("seed", range(6))
def test_drift_acceptance(stats_dir: Path, seed: int) -> None:
    cfg = _small_config(stats_dir)
    cfg.seed = seed
    cfg.drift.enabled = True
    cfg.drift.shutdown_step = 40
    cfg.drift.shutdown_rate_multiplier = 0.0
    cfg.drift.novel_step = 45
    cfg.drift.novel_scheme = "wash"
    graph = build_graph(cfg, EllipticStats(cfg.stats_dir))
    wash = [nd for nd in graph.nodes if nd.scheme == "wash"]
    assert wash and all(45 <= nd.step <= 49 for nd in wash)
    non_wash_illicit = [nd for nd in graph.nodes if nd.cls == "illicit" and nd.scheme != "wash"]
    assert max((nd.step for nd in non_wash_illicit), default=0) < 40


def _bfs(graph: dict[int, list[int]], src: int, dst: int, cap: int) -> int | None:
    # ponytail: re-implemented mirror of anchors._bfs_distance; keeps the test non-tautological.
    """BFS distance up to cap; None when farther."""
    if src == dst:
        return 0
    dist = {src: 0}
    queue = [src]
    for u in queue:
        if dist[u] >= cap:
            continue
        for v in graph[u]:
            if v not in dist:
                dist[v] = dist[u] + 1
                if v == dst:
                    return dist[v]
                queue.append(v)
    return None


def _undirected(edgelist: list[tuple[int, int]]) -> dict[int, list[int]]:
    adj: dict[int, list[int]] = {}
    for u, v in edgelist:
        adj.setdefault(u, []).append(v)
        adj.setdefault(v, []).append(u)
    return {k: sorted(lst) for k, lst in adj.items()}


def test_edge_attributes_file_consistent(stats_dir: Path) -> None:
    cfg = _small_config(stats_dir)
    _run(cfg)
    validate_edge_attributes(cfg.out_dir)
    attrs = pd.read_csv(cfg.out_dir / "elliptic_txs_edge_attributes.csv")
    assert len(attrs) == len(pd.read_csv(cfg.out_dir / "elliptic_txs_edgelist.csv"))


def test_manifest_retrieval_contract(stats_dir: Path) -> None:
    cfg = _small_config(stats_dir)
    cfg.anchors.per_1000_nodes = 5
    _run(cfg)
    manifest = json.loads((cfg.out_dir / "manifest.json").read_text())
    for key in (
        "anchor_registry",
        "retrieval_specs",
        "instance_ground_truth",
        "edge_attribute_ground_truth",
        "decoys",
        "holdout",
    ):
        assert key in manifest
    instances = {i["instance_id"]: i for i in manifest["instance_ground_truth"]}
    assert manifest["retrieval_specs"]["node_vector_dim"] == 167
    anchors = manifest["anchor_registry"]["anchors"]
    assert len(anchors) >= 1
    for a in anchors:
        inst = instances[a["instance_id"]]
        assert a["anchor_node_id"] == inst["anchor_node_id"]
        for k, nodes in a["k_hop_neighborhoods"].items():
            assert set(nodes) <= set(inst["node_ids"])
    edges = pd.read_csv(cfg.out_dir / "elliptic_txs_edgelist.csv")
    adj = _undirected(list(zip(edges["txId1"], edges["txId2"])))
    md = cfg.anchors.min_anchor_distance
    ids = [a["anchor_node_id"] for a in anchors]
    for i, a in enumerate(ids):
        for b in ids[i + 1 :]:
            d = _bfs(adj, a, b, md)
            assert d is None or d >= md, (a, b, d)
    for gt in manifest["ground_truth"]:
        assert "instance_id" in gt


def test_edge_attribute_invariants(stats_dir: Path) -> None:
    cfg = _small_config(stats_dir)
    cfg.drift.enabled = True
    cfg.drift.shutdown_step = 40
    cfg.drift.shutdown_rate_multiplier = 0.0
    cfg.drift.novel_step = 45
    cfg.drift.novel_scheme = "wash"
    _run(cfg)
    manifest = json.loads((cfg.out_dir / "manifest.json").read_text())
    edge_gt = manifest["edge_attribute_ground_truth"]
    by_inst: dict[int, list[dict]] = {}
    for e in edge_gt:
        by_inst.setdefault(e["instance_id"], []).append(e)
    seen_mixer = seen_peel = seen_wash = 0
    for inst in manifest["instance_ground_truth"]:
        entries = sorted(by_inst[inst["instance_id"]], key=lambda e: e["edge_id"])
        amounts = [e["amount"] for e in entries]
        if inst["pattern_type"] == "mixer":
            seen_mixer += 1
            total_in = sum(a for e, a in zip(entries, amounts) if e["role"] == "mixer_in")
            total_out = sum(a for e, a in zip(entries, amounts) if e["role"] == "mixer_out")
            fee_ratio = (total_in - total_out) / total_in
            fee_lo, fee_hi = cfg.edge_attributes.amount["mixer"]["fee_fraction"]
            assert fee_lo - 0.001 <= fee_ratio <= fee_hi + 0.001, (inst["instance_id"], fee_ratio)
        elif inst["pattern_type"] == "peel_chain":
            seen_peel += 1
            assert all(amounts[i] >= amounts[i + 1] for i in range(len(amounts) - 1)), amounts
            assert amounts[0] > amounts[-1], amounts
        elif inst["pattern_type"] == "wash":
            seen_wash += 1
            tol = cfg.edge_attributes.amount["wash"]["balance_tolerance"]
            in_out: dict[int, list[float]] = {}
            for e in entries:
                in_out.setdefault(e["txId1"], []).append(("out", e["amount"]))
                in_out.setdefault(e["txId2"], []).append(("in", e["amount"]))
            for node, flows in in_out.items():
                in_sum = sum(a for tag, a in flows if tag == "in")
                out_sum = sum(a for tag, a in flows if tag == "out")
                diff = abs(in_sum - out_sum)
                assert diff <= tol * max(in_sum, out_sum, 1e-9), (inst["instance_id"], node)
    assert seen_mixer >= 1 and seen_peel >= 1
    assert seen_wash >= 1, "wash instances expected with drift enabled"


def test_decoys_valid(stats_dir: Path) -> None:
    cfg = _small_config(stats_dir)
    cfg.p2p_edges_per_tx = 3
    _run(cfg)
    manifest = json.loads((cfg.out_dir / "manifest.json").read_text())
    schemes = {gt["txId"]: gt["scheme"] for gt in manifest["ground_truth"]}
    for dec in manifest["decoys"]:
        assert dec["motif"] in {"fan_in", "cycle"}
        assert all(schemes[n] == "p2p" for n in dec["nodes"])
    cfg2 = _small_config(stats_dir)
    cfg2.background.decoy_detection = False
    _run(cfg2)
    manifest2 = json.loads((cfg2.out_dir / "manifest.json").read_text())
    assert manifest2["decoys"] == []


def test_holdout_annotation(stats_dir: Path) -> None:
    cfg = _small_config(stats_dir)
    entry = {"pattern": "fanout", "step_min": 30, "step_max": 49, "train_excluded": True}
    cfg.holdout.entries = [entry]
    _run(cfg)
    manifest = json.loads((cfg.out_dir / "manifest.json").read_text())
    assert manifest["holdout"]["windows_active"] is True
    tagged = 0
    for inst in manifest["instance_ground_truth"]:
        lo, hi = inst["temporal_window"]
        overlaps = inst["pattern_type"] == "fanout" and max(lo, 30) <= min(hi, 49)
        assert inst["holdout"] == overlaps and inst["train_excluded"] == overlaps
        tagged += inst["holdout"]
    assert tagged >= 1
