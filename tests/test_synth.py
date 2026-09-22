from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from kyt_engine.synth.config import GeneratorConfig
from kyt_engine.synth.emit import write_dataset
from kyt_engine.synth.features import build_features_matrix
from kyt_engine.synth.graph import build_graph
from kyt_engine.synth.stats import CDF_GRID, CLASS_NAMES, N_FEATURES, N_STEPS, EllipticStats
from kyt_engine.synth.validate import load_elliptic, validate_dataset

OUT_FILES = [
    "elliptic_txs_features.csv",
    "elliptic_txs_classes.csv",
    "elliptic_txs_edgelist.csv",
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
