from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from .config import GeneratorConfig
from .emit import write_dataset
from .features import build_features_matrix
from .graph import build_graph
from .stats import CLASS_NAMES, EllipticStats


def _generate(config_path: Path) -> None:
    config = GeneratorConfig.from_yaml(config_path)
    need_cdf = config.features.mode != "semantic"
    if need_cdf:
        missing = [
            f"cdf_{c}.npy" for c in CLASS_NAMES if not (config.stats_dir / f"cdf_{c}.npy").exists()
        ]
        if missing:
            sys.exit(
                f"Missing stats artifacts {missing} in {config.stats_dir}; "
                "run `python -m kyt_engine._stats.compute` first."
            )
    if not (config.stats_dir / "volume.npy").exists():
        sys.exit(
            f"Missing volume.npy in {config.stats_dir}; "
            "run `python -m kyt_engine._stats.compute` first."
        )

    stats = EllipticStats(config.stats_dir, need_cdf=need_cdf)
    graph = build_graph(config, stats)
    if config.features.mode == "semantic":
        write_dataset(config, graph, None, config.out_dir)
    else:
        rng = np.random.default_rng(config.seed)
        features = build_features_matrix(rng, graph, stats)
        write_dataset(config, graph, features, config.out_dir)
    print(f"Generated {len(graph.nodes)} txs, {len(graph.edges)} edges -> {config.out_dir}")


def _validate(
    dir_path: Path, raw_dir: Path, run_distribution: bool, run_downstream_val: bool
) -> None:
    from .validate import validate_dataset, validate_edge_attributes, validate_semantic_features

    manifest = json.loads((dir_path / "manifest.json").read_text())
    validate_dataset(dir_path)
    print(f"OK: {dir_path} satisfies the dataset contract")
    try:
        validate_edge_attributes(dir_path)
    except FileNotFoundError:
        print(f"note: {dir_path}/elliptic_txs_edge_attributes.csv missing (older dataset)")
    else:
        print("OK: edge attributes file present and consistent")
    if manifest.get("config", {}).get("features", {}).get("mode") == "semantic":
        validate_semantic_features(dir_path)
        print("OK: semantic features consistent with edgelist/edge attributes")

    if run_distribution:
        report = run_distribution_checks(dir_path, raw_dir)
        out = dir_path / "distribution_report.json"
        out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"OK: distribution fidelity report -> {out}")
    if run_downstream_val:
        report = run_downstream_validation(dir_path, raw_dir)
        out = dir_path / "downstream_report.json"
        out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"OK: downstream transfer report -> {out}")

    if not (run_distribution or run_downstream_val):
        tip = "add --mmd and/or --downstream to validate against real Elliptic (needs data/raw)"
        print(f"tip: {tip}")


def run_distribution_checks(dir_path: Path, raw_dir: Path) -> dict:
    from .distribution import (
        distribution_report,
        full_features_for_run,
        load_raw_elliptic,
        sample_raw_full_features,
        structural_table,
    )

    manifest = json.loads((dir_path / "manifest.json").read_text())
    raw = load_raw_elliptic(raw_dir)
    rng = np.random.default_rng(distribution_seed)

    mode = manifest.get("config", {}).get("features", {}).get("mode") or "cdf"
    reports = {}
    s_real, s_synth, names = structural_table(raw, dir_path)
    reports["structural_shared"] = distribution_report(
        s_real, s_synth, names, rng=rng
    )
    if mode == "cdf":
        synth_full, feat_names = full_features_for_run(dir_path)
        real_full = sample_raw_full_features(raw, rng=rng)
        reports["cdf_full_165"] = distribution_report(
            real_full, synth_full, feat_names[:165], rng=rng
        )
    return {"feature_mode": mode, "reports": reports}


def run_downstream_validation(dir_path: Path, raw_dir: Path) -> dict:
    from .downstream import run_downstream

    return run_downstream(dir_path, raw_dir, seed=downstream_seed)


distribution_seed = 1
downstream_seed = 2


def main() -> None:
    parser = argparse.ArgumentParser(prog="kyt_engine.synth")
    sub = parser.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate", help="Generate a synthetic dataset")
    g.add_argument("--config", default="configs/generator.yaml")
    v = sub.add_parser("validate", help="Check a dataset against the format contract")
    v.add_argument("--dir", default="data/synthetic/run")
    v.add_argument("--raw-dir", default="data/raw", help="dir with real Elliptic txs (raw data)")
    v.add_argument("--mmd", action="store_true",
                   help="MMD/copula distribution fidelity vs real Elliptic (needs data/raw)")
    v.add_argument("--downstream", action="store_true",
                   help="RF transfer to held-out real Elliptic with F1/PR-AUC/ECE report")
    args = parser.parse_args()
    if args.cmd == "generate":
        _generate(Path(args.config))
    elif args.cmd == "validate":
        _validate(Path(args.dir), Path(args.raw_dir), args.mmd, args.downstream)


if __name__ == "__main__":
    main()
