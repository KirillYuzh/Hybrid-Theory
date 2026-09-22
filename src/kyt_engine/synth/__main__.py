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


def _validate(dir_path: Path) -> None:
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


def main() -> None:
    parser = argparse.ArgumentParser(prog="kyt_engine.synth")
    sub = parser.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate", help="Generate a synthetic dataset")
    g.add_argument("--config", default="configs/generator.yaml")
    v = sub.add_parser("validate", help="Check a dataset against the format contract")
    v.add_argument("--dir", default="data/synthetic/run")
    args = parser.parse_args()
    if args.cmd == "generate":
        _generate(Path(args.config))
    elif args.cmd == "validate":
        _validate(Path(args.dir))


if __name__ == "__main__":
    main()
