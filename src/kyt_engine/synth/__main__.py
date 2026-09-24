from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from .config import GeneratorConfig
from .emit import write_dataset
from .experiments import run_protocol, write_report
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
    dir_path: Path,
    raw_dir: Path,
    run_distribution: bool,
    run_downstream_val: bool,
    seeds: list[int] | None = None,
    space: str = "structural",
    config: Path | None = None,
    n_estimators: int = 200,
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
        report = run_protocol(
            config or Path("configs/generator.yaml"),
            raw_dir,
            seeds=seeds or list(range(downstream_seeds)),
            space=space,
            n_estimators=n_estimators,
        )
        out = write_report(dir_path / f"downstream_report_{space}.json", report)
        print(f"OK: downstream protocol ({report['n_seeds']} seeds, {report['scenario']}) -> {out}")
        print(f"   models: {', '.join(report['models'])}")
        for name, test in report["shift_tests"].items():
            flag = "significant" if test["welch_post_vs_pre"]["significant"] else "not significant"
            print(
                f"   {name}: F1 {test['pre_shift_f1']} -> {test['post_novel_f1']} "
                f"(post vs pre: {flag}, p={test['welch_post_vs_pre']['p']})"
            )

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
downstream_seeds = 20


def _sweep(args: argparse.Namespace) -> None:
    from .experiments import run_fragmentation_sweep, write_report

    grid = [float(x) for x in args.grid.split(",")]
    seeds = [int(x) for x in args.seeds.split(",")]
    axis = "supervision" if args.dimension == "supervision" else "labeled_ratio"
    report = run_fragmentation_sweep(
        Path(args.config),
        Path(args.raw_dir),
        seeds=seeds,
        labeled_ratios=grid if axis == "labeled_ratio" else [0.23],
        supervision_fractions=grid if axis == "supervision" else None,
        space=args.space,
        dimension=axis,
    )
    out = write_report(Path(args.out), report)
    print(f"OK: {axis} sweep ({len(grid)} points x {len(seeds)} seeds) -> {out}")
    for point in report["curve"]:
        sup = point["per_seed"][0]
        print(
            f"   {axis}={point[axis]}: F1={point['f1_mean']} +/- {point['f1_std']} "
            f"(supervised illicit={sup['n_supervised_illicit']}, "
            f"drop vs best {report['degradation_vs_best'][str(point[axis])]})"
        )


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
                   help="replicated transfer protocol (N seeds, Welch t-test, shift periods)")
    v.add_argument("--seeds", default="", help="comma-separated generator seeds for --downstream")
    v.add_argument("--space", default="structural", choices=["structural", "cdf_full"],
                   help="shared feature space for the downstream comparison")
    v.add_argument("--config", default="configs/generator.yaml",
                   help="config template the downstream protocol re-generates from")
    v.add_argument("--estimators", type=int, default=200,
                   help="trees per boosting model (lower = cheaper, noisier)")
    s = sub.add_parser("sweep", help="Fragmented-supervision sweep")
    s.add_argument("--config", default="configs/generator.yaml")
    s.add_argument("--raw-dir", default="data/raw")
    s.add_argument("--grid", default="0.05,0.10,0.23,0.30",
                   help="comma-separated sweep points")
    s.add_argument("--dimension", default="labeled_ratio",
                   choices=["labeled_ratio", "supervision"],
                   help="labeled_ratio = generator budget; supervision = subsample labels")
    s.add_argument("--seeds", default="0,1,2")
    s.add_argument("--space", default="structural", choices=["structural", "cdf_full"])
    s.add_argument("--out", default="data/synthetic/run/fragmentation_report.json")
    args = parser.parse_args()
    if args.cmd == "generate":
        _generate(Path(args.config))
    elif args.cmd == "validate":
        seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else None
        _validate(
            Path(args.dir), Path(args.raw_dir), args.mmd, args.downstream,
            seeds=seeds, space=args.space, config=Path(args.config),
            n_estimators=args.estimators,
        )
    elif args.cmd == "sweep":
        _sweep(args)


if __name__ == "__main__":
    main()
