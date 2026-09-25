import argparse
import json
from pathlib import Path

from .config import GeneratorConfig
from .emit import write_dataset
from .graph import build_graph
from .stats import VolumeProfile


def _generate(config_path: Path) -> None:
    config = GeneratorConfig.from_yaml(config_path)
    raw_path = Path("data/raw").resolve()
    output_path = config.out_dir.resolve()
    if output_path == raw_path or raw_path in output_path.parents:
        raise SystemExit("Refusing to write generated data inside data/raw")
    try:
        stats = VolumeProfile.from_dir(config.stats_dir)
    except FileNotFoundError as exc:
        raise SystemExit(str(exc)) from exc
    graph = build_graph(config, stats)
    write_dataset(config, graph, config.out_dir)
    print(f"Generated {len(graph.nodes)} transactions and {len(graph.edges)} edges")


def _validate(dir_path: Path) -> None:
    from .validate import (
        validate_behavior_dataset,
        validate_dataset,
        validate_edge_attributes,
        validate_semantic_features,
    )

    manifest = json.loads((dir_path / "manifest.json").read_text(encoding="utf-8"))
    validate_dataset(dir_path)
    validate_edge_attributes(dir_path)
    validate_semantic_features(dir_path)
    if manifest.get("config", {}).get("graph", {}).get("mode") != "behavior":
        raise SystemExit("dataset is not a behavior dataset")
    validate_behavior_dataset(dir_path)
    print(f"Validated behavior dataset: {dir_path}")


def _validate_real(args: argparse.Namespace) -> int:
    from .gnn_downstream import run_validate_real

    try:
        seeds = (
            [int(value) for value in args.seeds.split(",") if value]
            if args.seeds is not None
            else None
        )
    except ValueError as exc:
        print(f"strict validation status=error: invalid seeds: {exc}")
        return 4
    report = run_validate_real(
        Path(args.config),
        Path(args.raw_dir),
        Path(args.out),
        backend=args.backend,
        tier=args.tier,
        seeds=seeds,
        k=args.k,
        all_gnn=not args.sage_only,
    )
    print(f"strict validation status={report['status']} -> {args.out}")
    if report["status"] == "blocked":
        print(f"blocked: {report.get('error', {})}")
        return 3
    if report["status"] == "error":
        print(f"error: {report.get('error', {})}")
        return 4
    if report["benchmark_tier"] == "real" and (
        not report["acceptance_eligible"] or not report.get("acceptance", {}).get("target_met")
    ):
        return 2
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(prog="kyt_engine.synth")
    commands = parser.add_subparsers(dest="command", required=True)

    generate = commands.add_parser("generate", help="Generate a behavior dataset")
    generate.add_argument("--config", default="configs/generator_behavior.yaml")

    validate = commands.add_parser("validate", help="Validate a behavior dataset")
    validate.add_argument("--dir", default="data/synthetic/behavior_run")

    real = commands.add_parser("validate-real", help="Run strict downstream validation")
    real.add_argument("--config", default="configs/generator_behavior.yaml")
    real.add_argument("--raw-dir", default="data/raw")
    real.add_argument("--out", default="data/synthetic/behavior_run/strict_report.json")
    real.add_argument("--tier", choices=["smoke", "real"], default="smoke")
    real.add_argument("--backend", choices=["auto", "full", "sgc"], default=None)
    real.add_argument("--seeds", default=None)
    real.add_argument("--k", type=int, default=None)
    real.add_argument("--sage-only", action="store_true")

    args = parser.parse_args()
    if args.command == "generate":
        _generate(Path(args.config))
    elif args.command == "validate":
        _validate(Path(args.dir))
    else:
        code = _validate_real(args)
        if code:
            raise SystemExit(code)


if __name__ == "__main__":
    main()
