from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DriftConfig:
    enabled: bool = False
    shutdown_step: int = 40
    shutdown_rate_multiplier: float = 0.0
    novel_step: int = 45
    novel_scheme: str = "wash"


@dataclass
class GeneratorConfig:
    seed: int = 42
    n_txs: int = 10_000
    labeled_ratio: float = 0.23
    illicit_ratio_in_labeled: float = 0.10
    stats_dir: Path = Path("data/elliptic_stats")
    out_dir: Path = Path("data/synthetic/run")
    schemes: dict[str, dict[str, Any]] = field(default_factory=dict)
    p2p_edges_per_tx: float = 1.5
    drift: DriftConfig = field(default_factory=DriftConfig)

    @classmethod
    def from_yaml(cls, path: Path) -> GeneratorConfig:
        raw = yaml.safe_load(Path(path).read_text()) or {}
        defaults = cls()
        drift = raw.get("drift") or {}
        return cls(
            seed=int(raw.get("seed", defaults.seed)),
            n_txs=int(raw.get("n_txs", defaults.n_txs)),
            labeled_ratio=float(raw.get("labeled_ratio", defaults.labeled_ratio)),
            illicit_ratio_in_labeled=float(
                raw.get("illicit_ratio_in_labeled", defaults.illicit_ratio_in_labeled)
            ),
            stats_dir=Path(raw.get("stats_dir", defaults.stats_dir)),
            out_dir=Path(raw.get("out_dir", defaults.out_dir)),
            schemes=raw.get("schemes") or {},
            p2p_edges_per_tx=float(raw.get("p2p_edges_per_tx", defaults.p2p_edges_per_tx)),
            drift=DriftConfig(
                enabled=bool(drift.get("enabled", defaults.drift.enabled)),
                shutdown_step=int(drift.get("shutdown_step", defaults.drift.shutdown_step)),
                shutdown_rate_multiplier=float(
                    drift.get("shutdown_rate_multiplier", defaults.drift.shutdown_rate_multiplier)
                ),
                novel_step=int(drift.get("novel_step", defaults.drift.novel_step)),
                novel_scheme=str(drift.get("novel_scheme", defaults.drift.novel_scheme)),
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "n_txs": self.n_txs,
            "labeled_ratio": self.labeled_ratio,
            "illicit_ratio_in_labeled": self.illicit_ratio_in_labeled,
            "stats_dir": str(self.stats_dir),
            "schemes": self.schemes,
            "p2p_edges_per_tx": self.p2p_edges_per_tx,
            "drift": {
                "enabled": self.drift.enabled,
                "shutdown_step": self.drift.shutdown_step,
                "shutdown_rate_multiplier": self.drift.shutdown_rate_multiplier,
                "novel_step": self.drift.novel_step,
                "novel_scheme": self.drift.novel_scheme,
            },
        }
