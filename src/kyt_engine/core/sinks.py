from __future__ import annotations

from typing import Protocol

from kyt_engine.core.contracts import ScoreResult


class Sink(Protocol):
    name: str

    def write(self, result: ScoreResult) -> None: ...


class ConsoleSink:
    name = "console"

    def write(self, result: ScoreResult) -> None:
        print(f"[{result.tx_id}] risk={result.risk_score:.3f} zone={result.risk_zone} triage={result.triage_level}")