"""Синтетический генератор данных (GAN-подобный).

Генерирует синтетические графы транзакций с встроенными known patterns:
- money laundering (подмывание)
- chain-hopping (цепочки через посредников)
- smurfing (дробление крупных сумм)
- микшеры (объединение мелких сумм)

Файл: src/kyt_engine/data/synthetic.py
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class SyntheticConfig:
    n_nodes: int = 1000
    n_edges: int = 3000
    fraction_illegal: float = 0.10
    seed: int = 42


def _generate_addresses(n: int, rng: np.random.Generator) -> list[str]:
    return [f"0x{rng.bytes(20).hex()}" for _ in range(n)]


def _inject_laundering_pattern(
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    classes: pd.DataFrame,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Внедряет паттерн money laundering: один источник → много получателей → один получатель."""
    n_laundered = max(10, len(nodes) // 50)
    source = nodes["txId"].sample(n_laundered, random_state=int(rng.integers(0, 10000))).tolist()

    for src in source:
        n_hops = int(rng.integers(3, 8))
        current = src
        for _ in range(n_hops):
            next_tx = nodes["txId"].sample(1, random_state=int(rng.integers(0, 10000))).iloc[0]
            edges.loc[len(edges)] = {"txId1": current, "txId2": next_tx}
            current = next_tx

    return edges


def _inject_chain_hopping(
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    classes: pd.DataFrame,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Внедряет паттерн chain-hopping: цепочка транзакций через посредников."""
    n_chains = max(5, len(nodes) // 100)
    for _ in range(n_chains):
        chain_len = int(rng.integers(5, 20))
        chain = nodes["txId"].sample(chain_len, random_state=int(rng.integers(0, 10000))).tolist()
        for i in range(len(chain) - 1):
            edges.loc[len(edges)] = {"txId1": chain[i], "txId2": chain[i + 1]}

    return edges


def _inject_smurfing(
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    classes: pd.DataFrame,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Внедряет паттерн smurfing: крупная сумма разбивается на мелкие части."""
    n_smurfs = max(5, len(nodes) // 200)
    for _ in range(n_smurfs):
        big_tx = nodes["txId"].sample(1, random_state=int(rng.integers(0, 10000))).iloc[0]
        n_parts = int(rng.integers(10, 50))
        for _ in range(n_parts):
            small_tx = nodes["txId"].sample(1, random_state=int(rng.integers(0, 10000))).iloc[0]
            edges.loc[len(edges)] = {"txId1": big_tx, "txId2": small_tx}

    return edges


def _inject_mixer(
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    classes: pd.DataFrame,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Внедряет паттерн микшера: много входов → один выход → много выходов."""
    n_mixers = max(3, len(nodes) // 300)
    for _ in range(n_mixers):
        mixer_in = nodes["txId"].sample(1, random_state=int(rng.integers(0, 10000))).iloc[0]
        mixer_out = nodes["txId"].sample(1, random_state=int(rng.integers(0, 10000))).iloc[0]
        n_inputs = int(rng.integers(20, 100))
        n_outputs = int(rng.integers(20, 100))

        for _ in range(n_inputs):
            inp = nodes["txId"].sample(1, random_state=int(rng.integers(0, 10000))).iloc[0]
            edges.loc[len(edges)] = {"txId1": inp, "txId2": mixer_in}

        for _ in range(n_outputs):
            out = nodes["txId"].sample(1, random_state=int(rng.integers(0, 10000))).iloc[0]
            edges.loc[len(edges)] = {"txId1": mixer_out, "txId2": out}

    return edges


def generate_nodes(n: int, rng: np.random.Generator) -> pd.DataFrame:
    tx_ids = [f"tx_{i:06d}" for i in range(n)]
    base_ts = 1_580_000_000
    timestamps = rng.integers(base_ts, base_ts + 30 * 86400, size=n)
    hours = (timestamps % 86400) // 3600
    days_of_week = (timestamps // 86400) % 7
    return pd.DataFrame(
        {
            "txId": tx_ids,
            "timestamp": timestamps,
            "hour": hours,
            "day_of_week": days_of_week,
            "block_number": rng.integers(10_000_000, 12_000_000, size=n),
            "value": rng.exponential(5.0, size=n).round(4),
            "gas_price": rng.integers(1, 100, size=n),
            "gas_used": rng.integers(21_000, 500_000, size=n),
            "from_address": _generate_addresses(n, rng),
            "to_address": _generate_addresses(n, rng),
            "address": _generate_addresses(n, rng),
        }
    )


def generate_edges(tx_ids: np.ndarray, n_edges: int, rng: np.random.Generator) -> pd.DataFrame:
    idx = rng.integers(0, len(tx_ids), size=(n_edges, 2))
    return pd.DataFrame({"txId1": tx_ids[idx[:, 0]], "txId2": tx_ids[idx[:, 1]]})


def generate_classes(
    tx_ids: np.ndarray, fraction_illegal: float, rng: np.random.Generator
) -> pd.DataFrame:
    n = len(tx_ids)
    n_illegal = int(n * fraction_illegal)
    labels = np.array([1] * n_illegal + [0] * (n - n_illegal))
    rng.shuffle(labels)
    return pd.DataFrame({"txId": tx_ids, "label": labels})


def generate_dataset(
    output_dir: Optional[Path] = None,
    config: Optional[SyntheticConfig] = None,
) -> Path:
    """Generate synthetic Elliptic-like dataset with laundering patterns.

    Returns output directory.
    """
    cfg = config or SyntheticConfig()
    out = output_dir or Path("data/raw")
    out.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(cfg.seed)
    nodes = generate_nodes(cfg.n_nodes, rng)
    edges = generate_edges(nodes["txId"].values, cfg.n_edges, rng)
    classes = generate_classes(nodes["txId"].values, cfg.fraction_illegal, rng)

    # Внедряем известные паттерны
    edges = _inject_laundering_pattern(nodes, edges, classes, rng)
    edges = _inject_chain_hopping(nodes, edges, classes, rng)
    edges = _inject_smurfing(nodes, edges, classes, rng)
    edges = _inject_mixer(nodes, edges, classes, rng)

    nodes.to_csv(out / "nodes.csv", index=False)
    edges.to_csv(out / "edges.csv", index=False)
    classes.to_csv(out / "classes.csv", index=False)
    return out


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Generate synthetic dataset with laundering patterns")
    parser.add_argument("--output", type=Path, default=Path("data/raw"), help="Output directory")
    args = parser.parse_args()
    out = generate_dataset(args.output)
    print(f"Dataset written to {out}/")


if __name__ == "__main__":
    main()