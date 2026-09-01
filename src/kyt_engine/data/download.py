"""Download or generate Elliptic-like dataset for KYT engine testing."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path("data/raw")
N_NODES = 1000
N_EDGES = 3000
FRACTION_ILLEGAL = 0.10
SEED = 42


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
            "from_address": [f"0x{rng.bytes(20).hex()}" for _ in range(n)],
            "to_address": [f"0x{rng.bytes(20).hex()}" for _ in range(n)],
            "address": [f"0x{rng.bytes(20).hex()}" for _ in range(n)],
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


def generate_dataset(output_dir: Path | None = None) -> Path:
    """Generate synthetic Elliptic-like dataset. Returns output directory."""
    out = output_dir or DATA_DIR
    out.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(SEED)
    nodes = generate_nodes(N_NODES, rng)
    edges = generate_edges(nodes["txId"].values, N_EDGES, rng)
    classes = generate_classes(nodes["txId"].values, FRACTION_ILLEGAL, rng)

    nodes.to_csv(out / "nodes.csv", index=False)
    edges.to_csv(out / "edges.csv", index=False)
    classes.to_csv(out / "classes.csv", index=False)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Download / generate Elliptic-like dataset")
    parser.add_argument("--output", type=Path, default=DATA_DIR, help="Output directory")
    args = parser.parse_args()
    out = generate_dataset(args.output)
    print(f"Dataset written to {out}/")
    print(f"  nodes.csv   : {N_NODES} rows")
    print(f"  edges.csv   : {N_EDGES} rows")
    print(f"  classes.csv : {N_NODES} rows ({int(N_NODES * FRACTION_ILLEGAL)} illicit, "
          f"{N_NODES - int(N_NODES * FRACTION_ILLEGAL)} licit)")


if __name__ == "__main__":
    main()
