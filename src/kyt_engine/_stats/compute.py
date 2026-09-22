from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from kyt_engine.synth.stats import CDF_GRID, CLASS_NAMES, LABEL_TO_CLASS, N_FEATURES, N_STEPS

RAW_FEATURES = Path("data/raw/elliptic_txs_features.csv")
RAW_CLASSES = Path("data/raw/elliptic_txs_classes.csv")
OUT_DIR = Path("data/elliptic_stats")


def compute() -> None:
    """Empirical per-feature CDFs and tx/step volume from the real Elliptic dataset."""
    classes = pd.read_csv(RAW_CLASSES)
    classes["cls"] = classes["class"].map(LABEL_TO_CLASS)

    buffers: dict[str, list[np.ndarray]] = {c: [] for c in CLASS_NAMES}
    volume = np.zeros(N_STEPS, dtype=np.int64)

    feat_cols = [f"feat_{i}" for i in range(2, 2 + N_FEATURES)]
    chunksize = 50_000
    for chunk in pd.read_csv(RAW_FEATURES, header=None, chunksize=chunksize):
        chunk.columns = ["txId", "time_step"] + feat_cols
        merged = chunk.merge(classes[["txId", "cls"]], on="txId", how="left")
        for cls in CLASS_NAMES:
            mask = merged["cls"] == cls
            if mask.any():
                buffers[cls].append(merged.loc[mask, feat_cols].to_numpy(dtype=float))
        vol, _ = np.histogram(merged["time_step"].to_numpy(), bins=np.arange(1, N_STEPS + 2))
        volume += vol.astype(np.int64)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(OUT_DIR / "volume.npy", volume)
    for cls in CLASS_NAMES:
        arr = np.vstack(buffers[cls]) if buffers[cls] else np.zeros((1, N_FEATURES))
        if not buffers[cls]:
            print(f"WARN: no {cls} rows in raw classes; saved zero CDF")
        qs = np.empty((len(CDF_GRID), N_FEATURES))
        for f in range(N_FEATURES):
            qs[:, f] = np.quantile(arr[:, f], CDF_GRID)
        np.save(OUT_DIR / f"cdf_{cls}.npy", qs)
        print(f"{cls}: n={arr.shape[0]}, saved cdf_{cls}.npy")
    print(f"volume: sum={volume.sum()}, -> {OUT_DIR}")


if __name__ == "__main__":
    compute()
