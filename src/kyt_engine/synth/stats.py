from __future__ import annotations

from pathlib import Path

import numpy as np

N_FEATURES = 165
N_STEPS = 49
STEP_MIN = 1
CDF_GRID = np.linspace(0.0, 1.0, 1001)
CLASS_NAMES = ("illicit", "licit", "unknown")
LABEL_TO_CLASS = {"1": "illicit", "2": "licit", "unknown": "unknown"}
CLASS_TO_LABEL = {"illicit": "1", "licit": "2", "unknown": "unknown"}


class EllipticStats:
    """CDF artifacts of real Elliptic features per class + tx/step volume.

    `need_cdf=False` skips the feature CDFs: the `semantic` feature mode consumes only
    the temporal step distribution (`volume.npy`).
    """

    def __init__(self, root: Path, need_cdf: bool = True) -> None:
        self.cdf: dict[str, np.ndarray] = {}
        if need_cdf:
            for name in CLASS_NAMES:
                arr = np.load(root / f"cdf_{name}.npy")
                self.cdf[name] = arr
        self.volume = np.load(root / "volume.npy")
        self._step_weights = self.volume / self.volume.sum()

    def sample_features(self, rng: np.random.Generator, cls: str, n: int) -> np.ndarray:
        """Inverse-CDF sampling with jitter between neighboring grid bins."""
        q = self.cdf[cls]  # (1001, 165)
        if n == 0:
            return np.empty((0, q.shape[1]))
        idx = rng.integers(0, len(CDF_GRID) - 1, size=(n, 1))
        base = q[idx[:, 0], :]
        nxt = q[idx[:, 0] + 1, :]
        jitter = rng.uniform(0.0, 1.0, size=base.shape)
        return base + jitter * (nxt - base)

    def sample_step(self, rng: np.random.Generator, n: int) -> np.ndarray:
        cum = np.cumsum(self._step_weights)
        return (np.searchsorted(cum, rng.random(n)) + STEP_MIN).astype(int)
