from dataclasses import dataclass
from pathlib import Path

import numpy as np

N_STEPS = 49
STEP_MIN = 1
N_FEATURES = 165


@dataclass(frozen=True)
class VolumeProfile:
    volume: np.ndarray

    @classmethod
    def from_dir(cls, path: Path) -> "VolumeProfile":
        volume_path = Path(path) / "volume.npy"
        if not volume_path.is_file():
            raise FileNotFoundError(f"volume profile not found: {volume_path}")
        volume = np.load(volume_path, allow_pickle=False)
        if volume.shape != (N_STEPS,):
            raise ValueError(f"volume profile must have shape ({N_STEPS},)")
        if not np.issubdtype(volume.dtype, np.integer) or np.any(volume < 0):
            raise ValueError("volume profile must contain non-negative integers")
        return cls(volume.astype(np.int64, copy=True))


__all__ = ["N_FEATURES", "N_STEPS", "STEP_MIN", "VolumeProfile"]
