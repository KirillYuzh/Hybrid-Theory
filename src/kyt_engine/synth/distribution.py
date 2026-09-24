"""Quantitative distribution fidelity checks against the real Elliptic raw data.

Implements the D3 ("joint features") validation layer:
- marginal MMD per feature (univariate RBF kernel, per-feature bandwidth);
- bivariate MMD in copula space (rank transforms) for a sample of feature pairs —
  this probes the joint/correlation structure directly;
- Spearman correlation-gap (mean absolute off-diagonal difference of rank-correlation
  matrices).

Two comparisons are always reported: synthetic-vs-real and a synthetic-vs-own-split
baseline, so the user can judge whether the deviation is much larger than the
generator's own sampling noise.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from .features_structural import STRUCTURAL_FEATURES, structural_features
from .stats import LABEL_TO_CLASS

RAW_FEATURES = Path("data/raw/elliptic_txs_features.csv")
RAW_CLASSES = Path("data/raw/elliptic_txs_classes.csv")
RAW_EDGELIST = Path("data/raw/elliptic_txs_edgelist.csv")


def _check_raw(raw_dir: Path) -> None:
    missing = [
        p.name for p in (RAW_FEATURES, RAW_CLASSES, RAW_EDGELIST) if not (raw_dir / p.name).exists()
    ]
    if missing:
        raise FileNotFoundError(
            f"raw Elliptic data missing in {raw_dir}: {missing}; "
            "downstream/MMD validation requires the original dataset."
        )


def load_raw_elliptic(raw_dir: Path) -> dict:
    """time_step Series, edgelist, classes and the (203k x 167) feature matrix.

    The raw Elliptic features CSV contains repeated emission rows for a large
    tail of txIds (the same tx reappears across rolling-window timesteps). We
    keep the first occurrence per txId so the feature space matches the classes.
    """
    _check_raw(raw_dir)
    features = pd.read_csv(
        raw_dir / RAW_FEATURES.name, header=None, dtype={0: "int64", 1: "int64"}
    )
    features.columns = ["txId", "time_step"] + [
        f"feat_{i}" for i in range(2, 2 + features.shape[1] - 2)
    ]
    float_cols = [c for c in features.columns if c not in ("txId", "time_step")]
    features[float_cols] = features[float_cols].astype("float32")
    features = features.drop_duplicates(subset="txId", keep="first")
    classes = pd.read_csv(raw_dir / RAW_CLASSES.name)
    classes["cls"] = classes["class"].map(LABEL_TO_CLASS)
    edgelist = pd.read_csv(raw_dir / RAW_EDGELIST.name)
    time_step = features.set_index("txId")["time_step"].astype("int64")
    return {
        "features": features,
        "classes": classes,
        "edgelist": edgelist,
        "time_step": time_step,
    }


def _zscore_columns(X: np.ndarray, Y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pooled = np.vstack([np.asarray(X, dtype=np.float64), np.asarray(Y, dtype=np.float64)])
    mean = pooled.mean(axis=0, keepdims=True)
    std = pooled.std(axis=0, keepdims=True)
    std = np.where(std < 1e-12, 1.0, std)
    return (X - mean) / std, (Y - mean) / std


def _median_bandwidth(d: np.ndarray) -> float:
    """Median of the off-diagonal pairwise distances, floored to avoid collapsing."""
    vals = d[np.triu_indices_from(d, k=1)]
    med = float(np.median(vals))
    return med if med > 1e-8 else 1.0


def _rbf_kernel(d2: np.ndarray, sigma: float) -> np.ndarray:
    return np.exp(-0.5 * d2 / (sigma * sigma))


def _mmd2_for_1d(x: np.ndarray, y: np.ndarray) -> float:
    d = np.abs(x[:, None] - x[None, :])
    sigma = _median_bandwidth(d)
    kxx = _rbf_kernel(d * d, sigma)
    dyy = np.abs(y[:, None] - y[None, :])
    sigma_y = _median_bandwidth(dyy)
    kyy = _rbf_kernel(dyy * dyy, sigma_y)
    dxy = np.abs(x[:, None] - y[None, :])
    sigma_xy = max(_median_bandwidth(dxy), 0.5 * (sigma + sigma_y))
    kxy = _rbf_kernel(dxy * dxy, sigma_xy)
    return float(kxx.mean() + kyy.mean() - 2.0 * kxy.mean())


def marginal_mmd(
    X: np.ndarray, Y: np.ndarray, sample: int = 2000, rng: np.random.Generator | None = None
) -> np.ndarray:
    """Per-feature univariate MMD^2 (biased estimator) between standardized samples."""
    if rng is None:
        rng = np.random.default_rng(0)
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    if X.shape[0] > sample:
        X = X[rng.choice(X.shape[0], sample, replace=False)]
    if Y.shape[0] > sample:
        Y = Y[rng.choice(Y.shape[0], sample, replace=False)]
    Xs, Ys = _zscore_columns(X, Y)
    out = np.empty(Xs.shape[1])
    for j in range(Xs.shape[1]):
        out[j] = _mmd2_for_1d(Xs[:, j], Ys[:, j])
    return out


def _bivariate_mmd(Xa: np.ndarray, Xb: np.ndarray, sigma: float) -> float:
    dx = Xa[:, 0, None] - Xa[None, :, 0]
    dy = Xa[:, 1, None] - Xa[None, :, 1]
    kxx = _rbf_kernel(dx * dx + dy * dy, sigma)
    ex = Xb[:, 0, None] - Xb[None, :, 0]
    ey = Xb[:, 1, None] - Xb[None, :, 1]
    kyy = _rbf_kernel(ex * ex + ey * ey, sigma)
    gx = Xa[:, 0, None] - Xb[None, :, 0]
    gy = Xa[:, 1, None] - Xb[None, :, 1]
    kxy = _rbf_kernel(gx * gx + gy * gy, sigma)
    return float(kxx.mean() + kyy.mean() - 2.0 * kxy.mean())


def copula_bivariate_mmd(
    X: np.ndarray,
    Y: np.ndarray,
    n_pairs: int = 120,
    n: int = 800,
    rng: np.random.Generator | None = None,
) -> tuple[float, float, list[float]]:
    """Mean/p95 bivariate MMD over feature pairs, on empirical copulas (rank space).

    Both samples are transformed independently with the same per-column percentile
    ranks, then for each feature pair the two 2D clouds share one bandwidth (median of
    the pooled pair distances).
    """
    if rng is None:
        rng = np.random.default_rng(0)
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    xu = np.stack([rankdata(X[:, j]) / (X.shape[0] + 1) for j in range(X.shape[1])], axis=1)
    yu = np.stack([rankdata(Y[:, j]) / (Y.shape[0] + 1) for j in range(Y.shape[1])], axis=1)
    ix = rng.choice(X.shape[0], n, replace=False)
    iy = rng.choice(Y.shape[0], n, replace=False)
    xu_n, yu_n = xu[ix], yu[iy]

    p2 = X.shape[1] * (X.shape[1] - 1) // 2
    n_pairs = min(n_pairs, p2)
    if X.shape[1] < 2:
        return 0.0, 0.0, [0.0]
    if n_pairs < p2:
        rows = np.triu_indices(X.shape[1], k=1)
        pick = rng.choice(p2, n_pairs, replace=False)
        pairs = [(int(rows[0][k]), int(rows[1][k])) for k in pick]
    else:
        rows = np.triu_indices(X.shape[1], k=1)
        pairs = [(int(r), int(c)) for r, c in zip(rows[0], rows[1])]

    # pooled bandwidth over all chosen pairs (shares the same kernel scale)
    dists: list[np.ndarray] = []
    for a, b in pairs:
        cloud = np.vstack([xu_n[:, [a, b]], yu_n[:, [a, b]]])
        d2 = (
            (cloud[:, 0, None] - cloud[None, :, 0]) ** 2
            + (cloud[:, 1, None] - cloud[None, :, 1]) ** 2
        )
        dists.append(d2)
    sigma = _median_bandwidth(np.concatenate(dists))
    vals: list[float] = []
    for d2, (a, b) in zip(dists, pairs):
        xab = xu_n[:, [a, b]]
        yab = yu_n[:, [a, b]]
        vals.append(_bivariate_mmd(xab, yab, sigma))
    vals = np.asarray(vals)
    return float(vals.mean()), float(np.quantile(vals, 0.95)), vals.tolist()


def correlation_gap(X: np.ndarray, Y: np.ndarray, sample: int = 20000) -> float:
    """Mean absolute difference of off-diagonal Spearman correlations."""
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    if X.shape[0] > sample:
        X = X[np.random.default_rng(0).choice(X.shape[0], sample, replace=False)]
    if Y.shape[0] > sample:
        Y = Y[np.random.default_rng(0).choice(Y.shape[0], sample, replace=False)]
    rx = np.stack([rankdata(X[:, j]) for j in range(X.shape[1])], axis=1)
    ry = np.stack([rankdata(Y[:, j]) for j in range(Y.shape[1])], axis=1)
    cx = np.corrcoef(rx, rowvar=False)
    cy = np.corrcoef(ry, rowvar=False)
    p = X.shape[1]
    iu = np.triu_indices(p, k=1)
    return float(np.abs(cx[iu] - cy[iu]).mean())


def summarize_marginal(mmd: np.ndarray, names: list[str]) -> dict:
    order = np.argsort(mmd)[::-1]
    return {
        "mean": float(mmd.mean()),
        "median": float(np.median(mmd)),
        "max": float(mmd.max()),
        "top_worst": [{"feature": names[int(k)], "mmd2": float(mmd[int(k)])} for k in order[:5]],
    }


def distribution_report(
    real: np.ndarray,
    synth: np.ndarray,
    feature_names: list[str],
    rng: np.random.Generator | None = None,
    n_pairs: int = 120,
    sample: int = 2000,
) -> dict:
    """Synthetic-vs-real fidelity + synthetic-vs-own-split baseline (reference noise)."""
    if rng is None:
        rng = np.random.default_rng(0)
    real = np.asarray(real, dtype=np.float64)
    synth = np.asarray(synth, dtype=np.float64)

    mmd_real = marginal_mmd(real, synth, sample=sample, rng=rng)
    gap_real, cop_mean, cop_p95 = (None, None, None)
    gap_baseline, cop_base_mean = (None, None)
    if feature_names and len(feature_names) >= 2:
        gap_real = correlation_gap(real, synth)
        cop_mean, cop_p95, _ = copula_bivariate_mmd(real, synth, n_pairs=n_pairs, rng=rng)

    half = max(synth.shape[0] // 2, 1)
    a, b = synth[:half], synth[half : 2 * half]
    mmd_self = marginal_mmd(a, b, sample=sample, rng=rng)
    if feature_names and len(feature_names) >= 2:
        gap_baseline = correlation_gap(a, b)
        cop_base_mean, _, _ = copula_bivariate_mmd(a, b, n_pairs=n_pairs, rng=rng)

    def ratio(x: float | None, base: float | None) -> float | None:
        return None if x is None or base is None or base <= 1e-12 else x / base

    return {
        "feature_space": feature_names,
        "synthetic_vs_real": {
            "marginal_mmd2": summarize_marginal(mmd_real, feature_names),
            "copula_bivariate_mmd2_mean": cop_mean,
            "copula_bivariate_mmd2_p95": cop_p95,
            "correlation_gap": gap_real,
        },
        "synthetic_vs_own_split": {
            "marginal_mmd2": summarize_marginal(mmd_self, feature_names),
            "copula_bivariate_mmd2_mean": cop_base_mean,
            "correlation_gap": gap_baseline,
        },
        "deviation_ratio (real / own-split, <2 healthy)": {
            "marginal_mmd2": ratio(
                mmd_real.mean(), mmd_self.mean()
            ),
            "copula_bivariate_mmd2": ratio(cop_mean, cop_base_mean),
            "correlation_gap": ratio(gap_real, gap_baseline),
        },
    }


def structural_table(raw: dict, root: Path) -> tuple[np.ndarray, list[str]]:
    """Structural features for real Elliptic and the synthetic run, plus names."""
    real_edg = raw["edgelist"]
    real_ts = raw["time_step"]
    real_in = set(real_edg["txId1"]) | set(real_edg["txId2"])

    if real_in - set(real_ts.index):
        missing = real_in - set(real_ts.index)
        print(f"note: {len(missing)} real edgelist endpoints missing from features; dropping")
        real_edg = real_edg[
            real_edg["txId1"].isin(real_ts.index) & real_edg["txId2"].isin(real_ts.index)
        ]

    synth_feat = pd.read_csv(root / "elliptic_txs_features.csv", header=None)
    synth_feat.columns = ["txId", "time_step"] + [f"x{i}" for i in range(2, synth_feat.shape[1])]
    synth_ts = synth_feat.set_index("txId")["time_step"].astype("int64")
    synth_edg = pd.read_csv(root / "elliptic_txs_edgelist.csv")

    s_real = structural_features(real_edg, real_ts).to_numpy()
    s_synth = structural_features(synth_edg, synth_ts).to_numpy()
    return s_real, s_synth, list(STRUCTURAL_FEATURES)


def full_features_for_run(root: Path) -> tuple[np.ndarray, list[str]]:
    """The 165-column feature matrix of a `cdf`-mode run (same space as Elliptic)."""
    synth_feat = pd.read_csv(root / "elliptic_txs_features.csv", header=None)
    names = [f"feat_{i}" for i in range(2, 2 + synth_feat.shape[1] - 2)]
    return synth_feat.iloc[:, 2:].to_numpy(dtype=np.float64), names


def sample_raw_full_features(
    raw: dict, sample: int = 10000, rng: np.random.Generator | None = None
) -> np.ndarray:
    """Subsample of the real 165-column feature matrix (cdf space)."""
    if rng is None:
        rng = np.random.default_rng(0)
    X = raw["features"].iloc[:, 2:].to_numpy(dtype=np.float64)
    if X.shape[0] > sample:
        X = X[rng.choice(X.shape[0], sample, replace=False)]
    return X
