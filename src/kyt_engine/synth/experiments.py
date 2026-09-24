"""Evaluation protocol for synthetic-vs-real transfer (ADCC-Bench style).

The protocol is deliberately explicit instead of implicit:

* temporal splits only -- a random split is never used, because on Elliptic-style
  data it inflates Macro-F1 by up to 11% and can flip the model ranking;
* one shared preprocessing contract (scaler fit on the training period only) for
  every model inside a comparison, so RF / LightGBM / graph-conv / ensemble are
  measured on identical inputs;
* N generator seeds per configuration, with Welch's t-test to tell a real effect
  apart from seed noise (an ensemble is only claimed to be "better" when the test
  is significant -- otherwise it is variance reduction, not accuracy);
* separate metrics for the pre-shift (31-40) and post-novel (45-49) periods, so
  degradation under a temporal distribution shift is visible rather than averaged
  away;
* feature attribution (SHAP when the package is available, permutation importance
  always) to check that a model leans on domain indicators instead of artifacts.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse, stats

from .config import GeneratorConfig
from .distribution import load_raw_elliptic
from .downstream import classification_metrics
from .emit import write_dataset
from .features_structural import STRUCTURAL_FEATURES, structural_features
from .graph import build_graph
from .stats import LABEL_TO_CLASS, EllipticStats

TRAIN_MAX = 30
PERIODS: dict[str, tuple[int, int]] = {
    "pre_shift": (31, 40),
    "post_novel": (45, 49),
    "test_all": (41, 49),
}
ALPHA = 0.05
TARGET = "illicit"

SPLIT_POLICY: dict[str, Any] = {
    "split": "temporal by time_step; random splits are forbidden",
    "why": (
        "ADCC-Bench: random splitting inflates Macro-F1 by up to 11% on Elliptic-style "
        "data and can invert the model ranking, so any number here is temporal-only"
    ),
    "train_steps": f"<= {TRAIN_MAX}",
    "pre_shift_steps": f"{PERIODS['pre_shift'][0]}-{PERIODS['pre_shift'][1]}",
    "post_novel_steps": f"{PERIODS['post_novel'][0]}-{PERIODS['post_novel'][1]}",
    "test_steps": ">= 41",
    "preprocessing_contract": [
        "one shared feature space per comparison (structural or cdf_full, never mixed)",
        "StandardScaler fit on the training period only, applied unchanged to every period",
        "no label/target encoding anywhere",
        "identical pipeline for RF, LightGBM, graph-conv and the ensemble",
        "classes are imbalanced: class_weight='balanced' for RF, scale_pos_weight for LightGBM",
    ],
}


def normalized_adjacency(edgelist: pd.DataFrame, index: pd.Index) -> sparse.csr_matrix:
    """Row-normalized adjacency over `index` (isolated nodes keep a zero row)."""
    pos = pd.Series(np.arange(len(index), dtype=np.int64), index=index)
    keep = edgelist["txId1"].isin(pos.index) & edgelist["txId2"].isin(pos.index)
    rows = pos.loc[edgelist.loc[keep, "txId1"]].to_numpy()
    cols = pos.loc[edgelist.loc[keep, "txId2"]].to_numpy()
    data = np.ones(len(rows), dtype=np.float64)
    A = sparse.csr_matrix((data, (rows, cols)), shape=(len(index), len(index)))
    deg = np.asarray(A.sum(axis=1)).ravel()
    inv = np.zeros_like(deg)
    np.divide(1.0, deg, out=inv, where=deg > 0)
    return sparse.diags(inv) @ A


class GraphConvClassifier:
    """SGC-style graph convolution baseline: [X, ÂX, Â²X] -> logistic regression.

    A dependency-free stand-in for a full GNN (no torch in this environment). It answers
    the question the GNN comparison asks -- does neighbourhood aggregation help or hurt
    under temporal shift -- without pulling a deep-learning stack into the generator.
    """

    def __init__(self, hops: int = 2, C: float = 1.0, max_iter: int = 300, seed: int = 0):
        self.hops = hops
        self.C = C
        self.max_iter = max_iter
        self.seed = seed
        self._clf = None

    def _propagate(self, X: np.ndarray, A: sparse.csr_matrix) -> np.ndarray:
        blocks = [X]
        H = X
        for _ in range(self.hops):
            H = A @ H
            blocks.append(H)
        return np.hstack(blocks)

    def fit(
        self, X: np.ndarray, y: np.ndarray, A: sparse.csr_matrix, mask: np.ndarray
    ) -> GraphConvClassifier:
        from sklearn.linear_model import LogisticRegression

        Z = self._propagate(X, A)
        self._clf = LogisticRegression(
            C=self.C, max_iter=self.max_iter, class_weight="balanced", random_state=self.seed
        ).fit(Z[mask], y[mask])
        return self

    def predict_proba(
        self, X: np.ndarray, A: sparse.csr_matrix, mask: np.ndarray
    ) -> np.ndarray:
        return self._clf.predict_proba(self._propagate(X, A)[mask])


def build_models(seed: int, n_estimators: int = 200, hops: int = 2) -> dict[str, Any]:
    """Every model of the comparison, sharing the same feature matrix and split."""
    from sklearn.ensemble import RandomForestClassifier

    models: dict[str, Any] = {
        "rf": RandomForestClassifier(
            n_estimators=n_estimators, random_state=seed, class_weight="balanced", n_jobs=-1
        ),
        "graph_conv": GraphConvClassifier(hops=hops, seed=seed),
    }
    try:
        from lightgbm import LGBMClassifier

        models["lgbm"] = LGBMClassifier(
            n_estimators=n_estimators,
            learning_rate=0.1,
            random_state=seed,
            verbose=-1,
            is_unbalance=True,
        )
    except ImportError:
        pass
    return models


class SoftVote:
    """Mean of member probabilities -- the ensemble whose gains ADCC-Bench finds noisy."""

    def __init__(self, members: dict[str, Any]):
        self.members = members

    def predict_proba(
        self, X: np.ndarray, A: sparse.csr_matrix | None, mask: np.ndarray
    ) -> np.ndarray:
        probs = [predict_proba(name, model, X, A, mask) for name, model in self.members.items()]
        mean = np.mean(probs, axis=0)
        return np.column_stack([1.0 - mean, mean])


def fit_model(
    name: str, model: Any, X: np.ndarray, y: np.ndarray, A: sparse.csr_matrix, mask: np.ndarray
) -> Any:
    if name == "graph_conv":
        return model.fit(X, y, A, mask)
    return model.fit(X[mask], y[mask])


def predict_proba(
    name: str, model: Any, X: np.ndarray, A: sparse.csr_matrix | None, mask: np.ndarray
) -> np.ndarray:
    """Positive-class probability for the rows selected by `mask`."""
    if isinstance(model, SoftVote):
        return model.predict_proba(X, A, mask)[:, 1]
    if name == "graph_conv":
        return model.predict_proba(X, A, mask)[:, 1]
    return model.predict_proba(X[mask])[:, 1]


def welch(a: list[float], b: list[float]) -> dict[str, Any]:
    """Welch's t-test (unequal variances) plus Cohen's d -- the significance gate."""
    x, y = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    out: dict[str, Any] = {"n_a": len(x), "n_b": len(y), "mean_a": _m(x), "mean_b": _m(y)}
    if len(x) < 2 or len(y) < 2 or np.allclose(x, y):
        out.update({"t": None, "p": None, "cohens_d": _d(x, y), "significant": False})
        return out
    res = stats.ttest_ind(x, y, equal_var=False)
    out.update(
        {
            "t": round(float(res.statistic), 4),
            "p": float(res.pvalue),
            "cohens_d": _d(x, y),
            "significant": bool(res.pvalue < ALPHA),
        }
    )
    return out


def _m(values: np.ndarray) -> float:
    return round(float(np.mean(values)), 4) if len(values) else None


def _d(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2 or len(b) < 2:
        return None
    pooled = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1))
                    / (len(a) + len(b) - 2))
    return round(float((a.mean() - b.mean()) / pooled), 4) if pooled > 0 else None


def _period_masks(ts: np.ndarray) -> dict[str, np.ndarray]:
    masks = {"train": ts <= TRAIN_MAX}
    for name, (lo, hi) in PERIODS.items():
        masks[name] = (ts >= lo) & (ts <= hi)
    return masks


def load_space(
    run_dir: Path, space: str
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, np.ndarray]:
    """(features by txId, illicit label, edgelist, time_step) for a generated run."""
    classes = pd.read_csv(run_dir / "elliptic_txs_classes.csv")
    edgelist = pd.read_csv(run_dir / "elliptic_txs_edgelist.csv")
    labels = classes.set_index("txId")["class"].map(LABEL_TO_CLASS)
    raw = pd.read_csv(run_dir / "elliptic_txs_features.csv", header=None)
    raw.columns = ["txId", "time_step"] + [f"feat_{i}" for i in range(2, raw.shape[1])]
    raw = raw.set_index("txId")
    ts = raw["time_step"].to_numpy(dtype=np.int64)
    if space == "structural":
        feats = structural_features(edgelist, raw["time_step"].astype("int64"))
        return feats, (labels == TARGET).astype(float), edgelist, ts
    if space == "cdf_full":
        return raw.drop(columns=["time_step"]), (labels == TARGET).astype(float), edgelist, ts
    raise ValueError(f"unknown feature space: {space}")


def real_space(
    raw: dict, space: str
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, np.ndarray]:
    """Same feature space, label definition and time steps, computed on real Elliptic."""
    classes = raw["classes"]
    edgelist = raw["edgelist"]
    labels = classes.set_index("txId")["class"].map(LABEL_TO_CLASS)
    ts_series = raw["time_step"]
    ts = ts_series.to_numpy(dtype=np.int64)
    if space == "structural":
        in_index = set(ts_series.index)
        edg = edgelist[
            edgelist["txId1"].isin(in_index) & edgelist["txId2"].isin(in_index)
        ]
        feats = structural_features(edg, ts_series)
        return feats, (labels == TARGET).astype(float), edgelist, ts
    if space == "cdf_full":
        feats = raw["features"].set_index("txId").drop(columns=["time_step"])
        return feats, (labels == TARGET).astype(float), edgelist, ts
    raise ValueError(f"unknown feature space: {space}")


def standardize(train: np.ndarray, others: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Preprocessing contract: scaler fit on the training period only."""
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler().fit(train)
    return {name: scaler.transform(mat) for name, mat in others.items()}


def _make_config(
    template: Path | None,
    seed: int,
    space: str,
    drift: bool,
    out_dir: Path,
    n_txs: int | None = None,
    labeled_ratio: float | None = None,
    stats_dir: Path | None = None,
) -> GeneratorConfig:
    cfg = GeneratorConfig.from_yaml(template) if template else GeneratorConfig()
    cfg.seed = seed
    if stats_dir is not None:
        cfg.stats_dir = stats_dir
    cfg.out_dir = out_dir
    if n_txs is not None:
        cfg.n_txs = n_txs
    if labeled_ratio is not None:
        cfg.labeled_ratio = labeled_ratio
    if space == "structural":
        cfg.features.mode = "semantic"
    if drift:
        cfg.drift.enabled = True
        cfg.drift.shutdown_step = 40
        cfg.drift.shutdown_rate_multiplier = 0.0
        cfg.drift.novel_step = 45
        cfg.drift.novel_schemes = ["wash"]
    return cfg


def _generate(
    template: Path | None,
    seed: int,
    space: str,
    drift: bool,
    out_root: Path,
    n_txs: int | None = None,
    labeled_ratio: float | None = None,
    stats_dir: Path | None = None,
) -> Path:
    cfg = _make_config(
        template, seed, space, drift, out_root / f"seed{seed}", n_txs, labeled_ratio, stats_dir
    )
    stats_dir = EllipticStats(cfg.stats_dir, need_cdf=(cfg.features.mode == "cdf"))
    graph = build_graph(cfg, stats_dir)
    write_dataset(cfg, graph, None, cfg.out_dir)
    return cfg.out_dir


def _evaluate(
    models: dict[str, Any],
    X: np.ndarray,
    y: np.ndarray,
    A: sparse.csr_matrix | None,
    mask: np.ndarray,
) -> dict[str, dict]:
    if int(mask.sum()) == 0:
        raise ValueError("empty evaluation period: temporal split produced no rows")
    out: dict[str, dict] = {}
    for name, model in models.items():
        proba = predict_proba(name, model, X, A, mask)
        out[name] = classification_metrics(y[mask], proba)
    return out


def attribution(
    model: Any, X: np.ndarray, y: np.ndarray, names: list[str], seed: int, n_repeats: int = 5
) -> dict[str, Any]:
    """SHAP when available, permutation importance always."""
    report: dict[str, Any] = {}
    try:
        import shap

        explainer = shap.TreeExplainer(model)
        values = np.abs(explainer.shap_values(X[:2000], check_additivity=False))
        if values.ndim == 3:
            values = values[:, :, 1]
        report["shap_mean_abs"] = {n: round(float(v), 5) for n, v in zip(names, values.mean(0))}
        report["attribution_method"] = "shap.TreeExplainer(mean|SHAP|)"
    except Exception as exc:  # noqa: BLE001 - attribution is diagnostic, never fatal
        report["attribution_method"] = f"shap unavailable ({type(exc).__name__})"
    from sklearn.inspection import permutation_importance

    if len(np.unique(y)) < 2:
        report["permutation_importance"] = {}
        report["top_driver"] = None
        report["attribution_method"] += " (skipped: single-class sample)"
        return report
    perm = permutation_importance(
        model, X, y, n_repeats=n_repeats, random_state=seed, scoring="average_precision"
    )
    report["permutation_importance"] = {
        n: round(float(v), 5) for n, v in zip(names, perm.importances_mean)
    }
    top = max(report["permutation_importance"], key=report["permutation_importance"].get)
    report["top_driver"] = top
    return report


def run_protocol(
    template: Path | None,
    raw_dir: Path,
    seeds: list[int],
    space: str = "structural",
    drift: bool = True,
    n_estimators: int = 200,
    with_attribution: bool = True,
    n_txs: int | None = None,
    stats_dir: Path | None = None,
) -> dict[str, Any]:
    """ADCC-Bench style replicated transfer evaluation with shift-aware periods."""
    raw = load_raw_elliptic(raw_dir)
    r_feat, r_y, r_edg, r_ts = real_space(raw, space)
    r_masks = _period_masks(r_ts)
    r_adj = normalized_adjacency(r_edg, r_feat.index)
    names = list(STRUCTURAL_FEATURES) if space == "structural" else [str(c) for c in r_feat.columns]

    scaler_inputs = {"train": r_feat.to_numpy(float)[r_masks["train"]]}
    scaled = standardize(scaler_inputs["train"], {"all": r_feat.to_numpy(float)})
    X_real = scaled["all"]
    r_yv = r_y.to_numpy(float)

    oracle = build_models(seed=0, n_estimators=n_estimators)
    for name, model in oracle.items():
        fit_model(name, model, X_real, r_yv, r_adj, r_masks["train"])
    oracle_metrics = {
        period: _evaluate(oracle, X_real, r_yv, r_adj, mask)
        for period, mask in r_masks.items()
        if period != "train"
    }

    per_seed: list[dict[str, Any]] = []
    out_root = Path(tempfile.mkdtemp(prefix="kyt_protocol_"))
    for seed in seeds:
        run_dir = _generate(
            template, seed, space, drift, out_root, n_txs=n_txs, stats_dir=stats_dir
        )
        s_feat, s_y, s_edg, s_ts = load_space(run_dir, space)
        s_masks = _period_masks(s_ts)
        s_adj = normalized_adjacency(s_edg, s_feat.index)
        X_synth = standardize(
            X_real[r_masks["train"]], {"all": s_feat.to_numpy(float)}
        )["all"]
        s_yv = s_y.to_numpy(float)

        models = build_models(seed=seed, n_estimators=n_estimators)
        for name, model in models.items():
            fit_model(name, model, X_synth, s_yv, s_adj, s_masks["train"])
        members = dict(models)
        models["ensemble"] = SoftVote(members)

        entry: dict[str, Any] = {
            "seed": seed,
            "internal": {
                period: _evaluate(models, X_synth, s_yv, s_adj, mask)
                for period, mask in s_masks.items()
                if period != "train"
            },
            "transfer_to_real": {
                period: _evaluate(models, X_real, r_yv, r_adj, mask)
                for period, mask in r_masks.items()
                if period != "train"
            },
        }
        per_seed.append(entry)

    ttests: dict[str, Any] = {}
    model_names = sorted({name for e in per_seed for name in e["transfer_to_real"]["test_all"]})
    for period in ("pre_shift", "post_novel", "test_all"):
        for name in model_names:
            a = [e["transfer_to_real"][period][name]["f1"] for e in per_seed]
            b = [e["transfer_to_real"][period][name]["pr_auc"] for e in per_seed]
            ttests[f"{name}:{period}"] = {
                "f1_mean": _m(np.asarray(a)),
                "f1_std": _s(a),
                "pr_auc_mean": _m(np.asarray(b)),
                "pr_auc_std": _s(b),
            }
    shift_tests: dict[str, Any] = {}
    for name in ("rf", "lgbm", "graph_conv", "ensemble"):
        if not per_seed or f"{name}:pre_shift" not in ttests:
            continue
        pre = [e["transfer_to_real"]["pre_shift"][name]["f1"] for e in per_seed]
        post = [e["transfer_to_real"]["post_novel"][name]["f1"] for e in per_seed]
        shift_tests[name] = {
            "pre_shift_f1": _m(np.asarray(pre)),
            "post_novel_f1": _m(np.asarray(post)),
            "degradation": round(_m(np.asarray(pre)) - _m(np.asarray(post)), 4),
            "welch_post_vs_pre": welch(post, pre),
        }
    ensemble_tests: dict[str, Any] = {}
    for member in ("rf", "graph_conv", "lgbm"):
        if not per_seed or f"{member}:test_all" not in ttests:
            continue
        ens = [e["transfer_to_real"]["test_all"]["ensemble"]["f1"] for e in per_seed]
        single = [e["transfer_to_real"]["test_all"][member]["f1"] for e in per_seed]
        ensemble_tests[f"ensemble_vs_{member}"] = welch(ens, single)

    report: dict[str, Any] = {
        "protocol": SPLIT_POLICY,
        "feature_space": space,
        "scenario": "streaming_drift" if drift else "stationary",
        "seeds": seeds,
        "n_seeds": len(seeds),
        "periods": {k: f"{lo}-{hi}" for k, (lo, hi) in PERIODS.items()},
        "models": model_names,
        "oracle_real_trained": oracle_metrics,
        "per_seed": per_seed,
        "aggregate": ttests,
        "shift_tests": shift_tests,
        "ensemble_tests": ensemble_tests,
    }
    if with_attribution:
        report["attribution_real_rf"] = attribution(
            oracle["rf"], X_real[r_masks["test_all"]], r_yv[r_masks["test_all"]], names, seed=0
        )
    report["verdict"] = _verdict(shift_tests, ensemble_tests)
    return report


def _s(values: list[float]) -> float | None:
    return round(float(np.std(values, ddof=1)), 4) if len(values) > 1 else None


def _verdict(shift: dict[str, Any], ensemble: dict[str, Any]) -> dict[str, Any]:
    sig_shift = [k for k, v in shift.items() if v["welch_post_vs_pre"]["significant"]]
    sig_ens = [k for k, v in ensemble.items() if v["significant"]]
    return {
        "models_with_significant_post_novel_degradation": sig_shift,
        "ensemble_improvements_over_members": sig_ens,
        "reading": (
            "a significant post-novel drop means the model does not survive the temporal "
            "shift; a non-significant ensemble difference means the ensemble reduces "
            "variance rather than accuracy (report it as such)"
        ),
    }


def _stratified_subsample(
    labels: np.ndarray, mask: np.ndarray, fraction: float, seed: int
) -> np.ndarray:
    """Keep `fraction` of the labeled rows, stratified by class (deterministic)."""
    rng = np.random.default_rng(seed)
    keep = np.zeros_like(mask)
    for value in (0.0, 1.0):
        rows = np.flatnonzero(mask & (labels == value))
        if not len(rows):
            continue
        take = max(1, int(round(len(rows) * fraction)))
        keep[rng.permutation(rows)[:take]] = True
    return keep


def run_fragmentation_sweep(
    template: Path | None,
    raw_dir: Path,
    seeds: list[int],
    labeled_ratios: list[float],
    space: str = "structural",
    drift: bool = True,
    n_estimators: int = 200,
    n_txs: int | None = None,
    stats_dir: Path | None = None,
    dimension: str = "labeled_ratio",
    supervision_fractions: list[float] | None = None,
) -> dict[str, Any]:
    """Vary supervision and watch transfer quality degrade.

    `dimension="labeled_ratio"` changes the generator budget; `dimension="supervision"`
    keeps the budget but subsamples the labeled rows, which is the honest way to shrink
    supervision because planted schemes supply illicit labels regardless of the budget.
    """
    raw = load_raw_elliptic(raw_dir)
    r_feat, r_y, r_edg, r_ts = real_space(raw, space)
    r_masks = _period_masks(r_ts)
    r_adj = normalized_adjacency(r_edg, r_feat.index)
    X_real = standardize(
        r_feat.to_numpy(float)[r_masks["train"]], {"all": r_feat.to_numpy(float)}
    )["all"]
    r_yv = r_y.to_numpy(float)

    out_root = Path(tempfile.mkdtemp(prefix="kyt_frag_"))
    grid: list[tuple[str, float]] = (
        [(dimension, ratio) for ratio in labeled_ratios]
        if dimension == "labeled_ratio"
        else [(dimension, frac) for frac in (supervision_fractions or [0.02, 0.1, 0.5, 1.0])]
    )
    curve: list[dict[str, Any]] = []
    for axis, value in grid:
        per_ratio: list[dict[str, Any]] = []
        for seed in seeds:
            cfg = _make_config(
                template, seed, space, drift, out_root / f"{axis}{value}_seed{seed}",
                n_txs=n_txs,
                labeled_ratio=value if axis == "labeled_ratio" else None,
                stats_dir=stats_dir,
            )
            loader = EllipticStats(cfg.stats_dir, need_cdf=(cfg.features.mode == "cdf"))
            graph = build_graph(cfg, loader)
            write_dataset(cfg, graph, None, cfg.out_dir)
            s_feat, s_y, _s_edg, _ = load_space(cfg.out_dir, space)
            X_s = standardize(X_real[r_masks["train"]], {"all": s_feat.to_numpy(float)})["all"]
            s_yv = s_y.to_numpy(float)
            classes = pd.read_csv(cfg.out_dir / "elliptic_txs_classes.csv")
            class_values = classes["class"].to_numpy()
            labeled = np.zeros(len(s_yv), dtype=bool)
            labeled[class_values != "unknown"] = True
            train_mask = (
                _stratified_subsample(s_yv, labeled, value, seed)
                if axis == "supervision"
                else labeled
            )
            model = build_models(seed=seed, n_estimators=n_estimators)["rf"]
            model.fit(X_s[train_mask], s_yv[train_mask])
            per_ratio.append(
                {
                    "seed": seed,
                    "n_labeled": int(labeled.sum()),
                    "n_labeled_illicit": int((class_values == "1").sum()),
                    "n_supervised": int(train_mask.sum()),
                    "n_supervised_illicit": int(s_yv[train_mask].sum()),
                    "test_all": _evaluate(
                        {"rf": model}, X_real, r_yv, r_adj, r_masks["test_all"]
                    )["rf"],
                }
            )
        f1 = [e["test_all"]["f1"] for e in per_ratio]
        curve.append(
            {
                axis: value,
                "f1_mean": _m(np.asarray(f1)),
                "f1_std": _s(f1),
                "per_seed": per_ratio,
            }
        )
    best = max((c["f1_mean"] or 0.0) for c in curve)
    return {
        "protocol": SPLIT_POLICY,
        "feature_space": space,
        "scenario": "streaming_drift" if drift else "stationary",
        "seeds": seeds,
        "sweep": dimension,
        "curve": curve,
        "degradation_vs_best": {
            str(c[dimension]): round(best - (c["f1_mean"] or 0.0), 4) for c in curve
        },
    }


def write_report(path: Path, report: dict[str, Any]) -> Path:
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return path
