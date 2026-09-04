from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from kyt_engine.features._utils import find_best_threshold
from kyt_engine.training.train_real import load_elliptic_data, train_lightgbm

logger = logging.getLogger(__name__)


def pseudo_label(
    model: object,
    features: pd.DataFrame,
    label_map: dict[int, int],
    confidence_threshold: float = 0.95,
) -> dict:
    if confidence_threshold <= 0.5 or confidence_threshold > 1.0:
        raise ValueError(f"confidence_threshold must be in (0.5, 1.0], got: {confidence_threshold}")

    feature_cols = [c for c in features.columns if c not in ("txId", "time_step", "label")]
    X_all = features[feature_cols].copy()
    X_all = X_all.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    known_ids = set(label_map.keys())
    unknown_mask = ~features["txId"].isin(known_ids)
    X_unknown = X_all[unknown_mask].reset_index(drop=True)

    proba = model.predict_proba(X_unknown)[:, 1]

    high_conf_licit = proba <= (1.0 - confidence_threshold)
    high_conf_illicit = proba >= confidence_threshold

    pseudo_mask = high_conf_licit | high_conf_illicit
    n_pseudo_licit = int(high_conf_licit.sum())
    n_pseudo_illicit = int(high_conf_illicit.sum())
    n_pseudo_labeled = n_pseudo_licit + n_pseudo_illicit

    X_pseudo = X_unknown[pseudo_mask].reset_index(drop=True)
    y_pseudo = np.where(high_conf_illicit[pseudo_mask], 1, 0).astype(int)

    X_known = X_all[features["txId"].isin(known_ids)].reset_index(drop=True)
    y_known = np.array([label_map[int(tid)] for tid in features.loc[features["txId"].isin(known_ids), "txId"]], dtype=int)

    X_augmented = pd.concat([X_known, X_pseudo], ignore_index=True)
    y_augmented = np.concatenate([y_known, y_pseudo])

    if len(X_augmented) != len(X_known) + n_pseudo_labeled:
        raise ValueError(f"Augmented size mismatch: {len(X_augmented)} != {len(X_known)} + {n_pseudo_labeled}")
    if n_pseudo_labeled != n_pseudo_licit + n_pseudo_illicit:
        raise ValueError("Pseudo-label count mismatch")
    if len(y_augmented) != len(X_augmented):
        raise ValueError(f"y/X length mismatch: {len(y_augmented)} != {len(X_augmented)}")
    if X_augmented.columns.tolist() != X_unknown.columns.tolist():
        raise ValueError("Column mismatch between augmented and unknown data")

    logger.info("Pseudo-labels: %d (licit=%d, illicit=%d), total augmented: %d",
                n_pseudo_labeled, n_pseudo_licit, n_pseudo_illicit, len(X_augmented))

    return {
        "n_pseudo_labeled": n_pseudo_labeled,
        "n_pseudo_licit": n_pseudo_licit,
        "n_pseudo_illicit": n_pseudo_illicit,
        "X_augmented": X_augmented,
        "y_augmented": y_augmented,
        "feature_cols": feature_cols,
    }


def augment_illicit(
    X: pd.DataFrame,
    y: np.ndarray,
    n_copies: int = 2,
    noise_std: float = 0.01,
    rng: np.random.Generator | None = None,
) -> tuple[pd.DataFrame, np.ndarray]:
    if rng is None:
        rng = np.random.default_rng(42)

    illicit_mask = y == 1
    X_illicit = X[illicit_mask].values

    augmented_rows = []
    augmented_labels = []

    for _ in range(n_copies):
        noise = rng.normal(0, noise_std, size=X_illicit.shape)
        X_noisy = X_illicit + noise
        augmented_rows.append(X_noisy)
        augmented_labels.append(np.ones(len(X_noisy), dtype=int))

    if augmented_rows:
        X_augmented = np.vstack([X.values] + augmented_rows)
        y_augmented = np.concatenate([y] + augmented_labels)
    else:
        X_augmented = X.values.copy()
        y_augmented = y.copy()

    X_result = pd.DataFrame(X_augmented, columns=X.columns)
    y_result = y_augmented

    logger.info("Augmentation: original %d, illicit=%d, after augmentation: %d",
                len(X), int(illicit_mask.sum()), len(X_result))

    return X_result, y_result


def run_supplement_pipeline() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    features, label_map, edges = load_elliptic_data()

    labeled_mask = features["txId"].isin(set(label_map.keys()))
    features_labeled = features[labeled_mask].copy()
    features_labeled["label"] = features_labeled["txId"].map(label_map)

    feature_cols = [c for c in features_labeled.columns if c not in ("txId", "label")]
    feature_cols = [c for c in feature_cols if c != "time_step"]
    X_all_features = features_labeled[feature_cols].copy()
    X_all_features = X_all_features.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y_all = features_labeled["label"].values.astype(int)

    X_train_base, X_test, y_train_base, y_test = train_test_split(
        X_all_features, y_all, test_size=0.2, stratify=y_all, random_state=42,
    )

    print("=" * 60)
    print("STAGE 1: Baseline model (original data)")
    print("=" * 60)
    baseline = train_lightgbm(X_train_base, y_train_base, X_test, y_test, n_estimators=400)
    print(f"  Precision: {baseline['metrics']['precision']:.4f}")
    print(f"  Recall:    {baseline['metrics']['recall']:.4f}")
    print(f"  F1:        {baseline['metrics']['f1']:.4f}")
    print(f"  AUC-ROC:   {baseline['metrics']['auc_roc']:.4f}")
    print(f"  AUC-PR:    {baseline['metrics']['auc_pr']:.4f}")

    print("\n" + "=" * 60)
    print("STAGE 2: Pseudo-labels")
    print("=" * 60)
    pseudo_result = pseudo_label(
        baseline["model"], features, label_map, confidence_threshold=0.95,
    )

    X_pseudo = pseudo_result["X_augmented"]
    y_pseudo = pseudo_result["y_augmented"]

    unique_p, counts_p = np.unique(y_pseudo, return_counts=True)
    can_stratify_p = all(c >= 2 for c in counts_p)
    X_train_pseudo, X_test_p, y_train_pseudo, y_test_p = train_test_split(
        X_pseudo, y_pseudo, test_size=0.2, stratify=y_pseudo if can_stratify_p else None, random_state=42,
    )

    pseudo_model = train_lightgbm(X_train_pseudo, y_train_pseudo, X_test_p, y_test_p, n_estimators=400)
    print(f"  Pseudo-labels: {pseudo_result['n_pseudo_labeled']} "
          f"(licit={pseudo_result['n_pseudo_licit']}, illicit={pseudo_result['n_pseudo_illicit']})")
    print(f"  Precision: {pseudo_model['metrics']['precision']:.4f}")
    print(f"  Recall:    {pseudo_model['metrics']['recall']:.4f}")
    print(f"  F1:        {pseudo_model['metrics']['f1']:.4f}")
    print(f"  AUC-ROC:   {pseudo_model['metrics']['auc_roc']:.4f}")
    print(f"  AUC-PR:    {pseudo_model['metrics']['auc_pr']:.4f}")

    print("\n" + "=" * 60)
    print("STAGE 3: Illicit augmentation")
    print("=" * 60)
    X_aug, y_aug = augment_illicit(X_pseudo, y_pseudo, n_copies=2, noise_std=0.01)

    unique_a, counts_a = np.unique(y_aug, return_counts=True)
    can_stratify_a = all(c >= 2 for c in counts_a)
    X_train_aug, X_test_a, y_train_aug, y_test_a = train_test_split(
        X_aug, y_aug, test_size=0.2, stratify=y_aug if can_stratify_a else None, random_state=42,
    )

    aug_model = train_lightgbm(X_train_aug, y_train_aug, X_test_a, y_test_a, n_estimators=400)
    print(f"  After augmentation: {len(X_aug)} rows (was {len(X_pseudo)})")
    print(f"  Precision: {aug_model['metrics']['precision']:.4f}")
    print(f"  Recall:    {aug_model['metrics']['recall']:.4f}")
    print(f"  F1:        {aug_model['metrics']['f1']:.4f}")
    print(f"  AUC-ROC:   {aug_model['metrics']['auc_roc']:.4f}")
    print(f"  AUC-PR:    {aug_model['metrics']['auc_pr']:.4f}")

    print("\n" + "=" * 60)
    print("RESULTS COMPARISON")
    print("=" * 60)
    print(f"  {'Metric':<12} {'Baseline':>10} {'Pseudo':>10} {'Augment.':>10}")
    print(f"  {'-'*12} {'-'*10} {'-'*10} {'-'*10}")
    for m in ["precision", "recall", "f1", "auc_roc", "auc_pr"]:
        b = baseline["metrics"][m]
        p = pseudo_model["metrics"][m]
        a = aug_model["metrics"][m]
        print(f"  {m:<12} {b:>10.4f} {p:>10.4f} {a:>10.4f}")


if __name__ == "__main__":
    run_supplement_pipeline()
