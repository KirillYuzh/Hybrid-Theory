from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class UncertainSample:
    index: int
    k_score: float
    model_proba: float
    entropy: float
    priority: str
    feature_values: dict[str, float]


class UncertaintySampler:
    """Selects most uncertain transactions for human labeling.

    Uses model entropy + K-Score to identify the most informative samples.
    """

    def __init__(
        self,
        top_k: int = 1000,
        entropy_threshold: float = 0.7,
        kscore_threshold: float = 0.5,
    ):
        self._top_k = top_k
        self._entropy_threshold = entropy_threshold
        self._kscore_threshold = kscore_threshold

    def compute_entropy(self, proba: np.ndarray) -> np.ndarray:
        """Compute entropy of binary predictions.

        Max entropy at p=0.5 (most uncertain).
        """
        p = np.clip(proba, 1e-10, 1 - 1e-10)
        return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))

    def select_samples(
        self,
        proba: np.ndarray,
        k_scores: np.ndarray,
        features: pd.DataFrame | np.ndarray,
        top_k: int | None = None,
    ) -> list[UncertainSample]:
        """Select top_k most uncertain samples for human labeling."""
        k = top_k or self._top_k
        entropy = self.compute_entropy(proba)
        scores = entropy + k_scores
        top_indices = np.argsort(scores)[::-1][:k]

        n = len(proba)
        samples: list[UncertainSample] = []
        for idx in top_indices:
            i = int(idx)
            if i >= n:
                continue
            e = float(entropy[i])
            ks = float(np.asarray(k_scores)[i])

            if e > self._entropy_threshold and ks > self._kscore_threshold:
                priority = "HIGH"
            elif e > self._entropy_threshold or ks > self._kscore_threshold:
                priority = "MEDIUM"
            else:
                priority = "LOW"

            if isinstance(features, pd.DataFrame):
                feat_vals = {col: float(features.iloc[i][col]) for col in features.columns}
            else:
                feat_vals = {f"f{j}": float(features[i, j]) 
                            for j in range(min(len(features[i]), 167))}

            samples.append(
                UncertainSample(
                    index=i,
                    k_score=ks,
                    model_proba=float(proba[i]),
                    entropy=e,
                    priority=priority,
                    feature_values=feat_vals,
                )
            )
        return samples

    def export_for_labeling(
        self,
        samples: list[UncertainSample],
        path: Path | None = None,
    ) -> pd.DataFrame:
        """Export selected samples as CSV for human labeling."""
        rows = []
        for s in samples:
            row = {
                "index": s.index,
                "k_score": s.k_score,
                "model_proba": s.model_proba,
                "entropy": s.entropy,
                "priority": s.priority,
            }
            row.update(s.feature_values)
            rows.append(row)

        df = pd.DataFrame(rows)
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(path, index=False)
            logger.info("Exported %d samples to %s", len(df), path)
        return df


def init_model(n_estimators: int = 200, **kwargs) -> "LGBMClassifier":
    """Create a fresh LightGBM classifier with standard hyperparameters."""
    from lightgbm import LGBMClassifier

    defaults = dict(
        learning_rate=0.05,
        num_leaves=63,
        class_weight="balanced",
        random_state=42,
        verbose=-1,
        n_jobs=1,
    )
    defaults.update(kwargs)
    return LGBMClassifier(n_estimators=n_estimators, **defaults)


class FeedbackLoop:
    """Manages the feedback loop: labeled data -> model update."""

    def __init__(self, base_model_path: Path | None = None):
        self._model_path = base_model_path

    def load_labels(self, labeled_csv: Path) -> pd.DataFrame:
        """Load human-labeled CSV (adds 'human_label' column)."""
        df = pd.read_csv(labeled_csv)
        if "human_label" not in df.columns:
            raise ValueError(f"CSV must contain 'human_label' column. Found: {list(df.columns)}")
        return df

    def merge_with_training(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        labeled: pd.DataFrame,
    ) -> tuple[pd.DataFrame, np.ndarray]:
        """Merge new labels with existing training data.

        Deduplicates by index (new labels override old).
        """
        feat_cols = [c for c in labeled.columns if c not in ("index", "k_score", "model_proba", "entropy", "priority", "human_label")]
        new_X = labeled[feat_cols].copy()
        new_y = labeled["human_label"].values.astype(int)

        existing = X_train.copy()
        existing["_y"] = y_train

        new_data = new_X.copy()
        new_data["_y"] = new_y
        if "index" in labeled.columns:
            new_data.index = labeled["index"].values

        combined = pd.concat([existing, new_data], axis=0)
        combined = combined[~combined.index.duplicated(keep="last")]
        y_out = combined["_y"].values.astype(int)
        X_out = combined.drop(columns=["_y"])
        return X_out, y_out

    def incremental_retrain(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_val: pd.DataFrame,
        y_val: np.ndarray,
        n_rounds: int = 100,
        existing_model=None,
    ) -> dict[str, float]:
        """Retrain LightGBM with incremental iterations.

        If *existing_model* is provided its tree count is increased by
        *n_rounds* and it is refit on the new data. Otherwise a fresh model
        is created via :func:`init_model`.

        Returns metrics before and after retraining.
        """
        from sklearn.metrics import f1_score, roc_auc_score

        if existing_model is not None:
            model_before = existing_model
            n_new = existing_model.n_estimators + n_rounds
        else:
            model_before = init_model(n_estimators=50)
            n_new = 50 + n_rounds

        model_before.fit(X_train, y_train)
        base_proba = model_before.predict_proba(X_val)[:, 1]
        base_pred = (base_proba >= 0.5).astype(int)

        model_after = init_model(n_estimators=n_new)
        model_after.fit(X_train, y_train)
        new_proba = model_after.predict_proba(X_val)[:, 1]
        new_pred = (new_proba >= 0.5).astype(int)

        return {
            "before_f1": float(f1_score(y_val, base_pred, zero_division=0)),
            "before_auc": float(roc_auc_score(y_val, base_proba)),
            "after_f1": float(f1_score(y_val, new_pred, zero_division=0)),
            "after_auc": float(roc_auc_score(y_val, new_proba)),
            "model": model_after,
        }


class ActiveLearningPipeline:
    """Full active learning cycle: sample -> label -> retrain -> evaluate."""

    def __init__(
        self,
        top_k: int = 1000,
        entropy_threshold: float = 0.7,
    ):
        self._sampler = UncertaintySampler(top_k=top_k, entropy_threshold=entropy_threshold)
        self._feedback = FeedbackLoop()

    def run_cycle(
        self,
        model_proba: np.ndarray,
        k_scores: np.ndarray,
        features: pd.DataFrame,
        export_dir: Path | None = None,
    ) -> dict:
        """Run one active learning cycle."""
        samples = self._sampler.select_samples(model_proba, k_scores, features)
        priorities = {"HIGH": 0, "MEDIUM": 0, "LOW": 0}
        for s in samples:
            priorities[s.priority] += 1

        export_path = None
        if export_dir is not None:
            export_dir.mkdir(parents=True, exist_ok=True)
            export_path = export_dir / "labeling_batch.csv"
            self._sampler.export_for_labeling(samples, export_path)

        return {
            "n_selected": len(samples),
            "priority_distribution": priorities,
            "export_path": str(export_path) if export_path else None,
        }


class ActiveLearningSampler:
    """Сэмплирование адресов для ручной разметки (блок 21.1).

    Отбор на уровне адресов (поведенческих профилей), а не отдельных транзакций.
    Комбинированный скор: uncertainty + diversity + cost-sensitive.
    """

    def __init__(
        self,
        entropy_threshold: float = 0.5,
        diversity_weight: float = 0.3,
        cost_weight: float = 0.3,
        top_k: int = 100,
    ):
        self._entropy_threshold = entropy_threshold
        self._diversity_weight = diversity_weight
        self._cost_weight = cost_weight
        self._top_k = top_k
        self.sampled_addresses: set[str] = set()

    def compute_entropy(self, proba_matrix: np.ndarray) -> pd.Series:
        """Энтропия Шеннона для каждой строки матрицы вероятностей.

        Поддерживает 1D (бинарный) и 2D (мультиклассовый) вход.
        """
        arr = np.asarray(proba_matrix, dtype=float)
        if arr.ndim == 1:
            p = np.clip(arr, 1e-10, 1 - 1e-10)
            entropy = -(p * np.log2(p) + (1 - p) * np.log2(1 - p))
        else:
            p = np.clip(arr, 1e-10, 1.0)
            entropy = -np.sum(p * np.log2(p), axis=1)
        return pd.Series(entropy)

    def uncertainty_score(self, df: pd.DataFrame) -> pd.Series:
        """Комбинированный скор неопределённости.

        score = norm(entropy) + diversity_weight * novelty + cost_weight * norm(amount_usd)
        novelty = обратная близость к уже отобранным адресам (пустое множество → 1.0).
        """
        df = df.copy()
        entropy_col = "entropy" if "entropy" in df.columns else None
        amount_col = "amount_usd" if "amount_usd" in df.columns else None

        # Базовый скор неопределённости: чем выше энтропия (неопределённость), тем выше base
        if entropy_col is not None:
            base = df[entropy_col].astype(float).clip(0.0, 1.0).fillna(0.0)
        else:
            base = pd.Series(0.0, index=df.index)

        # Новизна: если адрес уже близок к отобранным, novelty снижается
        if "address" in df.columns and self.sampled_addresses:
            novelty = df["address"].apply(
                lambda a: 0.5 if a in self.sampled_addresses else 1.0
            ).astype(float)
        else:
            novelty = pd.Series(1.0, index=df.index)

        # Стоимость ошибки: нормализация суммы
        cost = pd.Series(0.0, index=df.index)
        if amount_col is not None:
            vals = df[amount_col].astype(float).fillna(0.0)
            mx = float(vals.max()) if len(vals) > 0 and vals.max() > 0 else 1.0
            cost = vals / mx

        return base + self._diversity_weight * novelty + self._cost_weight * cost

    def sample(self, features: pd.DataFrame) -> pd.DataFrame:
        """Возвращает подмножество строк для ручной разметки.

        Фильтрует строки с entropy > порога, группирует по адресу,
        агрегирует скор по адресу, выбирает top_k адресов и обновляет sampled_addresses.
        """
        if features.empty:
            return features.copy()

        df = features.copy()
        entropy_col = "entropy" if "entropy" in df.columns else None
        if entropy_col is not None:
            df = df[df[entropy_col].astype(float) > self._entropy_threshold].copy()

        if df.empty:
            return df

        df["_score"] = self.uncertainty_score(df)

        # Группировка по адресу (поведенческий профиль) — средний скор по адресу
        address_col = "address" if "address" in df.columns else df.index.name
        if address_col is None or address_col not in df.columns:
            df["address"] = df.index.astype(str)
            address_col = "address"

        addr_scores = df.groupby(address_col)["_score"].mean().sort_values(ascending=False)
        top_addrs = addr_scores.head(self._top_k).index.tolist()

        # Обновляем внутреннее состояние отобранных адресов
        self.sampled_addresses.update(top_addrs)

        result = df[df[address_col].isin(top_addrs)].drop(columns=["_score"])
        return result
