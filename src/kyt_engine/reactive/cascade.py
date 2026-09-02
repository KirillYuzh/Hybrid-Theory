"""Multi-stage Screening Cascade — каскадный фильтр транзакций.

Stage 0: Bloom Filter / HyperLogLog — 75% «абсолютно чистых» транзакций
Stage 1: LightGBM на 20 фичах — 20% зелёных, 5% дальше
Stage 2: Полный ансамбль + TGN — только для 5% «серых»
Stage 3: Ручной review — только 0.5% с max uncertainty

Экономия: вычислительные затраты снижаются в 10 раз.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass
from enum import Enum


class Stage(Enum):
    S0_BLOOM = 0
    S1_FAST = 1
    S2_FULL = 2
    S3_MANUAL = 3


@dataclass
class CascadeResult:
    """Результат прохождения транзакции через каскад."""
    tx_id: int
    stage: Stage
    score: float
    action: str  # PASS (зелёная), REVIEW (жёлтая), HOLD (красная)
    reason: str


class ScreeningCascade:
    """Каскад из 4 ступеней фильтрации.

    Каждая ступень решает: пропустить (PASS), отправить на следующий
    уровень (CONTINUE) или заблокировать (HOLD).
    """

    def __init__(
        self,
        fast_model=None,
        full_model=None,
        bloom_save_rate: float = 0.75,
        fast_save_rate: float = 0.20,
        hold_threshold: float = 0.7,
    ):
        self._fast_model = fast_model
        self._full_model = full_model
        self._bloom_save_rate = bloom_save_rate
        self._fast_save_rate = fast_save_rate
        self._hold_threshold = hold_threshold

    def process(self, features: pd.DataFrame, tx_ids: pd.Series) -> list[CascadeResult]:
        """Прогоняет транзакции через каскад, возвращая решения по каждой."""
        results: list[CascadeResult] = []
        n = len(features)

        # --- Stage 0: Bloom Filter (аппроксимация) ---
        # В реальном проде — Bloom Filter на множестве известных/чистых адресов.
        # В PoC — эвристика: «чистые» = известные биржи, повторяющиеся адреса.
        stage0_mask = self._stage0_filter(features)
        stage1_batch = features[~stage0_mask]

        # Транзакции, прошедшие Stage 0 → PASS без дальнейшего анализа
        for i in range(n):
            if stage0_mask.iloc[i]:
                results.append(CascadeResult(
                    tx_id=int(tx_ids.iloc[i]),
                    stage=Stage.S0_BLOOM,
                    score=0.0,
                    action="PASS",
                    reason="known_clean_address",
                ))

        # --- Stage 1: Fast model (LightGBM на 20 фичах) ---
        if len(stage1_batch) > 0 and self._fast_model is not None:
            # Отбираем быстрые фичи: топ-20 по важности (первые 20 f-фич)
            fast_cols = [c for c in stage1_batch.columns if c.startswith('f') and c[1:].isdigit() and int(c[1:]) < 20]
            X_fast = stage1_batch[fast_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
            fast_proba = self._fast_model.predict_proba(X_fast)[:, 1]

            for idx, proba in zip(stage1_batch.index, fast_proba):
                tx_id = int(tx_ids.loc[idx])
                if proba < self._hold_threshold:
                    results.append(CascadeResult(
                        tx_id=tx_id,
                        stage=Stage.S1_FAST,
                        score=float(proba),
                        action="PASS",
                        reason=f"fast_model_score={proba:.3f}",
                    ))
                else:
                    results.append(CascadeResult(
                        tx_id=tx_id,
                        stage=Stage.S1_FAST,
                        score=float(proba),
                        action="REVIEW",
                        reason="fast_model_high_risk",
                    ))
        elif len(stage1_batch) > 0:
            # Нет модели — все непрошедшие Stage 0 идут на REVIEW (эмуляция manual review)
            for idx in stage1_batch.index:
                tx_id = int(tx_ids.loc[idx])
                results.append(CascadeResult(
                    tx_id=tx_id,
                    stage=Stage.S1_FAST,
                    score=0.5,
                    action="REVIEW",
                    reason="no_model_manual_review",
                ))

        # --- Stage 2/3: Полный ансамбль / ручной review — для будущего ---
        # В PoC: все REVIEW идут на manual review (эмуляция).
        # В production: полный ансамбль + TGN здесь.
        results_sorted = sorted(results, key=lambda r: (r.stage.value, r.score), reverse=True)
        return results_sorted

    def _stage0_filter(self, features: pd.DataFrame) -> pd.Series:
        """Эвристика «чистого» адреса: низкая связность, известные биржи, повторяющиеся паттерны."""
        clean = pd.Series(False, index=features.index)

        # Низкая степень связности = скорее всего простой пользователь
        # Достаточно малой степени входа ИЛИ выхода — пользователь не хаб транзакций
        if "in_degree" in features.columns and "out_degree" in features.columns:
            low_connectivity = (
                (features["in_degree"].fillna(0) < 3) |
                (features["out_degree"].fillna(0) < 3)
            )
            clean |= low_connectivity

        # Повторяющиеся суммы (авторизация) — часто видно у легитимных
        if "amount_usd" in features.columns:
            repeated = features["amount_usd"].fillna(0).round(2).duplicated(keep=False)
            clean |= repeated

        # Ограничиваем долю прошедших 75%
        n_pass = int(len(features) * self._bloom_save_rate)
        idx_pass = clean[clean].index[:n_pass]
        mask = pd.Series(False, index=features.index)
        mask.loc[idx_pass] = True
        return mask

    def compute_savings(self, total: int, results: list[CascadeResult]) -> dict:
        """Вычисляет экономию вычислительных ресурсов."""
        n_pass = sum(1 for r in results if r.action == "PASS")
        n_review = sum(1 for r in results if r.action == "REVIEW")
        n_hold = sum(1 for r in results if r.action == "HOLD")

        # Стоимость полного пайплайна = C. Каскад = 0.01*C*0.75 + 0.2*C*0.2 + C*0.05
        cascade_cost = 0.01 * 0.75 * total + 0.2 * 0.20 * total + 1.0 * 0.05 * total
        full_cost = 1.0 * total

        return {
            "n_total": total,
            "n_pass": n_pass,
            "n_review": n_review,
            "n_hold": n_hold,
            "pass_pct": round(n_pass / total * 100, 1) if total else 0,
            "cost_savings_x": round(full_cost / cascade_cost, 2) if cascade_cost else 0,
        }