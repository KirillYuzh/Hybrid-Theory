"""Два ортогональных скора для Reactive-модуля.

AML-Risk (115-ФЗ): отмывание денег — детекция известных схем мошенничества
FX-Risk (173-ФЗ): валютный контроль нерезидентов — отслеживание конверсий и переводов
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass


@dataclass
class DualScoreResult:
    """Результат двойного скоринга: AML + FX."""
    aml_risk: float
    fx_risk: float
    combined_risk: float
    aml_zone: str
    fx_zone: str


class DualScorer:
    """Вычисляет AML-Risk и FX-Risk независимо, затем комбинирует."""

    def __init__(
        self,
        aml_threshold_red: float = 0.7,
        fx_threshold_red: float = 0.7,
        aml_weight: float = 0.7,
        fx_weight: float = 0.3,
    ):
        self._aml_threshold_red = aml_threshold_red
        self._fx_threshold_red = fx_threshold_red
        self._aml_weight = aml_weight
        self._fx_weight = fx_weight

    def score(self, features: pd.DataFrame) -> list[DualScoreResult]:
        """Вычисляет AML и FX скоры на основе фич транзакции."""
        aml_scores = self._compute_aml_risk(features)
        fx_scores = self._compute_fx_risk(features)

        combined = self._aml_weight * aml_scores + self._fx_weight * fx_scores

        results = []
        for i in range(len(features)):
            aml_risk = float(aml_scores[i])
            fx_risk = float(fx_scores[i])
            combined_risk = float(combined[i])
            results.append(DualScoreResult(
                aml_risk=aml_risk,
                fx_risk=fx_risk,
                combined_risk=combined_risk,
                aml_zone=self._zone(aml_risk, self._aml_threshold_red),
                fx_zone=self._zone(fx_risk, self._fx_threshold_red),
            ))
        return results

    def _compute_aml_risk(self, features: pd.DataFrame) -> np.ndarray:
        """AML риск — детекция отмывания. На основе отклонений от нормы."""
        if "in_degree" in features.columns:
            # Высокая связность с неизвестными адресами = повышенный AML риск
            degree_signal = features["in_degree"].fillna(0).clip(0, 100).values / 100.0
        else:
            degree_signal = np.zeros(len(features))

        # Временной сигнал: аномальная ночная активность
        if "hour_of_day" in features.columns:
            hour = features["hour_of_day"].fillna(0).values
            night_signal = np.where((hour < 6) | (hour > 22), 0.8, 0.2)
        else:
            night_signal = np.zeros(len(features))

        # Кол-во уникальных контрагентов (если есть)
        if "unique_counterparties" in features.columns:
            counterpart_signal = features["unique_counterparties"].fillna(1).clip(1, 100).values / 100.0
        else:
            counterpart_signal = np.zeros(len(features))

        # Комбинация (взвешенная)
        aml = 0.5 * degree_signal + 0.3 * night_signal + 0.2 * counterpart_signal
        return np.clip(aml, 0.0, 1.0)

    def _compute_fx_risk(self, features: pd.DataFrame) -> np.ndarray:
        """FX риск — валютный контроль нерезидентов (173-ФЗ)."""
        # Резиденты: низкий FX риск. Нерезиденты (известно из KYC): высокий.
        if "is_resident" in features.columns:
            resident_signal = features["is_resident"].fillna(1).values.astype(float)
        else:
            # По умолчанию: не знаем резидентство, средний риск
            resident_signal = np.full(len(features), 0.5)

        # Крупные суммы = повышенный FX риск
        if "amount_usd" in features.columns:
            amount_signal = features["amount_usd"].fillna(0).clip(0, 100_000).values / 100_000.0
        else:
            amount_signal = np.zeros(len(features))

        fx = 0.6 * (1.0 - resident_signal) + 0.4 * amount_signal
        return np.clip(fx, 0.0, 1.0)

    def _zone(self, score: float, threshold_red: float) -> str:
        if score < 0.3:
            return "GREEN"
        elif score < threshold_red:
            return "YELLOW"
        return "RED"