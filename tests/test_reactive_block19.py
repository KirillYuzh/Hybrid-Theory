"""Тесты для Блока 19: K-Score + Triage + DualScorer + Cascade."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from kyt_engine.models.kscore import KScoreCalculator
from kyt_engine.models.triage import AlertPriority, TriageSystem
from kyt_engine.reactive.scorer import DualScorer
from kyt_engine.reactive.cascade import ScreeningCascade, Stage


@pytest.fixture
def sample_features() -> pd.DataFrame:
    """Создаёт синтетический набор фич для тестов."""
    n = 100
    rng = np.random.default_rng(42)

    # Два адреса: один стабильный (малые z-scores), один аномальный (большие)
    addresses = ["addr_a"] * (n // 2) + ["addr_b"] * (n // 2)

    data = {
        "txId": list(range(n)),
        "time_step": list(range(n)),
        "from_address": addresses,
        "to_address": addresses,
        "in_degree": rng.poisson(2, n),
        "out_degree": rng.poisson(2, n),
        "amount_usd": rng.uniform(10, 1000, n),
        "hour_of_day": rng.integers(0, 24, n),
    }
    return pd.DataFrame(data)


class TestKScore:
    def test_fit_and_score(self, sample_features):
        kscore = KScoreCalculator(window_days=(30, 60, 90))
        kscore.fit(sample_features, sample_features["time_step"])

        scores = kscore.score(sample_features)
        assert len(scores) == len(sample_features)
        assert scores.min() >= 0.0
        assert scores.max() <= 1.0

    def test_classify(self, sample_features):
        kscore = KScoreCalculator()
        kscore.fit(sample_features, sample_features["time_step"])
        scores = kscore.score(sample_features)
        zones = kscore.classify(scores)

        assert zones.isin(["GREEN", "YELLOW", "RED"]).all()

    def test_update_rolling_window(self, sample_features):
        kscore = KScoreCalculator()
        kscore.fit(sample_features.iloc[:50], sample_features["time_step"].iloc[:50])
        kscore.update(sample_features.iloc[50:], time_step=45)

        scores = kscore.score(sample_features)
        assert len(scores) == len(sample_features)


class TestTriage:
    def test_triage_levels(self):
        triage = TriageSystem()

        k_scores = pd.Series([0.1, 0.5, 0.8, 0.2, 0.9])
        proba = pd.Series([0.95, 0.6, 0.5, 0.85, 0.55])
        entropy = pd.Series([0.1, 0.5, 0.4, 0.3, 0.2])

        result = triage.triage(k_scores, proba, entropy)

        assert result["priority"].iloc[0] == AlertPriority.AUTO_CLOSE  # 0.1 + 0.95 conf
        assert result["priority"].iloc[2] == AlertPriority.ESCALATION   # 0.8 kscore
        assert result["priority"].iloc[1] == AlertPriority.PRIORITY     # 0.5 kscore

    def test_cost_sensitive_routing(self):
        triage = TriageSystem(high_value_threshold=100_000.0)

        k_scores = pd.Series([0.2, 0.2])
        proba = pd.Series([0.95, 0.95])
        entropy = pd.Series([0.1, 0.1])
        amounts = pd.Series([50.0, 500_000.0])  # второй — high-value

        result = triage.triage(k_scores, proba, entropy, amount_usd=amounts)

        # Первая: маленькая сумма → автозакрытие
        assert result["priority"].iloc[0] == AlertPriority.AUTO_CLOSE
        # Вторая: high-value → НЕ автозакрытие (даже при высокой уверенности)
        assert result["priority"].iloc[1] != AlertPriority.AUTO_CLOSE

    def test_statistics(self):
        triage = TriageSystem()
        k_scores = pd.Series([0.1, 0.5, 0.8])
        proba = pd.Series([0.95, 0.6, 0.5])
        entropy = pd.Series([0.1, 0.5, 0.4])

        result = triage.triage(k_scores, proba, entropy)
        stats = triage.statistics(result)

        assert stats["total"] == 3
        assert stats["auto_close_pct"] > 0
        assert stats["escalation_pct"] > 0


class TestDualScorer:
    def test_dual_score(self, sample_features):
        scorer = DualScorer()
        results = scorer.score(sample_features)

        assert len(results) == len(sample_features)
        for r in results:
            assert 0.0 <= r.aml_risk <= 1.0
            assert 0.0 <= r.fx_risk <= 1.0
            assert r.aml_zone in ("GREEN", "YELLOW", "RED")
            assert r.fx_zone in ("GREEN", "YELLOW", "RED")

    def test_fx_risk_resident_vs_nonresident(self, sample_features):
        scorer = DualScorer()
        df = sample_features.copy()
        df["is_resident"] = [1] * 50 + [0] * 50

        results = scorer.score(df)
        resident_fx = np.mean([r.fx_risk for r in results[:50]])
        nonresident_fx = np.mean([r.fx_risk for r in results[50:]])

        # Нерезиденты должны иметь более высокий FX риск
        assert nonresident_fx > resident_fx


class TestScreeningCascade:
    def test_cascade_basic(self, sample_features):
        cascade = ScreeningCascade()
        results = cascade.process(sample_features, sample_features["txId"])

        assert len(results) == len(sample_features)
        for r in results:
            assert r.action in ("PASS", "REVIEW", "HOLD")
            assert 0.0 <= r.score <= 1.0

    def test_cascade_savings(self, sample_features):
        cascade = ScreeningCascade(bloom_save_rate=0.75)
        results = cascade.process(sample_features, sample_features["txId"])
        savings = cascade.compute_savings(len(sample_features), results)

        # Большинство должно пройти через Stage 0
        assert savings["n_total"] == len(sample_features)
        assert savings["pass_pct"] > 50  # как минимум 75% проходит через селективный фильтр
        assert savings["cost_savings_x"] > 1  # экономия значительна

    def test_cascade_with_model(self, sample_features):
        """Проверяет каскад с моделью (эмуляция fast model)."""
        class FakeModel:
            def predict_proba(self, X):
                return np.column_stack([1 - np.linspace(0, 1, len(X)), np.linspace(0, 1, len(X))])

        cascade = ScreeningCascade(
            fast_model=FakeModel(),
            bloom_save_rate=0.5,  # пропустить половину через stage 0
            hold_threshold=0.7,
        )
        results = cascade.process(sample_features, sample_features["txId"])

        # Часть проходит stage 0, часть — через модель
        stages = {r.stage for r in results}
        assert Stage.S0_BLOOM in stages
        # С моделью должно быть больше REVIEW (высокие proba)
        n_review = sum(1 for r in results if r.action == "REVIEW")
        assert n_review > 0