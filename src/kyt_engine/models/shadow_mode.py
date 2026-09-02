from __future__ import annotations

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional
from enum import Enum


class SystemMode(Enum):
    SHADOW = "shadow"
    PRODUCTION = "production"


@dataclass
class ShadowConfig:
    min_shadow_days: int = 90
    max_shadow_days: int = 180
    target_traffic_share: float = 1.0
    fp_rate_target: float = 0.15
    compliance_match_target: float = 0.99
    weekly_review_enabled: bool = True
    dry_run: bool = True


@dataclass
class ShadowComparison:
    tx_id: int
    timestamp: pd.Timestamp
    
    old_decision: str
    old_reason: str
    
    model_decision: str
    model_confidence: float
    model_score: float
    
    is_agreement: bool
    agreement_type: str
    risk_amount_usd: float | None = None


@dataclass
class ShadowMetrics:
    period_start: pd.Timestamp
    period_end: pd.Timestamp
    total_transactions: int
    
    agreement_rate: float
    agreement_by_decision: dict[str, float]
    
    fp_rate_new: float
    fp_rate_old: float | None
    fn_rate_new: float
    fn_rate_old: float | None
    
    compliance_match_rate: float
    
    avg_model_latency_ms: float
    avg_old_process_latency_ms: float
    
    escalated_disagreements: int
    high_value_disagreements: int
    
    transition_ready: bool
    transition_blockers: list[str] = field(default_factory=list)


@dataclass
class ShadowReport:
    report_id: str
    generated_at: pd.Timestamp
    
    config: ShadowConfig
    
    cumulative_metrics: ShadowMetrics
    weekly_metrics: list[ShadowMetrics]
    
    fp_rate_trajectory: list[tuple[pd.Timestamp, float]]
    compliance_trajectory: list[tuple[pd.Timestamp, float]]
    
    transition_criteria_status: dict[str, dict]
    
    current_mode: SystemMode
    projected_production_date: pd.Timestamp | None
    
    total_cost_saved_usd: float | None = None
    total_false_positives_averted: int = 0


class ShadowModeEngine:
    def __init__(
        self,
        model_scorer,
        old_process_func,
        compliance_checker=None,
        config: ShadowConfig | None = None,
    ):
        self._model = model_scorer
        self._old_process = old_process_func
        self._compliance = compliance_checker
        self._config = config or ShadowConfig()
        
        self._comparisons: list[ShadowComparison] = []
        self._weekly_batches: list[list[ShadowComparison]] = []
        
        self._current_mode = SystemMode.SHADOW
        self._shadow_start: pd.Timestamp | None = None
        self._last_review: pd.Timestamp | None = None
        
        self._fp_samples: list[bool] = []
        self._compliance_samples: list[bool] = []
        
        self._transaction_costs: dict[int, float] = {}
        self._cost_saved: float = 0.0
        self._fp_averted: int = 0

    def start(self) -> None:
        self._shadow_start = pd.Timestamp.now()
        self._last_review = self._shadow_start
        self._current_mode = SystemMode.SHADOW

    def process_transaction(
        self,
        tx_id: int,
        features: pd.DataFrame,
        old_decision: str,
        old_reason: str,
        risk_amount_usd: float | None = None,
        ground_truth_label: bool | None = None,
    ) -> ShadowComparison:
        model_score = self._model.score(features, pd.Series([tx_id]))[0]
        
        model_decision = self._decide_from_score(model_score.risk_score)
        model_confidence = max(model_score.lgbm_proba, 1 - model_score.lgbm_proba)
        
        is_agreement = old_decision == model_decision
        agreement_type = self._get_agreement_type(old_decision, model_decision)
        
        comparison = ShadowComparison(
            tx_id=tx_id,
            timestamp=pd.Timestamp.now(),
            old_decision=old_decision,
            old_reason=old_reason,
            model_decision=model_decision,
            model_confidence=float(model_confidence),
            model_score=float(model_score.risk_score),
            is_agreement=is_agreement,
            agreement_type=agreement_type,
            risk_amount_usd=risk_amount_usd,
        )
        
        self._comparisons.append(comparison)
        
        if ground_truth_label is not None:
            self._update_fp_tracking(comparison, ground_truth_label)
        
        if risk_amount_usd is not None:
            self._transaction_costs[tx_id] = risk_amount_usd
            self._update_cost_savings(comparison, risk_amount_usd)
        
        return comparison

    def _decide_from_score(self, score: float) -> str:
        if score < 0.3:
            return "APPROVE"
        elif score < 0.7:
            return "REVIEW"
        else:
            return "BLOCK"

    def _get_agreement_type(self, old: str, model: str) -> str:
        if old == model:
            return "full_agreement"
        
        old_order = {"APPROVE": 0, "REVIEW": 1, "BLOCK": 2}
        diff = old_order[model] - old_order[old]
        
        if diff == 1:
            return "model_more_conservative"
        elif diff == -1:
            return "model_more_aggressive"
        elif diff > 1:
            return "model_blocks_old_reviewed"
        else:
            return "model_approves_old_blocked"

    def _update_fp_tracking(
        self,
        comparison: ShadowComparison,
        ground_truth: bool,
    ) -> None:
        is_fp_new = (comparison.model_decision in ("BLOCK", "REVIEW")) and (not ground_truth)
        is_fp_old = (comparison.old_decision in ("BLOCK", "REVIEW")) and (not ground_truth)
        
        self._fp_samples.append(is_fp_new)
        
        is_compliant = True
        if self._compliance:
            is_compliant = self._compliance.check(
                comparison.tx_id,
                comparison.model_decision,
                comparison.model_score,
            )
        self._compliance_samples.append(is_compliant)

    def _update_cost_savings(
        self,
        comparison: ShadowComparison,
        risk_amount_usd: float,
    ) -> None:
        if comparison.agreement_type == "full_agreement":
            return
        
        if comparison.model_decision == "APPROVE" and comparison.old_decision in ("BLOCK", "REVIEW"):
            self._cost_saved += risk_amount_usd * 0.001
            if not comparison.is_agreement:
                self._fp_averted += 1
        elif comparison.model_decision in ("BLOCK", "REVIEW") and comparison.old_decision == "APPROVE":
            self._cost_saved -= risk_amount_usd * 0.01

    def _compute_cumulative_metrics(self) -> ShadowMetrics:
        if not self._comparisons:
            return ShadowMetrics(
                period_start=pd.Timestamp.now(),
                period_end=pd.Timestamp.now(),
                total_transactions=0,
                agreement_rate=0.0,
                agreement_by_decision={},
                fp_rate_new=0.0,
                fp_rate_old=None,
                fn_rate_new=0.0,
                fn_rate_old=None,
                compliance_match_rate=0.0,
                avg_model_latency_ms=0.0,
                avg_old_process_latency_ms=0.0,
                escalated_disagreements=0,
                high_value_disagreements=0,
                transition_ready=False,
            )
        
        n = len(self._comparisons)
        agreements = sum(1 for c in self._comparisons if c.is_agreement)
        
        agreement_by_type: dict[str, int] = {}
        for c in self._comparisons:
            t = c.agreement_type
            agreement_by_type[t] = agreement_by_type.get(t, 0) + 1
        
        disagreement_mask = [not c.is_agreement for c in self._comparisons]
        escalated = sum(
            1 for c in self._comparisons
            if not c.is_agreement and c.model_decision == "BLOCK"
        )
        high_value = sum(
            1 for c in self._comparisons
            if not c.is_agreement
            and c.risk_amount_usd is not None
            and c.risk_amount_usd > 100_000
        )
        
        fp_rate = sum(self._fp_samples) / len(self._fp_samples) if self._fp_samples else 0.0
        compliance_rate = sum(self._compliance_samples) / len(self._compliance_samples) if self._compliance_samples else 0.0
        
        criteria = self._check_transition_criteria(fp_rate, compliance_rate)
        
        return ShadowMetrics(
            period_start=self._comparisons[0].timestamp,
            period_end=self._comparisons[-1].timestamp,
            total_transactions=n,
            agreement_rate=agreements / n,
            agreement_by_decision={k: v / n for k, v in agreement_by_type.items()},
            fp_rate_new=fp_rate,
            fp_rate_old=None,
            fn_rate_new=0.0,
            fn_rate_old=None,
            compliance_match_rate=compliance_rate,
            avg_model_latency_ms=0.0,
            avg_old_process_latency_ms=0.0,
            escalated_disagreements=escalated,
            high_value_disagreements=high_value,
            transition_ready=criteria["ready"],
            transition_blockers=criteria["blockers"],
        )

    def _check_transition_criteria(
        self,
        fp_rate: float,
        compliance_rate: float,
    ) -> dict:
        blockers = []
        
        if fp_rate >= self._config.fp_rate_target:
            blockers.append(f"FP-rate {fp_rate:.1%} >= target {self._config.fp_rate_target:.1%}")
        
        if compliance_rate < self._config.compliance_match_target:
            blockers.append(f"Compliance match {compliance_rate:.1%} < target {self._config.compliance_match_target:.1%}")
        
        if len(self._comparisons) < 1000:
            blockers.append(f"Insufficient transactions ({len(self._comparisons)} < 1000)")
        
        shadow_days = (pd.Timestamp.now() - self._shadow_start).days if self._shadow_start else 0
        if shadow_days < self._config.min_shadow_days:
            blockers.append(f"Shadow period {shadow_days} days < minimum {self._config.min_shadow_days} days")
        
        return {
            "ready": len(blockers) == 0,
            "fp_rate_criterion": fp_rate < self._config.fp_rate_target,
            "compliance_criterion": compliance_rate >= self._config.compliance_match_target,
            "sample_size_criterion": len(self._comparisons) >= 1000,
            "min_period_criterion": shadow_days >= self._config.min_shadow_days,
            "blockers": blockers,
        }

    def generate_weekly_report(self) -> ShadowReport:
        metrics = self._compute_cumulative_metrics()
        
        fp_traj = []
        compliance_traj = []
        
        if len(self._fp_samples) > 0:
            cumulative_fp = np.cumsum(self._fp_samples) / np.arange(1, len(self._fp_samples) + 1)
            for i, val in enumerate(cumulative_fp[::max(1, len(cumulative_fp) // 20)]):
                fp_traj.append((self._comparisons[min(i * max(1, len(cumulative_fp) // 20), len(self._comparisons) - 1)].timestamp, val))
        
        if len(self._compliance_samples) > 0:
            cumulative_compliance = np.cumsum(self._compliance_samples) / np.arange(1, len(self._compliance_samples) + 1)
            for i, val in enumerate(cumulative_compliance[::max(1, len(cumulative_compliance) // 20)]):
                compliance_traj.append((self._comparisons[min(i * max(1, len(cumulative_compliance) // 20), len(self._comparisons) - 1)].timestamp, val))
        
        criteria = self._check_transition_criteria(metrics.fp_rate_new, metrics.compliance_match_rate)
        
        projected_date = None
        if not criteria["ready"] and self._shadow_start:
            remaining_days = self._config.max_shadow_days - (pd.Timestamp.now() - self._shadow_start).days
            if remaining_days > 0:
                projected_date = pd.Timestamp.now() + timedelta(days=remaining_days)
        
        return ShadowReport(
            report_id=f"shadow_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}",
            generated_at=pd.Timestamp.now(),
            config=self._config,
            cumulative_metrics=metrics,
            weekly_metrics=[],
            fp_rate_trajectory=fp_traj,
            compliance_trajectory=compliance_traj,
            transition_criteria_status=criteria,
            current_mode=self._current_mode,
            projected_production_date=projected_date,
            total_cost_saved_usd=self._cost_saved if self._cost_saved else None,
            total_false_positives_averted=self._fp_averted,
        )

    def can_transition_to_production(self) -> tuple[bool, list[str]]:
        if self._current_mode != SystemMode.SHADOW:
            return False, ["Already in production mode"]
        
        metrics = self._compute_cumulative_metrics()
        criteria = self._check_transition_criteria(
            metrics.fp_rate_new,
            metrics.compliance_match_rate,
        )
        
        return criteria["ready"], criteria["blockers"]

    def transition_to_production(self) -> bool:
        can_transition, blockers = self.can_transition_to_production()
        
        if not can_transition:
            return False
        
        self._current_mode = SystemMode.PRODUCTION
        
        final_report = self.generate_weekly_report()
        
        return True

    def get_disagreements(
        self,
        min_risk_amount: float | None = None,
        agreement_type_filter: str | None = None,
        limit: int = 100,
    ) -> list[ShadowComparison]:
        filtered = self._comparisons
        
        if not filtered:
            return []
        
        filtered = [c for c in filtered if not c.is_agreement]
        
        if min_risk_amount is not None:
            filtered = [c for c in filtered if c.risk_amount_usd is not None and c.risk_amount_usd >= min_risk_amount]
        
        if agreement_type_filter:
            filtered = [c for c in filtered if c.agreement_type == agreement_type_filter]
        
        filtered.sort(key=lambda c: c.risk_amount_usd or 0, reverse=True)
        
        return filtered[:limit]

    def get_summary(self) -> dict:
        metrics = self._compute_cumulative_metrics()
        criteria = self._check_transition_criteria(metrics.fp_rate_new, metrics.compliance_match_rate)
        
        shadow_days = (pd.Timestamp.now() - self._shadow_start).days if self._shadow_start else 0
        
        return {
            "mode": self._current_mode.value,
            "shadow_days": shadow_days,
            "total_transactions": len(self._comparisons),
            "agreement_rate": metrics.agreement_rate,
            "fp_rate": metrics.fp_rate_new,
            "compliance_match_rate": metrics.compliance_match_rate,
            "transition_ready": criteria["ready"],
            "blockers": criteria["blockers"],
            "cost_saved_usd": self._cost_saved,
            "fp_averted": self._fp_averted,
        }
