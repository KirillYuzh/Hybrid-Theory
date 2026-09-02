"""Reactive module — fast-path scoring for every transaction.

Business question: "Is this transaction similar to a known fraudulent scheme?"
SLA: < 200 ms per transaction.
Metrics: Precision@RED, Recall@RED, FPR (user-level), p95 latency.

Components:
- KScoreCalculator: deviation from behavioral baseline (0 = normal, 1 = unusual)
- TriageSystem: three-level routing (auto_close / priority / escalation)
- DualScorer: AML-Risk (115-FZ) + FX-Risk (173-FZ) orthogonal scores
- CascadeFilter: multi-stage screening (Stage 0-3) for 10x compute savings
"""