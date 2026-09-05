from dataclasses import dataclass, field


@dataclass(frozen=True)
class TxRecord:
    tx_id: str
    address: str
    from_address: str
    to_address: str
    value: float
    gas_price: float
    gas_used: float
    timestamp: float
    block_number: int
    features: dict[str, float] | None = None


@dataclass
class FeatureVector:
    values: dict[str, float]

    def as_dict(self) -> dict[str, float]:
        return dict(self.values)


@dataclass
class ScoreResult:
    tx_id: str
    risk_score: float
    risk_zone: str
    triage_level: str
    lgbm_proba: float
    k_score: float
    vae_anomaly: float
    external_risk: float
    reasons: list[dict[str, float | str]]


@dataclass
class AuditRecord:
    audit_id: str
    tx_id: str
    requested_by: str
    model_version: str
    input_snapshot: dict[str, float | str]
    features_hash: str
    scores: dict[str, float]
    risk_score: float
    risk_zone: str
    triage_level: str
    reasons: list[dict[str, float | str]]
    created_at: str