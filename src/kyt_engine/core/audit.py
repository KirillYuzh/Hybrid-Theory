import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from kyt_engine.core.contracts import AuditRecord, ScoreResult, TxRecord


def _features_hash(values: dict[str, float]) -> str:
    payload = json.dumps(values, sort_keys=True, default=str).encode()
    return hashlib.sha256(payload).hexdigest()


class AuditLog:
    def __init__(self, path: Path) -> None:
        self._path = path
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, tx: TxRecord, score: ScoreResult, requested_by: str, model_version: str) -> None:
        record = AuditRecord(
            audit_id=str(uuid4()),
            tx_id=tx.tx_id,
            requested_by=requested_by,
            model_version=model_version,
            input_snapshot={
                "from_address": tx.from_address,
                "to_address": tx.to_address,
                "value": tx.value,
                "gas_price": tx.gas_price,
                "gas_used": tx.gas_used,
                "timestamp": tx.timestamp,
                "block_number": tx.block_number,
            },
            features_hash=_features_hash(score.reasons and {
                r["feature"]: float(r["value"]) for r in score.reasons
            } or {}),
            scores={
                "lgbm_proba": score.lgbm_proba,
                "k_score": score.k_score,
                "vae_anomaly": score.vae_anomaly,
                "external_risk": score.external_risk,
            },
            risk_score=score.risk_score,
            risk_zone=score.risk_zone,
            triage_level=score.triage_level,
            reasons=score.reasons,
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        with self._path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record.__dict__, ensure_ascii=False) + "\n")