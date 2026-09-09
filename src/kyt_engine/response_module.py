from kyt_engine.core.contracts import ScoreResult


class ResponseModule:
    def pause_contract(self, tx_id: str) -> dict:
        return {"action": "pause", "tx_id": tx_id, "status": "initiated"}

    def revoke_allowance(self, tx_id: str, spender: str) -> dict:
        return {"action": "revoke", "tx_id": tx_id, "spender": spender, "status": "initiated"}

    def emergency_withdraw(self, tx_id: str, to: str) -> dict:
        return {"action": "emergency_withdraw", "tx_id": tx_id, "to": to, "status": "initiated"}


def execute_response(result: ScoreResult, module: ResponseModule) -> dict:
    triage_level = result.triage_level

    if triage_level == "BLOCKED":
        module.pause_contract(result.tx_id)
        module.revoke_allowance(result.tx_id, spender="0x0000000000000000000000000000000000000000")
        module.emergency_withdraw(result.tx_id, to="0x0000000000000000000000000000000000000000")
        return {"status": "executed", "actions": ["pause", "revoke", "emergency_withdraw"], "tx_id": result.tx_id}

    if triage_level == "FLAGGED":
        return {"status": "logged", "decision": "FLAGGED", "tx_id": result.tx_id}

    if triage_level == "APPROVED":
        return {"status": "approved", "decision": "APPROVED", "tx_id": result.tx_id}

    return {"status": "unknown", "decision": triage_level, "tx_id": result.tx_id}