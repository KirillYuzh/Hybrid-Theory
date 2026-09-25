from math import isfinite
from typing import Any

import numpy as np


class BehaviorDriftError(ValueError):
    pass


def validate_drift_schedule(
    schedule: dict[str, Any],
    step_min: int = 1,
    step_max: int = 49,
    known_profile_ids: set[str] | None = None,
) -> dict[str, Any]:
    """Validate a contiguous drift schedule.

    Parameters
    ----------
    schedule : dict[str, Any]
        Schedule containing an ordered ``phases`` list.
    step_min : int, default 1
        First time step that phases must cover.
    step_max : int, default 49
        Last time step that phases must cover.
    known_profile_ids : set[str] or None, optional
        Allowed profile IDs for phase-level novel profiles. If omitted, profile
        registration is not checked by this function.

    Returns
    -------
    dict[str, Any]
        Validated schedule containing copies of the original phase mappings.

    Raises
    ------
    BehaviorDriftError
        If phase fields, bounds, probability, IDs, or coverage violate the schedule
        contract.
    """
    phases = schedule.get("phases")
    if not isinstance(phases, list) or not phases:
        raise BehaviorDriftError("drift.phases must be a non-empty list")
    expected_start = step_min
    seen: set[str] = set()
    validated: list[dict[str, Any]] = []
    for phase in phases:
        if not isinstance(phase, dict):
            raise BehaviorDriftError("drift phase must be a mapping")
        required = {
            "phase_id",
            "step_min",
            "step_max",
            "illicit_keep_probability",
            "novel_profile_ids",
        }
        if set(phase) != required:
            raise BehaviorDriftError("drift phase fields do not match the contract")
        phase_id = phase["phase_id"]
        lo = phase["step_min"]
        hi = phase["step_max"]
        probability = phase["illicit_keep_probability"]
        novel = phase["novel_profile_ids"]
        if not isinstance(phase_id, str) or not phase_id or phase_id in seen:
            raise BehaviorDriftError("drift phase IDs must be unique strings")
        if (
            isinstance(lo, bool)
            or isinstance(hi, bool)
            or not isinstance(lo, int)
            or not isinstance(hi, int)
        ):
            raise BehaviorDriftError("drift phase bounds must be integers")
        if lo != expected_start or hi < lo or hi > step_max:
            raise BehaviorDriftError("drift phases must be contiguous and bounded")
        if isinstance(probability, bool) or not isinstance(probability, (int, float)):
            raise BehaviorDriftError("drift probability must be numeric")
        if not isfinite(float(probability)) or not 0 <= probability <= 1:
            raise BehaviorDriftError("drift probability must be in [0, 1]")
        if not isinstance(novel, list) or any(not isinstance(item, str) for item in novel):
            raise BehaviorDriftError("novel_profile_ids must be a string list")
        if len(novel) != len(set(novel)):
            raise BehaviorDriftError("novel_profile_ids must be unique")
        if known_profile_ids is not None and not set(novel) <= known_profile_ids:
            raise BehaviorDriftError("novel profile is not registered")
        seen.add(phase_id)
        validated.append(dict(phase))
        expected_start = hi + 1
    if expected_start != step_max + 1:
        raise BehaviorDriftError("drift phases must cover the complete timeline")
    return {"phases": validated}


def phase_at(schedule: dict[str, Any], step: int) -> dict[str, Any]:
    for phase in schedule["phases"]:
        if phase["step_min"] <= step <= phase["step_max"]:
            return phase
    raise BehaviorDriftError(f"no drift phase for step {step}")


def profile_is_eligible(profile: Any, step: int, schedule: dict[str, Any]) -> bool:
    phase = phase_at(schedule, step)
    if profile.negative_kind == "novel_illicit":
        return profile.profile_id in phase["novel_profile_ids"]
    if profile.class_policy == "illicit" and phase["illicit_keep_probability"] == 0:
        return False
    return True


def allocate_step_slots(volume: np.ndarray, n_txs: int) -> list[int]:
    """Allocate exact transaction slots across volume-weighted time steps.

    Integer counts use largest-remainder rounding with stable tie ordering. A
    non-positive transaction count produces an empty allocation; an all-zero
    weight profile falls back to uniform weights.

    Parameters
    ----------
    volume : numpy.ndarray
        Finite non-negative volume weights, flattened before allocation.
    n_txs : int
        Exact number of transaction slots to distribute.

    Returns
    -------
    list[int]
        One-based time steps in ascending order, repeated according to allocation.

    Raises
    ------
    BehaviorDriftError
        If a positive allocation is requested from an empty or invalid weight profile.
    """
    if n_txs <= 0:
        return []
    weights = np.asarray(volume, dtype=float).reshape(-1)
    if len(weights) == 0 or not np.all(np.isfinite(weights)) or np.any(weights < 0):
        raise BehaviorDriftError("volume must contain finite non-negative weights")
    if float(weights.sum()) <= 0:
        weights = np.ones(len(weights), dtype=float)
    exact = weights / weights.sum() * n_txs
    counts = np.floor(exact).astype(int)
    remainder = n_txs - int(counts.sum())
    order = np.argsort(-(exact - counts), kind="stable")
    for index in order[:remainder]:
        counts[index] += 1
    return [int(step + 1) for step, count in enumerate(counts) for _ in range(int(count))]


__all__ = [
    "BehaviorDriftError",
    "allocate_step_slots",
    "phase_at",
    "profile_is_eligible",
    "validate_drift_schedule",
]
