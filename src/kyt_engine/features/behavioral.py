import numpy as np
import pandas as pd
from collections import Counter

from kyt_engine.features._utils import (
    counting_entropy,
    discretized_entropy,
    extract_counterparties,
    safe_float,
)


def _reaction_speed_features(group: pd.DataFrame) -> dict[str, float]:
    sorted_group = group.sort_values("timestamp")
    address = sorted_group["address"].iloc[0]

    reaction_times: list[float] = []
    for i in range(1, len(sorted_group)):
        prev = sorted_group.iloc[i - 1]
        curr = sorted_group.iloc[i]
        is_incoming = prev["from_address"] == address and curr["to_address"] == address
        is_outgoing = prev["to_address"] == address and curr["from_address"] == address
        if is_incoming or is_outgoing:
            reaction_times.append(curr["timestamp"] - prev["timestamp"])

    if len(reaction_times) == 0:
        return {
            "reaction_speed_mean": 0.0,
            "reaction_speed_median": 0.0,
            "fast_reaction_ratio": 0.0,
        }

    rt = np.array(reaction_times)
    median_rt = np.median(rt)
    return {
        "reaction_speed_mean": float(np.mean(rt)),
        "reaction_speed_median": float(median_rt),
        "fast_reaction_ratio": float(np.mean(rt < 0.5 * median_rt)) if median_rt > 0 else 0.0,
    }


def _circadian_rhythm_features(group: pd.DataFrame) -> dict[str, float]:
    hours = group["hour"].values.astype(int)
    hour_counts = np.bincount(hours, minlength=24).astype(float)
    total = hour_counts.sum()

    if total == 0:
        return {
            "circadian_regularity": 0.0,
            "peak_activity_hour": 0.0,
            "nocturnal_ratio": 0.0,
            "work_hours_ratio": 0.0,
            "circadian_amplitude": 0.0,
        }

    night_mask = np.array([1 if h in set(range(0, 6)) | set(range(22, 24)) else 0 for h in range(24)])
    work_mask = np.array([1 if h in set(range(9, 17)) else 0 for h in range(24)])

    peak_idx = np.argmax(hour_counts)
    trough_idx = np.argmin(hour_counts)

    return {
        "circadian_regularity": 1.0 - np.std(hours) / 12.0,
        "peak_activity_hour": float(peak_idx),
        "nocturnal_ratio": np.sum(hour_counts * night_mask) / total,
        "work_hours_ratio": np.sum(hour_counts * work_mask) / total,
        "circadian_amplitude": (hour_counts[peak_idx] - hour_counts[trough_idx]) / total,
    }


def _gas_strategy_features(group: pd.DataFrame, global_gas_median: float) -> dict[str, float]:
    gas_prices = group["gas_price"].values.astype(float)

    if len(gas_prices) == 0:
        return {
            "gas_aggressiveness": 0.0, "gas_bid_ratio": 0.0,
            "gas_patience": 0.0, "gas_optimization_score": 0.0,
            "gas_predictability": 0.0, "mean_gas_percentile": 0.0,
        }

    sorted_gas = np.sort(gas_prices)
    ranks = np.array([np.searchsorted(sorted_gas, g) for g in gas_prices])
    gas_mean = np.mean(gas_prices)

    features = {
        "gas_aggressiveness": float(np.mean(ranks / len(gas_prices))),
        "gas_optimization_score": 1.0 - min(np.std(gas_prices) / gas_mean, 1.0) if gas_mean > 0 else 0.0,
        "mean_gas_percentile": float(np.mean(ranks) / len(gas_prices)),
    }

    if global_gas_median > 0:
        features["gas_bid_ratio"] = float(np.mean(gas_prices > global_gas_median))
        features["gas_patience"] = float(np.mean(gas_prices < global_gas_median))
    else:
        features["gas_bid_ratio"] = 0.0
        features["gas_patience"] = 0.0

    if len(gas_prices) > 2:
        n = len(gas_prices)
        gas_sorted_by_time = group.sort_values("timestamp")["gas_price"].values.astype(float)
        acf1 = np.corrcoef(gas_sorted_by_time[:n - 1], gas_sorted_by_time[1:])[0, 1] if n > 1 else 0.0
        features["gas_predictability"] = float(max(0.0, acf1))
    else:
        features["gas_predictability"] = 0.0

    return features


def _transaction_interval_features(group: pd.DataFrame) -> dict[str, float]:
    intervals = group["interval"].dropna().values.astype(float)

    if len(intervals) == 0:
        return {
            "burst_ratio": 0.0,
            "long_pause_ratio": 0.0,
            "rapid_fire_ratio": 0.0,
        }

    median_int = np.median(intervals)
    burst_threshold = np.percentile(intervals, 25)

    return {
        "burst_ratio": float(np.mean(intervals < burst_threshold)),
        "long_pause_ratio": float(np.mean(intervals > 3 * median_int)) if median_int > 0 else 0.0,
        "rapid_fire_ratio": float(np.mean(intervals < 0.1 * median_int)) if median_int > 0 else 0.0,
    }


def _shannon_entropy_features(group: pd.DataFrame) -> dict[str, float]:
    address = group["address"].iloc[0]
    counterparties = extract_counterparties(group, address)

    return {
        "value_distribution_entropy": discretized_entropy(group["value"].values.astype(float)),
        "behavioral_counterparty_entropy": counting_entropy(counterparties),
        "temporal_entropy": _temporal_entropy(group),
        "behavioral_gas_entropy": discretized_entropy(group["gas_price"].values.astype(float)),
    }


def _temporal_entropy(group: pd.DataFrame) -> float:
    hours = group["hour"].values.astype(int)
    hour_counts = np.bincount(hours, minlength=24).astype(float)
    total = hour_counts.sum()
    if total > 0:
        probs = hour_counts / total
        probs = probs[probs > 0]
        return float(-np.sum(probs * np.log2(probs)))
    return 0.0


def _counterparty_diversity_features(group: pd.DataFrame) -> dict[str, float]:
    address = group["address"].iloc[0]
    counterparties = extract_counterparties(group, address)

    if len(counterparties) == 0:
        return {
            "counterparty_diversity": 0.0, "counterparty_growth_rate": 0.0,
            "counterparty_stability": 0.0, "new_vs_returned_ratio": 0.0,
            "counterparty_herfindahl": 0.0,
        }

    unique_cp = len(set(counterparties))
    cp_counter = Counter(counterparties)

    sorted_group = group.sort_values("timestamp")
    seen: set[str] = set()
    new_per_tx: list[float] = []
    for _, row in sorted_group.iterrows():
        cp = str(row["to_address"]) if str(row["from_address"]) == address else str(row["from_address"])
        was_new = cp not in seen
        seen.add(cp)
        new_per_tx.append(1.0 if was_new else 0.0)

    growth_rate = 0.0
    if len(new_per_tx) >= 2:
        half = len(new_per_tx) // 2
        first_half_rate = np.mean(new_per_tx[:half])
        second_half_rate = np.mean(new_per_tx[half:])
        growth_rate = second_half_rate - first_half_rate

    seen_ordered: set[str] = set()
    new_count = 0
    return_count = 0
    for cp in counterparties:
        if cp not in seen_ordered:
            new_count += 1
            seen_ordered.add(cp)
        else:
            return_count += 1

    new_vs_returned = new_count / return_count if return_count > 0 else float(new_count)

    freqs = np.array(list(cp_counter.values()), dtype=float)
    freqs = freqs / freqs.sum()
    herfindahl = 1.0 - np.sum(freqs ** 2)

    return {
        "counterparty_diversity": unique_cp / len(counterparties),
        "counterparty_growth_rate": float(growth_rate),
        "counterparty_stability": sum(1 for c in cp_counter.values() if c > 1) / unique_cp if unique_cp > 0 else 0.0,
        "new_vs_returned_ratio": float(new_vs_returned),
        "counterparty_herfindahl": float(herfindahl),
    }


_NUM_BEHAVIORAL_FEATURES = 26


def extract_behavioral_features(
    df: pd.DataFrame,
    global_gas_median: float | None = None,
) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    df = df.copy()
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce").fillna(0.0)
    df["gas_price"] = pd.to_numeric(df["gas_price"], errors="coerce").fillna(0.0)
    df["gas_used"] = pd.to_numeric(df["gas_used"], errors="coerce").fillna(0.0)

    ts = pd.to_datetime(df["timestamp"], unit="s", errors="coerce")
    fallback = pd.to_datetime(df["timestamp"], errors="coerce")
    ts = ts.fillna(fallback)

    df["hour"] = ts.dt.hour.fillna(0).astype(int)
    df["day_of_week"] = ts.dt.dayofweek.fillna(0).astype(int)

    df = df.sort_values(["address", "timestamp"])
    df["interval"] = df.groupby("address")["timestamp"].diff().fillna(0.0).clip(lower=0.0)

    if global_gas_median is None:
        global_gas_median = float(df["gas_price"].median())

    all_features: list[dict[str, float]] = []
    addresses: list[str] = []

    for address, group in df.groupby("address"):
        feat: dict[str, float] = {}
        feat.update(_reaction_speed_features(group))
        feat.update(_circadian_rhythm_features(group))
        feat.update(_gas_strategy_features(group, global_gas_median))
        feat.update(_transaction_interval_features(group))
        feat.update(_shannon_entropy_features(group))
        feat.update(_counterparty_diversity_features(group))

        all_features.append(feat)
        addresses.append(address)

    result = pd.DataFrame(all_features, index=addresses)
    result.index.name = "address"

    for col in result.columns:
        result[col] = result[col].astype(np.float64).fillna(0.0)

    if result.shape[1] != _NUM_BEHAVIORAL_FEATURES:
        raise ValueError(
            f"Expected {_NUM_BEHAVIORAL_FEATURES} behavioral features, got {result.shape[1]}"
        )

    return result