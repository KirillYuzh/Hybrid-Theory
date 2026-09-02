from __future__ import annotations

import numpy as np
import pandas as pd

from kyt_engine.features._utils import (
    counting_entropy,
    discretized_entropy,
    safe_float,
    safe_linregress,
)

WINDOWS = {"30d": 2, "60d": 4, "90d": 6}
WINDOW_DAYS = {"30d": 30, "60d": 60, "90d": 90}


def _window_activity(addr_df: pd.DataFrame, step_from: int, step_to: int, window_days: int = 30) -> dict[str, float]:
    w = addr_df[(addr_df["time_step"] >= step_from) & (addr_df["time_step"] <= step_to)]
    f: dict[str, float] = {}

    has_value = "value" in w.columns
    has_gas = "gas_price" in w.columns

    f["tx_count"] = safe_float(len(w))
    f["poisson_lambda"] = safe_float(len(w) / window_days) if window_days > 0 else 0.0

    ts_sorted = np.sort(w["time_step"].values) if len(w) > 0 else np.array([], dtype=int)
    if len(ts_sorted) >= 2:
        intervals = np.diff(ts_sorted).astype(float)
        f["avg_interval"] = safe_float(np.mean(intervals))
        med_int = float(np.median(intervals))
        f["burstiness"] = safe_float(np.mean(intervals < med_int)) if med_int > 0 else 0.0
    else:
        f["avg_interval"] = 0.0
        f["burstiness"] = 0.0

    if has_value:
        vals = w["value"].values.astype(np.float64)
    else:
        vals = np.array([0.0])
    f["value_mean"] = safe_float(np.mean(vals))
    f["value_std"] = safe_float(np.std(vals))
    f["value_median"] = safe_float(np.median(vals))
    vm = float(np.mean(vals))
    f["value_cv"] = safe_float(float(np.std(vals)) / vm) if vm > 0 else 0.0
    f["value_max_ratio"] = safe_float(float(np.max(vals)) / vm) if vm > 0 else 0.0

    if len(vals) >= 2:
        slope, _ = safe_linregress(np.arange(len(vals), dtype=float), vals)
        f["value_trend"] = safe_float(slope)
    else:
        f["value_trend"] = 0.0

    if has_gas:
        gas = w["gas_price"].values.astype(np.float64)
    else:
        gas = np.array([0.0])
    f["gas_mean"] = safe_float(np.mean(gas))
    f["gas_std"] = safe_float(np.std(gas))
    gm = float(np.mean(gas))
    f["gas_cv"] = safe_float(float(np.std(gas)) / gm) if gm > 0 else 0.0

    if len(gas) >= 2:
        slope, _ = safe_linregress(np.arange(len(gas), dtype=float), gas)
        f["gas_trend"] = safe_float(slope)
    else:
        f["gas_trend"] = 0.0

    # Газовая стратегия: волатильность газа и премия (газ/медиана)
    gas_sorted = w.sort_values("time_step")["gas_price"].values.astype(np.float64) if has_gas and len(w) > 0 else np.array([], dtype=float)
    gas_med = float(np.median(gas)) if len(gas) > 0 else 0.0
    if len(gas_sorted) >= 4:
        f["gas_volatility"] = safe_float(float(np.std(np.diff(gas_sorted))))
    else:
        f["gas_volatility"] = 0.0
    f["gas_premium"] = safe_float(gm / gas_med) if gas_med > 0 else 0.0

    time_steps = w["time_step"].values if len(w) > 0 else np.array([], dtype=int)
    hours = (time_steps % 24).astype(int) if len(time_steps) > 0 else np.array([], dtype=int)
    if len(hours) > 0:
        hour_counts = np.bincount(hours, minlength=24).astype(float)
        total = float(hour_counts.sum())
        f["hour_mean"] = safe_float(float(np.mean(hours)))
        if total > 0:
            probs = hour_counts / total
            nz = probs[probs > 0]
            f["hour_entropy"] = safe_float(float(-np.sum(nz * np.log2(nz))))
        else:
            f["hour_entropy"] = 0.0
        night = set(range(0, 6)) | set(range(22, 24))
        work = set(range(9, 17))
        weekend = set(range(5, 7))
        f["night_ratio"] = safe_float(sum(hour_counts[h] for h in night) / total) if total > 0 else 0.0
        f["work_hours_ratio"] = safe_float(sum(hour_counts[h] for h in work) / total) if total > 0 else 0.0
        f["weekend_ratio"] = safe_float(sum(hour_counts[h] for h in weekend) / total) if total > 0 else 0.0
        f["peak_hour"] = safe_float(float(np.argmax(hour_counts)))
        # Циркадный ритм: размах активных часов (активный период)
        active_hours = np.where(hour_counts > 0)[0]
        f["activity_period"] = safe_float(float(np.max(active_hours) - np.min(active_hours))) if len(active_hours) > 0 else 0.0
        if total > 0:
            peak = float(np.max(hour_counts))
            trough = float(np.min(hour_counts))
            f["circadian_amplitude"] = safe_float((peak - trough) / total)
        else:
            f["circadian_amplitude"] = 0.0
    else:
        f["hour_mean"] = 0.0
        f["hour_entropy"] = 0.0
        f["night_ratio"] = 0.0
        f["work_hours_ratio"] = 0.0
        f["weekend_ratio"] = 0.0
        f["peak_hour"] = 0.0
        f["circadian_amplitude"] = 0.0
        f["activity_period"] = 0.0

    dows = (time_steps % 7).astype(int) if len(time_steps) > 0 else np.array([], dtype=int)
    if len(dows) > 0:
        dow_counts = np.bincount(dows, minlength=7).astype(float)
        total_dow = float(dow_counts.sum())
        if total_dow > 0:
            probs = dow_counts / total_dow
            nz = probs[probs > 0]
            f["dow_entropy"] = safe_float(float(-np.sum(nz * np.log2(nz))))
        else:
            f["dow_entropy"] = 0.0
    else:
        f["dow_entropy"] = 0.0

    froms = w["from_address"].astype(str).values if len(w) > 0 else np.array([], dtype=str)
    tos = w["to_address"].astype(str).values if len(w) > 0 else np.array([], dtype=str)
    addr_val = str(w["from_address"].iloc[0]) if len(w) > 0 else ""

    in_counterparties = np.unique(tos[tos != addr_val]) if len(tos) > 0 else np.array([], dtype=str)
    out_counterparties = np.unique(froms[froms != addr_val]) if len(froms) > 0 else np.array([], dtype=str)
    unique_in = len(in_counterparties)
    unique_out = len(out_counterparties)
    f["unique_in"] = safe_float(unique_in)
    f["unique_out"] = safe_float(unique_out)
    f["unique_total"] = safe_float(unique_in + unique_out)
    f["in_out_ratio"] = safe_float(unique_in / unique_out) if unique_out > 0 else 0.0

    in_partners = set(in_counterparties.tolist())
    out_partners = set(out_counterparties.tolist())
    recip = len(in_partners & out_partners)
    total_cp = len(in_partners | out_partners)
    f["reciprocity"] = safe_float(recip / total_cp) if total_cp > 0 else 0.0

    # Сетевое разнообразие: энтропия Шеннона распределений контрагентов
    if unique_in > 0:
        cp_in, cnt_in = np.unique(tos[tos != addr_val] if len(tos) > 0 else np.array([], dtype=str), return_counts=True)
        if cnt_in.sum() > 0:
            p_in = cnt_in.astype(float) / cnt_in.sum()
            f["entropy_in"] = safe_float(float(-np.sum(p_in * np.log2(p_in))))
        else:
            f["entropy_in"] = 0.0
    else:
        f["entropy_in"] = 0.0

    if unique_out > 0:
        cp_out, cnt_out = np.unique(froms[froms != addr_val] if len(froms) > 0 else np.array([], dtype=str), return_counts=True)
        if cnt_out.sum() > 0:
            p_out = cnt_out.astype(float) / cnt_out.sum()
            f["entropy_out"] = safe_float(float(-np.sum(p_out * np.log2(p_out))))
        else:
            f["entropy_out"] = 0.0
    else:
        f["entropy_out"] = 0.0

    all_cp = np.concatenate([in_counterparties, out_counterparties]) if total_cp > 0 else np.array([], dtype=int)
    if len(all_cp) > 0:
        _, cnt_all = np.unique(all_cp, return_counts=True)
        if cnt_all.sum() > 0:
            p_all = cnt_all.astype(float) / cnt_all.sum()
            f["entropy_total"] = safe_float(float(-np.sum(p_all * np.log2(p_all))))
        else:
            f["entropy_total"] = 0.0
    else:
        f["entropy_total"] = 0.0

    if len(all_cp) > 0:
        cp_vals, cp_counts_arr = np.unique(all_cp, return_counts=True)
        total_tx = int(cp_counts_arr.sum())
        if total_tx > 0:
            freqs = cp_counts_arr.astype(float) / total_tx
            f["degree_concentration"] = safe_float(float(np.sum(freqs ** 2)))
        else:
            f["degree_concentration"] = 0.0
        first_time = int(np.sum(cp_counts_arr == 1))
        f["new_cp_ratio"] = safe_float(first_time / len(cp_vals))
    else:
        f["degree_concentration"] = 0.0
        f["new_cp_ratio"] = 0.0

    v_log = np.log1p(vals)
    f["value_entropy"] = safe_float(discretized_entropy(v_log))

    cp_list = []
    for p in froms[froms != addr_val]:
        cp_list.append(str(p))
    for p in tos[tos == addr_val]:
        cp_list.append(str(p))
    f["counterparty_entropy"] = safe_float(counting_entropy(cp_list))

    f["temporal_entropy"] = safe_float(discretized_entropy(hours.astype(float))) if len(hours) > 0 else 0.0
    f["gas_entropy"] = safe_float(discretized_entropy(gas))

    return f


def compute_behavioral_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ("value", "gas_price", "in_degree", "out_degree"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    if "from_address" not in df.columns:
        df["from_address"] = df["txId"]
    if "to_address" not in df.columns:
        df["to_address"] = df["txId"]

    addr_groups = df.groupby("from_address")
    all_features: list[dict[str, float]] = []
    addresses: list[str] = []

    for addr, addr_df in addr_groups:
        addr_df = addr_df.sort_values("time_step")
        last_ts = int(addr_df["time_step"].max())

        feat: dict[str, float] = {}

        window_data: dict[str, dict[str, float]] = {}
        for name, size in WINDOWS.items():
            step_from = max(1, last_ts - size + 1)
            step_to = last_ts
            wd = _window_activity(addr_df, step_from, step_to, WINDOW_DAYS[name])
            for k, v in wd.items():
                feat[f"{k}_{name}"] = v
            window_data[name] = wd

        a30 = window_data["30d"]
        a60 = window_data["60d"]
        a90 = window_data["90d"]

        feat["activity_ratio_30d_90d"] = safe_float(
            a30["tx_count"] / a90["tx_count"] if a90["tx_count"] > 0 else 0.0
        )

        feat["value_delta_30d_60d"] = safe_float(a30["value_mean"] - a60["value_mean"])
        feat["value_delta_30d_90d"] = safe_float(a30["value_mean"] - a90["value_mean"])
        feat["gas_delta_30d_60d"] = safe_float(a30["gas_mean"] - a60["gas_mean"])
        feat["gas_delta_30d_90d"] = safe_float(a30["gas_mean"] - a90["gas_mean"])
        feat["activity_delta_30d_60d"] = safe_float(a30["tx_count"] - a60["tx_count"])
        feat["activity_delta_30d_90d"] = safe_float(a30["tx_count"] - a90["tx_count"])

        vals = addr_df["value"].values.astype(np.float64) if "value" in addr_df.columns else np.array([0.0])
        gas = addr_df["gas_price"].values.astype(np.float64) if "gas_price" in addr_df.columns else np.array([0.0])

        if len(vals) >= 3:
            v_slope, _ = safe_linregress(np.arange(len(vals), dtype=float), vals)
            feat["value_acceleration"] = safe_float(v_slope)
        else:
            feat["value_acceleration"] = 0.0

        if len(gas) >= 3:
            g_slope, _ = safe_linregress(np.arange(len(gas), dtype=float), gas)
            feat["gas_acceleration"] = safe_float(g_slope)
        else:
            feat["gas_acceleration"] = 0.0

        # Скорость реакции: время между получением и действием (получение → исходящий)
        # Приближение: интервалы между входящими и следующими исходящими транзакциями
        if "to_address" in addr_df.columns and "from_address" in addr_df.columns:
            addr_ts = addr_df.sort_values("time_step")
            if len(addr_ts) >= 2:
                reaction_intervals: list[float] = []
                for i in range(1, len(addr_ts)):
                    prev_to = str(addr_ts.iloc[i - 1]["to_address"])
                    cur_from = str(addr_ts.iloc[i]["from_address"])
                    cur_time = int(addr_ts.iloc[i]["time_step"])
                    prev_time = int(addr_ts.iloc[i - 1]["time_step"])
                    # Если предыдущая транзакция получена, а текущая отправлена — интервал реакции
                    if prev_to != prev_from and cur_from == addr and cur_time > prev_time:
                        reaction_intervals.append(float(cur_time - prev_time))
                if len(reaction_intervals) > 0:
                    feat["reaction_median"] = safe_float(float(np.median(reaction_intervals)))
                    r_mean = float(np.mean(reaction_intervals))
                    r_std = float(np.std(reaction_intervals))
                    feat["reaction_cv"] = safe_float(r_std / r_mean) if r_mean > 0 else 0.0
                else:
                    feat["reaction_median"] = 0.0
                    feat["reaction_cv"] = 0.0
            else:
                feat["reaction_median"] = 0.0
                feat["reaction_cv"] = 0.0
        else:
            feat["reaction_median"] = 0.0
            feat["reaction_cv"] = 0.0

        all_features.append(feat)
        addresses.append(addr)

    result = pd.DataFrame(all_features, index=addresses)
    result.index.name = "address"

    for col in result.columns:
        result[col] = result[col].astype(np.float64).fillna(0.0)

    return result
