from __future__ import annotations

import numpy as np
import pandas as pd

from kyt_engine.features._utils import (
    counting_entropy,
    safe_float,
    safe_kurtosis,
    safe_linregress,
    safe_skew,
)

_NUM_BASE_FEATURES = 165


def _entropy(arr: np.ndarray) -> float:
    if len(arr) == 0:
        return 0.0
    p99 = float(np.percentile(arr, 99)) + 1e-9
    edges = np.linspace(0, p99, 20)
    counts = np.bincount(np.digitize(arr, edges))
    probs = counts[counts > 0] / float(counts.sum())
    return float(-np.sum(probs * np.log2(probs)))


def _acf(series: np.ndarray, lag: int) -> float:
    n = len(series)
    if n <= lag + 1:
        return 0.0
    x = series[: n - lag]
    y = series[lag:]
    sx = float(np.std(x))
    sy = float(np.std(y))
    if sx == 0.0 or sy == 0.0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def _hurst(series: np.ndarray) -> float:
    n = len(series)
    if n < 20:
        return 0.5
    max_k = min(int(np.floor(n / 2)), 49)
    if max_k < 2:
        return 0.5
    lags = np.arange(2, max_k + 1, dtype=np.float64)
    tau = np.array(
        [float(np.std(np.subtract(series[lag:], series[:-lag]))) for lag in lags.astype(int)]
    )
    mask = tau > 0
    tau = tau[mask]
    lags = lags[mask]
    if len(lags) < 2:
        return 0.5
    try:
        coeffs = np.polyfit(np.log(lags), np.log(tau), 1)
        return float(np.clip(coeffs[0], 0.0, 1.0))
    except (np.linalg.LinAlgError, ValueError):
        return 0.5


def _lin_trend(y: np.ndarray) -> tuple[float, float]:
    n = len(y)
    if n < 2:
        return 0.0, 0.0
    x = np.arange(n, dtype=np.float64)
    return safe_linregress(x, y)


def _val_feats(group: pd.DataFrame) -> dict[str, float]:
    v = group["value"].to_numpy(dtype=np.float64)
    lv = np.log1p(v)
    f: dict[str, float] = {}

    f["value_mean"] = safe_float(np.mean(v))
    f["value_std"] = safe_float(np.std(v))
    f["value_min"] = safe_float(np.min(v))
    f["value_max"] = safe_float(np.max(v))
    f["value_median"] = safe_float(np.median(v))
    f["value_q25"] = safe_float(np.percentile(v, 25))
    f["value_q75"] = safe_float(np.percentile(v, 75))
    f["value_skew"] = safe_float(safe_skew(v))
    f["value_kurtosis"] = safe_float(safe_kurtosis(v))
    f["value_range"] = safe_float(float(np.max(v)) - float(np.min(v)))
    f["value_iqr"] = safe_float(float(np.percentile(v, 75)) - float(np.percentile(v, 25)))
    f["value_sum"] = safe_float(np.sum(v))
    vm = float(np.mean(v))
    f["value_cv"] = safe_float(float(np.std(v)) / vm) if vm > 0 else 0.0
    f["value_log_mean"] = safe_float(np.mean(lv))
    f["value_log_std"] = safe_float(np.std(lv))
    f["value_entropy"] = safe_float(_entropy(v))

    f["value_positive_ratio"] = safe_float(float(np.mean(v > 0)))
    f["value_zero_ratio"] = safe_float(float(np.mean(v == 0)))
    vmin = float(np.min(v))
    f["value_max_min_ratio"] = safe_float(float(np.max(v)) / vmin) if vmin > 0 else 0.0
    f["value_skew_mean"] = safe_float(float(safe_skew(v)) * vm) if len(v) > 2 else 0.0
    q75 = float(np.percentile(v, 75))
    q25 = float(np.percentile(v, 25))
    iqr = q75 - q25
    f["value_upper_outlier_ratio"] = safe_float(float(np.mean(v > q75 + 1.5 * iqr)))
    f["value_lower_outlier_ratio"] = safe_float(float(np.mean(v < q25 - 1.5 * iqr)))
    f["value_concentration"] = safe_float(float(np.max(v)) / float(np.sum(v))) if float(np.sum(v)) > 0 else 0.0
    sv = np.sort(v)[::-1]
    t5 = float(np.sum(sv[: min(5, len(sv))]))
    f["value_dominance"] = safe_float(t5 / float(np.sum(v))) if float(np.sum(v)) > 0 else 0.0
    med = float(np.median(v))
    f["value_small_ratio"] = safe_float(float(np.mean(v < 0.1 * med))) if med > 0 else 0.0
    f["value_large_ratio"] = safe_float(float(np.mean(v > 10 * med))) if med > 0 else 0.0

    return f


def _gas_feats(group: pd.DataFrame) -> dict[str, float]:
    v = group["gas_price"].to_numpy(dtype=np.float64)
    lv = np.log1p(v)
    f: dict[str, float] = {}

    f["gas_mean"] = safe_float(np.mean(v))
    f["gas_std"] = safe_float(np.std(v))
    f["gas_min"] = safe_float(np.min(v))
    f["gas_max"] = safe_float(np.max(v))
    f["gas_median"] = safe_float(np.median(v))
    f["gas_q25"] = safe_float(np.percentile(v, 25))
    f["gas_q75"] = safe_float(np.percentile(v, 75))
    f["gas_skew"] = safe_float(safe_skew(v))
    f["gas_kurtosis"] = safe_float(safe_kurtosis(v))
    f["gas_range"] = safe_float(float(np.max(v)) - float(np.min(v)))
    f["gas_iqr"] = safe_float(float(np.percentile(v, 75)) - float(np.percentile(v, 25)))
    f["gas_sum"] = safe_float(np.sum(v))
    vm = float(np.mean(v))
    f["gas_cv"] = safe_float(float(np.std(v)) / vm) if vm > 0 else 0.0
    f["gas_log_mean"] = safe_float(np.mean(lv))
    f["gas_log_std"] = safe_float(np.std(lv))
    f["gas_entropy"] = safe_float(_entropy(v))

    f["gas_positive_ratio"] = safe_float(float(np.mean(v > 0)))
    f["gas_zero_ratio"] = safe_float(float(np.mean(v == 0)))
    vmin = float(np.min(v))
    f["gas_max_min_ratio"] = safe_float(float(np.max(v)) / vmin) if vmin > 0 else 0.0
    f["gas_skew_mean"] = safe_float(float(safe_skew(v)) * vm) if len(v) > 2 else 0.0
    q75 = float(np.percentile(v, 75))
    q25 = float(np.percentile(v, 25))
    iqr = q75 - q25
    f["gas_upper_outlier_ratio"] = safe_float(float(np.mean(v > q75 + 1.5 * iqr)))
    f["gas_lower_outlier_ratio"] = safe_float(float(np.mean(v < q25 - 1.5 * iqr)))
    f["gas_concentration"] = safe_float(float(np.max(v)) / float(np.sum(v))) if float(np.sum(v)) > 0 else 0.0
    sv = np.sort(v)[::-1]
    t5 = float(np.sum(sv[: min(5, len(sv))]))
    f["gas_dominance"] = safe_float(t5 / float(np.sum(v))) if float(np.sum(v)) > 0 else 0.0
    med = float(np.median(v))
    f["gas_small_ratio"] = safe_float(float(np.mean(v < 0.1 * med))) if med > 0 else 0.0
    f["gas_large_ratio"] = safe_float(float(np.mean(v > 10 * med))) if med > 0 else 0.0

    return f


def _int_feats(group: pd.DataFrame) -> dict[str, float]:
    iv = group["interval"].dropna().to_numpy(dtype=np.float64)
    f: dict[str, float] = {}

    names = [
        "interval_mean", "interval_std", "interval_min", "interval_max",
        "interval_median", "interval_q25", "interval_q75", "interval_skew",
        "interval_kurtosis", "interval_range", "interval_cv", "interval_log_mean",
        "interval_log_std", "interval_entropy", "interval_burstiness",
        "interval_regularity",
    ]
    if len(iv) == 0:
        for n in names:
            f[n] = 0.0
        return f

    liv = np.log1p(iv)
    mi = float(np.mean(iv))

    f["interval_mean"] = safe_float(mi)
    f["interval_std"] = safe_float(float(np.std(iv)))
    f["interval_min"] = safe_float(float(np.min(iv)))
    f["interval_max"] = safe_float(float(np.max(iv)))
    f["interval_median"] = safe_float(float(np.median(iv)))
    f["interval_q25"] = safe_float(float(np.percentile(iv, 25)))
    f["interval_q75"] = safe_float(float(np.percentile(iv, 75)))
    f["interval_skew"] = safe_float(safe_skew(iv))
    f["interval_kurtosis"] = safe_float(safe_kurtosis(iv))
    f["interval_range"] = safe_float(float(np.max(iv)) - float(np.min(iv)))
    f["interval_cv"] = safe_float(float(np.std(iv)) / mi) if mi > 0 else 0.0
    f["interval_log_mean"] = safe_float(np.mean(liv))
    f["interval_log_std"] = safe_float(np.std(liv))
    f["interval_entropy"] = safe_float(_entropy(iv))
    med = float(np.median(iv))
    f["interval_burstiness"] = safe_float(float(np.mean(iv < 0.1 * med))) if med > 0 else 0.0
    f["interval_regularity"] = safe_float(1.0 - min(float(np.std(iv)) / mi, 1.0)) if mi > 0 else 0.0

    return f


def _tod_feats(group: pd.DataFrame) -> dict[str, float]:
    hours = group["hour"].to_numpy(dtype=np.intp)
    f: dict[str, float] = {}

    hc = np.bincount(hours, minlength=24).astype(np.float64)
    total = float(hc.sum())

    f["hour_mean"] = safe_float(float(np.mean(hours)))
    f["hour_std"] = safe_float(float(np.std(hours)))

    if total > 0:
        probs = hc / total
        nz = probs[probs > 0]
        f["hour_entropy"] = safe_float(float(-np.sum(nz * np.log2(nz))))
    else:
        f["hour_entropy"] = 0.0

    night = set(range(0, 6)) | set(range(22, 24))
    morning = set(range(6, 12))
    afternoon = set(range(12, 18))
    evening = set(range(18, 22))

    nc = sum(hc[h] for h in night)
    mc = sum(hc[h] for h in morning)
    ac = sum(hc[h] for h in afternoon)
    ec = sum(hc[h] for h in evening)

    f["night_ratio"] = safe_float(nc / total) if total > 0 else 0.0
    f["morning_ratio"] = safe_float(mc / total) if total > 0 else 0.0
    f["afternoon_ratio"] = safe_float(ac / total) if total > 0 else 0.0
    f["evening_ratio"] = safe_float(ec / total) if total > 0 else 0.0
    f["peak_hour"] = safe_float(float(np.argmax(hc)))
    f["hour_concentration"] = safe_float(float(np.max(hc)) / total) if total > 0 else 0.0

    nz = hc[hc > 0]
    if len(nz) >= 2:
        mh = float(np.mean(nz))
        f["hour_bimodality"] = safe_float(1.0 - float(np.std(nz)) / mh) if mh > 0 else 0.0
    else:
        f["hour_bimodality"] = 0.0

    return f


def _dow_feats(group: pd.DataFrame) -> dict[str, float]:
    dows = group["day_of_week"].to_numpy(dtype=np.intp)
    f: dict[str, float] = {}

    dc = np.bincount(dows, minlength=7).astype(np.float64)
    total = float(dc.sum())

    if total > 0:
        probs = dc / total
        probs = probs[probs > 0]
        f["dow_entropy"] = safe_float(float(-np.sum(probs * np.log2(probs))))
    else:
        f["dow_entropy"] = 0.0

    weekend = float(dc[5] + dc[6])
    midweek = float(dc[1] + dc[2] + dc[3])
    endweek = float(dc[4] + dc[5] + dc[6])

    f["weekend_ratio"] = safe_float(weekend / total) if total > 0 else 0.0
    f["midweek_ratio"] = safe_float(midweek / total) if total > 0 else 0.0
    f["endweek_ratio"] = safe_float(endweek / total) if total > 0 else 0.0
    f["dow_concentration"] = safe_float(float(np.max(dc)) / total) if total > 0 else 0.0

    active = int(np.sum(dc > 0))
    f["dow_regularity"] = safe_float(active / 7.0)

    dates = sorted(group["date"].dropna().unique().tolist())
    if len(dates) > 1:
        dmax = pd.Timestamp(dates[-1])
        dmin = pd.Timestamp(dates[0])
        f["day_span"] = safe_float((dmax - dmin).total_seconds() / 86400.0)
    else:
        f["day_span"] = 0.0

    return f


def _net_feats(group: pd.DataFrame) -> dict[str, float]:
    f: dict[str, float] = {}
    addr = str(group["address"].iloc[0])

    incoming = group[group["to_address"] == addr]
    outgoing = group[group["from_address"] == addr]

    in_deg = int(incoming["from_address"].nunique())
    out_deg = int(outgoing["to_address"].nunique())

    all_partners = set(group["from_address"].tolist()) | set(group["to_address"].tolist())
    all_partners.discard(addr)
    total_deg = len(all_partners)

    f["in_degree"] = safe_float(in_deg)
    f["out_degree"] = safe_float(out_deg)
    f["total_degree"] = safe_float(total_deg)
    f["in_out_ratio"] = safe_float(in_deg / out_deg) if out_deg > 0 else 0.0

    unique_in = int(incoming["from_address"].nunique())
    unique_out = int(outgoing["to_address"].nunique())
    f["unique_in"] = safe_float(unique_in)
    f["unique_out"] = safe_float(unique_out)
    f["unique_total"] = safe_float(total_deg)

    from_mask = group["from_address"] != addr
    to_mask = group["to_address"] != addr
    from_counts = group.loc[from_mask, "from_address"].value_counts()
    to_counts = group.loc[to_mask, "to_address"].value_counts()
    partner_counts = from_counts.add(to_counts, fill_value=0)

    top_count = int(partner_counts.max()) if len(partner_counts) > 0 else 0
    f["degree_concentration"] = safe_float(top_count / len(group))

    total_val = float(group["value"].sum())
    in_val = float(incoming["value"].sum())
    out_val = float(outgoing["value"].sum())

    f["in_value_ratio"] = safe_float(in_val / total_val) if total_val > 0 else 0.0
    f["out_value_ratio"] = safe_float(out_val / total_val) if total_val > 0 else 0.0

    in_partners = set(incoming["from_address"].unique())
    out_partners = set(outgoing["to_address"].unique())
    mutual = in_partners & out_partners
    f["mutual_ratio"] = safe_float(len(mutual) / total_deg) if total_deg > 0 else 0.0

    bidirectional = 0
    for p in mutual:
        has_in = bool(((group["from_address"] == p) & (group["to_address"] == addr)).any())
        has_out = bool(((group["from_address"] == addr) & (group["to_address"] == p)).any())
        if has_in and has_out:
            bidirectional += 1
    f["reciprocity"] = safe_float(bidirectional / len(mutual)) if mutual else 0.0

    f["hub_score"] = safe_float(out_deg / total_deg) if total_deg > 0 else 0.0
    f["authority_score"] = safe_float(in_deg / total_deg) if total_deg > 0 else 0.0
    f["pagerank_approx"] = safe_float(in_val / total_val) if total_val > 0 else 0.0
    f["star_ratio"] = safe_float(top_count / len(group)) if len(group) > 0 else 0.0

    return f


def _cp_feats(group: pd.DataFrame) -> dict[str, float]:
    f: dict[str, float] = {}
    addr = str(group["address"].iloc[0])

    from_addr = group["from_address"].astype(str)
    to_addr = group["to_address"].astype(str)
    is_out = from_addr == addr
    is_in = to_addr == addr

    cps = np.where(is_out, to_addr, np.where(is_in, from_addr, "")).tolist()
    cps = [c for c in cps if c]

    empty = {
        "unique_counterparties": 0.0, "counterparty_concentration": 0.0,
        "top_counterparty_ratio": 0.0, "top5_counterparty_ratio": 0.0,
        "new_counterparty_ratio": 0.0, "return_counterparty_ratio": 0.0,
        "counterparty_reciprocity": 0.0, "counterparty_entropy": 0.0,
        "counterparty_churn": 0.0, "stable_counterparty_ratio": 0.0,
        "bridge_counterparty_ratio": 0.0, "counterparty_hhi": 0.0,
    }
    if not cps:
        return empty

    from collections import Counter
    cp_c = Counter(cps)
    uc = len(cp_c)
    tx = len(cps)

    f["unique_counterparties"] = safe_float(uc)
    top_c = cp_c.most_common(1)[0][1]
    f["counterparty_concentration"] = safe_float(top_c / tx) if tx > 0 else 0.0
    f["top_counterparty_ratio"] = safe_float(top_c / tx) if tx > 0 else 0.0
    t5 = sum(c for _, c in cp_c.most_common(5))
    f["top5_counterparty_ratio"] = safe_float(t5 / tx) if tx > 0 else 0.0

    sorted_group = group.sort_values("timestamp")
    ordered = np.where(
        sorted_group["from_address"].astype(str) == addr,
        sorted_group["to_address"].astype(str),
        np.where(
            sorted_group["to_address"].astype(str) == addr,
            sorted_group["from_address"].astype(str),
            "",
        ),
    ).tolist()
    ordered = [c for c in ordered if c]

    seen: set[str] = set()
    new_c = 0
    ret_c = 0
    for cp in ordered:
        if cp not in seen:
            new_c += 1
            seen.add(cp)
        else:
            ret_c += 1

    f["new_counterparty_ratio"] = safe_float(new_c / tx) if tx > 0 else 0.0
    f["return_counterparty_ratio"] = safe_float(ret_c / tx) if tx > 0 else 0.0

    out_addrs = set(group.loc[group["from_address"] == addr, "to_address"].astype(str))
    in_addrs = set(group.loc[group["to_address"] == addr, "from_address"].astype(str))
    recip = len(out_addrs & in_addrs)
    f["counterparty_reciprocity"] = safe_float(recip / uc) if uc > 0 else 0.0
    f["counterparty_entropy"] = safe_float(counting_entropy(cps))

    if len(ordered) > 1:
        changes = sum(1 for i in range(1, len(ordered)) if ordered[i] != ordered[i - 1])
        f["counterparty_churn"] = safe_float(changes / (len(ordered) - 1))
    else:
        f["counterparty_churn"] = 0.0

    f["stable_counterparty_ratio"] = safe_float(
        sum(1 for c in cp_c.values() if c > 1) / uc
    ) if uc > 0 else 0.0

    in_partners = set(group[group["to_address"] == addr]["from_address"].unique())
    out_partners = set(group[group["from_address"] == addr]["to_address"].unique())
    bridge = (in_partners - {addr}) & (out_partners - {addr})
    f["bridge_counterparty_ratio"] = safe_float(len(bridge) / uc) if uc > 0 else 0.0

    freqs = np.array(list(cp_c.values()), dtype=np.float64)
    freqs = freqs / freqs.sum()
    f["counterparty_hhi"] = safe_float(float(np.sum(freqs ** 2)))

    return f


def _vtrend_feats(group: pd.DataFrame) -> dict[str, float]:
    f: dict[str, float] = {}
    sg = group.sort_values("timestamp")
    vals = sg["value"].to_numpy(dtype=np.float64)
    n = len(vals)

    slope, r2 = _lin_trend(vals)
    f["value_trend_slope"] = safe_float(slope)
    f["value_trend_r2"] = safe_float(r2)

    f["value_momentum_3"] = safe_float(float(np.mean(vals[-3:]) - np.mean(vals[:-3]))) if n > 3 else 0.0
    f["value_momentum_5"] = safe_float(float(np.mean(vals[-5:]) - np.mean(vals[:-5]))) if n > 5 else 0.0
    f["value_momentum_10"] = safe_float(float(np.mean(vals[-10:]) - np.mean(vals[:-10]))) if n > 10 else 0.0

    if n >= 3:
        d = np.diff(vals)
        f["value_acceleration"] = safe_float(float(np.mean(np.diff(d))))
    else:
        f["value_acceleration"] = 0.0

    if n >= 4:
        d2 = np.diff(np.diff(vals))
        f["value_jerk"] = safe_float(float(np.mean(np.diff(d2))))
    else:
        f["value_jerk"] = 0.0

    for w, nm in [(3, "value_volatility_3"), (5, "value_volatility_5"), (10, "value_volatility_10")]:
        if n >= w:
            rs = np.array([float(np.std(vals[i : i + w])) for i in range(n - w + 1)])
            f[nm] = safe_float(float(np.mean(rs)))
        else:
            f[nm] = 0.0

    if n >= 3:
        rm = np.array([float(np.mean(vals[i : i + 3])) for i in range(n - 2)])
        rm_mean = float(np.mean(rm))
        f["trend_consistency"] = safe_float(float(np.std(rm)) / rm_mean) if rm_mean > 0 else 0.0
    else:
        f["trend_consistency"] = 0.0

    if n >= 2:
        signs = np.sign(np.diff(vals))
        streak = 1
        mx = 1
        for i in range(1, len(signs)):
            if signs[i] == signs[i - 1] and signs[i] != 0:
                streak += 1
                mx = max(mx, streak)
            else:
                streak = 1
        f["value_max_streak"] = safe_float(mx / n)
    else:
        f["value_max_streak"] = 0.0

    return f


def _gtrend_feats(group: pd.DataFrame) -> dict[str, float]:
    f: dict[str, float] = {}
    sg = group.sort_values("timestamp")
    g = sg["gas_price"].to_numpy(dtype=np.float64)
    n = len(g)

    slope, r2 = _lin_trend(g)
    f["gas_trend_slope"] = safe_float(slope)
    f["gas_trend_r2"] = safe_float(r2)

    f["gas_momentum_3"] = safe_float(float(np.mean(g[-3:]) - np.mean(g[:-3]))) if n > 3 else 0.0
    f["gas_momentum_5"] = safe_float(float(np.mean(g[-5:]) - np.mean(g[:-5]))) if n > 5 else 0.0

    if n >= 2:
        gc = np.diff(g)
        f["gas_change_mean"] = safe_float(float(np.mean(gc)))
        f["gas_change_std"] = safe_float(float(np.std(gc)))
    else:
        f["gas_change_mean"] = 0.0
        f["gas_change_std"] = 0.0

    if n >= 5:
        gu = sg["gas_used"].to_numpy(dtype=np.float64)
        eff = gu / (g + 1e-12)
        es, _ = _lin_trend(eff)
        f["gas_efficiency_trend"] = safe_float(es)
    else:
        f["gas_efficiency_trend"] = 0.0

    if n >= 2:
        spikes = int(np.sum(g[1:] > 3 * g[:-1]))
        f["gas_spike_freq"] = safe_float(spikes / (n - 1))
    else:
        f["gas_spike_freq"] = 0.0

    return f


def _acf_feats(group: pd.DataFrame) -> dict[str, float]:
    f: dict[str, float] = {}
    sg = group.sort_values("timestamp")
    vals = sg["value"].to_numpy(dtype=np.float64)
    gas = sg["gas_price"].to_numpy(dtype=np.float64)
    iv = group["interval"].dropna().to_numpy(dtype=np.float64)

    for lag in [1, 2, 3, 5, 10]:
        f[f"value_acf_{lag}"] = safe_float(_acf(vals, lag))
    for lag in [1, 2, 3]:
        f[f"gas_acf_{lag}"] = safe_float(_acf(gas, lag))
    for lag in [1, 2, 3]:
        f[f"interval_acf_{lag}"] = safe_float(_acf(iv, lag))

    acf_vals = [_acf(vals, lag) for lag in range(1, min(len(vals), 11))]
    if len(acf_vals) >= 3:
        aa = np.array(acf_vals[:5], dtype=np.float64)
        abs_a = np.abs(aa)
        hi = int(np.argmax(abs_a < 0.5 * abs_a[0])) if abs_a[0] > 0 else len(abs_a)
        f["acf_decay_rate"] = safe_float(1.0 - abs_a[min(hi, len(abs_a) - 1)] / abs_a[0]) if abs_a[0] > 0 else 0.0
        s = np.sign(aa)
        sc = int(np.sum(s[1:] != s[:-1]))
        f["acf_sign_changes"] = safe_float(sc / len(s))
    else:
        f["acf_decay_rate"] = 0.0
        f["acf_sign_changes"] = 0.0

    if len(vals) >= 4:
        a1 = _acf(vals, 1)
        a2 = _acf(vals, 2)
        f["periodicity_score"] = safe_float(max(0.0, a2 - a1 * a1))
    else:
        f["periodicity_score"] = 0.0

    if len(vals) >= 10:
        mv = float(np.mean(vals))
        cd = np.cumsum(vals - mv)
        r = float(np.max(cd) - np.min(cd))
        sv = float(np.std(vals))
        f["stationarity_score"] = safe_float(r / sv) if sv > 0 else 0.0
    else:
        f["stationarity_score"] = 0.0

    f["hurst_exponent"] = safe_float(_hurst(vals))
    return f


def _lag_feats(group: pd.DataFrame) -> dict[str, float]:
    f: dict[str, float] = {}
    sg = group.sort_values("timestamp")
    vals = sg["value"].to_numpy(dtype=np.float64)
    gas = sg["gas_price"].to_numpy(dtype=np.float64)
    blks = sg["block_number"].to_numpy(dtype=np.float64)
    ts = sg["timestamp"].to_numpy(dtype=np.float64)
    iv = group["interval"].dropna().to_numpy(dtype=np.float64)

    for lag in [1, 2, 3]:
        n = len(vals)
        if n > lag and float(np.std(vals[: n - lag])) > 0 and float(np.std(vals[lag:])) > 0:
            f[f"value_lag_{lag}"] = safe_float(float(np.corrcoef(vals[: n - lag], vals[lag:])[0, 1]))
        else:
            f[f"value_lag_{lag}"] = 0.0

    for lag in [1, 2, 3]:
        n = len(gas)
        if n > lag and float(np.std(gas[: n - lag])) > 0 and float(np.std(gas[lag:])) > 0:
            f[f"gas_lag_{lag}"] = safe_float(float(np.corrcoef(gas[: n - lag], gas[lag:])[0, 1]))
        else:
            f[f"gas_lag_{lag}"] = 0.0

    n = len(vals)
    if n >= 3 and float(np.std(vals)) > 0 and float(np.std(gas)) > 0:
        f["value_gas_corr"] = safe_float(float(np.corrcoef(vals, gas)[0, 1]))
    else:
        f["value_gas_corr"] = 0.0

    if n >= 3:
        if float(np.std(vals)) > 0 and float(np.std(blks)) > 0:
            f["value_block_corr"] = safe_float(float(np.corrcoef(vals, blks)[0, 1]))
        else:
            f["value_block_corr"] = 0.0
        if float(np.std(gas)) > 0 and float(np.std(blks)) > 0:
            f["gas_block_corr"] = safe_float(float(np.corrcoef(gas, blks)[0, 1]))
        else:
            f["gas_block_corr"] = 0.0
    else:
        f["value_block_corr"] = 0.0
        f["gas_block_corr"] = 0.0

    if n >= 3:
        if float(np.std(vals)) > 0 and float(np.std(ts)) > 0:
            f["value_ts_corr"] = safe_float(float(np.corrcoef(vals, ts)[0, 1]))
        else:
            f["value_ts_corr"] = 0.0
        if float(np.std(gas)) > 0 and float(np.std(ts)) > 0:
            f["gas_ts_corr"] = safe_float(float(np.corrcoef(gas, ts)[0, 1]))
        else:
            f["gas_ts_corr"] = 0.0
    else:
        f["value_ts_corr"] = 0.0
        f["gas_ts_corr"] = 0.0

    if len(iv) >= 3 and float(np.std(iv)) > 0:
        vf = vals[1:]
        if len(vf) == len(iv) and float(np.std(vf)) > 0:
            f["interval_value_corr"] = safe_float(float(np.corrcoef(iv, vf)[0, 1]))
        else:
            f["interval_value_corr"] = 0.0
    else:
        f["interval_value_corr"] = 0.0

    return f


def _blk_feats(group: pd.DataFrame) -> dict[str, float]:
    f: dict[str, float] = {}
    blocks = group["block_number"].to_numpy()
    uniq = np.unique(blocks)

    f["block_span"] = safe_float(float(np.max(blocks) - np.min(blocks))) if len(blocks) > 0 else 0.0
    f["blocks_per_tx"] = safe_float(len(uniq) / len(blocks)) if len(blocks) > 0 else 0.0
    f["unique_blocks_ratio"] = safe_float(len(uniq) / len(blocks)) if len(blocks) > 0 else 0.0

    if len(uniq) > 0:
        btc = group.groupby("block_number").size()
        f["block_reuse_ratio"] = safe_float(1.0 - btc.nunique() / len(btc))
    else:
        f["block_reuse_ratio"] = 0.0

    return f


def extract_base_features(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    df = df.copy()
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce").fillna(0.0)
    df["gas_price"] = pd.to_numeric(df["gas_price"], errors="coerce").fillna(0.0)
    df["gas_used"] = pd.to_numeric(df["gas_used"], errors="coerce").fillna(0.0)
    df["block_number"] = pd.to_numeric(df["block_number"], errors="coerce").fillna(0.0).astype(np.int64)

    ts = pd.to_datetime(df["timestamp"], unit="s", errors="coerce")
    fb = pd.to_datetime(df["timestamp"], errors="coerce")
    ts = ts.fillna(fb)

    df["hour"] = ts.dt.hour.fillna(0).astype(np.intp)
    df["day_of_week"] = ts.dt.dayofweek.fillna(0).astype(np.intp)
    df["date"] = ts.dt.strftime("%Y-%m-%d")

    df = df.sort_values(["address", "timestamp"])
    df["interval"] = df.groupby("address")["timestamp"].diff().fillna(0.0).clip(lower=0.0)

    all_f: list[dict[str, float]] = []
    addrs: list[str] = []

    for addr, grp in df.groupby("address"):
        feat: dict[str, float] = {}
        feat.update(_val_feats(grp))
        feat.update(_gas_feats(grp))
        feat.update(_int_feats(grp))
        feat.update(_tod_feats(grp))
        feat.update(_dow_feats(grp))
        feat.update(_net_feats(grp))
        feat.update(_cp_feats(grp))
        feat.update(_vtrend_feats(grp))
        feat.update(_gtrend_feats(grp))
        feat.update(_acf_feats(grp))
        feat.update(_lag_feats(grp))
        feat.update(_blk_feats(grp))
        all_f.append(feat)
        addrs.append(str(addr))

    result = pd.DataFrame(all_f, index=addrs)
    result.index.name = "address"

    for col in result.columns:
        result[col] = result[col].astype(np.float64).fillna(0.0)

    if result.shape[1] != _NUM_BASE_FEATURES:
        raise ValueError(
            f"Expected {_NUM_BASE_FEATURES} base features, got {result.shape[1]}"
        )

    return result
