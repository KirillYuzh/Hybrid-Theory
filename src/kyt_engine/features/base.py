import numpy as np
import pandas as pd
from collections import Counter
from scipy import stats as sp_stats


def _entropy(arr: np.ndarray) -> float:
    if len(arr) == 0:
        return 0.0
    p99 = float(np.percentile(arr, 99)) + 1e-9
    edges = np.linspace(0, p99, 20)
    counts = np.bincount(np.digitize(arr, edges))
    probs = counts[counts > 0] / float(counts.sum())
    return float(-np.sum(probs * np.log2(probs + 1e-12)))


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
    res = sp_stats.linregress(x, y)
    return float(res.slope), float(res.rvalue) ** 2


def _percentile_stats(v: np.ndarray) -> dict[str, float]:
    return {
        "min": float(np.min(v)),
        "max": float(np.max(v)),
        "median": float(np.median(v)),
        "q25": float(np.percentile(v, 25)),
        "q75": float(np.percentile(v, 75)),
    }


def _basic_stats(v: np.ndarray, prefix: str) -> dict[str, float]:
    if len(v) == 0:
        return {f"{prefix}_mean": 0.0, f"{prefix}_std": 0.0}

    lv = np.log1p(v)
    vm = float(np.mean(v))
    vs = float(np.std(v))
    ps = _percentile_stats(v)

    iqr = ps["q75"] - ps["q25"]
    vsum = float(np.sum(v))
    vmin = ps["min"]
    med = ps["median"]

    f = {
        f"{prefix}_mean": vm,
        f"{prefix}_std": vs,
        f"{prefix}_min": ps["min"],
        f"{prefix}_max": ps["max"],
        f"{prefix}_median": ps["median"],
        f"{prefix}_q25": ps["q25"],
        f"{prefix}_q75": ps["q75"],
        f"{prefix}_skew": float(sp_stats.skew(v, bias=False)),
        f"{prefix}_kurtosis": float(sp_stats.kurtosis(v, bias=False)),
        f"{prefix}_range": ps["max"] - ps["min"],
        f"{prefix}_iqr": iqr,
        f"{prefix}_sum": vsum,
        f"{prefix}_cv": vs / vm if vm > 0 else 0.0,
        f"{prefix}_log_mean": float(np.mean(lv)),
        f"{prefix}_log_std": float(np.std(lv)),
        f"{prefix}_entropy": _entropy(v),
        f"{prefix}_max_min_ratio": ps["max"] / vmin if vmin > 0 else 0.0,
        f"{prefix}_skew_mean": float(sp_stats.skew(v, bias=False)) * vm if len(v) > 2 else 0.0,
        f"{prefix}_upper_outlier_ratio": float(np.mean(v > ps["q75"] + 1.5 * iqr)),
        f"{prefix}_lower_outlier_ratio": float(np.mean(v < ps["q25"] - 1.5 * iqr)),
        f"{prefix}_concentration": ps["max"] / vsum if vsum > 0 else 0.0,
    }

    sv = np.sort(v)[::-1]
    t5 = float(np.sum(sv[: min(5, len(sv))]))
    f[f"{prefix}_dominance"] = t5 / vsum if vsum > 0 else 0.0
    f[f"{prefix}_small_ratio"] = float(np.mean(v < 0.1 * med)) if med > 0 else 0.0
    f[f"{prefix}_large_ratio"] = float(np.mean(v > 10 * med)) if med > 0 else 0.0
    f[f"{prefix}_positive_ratio"] = float(np.mean(v > 0))
    f[f"{prefix}_zero_ratio"] = float(np.mean(v == 0))
    return f


def _val_feats(group: pd.DataFrame) -> dict[str, float]:
    return _basic_stats(group["value"].to_numpy(dtype=np.float64), "value")


def _gas_feats(group: pd.DataFrame) -> dict[str, float]:
    return _basic_stats(group["gas_price"].to_numpy(dtype=np.float64), "gas")


def _int_feats(group: pd.DataFrame) -> dict[str, float]:
    iv = group["interval"].dropna().to_numpy(dtype=np.float64)
    names = [
        "interval_mean", "interval_std", "interval_min", "interval_max",
        "interval_median", "interval_q25", "interval_q75", "interval_skew",
        "interval_kurtosis", "interval_range", "interval_cv", "interval_log_mean",
        "interval_log_std", "interval_entropy", "interval_burstiness",
        "interval_regularity",
    ]
    if len(iv) == 0:
        return {n: 0.0 for n in names}

    liv = np.log1p(iv)
    mi = float(np.mean(iv))
    ps = _percentile_stats(iv)

    f = {
        "interval_mean": mi,
        "interval_std": float(np.std(iv)),
        "interval_min": ps["min"],
        "interval_max": ps["max"],
        "interval_median": ps["median"],
        "interval_q25": ps["q25"],
        "interval_q75": ps["q75"],
        "interval_skew": float(sp_stats.skew(iv, bias=False)),
        "interval_kurtosis": float(sp_stats.kurtosis(iv, bias=False)),
        "interval_range": ps["max"] - ps["min"],
        "interval_cv": float(np.std(iv)) / mi if mi > 0 else 0.0,
        "interval_log_mean": float(np.mean(liv)),
        "interval_log_std": float(np.std(liv)),
        "interval_entropy": _entropy(iv),
    }
    f["interval_burstiness"] = float(np.mean(iv < 0.1 * med)) if (med := ps["median"]) > 0 else 0.0
    f["interval_regularity"] = 1.0 - min(float(np.std(iv)) / mi, 1.0) if mi > 0 else 0.0
    return f


def _tod_feats(group: pd.DataFrame) -> dict[str, float]:
    hours = group["hour"].to_numpy(dtype=np.intp)
    hc = np.bincount(hours, minlength=24).astype(np.float64)
    total = float(hc.sum())

    f = {
        "hour_mean": float(np.mean(hours)),
        "hour_std": float(np.std(hours)),
    }
    if total > 0:
        probs = hc / total
        nz = probs[probs > 0]
        f["hour_entropy"] = float(-np.sum(nz * np.log2(nz)))
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

    f["night_ratio"] = nc / total if total > 0 else 0.0
    f["morning_ratio"] = mc / total if total > 0 else 0.0
    f["afternoon_ratio"] = ac / total if total > 0 else 0.0
    f["evening_ratio"] = ec / total if total > 0 else 0.0
    f["peak_hour"] = float(np.argmax(hc))
    f["hour_concentration"] = float(np.max(hc)) / total if total > 0 else 0.0

    nz = hc[hc > 0]
    if len(nz) >= 2:
        mh = float(np.mean(nz))
        f["hour_bimodality"] = 1.0 - float(np.std(nz)) / mh if mh > 0 else 0.0
    else:
        f["hour_bimodality"] = 0.0
    return f


def _dow_feats(group: pd.DataFrame) -> dict[str, float]:
    dows = group["day_of_week"].to_numpy(dtype=np.intp)
    dc = np.bincount(dows, minlength=7).astype(np.float64)
    total = float(dc.sum())

    f = {}
    if total > 0:
        probs = dc / total
        probs = probs[probs > 0]
        f["dow_entropy"] = float(-np.sum(probs * np.log2(probs)))
    else:
        f["dow_entropy"] = 0.0

    weekend = float(dc[5] + dc[6])
    midweek = float(dc[1] + dc[2] + dc[3])
    endweek = float(dc[4] + dc[5] + dc[6])

    f["weekend_ratio"] = weekend / total if total > 0 else 0.0
    f["midweek_ratio"] = midweek / total if total > 0 else 0.0
    f["endweek_ratio"] = endweek / total if total > 0 else 0.0
    f["dow_concentration"] = float(np.max(dc)) / total if total > 0 else 0.0
    active = int(np.sum(dc > 0))
    f["dow_regularity"] = active / 7.0

    dates = sorted(group["date"].dropna().unique().tolist())
    if len(dates) > 1:
        dmax = pd.Timestamp(dates[-1])
        dmin = pd.Timestamp(dates[0])
        f["day_span"] = (dmax - dmin).total_seconds() / 86400.0
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

    f["in_degree"] = float(in_deg)
    f["out_degree"] = float(out_deg)
    f["total_degree"] = float(total_deg)
    f["in_out_ratio"] = in_deg / out_deg if out_deg > 0 else 0.0
    f["unique_in"] = float(in_deg)
    f["unique_out"] = float(out_deg)
    f["unique_total"] = float(total_deg)

    from_mask = group["from_address"] != addr
    to_mask = group["to_address"] != addr
    from_counts = group.loc[from_mask, "from_address"].value_counts()
    to_counts = group.loc[to_mask, "to_address"].value_counts()
    partner_counts = from_counts.add(to_counts, fill_value=0)
    top_count = int(partner_counts.max()) if len(partner_counts) > 0 else 0
    f["degree_concentration"] = top_count / len(group)

    total_val = float(group["value"].sum())
    in_val = float(incoming["value"].sum())
    out_val = float(outgoing["value"].sum())
    f["in_value_ratio"] = in_val / total_val if total_val > 0 else 0.0
    f["out_value_ratio"] = out_val / total_val if total_val > 0 else 0.0

    in_partners = set(incoming["from_address"].unique())
    out_partners = set(outgoing["to_address"].unique())
    mutual = in_partners & out_partners
    f["mutual_ratio"] = len(mutual) / total_deg if total_deg > 0 else 0.0

    bidirectional = 0
    for p in mutual:
        has_in = bool(((group["from_address"] == p) & (group["to_address"] == addr)).any())
        has_out = bool(((group["from_address"] == addr) & (group["to_address"] == p)).any())
        if has_in and has_out:
            bidirectional += 1
    f["reciprocity"] = bidirectional / len(mutual) if mutual else 0.0
    f["hub_score"] = out_deg / total_deg if total_deg > 0 else 0.0
    f["authority_score"] = in_deg / total_deg if total_deg > 0 else 0.0
    f["pagerank_approx"] = in_val / total_val if total_val > 0 else 0.0
    f["star_ratio"] = top_count / len(group) if len(group) > 0 else 0.0
    return f


def _extract_counterparties(group: pd.DataFrame, addr: str) -> list[str]:
    from_addr = group["from_address"].astype(str)
    to_addr = group["to_address"].astype(str)
    is_out = from_addr == addr
    is_in = to_addr == addr
    cps = np.where(is_out, to_addr, np.where(is_in, from_addr, "")).tolist()
    return [c for c in cps if c]


def _cp_feats(group: pd.DataFrame) -> dict[str, float]:
    f: dict[str, float] = {}
    addr = str(group["address"].iloc[0])
    cps = _extract_counterparties(group, addr)

    if not cps:
        empty_keys = [
            "unique_counterparties", "counterparty_concentration",
            "top_counterparty_ratio", "top5_counterparty_ratio",
            "new_counterparty_ratio", "return_counterparty_ratio",
            "counterparty_reciprocity", "counterparty_entropy",
            "counterparty_churn", "stable_counterparty_ratio",
            "bridge_counterparty_ratio", "counterparty_hhi",
        ]
        return {k: 0.0 for k in empty_keys}

    cp_c = Counter(cps)
    uc = len(cp_c)
    tx = len(cps)

    f["unique_counterparties"] = float(uc)
    top_c = cp_c.most_common(1)[0][1]
    f["counterparty_concentration"] = top_c / tx if tx > 0 else 0.0
    f["top_counterparty_ratio"] = top_c / tx if tx > 0 else 0.0
    t5 = sum(c for _, c in cp_c.most_common(5))
    f["top5_counterparty_ratio"] = t5 / tx if tx > 0 else 0.0

    sorted_group = group.sort_values("timestamp")
    ordered = _extract_counterparties(sorted_group, addr)

    seen: set[str] = set()
    new_c = 0
    ret_c = 0
    for cp in ordered:
        if cp not in seen:
            new_c += 1
            seen.add(cp)
        else:
            ret_c += 1
    f["new_counterparty_ratio"] = new_c / tx if tx > 0 else 0.0
    f["return_counterparty_ratio"] = ret_c / tx if tx > 0 else 0.0

    out_addrs = set(group.loc[group["from_address"] == addr, "to_address"].astype(str))
    in_addrs = set(group.loc[group["to_address"] == addr, "from_address"].astype(str))
    recip = len(out_addrs & in_addrs)
    f["counterparty_reciprocity"] = recip / uc if uc > 0 else 0.0
    f["counterparty_entropy"] = _counting_entropy(cps)

    if len(ordered) > 1:
        changes = sum(1 for i in range(1, len(ordered)) if ordered[i] != ordered[i - 1])
        f["counterparty_churn"] = changes / (len(ordered) - 1)
    else:
        f["counterparty_churn"] = 0.0

    f["stable_counterparty_ratio"] = sum(1 for c in cp_c.values() if c > 1) / uc if uc > 0 else 0.0

    in_partners = in_addrs
    out_partners = out_addrs
    bridge = (in_partners - {addr}) & (out_partners - {addr})
    f["bridge_counterparty_ratio"] = len(bridge) / uc if uc > 0 else 0.0

    freqs = np.array(list(cp_c.values()), dtype=np.float64)
    freqs = freqs / freqs.sum()
    f["counterparty_hhi"] = float(np.sum(freqs ** 2))
    return f


def _trend_feats(series: np.ndarray, prefix: str, include_momentum: bool = True,
                 include_volatility: bool = True, include_jerk: bool = False) -> dict[str, float]:
    f: dict[str, float] = {}
    n = len(series)
    if n == 0:
        return f

    slope, r2 = _lin_trend(series)
    f[f"{prefix}_trend_slope"] = slope
    f[f"{prefix}_trend_r2"] = r2

    if include_momentum:
        for w in [3, 5, 10]:
            if n > w:
                f[f"{prefix}_momentum_{w}"] = float(np.mean(series[-w:]) - np.mean(series[:-w]))
            else:
                f[f"{prefix}_momentum_{w}"] = 0.0

    if include_jerk and n >= 3:
        d = np.diff(series)
        f[f"{prefix}_acceleration"] = float(np.mean(np.diff(d)))
    elif include_jerk:
        f[f"{prefix}_acceleration"] = 0.0

    if include_jerk and n >= 4:
        d2 = np.diff(np.diff(series))
        f[f"{prefix}_jerk"] = float(np.mean(np.diff(d2)))
    elif include_jerk:
        f[f"{prefix}_jerk"] = 0.0

    if include_volatility:
        for w, nm in [(3, f"{prefix}_volatility_3"), (5, f"{prefix}_volatility_5"), (10, f"{prefix}_volatility_10")]:
            if n >= w:
                rs = np.array([float(np.std(series[i : i + w])) for i in range(n - w + 1)])
                f[nm] = float(np.mean(rs))
            else:
                f[nm] = 0.0

    if n >= 3:
        rm = np.array([float(np.mean(series[i : i + 3])) for i in range(n - 2)])
        rm_mean = float(np.mean(rm))
        f[f"{prefix}_trend_consistency"] = float(np.std(rm)) / rm_mean if rm_mean > 0 else 0.0
    else:
        f[f"{prefix}_trend_consistency"] = 0.0

    if n >= 2:
        signs = np.sign(np.diff(series))
        streak = 1
        mx = 1
        for i in range(1, len(signs)):
            if signs[i] == signs[i - 1] and signs[i] != 0:
                streak += 1
                mx = max(mx, streak)
            else:
                streak = 1
        f[f"{prefix}_max_streak"] = mx / n
    else:
        f[f"{prefix}_max_streak"] = 0.0
    return f


def _vtrend_feats(group: pd.DataFrame) -> dict[str, float]:
    sg = group.sort_values("timestamp")
    vals = sg["value"].to_numpy(dtype=np.float64)
    return _trend_feats(vals, "value", include_jerk=True)


def _gtrend_feats(group: pd.DataFrame) -> dict[str, float]:
    f: dict[str, float] = {}
    sg = group.sort_values("timestamp")
    g = sg["gas_price"].to_numpy(dtype=np.float64)
    n = len(g)

    trend = _trend_feats(g, "gas", include_momentum=True, include_volatility=False, include_jerk=False)
    f.update(trend)

    if n >= 2:
        gc = np.diff(g)
        f["gas_change_mean"] = float(np.mean(gc))
        f["gas_change_std"] = float(np.std(gc))
    else:
        f["gas_change_mean"] = 0.0
        f["gas_change_std"] = 0.0

    if n >= 5:
        gu = sg["gas_used"].to_numpy(dtype=np.float64)
        eff = gu / (g + 1e-12)
        es, _ = _lin_trend(eff)
        f["gas_efficiency_trend"] = es
    else:
        f["gas_efficiency_trend"] = 0.0

    if n >= 2:
        spikes = int(np.sum(g[1:] > 3 * g[:-1]))
        f["gas_spike_freq"] = spikes / (n - 1)
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
        f[f"value_acf_{lag}"] = _acf(vals, lag)
    for lag in [1, 2, 3]:
        f[f"gas_acf_{lag}"] = _acf(gas, lag)
    for lag in [1, 2, 3]:
        f[f"interval_acf_{lag}"] = _acf(iv, lag)

    acf_vals = [_acf(vals, lag) for lag in range(1, min(len(vals), 11))]
    if len(acf_vals) >= 3:
        aa = np.array(acf_vals[:5], dtype=np.float64)
        abs_a = np.abs(aa)
        hi = int(np.argmax(abs_a < 0.5 * abs_a[0])) if abs_a[0] > 0 else len(abs_a)
        f["acf_decay_rate"] = 1.0 - abs_a[min(hi, len(abs_a) - 1)] / abs_a[0] if abs_a[0] > 0 else 0.0
        s = np.sign(aa)
        sc = int(np.sum(s[1:] != s[:-1]))
        f["acf_sign_changes"] = sc / len(s)
    else:
        f["acf_decay_rate"] = 0.0
        f["acf_sign_changes"] = 0.0

    if len(vals) >= 4:
        a1 = _acf(vals, 1)
        a2 = _acf(vals, 2)
        f["periodicity_score"] = max(0.0, a2 - a1 * a1)
    else:
        f["periodicity_score"] = 0.0

    if len(vals) >= 10:
        mv = float(np.mean(vals))
        cd = np.cumsum(vals - mv)
        r = float(np.max(cd) - np.min(cd))
        sv = float(np.std(vals))
        f["stationarity_score"] = r / sv if sv > 0 else 0.0
    else:
        f["stationarity_score"] = 0.0

    f["hurst_exponent"] = _hurst(vals)
    return f


def _corr_pair(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3 or len(b) < 3:
        return 0.0
    sa = float(np.std(a))
    sb = float(np.std(b))
    if sa == 0.0 or sb == 0.0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _lag_feats(group: pd.DataFrame) -> dict[str, float]:
    f: dict[str, float] = {}
    sg = group.sort_values("timestamp")
    vals = sg["value"].to_numpy(dtype=np.float64)
    gas = sg["gas_price"].to_numpy(dtype=np.float64)
    blks = sg["block_number"].to_numpy(dtype=np.float64)
    ts = sg["timestamp"].to_numpy(dtype=np.float64)
    iv = group["interval"].dropna().to_numpy(dtype=np.float64)

    for lag in [1, 2, 3]:
        f[f"value_lag_{lag}"] = _corr_pair(vals[:-lag], vals[lag:])
        f[f"gas_lag_{lag}"] = _corr_pair(gas[:-lag], gas[lag:])

    f["value_gas_corr"] = _corr_pair(vals, gas)
    f["value_block_corr"] = _corr_pair(vals, blks)
    f["gas_block_corr"] = _corr_pair(gas, blks)
    f["value_ts_corr"] = _corr_pair(vals, ts)
    f["gas_ts_corr"] = _corr_pair(gas, ts)

    if len(iv) >= 3:
        vf = vals[1:]
        if len(vf) == len(iv):
            f["interval_value_corr"] = _corr_pair(iv, vf)
        else:
            f["interval_value_corr"] = 0.0
    else:
        f["interval_value_corr"] = 0.0
    return f


def _blk_feats(group: pd.DataFrame) -> dict[str, float]:
    f: dict[str, float] = {}
    blocks = group["block_number"].to_numpy()
    if len(blocks) == 0:
        return {"block_span": 0.0, "blocks_per_tx": 0.0, "unique_blocks_ratio": 0.0, "block_reuse_ratio": 0.0}

    uniq = np.unique(blocks)
    f["block_span"] = float(np.max(blocks) - np.min(blocks))
    f["blocks_per_tx"] = len(uniq) / len(blocks)
    f["unique_blocks_ratio"] = len(uniq) / len(blocks)

    btc = group.groupby("block_number").size()
    f["block_reuse_ratio"] = 1.0 - btc.nunique() / len(btc) if len(btc) > 0 else 0.0
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
    return result.astype(np.float64).fillna(0.0)


def _counting_entropy(values: list[str]) -> float:
    if not values:
        return 0.0
    counts = Counter(values)
    total = sum(counts.values())
    if total == 0:
        return 0.0
    probs = np.array([c / total for c in counts.values()])
    return float(-np.sum(probs * np.log2(probs + 1e-12)))