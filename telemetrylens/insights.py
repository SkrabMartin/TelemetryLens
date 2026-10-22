"""Operating statistics, threshold violations and explainable anomaly flags.

All methods are rule-based or descriptive statistics. Nothing here forecasts.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy import stats

from . import schema
from .quality import sample_durations

PERCENTILES = [0.05, 0.5, 0.95]
# Episodes separated by at most this many samples below the limit are merged, so
# noise around a threshold does not split one excursion into many.
EPISODE_MERGE_SAMPLES = 5


def operating_stats(df: pd.DataFrame, sensors: list[str]) -> pd.DataFrame:
    """Descriptive statistics per machine and sensor, computed on raw (not resampled) samples."""
    rows = []
    for (mid, sensor), s in df.melt(id_vars=[schema.MACHINE_ID], value_vars=sensors, var_name="sensor").groupby(
        [schema.MACHINE_ID, "sensor"], sort=True
    )["value"]:
        v = s.dropna()
        q = v.quantile(PERCENTILES) if len(v) else pd.Series(np.nan, index=PERCENTILES)
        rows.append(
            dict(
                machine_id=mid,
                sensor=sensor,
                unit=schema.SENSORS[sensor].unit,
                n=len(v),
                mean=v.mean(),
                std=v.std(),
                min=v.min(),
                p05=q[0.05],
                median=q[0.5],
                p95=q[0.95],
                max=v.max(),
            )
        )
    return pd.DataFrame(rows)


def _episodes(mask: np.ndarray, merge: int = EPISODE_MERGE_SAMPLES) -> list[tuple[int, int]]:
    if not mask.any():
        return []
    d = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    runs = list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1) - 1, strict=True))
    merged = [runs[0]]
    for a, b in runs[1:]:
        if a - merged[-1][1] - 1 <= merge:
            merged[-1] = (merged[-1][0], b)
        else:
            merged.append((a, b))
    return merged


def violation_episodes(df: pd.DataFrame, sensor: str, limit: float) -> pd.DataFrame:
    """Consecutive runs of samples at or above ``limit``, per machine."""
    rows = []
    has_phase = "phase" in df.columns
    for mid, g in df.groupby(schema.MACHINE_ID, sort=True):
        v = g[sensor].to_numpy()
        ts = g[schema.TIMESTAMP]
        for a, b in _episodes(np.nan_to_num(v, nan=-np.inf) >= limit):
            rows.append(
                dict(
                    machine_id=mid,
                    sensor=sensor,
                    start=ts.iloc[a],
                    end=ts.iloc[b],
                    duration_s=(ts.iloc[b] - ts.iloc[a]).total_seconds(),
                    samples=int(np.sum(v[a : b + 1] >= limit)),
                    peak=float(np.nanmax(v[a : b + 1])),
                    phase=g["phase"].iloc[a] if has_phase else None,
                )
            )
    return pd.DataFrame(
        rows, columns=["machine_id", "sensor", "start", "end", "duration_s", "samples", "peak", "phase"]
    )


def threshold_summary(
    df: pd.DataFrame, sensors: list[str], thresholds: dict[str, dict[str, float | None]]
) -> pd.DataFrame:
    """How often each machine exceeds warn / critical limits, and whether it happens at start-up."""
    dt = sample_durations(df)
    rows = []
    for sensor in sensors:
        lim = thresholds.get(sensor, {})
        warn, crit = lim.get("warn"), lim.get("critical")
        if warn is None and crit is None:
            continue
        low = warn if warn is not None else crit
        for mid, g in df.groupby(schema.MACHINE_ID, sort=True):
            v = g[sensor]
            observed = int(v.notna().sum())
            above_warn = v >= low
            above_crit = v >= crit if crit is not None else pd.Series(False, index=v.index)
            n_warn = int(above_warn.sum())
            startup = int((above_warn & (g["phase"] == "start-up")).sum()) if "phase" in g else 0
            rows.append(
                dict(
                    machine_id=mid,
                    sensor=sensor,
                    warn=warn,
                    critical=crit,
                    observed=observed,
                    above_warn=n_warn,
                    above_critical=int(above_crit.sum()),
                    share_above_warn=n_warn / observed if observed else np.nan,
                    time_above_warn_s=float(dt[above_warn[above_warn].index].sum()),
                    episodes=len(_episodes(above_warn.to_numpy())),
                    peak=float(v.max()) if observed else np.nan,
                    startup_share=startup / n_warn if n_warn else np.nan,
                )
            )
    return pd.DataFrame(rows)


def robust_zscores(df: pd.DataFrame, sensor: str, group_by: str = "phase") -> pd.Series:
    """Robust z-score of each sample against its own machine and operating phase.

    z = (x - median) / (1.4826 * MAD), with a small floor on the scale so that near
    constant groups (a stopped machine) do not turn rounding noise into anomalies.
    """
    keys = [df[schema.MACHINE_ID]] + ([df[group_by]] if group_by in df.columns else [])
    g = df[sensor].groupby(keys)
    med = g.transform("median")
    mad = (df[sensor] - med).abs().groupby(keys).transform("median") * 1.4826
    spread = df[sensor].quantile(0.95) - df[sensor].quantile(0.05)
    floor = max(0.02 * spread, 1e-6) if np.isfinite(spread) else 1e-6
    return (df[sensor] - med) / mad.clip(lower=floor)


def flag_anomalies(
    df: pd.DataFrame,
    sensors: list[str],
    z_limit: float = 6.0,
    group_by: str = "phase",
    direction: str = "high",
) -> pd.DataFrame:
    """Long table of flagged samples: machine, sensor, timestamp, value, z, phase, direction.

    ``direction`` is "high" (default: overheating, vibration, overload) or "both". Low
    values are mostly warm-up after a stop, which is expected behaviour.
    """
    parts = []
    for sensor in sensors:
        z = robust_zscores(df, sensor, group_by)
        hit = z >= z_limit if direction == "high" else z.abs() >= z_limit
        if not hit.any():
            continue
        sub = df.loc[hit, [schema.TIMESTAMP, schema.MACHINE_ID, sensor] + (["phase"] if "phase" in df else [])]
        sub = sub.rename(columns={sensor: "value"}).assign(sensor=sensor, z=z[hit].round(2))
        sub["direction"] = np.where(sub.z > 0, "high", "low")
        parts.append(sub)
    if not parts:
        return pd.DataFrame(columns=[schema.TIMESTAMP, schema.MACHINE_ID, "value", "phase", "sensor", "z", "direction"])
    return pd.concat(parts, ignore_index=True).sort_values([schema.MACHINE_ID, schema.TIMESTAMP])


def anomaly_summary(anomalies: pd.DataFrame) -> pd.DataFrame:
    if anomalies.empty:
        return pd.DataFrame(columns=["machine_id", "sensor", "flags", "high", "low", "max_abs_z"])
    g = anomalies.groupby([schema.MACHINE_ID, "sensor"])
    return pd.DataFrame(
        {
            "flags": g.size(),
            "high": g.direction.apply(lambda s: int((s == "high").sum())),
            "low": g.direction.apply(lambda s: int((s == "low").sum())),
            "max_abs_z": g.z.apply(lambda s: float(s.abs().max())),
        }
    ).reset_index()


def state_summary(df: pd.DataFrame, sensors: list[str]) -> pd.DataFrame:
    """Time share and mean sensor values per machine and status."""
    if schema.STATUS not in df.columns:
        return pd.DataFrame()
    d = df.assign(_dt=sample_durations(df))
    g = d.groupby([schema.MACHINE_ID, schema.STATUS])
    out = g[sensors].mean()
    out.insert(0, "hours", g._dt.sum() / 3600)
    out.insert(1, "time_share", out.hours / out.hours.groupby(level=0).transform("sum"))
    return out.reset_index()


def correlations(df: pd.DataFrame, sensors: list[str], method: str = "spearman") -> pd.DataFrame:
    return df[sensors].corr(method=method)


def correlation_by_machine(df: pd.DataFrame, x: str, y: str, method: str = "spearman") -> pd.DataFrame:
    """Correlation of two sensors per machine plus pooled, to expose Simpson-type effects."""
    rows = [
        dict(machine_id=mid, n=int(g[[x, y]].dropna().shape[0]), r=g[x].corr(g[y], method=method))
        for mid, g in df.groupby(schema.MACHINE_ID, sort=True)
    ]
    rows.append(
        dict(machine_id="All machines pooled", n=int(df[[x, y]].dropna().shape[0]), r=df[x].corr(df[y], method=method))
    )
    return pd.DataFrame(rows)


def compare_windows(df: pd.DataFrame, sensors: list[str], window_a: tuple, window_b: tuple) -> pd.DataFrame:
    """Per machine and sensor statistics in two time windows, with a Mann-Whitney U test.

    The p-value assumes independent samples. Telemetry is autocorrelated, so treat it
    as a rough indicator and look at the effect size (difference of medians) first.
    """
    ts = df[schema.TIMESTAMP]
    a = df[(ts >= window_a[0]) & (ts <= window_a[1])]
    b = df[(ts >= window_b[0]) & (ts <= window_b[1])]
    rows = []
    for mid in sorted(set(a[schema.MACHINE_ID]) | set(b[schema.MACHINE_ID])):
        ga, gb = a[a[schema.MACHINE_ID] == mid], b[b[schema.MACHINE_ID] == mid]
        for s in sensors:
            va, vb = ga[s].dropna(), gb[s].dropna()
            p = np.nan
            if (
                len(va) >= 5
                and len(vb) >= 5
                and not (va.nunique() == 1 and vb.nunique() == 1 and va.iat[0] == vb.iat[0])
            ):
                p = float(stats.mannwhitneyu(va, vb, alternative="two-sided").pvalue)
            rows.append(
                dict(
                    machine_id=mid,
                    sensor=s,
                    n_a=len(va),
                    n_b=len(vb),
                    median_a=va.median(),
                    median_b=vb.median(),
                    delta_median=vb.median() - va.median(),
                    mean_a=va.mean(),
                    mean_b=vb.mean(),
                    p95_a=va.quantile(0.95) if len(va) else np.nan,
                    p95_b=vb.quantile(0.95) if len(vb) else np.nan,
                    p_value=p,
                )
            )
    return pd.DataFrame(rows)


def downsample(df: pd.DataFrame, sensor: str, max_points: int = 4000) -> tuple[pd.DataFrame, str | None]:
    """Reduce points per machine for plotting using the sensor's aggregation policy.

    Statistics elsewhere are always computed from the full data; this is display only.
    Gaps stay visible because empty buckets are kept as NaN.
    """
    cols = [schema.TIMESTAMP, schema.MACHINE_ID, sensor]
    per_machine = df.groupby(schema.MACHINE_ID).size()
    if per_machine.empty or per_machine.max() <= max_points:
        return df[cols], None
    span = (df[schema.TIMESTAMP].max() - df[schema.TIMESTAMP].min()).total_seconds()
    bucket_s = max(1, math.ceil(span / max_points))
    agg = schema.SENSORS[sensor].resample_agg
    out = (
        df.set_index(schema.TIMESTAMP)
        .groupby(schema.MACHINE_ID)[sensor]
        .resample(f"{bucket_s}s")
        .agg(agg)
        .reset_index()
    )
    return out[cols], f"Display resampled to {bucket_s} s buckets ({agg} per bucket); statistics use all samples."
