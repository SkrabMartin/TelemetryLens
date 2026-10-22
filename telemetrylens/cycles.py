"""Operating cycles, phases (start-up / steady / shutdown) and time-in-state."""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import schema
from .quality import gap_thresholds, sample_durations

STARTUP_WINDOW_S = 60.0
SHUTDOWN_WINDOW_S = 30.0


def cycle_keys(df: pd.DataFrame) -> tuple[pd.Series, str]:
    """A cycle label per row (NaN outside cycles) and how cycles were identified."""
    if schema.CYCLE_ID in df.columns and df[schema.CYCLE_ID].notna().any():
        key = df[schema.MACHINE_ID] + "#" + df[schema.CYCLE_ID].astype("string")
        return key.where(df[schema.CYCLE_ID].notna()), "cycle_id column"
    if schema.STATUS not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype="object"), "none"
    active = df[schema.STATUS].isin(schema.ACTIVE_STATUSES)
    m = df[schema.MACHINE_ID]
    dt = df.groupby(m)[schema.TIMESTAMP].diff().dt.total_seconds()
    after_gap = dt > m.map(gap_thresholds(df))
    prev_active = active.groupby(m).shift(1, fill_value=False)
    starts = active & (~prev_active | after_gap)
    number = starts.astype(int).groupby(m).cumsum()
    key = m + "#" + number.astype(str)
    return key.where(active), "status transitions"


def add_cycle_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """Add ``cycle``, ``t_in_cycle_s`` and ``phase`` columns."""
    out = df.copy()
    key, source = cycle_keys(out)
    out["cycle"] = key
    grp = out.groupby("cycle")[schema.TIMESTAMP]
    start, end = grp.transform("min"), grp.transform("max")
    out["t_in_cycle_s"] = (out[schema.TIMESTAMP] - start).dt.total_seconds()
    to_end = (end - out[schema.TIMESTAMP]).dt.total_seconds()
    in_cycle = key.notna()
    phase = np.select(
        [in_cycle & (out.t_in_cycle_s < STARTUP_WINDOW_S), in_cycle & (to_end < SHUTDOWN_WINDOW_S), in_cycle],
        ["start-up", "shutdown", "steady"],
        default="",
    )
    phase = pd.Series(phase, index=out.index)
    if schema.STATUS in out.columns:
        phase = phase.where(phase != "", out[schema.STATUS].fillna("no status"))
    out["phase"] = phase.replace("", "outside cycle")
    return out, source


def cycle_table(df: pd.DataFrame, sensors: list[str]) -> pd.DataFrame:
    """One row per cycle. Expects the columns from :func:`add_cycle_columns`."""
    c = df[df["cycle"].notna()].copy()
    if c.empty:
        return pd.DataFrame()
    c["_dt"] = sample_durations(df).loc[c.index]
    g = c.groupby("cycle", sort=False)
    out = pd.DataFrame(
        {
            "machine_id": g[schema.MACHINE_ID].first(),
            "cycle_id": g[schema.CYCLE_ID].first()
            if schema.CYCLE_ID in c
            else g["cycle"].first().str.split("#").str[-1].astype(int),
            "start": g[schema.TIMESTAMP].min(),
            "end": g[schema.TIMESTAMP].max(),
            "samples": g.size(),
        }
    )
    out["duration_s"] = (out.end - out.start).dt.total_seconds()
    if schema.STATUS in c:
        out["warning_share"] = g[schema.STATUS].apply(lambda s: (s == "warning").mean())
    if "temperature_c" in sensors:
        out["temp_mean_c"] = g.temperature_c.mean()
        out["temp_max_c"] = g.temperature_c.max()
        out["temp_rise_c"] = g.temperature_c.last() - g.temperature_c.first()
    if "vibration_mm_s" in sensors:
        out["vib_max_mm_s"] = g.vibration_mm_s.max()
        out["vib_p95_mm_s"] = g.vibration_mm_s.quantile(0.95)
    if "rpm" in sensors:
        out["rpm_mean"] = g.rpm.mean()
    if "power_kw" in sensors:
        out["power_mean_kw"] = g.power_kw.mean()
        c["_kwh"] = c.power_kw * c._dt / 3600.0
        out["energy_kwh"] = c.groupby("cycle", sort=False)._kwh.sum()
    return out.sort_values(["machine_id", "start"]).reset_index(names="cycle")


def time_in_state(df: pd.DataFrame) -> pd.DataFrame:
    """Hours per machine and status, weighted by sample duration. Gaps count as 'no data'."""
    if schema.STATUS not in df.columns:
        return pd.DataFrame()
    d = df.assign(_dt=sample_durations(df), _status=df[schema.STATUS].fillna("no status"))
    hours = d.pivot_table(index=schema.MACHINE_ID, columns="_status", values="_dt", aggfunc="sum", fill_value=0) / 3600
    span = d.groupby(schema.MACHINE_ID)[schema.TIMESTAMP].agg(lambda s: (s.max() - s.min()).total_seconds()) / 3600
    hours["no data"] = (span - hours.sum(axis=1)).clip(lower=0)
    order = [s for s in [*schema.STATUSES, schema.UNKNOWN_STATUS, "no status", "no data"] if s in hours.columns]
    hours = hours[order]
    hours.columns.name = None
    return hours


def utilisation(df: pd.DataFrame) -> pd.DataFrame:
    h = time_in_state(df)
    if h.empty:
        return h
    active = h[[c for c in h.columns if c in schema.ACTIVE_STATUSES]].sum(axis=1)
    observed = h.drop(columns=["no data"]).sum(axis=1)
    return pd.DataFrame(
        {
            "observed_h": observed,
            "active_h": active,
            "idle_h": h.get("idle", 0.0),
            "stopped_h": h.get("stopped", 0.0),
            "no_data_h": h["no data"],
            "active_share": active / observed.replace(0, np.nan),
        }
    )
