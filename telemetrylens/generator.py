"""Synthetic telemetry for CNC mills and industrial motors.

The generator simulates start-up, stable operation, idle time and shutdown, and
then injects a known set of data faults. Every injected fault is returned in a
ground-truth table, so tests (and users) can check that the analysis finds
exactly what was planted.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.signal import lfilter

from . import schema


@dataclass(frozen=True)
class MachineProfile:
    machine_id: str
    kind: str
    nominal_rpm: float
    nominal_torque_nm: float
    ambient_c: float
    temp_rise_c: float  # steady-state rise above ambient at nominal power
    thermal_tau_s: float
    vib_per_krpm: float
    interval_s: float


PROFILES: list[MachineProfile] = [
    MachineProfile("CNC_01", "CNC mill", 1500, 25, 23.0, 26.0, 600, 0.9, 1.0),
    MachineProfile("CNC_02", "CNC mill", 2200, 20, 24.0, 30.0, 600, 0.9, 1.0),
    MachineProfile("CNC_03", "CNC mill", 1800, 30, 22.5, 28.0, 650, 0.9, 1.0),
    MachineProfile("MOTOR_01", "Electric motor", 2950, 60, 25.0, 32.0, 900, 0.8, 2.0),
    MachineProfile("MOTOR_02", "Electric motor", 1475, 90, 24.0, 35.0, 900, 0.8, 2.0),
]

STARTUP_S = 45.0
SHUTDOWN_S = 25.0
EFFICIENCY = 0.9
VOLTAGE_V = 400.0
POWER_FACTOR = 0.85

N_VIBRATION_SPIKES = 12
N_DUPLICATES = 20
N_OUT_OF_ORDER = 10
N_TEMP_SENTINELS = 5
N_RPM_SPIKES = 3
DROPOUT_S = 10 * 60
GAP_S = 15 * 60


@dataclass
class SyntheticDataset:
    data: pd.DataFrame
    ground_truth: pd.DataFrame


def _clipped_noise(rng: np.random.Generator, sigma: float, n: int) -> np.ndarray:
    return np.clip(rng.normal(0.0, sigma, n), -3 * sigma, 3 * sigma)


def _schedule(rng: np.random.Generator, horizon_s: float) -> list[tuple[str, float]]:
    segments: list[tuple[str, float]] = [
        ("stopped", rng.uniform(120, 600)),
        ("idle", rng.uniform(90, 240)),
    ]
    t = sum(d for _, d in segments)
    while t < horizon_s:
        run = rng.uniform(8 * 60, 25 * 60)
        segments.append(("running", run))
        t += run
        if rng.random() < 0.12:
            pause = ("stopped", rng.uniform(10 * 60, 20 * 60))
        else:
            pause = ("idle", rng.uniform(2 * 60, 6 * 60))
        segments.append(pause)
        t += pause[1]
    return segments


def _simulate_machine(p: MachineProfile, index: int, horizon_s: float, rng: np.random.Generator) -> pd.DataFrame:
    segments = _schedule(rng, horizon_s)
    seg_state = np.array([s for s, _ in segments])
    seg_len = np.array([d for _, d in segments])
    seg_end = np.cumsum(seg_len)
    seg_start = seg_end - seg_len
    is_run_seg = seg_state == "running"
    seg_cycle = np.where(is_run_seg, np.cumsum(is_run_seg) + 100 * (index + 1), np.nan)
    seg_load = rng.uniform(0.6, 1.0, len(segments))
    seg_speed = rng.uniform(0.85, 1.05, len(segments))

    t = np.arange(0.0, horizon_s, p.interval_s)
    n = len(t)
    seg = np.searchsorted(seg_end, t, side="right")
    state = seg_state[seg]
    t_in = t - seg_start[seg]
    length = seg_len[seg]
    running = state == "running"
    idle = state == "idle"

    ramp = np.clip(t_in / STARTUP_S, 0, 1) * np.clip((length - t_in) / SHUTDOWN_S, 0, 1)
    ramp = np.where(running, ramp, 0.0)
    load = np.clip(seg_load[seg] + 0.05 * np.sin(2 * np.pi * t / 300.0 + index), 0.3, 1.05)

    rpm = p.nominal_rpm * seg_speed[seg] * ramp + _clipped_noise(rng, 2.0, n) * ramp
    torque = p.nominal_torque_nm * load * ramp * (1 + _clipped_noise(rng, 0.02, n))
    mech_kw = torque * rpm * 2 * np.pi / 60.0 / 1000.0
    base_kw = np.select([running, idle], [0.5, 0.35], 0.02)
    power = mech_kw / EFFICIENCY + base_kw + _clipped_noise(rng, 0.03, n)
    current = (
        power * 1000.0 / (np.sqrt(3) * VOLTAGE_V * POWER_FACTOR)
        + np.select([running, idle], [1.2, 0.8], 0.0)
        + _clipped_noise(rng, 0.05, n)
    )

    nominal_kw = p.nominal_rpm * p.nominal_torque_nm * 2 * np.pi / 60.0 / 1000.0 / EFFICIENCY
    temp_target = p.ambient_c + p.temp_rise_c * np.clip(power, 0, None) / nominal_kw

    vibration = np.select(
        [running, idle],
        [
            0.25
            + p.vib_per_krpm * rpm / 1000.0 * (0.7 + 0.5 * load)
            + 1.0 * np.exp(-(((t_in - 25.0) / 8.0) ** 2)),  # resonance pass at start-up
            0.15,
        ],
        0.05,
    ) + _clipped_noise(rng, 0.08, n) * np.where(running, 1.0, 0.3)

    return pd.DataFrame(
        {
            "t": t,
            "machine_id": p.machine_id,
            "cycle_id": seg_cycle[seg],
            "state": state,
            "seg": seg,
            "t_in": t_in,
            "seg_len": length,
            "ramp": ramp,
            "temp_target": temp_target,
            "vibration_mm_s": np.clip(vibration, 0.02, None),
            "rpm": np.clip(rpm, 0, None),
            "current_a": np.clip(current, 0, None),
            "power_kw": np.clip(power, 0, None),
            "torque_nm": np.clip(torque, 0, None),
        }
    )


def _thermal(target: np.ndarray, start: float, interval_s: float, tau_s: float) -> np.ndarray:
    a = np.exp(-interval_s / tau_s)
    y, _ = lfilter([1 - a], [1, -a], target, zi=[a * start])
    return y


def generate(
    n_machines: int = 5,
    hours: float = 8.0,
    start: str = "2026-10-22T06:00:00Z",
    seed: int = 42,
    inject_faults: bool = True,
) -> SyntheticDataset:
    """Generate telemetry for ``n_machines`` (1-5) over ``hours`` of operation."""
    if not 1 <= n_machines <= len(PROFILES):
        raise ValueError(f"n_machines must be between 1 and {len(PROFILES)}")
    rng = np.random.default_rng(seed)
    horizon_s = hours * 3600.0
    t0 = pd.Timestamp(start)
    t0 = t0.tz_localize("UTC") if t0.tzinfo is None else t0.tz_convert("UTC")
    profiles = {p.machine_id: p for p in PROFILES[:n_machines]}
    truth: list[dict] = []

    frames = []
    for i, p in enumerate(profiles.values()):
        m = _simulate_machine(p, i, horizon_s, rng)

        # Random missing readings (sensor noise / dropped packets), ~0.2% per sensor.
        for col in schema.SENSOR_NAMES:
            if col == "temperature_c":
                continue
            mask = rng.random(len(m)) < 0.002
            m.loc[mask, col] = np.nan

        if inject_faults and p.machine_id == "CNC_02":
            # Cooling fault: extra heat builds up during one long cycle mid-shift.
            runs = m[m.state == "running"].groupby("seg").agg(start=("t", "min"), length=("seg_len", "first"))
            middle = runs[(runs.start > horizon_s * 0.35) & (runs.start < horizon_s * 0.7)]
            seg_id = (middle if len(middle) else runs).length.idxmax()
            in_seg = (m.seg == seg_id).to_numpy()
            extra = 75.0 * (m.t_in / m.seg_len).to_numpy()
            m["temp_target"] = m.temp_target + np.where(in_seg, extra, 0.0)
            cyc = m.loc[in_seg, "cycle_id"].iloc[0]
            truth.append(
                dict(
                    fault="cooling_fault",
                    machine_id=p.machine_id,
                    sensor="temperature_c",
                    start=m.loc[in_seg, "t"].min(),
                    end=m.loc[in_seg, "t"].max(),
                    n_samples=int(in_seg.sum()),
                    note=f"Temperature climbs during cycle {int(cyc)} while load is normal.",
                )
            )

        m["temperature_c"] = _thermal(
            m.temp_target.to_numpy(), p.ambient_c, p.interval_s, p.thermal_tau_s
        ) + _clipped_noise(rng, 0.15, len(m))

        if inject_faults and p.machine_id == "CNC_03":
            steady = np.flatnonzero(
                (m.state == "running").to_numpy() & (m.ramp >= 1).to_numpy() & m.vibration_mm_s.notna().to_numpy()
            )
            idx = np.sort(rng.choice(steady, N_VIBRATION_SPIKES, replace=False))
            m.loc[m.index[idx], "vibration_mm_s"] = rng.uniform(8.0, 14.0, N_VIBRATION_SPIKES)
            for j in idx:
                truth.append(
                    dict(
                        fault="vibration_spike",
                        machine_id=p.machine_id,
                        sensor="vibration_mm_s",
                        start=m.t.iat[j],
                        end=m.t.iat[j],
                        n_samples=1,
                        note="Single-sample impact spike above the critical zone.",
                    )
                )

        if inject_faults and p.machine_id == "MOTOR_01":
            s0 = round(horizon_s * 0.30 / p.interval_s) * p.interval_s
            mask = (m.t >= s0) & (m.t < s0 + DROPOUT_S)
            m.loc[mask, "temperature_c"] = np.nan
            truth.append(
                dict(
                    fault="sensor_dropout",
                    machine_id=p.machine_id,
                    sensor="temperature_c",
                    start=s0,
                    end=s0 + DROPOUT_S - p.interval_s,
                    n_samples=int(mask.sum()),
                    note="Temperature sensor disconnected for 10 minutes.",
                )
            )

        if inject_faults and p.machine_id == "MOTOR_02":
            # Jittered, irregular sampling plus implausible sentinel values.
            m["t"] = m.t + rng.uniform(-0.6, 0.6, len(m))
            m = m[rng.random(len(m)) >= 0.03].reset_index(drop=True)
            for col, value, k, fault in [
                ("temperature_c", -999.0, N_TEMP_SENTINELS, "implausible_temperature"),
                ("rpm", 99999.0, N_RPM_SPIKES, "implausible_rpm"),
            ]:
                idx = rng.choice(np.flatnonzero(m[col].notna().to_numpy()), k, replace=False)
                m.loc[m.index[idx], col] = value
                truth.append(
                    dict(
                        fault=fault,
                        machine_id=p.machine_id,
                        sensor=col,
                        start=m.t.iloc[idx].min(),
                        end=m.t.iloc[idx].max(),
                        n_samples=k,
                        note=f"Sentinel value {value:g} written by the controller.",
                    )
                )
            truth.append(
                dict(
                    fault="irregular_sampling",
                    machine_id=p.machine_id,
                    sensor=None,
                    start=m.t.min(),
                    end=m.t.max(),
                    n_samples=len(m),
                    note="2 s nominal interval with ±0.6 s jitter and ~3% dropped samples.",
                )
            )

        if inject_faults and p.machine_id == "CNC_01":
            g0 = round(horizon_s * 0.40)
            gap = (m.t >= g0) & (m.t < g0 + GAP_S)
            truth.append(
                dict(
                    fault="data_gap",
                    machine_id=p.machine_id,
                    sensor=None,
                    start=float(g0),
                    end=float(g0 + GAP_S),
                    n_samples=int(gap.sum()),
                    note="Logger offline for 15 minutes: rows are missing entirely.",
                )
            )
            m = m[~gap].reset_index(drop=True)

        # Controller flags a warning while running above warn thresholds.
        temp_warn = schema.SENSORS["temperature_c"].warn
        vib_warn = schema.SENSORS["vibration_mm_s"].warn
        over = (m.temperature_c.between(temp_warn, schema.SENSORS["temperature_c"].plausible_max)) | (
            m.vibration_mm_s > vib_warn
        )
        m["status"] = np.where((m.state == "running") & over, "warning", m.state)
        frames.append(m)

    df = pd.concat(frames, ignore_index=True)
    df["timestamp"] = t0 + pd.to_timedelta(df.t, unit="s")
    df = df.sort_values(["timestamp", "machine_id"], kind="stable").reset_index(drop=True)

    if inject_faults and "CNC_01" in profiles:
        rows = df.index[df.machine_id == "CNC_01"]
        dup_idx = np.sort(rng.choice(rows, N_DUPLICATES, replace=False))
        df = pd.concat([df, df.loc[dup_idx]]).sort_index(kind="stable").reset_index(drop=True)
        truth.append(
            dict(
                fault="duplicate_rows",
                machine_id="CNC_01",
                sensor=None,
                start=None,
                end=None,
                n_samples=N_DUPLICATES,
                note="Exact duplicate rows re-sent by the logger.",
            )
        )

    if inject_faults and "CNC_03" in profiles:
        pos = np.flatnonzero((df.machine_id == "CNC_03").to_numpy())
        picks = rng.choice(np.arange(10, len(pos) - 10), N_OUT_OF_ORDER, replace=False)
        order = np.arange(len(df))
        for k in picks:
            a, b = pos[k], pos[k + 1]
            order[a], order[b] = order[b], order[a]
        df = df.iloc[order].reset_index(drop=True)
        truth.append(
            dict(
                fault="out_of_order_rows",
                machine_id="CNC_03",
                sensor=None,
                start=None,
                end=None,
                n_samples=N_OUT_OF_ORDER,
                note="Adjacent rows swapped in the file (logger buffering).",
            )
        )

    out = pd.DataFrame(
        {
            "timestamp": df.timestamp,
            "machine_id": df.machine_id,
            "cycle_id": df.cycle_id.astype("Int64"),
            "temperature_c": df.temperature_c.round(2),
            "vibration_mm_s": df.vibration_mm_s.round(3),
            "rpm": df.rpm.round(0),
            "current_a": df.current_a.round(2),
            "power_kw": df.power_kw.round(3),
            "torque_nm": df.torque_nm.round(2),
            "status": df.status,
        }
    )

    gt = pd.DataFrame(truth, columns=["fault", "machine_id", "sensor", "start", "end", "n_samples", "note"])
    for col in ("start", "end"):
        gt[col] = t0 + pd.to_timedelta(pd.to_numeric(gt[col]), unit="s")
    return SyntheticDataset(out, gt)


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    """Serialise telemetry the way an equipment export would (ISO 8601, UTC, 'Z')."""
    out = df.copy()
    ts = out["timestamp"].dt.tz_convert("UTC").dt.tz_localize(None).to_numpy()
    out["timestamp"] = np.char.add(np.datetime_as_string(ts, unit="ms"), "Z")
    return out.to_csv(index=False).encode("utf-8")
