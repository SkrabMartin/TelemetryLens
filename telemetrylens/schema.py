"""Data dictionary: canonical telemetry fields, units, plausible ranges and policies.

Everything that decides how a value is judged (plausibility, thresholds,
resampling aggregation) lives here so it can be shown to the user and written
into exported reports.
"""

from __future__ import annotations

from dataclasses import dataclass

TIMESTAMP = "timestamp"
MACHINE_ID = "machine_id"
CYCLE_ID = "cycle_id"
STATUS = "status"


@dataclass(frozen=True)
class SensorSpec:
    name: str
    label: str
    unit: str
    description: str
    plausible_min: float
    plausible_max: float
    # Resampling policy used for plotting only. Statistics always use raw samples.
    resample_agg: str
    warn: float | None = None
    critical: float | None = None


SENSORS: dict[str, SensorSpec] = {
    s.name: s
    for s in [
        SensorSpec(
            "temperature_c",
            "Temperature",
            "°C",
            "Motor or spindle temperature",
            -20.0,
            150.0,
            "mean",
            warn=75.0,
            critical=90.0,
        ),
        SensorSpec(
            "vibration_mm_s",
            "Vibration",
            "mm/s",
            "RMS vibration velocity (ISO 10816 style zones)",
            0.0,
            50.0,
            "max",
            warn=4.5,
            critical=7.1,
        ),
        SensorSpec(
            "rpm",
            "Speed",
            "rpm",
            "Spindle or motor rotational speed",
            0.0,
            12000.0,
            "mean",
        ),
        SensorSpec(
            "current_a",
            "Current",
            "A",
            "Motor phase current",
            0.0,
            200.0,
            "mean",
        ),
        SensorSpec(
            "power_kw",
            "Power",
            "kW",
            "Electrical power consumption",
            0.0,
            100.0,
            "mean",
        ),
        SensorSpec(
            "torque_nm",
            "Torque",
            "Nm",
            "Mechanical load on the spindle or shaft",
            0.0,
            500.0,
            "mean",
        ),
    ]
}

SENSOR_NAMES: list[str] = list(SENSORS)

STATUSES: list[str] = ["running", "warning", "idle", "stopped"]
ACTIVE_STATUSES: frozenset[str] = frozenset({"running", "warning"})
UNKNOWN_STATUS = "unknown"

# Common spellings found in equipment exports, normalised to canonical states.
STATUS_ALIASES: dict[str, str] = {
    "run": "running",
    "running": "running",
    "active": "running",
    "on": "running",
    "busy": "running",
    "producing": "running",
    "1": "running",
    "warn": "warning",
    "warning": "warning",
    "alarm": "warning",
    "alert": "warning",
    "idle": "idle",
    "standby": "idle",
    "ready": "idle",
    "waiting": "idle",
    "stop": "stopped",
    "stopped": "stopped",
    "off": "stopped",
    "down": "stopped",
    "halt": "stopped",
    "0": "stopped",
}

# Column-name aliases used to suggest a mapping for unfamiliar CSV headers.
COLUMN_ALIASES: dict[str, list[str]] = {
    TIMESTAMP: ["timestamp", "time", "ts", "datetime", "date_time", "time_utc", "sample_time"],
    MACHINE_ID: ["machine_id", "machine", "device_id", "device", "asset_id", "asset", "unit_id"],
    CYCLE_ID: ["cycle_id", "cycle", "job_id", "batch_id", "run_id"],
    STATUS: ["status", "state", "machine_status", "mode", "machine_state"],
    "temperature_c": ["temperature_c", "temperature", "temp", "temp_c", "spindle_temp", "motor_temp"],
    "vibration_mm_s": ["vibration_mm_s", "vibration", "vib", "vibration_rms", "vib_rms"],
    "rpm": ["rpm", "speed", "spindle_speed", "motor_speed", "speed_rpm"],
    "current_a": ["current_a", "current", "amps", "motor_current", "current_amp"],
    "power_kw": ["power_kw", "power", "kw", "power_consumption"],
    "torque_nm": ["torque_nm", "torque", "load_nm", "load"],
}

# Supported source units and how to convert them to the canonical unit.
UNIT_CONVERSIONS: dict[str, dict[str, tuple[float, float]]] = {
    # sensor -> unit -> (scale, offset) applied as value * scale + offset
    "temperature_c": {"°C": (1.0, 0.0), "°F": (5.0 / 9.0, -32.0 * 5.0 / 9.0), "K": (1.0, -273.15)},
    "vibration_mm_s": {"mm/s": (1.0, 0.0), "in/s": (25.4, 0.0)},
    "power_kw": {"kW": (1.0, 0.0), "W": (0.001, 0.0)},
    "current_a": {"A": (1.0, 0.0), "mA": (0.001, 0.0)},
    "torque_nm": {"Nm": (1.0, 0.0), "lbf·ft": (1.3558179, 0.0)},
    "rpm": {"rpm": (1.0, 0.0), "Hz": (60.0, 0.0)},
}


def label(sensor: str) -> str:
    spec = SENSORS[sensor]
    return f"{spec.label} ({spec.unit})"


def default_thresholds() -> dict[str, dict[str, float | None]]:
    return {n: {"warn": s.warn, "critical": s.critical} for n, s in SENSORS.items()}
