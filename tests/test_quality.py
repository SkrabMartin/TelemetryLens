import numpy as np
import pandas as pd

from telemetrylens import ingest, quality
from tests.conftest import make_csv


def test_synthetic_quality_matches_ground_truth(raw_quality, synthetic):
    m = raw_quality.machines.set_index("machine_id")
    gt = synthetic.ground_truth.set_index("fault")

    assert m.loc["CNC_01", "exact_duplicates"] == 20
    assert m[["exact_duplicates"]].drop(index="CNC_01").eq(0).all().all()

    assert len(raw_quality.gaps) == 1
    gap = raw_quality.gaps.iloc[0]
    assert gap.machine_id == "CNC_01"
    assert gap.missing_samples == gt.loc["data_gap", "n_samples"] == 900

    d = raw_quality.dropouts
    assert d[["machine_id", "sensor", "samples"]].values.tolist() == [["MOTOR_01", "temperature_c", 300]]

    imp = raw_quality.implausible
    assert imp.loc["MOTOR_02", "temperature_c"] == 5
    assert imp.loc["MOTOR_02", "rpm"] == 3
    assert imp.to_numpy().sum() == 8

    irregular = m.irregular_intervals
    assert irregular["MOTOR_02"] > 0
    assert irregular.drop(index="MOTOR_02").eq(0).all()
    assert m.loc["CNC_01", "coverage"] == (28800 - 900) / 28800


def test_cleaning_logs_every_decision(cleaned, parsed):
    assert cleaned.rows_before - cleaned.rows_after == 20
    assert cleaned.masked == {
        "temperature_c": 5,
        "vibration_mm_s": 0,
        "rpm": 3,
        "current_a": 0,
        "power_kw": 0,
        "torque_nm": 0,
    }
    text = " ".join(cleaned.decisions)
    assert "20 exact duplicate rows" in text
    assert "not interpolated" in text
    # The parsed (pre-cleaning) frame is not modified.
    assert (parsed.data.temperature_c == -999).sum() == 5


def test_cleaning_does_not_interpolate(cleaned, parsed):
    before = parsed.data.drop_duplicates(subset=[c for c in parsed.data.columns if c != "source_row"])
    for s in parsed.sensors:
        expected = before[s].isna().sum() + cleaned.masked[s]
        assert cleaned.data[s].isna().sum() == expected


def _conflict_frame():
    return ingest.parse(
        make_csv(
            [
                "2026-10-22T10:00:00Z,M1,40,1.0,running",
                "2026-10-22T10:00:00Z,M1,44,1.0,running",
                "2026-10-22T10:00:01Z,M1,41,1.0,running",
            ]
        )
    )


def test_conflicting_timestamps_policies():
    p = _conflict_frame()
    q = quality.quality_report(p.data, p.sensors)
    assert q.machines.conflicting_timestamps.iloc[0] == 1
    first = quality.clean(p.data, p.sensors, quality.CleaningOptions(conflicting_timestamps="keep_first"))
    assert first.data.temperature_c.tolist() == [40, 41]
    mean = quality.clean(p.data, p.sensors, quality.CleaningOptions(conflicting_timestamps="mean"))
    assert mean.data.temperature_c.tolist() == [42, 41]
    keep = quality.clean(p.data, p.sensors, quality.CleaningOptions(conflicting_timestamps="keep_all"))
    assert len(keep.data) == 3


def test_sample_durations_do_not_bridge_gaps():
    ts = pd.to_datetime(
        [
            "2026-01-01T00:00:00Z",
            "2026-01-01T00:00:01Z",
            "2026-01-01T00:00:02Z",
            "2026-01-01T00:10:00Z",
            "2026-01-01T00:10:01Z",
        ]
    )
    df = pd.DataFrame({"timestamp": ts, "machine_id": "M", "temperature_c": np.arange(5.0)})
    assert quality.sample_durations(df).tolist() == [1, 1, 0, 1, 1]
