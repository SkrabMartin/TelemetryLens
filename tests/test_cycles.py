import pandas as pd

from telemetrylens import cycles


def _status_frame():
    ts = pd.date_range("2026-01-01", periods=12, freq="10s", tz="UTC")
    status = [
        "idle",
        "running",
        "running",
        "warning",
        "idle",
        "idle",
        "running",
        "running",
        "stopped",
        "stopped",
        "idle",
        "idle",
    ]
    return pd.DataFrame({"timestamp": ts, "machine_id": "M", "status": status, "power_kw": 1.0})


def test_cycles_from_status_transitions():
    df, source = cycles.add_cycle_columns(_status_frame())
    assert source == "status transitions"
    assert df.cycle.dropna().unique().tolist() == ["M#1", "M#2"]
    ct = cycles.cycle_table(df, ["power_kw"])
    assert ct.duration_s.tolist() == [20.0, 10.0]


def test_cycles_from_cycle_id(analysed):
    ct = cycles.cycle_table(analysed, ["temperature_c", "power_kw"])
    assert ct.groupby("machine_id").size().min() >= 15
    assert (ct.duration_s > 0).all()
    hottest = ct.loc[ct.temp_max_c.idxmax()]
    assert hottest.machine_id == "CNC_02"


def test_phases():
    df, _ = cycles.add_cycle_columns(_status_frame())
    assert df.phase.tolist()[:4] == ["idle", "start-up", "start-up", "start-up"]


def test_time_in_state_counts_gaps_as_no_data(analysed):
    h = cycles.time_in_state(analysed)
    assert h.loc["CNC_01", "no data"] > 0.24
    assert h.drop(index="CNC_01")["no data"].max() < 0.01
    u = cycles.utilisation(analysed)
    assert u.active_share.between(0.5, 0.95).all()
