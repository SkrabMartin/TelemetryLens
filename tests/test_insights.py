import numpy as np
import pandas as pd
import pytest

from telemetrylens import insights, schema


def test_vibration_threshold_counts_match_planted_spikes(analysed):
    th = insights.threshold_summary(analysed, ["vibration_mm_s"], schema.default_thresholds()).set_index("machine_id")
    assert th.loc["CNC_03", "above_critical"] == 12
    assert th.loc["CNC_03", "episodes"] == 12
    assert th.above_warn.drop(index="CNC_03").eq(0).all()
    assert th.loc["CNC_03", "startup_share"] == 0


def test_cooling_fault_is_one_steady_state_episode(analysed, synthetic):
    gt = synthetic.ground_truth.set_index("fault").loc["cooling_fault"]
    ep = insights.violation_episodes(analysed, "temperature_c", 75)
    assert ep.machine_id.tolist() == ["CNC_02"]
    e = ep.iloc[0]
    assert gt.start <= e.start <= gt.end
    assert e.phase == "steady"
    assert e.peak > 90


def test_anomalies_find_spikes_without_false_positives(analysed, synthetic):
    an = insights.flag_anomalies(analysed, schema.SENSOR_NAMES, z_limit=6)
    spikes = synthetic.ground_truth[synthetic.ground_truth.fault == "vibration_spike"].start
    vib = an[an.sensor == "vibration_mm_s"]
    assert set(vib.timestamp) == set(spikes)
    other = an[(an.sensor == "temperature_c")]
    assert set(other.machine_id) == {"CNC_02"}
    assert set(an.sensor) == {"vibration_mm_s", "temperature_c"}


def test_threshold_summary_on_hand_made_series():
    ts = pd.date_range("2026-01-01", periods=10, freq="1s", tz="UTC")
    v = [1, 5, 5, 1, 1, 1, 1, 1, 1, 8]
    df = pd.DataFrame({"timestamp": ts, "machine_id": "M", "vibration_mm_s": v})
    th = insights.threshold_summary(df, ["vibration_mm_s"], {"vibration_mm_s": {"warn": 4.5, "critical": 7.1}})
    row = th.iloc[0]
    assert row.above_warn == 3
    assert row.above_critical == 1
    assert row.episodes == 2  # the 6-sample quiet stretch is longer than the merge window
    assert row.time_above_warn_s == 3
    assert row.share_above_warn == pytest.approx(0.3)


def test_episodes_merge_short_dips():
    ts = pd.date_range("2026-01-01", periods=8, freq="1s", tz="UTC")
    df = pd.DataFrame({"timestamp": ts, "machine_id": "M", "temperature_c": [80, 74, 80, 80, 70, 70, 70, 70]})
    ep = insights.violation_episodes(df, "temperature_c", 75)
    assert len(ep) == 1
    assert ep.iloc[0].samples == 3


def test_operating_stats_are_exact():
    df = pd.DataFrame({"machine_id": ["A"] * 4 + ["B"] * 2, "temperature_c": [1.0, 2.0, 3.0, np.nan, 10.0, 20.0]})
    s = insights.operating_stats(df, ["temperature_c"]).set_index("machine_id")
    assert s.loc["A", "n"] == 3 and s.loc["A", "mean"] == 2.0 and s.loc["A", "max"] == 3.0
    assert s.loc["B", "median"] == 15.0


def test_downsampling_changes_the_plot_not_the_statistics(analysed):
    before = insights.operating_stats(analysed, ["vibration_mm_s"])
    plot, note = insights.downsample(analysed, "vibration_mm_s", max_points=2000)
    assert note and len(plot) < len(analysed)
    # Max aggregation keeps every spike visible.
    assert (
        plot.groupby("machine_id").vibration_mm_s.max().round(3).to_dict()
        == analysed.groupby("machine_id").vibration_mm_s.max().round(3).to_dict()
    )
    pd.testing.assert_frame_equal(before, insights.operating_stats(analysed, ["vibration_mm_s"]))


def test_pooled_correlation_can_differ_from_per_machine():
    rng = np.random.default_rng(0)
    a = pd.DataFrame({"machine_id": "A", "power_kw": rng.normal(2, 0.1, 200)})
    a["vibration_mm_s"] = 5 - a.power_kw + rng.normal(0, 0.05, 200)
    b = pd.DataFrame({"machine_id": "B", "power_kw": rng.normal(10, 0.1, 200)})
    b["vibration_mm_s"] = 20 - b.power_kw + rng.normal(0, 0.05, 200)
    r = insights.correlation_by_machine(pd.concat([a, b]), "power_kw", "vibration_mm_s").set_index("machine_id").r
    assert r["A"] < -0.5 and r["B"] < -0.5
    assert r["All machines pooled"] > 0.5


def test_compare_windows(analysed):
    t0 = analysed.timestamp.min()
    cmp = insights.compare_windows(
        analysed,
        ["temperature_c"],
        (t0, t0 + pd.Timedelta("1h")),
        (t0 + pd.Timedelta("3h"), t0 + pd.Timedelta("4h")),
    ).set_index("machine_id")
    assert cmp.loc["CNC_02", "median_b"] > cmp.loc["CNC_02", "median_a"]
    assert cmp.p_value.between(0, 1).all()
