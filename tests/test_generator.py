import pandas as pd

from telemetrylens import generator, schema


def test_shape_and_columns(synthetic):
    df = synthetic.data
    assert list(df.columns) == [
        "timestamp",
        "machine_id",
        "cycle_id",
        *schema.SENSOR_NAMES[:1],
        "vibration_mm_s",
        "rpm",
        "current_a",
        "power_kw",
        "torque_nm",
        "status",
    ]
    assert sorted(df.machine_id.unique()) == ["CNC_01", "CNC_02", "CNC_03", "MOTOR_01", "MOTOR_02"]
    assert set(df.status.unique()) <= set(schema.STATUSES)
    assert len(df) > 100_000


def test_is_deterministic():
    a = generator.generate(n_machines=3, hours=1, seed=7).data
    b = generator.generate(n_machines=3, hours=1, seed=7).data
    pd.testing.assert_frame_equal(a, b)


def test_ground_truth_lists_every_fault(synthetic):
    faults = set(synthetic.ground_truth.fault)
    assert faults == {
        "cooling_fault",
        "vibration_spike",
        "sensor_dropout",
        "data_gap",
        "duplicate_rows",
        "out_of_order_rows",
        "implausible_temperature",
        "implausible_rpm",
        "irregular_sampling",
    }


def test_clean_run_has_no_faults():
    ds = generator.generate(n_machines=3, hours=2, seed=1, inject_faults=False)
    assert ds.ground_truth.empty
    assert ds.data.vibration_mm_s.max() < schema.SENSORS["vibration_mm_s"].warn
    assert not ds.data.duplicated().any()


def test_csv_round_trip_uses_utc_z(synthetic):
    text = generator.to_csv_bytes(synthetic.data.head(3)).decode()
    assert text.splitlines()[1].startswith("2026-10-22T06:00:00.000Z,CNC_01")
