import pandas as pd
import pytest

from telemetrylens import ingest
from tests.conftest import make_csv


def checks(parsed):
    return {i.check: i for i in parsed.issues}


def test_suggest_mapping_handles_aliases():
    m = ingest.suggest_mapping(["Time", "Device ID", "Temp", "Vib RMS", "Spindle Speed", "State"])
    assert m["timestamp"] == "Time"
    assert m["machine_id"] == "Device ID"
    assert m["temperature_c"] == "Temp"
    assert m["vibration_mm_s"] == "Vib RMS"
    assert m["rpm"] == "Spindle Speed"
    assert m["status"] == "State"
    assert m["power_kw"] is None


def test_badly_formatted_timestamps_are_reported_and_excluded():
    p = ingest.parse(
        make_csv(
            [
                "2026-10-22T10:00:00Z,M1,40,1.0,running",
                "not a time,M1,41,1.0,running",
                "2026-10-22T10:00:02Z,M1,42,1.0,running",
            ]
        )
    )
    assert p.ok
    assert checks(p)["bad_timestamp"].rows == 1
    assert checks(p)["rows_excluded"].rows == 1
    assert len(p.data) == 2


def test_reordered_rows_are_flagged_and_sorted():
    p = ingest.parse(
        make_csv(
            [
                "2026-10-22T10:00:02Z,M1,42,1.0,running",
                "2026-10-22T10:00:00Z,M1,40,1.0,running",
                "2026-10-22T10:00:01Z,M1,41,1.0,running",
            ]
        )
    )
    assert checks(p)["out_of_order"].rows == 1
    assert p.data.temperature_c.tolist() == [40, 41, 42]
    assert p.data.source_row.tolist() == [3, 4, 2]


def test_naive_timestamps_use_the_chosen_timezone():
    content = make_csv(["2026-07-01 12:00:00,M1,40,1.0,running"])
    cfg = ingest.ImportConfig(
        mapping=ingest.suggest_mapping(ingest.read_csv(content).columns.tolist()), source_timezone="Europe/Prague"
    )
    p = ingest.parse(content, cfg)
    assert p.data.timestamp.iloc[0] == pd.Timestamp("2026-07-01T10:00:00Z")
    assert checks(p)["naive_timestamps"].rows == 1


def test_mixed_offsets_are_normalised_to_utc():
    p = ingest.parse(
        make_csv(
            [
                "2026-10-22T12:00:00+02:00,M1,40,1.0,running",
                "2026-10-22T10:00:01Z,M1,41,1.0,running",
            ]
        )
    )
    assert p.data.timestamp.tolist() == [pd.Timestamp("2026-10-22T10:00:00Z"), pd.Timestamp("2026-10-22T10:00:01Z")]


def test_machine_ids_with_whitespace_are_merged():
    p = ingest.parse(
        make_csv(
            [
                "2026-10-22T10:00:00Z,M1,40,1.0,running",
                "2026-10-22T10:00:01Z, M1 ,41,1.0,running",
            ]
        )
    )
    assert p.data.machine_id.unique().tolist() == ["M1"]
    assert "machine_id_whitespace" in checks(p)


def test_fahrenheit_is_converted():
    content = make_csv(["2026-10-22T10:00:00Z,M1,212,1.0,running"])
    cfg = ingest.ImportConfig(
        mapping=ingest.suggest_mapping(ingest.read_csv(content).columns.tolist()), units={"temperature_c": "°F"}
    )
    p = ingest.parse(content, cfg)
    assert p.data.temperature_c.iloc[0] == pytest.approx(100.0)
    assert "unit_conversion" in checks(p)


def test_status_aliases_and_unknown_values():
    p = ingest.parse(
        make_csv(
            [
                "2026-10-22T10:00:00Z,M1,40,1.0,RUN",
                "2026-10-22T10:00:01Z,M1,40,1.0,Alarm",
                "2026-10-22T10:00:02Z,M1,40,1.0,off",
                "2026-10-22T10:00:03Z,M1,40,1.0,maintenance",
            ]
        )
    )
    assert p.data.status.tolist() == ["running", "warning", "stopped", "unknown"]
    assert checks(p)["unknown_status"].rows == 1


def test_non_numeric_values_become_missing_and_comma_decimals_parse():
    p = ingest.parse(
        make_csv(
            [
                "2026-10-22T10:00:00Z,M1,ERR,1.0,running",
                '2026-10-22T10:00:01Z,M1,"40,5",1.0,running',
                "2026-10-22T10:00:02Z,M1,,1.0,running",
            ]
        )
    )
    assert p.data.temperature_c.isna().tolist() == [True, False, True]
    assert p.data.temperature_c.iloc[1] == 40.5
    assert checks(p)["non_numeric"].rows == 1


def test_file_without_usable_signal_is_an_error():
    p = ingest.parse(
        make_csv(
            [
                "2026-10-22T10:00:00Z,M1,,,running",
                "2026-10-22T10:00:01Z,M1,n/a,-,running",
            ]
        )
    )
    assert not p.ok
    assert "no_signal" in checks(p)


def test_missing_required_column_is_an_error():
    p = ingest.parse(make_csv(["40,1.0"], header="temperature_c,vibration_mm_s"))
    assert not p.ok
    assert checks(p)["missing_column"].severity == "error"


def test_empty_file_is_an_error():
    p = ingest.parse(b"timestamp,machine_id,temperature_c\n")
    assert not p.ok


def test_import_id_depends_on_content_and_settings(csv_bytes):
    cfg = ingest.ImportConfig(mapping=ingest.suggest_mapping(ingest.read_csv(csv_bytes).columns.tolist()))
    a = ingest.import_id(csv_bytes, cfg)
    assert a == ingest.import_id(csv_bytes, cfg)
    assert a != ingest.import_id(csv_bytes + b"\n", cfg)
    cfg.source_timezone = "Europe/Prague"
    assert a != ingest.import_id(csv_bytes, cfg)


def test_raw_data_is_preserved(parsed, csv_bytes):
    assert len(parsed.raw) == len(ingest.read_csv(csv_bytes))
    assert parsed.raw.iloc[0, 0] == "2026-10-22T06:00:00.000Z"
