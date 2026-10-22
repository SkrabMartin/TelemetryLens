import pandas as pd
import pytest

from telemetrylens import cycles, generator, ingest, quality


@pytest.fixture(scope="session")
def synthetic():
    return generator.generate(n_machines=5, hours=8, seed=42)


@pytest.fixture(scope="session")
def csv_bytes(synthetic):
    return generator.to_csv_bytes(synthetic.data)


@pytest.fixture(scope="session")
def parsed(csv_bytes):
    return ingest.parse(csv_bytes)


@pytest.fixture(scope="session")
def raw_quality(parsed):
    return quality.quality_report(parsed.data, parsed.sensors)


@pytest.fixture(scope="session")
def cleaned(parsed):
    return quality.clean(parsed.data, parsed.sensors)


@pytest.fixture(scope="session")
def analysed(cleaned):
    df, _ = cycles.add_cycle_columns(cleaned.data)
    return df


def make_csv(rows: list[str], header: str = "timestamp,machine_id,temperature_c,vibration_mm_s,status") -> bytes:
    return ("\n".join([header, *rows]) + "\n").encode()


def frame(values: dict) -> pd.DataFrame:
    return pd.DataFrame(values)
