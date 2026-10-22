"""CSV import, column mapping and validation.

The imported file is kept untouched (``raw``). Parsing produces a separate,
canonical frame plus a list of validation issues; nothing is dropped silently.
"""

from __future__ import annotations

import hashlib
import io
import json
import re
from dataclasses import dataclass, field

import pandas as pd

from . import schema

_TZ_SUFFIX = re.compile(r"(?:Z|[+-]\d{2}:?\d{2})$", re.IGNORECASE)


@dataclass
class Issue:
    severity: str  # "error" | "warning" | "info"
    check: str
    message: str
    rows: int = 0


@dataclass
class ImportConfig:
    """User decisions about how to interpret a file. Part of the import identity."""

    mapping: dict[str, str | None]
    source_timezone: str = "UTC"
    units: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"mapping": self.mapping, "source_timezone": self.source_timezone, "units": self.units}


@dataclass
class ParsedData:
    raw: pd.DataFrame
    data: pd.DataFrame
    issues: list[Issue]
    config: ImportConfig
    import_id: str
    sensors: list[str]

    @property
    def ok(self) -> bool:
        return not any(i.severity == "error" for i in self.issues)


def read_csv(source: bytes | str) -> pd.DataFrame:
    """Read a CSV exactly as stored: every column as text, no type guessing."""
    if isinstance(source, bytes):
        source = io.BytesIO(source)
    return pd.read_csv(source, dtype=str, keep_default_na=False, na_values=[], skipinitialspace=True)


def _norm(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.strip().lower()).strip("_")


def suggest_mapping(columns: list[str]) -> dict[str, str | None]:
    """Guess which CSV column holds each canonical field, by name."""
    normalised = {_norm(c): c for c in columns}
    mapping: dict[str, str | None] = {}
    used: set[str] = set()
    for canonical, aliases in schema.COLUMN_ALIASES.items():
        mapping[canonical] = None
        for alias in aliases:
            col = normalised.get(alias)
            if col is not None and col not in used:
                mapping[canonical] = col
                used.add(col)
                break
    return mapping


def import_id(content: bytes, config: ImportConfig) -> str:
    h = hashlib.sha256(content)
    h.update(json.dumps(config.to_dict(), sort_keys=True).encode())
    return h.hexdigest()[:12]


_MISSING_TOKENS = {"", "na", "n/a", "nan", "null", "none", "-", "--"}


def _is_blank(s: pd.Series) -> pd.Series:
    return s.str.strip().str.lower().isin(_MISSING_TOKENS)


def parse_timestamps(values: pd.Series, source_timezone: str = "UTC") -> tuple[pd.Series, int]:
    """Parse to UTC. Values without an offset are interpreted in ``source_timezone``.

    Returns the parsed series and how many naive (offset-less) values were seen.
    """
    text = values.astype(str).str.strip()
    aware = text.str.contains(_TZ_SUFFIX)
    out = pd.Series(pd.NaT, index=values.index, dtype="datetime64[ns, UTC]")
    if aware.any():
        out[aware] = pd.to_datetime(text[aware], utc=True, errors="coerce", format="ISO8601")
        bad = aware & out.isna()
        if bad.any():
            out[bad] = pd.to_datetime(text[bad], utc=True, errors="coerce", format="mixed")
    naive = ~aware
    if naive.any():
        parsed = pd.to_datetime(text[naive], errors="coerce", format="ISO8601")
        bad = parsed.isna()
        if bad.any():
            parsed[bad] = pd.to_datetime(text[naive][bad], errors="coerce", format="mixed")
        localized = parsed.dt.tz_localize(source_timezone, ambiguous="NaT", nonexistent="NaT")
        out[naive] = localized.dt.tz_convert("UTC")
    return out, int(naive.sum())


def parse(content: bytes, config: ImportConfig | None = None) -> ParsedData:
    raw = read_csv(content)
    if config is None:
        config = ImportConfig(mapping=suggest_mapping(list(raw.columns)))
    issues: list[Issue] = []
    m = config.mapping
    sensors = [s for s in schema.SENSOR_NAMES if m.get(s)]
    iid = import_id(content, config)

    def fail(check: str, message: str) -> ParsedData:
        issues.append(Issue("error", check, message))
        return ParsedData(raw, pd.DataFrame(), issues, config, iid, sensors)

    if raw.empty:
        return fail("empty_file", "The file contains no data rows.")
    for req in (schema.TIMESTAMP, schema.MACHINE_ID):
        if not m.get(req) or m[req] not in raw.columns:
            return fail("missing_column", f"No column is mapped to required field '{req}'.")
    if not sensors:
        return fail("no_sensors", "No sensor column is mapped, so there is nothing to analyse.")
    mapped = [c for c in m.values() if c]
    if len(mapped) != len(set(mapped)):
        return fail("mapping_conflict", "The same CSV column is mapped to more than one field.")

    df = pd.DataFrame(index=raw.index)
    df["source_row"] = raw.index + 2  # 1-based line number in the file, after the header

    ts, n_naive = parse_timestamps(raw[m[schema.TIMESTAMP]], config.source_timezone)
    df[schema.TIMESTAMP] = ts
    if n_naive:
        issues.append(
            Issue(
                "info",
                "naive_timestamps",
                f"{n_naive:,} timestamps had no timezone offset and were read as {config.source_timezone}.",
                n_naive,
            )
        )
    bad_ts = int(ts.isna().sum())
    if bad_ts:
        issues.append(Issue("warning", "bad_timestamp", f"{bad_ts:,} rows have an unreadable timestamp.", bad_ts))

    mid = raw[m[schema.MACHINE_ID]].str.strip()
    df[schema.MACHINE_ID] = mid.mask(_is_blank(mid))
    bad_mid = int(df[schema.MACHINE_ID].isna().sum())
    if bad_mid:
        issues.append(Issue("warning", "missing_machine_id", f"{bad_mid:,} rows have no machine identifier.", bad_mid))
    stripped_variants = raw[m[schema.MACHINE_ID]].nunique() - mid.nunique()
    if stripped_variants > 0:
        issues.append(
            Issue(
                "info",
                "machine_id_whitespace",
                f"{stripped_variants} machine identifiers differed only by surrounding spaces and were merged.",
            )
        )

    if m.get(schema.CYCLE_ID):
        c = raw[m[schema.CYCLE_ID]]
        df[schema.CYCLE_ID] = pd.to_numeric(c.mask(_is_blank(c)), errors="coerce")
        if df[schema.CYCLE_ID].dropna().mod(1).ne(0).any():
            # Non-integer cycle ids: keep as text labels instead.
            df[schema.CYCLE_ID] = c.mask(_is_blank(c))
        else:
            df[schema.CYCLE_ID] = df[schema.CYCLE_ID].astype("Int64")

    if m.get(schema.STATUS):
        s = raw[m[schema.STATUS]].str.strip().str.lower()
        norm = s.map(schema.STATUS_ALIASES)
        blank = _is_blank(s)
        unknown = norm.isna() & ~blank
        df[schema.STATUS] = norm.where(~unknown, schema.UNKNOWN_STATUS).where(~blank, None)
        if unknown.any():
            values = ", ".join(sorted(s[unknown].unique())[:5])
            issues.append(
                Issue(
                    "warning",
                    "unknown_status",
                    f"{int(unknown.sum()):,} status values were not recognised ({values}) and are labelled 'unknown'.",
                    int(unknown.sum()),
                )
            )

    for sensor in sensors:
        col = raw[m[sensor]]
        blank = _is_blank(col)
        values = pd.to_numeric(col.str.replace(",", ".", regex=False).where(~blank), errors="coerce")
        non_numeric = int((values.isna() & ~blank).sum())
        if non_numeric:
            issues.append(
                Issue(
                    "warning",
                    "non_numeric",
                    f"{schema.label(sensor)}: {non_numeric:,} values are not numbers and were treated as missing.",
                    non_numeric,
                )
            )
        unit = config.units.get(sensor)
        if unit and unit in schema.UNIT_CONVERSIONS.get(sensor, {}) and unit != schema.SENSORS[sensor].unit:
            scale, offset = schema.UNIT_CONVERSIONS[sensor][unit]
            values = values * scale + offset
            issues.append(
                Issue(
                    "info",
                    "unit_conversion",
                    f"{schema.SENSORS[sensor].label} converted from {unit} to {schema.SENSORS[sensor].unit}.",
                )
            )
        df[sensor] = values.astype(float)

    if df[sensors].isna().all().all():
        return fail("no_signal", "None of the mapped sensor columns contain a usable numeric value.")

    unusable = df[schema.TIMESTAMP].isna() | df[schema.MACHINE_ID].isna()
    if unusable.all():
        return fail("no_usable_rows", "No row has both a readable timestamp and a machine identifier.")

    # Sort order is checked per machine in file order, before any reordering.
    usable = df[~unusable]
    backwards = usable.groupby(schema.MACHINE_ID, sort=False)[schema.TIMESTAMP].diff() < pd.Timedelta(0)
    n_back = int(backwards.sum())
    if n_back:
        issues.append(
            Issue(
                "warning",
                "out_of_order",
                f"{n_back:,} rows go back in time relative to the previous row of the same machine. "
                "Rows were sorted by machine and timestamp.",
                n_back,
            )
        )

    df = df[~unusable].sort_values([schema.MACHINE_ID, schema.TIMESTAMP, "source_row"], kind="stable")
    df = df.reset_index(drop=True)
    if unusable.any():
        issues.append(
            Issue(
                "warning",
                "rows_excluded",
                f"{int(unusable.sum()):,} rows were excluded because timestamp or machine id is missing.",
                int(unusable.sum()),
            )
        )
    df[schema.MACHINE_ID] = df[schema.MACHINE_ID].astype(str)
    return ParsedData(raw, df, issues, config, iid, sensors)
