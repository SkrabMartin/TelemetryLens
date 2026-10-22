"""Deterministic data-quality checks and an explicit, logged cleaning step."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import schema

GAP_FACTOR = 5.0  # an interval longer than 5x the median interval is a gap
MIN_GAP_S = 10.0
IRREGULAR_TOLERANCE = 0.5  # +-50% around the median interval
MIN_DROPOUT_S = 30.0  # missing runs shorter than this are counted, not listed


@dataclass
class QualityReport:
    machines: pd.DataFrame  # one row per machine
    missing: pd.DataFrame  # machine x sensor: missing share
    implausible: pd.DataFrame  # machine x sensor: count outside plausible range
    gaps: pd.DataFrame  # time gaps with no rows
    dropouts: pd.DataFrame  # long runs where one sensor is missing
    intervals: pd.DataFrame  # sampled interval values for the histogram

    @property
    def totals(self) -> dict[str, float]:
        m = self.machines
        return {
            "rows": int(m.rows.sum()),
            "machines": len(m),
            "exact_duplicates": int(m.exact_duplicates.sum()),
            "conflicting_timestamps": int(m.conflicting_timestamps.sum()),
            "gaps": len(self.gaps),
            "gap_time_s": float(self.gaps.duration_s.sum()) if len(self.gaps) else 0.0,
            "implausible": int(self.implausible.to_numpy().sum()),
            "missing_cells": int(m.missing_cells.sum()),
            "irregular_intervals": int(m.irregular_intervals.sum()),
            "dropouts": len(self.dropouts),
        }


def _value_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c != "source_row"]


def median_intervals(df: pd.DataFrame) -> pd.Series:
    dt = df.groupby(schema.MACHINE_ID)[schema.TIMESTAMP].diff().dt.total_seconds()
    return dt[dt > 0].groupby(df[schema.MACHINE_ID]).median()


def gap_thresholds(df: pd.DataFrame) -> pd.Series:
    return (median_intervals(df) * GAP_FACTOR).clip(lower=MIN_GAP_S)


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Start/end (inclusive) positions of consecutive True runs."""
    if not mask.any():
        return []
    d = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1) - 1, strict=True))


def quality_report(df: pd.DataFrame, sensors: list[str]) -> QualityReport:
    rows, gaps, dropouts, interval_samples = [], [], [], []
    missing, implausible = {}, {}
    value_cols = _value_columns(df)

    for mid, g in df.groupby(schema.MACHINE_ID, sort=True):
        ts = g[schema.TIMESTAMP]
        dt = ts.diff().dt.total_seconds().to_numpy()[1:]
        positive = dt[dt > 0]
        med = float(np.median(positive)) if len(positive) else np.nan
        gap_thr = max(med * GAP_FACTOR, MIN_GAP_S) if np.isfinite(med) else np.inf
        is_gap = dt > gap_thr
        regular = positive[positive <= gap_thr]
        irregular = int((np.abs(regular - med) > IRREGULAR_TOLERANCE * med).sum()) if np.isfinite(med) else 0

        exact_dup = int(g.duplicated(subset=value_cols).sum())
        dedup = g.drop_duplicates(subset=value_cols)
        same_ts = dedup.duplicated(subset=[schema.TIMESTAMP], keep=False)
        conflicting = int(dedup[same_ts][schema.TIMESTAMP].nunique())

        span_s = (ts.iloc[-1] - ts.iloc[0]).total_seconds() if len(g) > 1 else 0.0
        expected = span_s / med + 1 if np.isfinite(med) and med > 0 else len(g)
        unique_ts = ts.nunique()
        coverage = min(unique_ts / expected, 1.0) if expected else 1.0

        miss = g[sensors].isna()
        missing[mid] = miss.mean()
        lo = pd.Series({s: schema.SENSORS[s].plausible_min for s in sensors})
        hi = pd.Series({s: schema.SENSORS[s].plausible_max for s in sensors})
        vals = g[sensors]
        bad = (vals.lt(lo) | vals.gt(hi)) & vals.notna()
        implausible[mid] = bad.sum()
        valid_share = float(1 - (miss | bad).to_numpy().mean())

        rows.append(
            dict(
                machine_id=mid,
                rows=len(g),
                first=ts.iloc[0],
                last=ts.iloc[-1],
                span_s=span_s,
                median_interval_s=med,
                expected_samples=int(round(expected)),
                coverage=coverage,
                missing_sample_rate=1 - coverage,
                gaps=int(is_gap.sum()),
                irregular_intervals=irregular,
                irregular_share=irregular / max(len(regular), 1),
                exact_duplicates=exact_dup,
                conflicting_timestamps=conflicting,
                missing_cells=int(miss.to_numpy().sum()),
                implausible_values=int(bad.to_numpy().sum()),
                valid_value_share=valid_share,
            )
        )

        for i in np.flatnonzero(is_gap):
            start, end = ts.iloc[i], ts.iloc[i + 1]
            gaps.append(
                dict(
                    machine_id=mid,
                    gap_start=start,
                    gap_end=end,
                    duration_s=dt[i],
                    missing_samples=int(round(dt[i] / med)) - 1,
                )
            )

        tsv = ts.to_numpy()
        for s in sensors:
            for a, b in _runs(miss[s].to_numpy()):
                end = tsv[b + 1] if b + 1 < len(tsv) else tsv[b]
                start = tsv[a - 1] if a > 0 else tsv[a]
                dur = (end - start) / np.timedelta64(1, "s")
                if dur >= MIN_DROPOUT_S:
                    dropouts.append(
                        dict(
                            machine_id=mid,
                            sensor=s,
                            first_missing=ts.iloc[a],
                            last_missing=ts.iloc[b],
                            samples=int(b - a + 1),
                            duration_s=float(dur),
                        )
                    )

        sample = positive if len(positive) <= 20000 else positive[:: len(positive) // 20000 + 1]
        interval_samples.append(pd.DataFrame({"machine_id": mid, "interval_s": sample}))

    machines = pd.DataFrame(rows)
    return QualityReport(
        machines=machines,
        missing=pd.DataFrame(missing).T.reindex(columns=sensors),
        implausible=pd.DataFrame(implausible).T.reindex(columns=sensors).fillna(0).astype(int),
        gaps=pd.DataFrame(gaps, columns=["machine_id", "gap_start", "gap_end", "duration_s", "missing_samples"]),
        dropouts=pd.DataFrame(
            dropouts, columns=["machine_id", "sensor", "first_missing", "last_missing", "samples", "duration_s"]
        ),
        intervals=pd.concat(interval_samples, ignore_index=True) if interval_samples else pd.DataFrame(),
    )


@dataclass
class CleaningOptions:
    drop_exact_duplicates: bool = True
    # "keep_first" | "mean" | "keep_all" for rows sharing machine+timestamp with different values
    conflicting_timestamps: str = "keep_first"
    mask_implausible: bool = True

    def to_dict(self) -> dict:
        return dict(self.__dict__)


@dataclass
class CleanResult:
    data: pd.DataFrame
    decisions: list[str] = field(default_factory=list)
    rows_before: int = 0
    rows_after: int = 0
    masked: dict[str, int] = field(default_factory=dict)


def clean(df: pd.DataFrame, sensors: list[str], options: CleaningOptions | None = None) -> CleanResult:
    """Apply cleaning decisions and log each one. Never interpolates missing data."""
    options = options or CleaningOptions()
    out = df.copy()
    decisions: list[str] = []
    n0 = len(out)

    if options.drop_exact_duplicates:
        dup = out.duplicated(subset=_value_columns(out))
        out = out[~dup]
        decisions.append(f"Removed {int(dup.sum()):,} exact duplicate rows (kept the first occurrence).")
    else:
        decisions.append("Kept exact duplicate rows.")

    key = [schema.MACHINE_ID, schema.TIMESTAMP]
    conflict = out.duplicated(subset=key, keep=False)
    n_conflict = int(out[conflict].groupby(key).ngroups) if conflict.any() else 0
    if n_conflict and options.conflicting_timestamps == "keep_first":
        out = out[~out.duplicated(subset=key, keep="first")]
        decisions.append(f"{n_conflict:,} timestamps had conflicting rows; kept the first row in file order.")
    elif n_conflict and options.conflicting_timestamps == "mean":
        first = out.groupby(key, sort=False).first()
        first[sensors] = out.groupby(key, sort=False)[sensors].mean()
        out = first.reset_index()[out.columns]
        decisions.append(f"{n_conflict:,} timestamps had conflicting rows; sensor values were averaged.")
    elif n_conflict:
        decisions.append(f"{n_conflict:,} timestamps have conflicting rows; all were kept.")

    masked: dict[str, int] = {}
    if options.mask_implausible:
        for s in sensors:
            spec = schema.SENSORS[s]
            bad = (out[s] < spec.plausible_min) | (out[s] > spec.plausible_max)
            masked[s] = int(bad.sum())
            if masked[s]:
                out.loc[bad, s] = np.nan
                decisions.append(
                    f"{spec.label}: {masked[s]:,} values outside the plausible range "
                    f"{spec.plausible_min:g} to {spec.plausible_max:g} {spec.unit} were set to missing."
                )
    else:
        decisions.append("Implausible values were kept as recorded.")

    decisions.append("Missing readings and gaps were not interpolated; statistics use observed samples only.")
    out = out.sort_values(key, kind="stable").reset_index(drop=True)
    return CleanResult(out, decisions, n0, len(out), masked)


def sample_durations(df: pd.DataFrame) -> pd.Series:
    """Seconds each sample represents (time to the next sample), with gaps counted as zero.

    Used for time-weighted shares, so a logger outage is not credited to the last state seen.
    """
    nxt = df.groupby(schema.MACHINE_ID)[schema.TIMESTAMP].shift(-1)
    dt = (nxt - df[schema.TIMESTAMP]).dt.total_seconds()
    thr = df[schema.MACHINE_ID].map(gap_thresholds(df))
    med = df[schema.MACHINE_ID].map(median_intervals(df))
    dt = dt.fillna(med)
    return dt.where(dt <= thr, 0.0).fillna(0.0)
