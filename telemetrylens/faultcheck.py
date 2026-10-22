"""Compare what the analysis found against the generator's ground truth."""

from __future__ import annotations

import pandas as pd

from . import schema


def detection_table(
    ground_truth: pd.DataFrame,
    quality,  # QualityReport on the parsed (uncleaned) data
    issues: list,
    temp_episodes: pd.DataFrame,
    anomalies: pd.DataFrame,
) -> pd.DataFrame:
    """One row per injected fault type: what was planted and what TelemetryLens reports."""
    rows = []
    gt = ground_truth
    by_issue = {i.check: i.rows for i in issues}
    machines = quality.machines.set_index(schema.MACHINE_ID)

    for fault, g in gt.groupby("fault", sort=False):
        mid = g.machine_id.iloc[0]
        planted = int(g.n_samples.sum())
        if fault == "cooling_fault":
            ep = temp_episodes[temp_episodes.machine_id == mid]
            hit = ep[(ep.start <= g.end.iloc[0] + pd.Timedelta("15min")) & (ep.end >= g.start.iloc[0])]
            found, detail = len(hit) > 0, f"{len(hit)} temperature episode(s) above warn"
        elif fault == "vibration_spike":
            a = anomalies[(anomalies.machine_id == mid) & (anomalies.sensor == "vibration_mm_s")]
            matched = a[schema.TIMESTAMP].isin(g.start).sum()
            found, detail = matched == len(g), f"{matched} of {len(g)} spikes flagged"
        elif fault == "sensor_dropout":
            d = quality.dropouts
            hit = d[(d.machine_id == mid) & (d.sensor == g.sensor.iloc[0])]
            found = bool(len(hit)) and int(hit.samples.max()) == planted
            detail = f"{int(hit.samples.max()) if len(hit) else 0} consecutive missing samples"
        elif fault == "data_gap":
            q = quality.gaps[quality.gaps.machine_id == mid]
            found = bool(len(q)) and int(q.missing_samples.max()) == planted
            detail = f"{len(q)} gap(s), {int(q.missing_samples.sum()) if len(q) else 0} missing samples"
        elif fault == "duplicate_rows":
            n = int(machines.loc[mid, "exact_duplicates"]) if mid in machines.index else 0
            found, detail = n == planted, f"{n} exact duplicates"
        elif fault in ("implausible_temperature", "implausible_rpm"):
            n = int(quality.implausible.loc[mid, g.sensor.iloc[0]]) if mid in quality.implausible.index else 0
            found, detail = n == planted, f"{n} implausible values"
        elif fault == "out_of_order_rows":
            n = by_issue.get("out_of_order", 0)
            found, detail = n == planted, f"{n} rows out of order"
        elif fault == "irregular_sampling":
            share = float(machines.loc[mid, "irregular_share"]) if mid in machines.index else 0.0
            found, detail = share > 0, f"{share:.1%} irregular intervals"
        else:
            found, detail = False, "no check defined"
        rows.append(
            dict(
                fault=fault, machine_id=mid, planted=planted, detected=bool(found), finding=detail, note=g.note.iloc[0]
            )
        )
    return pd.DataFrame(rows)
