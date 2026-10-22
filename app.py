"""TelemetryLens: explainable analysis of industrial machine telemetry.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import io
import json
from dataclasses import asdict
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st

from telemetrylens import charts, cycles, faultcheck, generator, ingest, insights, quality, report, schema

st.set_page_config(page_title="TelemetryLens", page_icon="📈", layout="wide")


# ---------------------------------------------------------------- cached steps
@st.cache_data(show_spinner="Generating synthetic telemetry…", max_entries=4)
def demo_data(n_machines: int, hours: float, seed: int) -> tuple[bytes, pd.DataFrame]:
    ds = generator.generate(n_machines=n_machines, hours=hours, seed=seed)
    return generator.to_csv_bytes(ds.data), ds.ground_truth


@st.cache_data(show_spinner="Reading CSV header…", max_entries=4)
def csv_columns(content: bytes) -> list[str]:
    return list(pd.read_csv(io.BytesIO(content), nrows=5, dtype=str).columns)


@st.cache_data(show_spinner="Parsing and validating…", max_entries=4)
def parse(content: bytes, config_json: str) -> ingest.ParsedData:
    cfg = json.loads(config_json)
    return ingest.parse(content, ingest.ImportConfig(**cfg))


@st.cache_data(show_spinner="Checking data quality and cleaning…", max_entries=4)
def prepare(key: str, _parsed: ingest.ParsedData, options_json: str):
    options = quality.CleaningOptions(**json.loads(options_json))
    raw_quality = quality.quality_report(_parsed.data, _parsed.sensors)
    cleaned = quality.clean(_parsed.data, _parsed.sensors, options)
    data, cycle_source = cycles.add_cycle_columns(cleaned.data)
    clean_quality = quality.quality_report(cleaned.data, _parsed.sensors)
    return raw_quality, cleaned, data, cycle_source, clean_quality


@st.cache_data(show_spinner="Computing insights…", max_entries=8)
def analyse(
    key: str,
    _df: pd.DataFrame,
    sensors: tuple[str, ...],
    thresholds_json: str,
    z_limit: float,
    group_by: str,
    direction: str,
):
    sensors = list(sensors)
    thresholds = json.loads(thresholds_json)
    return dict(
        stats=insights.operating_stats(_df, sensors),
        thresholds=insights.threshold_summary(_df, sensors, thresholds),
        anomalies=insights.flag_anomalies(_df, sensors, z_limit, group_by, direction),
        states=insights.state_summary(_df, sensors),
        hours=cycles.time_in_state(_df),
        utilisation=cycles.utilisation(_df),
        cycles=cycles.cycle_table(_df, sensors),
        episodes=pd.concat(
            [
                insights.violation_episodes(_df, s, lim["warn"] if lim.get("warn") is not None else lim["critical"])
                for s, lim in thresholds.items()
                if s in sensors and (lim.get("warn") is not None or lim.get("critical") is not None)
            ]
            or [pd.DataFrame()],
            ignore_index=True,
        ),
    )


def to_py(ts: pd.Timestamp) -> datetime:
    return ts.tz_convert("UTC").to_pydatetime().replace(tzinfo=None)


def to_ts(dt: datetime) -> pd.Timestamp:
    return pd.Timestamp(dt).tz_localize("UTC")


def fmt_duration(seconds: float) -> str:
    if not np.isfinite(seconds):
        return "–"
    seconds = int(round(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h} h {m:02d} min" if h else (f"{m} min {s:02d} s" if m else f"{s} s")


def show_table(df: pd.DataFrame, **kwargs):
    if df is None or df.empty:
        st.caption("Nothing to show for the current selection.")
        return
    st.dataframe(df, hide_index=True, width="stretch", **kwargs)


def pct(col: str, label: str | None = None):
    return st.column_config.NumberColumn(label or col, format="percent")


# --------------------------------------------------------------------- sidebar
st.sidebar.title("TelemetryLens")
st.sidebar.caption("Explainable telemetry analysis. Statistics and rules only, no predictions.")

source = st.sidebar.radio("Data source", ["Synthetic demo data", "Upload CSV"], horizontal=True)
ground_truth = None
if source == "Synthetic demo data":
    c1, c2 = st.sidebar.columns(2)
    n_machines = c1.slider("Machines", 1, 5, 5)
    hours = c2.slider("Hours", 1, 24, 8)
    seed = st.sidebar.number_input("Random seed", 0, 10_000, 42, step=1)
    content, ground_truth = demo_data(n_machines, float(hours), int(seed))
    source_name = f"synthetic_{n_machines}m_{hours}h_seed{seed}.csv"
    st.sidebar.download_button("Download demo CSV", content, file_name=source_name, mime="text/csv")
else:
    upload = st.sidebar.file_uploader("Telemetry CSV", type=["csv"])
    if upload is None:
        st.title("TelemetryLens")
        st.info(
            "Upload a CSV with one row per measurement: a timestamp, a machine id and one or more sensor "
            "columns (temperature, vibration, rpm, current, power, torque) plus an optional status and cycle id. "
            "Or switch to **Synthetic demo data** in the sidebar."
        )
        st.code(
            "timestamp,machine_id,cycle_id,temperature_c,vibration_mm_s,rpm,current_a,power_kw,status\n"
            "2026-10-22T10:00:00Z,CNC_01,101,45.2,1.8,1500,8.2,3.5,running",
            language="text",
        )
        st.stop()
    content = upload.getvalue()
    source_name = upload.name

columns = csv_columns(content)
suggested = ingest.suggest_mapping(columns)
with st.sidebar.expander("Import settings", expanded=source == "Upload CSV"):
    st.caption("Confirm which column holds each field. Unmapped sensors are ignored.")
    options = ["—"] + columns
    mapping: dict[str, str | None] = {}
    for field in [schema.TIMESTAMP, schema.MACHINE_ID, schema.CYCLE_ID, schema.STATUS, *schema.SENSOR_NAMES]:
        default = suggested.get(field)
        name = schema.label(field) if field in schema.SENSORS else field
        choice = st.selectbox(
            name, options, index=options.index(default) if default in options else 0, key=f"map_{field}"
        )
        mapping[field] = None if choice == "—" else choice
    tz = st.text_input(
        "Timezone for timestamps without offset",
        "UTC",
        help="IANA name such as Europe/Prague. Timestamps with Z or +hh:mm keep their offset.",
    )
    units = {}
    for sensor, conv in schema.UNIT_CONVERSIONS.items():
        if mapping.get(sensor) and len(conv) > 1:
            units[sensor] = st.selectbox(
                f"Unit of {schema.SENSORS[sensor].label} in file", list(conv), key=f"unit_{sensor}"
            )

config = ingest.ImportConfig(
    mapping=mapping,
    source_timezone=tz.strip() or "UTC",
    units={k: v for k, v in units.items() if v != schema.SENSORS[k].unit},
)
try:
    parsed = parse(content, json.dumps(config.to_dict(), sort_keys=True))
except Exception as exc:  # e.g. an unknown timezone name
    st.error(f"Could not import the file: {exc}")
    st.stop()

if not parsed.ok:
    st.title("TelemetryLens")
    st.error("The file cannot be analysed yet.")
    for i in parsed.issues:
        (st.error if i.severity == "error" else st.warning)(i.message)
    st.stop()

with st.sidebar.expander("Cleaning", expanded=False):
    opts = quality.CleaningOptions(
        drop_exact_duplicates=st.checkbox("Remove exact duplicate rows", True),
        conflicting_timestamps=st.selectbox(
            "Rows sharing a timestamp with different values",
            ["keep_first", "mean", "keep_all"],
            format_func={"keep_first": "Keep first in file", "mean": "Average values", "keep_all": "Keep all"}.get,
        ),
        mask_implausible=st.checkbox("Set implausible values to missing", True),
    )
    st.caption("Missing data and gaps are never interpolated.")

options_json = json.dumps(opts.to_dict(), sort_keys=True)
raw_q, cleaned, full, cycle_source, clean_q = prepare(parsed.import_id + options_json, parsed, options_json)
sensors = parsed.sensors
all_machines = sorted(full[schema.MACHINE_ID].unique())
colors = charts.machine_colors(all_machines)

st.sidebar.subheader("Filters")
st.sidebar.caption("Applied to every view and to the exported report.")
machines = st.sidebar.multiselect("Machines", all_machines, default=all_machines)
t_min, t_max = full[schema.TIMESTAMP].min(), full[schema.TIMESTAMP].max()
if t_max > t_min:
    window = st.sidebar.slider(
        "Time window (UTC)", to_py(t_min), to_py(t_max), (to_py(t_min), to_py(t_max)), format="MM-DD HH:mm"
    )
    w_start, w_end = to_ts(window[0]), to_ts(window[1])
else:
    w_start, w_end = t_min, t_max
statuses_all = sorted(full[schema.STATUS].dropna().unique()) if schema.STATUS in full else []
statuses = st.sidebar.multiselect("Machine status", statuses_all, default=statuses_all) if statuses_all else []

with st.sidebar.expander("Thresholds", expanded=False):
    thresholds: dict[str, dict[str, float | None]] = {}
    for sensor in sensors:
        spec = schema.SENSORS[sensor]
        c1, c2 = st.columns(2)
        warn = c1.number_input(f"{spec.label} warn", value=spec.warn, key=f"w_{sensor}", format="%g")
        crit = c2.number_input(f"{spec.label} critical", value=spec.critical, key=f"c_{sensor}", format="%g")
        thresholds[sensor] = {"warn": warn, "critical": crit}
    st.caption("Vibration defaults follow ISO 10816-style zones for medium machines. Leave empty for none.")

with st.sidebar.expander("Anomaly flags", expanded=False):
    z_limit = st.slider("Robust z-score limit", 3.0, 12.0, 6.0, 0.5)
    group_by = st.selectbox(
        "Compare each sample with",
        ["phase", "status"],
        format_func={"phase": "same machine and operating phase", "status": "same machine and status"}.get,
    )
    direction = st.selectbox(
        "Flag", ["high", "both"], format_func={"high": "high values only", "both": "high and low values"}.get
    )

mask = full[schema.MACHINE_ID].isin(machines) & full[schema.TIMESTAMP].between(w_start, w_end)
if statuses_all:
    mask &= full[schema.STATUS].isin(statuses) | (full[schema.STATUS].isna() & (len(statuses) == len(statuses_all)))
df = full[mask]
filters = dict(
    machines=machines, start=w_start, end=w_end, statuses=statuses if len(statuses) != len(statuses_all) else []
)
filter_key = json.dumps(dict(filters, start=str(w_start), end=str(w_end)), sort_keys=True)
thresholds_json = json.dumps(thresholds, sort_keys=True)
analysis_key = "|".join(
    [parsed.import_id, options_json, filter_key, thresholds_json, str(z_limit), group_by, direction]
)

st.title("TelemetryLens")
st.caption(
    f"`{source_name}` · import `{parsed.import_id}` · {len(full):,} rows after cleaning · "
    f"{len(all_machines)} machines · {t_min:%Y-%m-%d %H:%M} to {t_max:%Y-%m-%d %H:%M} UTC"
)
if df.empty:
    st.warning("No data matches the current filters.")
    st.stop()

A = analyse(analysis_key, df, tuple(sensors), thresholds_json, z_limit, group_by, direction)
totals = raw_q.totals
issues = parsed.issues

tab_ins, tab_q, tab_cyc, tab_cmp, tab_rep, tab_dict = st.tabs(
    ["Insights", "Data quality", "Cycles", "Compare windows", "Report", "Data dictionary"]
)

# --------------------------------------------------------------------- insights
with tab_ins:
    valid = float((raw_q.machines.valid_value_share * raw_q.machines.rows).sum() / raw_q.machines.rows.sum())
    problems = []
    if totals["gaps"]:
        problems.append(f"{totals['gaps']} data gap(s) totalling {fmt_duration(totals['gap_time_s'])}")
    if totals["dropouts"]:
        problems.append(f"{totals['dropouts']} sensor dropout(s)")
    if totals["exact_duplicates"]:
        problems.append(f"{totals['exact_duplicates']} duplicate rows")
    if totals["implausible"]:
        problems.append(f"{totals['implausible']} implausible values")
    if totals["irregular_intervals"]:
        problems.append(f"{totals['irregular_intervals']:,} irregular sampling intervals")
    n_warn_issues = sum(i.severity == "warning" for i in issues)
    msg = f"**Data quality:** {valid:.1%} of sensor values are present and plausible."
    if problems or n_warn_issues:
        st.warning(
            msg
            + " Found "
            + ", ".join(problems or [f"{n_warn_issues} validation warnings"])
            + ". See the Data quality tab before relying on these numbers."
        )
    else:
        st.success(msg + " No gaps, duplicates or implausible values were found.")

    th, an = A["thresholds"], A["anomalies"]
    util = A["utilisation"]
    k = st.columns(6)
    k[0].metric("Machines", len(machines))
    k[1].metric("Samples", f"{len(df):,}")
    k[2].metric("Time window", fmt_duration((w_end - w_start).total_seconds()))
    k[3].metric(
        "Active share",
        f"{util.active_h.sum() / util.observed_h.sum():.0%}" if len(util) else "–",
        help="Running or warning, as a share of observed time (gaps excluded).",
    )
    k[4].metric(
        "Threshold episodes",
        int(th.episodes.sum()) if len(th) else 0,
        help="Consecutive runs at or above the warn limit.",
    )
    k[5].metric("Anomaly flags", f"{len(an):,}", help=f"Samples with robust z ≥ {z_limit:g}.")

    st.subheader("Sensor trends")
    c1, c2 = st.columns([3, 1])
    default_sensors = [s for s in ["temperature_c", "vibration_mm_s", "rpm", "power_kw"] if s in sensors]
    plot_sensors = c1.multiselect("Sensors", sensors, default=default_sensors or sensors[:1], format_func=schema.label)
    show_flags = c2.toggle("Mark anomaly flags", True)
    notes = set()
    for sensor in plot_sensors:
        data, note = insights.downsample(df, sensor)
        if note:
            notes.add(note)
        marks = an[an.sensor == sensor] if show_flags else None
        st.plotly_chart(
            charts.timeseries(data, sensor, colors, thresholds.get(sensor), marks), width="stretch", key=f"ts_{sensor}"
        )
    for note in sorted(notes):
        st.caption(note + " Gaps appear as breaks in the line.")

    st.subheader("Threshold violations")
    st.caption("How often each machine reaches its warn or critical limit, and whether that happens only at start-up.")
    show_table(
        th[th.above_warn > 0] if len(th) and st.toggle("Only machines with violations", True) else th,
        column_config={
            "share_above_warn": pct("share_above_warn", "share ≥ warn"),
            "startup_share": pct("startup_share", "share at start-up"),
            "time_above_warn_s": st.column_config.NumberColumn("time ≥ warn (s)", format="%.0f"),
            "peak": st.column_config.NumberColumn(format="%.2f"),
        },
    )
    if not A["episodes"].empty:
        with st.expander(f"Episodes ({len(A['episodes'])})"):
            show_table(A["episodes"].sort_values("start"))

    st.subheader("Anomaly flags")
    st.caption(
        f"A sample is flagged when it lies at least {z_limit:g} robust standard deviations "
        f"(1.4826 × MAD) from the median of the same machine and {group_by}. This is a transparent rule, not a model."
    )
    c1, c2 = st.columns([1, 2])
    with c1:
        show_table(insights.anomaly_summary(an))
    with c2:
        show_table(an.sort_values("z", ascending=False).head(500))

    st.subheader("Machine comparison")
    c1, c2 = st.columns([1, 3])
    cmp_sensor = c1.selectbox(
        "Sensor",
        sensors,
        index=sensors.index("temperature_c") if "temperature_c" in sensors else 0,
        format_func=schema.label,
        key="cmp_sensor",
    )
    active_only = (
        c1.toggle("Active samples only", True, help="Running or warning status.") if schema.STATUS in df else False
    )
    sub = df[df[schema.STATUS].isin(schema.ACTIVE_STATUSES)] if active_only else df
    c2.plotly_chart(charts.box_by_machine(sub, cmp_sensor, colors), width="stretch")
    stats_tbl = insights.operating_stats(sub, sensors)
    show_table(
        stats_tbl[stats_tbl.sensor == cmp_sensor].drop(columns="sensor"),
        column_config={
            c: st.column_config.NumberColumn(format="%.2f")
            for c in ["mean", "std", "min", "p05", "median", "p95", "max"]
        },
    )

    if not A["states"].empty:
        st.subheader("Operating states")
        c1, c2 = st.columns([2, 3])
        c1.plotly_chart(charts.stacked_hours(A["hours"], "Time in state"), width="stretch")
        with c2:
            show_table(
                A["states"],
                column_config={
                    "time_share": pct("time_share", "time share"),
                    "hours": st.column_config.NumberColumn(format="%.2f"),
                    **{s: st.column_config.NumberColumn(schema.label(s), format="%.2f") for s in sensors},
                },
            )

    if len(sensors) >= 2:
        st.subheader("Relationships")
        c1, c2, c3 = st.columns(3)
        x = c1.selectbox(
            "x", sensors, index=sensors.index("power_kw") if "power_kw" in sensors else 0, format_func=schema.label
        )
        y = c2.selectbox(
            "y",
            sensors,
            index=sensors.index("vibration_mm_s") if "vibration_mm_s" in sensors else 1,
            format_func=schema.label,
        )
        rel_active = c3.toggle("Active samples only", True, key="rel_active") if schema.STATUS in df else False
        rel = df[df[schema.STATUS].isin(schema.ACTIVE_STATUSES)] if rel_active else df
        c1, c2 = st.columns([3, 2])
        c1.plotly_chart(charts.scatter(rel, x, y, colors), width="stretch")
        with c2:
            st.markdown("**Spearman correlation, per machine and pooled**")
            show_table(
                insights.correlation_by_machine(rel, x, y),
                column_config={"r": st.column_config.NumberColumn(format="%.2f")},
            )
            st.caption(
                "Correlation does not prove a mechanical cause. Idle and cutting regimes, or machines of different "
                "size, can create a strong pooled correlation that does not hold within any one machine."
            )
        with st.expander("Correlation matrix (all sensors)"):
            st.plotly_chart(
                charts.heatmap(insights.correlations(rel, sensors), "Spearman correlation"), width="stretch"
            )

# ------------------------------------------------------------------ data quality
with tab_q:
    st.caption("Checks run on the imported data before cleaning, so the numbers describe the file as delivered.")
    k = st.columns(6)
    k[0].metric("Rows", f"{totals['rows']:,}")
    k[1].metric("Valid values", f"{valid:.2%}")
    k[2].metric("Data gaps", totals["gaps"], help="Intervals longer than 5x the machine's median interval.")
    k[3].metric("Duplicates", totals["exact_duplicates"])
    k[4].metric("Implausible", totals["implausible"])
    k[5].metric(
        "Irregular intervals",
        f"{totals['irregular_intervals']:,}",
        help="Intervals more than 50% away from the median, excluding gaps.",
    )

    st.subheader("Validation findings")
    if issues:
        for i in issues:
            {"error": st.error, "warning": st.warning}.get(i.severity, st.info)(i.message)
    else:
        st.success("No validation issues: timestamps, machine ids, statuses and numbers all parsed.")

    st.subheader("Per machine")
    show_table(
        raw_q.machines,
        column_config={
            "coverage": pct("coverage"),
            "missing_sample_rate": pct("missing_sample_rate", "missing sample rate"),
            "irregular_share": pct("irregular_share", "irregular share"),
            "valid_value_share": pct("valid_value_share", "valid values"),
            "median_interval_s": st.column_config.NumberColumn("median interval (s)", format="%.3f"),
            "span_s": st.column_config.NumberColumn("span (s)", format="%.0f"),
        },
    )

    c1, c2 = st.columns(2)
    c1.plotly_chart(
        charts.heatmap(raw_q.missing * 100, "Missing values per sensor (%)", diverging=False, fmt=".2f"),
        width="stretch",
    )
    c2.plotly_chart(
        charts.histogram(
            raw_q.intervals,
            "interval_s",
            colors,
            "Sampling intervals",
            log_y=True,
            xaxis_title="seconds between samples",
        ),
        width="stretch",
    )

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Data gaps** (no rows at all)")
        show_table(raw_q.gaps)
    with c2:
        st.markdown("**Sensor dropouts** (one sensor missing for ≥ 30 s)")
        show_table(raw_q.dropouts)
    st.markdown("**Implausible values** per sensor (outside the plausible range in the data dictionary)")
    show_table(raw_q.implausible.reset_index(names="machine_id"))

    st.subheader("Cleaning")
    c1, c2 = st.columns([2, 1])
    with c1:
        for d in cleaned.decisions:
            st.markdown(f"- {d}")
    c2.metric("Rows before → after", f"{cleaned.rows_before:,} → {cleaned.rows_after:,}")
    with st.expander("Original rows as imported (first 200, untouched)"):
        st.dataframe(parsed.raw.head(200), width="stretch")
    with st.expander("Cleaned rows (first 200)"):
        st.dataframe(full.head(200), width="stretch", hide_index=True)

# -------------------------------------------------------------------- cycles
with tab_cyc:
    ct = A["cycles"]
    st.caption(
        f"Cycles identified from the **{cycle_source}**. Start-up is the first "
        f"{cycles.STARTUP_WINDOW_S:.0f} s of a cycle, shutdown the last {cycles.SHUTDOWN_WINDOW_S:.0f} s."
    )
    if ct.empty:
        st.info("No operating cycles found. Map a cycle_id or status column to enable cycle views.")
    else:
        util = A["utilisation"]
        c1, c2 = st.columns([3, 2])
        c1.plotly_chart(charts.stacked_hours(A["hours"], "Active, idle, stopped and missing time"), width="stretch")
        with c2:
            show_table(
                util.reset_index(),
                column_config={
                    "active_share": pct("active_share", "active share"),
                    **{
                        c: st.column_config.NumberColumn(c.replace("_h", " (h)"), format="%.2f")
                        for c in ["observed_h", "active_h", "idle_h", "stopped_h", "no_data_h"]
                    },
                },
            )
        per_machine = (
            ct.groupby("machine_id")
            .agg(
                cycles=("cycle", "size"),
                mean_duration_min=("duration_s", lambda s: s.mean() / 60),
                max_duration_min=("duration_s", lambda s: s.max() / 60),
                **({"energy_kwh": ("energy_kwh", "sum")} if "energy_kwh" in ct else {}),
            )
            .reset_index()
        )
        show_table(
            per_machine,
            column_config={
                c: st.column_config.NumberColumn(format="%.1f")
                for c in ["mean_duration_min", "max_duration_min", "energy_kwh"]
            },
        )
        st.plotly_chart(
            charts.box_by_machine(
                ct.assign(duration_min=ct.duration_s / 60),
                "duration_min",
                colors,
                "Cycle durations",
                yaxis_title="minutes",
            ),
            width="stretch",
        )
        st.subheader("All cycles")
        show_table(
            ct,
            column_config={
                "warning_share": pct("warning_share", "warning share"),
                "duration_s": st.column_config.NumberColumn("duration (s)", format="%.0f"),
                **{
                    c: st.column_config.NumberColumn(format="%.2f")
                    for c in ct.columns
                    if c.endswith(("_c", "_mm_s", "_kw", "_kwh")) or c == "rpm_mean"
                },
            },
        )

        st.subheader("Cycle detail")
        c1, c2 = st.columns(2)
        cyc_machines = sorted(ct.machine_id.unique())
        hottest_machine = (
            ct.loc[ct.temp_max_c.idxmax(), "machine_id"]
            if "temp_max_c" in ct and ct.temp_max_c.notna().any()
            else cyc_machines[0]
        )
        cm = c1.selectbox(
            "Machine",
            cyc_machines,
            index=cyc_machines.index(hottest_machine),
            key="cyc_machine",
            help="Defaults to the machine with the hottest cycle.",
        )
        mc = ct[ct.machine_id == cm]
        hottest = (
            mc.sort_values("temp_max_c", ascending=False).cycle.iloc[0] if "temp_max_c" in mc else mc.cycle.iloc[0]
        )
        labels = {
            r.cycle: f"{r.cycle.split('#')[-1]} · {r.start:%H:%M} · {r.duration_s / 60:.0f} min"
            for r in mc.itertuples()
        }
        sel = c2.selectbox(
            "Cycle",
            list(labels),
            index=list(labels).index(hottest),
            format_func=labels.get,
            help="Defaults to the hottest cycle of the machine.",
        )
        st.caption(
            "The selected cycle is drawn in colour over every other cycle of the same machine (grey), "
            "aligned on time since the cycle started."
        )
        md = df[df[schema.MACHINE_ID] == cm]
        cols = st.columns(2)
        for i, sensor in enumerate(sensors):
            cols[i % 2].plotly_chart(
                charts.cycle_overlay(md, sensor, sel, colors[cm], thresholds.get(sensor)),
                width="stretch",
                key=f"cyc_{sensor}",
            )

# ----------------------------------------------------------- compare windows
with tab_cmp:
    st.caption(
        "Compare two periods, for example before and after a shift change. Uses the machine and status "
        "filters from the sidebar."
    )
    lo, hi = to_py(w_start), to_py(w_end)
    mid = lo + (hi - lo) / 2
    if hi > lo:
        c1, c2 = st.columns(2)
        wa = c1.slider("Window A", lo, hi, (lo, mid), format="MM-DD HH:mm", key="wa")
        wb = c2.slider("Window B", lo, hi, (mid, hi), format="MM-DD HH:mm", key="wb")
        comp = insights.compare_windows(df, sensors, (to_ts(wa[0]), to_ts(wa[1])), (to_ts(wb[0]), to_ts(wb[1])))
        comp_sensor = st.selectbox("Sensor", sensors, format_func=schema.label, key="comp_sensor")
        show_table(
            comp[comp.sensor == comp_sensor].drop(columns="sensor"),
            column_config={
                c: st.column_config.NumberColumn(format="%.3f")
                for c in comp.columns
                if c not in ("machine_id", "n_a", "n_b", "p_value")
            }
            | {"p_value": st.column_config.NumberColumn("p (Mann-Whitney)", format="%.2e")},
        )
        st.caption(
            "Telemetry samples are autocorrelated, so p-values overstate certainty. "
            "Judge the difference in medians against the sensor's normal spread first."
        )
        with st.expander("All sensors"):
            show_table(comp)
    else:
        st.info("The selected time window is a single instant.")

# --------------------------------------------------------------------- report
with tab_rep:
    tables = {
        "Data quality per machine": raw_q.machines,
        "Data gaps": raw_q.gaps,
        "Sensor dropouts": raw_q.dropouts,
        "Threshold summary": A["thresholds"],
        "Threshold episodes": A["episodes"],
        "Anomaly summary": insights.anomaly_summary(A["anomalies"]),
        "Operating states": A["states"],
        "Utilisation": A["utilisation"].reset_index() if len(A["utilisation"]) else A["utilisation"],
        "Operating statistics": A["stats"],
    }
    ctx = report.ReportContext(
        source_name=source_name,
        import_id=parsed.import_id,
        import_config=parsed.config.to_dict(),
        cleaning_options=opts.to_dict(),
        cleaning_decisions=cleaned.decisions,
        validation_issues=[asdict(i) for i in issues],
        filters=filters,
        thresholds=thresholds,
        anomaly_settings=dict(z_limit=z_limit, group_by=group_by, direction=direction),
        tables=tables,
    )
    md = report.to_markdown(ctx)
    st.caption(
        f"Analysis id `{ctx.analysis_id}`. The report states the time window, filters, thresholds and every "
        "cleaning decision, so the same CSV and settings reproduce it exactly."
    )
    c = st.columns(3)
    c[0].download_button(
        "Download report (Markdown)", md, file_name=f"telemetrylens_{ctx.analysis_id}.md", mime="text/markdown"
    )
    c[1].download_button(
        "Download report (JSON)",
        report.to_json(ctx),
        file_name=f"telemetrylens_{ctx.analysis_id}.json",
        mime="application/json",
    )
    export = df.drop(columns=["t_in_cycle_s"]).copy()
    c[2].download_button(
        "Download filtered, cleaned CSV",
        export.to_csv(index=False).encode(),
        file_name=f"telemetrylens_{ctx.analysis_id}_clean.csv",
        mime="text/csv",
    )
    with st.container(border=True):
        st.markdown(md)

# ------------------------------------------------------------- data dictionary
with tab_dict:
    st.subheader("Sensors")
    show_table(
        pd.DataFrame(
            [
                dict(
                    field=s.name,
                    label=s.label,
                    unit=s.unit,
                    description=s.description,
                    plausible_range=f"{s.plausible_min:g} to {s.plausible_max:g}",
                    default_warn=s.warn,
                    default_critical=s.critical,
                    plot_resampling=s.resample_agg,
                    mapped_from=parsed.config.mapping.get(s.name) or "not mapped",
                )
                for s in schema.SENSORS.values()
            ]
        )
    )
    st.caption(
        "Resampling is used only to draw long time series. Temperature is averaged per bucket, vibration "
        "keeps the bucket maximum so short spikes stay visible."
    )
    st.subheader("Statuses")
    st.markdown(
        "Canonical states: "
        + ", ".join(f"`{s}`" for s in schema.STATUSES)
        + ". Running and warning count as active time."
    )
    show_table(pd.DataFrame(sorted(schema.STATUS_ALIASES.items()), columns=["value in file", "read as"]))

    if ground_truth is not None:
        st.subheader("Demo ground truth")
        st.caption(
            "Faults the generator planted, and whether TelemetryLens finds them with the default settings "
            "(computed on all machines and the full time range)."
        )
        temp_lim = thresholds.get("temperature_c", {}).get("warn") or 75.0
        full_anoms = insights.flag_anomalies(full, sensors, z_limit, group_by, direction)
        det = faultcheck.detection_table(
            ground_truth,
            raw_q,
            issues,
            insights.violation_episodes(full, "temperature_c", temp_lim)
            if "temperature_c" in sensors
            else pd.DataFrame(columns=["machine_id", "start", "end"]),
            full_anoms,
        )
        show_table(det, column_config={"detected": st.column_config.CheckboxColumn()})
