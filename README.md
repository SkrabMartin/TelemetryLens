# TelemetryLens

TelemetryLens turns raw machine telemetry from CNC mills and industrial motors into explainable
data-quality reports, operating-cycle views and engineering insights. It is a local Streamlit app
built on Pandas, NumPy, SciPy and Plotly. It uses only rules and descriptive statistics, with no
predictive models: predictive maintenance belongs to the separate MachineGuard project.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py               # opens http://localhost:8501
```

The app opens on synthetic demo data (5 machines, 8 hours, about 114k rows). To analyse your own file,
pick **Upload CSV** in the sidebar and confirm the column mapping under **Import settings**.

```bash
pytest                             # 41 tests, about 20 s (includes one full app render)
pytest -m "not slow"               # skip the app render
ruff check . && ruff format --check .
python scripts/generate_data.py --machines 5 --hours 8 --seed 42 --out data/sample_telemetry.csv
python scripts/benchmark.py --hours 1 8 24
```

## Input format

One row per measurement. Only `timestamp`, `machine_id` and at least one sensor are required.

```csv
timestamp,machine_id,cycle_id,temperature_c,vibration_mm_s,rpm,current_a,power_kw,torque_nm,status
2026-10-22T10:00:00Z,CNC_01,101,45.2,1.8,1500,8.2,3.5,22.1,running
```

| Field | Unit | Plausible range | Default warn / critical | Plot resampling |
|---|---|---|---|---|
| temperature_c | °C | -20 to 150 | 75 / 90 | mean |
| vibration_mm_s | mm/s RMS | 0 to 50 | 4.5 / 7.1 (ISO 10816-style zones) | max |
| rpm | rpm | 0 to 12000 | none | mean |
| current_a | A | 0 to 200 | none | mean |
| power_kw | kW | 0 to 100 | none | mean |
| torque_nm | Nm | 0 to 500 | none | mean |
| status | running, warning, idle, stopped | aliases such as RUN, alarm, standby, off are mapped | | |

Column names are matched by alias (`Time`, `Device ID`, `Temp`, `State`, ...) and can be changed in the
sidebar. Timestamps with `Z` or an offset keep it; timestamps without one are read in the timezone you
choose. Temperature in °F or K, vibration in in/s and power in W are converted on import.

## What the app does

| Tab | Content |
|---|---|
| **Insights** (main) | Data-quality banner, KPIs, sensor trends per machine with threshold lines and anomaly markers, threshold violations and episodes (with the share that happens at start-up), anomaly flags, machine comparison, time in each state, sensor relationships with per-machine vs pooled correlation |
| **Data quality** | Validation findings, per-machine coverage, missing values per sensor, sampling-interval histogram, data gaps, sensor dropouts, implausible values, the cleaning log, original vs cleaned rows |
| **Cycles** | Cycles from `cycle_id` or from status transitions, active/idle/stopped/no-data hours, cycle durations, a table of all cycles, and a drill-down that overlays one cycle on all others of the same machine |
| **Compare windows** | Two time windows side by side (for example before and after a shift change) with medians, p95 and a Mann-Whitney test |
| **Report** | Markdown and JSON report plus the filtered, cleaned CSV |
| **Data dictionary** | Units, ranges, thresholds, status aliases, and for demo data the planted faults with a detected / not detected check |

Sidebar filters (machines, time window, status) apply to every tab and to the exported report.

## Design decisions

- **The original file is kept untouched.** Parsing creates a separate frame with a `source_row` column
  pointing back to the line in the CSV.
- **Nothing is dropped silently.** Unreadable timestamps, unknown statuses, non-numeric values,
  out-of-order rows and unit conversions all produce a visible validation finding.
- **Cleaning is explicit and logged:** exact duplicates removed, conflicting timestamps kept first,
  averaged or kept (your choice), implausible values set to missing. Missing data and gaps are never
  interpolated.
- **Quality checks are deterministic.** A gap is an interval longer than 5× the machine's median
  interval (at least 10 s). An interval is irregular when it is more than 50% away from the median. A
  dropout is one sensor missing for at least 30 s.
- **Time-weighted figures do not bridge gaps.** A logger outage counts as "no data", not as the last
  state seen.
- **Anomaly flags are a transparent rule:** a sample is flagged when its robust z-score
  `(x − median) / (1.4826 × MAD)` against the same machine and operating phase (start-up, steady,
  shutdown, idle, stopped) is 6 or more. High values only by default, because low temperatures are
  mostly warm-up after a stop.
- **Threshold episodes** merge excursions separated by 5 or fewer samples, so noise around a limit does
  not split one event into many.
- **Downsampling is for display only.** Long series are bucketed per the sensor's policy (mean for
  temperature, max for vibration so spikes stay visible). Statistics always use every sample.
- **Reports are reproducible.** The import id hashes the file and the import settings. The analysis id
  also covers cleaning, filters, thresholds and anomaly settings. The same inputs give a byte-identical
  report.

## Synthetic data and ground truth

`telemetrylens/generator.py` simulates start-up ramps (including a short resonance pass in vibration),
steady cutting at varying load, idle time, shutdowns and occasional stops, with a first-order thermal
model. It then plants known faults and returns them as a ground-truth table:

| Fault | Machine | Planted | Found by |
|---|---|---|---|
| Cooling fault: temperature climbs above 90 °C during one cycle | CNC_02 | 1 cycle | threshold episode, anomaly flags |
| Vibration impact spikes, 8 to 14 mm/s | CNC_03 | 12 samples | critical threshold count and anomaly flags, exactly 12 |
| Temperature sensor dropout | MOTOR_01 | 300 samples (10 min) | dropout list |
| Logger offline | CNC_01 | 900 samples (15 min) | gap list, coverage 96.9% |
| Exact duplicate rows | CNC_01 | 20 | duplicate count |
| Rows swapped in the file | CNC_03 | 10 | out-of-order finding |
| Sentinel values −999 °C and 99999 rpm | MOTOR_02 | 5 + 3 | implausible values |
| ±0.6 s jitter and ~3% dropped samples | MOTOR_02 | whole series | irregular-interval share |

The tests check these counts exactly, along with zero false threshold violations and zero anomaly flags
on the healthy machines.

## Performance (5 machines, single laptop-class CPU)

| hours | rows | CSV MB | parse s | quality s | clean + cycles s | insights s | total s |
|---|---|---|---|---|---|---|---|
| 1 | 13,456 | 1.0 | 0.18 | 0.12 | 0.05 | 0.18 | 0.54 |
| 8 | 113,908 | 8.8 | 1.21 | 0.33 | 0.21 | 0.65 | 2.40 |
| 24 | 343,420 | 26.7 | 3.33 | 1.25 | 0.81 | 2.19 | 7.58 |
| 48 | 687,692 | 53.6 | 8.04 | 1.69 | 1.87 | 3.73 | 15.34 |

Steps are cached, so changing a filter or threshold reruns only the analysis. Files up to about
50 MB stay comfortable. Parsing is the bottleneck, which makes it the natural place to try Polars or
Parquet later.

## Limitations

- Anomaly flags compare each sample with its own machine and phase. A machine that is abnormal for the
  entire file looks normal to it, so use the machine comparison and thresholds for that case.
- Status-derived cycles depend on the status column. Without `cycle_id` or `status`, cycle views are
  unavailable.
- Window-comparison p-values assume independent samples. Telemetry is autocorrelated, so they overstate
  certainty.
- Units are converted only for the listed source units. Mixed units within one column are not detected.

## Repository layout

```
telemetrylens/
├── .streamlit/
│   └── config.toml                     Streamlit theme and server settings
├── data/
│   ├── sample_telemetry.csv            demo dataset: 5 machines, 8 hours, ~114k rows
│   └── sample_telemetry_ground_truth.csv  faults planted in the demo dataset
├── scripts/
│   ├── benchmark.py                    timing of parse, quality, cycles and insights per file size
│   └── generate_data.py                CLI for the synthetic data generator
├── telemetrylens/                      analysis package (no Streamlit imports)
│   ├── __init__.py
│   ├── schema.py                       data dictionary: fields, units, ranges, thresholds, aliases
│   ├── generator.py                    synthetic telemetry with ground-truth faults
│   ├── ingest.py                       CSV import, column mapping, validation, timezone and unit handling
│   ├── quality.py                      quality report and logged cleaning
│   ├── cycles.py                       cycles, operating phases, time in state
│   ├── insights.py                     statistics, thresholds, anomaly flags, correlations, window comparison
│   ├── report.py                       reproducible Markdown and JSON reports
│   ├── charts.py                       Plotly figures with a fixed, colour-blind-checked palette
│   └── faultcheck.py                   planted vs detected faults
├── tests/
│   ├── conftest.py                     shared fixtures (generated dataset)
│   ├── test_app.py                     full app render (marked slow)
│   ├── test_cycles.py
│   ├── test_generator.py
│   ├── test_ingest.py
│   ├── test_insights.py
│   ├── test_quality.py
│   └── test_report_and_faults.py
├── .gitignore
├── app.py                              Streamlit UI: sidebar filters and the six tabs
├── pyproject.toml                      package metadata, pytest and Ruff settings
├── README.md
└── requirements.txt                    runtime and dev dependencies
```
