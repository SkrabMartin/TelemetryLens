"""Reproducible report export (Markdown and JSON).

A report records the import identity, every cleaning decision, the filters and
thresholds in force, so a colleague can reproduce the numbers from the same CSV.
Output is deterministic: identical inputs produce byte-identical reports.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

CAVEATS = [
    "Correlation between sensors does not establish a mechanical cause; different operating "
    "regimes (idle vs. cutting) can create misleading pooled relationships.",
    "Anomaly flags are robust z-scores against the same machine and operating phase. They "
    "highlight unusual samples; they are not predictions of failure.",
    "Missing readings and data gaps were not interpolated. Time-weighted figures treat gaps as 'no data'.",
    "Window comparison p-values assume independent samples; telemetry is autocorrelated, so "
    "read them as indicative and focus on the size of the difference.",
]


@dataclass
class ReportContext:
    source_name: str
    import_id: str
    import_config: dict
    cleaning_options: dict
    cleaning_decisions: list[str]
    validation_issues: list[dict]
    filters: dict
    thresholds: dict
    anomaly_settings: dict
    tables: dict[str, pd.DataFrame] = field(default_factory=dict)

    def settings(self) -> dict:
        return {
            "import_id": self.import_id,
            "import_config": self.import_config,
            "cleaning_options": self.cleaning_options,
            "filters": self.filters,
            "thresholds": self.thresholds,
            "anomaly_settings": self.anomaly_settings,
        }

    @property
    def analysis_id(self) -> str:
        blob = json.dumps(self.settings(), sort_keys=True, default=str).encode()
        return hashlib.sha256(blob).hexdigest()[:12]


def _fmt(v) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)) or v is pd.NaT:
        return "–"
    if isinstance(v, pd.Timestamp):
        return v.strftime("%Y-%m-%d %H:%M:%S") + ("Z" if v.tzinfo else "")
    if isinstance(v, (float, np.floating)):
        return f"{v:,.3f}".rstrip("0").rstrip(".") if abs(v) < 1e6 else f"{v:.3e}"
    return str(v).replace("|", "\\|")


def markdown_table(df: pd.DataFrame, max_rows: int = 50) -> str:
    if df is None or df.empty:
        return "_None._\n"
    shown = df.head(max_rows)
    lines = ["| " + " | ".join(map(str, shown.columns)) + " |", "|" + "---|" * len(shown.columns)]
    for row in shown.itertuples(index=False):
        lines.append("| " + " | ".join(_fmt(v) for v in row) + " |")
    if len(df) > max_rows:
        lines.append(f"\n_{len(df) - max_rows} more rows not shown._")
    return "\n".join(lines) + "\n"


def to_markdown(ctx: ReportContext) -> str:
    f = ctx.filters
    out = [
        "# TelemetryLens analysis report",
        "",
        f"- **Source file:** `{ctx.source_name}`",
        f"- **Import id:** `{ctx.import_id}` (hash of file content and import settings)",
        f"- **Analysis id:** `{ctx.analysis_id}` (hash of import id, cleaning, filters and thresholds)",
        f"- **Time window:** {_fmt(f.get('start'))} to {_fmt(f.get('end'))} (UTC)",
        f"- **Machines:** {', '.join(f.get('machines', [])) or 'all'}",
        f"- **Statuses:** {', '.join(f.get('statuses', [])) or 'all'}",
        "",
        "## Import settings",
        "",
        f"- Timestamps without offset interpreted as: `{ctx.import_config.get('source_timezone')}`",
        "- Column mapping: "
        + ", ".join(f"`{k}` ← `{v}`" for k, v in ctx.import_config.get("mapping", {}).items() if v),
    ]
    units = {k: v for k, v in ctx.import_config.get("units", {}).items() if v}
    if units:
        out.append("- Source units: " + ", ".join(f"`{k}` in {v}" for k, v in units.items()))
    out += ["", "## Validation findings", ""]
    if ctx.validation_issues:
        out += [f"- **{i['severity']}** ({i['check']}): {i['message']}" for i in ctx.validation_issues]
    else:
        out.append("- No validation issues.")
    out += ["", "## Cleaning decisions", ""]
    out += [f"- {d}" for d in ctx.cleaning_decisions]
    out += ["", "## Thresholds and anomaly settings", ""]
    for sensor, lim in ctx.thresholds.items():
        if lim.get("warn") is not None or lim.get("critical") is not None:
            out.append(f"- `{sensor}`: warn ≥ {_fmt(lim.get('warn'))}, critical ≥ {_fmt(lim.get('critical'))}")
    out.append(
        "- Anomalies: robust z-score ≥ "
        f"{ctx.anomaly_settings.get('z_limit')} against machine and {ctx.anomaly_settings.get('group_by')}, "
        f"direction: {ctx.anomaly_settings.get('direction')}"
    )
    for title, table in ctx.tables.items():
        out += ["", f"## {title}", "", markdown_table(table)]
    out += ["## Caveats", ""] + [f"- {c}" for c in CAVEATS] + [""]
    return "\n".join(out)


def to_json(ctx: ReportContext) -> str:
    def records(df: pd.DataFrame) -> list[dict]:
        return json.loads(df.to_json(orient="records", date_format="iso", double_precision=6))

    payload = {
        "source_name": ctx.source_name,
        "analysis_id": ctx.analysis_id,
        **ctx.settings(),
        "validation_issues": ctx.validation_issues,
        "cleaning_decisions": ctx.cleaning_decisions,
        "tables": {k: records(v) for k, v in ctx.tables.items()},
        "caveats": CAVEATS,
    }
    return json.dumps(payload, indent=2, sort_keys=False, default=str)
