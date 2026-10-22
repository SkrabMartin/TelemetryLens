"""Plotly figures with a fixed, colour-blind-checked palette.

Machine colours follow the machine, not its position in a filtered list, so a
filter never repaints the remaining machines.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from . import schema

# Categorical slots in fixed order (validated for adjacent-pair CVD separation).
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
OTHER = "#8a8985"
STATUS_COLORS = {
    "running": "#0ca30c",
    "warning": "#fab219",
    "idle": "#a3a29c",
    "stopped": "#52514e",
    "unknown": "#d03b3b",
    "no status": "#c3c2b7",
    "no data": "#e6e5e1",
}
WARN = "#fab219"
CRITICAL = "#d03b3b"
DIVERGING = [[0.0, "#2a78d6"], [0.5, "#e8e7e3"], [1.0, "#e34948"]]
SEQUENTIAL = [[0.0, "#cde2fb"], [0.5, "#3987e5"], [1.0, "#0d366b"]]

LAYOUT = dict(
    margin=dict(l=10, r=10, t=72, b=10),
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="left", x=0),
)


def _title(text: str) -> dict:
    return dict(text=text, font_size=14, x=0, xref="paper", y=0.98, yref="container", yanchor="top")


def machine_colors(machines: list[str]) -> dict[str, str]:
    return {m: SERIES[i] if i < len(SERIES) else OTHER for i, m in enumerate(sorted(machines))}


def timeseries(
    data: pd.DataFrame,
    sensor: str,
    colors: dict[str, str],
    thresholds: dict | None = None,
    markers: pd.DataFrame | None = None,
    height: int = 300,
) -> go.Figure:
    fig = go.Figure()
    for mid, g in data.groupby(schema.MACHINE_ID, sort=True):
        fig.add_trace(
            go.Scatter(
                x=g[schema.TIMESTAMP],
                y=g[sensor],
                mode="lines",
                name=mid,
                line=dict(width=1.5, color=colors.get(mid, OTHER)),
                connectgaps=False,
                hovertemplate=f"{mid}: %{{y:.2f}} {schema.SENSORS[sensor].unit}<extra></extra>",
            )
        )
    for level, color in (("warn", WARN), ("critical", CRITICAL)):
        value = (thresholds or {}).get(level)
        if value is not None:
            fig.add_hline(
                y=value,
                line=dict(color=color, width=1.5, dash="dash"),
                annotation_text=f"{level} {value:g}",
                annotation_position="top left",
                annotation_font_size=11,
            )
    if markers is not None and not markers.empty:
        fig.add_trace(
            go.Scatter(
                x=markers[schema.TIMESTAMP],
                y=markers["value"],
                mode="markers",
                name="Anomaly flag",
                marker=dict(size=9, color="rgba(0,0,0,0)", line=dict(color=CRITICAL, width=2), symbol="circle"),
                customdata=np.stack([markers[schema.MACHINE_ID], markers["z"]], axis=-1),
                hovertemplate="%{customdata[0]} flagged: %{y:.2f}, z=%{customdata[1]}<extra></extra>",
            )
        )
    fig.update_layout(title=_title(schema.label(sensor)), height=height, **LAYOUT)
    fig.update_yaxes(title=None)
    return fig


def box_by_machine(
    data: pd.DataFrame, column: str, colors: dict[str, str], title: str | None = None, yaxis_title: str | None = None
) -> go.Figure:
    fig = go.Figure()
    for mid, g in data.groupby(schema.MACHINE_ID, sort=True):
        fig.add_trace(
            go.Box(
                y=g[column].dropna(),
                name=mid,
                marker_color=colors.get(mid, OTHER),
                line_width=1.5,
                boxpoints="outliers",
                marker_size=4,
                hoverinfo="y+name",
            )
        )
    if title is None:
        title = f"{schema.label(column)} by machine"
    fig.update_layout(
        title=_title(title), height=340, showlegend=False, margin=LAYOUT["margin"], yaxis_title=yaxis_title
    )
    return fig


def stacked_hours(hours: pd.DataFrame, title: str) -> go.Figure:
    fig = go.Figure()
    for col in hours.columns:
        fig.add_trace(
            go.Bar(
                y=hours.index,
                x=hours[col],
                name=col,
                orientation="h",
                marker=dict(color=STATUS_COLORS.get(col, OTHER), line=dict(width=1, color="rgba(255,255,255,0.8)")),
                hovertemplate=f"%{{y}} {col}: %{{x:.2f}} h<extra></extra>",
            )
        )
    fig.update_layout(
        barmode="stack",
        title=_title(title),
        height=80 + 40 * len(hours),
        xaxis_title="hours",
        margin=LAYOUT["margin"],
        legend=dict(LAYOUT["legend"], traceorder="normal"),
    )
    fig.update_yaxes(autorange="reversed")
    return fig


def heatmap(matrix: pd.DataFrame, title: str, diverging: bool = True, fmt: str = ".2f") -> go.Figure:
    fig = go.Figure(
        go.Heatmap(
            z=matrix.to_numpy(),
            x=list(matrix.columns),
            y=list(matrix.index),
            colorscale=DIVERGING if diverging else SEQUENTIAL,
            zmin=-1 if diverging else 0,
            zmax=1 if diverging else max(float(np.nanmax(matrix.to_numpy())), 1e-9),
            text=matrix.to_numpy(),
            texttemplate=f"%{{text:{fmt}}}",
            hovertemplate="%{y} / %{x}: %{z:.3f}<extra></extra>",
        )
    )
    fig.update_layout(title=_title(title), height=120 + 36 * len(matrix), margin=LAYOUT["margin"])
    fig.update_yaxes(autorange="reversed")
    return fig


def scatter(data: pd.DataFrame, x: str, y: str, colors: dict[str, str], max_points: int = 6000) -> go.Figure:
    d = data[[schema.MACHINE_ID, x, y]].dropna()
    if len(d) > max_points:
        d = d.sample(max_points, random_state=0)
    fig = go.Figure()
    for mid, g in d.groupby(schema.MACHINE_ID, sort=True):
        fig.add_trace(
            go.Scatter(
                x=g[x],
                y=g[y],
                mode="markers",
                name=mid,
                marker=dict(size=5, color=colors.get(mid, OTHER), opacity=0.55),
                hovertemplate=f"{mid}: %{{x:.2f}}, %{{y:.2f}}<extra></extra>",
            )
        )
    fig.update_layout(
        title=_title(f"{schema.SENSORS[y].label} vs {schema.SENSORS[x].label}"),
        height=380,
        margin=LAYOUT["margin"],
        legend=LAYOUT["legend"],
        xaxis_title=schema.label(x),
        yaxis_title=schema.label(y),
    )
    return fig


def histogram(
    data: pd.DataFrame,
    column: str,
    colors: dict[str, str],
    title: str,
    log_y: bool = False,
    xaxis_title: str | None = None,
) -> go.Figure:
    fig = go.Figure()
    for mid, g in data.groupby(schema.MACHINE_ID, sort=True):
        fig.add_trace(go.Histogram(x=g[column], name=mid, marker_color=colors.get(mid, OTHER), opacity=0.75, nbinsx=60))
    fig.update_layout(
        barmode="overlay",
        title=_title(title),
        height=320,
        margin=LAYOUT["margin"],
        legend=LAYOUT["legend"],
        xaxis_title=xaxis_title,
    )
    if log_y:
        fig.update_yaxes(type="log")
    return fig


def cycle_overlay(
    data: pd.DataFrame,
    sensor: str,
    selected: str,
    color: str,
    thresholds: dict | None = None,
    max_cycles: int = 60,
) -> go.Figure:
    """All cycles of one machine against time-in-cycle (grey), the selected one highlighted."""
    fig = go.Figure()
    cycles = list(data["cycle"].dropna().unique())[:max_cycles]
    for c in cycles:
        if c == selected:
            continue
        g = data[data["cycle"] == c]
        step = max(1, len(g) // 300)
        fig.add_trace(
            go.Scatter(
                x=g.t_in_cycle_s.iloc[::step] / 60,
                y=g[sensor].iloc[::step],
                mode="lines",
                line=dict(width=1, color="rgba(138,137,133,0.35)"),
                name=str(c),
                showlegend=False,
                hoverinfo="skip",
            )
        )
    g = data[data["cycle"] == selected]
    fig.add_trace(
        go.Scatter(
            x=g.t_in_cycle_s / 60,
            y=g[sensor],
            mode="lines",
            line=dict(width=2.5, color=color),
            name=f"cycle {selected.split('#')[-1]}",
            hovertemplate="%{x:.1f} min: %{y:.2f}<extra></extra>",
        )
    )
    for level, col in (("warn", WARN), ("critical", CRITICAL)):
        value = (thresholds or {}).get(level)
        if value is not None:
            fig.add_hline(y=value, line=dict(color=col, width=1.5, dash="dash"))
    fig.update_layout(
        title=_title(schema.label(sensor)),
        height=260,
        margin=LAYOUT["margin"],
        xaxis_title="minutes into cycle",
        legend=LAYOUT["legend"],
    )
    return fig
