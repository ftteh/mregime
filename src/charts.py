"""
Plotly figure builders for the participation / rotation / valuation /
standardized-signal panels.

Pure functions (no Streamlit): each takes pandas data and returns a
`go.Figure`, so tests can build every figure offline and app.py only decides
layout. Styling matches the dashboard's dark theme (transparent background,
faint grid) while borrowing the research-chart conventions: shaded extreme
zones, labelled end-points and dashed reference levels.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Palette (validated for contrast on the dark dashboard background)
INK = "#f2f4f5"
MUTED = "rgba(255,255,255,0.55)"
GRID = "rgba(255,255,255,0.08)"
RED = "#e8615f"
GREEN = "#2ecc8f"
BLUE = "#5da9e9"
TEAL = "#3cc5b0"
GOLD = "#c9a45c"
STEEL = "#8aa4c8"
RED_ZONE = "rgba(232,97,95,0.14)"
GREEN_ZONE = "rgba(46,204,143,0.12)"
NEUTRAL_ZONE = "rgba(255,255,255,0.05)"


def _base_layout(fig: go.Figure, height: int, *, legend: bool = False, top: int = 40) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=70, t=top, b=10),
        hovermode="x unified",
        showlegend=legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, font=dict(size=11)),
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="rgba(255,255,255,0.9)"),
    )
    fig.update_xaxes(gridcolor=GRID, type="date", showspikes=False)
    fig.update_yaxes(gridcolor=GRID, zeroline=False)
    return fig


def _clip(s: pd.Series | None, start: pd.Timestamp | None) -> pd.Series:
    if s is None or len(s) == 0:
        return pd.Series(dtype=float)
    out = pd.to_numeric(s, errors="coerce").dropna()
    if getattr(out.index, "tz", None) is not None:
        out.index = out.index.tz_localize(None)
    out = out.sort_index()
    if start is not None:
        out = out.loc[out.index >= pd.Timestamp(start)]
    return out


def _end_point(
    fig: go.Figure,
    s: pd.Series,
    text: str,
    color: str,
    *,
    row: int | None = None,
    col: int | None = None,
    size: int = 10,
    yshift: int = 0,
) -> None:
    """Dot on the last observation plus a bold value label to its right."""
    if s.empty:
        return
    x, y = s.index[-1], float(s.iloc[-1])
    kw = {} if row is None else {"row": row, "col": col}
    fig.add_trace(
        go.Scatter(
            x=[x], y=[y], mode="markers",
            marker=dict(size=size, color=color, line=dict(color="#ffffff", width=1.2)),
            showlegend=False, hoverinfo="skip",
        ),
        **kw,
    )
    fig.add_annotation(
        x=x, y=y, text=f"<b>{text}</b>", showarrow=False, xanchor="left", xshift=10,
        yshift=yshift, font=dict(size=13, color=color), **kw,
    )


def add_date_marker(
    fig: go.Figure,
    s: pd.Series,
    ts: pd.Timestamp | None,
    *,
    row: int | None = None,
    col: int | None = None,
) -> None:
    """Red highlight dot at `ts` (the dashboard-wide 'Highlight date')."""
    if ts is None or s is None or s.empty:
        return
    ts = pd.Timestamp(ts).normalize()
    if ts < s.index.min() or ts > s.index.max():
        return
    y = s.asof(ts)
    if pd.isna(y):
        return
    kw = {} if row is None else {"row": row, "col": col}
    fig.add_trace(
        go.Scatter(
            x=[ts], y=[float(y)], mode="markers",
            marker=dict(size=11, color="#ff2d2d", line=dict(color="#ffffff", width=1.2)),
            showlegend=False, hovertemplate="Marker %{x|%Y-%m-%d}<br>%{y:.4g}<extra></extra>",
        ),
        **kw,
    )


# ---------------------------------------------------------------------------
# 1. Overbought / oversold breadth (two panels)
# ---------------------------------------------------------------------------
def obos_figure(
    df: pd.DataFrame,
    start: pd.Timestamp | None,
    zone_lo: float,
    zone_hi: float,
    marker_ts: pd.Timestamp | None = None,
    height: int = 620,
) -> go.Figure:
    spread = _clip(df.get("spread"), start)
    ob = _clip(df.get("ob_pct"), start)
    os_ = _clip(df.get("os_pct"), start)
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.11,
        subplot_titles=("50-DAY MOVING AVERAGE BREADTH SPREAD",
                        "OVERBOUGHT vs OVERSOLD CONSTITUENTS"),
    )
    if not spread.empty:
        y_top = max(float(spread.max()), zone_hi) + 8
        y_bot = min(float(spread.min()), zone_lo) - 8
        fig.add_hrect(y0=zone_hi, y1=y_top, fillcolor=RED_ZONE, line_width=0, layer="below", row=1, col=1, exclude_empty_subplots=False)
        fig.add_hrect(y0=y_bot, y1=zone_lo, fillcolor=GREEN_ZONE, line_width=0, layer="below", row=1, col=1, exclude_empty_subplots=False)
        fig.add_hline(y=0, line_color="rgba(255,255,255,0.55)", line_width=1, row=1, col=1, exclude_empty_subplots=False)
        fig.add_trace(
            go.Scatter(x=spread.index, y=spread.values, mode="lines", name="OB − OS spread",
                       line=dict(color=BLUE, width=2.4),
                       hovertemplate="%{x|%Y-%m-%d}<br>spread %{y:+.1f} pp<extra></extra>"),
            row=1, col=1,
        )
        _end_point(fig, spread, f"{spread.iloc[-1]:+.1f}", BLUE, row=1, col=1)
        add_date_marker(fig, spread, marker_ts, row=1, col=1)
        fig.update_yaxes(range=[y_bot, y_top], title_text="Overbought − Oversold (pp)",
                         tickformat="+d", row=1, col=1)
    if not ob.empty:
        fig.add_trace(
            go.Scatter(x=ob.index, y=ob.values, mode="lines", name="Overbought %",
                       line=dict(color=RED, width=2.1),
                       hovertemplate="OB %{y:.1f}%<extra></extra>"),
            row=2, col=1,
        )
    if not os_.empty:
        fig.add_trace(
            go.Scatter(x=os_.index, y=os_.values, mode="lines", name="Oversold %",
                       line=dict(color=GREEN, width=2.1),
                       hovertemplate="OS %{y:.1f}%<extra></extra>"),
            row=2, col=1,
        )
    if not ob.empty and not os_.empty:
        ob_above = float(ob.iloc[-1]) >= float(os_.iloc[-1])
        _end_point(fig, ob, f"OB {ob.iloc[-1]:.1f}%", RED, row=2, col=1, yshift=10 if ob_above else -10)
        _end_point(fig, os_, f"OS {os_.iloc[-1]:.1f}%", GREEN, row=2, col=1, yshift=-10 if ob_above else 10)
    fig.update_yaxes(title_text="Share of constituents", ticksuffix="%", rangemode="tozero", row=2, col=1)
    _base_layout(fig, height, legend=False, top=36)
    fig.update_annotations(selector=dict(text="50-DAY MOVING AVERAGE BREADTH SPREAD"),
                           font=dict(size=13, color=INK), x=0, xanchor="left")
    fig.update_annotations(selector=dict(text="OVERBOUGHT vs OVERSOLD CONSTITUENTS"),
                           font=dict(size=13, color=INK), x=0, xanchor="left")
    return fig


# ---------------------------------------------------------------------------
# 2. Short-term trend breadth (index + 200DMA over 10/20DMA breadth)
# ---------------------------------------------------------------------------
def trend_breadth_figure(
    breadth: pd.Series,
    index_px: pd.Series,
    start: pd.Timestamp | None,
    *,
    index_name: str = "S&P 500",
    bands: tuple[float, float, float] = (20.0, 50.0, 80.0),
    marker_ts: pd.Timestamp | None = None,
    height: int = 600,
) -> go.Figure:
    px_full = _clip(index_px, None)
    ma200 = px_full.rolling(200, min_periods=200).mean() if len(px_full) else px_full
    px = _clip(px_full, start)
    ma = _clip(ma200, start)
    b = _clip(breadth, start)
    lo, mid, hi = bands
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.09, row_heights=[0.6, 0.4],
        subplot_titles=(index_name.upper(), "SHORT-TERM TREND BREADTH — 10DMA > 20DMA"),
    )
    if not px.empty:
        fig.add_trace(go.Scatter(x=px.index, y=px.values, mode="lines", name=index_name,
                                 line=dict(color=INK, width=1.8),
                                 hovertemplate=f"{index_name} %{{y:,.0f}}<extra></extra>"), row=1, col=1)
        add_date_marker(fig, px, marker_ts, row=1, col=1)
    if not ma.empty:
        fig.add_trace(go.Scatter(x=ma.index, y=ma.values, mode="lines", name="200DMA",
                                 line=dict(color=GOLD, width=1.3),
                                 hovertemplate="200DMA %{y:,.0f}<extra></extra>"), row=1, col=1)
    if not b.empty:
        fig.add_hrect(y0=hi, y1=100, fillcolor=NEUTRAL_ZONE, line_width=0, layer="below", row=2, col=1, exclude_empty_subplots=False)
        fig.add_hrect(y0=0, y1=lo, fillcolor=NEUTRAL_ZONE, line_width=0, layer="below", row=2, col=1, exclude_empty_subplots=False)
        fig.add_hline(y=mid, line_dash="dash", line_color="rgba(255,255,255,0.45)", line_width=1, row=2, col=1, exclude_empty_subplots=False)
        fig.add_trace(go.Scatter(x=b.index, y=b.values, mode="lines", name="10/20DMA breadth",
                                 line=dict(color=TEAL, width=1.8),
                                 hovertemplate="breadth %{y:.1f}%<extra></extra>"), row=2, col=1)
        _end_point(fig, b, f"{b.iloc[-1]:.1f}%", TEAL, row=2, col=1, size=9)
        add_date_marker(fig, b, marker_ts, row=2, col=1)
    fig.update_yaxes(title_text="Index level", tickformat=",d", row=1, col=1)
    fig.update_yaxes(range=[0, 100], tickvals=[lo, mid, hi], ticksuffix="%", row=2, col=1)
    _base_layout(fig, height, legend=True, top=44)
    for t in (index_name.upper(), "SHORT-TERM TREND BREADTH — 10DMA > 20DMA"):
        fig.update_annotations(selector=dict(text=t), font=dict(size=12, color=MUTED), x=0, xanchor="left")
    return fig


# ---------------------------------------------------------------------------
# 3. Relative strength (style rotation)
# ---------------------------------------------------------------------------
def relative_strength_figure(
    rebased: pd.DataFrame,
    pair_label: str,
    marker_ts: pd.Timestamp | None = None,
    height: int = 440,
) -> go.Figure:
    fig = go.Figure()
    r = _clip(rebased.get("ratio"), None)
    f = _clip(rebased.get("ma_fast"), None)
    s = _clip(rebased.get("ma_slow"), None)
    if not r.empty:
        fig.add_trace(go.Scatter(x=r.index, y=r.values, mode="lines", name=pair_label,
                                 line=dict(color=INK, width=2.4),
                                 hovertemplate=f"{pair_label} %{{y:.1f}}<extra></extra>"))
    if not f.empty:
        fig.add_trace(go.Scatter(x=f.index, y=f.values, mode="lines", name="50D relative MA",
                                 line=dict(color=TEAL, width=2.0),
                                 hovertemplate="50D MA %{y:.1f}<extra></extra>"))
    if not s.empty:
        fig.add_trace(go.Scatter(x=s.index, y=s.values, mode="lines", name="200D relative MA",
                                 line=dict(color=STEEL, width=1.6),
                                 hovertemplate="200D MA %{y:.1f}<extra></extra>"))
    if not r.empty:
        _end_point(fig, r, f"{r.iloc[-1]:.1f}", TEAL)
        add_date_marker(fig, r, marker_ts)
        fig.add_hline(y=100, line_dash="dot", line_color="rgba(255,255,255,0.25)", line_width=1)
    _base_layout(fig, height, legend=True, top=30)
    fig.update_yaxes(side="right", title_text="Ratio (rebased = 100)")
    return fig


# ---------------------------------------------------------------------------
# 4. Valuation (Nasdaq 100 forward + trailing P/E)
# ---------------------------------------------------------------------------
def valuation_figure(
    trailing: pd.Series,
    forward: pd.Series,
    start: pd.Timestamp | None,
    trailing_avg: float | None,
    forward_avg: float | None = None,
    marker_ts: pd.Timestamp | None = None,
    height: int = 440,
) -> go.Figure:
    tr = _clip(trailing, start)
    fw = _clip(forward, start)
    fig = go.Figure()
    floor = None
    vals = [x for x in (tr.min() if not tr.empty else None, fw.min() if not fw.empty else None) if x is not None]
    if vals:
        floor = float(min(vals)) * 0.92
    if not tr.empty:
        fig.add_trace(go.Scatter(
            x=tr.index, y=tr.values, mode="lines", name="Trailing P/E (QQQ)",
            line=dict(color=GREEN, width=2.2, shape="hv"),
            fill="tozeroy" if floor is None else None,
            hovertemplate="trailing %{y:.1f}x<extra></extra>",
        ))
        if floor is not None:
            # Fill to a floor just below the range (tozeroy would flatten the line).
            fig.add_trace(go.Scatter(
                x=list(tr.index) + list(tr.index[::-1]),
                y=list(tr.values) + [floor] * len(tr),
                fill="toself", fillcolor="rgba(46,204,143,0.08)", line=dict(width=0),
                hoverinfo="skip", showlegend=False,
            ))
        if trailing_avg is not None and not np.isnan(trailing_avg):
            fig.add_hline(
                y=trailing_avg, line_dash="dash", line_color="rgba(255,255,255,0.45)", line_width=1.2,
                annotation_text=f"5Y TRAILING AVERAGE  |  {trailing_avg:.1f}x",
                annotation_position="top left", annotation_font=dict(size=11, color=MUTED),
            )
        _end_point(fig, tr, f"{tr.iloc[-1]:.1f}x", GREEN, yshift=10)
        add_date_marker(fig, tr, marker_ts)
    if not fw.empty:
        fig.add_trace(go.Scatter(
            x=fw.index, y=fw.values, mode="lines", name="Forward P/E (next 12M consensus)",
            line=dict(color=BLUE, width=2.4),
            hovertemplate="forward %{y:.1f}x<extra></extra>",
        ))
        if forward_avg is not None and not np.isnan(forward_avg):
            fig.add_trace(go.Scatter(
                x=[fw.index[0], fw.index[-1]], y=[forward_avg, forward_avg], mode="lines",
                line=dict(color=BLUE, width=1, dash="dot"), name=f"Forward avg {forward_avg:.1f}x",
                hoverinfo="skip",
            ))
        _end_point(fig, fw, f"{fw.iloc[-1]:.1f}x", BLUE, yshift=-10)
    _base_layout(fig, height, legend=True, top=30)
    fig.update_yaxes(ticksuffix="x", range=None if floor is None else [floor, None])
    return fig


# ---------------------------------------------------------------------------
# 5. Standardized signal (z-score with signed fill)
# ---------------------------------------------------------------------------
def zscore_figure(
    z: pd.Series,
    title: str,
    marker_ts: pd.Timestamp | None = None,
    height: int = 420,
) -> go.Figure:
    z = _clip(z, None)
    fig = go.Figure()
    if not z.empty:
        pos = z.clip(lower=0)
        neg = z.clip(upper=0)
        fig.add_trace(go.Scatter(x=z.index, y=pos.values, mode="lines", line=dict(width=0),
                                 fill="tozeroy", fillcolor="rgba(46,204,143,0.22)",
                                 hoverinfo="skip", showlegend=False))
        fig.add_trace(go.Scatter(x=z.index, y=neg.values, mode="lines", line=dict(width=0),
                                 fill="tozeroy", fillcolor="rgba(232,97,95,0.24)",
                                 hoverinfo="skip", showlegend=False))
        fig.add_trace(go.Scatter(x=z.index, y=z.values, mode="lines", name=title,
                                 line=dict(color=INK, width=1.9),
                                 hovertemplate="%{x|%Y-%m-%d}<br>z = %{y:+.2f}σ<extra></extra>"))
        fig.add_hline(y=0, line_color="rgba(255,255,255,0.5)", line_width=1)
        for lvl in (2, -2):
            fig.add_hline(y=lvl, line_dash="dot", line_color="rgba(255,255,255,0.25)", line_width=1,
                          annotation_text=f"{lvl:+d}σ", annotation_position="right",
                          annotation_font=dict(size=10, color=MUTED))
        color = GREEN if z.iloc[-1] >= 0 else RED
        _end_point(fig, z, f"{z.iloc[-1]:+.1f}σ", color)
        add_date_marker(fig, z, marker_ts)
        span = float(max(abs(z.min()), abs(z.max()), 2.5)) + 0.5
        fig.update_yaxes(range=[-span, span])
    _base_layout(fig, height, legend=False, top=20)
    fig.update_yaxes(title_text="z-score (σ)")
    return fig
