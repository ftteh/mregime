"""
Plotly figures for the research desk.

Pure builders (no Streamlit): data in, `go.Figure` out, so every figure can be
built in tests. Two measures with different scales get stacked panels rather than
a secondary y-axis: a dual axis invites reading meaning into where two arbitrary
scales happen to cross. Series colors follow a fixed order so an entity keeps its
color when a sibling disappears.
"""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src import research as R
from src.charts import BLUE, GOLD, GREEN, GRID, GREEN_ZONE, INK, MUTED, RED, RED_ZONE, _base_layout, _clip, _end_point

# Fixed categorical order for multi-series panels (validated on the dark surface).
SERIES = ["#3f8fd6", "#b28b3a", "#27a08d", "#a56fe0", "#d0664f"]
NEUTRAL = "rgba(255,255,255,0.45)"


def _fmt_axis(fig: go.Figure) -> None:
    fig.update_xaxes(gridcolor=GRID, showspikes=False)
    fig.update_yaxes(gridcolor=GRID, zeroline=False)


def _line(fig: go.Figure, s: pd.Series, name: str, color: str, *, row: int | None = None, width: float = 2.0, dash: str | None = None, fill: str | None = None, fillcolor: str | None = None) -> None:
    kw = {} if row is None else {"row": row, "col": 1}
    fig.add_trace(
        go.Scatter(x=s.index, y=s.values, mode="lines", name=name, line=dict(color=color, width=width, dash=dash),
                   fill=fill, fillcolor=fillcolor),
        **kw,
    )


def _hline(fig: go.Figure, y: float, *, text: str = "", color: str = NEUTRAL, row: int | None = None, dash: str = "dash") -> None:
    kw = {} if row is None else {"row": row, "col": 1}
    fig.add_hline(y=y, line=dict(color=color, width=1, dash=dash), annotation_text=text,
                  annotation_position="top left", annotation_font=dict(size=10, color=MUTED), **kw)


def _zone(fig: go.Figure, y0: float, y1: float, color: str, row: int | None = None) -> None:
    kw = {} if row is None else {"row": row, "col": 1}
    fig.add_hrect(y0=y0, y1=y1, fillcolor=color, line_width=0, layer="below", **kw)


def stacked(
    top: list[tuple[str, pd.Series]],
    bottom: list[tuple[str, pd.Series]],
    *,
    start: pd.Timestamp | None = None,
    height: int = 460,
    top_title: str = "",
    bottom_title: str = "",
    bottom_lines: list[tuple[float, str]] | None = None,
    bottom_zones: list[tuple[float, float, str]] | None = None,
    bottom_range: tuple[float, float] | None = None,
    log_top: bool = False,
    fmt: str = "{:,.2f}",
    bottom_fmt: str = "{:.1f}",
) -> go.Figure:
    """Context on top, the signal underneath, on one shared time axis."""
    top_title = top_title or (top[0][0] if len(top) == 1 else "")
    bottom_title = bottom_title or (bottom[0][0] if len(bottom) == 1 else "")
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.09, row_heights=[0.58, 0.42],
                        subplot_titles=(top_title, bottom_title))
    for i, (name, s) in enumerate(top):
        c = _clip(s, start)
        _line(fig, c, name, SERIES[i % len(SERIES)] if len(top) > 1 else INK, row=1, width=2.0 if i == 0 else 1.4)
        if i == 0:
            _end_point(fig, c, fmt.format(c.iloc[-1]) if not c.empty else "", INK, row=1, col=1)
    for i, (name, s) in enumerate(bottom):
        c = _clip(s, start)
        col = SERIES[(i + 2) % len(SERIES)]
        _line(fig, c, name, col, row=2)
        if i == 0:
            _end_point(fig, c, bottom_fmt.format(c.iloc[-1]) if not c.empty else "", col, row=2, col=1)
    for y0, y1, col in bottom_zones or []:
        _zone(fig, y0, y1, col, row=2)
    for y, text in bottom_lines or []:
        _hline(fig, y, text=text, row=2)
    _base_layout(fig, height, legend=len(top) + len(bottom) > 2, top=46)
    if log_top:
        fig.update_yaxes(type="log", row=1, col=1)
    if bottom_range:
        fig.update_yaxes(range=list(bottom_range), row=2, col=1)
    fig.update_annotations(font=dict(size=12, color=MUTED))
    return fig


def lines(
    series: list[tuple[str, pd.Series]],
    *,
    start: pd.Timestamp | None = None,
    height: int = 320,
    levels: list[tuple[float, str]] | None = None,
    zones: list[tuple[float, float, str]] | None = None,
    fmt: str = "{:.2f}",
    fill_first: bool = False,
    title: str = "",
) -> go.Figure:
    fig = go.Figure()
    for i, (name, s) in enumerate(series):
        c = _clip(s, start)
        col = SERIES[i % len(SERIES)]
        _line(fig, c, name, col, fill="tozeroy" if fill_first and i == 0 else None,
              fillcolor="rgba(93,169,233,0.10)" if fill_first and i == 0 else None)
        if not c.empty:
            _end_point(fig, c, fmt.format(c.iloc[-1]), col)
    for y0, y1, col in zones or []:
        _zone(fig, y0, y1, col)
    for y, text in levels or []:
        _hline(fig, y, text=text)
    _base_layout(fig, height, legend=len(series) >= 2, top=46 if title else 30)
    if title:
        fig.update_layout(title=dict(text=title, font=dict(size=13, color=MUTED), x=0, xanchor="left"))
    return fig


def signed_bars(
    s: pd.Series,
    *,
    start: pd.Timestamp | None = None,
    height: int = 300,
    levels: list[tuple[float, str]] | None = None,
    title: str = "",
    highlight_last: bool = True,
) -> go.Figure:
    c = _clip(s, start)
    colors = [GREEN if v >= 0 else RED for v in c.values]
    fig = go.Figure(go.Bar(x=c.index, y=c.values, marker_color=colors, marker_line_width=0, name="",
                           hovertemplate="%{x|%b %Y}: %{y:+.2f}<extra></extra>"))
    if highlight_last and not c.empty:
        _end_point(fig, c, f"{c.iloc[-1]:+.2f}", INK)
    for y, text in levels or []:
        _hline(fig, y, text=text)
    _base_layout(fig, height, top=46 if title else 30)
    fig.update_layout(bargap=0.15)
    if title:
        fig.update_layout(title=dict(text=title, font=dict(size=13, color=MUTED), x=0, xanchor="left"))
    return fig


def hbars(df: pd.DataFrame, label: str, value: str, *, height: int | None = None, title: str = "", suffix: str = "%", ref: float | None = None, ref_label: str = "") -> go.Figure:
    """Ranked horizontal bars (first row of `df` drawn at the top)."""
    d = df.iloc[::-1]
    colors = [GREEN if v >= 0 else RED for v in d[value]]
    fig = go.Figure(go.Bar(
        x=d[value], y=d[label], orientation="h", marker_color=colors, marker_line_width=0,
        text=[f"{v:+.1f}{suffix}" for v in d[value]], textposition="outside", textfont=dict(size=11, color=INK),
        cliponaxis=False, hovertemplate="%{y}: %{x:+.2f}" + suffix + "<extra></extra>",
    ))
    if ref is not None:
        fig.add_vline(x=ref, line=dict(color=GOLD, width=1, dash="dash"), annotation_text=ref_label,
                      annotation_position="top", annotation_font=dict(size=10, color=MUTED))
    _base_layout(fig, height or max(260, 24 * len(d) + 70), top=40 if title else 16)
    fig.update_layout(hovermode="closest", margin=dict(l=10, r=64, t=40 if title else 16, b=10), bargap=0.28)
    fig.update_xaxes(type="linear", gridcolor=GRID, zeroline=True, zerolinecolor=NEUTRAL)
    fig.update_yaxes(type="category", showgrid=False)
    if title:
        fig.update_layout(title=dict(text=title, font=dict(size=13, color=MUTED), x=0, xanchor="left"))
    return fig


def crowding_dumbbell(table: pd.DataFrame) -> go.Figure:
    """All-time vs 3-year positioning percentile for each market.

    A long connector is the informative case: the market looks extreme against
    one horizon but not the other, which usually means a structural position.
    """
    d = table.iloc[::-1]
    fig = go.Figure()
    fig.add_vrect(x0=0, x1=10, fillcolor=GREEN_ZONE, line_width=0, layer="below")
    fig.add_vrect(x0=90, x1=100, fillcolor=RED_ZONE, line_width=0, layer="below")
    for _, r in d.iterrows():
        fig.add_trace(go.Scatter(x=[r["pct_all"], r["pct_3y"]], y=[r["market"]] * 2, mode="lines",
                                 line=dict(color=NEUTRAL, width=2), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=d["pct_all"], y=d["market"], mode="markers", name="All history (since 2006)",
                             marker=dict(size=12, color=BLUE, line=dict(color="#0e1117", width=2)),
                             hovertemplate="%{y}: %{x:.0f}th pct, all history<extra></extra>"))
    fig.add_trace(go.Scatter(x=d["pct_3y"], y=d["market"], mode="markers", name="Last 3 years",
                             marker=dict(size=12, color=GOLD, symbol="diamond", line=dict(color="#0e1117", width=2)),
                             hovertemplate="%{y}: %{x:.0f}th pct, last 3 years<extra></extra>"))
    _base_layout(fig, 72 + 34 * len(d), legend=True, top=34)
    fig.update_layout(hovermode="closest", margin=dict(l=10, r=20, t=34, b=10))
    fig.update_xaxes(type="linear", range=[-3, 103], title_text="Net position percentile", gridcolor=GRID)
    fig.update_yaxes(type="category", showgrid=False)
    return fig


def cot_figure(price: pd.Series, net_oi: pd.Series, name: str, start: pd.Timestamp | None = None) -> go.Figure:
    rank = R.expanding_rank(net_oi, 104)
    return stacked(
        [(name, price)], [("Positioning percentile", rank)], start=start, height=420,
        top_title=f"{name}: underlying price", bottom_title="Net position percentile, expanding history",
        bottom_zones=[(0, 10, GREEN_ZONE), (90, 100, RED_ZONE)], bottom_range=(0, 100), bottom_fmt="{:.0f}",
    )


def divergence_scatter(internals: pd.DataFrame, index_px: pd.Series, *, min_adv: int = 300) -> go.Figure:
    """Each dot is a session: advancing constituents vs the index's return."""
    ret = R.clean(index_px).pct_change(fill_method=None) * 100
    df = internals[["adv"]].join(ret.rename("ret"), how="inner").dropna()
    df = df[df["ret"] < 0]
    hit = df[df["adv"] >= min_adv]
    rest = df[df["adv"] < min_adv]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=rest["adv"], y=rest["ret"], mode="markers", name="Down days",
                             marker=dict(size=7, color=BLUE, opacity=0.45, line=dict(width=0)),
                             hovertemplate="%{customdata}<br>%{x} advancers, index %{y:.2f}%<extra></extra>",
                             customdata=rest.index.strftime("%Y-%m-%d")))
    fig.add_trace(go.Scatter(x=hit["adv"], y=hit["ret"], mode="markers", name=f"{min_adv}+ advancers",
                             marker=dict(size=11, color=GOLD, line=dict(color="#0e1117", width=2)),
                             hovertemplate="%{customdata}<br>%{x} advancers, index %{y:.2f}%<extra></extra>",
                             customdata=hit.index.strftime("%Y-%m-%d")))
    _base_layout(fig, 340, legend=True, top=34)
    fig.update_layout(hovermode="closest")
    fig.update_xaxes(type="linear", title_text="Advancing S&P 500 members", gridcolor=GRID)
    fig.update_yaxes(title_text="Index return, %", gridcolor=GRID)
    return fig
