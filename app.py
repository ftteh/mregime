"""
Institutional Quant Regime Dashboard
------------------------------------
Daily market-position monitor combining credit/liquidity, breadth, sentiment,
positioning, and valuation into a single 0-100 composite regime score.

Run:
    streamlit run app.py
"""
from __future__ import annotations

from datetime import date, datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.config import (
    BUCKET_WEIGHTS,
    FRED_API_KEY,
    NASDAQ_DATA_LINK_API_KEY,
    INDICATORS,
    INDICATORS_BY_KEY,
    regime_label,
)
from src.indicators import (
    build_raw,
    cluster_signal,
    composite,
    latest_percentile,
    orient_score,
    score_indicators,
)


# ---------------------------------------------------------------------------
# Page config + style
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Quant Regime Dashboard",
    page_icon="■",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container { padding-top: 1.4rem; padding-bottom: 2rem; }
    .metric-big { font-size: 2.6rem; font-weight: 700; line-height: 1; }
    .sub { color: #888; font-size: 0.85rem; }
    .pill {
        display:inline-block; padding:2px 10px; border-radius: 999px;
        font-size: 0.75rem; font-weight:600; color:#fff; margin-right:6px;
    }
    .card {
        background: rgba(255,255,255,0.02);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 12px; padding: 14px 18px; margin-bottom: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Data (cached)
# ---------------------------------------------------------------------------
@st.cache_data(ttl=60 * 60, show_spinner="Pulling institutional data…")
def load_all():
    raw = build_raw()
    scores = score_indicators(raw)
    comp = composite(scores)
    cluster = cluster_signal(scores)
    return raw, scores, comp, cluster


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("Quant Regime")
    st.caption("Daily institutional market-position monitor")

    if not FRED_API_KEY:
        st.error(
            "**FRED_API_KEY missing.** Credit & Liquidity pillar (40% of composite) "
            "needs FRED data. Get a free key in 30s: "
            "[fred.stlouisfed.org/api_key](https://fred.stlouisfed.org/docs/api/api_key.html) "
            "→ put it in `.env` → restart."
        )
    if not NASDAQ_DATA_LINK_API_KEY:
        st.info(
            "**AAII sentiment** is optional: add a free Nasdaq Data Link key as "
            "`NASDAQ_DATA_LINK_API_KEY` in `.env` to unlock it. "
            "(NAAIM + F&G + VIX + SKEW already cover sentiment.)"
        )

    if st.button("Refresh data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    st.markdown("---")
    st.markdown("### Composite weights")
    for b, w in BUCKET_WEIGHTS.items():
        st.progress(w, text=f"{b.replace('_', ' ').title()} — {int(w*100)}%")

    st.markdown("---")
    st.markdown("### How to read")
    st.markdown(
        "- **>85**: Extreme complacency → de-risk, tail hedges  \n"
        "- **65–85**: Complacent → trim, tighten stops  \n"
        "- **45–65**: Neutral  \n"
        "- **15–35**: Fearful → watch for stabilization  \n"
        "- **<15**: Panic / capitulation → accumulate"
    )


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------
raw, scores, comp, cluster = load_all()

# All time-series charts share this x-axis window (aligned duration)
CHART_LOOKBACK_YEARS = 3


def _chart_date_bounds() -> tuple[pd.Timestamp, pd.Timestamp]:
    end = pd.Timestamp(datetime.now().date())
    start = end - pd.DateOffset(years=CHART_LOOKBACK_YEARS)
    return start, end


CHART_START, CHART_END = _chart_date_bounds()


def _normalize_series_index(s: pd.Series) -> pd.Series:
    out = s.copy()
    if out.index.tz is not None:
        out.index = out.index.tz_localize(None)
    return out.sort_index()


def _clip_series_to_chart_window(s: pd.Series) -> pd.Series:
    """Restrict to [CHART_START, CHART_END] for plotting only."""
    s = _normalize_series_index(s)
    return s.loc[(s.index >= CHART_START) & (s.index <= CHART_END)]


# Latest top-risk score per indicator key (0 = safer, 100 = riskier) — drives chart line colors
SCORE_BY_KEY: dict[str, float] = scores.set_index("key")["score"].to_dict()


def _top_risk_to_line_color(score: float | None) -> str:
    """Map 0–100 top-risk score to a line color: green (safe) → red (risky)."""
    if score is None or (isinstance(score, (int, float)) and np.isnan(score)):
        return "#7f8c8d"
    x = float(np.clip(float(score), 0.0, 100.0))
    xs = np.array([0.0, 15.0, 35.0, 45.0, 55.0, 65.0, 85.0, 100.0], dtype=float)
    r = np.array([0x16, 0x27, 0x34, 0x95, 0xF1, 0xE6, 0xC0, 0xC0], dtype=float)
    g = np.array([0xA0, 0xAE, 0x98, 0xA5, 0xC4, 0x7E, 0x39, 0x39], dtype=float)
    b = np.array([0x85, 0x60, 0xDB, 0xA6, 0x0F, 0x22, 0x2B, 0x2B], dtype=float)
    return (
        f"rgb({int(np.interp(x, xs, r))},"
        f"{int(np.interp(x, xs, g))},"
        f"{int(np.interp(x, xs, b))})"
    )


def _rgb_to_rgba(rgb: str, alpha: float = 0.22) -> str:
    inner = rgb.replace("rgb(", "").replace(")", "").strip()
    return f"rgba({inner},{alpha})"


def _add_chart_date_marker(
    fig: go.Figure,
    plot_s: pd.Series,
    marker_ts: pd.Timestamp | None,
) -> None:
    """Small red dot on `marker_ts` using values from the clipped series (matches the line)."""
    if marker_ts is None or plot_s.empty:
        return
    ts = pd.Timestamp(marker_ts).normalize()
    lo, hi = plot_s.index.min(), plot_s.index.max()
    if hasattr(lo, "normalize"):
        lo, hi = lo.normalize(), hi.normalize()
    if ts < lo or ts > hi:
        return
    y = plot_s.asof(ts)
    if pd.isna(y):
        return
    fig.add_trace(
        go.Scatter(
            x=[ts],
            y=[float(y)],
            mode="markers",
            marker=dict(
                size=11,
                color="#ff2d2d",
                line=dict(color="#ffffff", width=1.2),
            ),
            name="marker",
            showlegend=False,
            hovertemplate="Marker %{x|%Y-%m-%d}<br>value: %{y:.6g}<extra></extra>",
        )
    )


def _score_for_chart(indicator_key: str | None, series: pd.Series | None, extra_direction: str | None) -> float:
    """Resolve top-risk score: table row, or compute from series + direction for bonus charts."""
    if indicator_key and indicator_key in SCORE_BY_KEY:
        return float(SCORE_BY_KEY[indicator_key])
    if extra_direction and series is not None and not series.empty:
        pct = latest_percentile(series)
        return float(orient_score(pct, extra_direction))  # type: ignore[arg-type]
    return float("nan")


def _line(
    series: pd.Series | None,
    title: str,
    ref_lines: list[tuple[float, str]] | None = None,
    height: int = 240,
    *,
    indicator_key: str | None = None,
    extra_direction: str | None = None,
    marker_ts: pd.Timestamp | None = None,
) -> None:
    if series is None or series.empty:
        st.info(f"{title}: no data")
        return
    score = _score_for_chart(indicator_key, series, extra_direction)
    plot_s = _clip_series_to_chart_window(series)
    if plot_s.empty:
        st.info(f"{title}: no data in last {CHART_LOOKBACK_YEARS} years")
        return
    line_color = _top_risk_to_line_color(score if not np.isnan(score) else None)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=plot_s.index,
            y=plot_s.values,
            mode="lines",
            line=dict(width=2.8, color=line_color),
            fill="tozeroy",
            fillcolor=_rgb_to_rgba(line_color, 0.18),
            name=title,
        )
    )
    if ref_lines:
        for y, lbl in ref_lines:
            fig.add_hline(
                y=y,
                line_dash="dot",
                line_color="rgba(255,255,255,0.35)",
                annotation_text=lbl,
                annotation_position="right",
                annotation_font_size=10,
            )
    _add_chart_date_marker(fig, plot_s, marker_ts)
    subt = ""
    if not np.isnan(score):
        subt = f" — current top-risk score {score:.0f}"
    fig.update_layout(
        title=dict(text=f"{title}{subt}", font=dict(size=14)),
        height=height,
        margin=dict(l=10, r=10, t=48, b=10),
        hovermode="x unified",
        showlegend=False,
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="rgba(255,255,255,0.9)"),
        xaxis=dict(
            gridcolor="rgba(255,255,255,0.08)",
            range=[CHART_START, CHART_END],
            type="date",
        ),
        yaxis=dict(gridcolor="rgba(255,255,255,0.08)"),
    )
    st.plotly_chart(fig, use_container_width=True)


composite_score = comp["composite"]
label, emoji, color = regime_label(composite_score) if not np.isnan(composite_score) else ("NO DATA", "?", "#555")


# ---------------------------------------------------------------------------
# HEADER — the "daily glance"
# ---------------------------------------------------------------------------
st.markdown(f"## Market Regime — {datetime.now().strftime('%A, %b %d %Y')}")

hc1, hc2, hc3, hc4 = st.columns([1.4, 1, 1, 1.2])

with hc1:
    st.markdown("**Composite Regime Score**")
    score_val = composite_score if not np.isnan(composite_score) else 0
    gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score_val,
        number={"font": {"size": 56, "color": color}, "valueformat": ".1f"},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "rgba(255,255,255,0.5)"},
            "bar": {"color": "rgba(0,0,0,0)", "thickness": 0},
            "bgcolor": "rgba(0,0,0,0)",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 15],   "color": "#16a085"},
                {"range": [15, 35],  "color": "#27ae60"},
                {"range": [35, 45],  "color": "#3498db"},
                {"range": [45, 55],  "color": "#95a5a6"},
                {"range": [55, 65],  "color": "#f1c40f"},
                {"range": [65, 85],  "color": "#e67e22"},
                {"range": [85, 100], "color": "#c0392b"},
            ],
            "threshold": {
                "line": {"color": "#ffffff", "width": 7},
                "thickness": 1.0,
                "value": score_val,
            },
        },
    ))
    gauge.update_layout(
        height=260,
        margin=dict(l=10, r=10, t=30, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
    )
    st.plotly_chart(gauge, use_container_width=True)
    st.markdown(
        f"<div style='text-align:center;'><span class='pill' style='background:{color}'>{emoji}</span>"
        f"<b>{label}</b></div>",
        unsafe_allow_html=True,
    )

with hc2:
    spx = raw.series.get("spx", pd.Series(dtype=float))
    if not spx.empty:
        last = spx.iloc[-1]
        prev = spx.iloc[-2] if len(spx) > 1 else last
        chg = (last / prev - 1) * 100
        st.metric("S&P 500", f"{last:,.2f}", f"{chg:+.2f}%")
    vx = raw.series.get("vix", pd.Series(dtype=float))
    if not vx.empty:
        st.metric("VIX", f"{vx.iloc[-1]:.2f}")
    hy = raw.series.get("hy_spread", pd.Series(dtype=float))
    if not hy.empty:
        st.metric("HY Spread (bps)", f"{hy.iloc[-1]*100:.0f}")

with hc3:
    fg = raw.series.get("fear_greed", pd.Series(dtype=float))
    if not fg.empty:
        st.metric("Fear & Greed", f"{fg.iloc[-1]:.0f}")
    naaim = raw.series.get("naaim", pd.Series(dtype=float))
    if not naaim.empty:
        st.metric("NAAIM Exposure", f"{naaim.iloc[-1]:.1f}")
    aaii = raw.series.get("aaii_bull_bear", pd.Series(dtype=float))
    if not aaii.empty:
        st.metric("AAII Bull−Bear", f"{aaii.iloc[-1]:+.1f}")

with hc4:
    st.markdown("**Cluster signals**")
    tcnt = cluster["top_cluster_count"]
    bcnt = cluster["bottom_cluster_count"]
    if tcnt >= 4:
        st.error(f"TOP CLUSTER: {tcnt} indicators in extreme complacency")
    elif bcnt >= 4:
        st.success(f"BOTTOM CLUSTER: {bcnt} indicators in extreme fear")
    else:
        st.info(f"No extreme cluster ({tcnt} top / {bcnt} bottom)")
    st.caption("Funds act on 3–5+ aligned extremes, not single signals.")


# ---------------------------------------------------------------------------
# BUCKET SCORES — the four pillars
# ---------------------------------------------------------------------------
st.markdown("### The Four Pillars")
bc = st.columns(4)
pillar_order = ["credit_liquidity", "breadth_momentum", "sentiment_positioning", "valuation"]
pillar_names = {
    "credit_liquidity": "Credit & Liquidity",
    "breadth_momentum": "Breadth & Momentum",
    "sentiment_positioning": "Sentiment & Positioning",
    "valuation": "Valuation",
}
for col, bkey in zip(bc, pillar_order):
    info = comp["buckets"].get(bkey, {})
    s = info.get("score", np.nan)
    n = info.get("n", 0)
    w = int(BUCKET_WEIGHTS[bkey] * 100)
    with col:
        st.markdown(f"**{pillar_names[bkey]}** · {w}%")
        if np.isnan(s):
            st.markdown("<div class='metric-big'>—</div>", unsafe_allow_html=True)
            st.caption("no data")
        else:
            st.markdown(f"<div class='metric-big' style='color:{regime_label(s)[2]}'>{s:.0f}</div>", unsafe_allow_html=True)
            st.caption(f"{n} indicators · {regime_label(s)[0].split('—')[0].strip()}")


# ---------------------------------------------------------------------------
# INDICATOR TABLE
# ---------------------------------------------------------------------------
st.markdown("### Indicator breakdown")
tbl = scores.copy()
tbl["raw"] = tbl["raw"].apply(lambda x: f"{x:,.2f}" if pd.notna(x) else "—")
tbl["percentile"] = tbl["percentile"].round(1)
tbl["score"] = tbl["score"].round(1)
tbl["bucket"] = tbl["bucket"].map(pillar_names)
tbl["as_of"] = pd.to_datetime(tbl["as_of"], errors="coerce").dt.strftime("%Y-%m-%d").fillna("—")


def _highlight_score(v):
    if not isinstance(v, (int, float)) or pd.isna(v):
        return ""
    if v >= 85: return "background-color: #c0392b; color:white;"
    if v >= 65: return "background-color: #e67e22; color:white;"
    if v >= 55: return "background-color: #f1c40f;"
    if v >= 45: return "background-color: #95a5a6;"
    if v >= 35: return "background-color: #3498db; color:white;"
    if v >= 15: return "background-color: #27ae60; color:white;"
    return "background-color: #16a085; color:white;"


display_cols = ["label", "bucket", "raw", "percentile", "score", "as_of", "n_obs"]
styled = tbl[display_cols].rename(columns={
    "label": "Indicator",
    "bucket": "Pillar",
    "raw": "Latest",
    "percentile": "3Y %ile",
    "score": "Top-risk score",
    "as_of": "As of",
    "n_obs": "Obs",
}).style.map(_highlight_score, subset=["Top-risk score"])

st.dataframe(styled, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# CHART GRID — the key time series
# ---------------------------------------------------------------------------
st.markdown("### Time-series")
st.caption(
    f"All charts use the **same {CHART_LOOKBACK_YEARS}-year window** "
    f"({CHART_START.strftime('%Y-%m-%d')} → {CHART_END.strftime('%Y-%m-%d')}). "
    "Line color = **top-risk score**: **green** = safer, **red** = riskier (indicator table scale)."
)

if "ts_marker_date" not in st.session_state:
    st.session_state.ts_marker_date = None

_m1, _m2, _m3, _m4 = st.columns([1.15, 0.85, 0.85, 2.2])
with _m1:
    _pick = st.date_input(
        "Highlight date",
        value=date.today(),
        min_value=CHART_START.date(),
        max_value=CHART_END.date(),
        key="ts_marker_date_picker",
    )
with _m2:
    if st.button("Set marker", key="ts_marker_set", use_container_width=True, help="Place a red dot on all charts at this date"):
        st.session_state.ts_marker_date = pd.Timestamp(_pick).normalize()
with _m3:
    if st.button("Clear marker", key="ts_marker_clear", use_container_width=True):
        st.session_state.ts_marker_date = None
with _m4:
    _md = st.session_state.ts_marker_date
    if _md is not None:
        st.success(f"Marker: **{_md.strftime('%Y-%m-%d')}** — red dot on each series (change date + Set again to move)")
    else:
        st.caption("No marker — choose a date and click **Set marker**.")

ts_marker: pd.Timestamp | None = st.session_state.ts_marker_date

rc1, rc2 = st.columns(2)
with rc1:
    _line(
        raw.series.get("hy_spread"),
        "HY Credit Spread (%)",
        ref_lines=[(4.0, "tight → top risk"), (8.0, "stress")],
        indicator_key="hy_spread",
        marker_ts=ts_marker,
    )
    _line(
        raw.series.get("vix"),
        "VIX",
        ref_lines=[(13, "complacency"), (30, "fear")],
        indicator_key="vix",
        marker_ts=ts_marker,
    )
    _line(
        raw.series.get("fear_greed"),
        "CNN Fear & Greed",
        ref_lines=[(25, "extreme fear"), (75, "extreme greed")],
        indicator_key="fear_greed",
        marker_ts=ts_marker,
    )
    _line(
        raw.series.get("aaii_bull_bear"),
        "AAII Bull−Bear Spread (%)",
        ref_lines=[(-20, "panic"), (20, "euphoria")],
        indicator_key="aaii_bull_bear",
        marker_ts=ts_marker,
    )
    _line(
        raw.series.get("move_vix_div"),
        "MOVE / VIX  (bond-vol vs equity-vol)",
        ref_lines=[(6, "bonds 'lying' — watch")],
        extra_direction="risk_high_is_top",
        marker_ts=ts_marker,
    )

with rc2:
    nl = raw.series.get("net_liquidity")
    _line(
        nl,
        "Fed Net Liquidity (WALCL − TGA − RRP, $B)",
        indicator_key="net_liquidity",
        marker_ts=ts_marker,
    )
    _line(
        raw.series.get("pct_above_200dma"),
        "% SP500 above 200DMA",
        ref_lines=[(20, "washout"), (80, "extended")],
        indicator_key="pct_above_200dma",
        marker_ts=ts_marker,
    )
    _line(
        raw.series.get("naaim"),
        "NAAIM Exposure",
        ref_lines=[(30, "defensive"), (100, "leveraged")],
        indicator_key="naaim",
        marker_ts=ts_marker,
    )
    _line(
        raw.series.get("skew"),
        "CBOE SKEW",
        ref_lines=[(130, "tail hedging elevated")],
        indicator_key="skew",
        marker_ts=ts_marker,
    )
    _line(
        raw.series.get("corr_cluster"),
        "SPY vs (TLT+GLD)/2  20D correlation",
        ref_lines=[(0.6, "liquidity event")],
        extra_direction="risk_high_is_top",
        marker_ts=ts_marker,
    )


# ---------------------------------------------------------------------------
# WHAT PROS WATCH — the divergences
# ---------------------------------------------------------------------------
st.markdown("### Pro Watchlist — divergences & confluence")
mv = raw.series.get("move_vix_div", pd.Series(dtype=float))
cc = raw.series.get("corr_cluster", pd.Series(dtype=float))

dc1, dc2, dc3 = st.columns(3)
with dc1:
    if not mv.empty:
        z = (mv.iloc[-1] - mv.rolling(252).mean().iloc[-1]) / mv.rolling(252).std().iloc[-1]
        if z > 1.5:
            st.warning(f"MOVE/VIX z-score = {z:+.2f} — bond market stressed while equity calm (equity is lying).")
        elif z < -1.0:
            st.info(f"MOVE/VIX z-score = {z:+.2f} — bond vol suppressed vs equity.")
        else:
            st.success(f"MOVE/VIX z-score = {z:+.2f} — bond vs equity vol aligned.")

with dc2:
    if not cc.empty:
        last = cc.iloc[-1]
        if last > 0.6:
            st.warning(f"SPY-hedge correlation = {last:.2f} — liquidity event / everything selling together.")
        elif last < -0.3:
            st.success(f"SPY-hedge correlation = {last:.2f} — healthy risk-on / risk-off separation.")
        else:
            st.info(f"SPY-hedge correlation = {last:.2f} — normal regime.")

with dc3:
    dix = raw.series.get("dix", pd.Series(dtype=float))
    if not dix.empty:
        last = dix.iloc[-1]
        avg = dix.tail(60).mean()
        if last > avg + 1.0:
            st.success(f"DIX {last:.1f} > 60D avg {avg:.1f} — hidden institutional buying.")
        elif last < avg - 1.0:
            st.warning(f"DIX {last:.1f} < 60D avg {avg:.1f} — dark pool selling pressure.")
        else:
            st.info(f"DIX {last:.1f} ≈ 60D avg {avg:.1f} — neutral dark-pool flow.")
    else:
        st.caption("DIX: source unavailable (squeezemetrics). Optional indicator.")


# ---------------------------------------------------------------------------
# CLUSTER DETAIL
# ---------------------------------------------------------------------------
with st.expander("Why is the composite where it is?  (extreme-cluster detail)"):
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Top-risk extremes (score ≥ 85)**")
        if cluster["top_names"]:
            for n in cluster["top_names"]:
                st.markdown(f"- {n}")
        else:
            st.caption("None.")
    with c2:
        st.markdown("**Bottom extremes (score ≤ 15)**")
        if cluster["bottom_names"]:
            for n in cluster["bottom_names"]:
                st.markdown(f"- {n}")
        else:
            st.caption("None.")


# ---------------------------------------------------------------------------
# SOURCES
# ---------------------------------------------------------------------------
with st.expander("Data sources & methodology"):
    src_rows = [
        {"Indicator": i.label, "Bucket": i.bucket, "Source": i.source,
         "Direction": i.direction, "Note": i.description}
        for i in INDICATORS
    ]
    st.dataframe(pd.DataFrame(src_rows), use_container_width=True, hide_index=True)
    st.markdown(
        "**Scoring**: every indicator is converted to a 3-year rolling percentile, "
        "oriented so 100 = complacency / top-risk and 0 = panic / bottom-setup. "
        "Bucket scores are equal-weighted within bucket; composite is a weighted "
        "average across the 4 buckets using the weights shown in the sidebar."
    )


st.caption(
    "Not investment advice. Use cluster confluence (3–5 aligned extremes) rather "
    "than any single indicator. Tops are processes; bottoms are events."
)
