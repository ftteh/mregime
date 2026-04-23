"""
Institutional Quant Regime Dashboard
------------------------------------
Daily market-position monitor combining credit/liquidity, breadth, sentiment,
positioning, and valuation into a single 0-100 composite regime score.

Run:
    streamlit run app.py
"""
from __future__ import annotations

import os
from datetime import date, datetime

import streamlit as st

# Must be the first Streamlit command (required by Streamlit).
st.set_page_config(
    page_title="Quant Regime Dashboard",
    page_icon="■",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Streamlit Community Cloud: paste keys in the dashboard "Secrets" UI — copy here
# so src.config sees them via os.environ (local dev still uses `.env`).
try:
    for _secret in ("FRED_API_KEY", "NASDAQ_DATA_LINK_API_KEY"):
        if _secret in st.secrets:
            os.environ.setdefault(_secret, str(st.secrets[_secret]))
except Exception:
    pass

import numpy as np
import pandas as pd
import plotly.graph_objects as go

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
    exposure_recommendation,
    historical_composite,
    latest_percentile,
    orient_score,
    pillar_momentum,
    score_indicators,
)


# ---------------------------------------------------------------------------
# Global CSS
# ---------------------------------------------------------------------------
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
    .mom-label {
        font-size: 1rem; color: #ccc; margin-bottom: 2px;
    }
    .mom-today {
        font-size: 2rem; font-weight: 700; color: #ffffff; line-height: 1;
        margin-right: 6px;
    }
    .mom-pill {
        display: inline-block;
        padding: 6px 14px;
        border-radius: 999px;
        font-size: 1rem;
        font-weight: 700;
        color: #ffffff;
        margin-right: 8px;
        margin-top: 6px;
        letter-spacing: 0.2px;
        box-shadow: 0 1px 2px rgba(0,0,0,0.35);
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
    momentum = pillar_momentum(raw)
    comp_hist = historical_composite(raw)
    return raw, scores, comp, cluster, momentum, comp_hist


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

    if st.button("Refresh data", width="stretch"):
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
raw, scores, comp, cluster, momentum, comp_hist = load_all()

# All time-series charts share this x-axis window (aligned duration)
CHART_LOOKBACK_YEARS = 3


def _chart_date_bounds() -> tuple[pd.Timestamp, pd.Timestamp]:
    end = pd.Timestamp(datetime.now().date())
    start = end - pd.DateOffset(years=CHART_LOOKBACK_YEARS)
    return start, end


CHART_START, CHART_END = _chart_date_bounds()


def _normalize_series_index(s: pd.Series) -> pd.Series:
    """
    Return a copy of `s` with a clean, tz-naive, monotonically-sorted
    DatetimeIndex. Defensive: some upstream fetchers occasionally emit a
    plain `Index` (string/int) when a source returns an unexpected shape —
    we coerce rather than crash the dashboard.
    """
    out = s.copy()
    # Coerce non-datetime indices (e.g. empty Index, object dtype from a
    # scrape fallback) into DatetimeIndex; drop rows that don't parse.
    if not isinstance(out.index, pd.DatetimeIndex):
        try:
            out.index = pd.to_datetime(out.index, errors="coerce")
            out = out[out.index.notna()]
        except Exception:
            return pd.Series(dtype=out.dtype if hasattr(out, "dtype") else float)
    if out.empty or not isinstance(out.index, pd.DatetimeIndex):
        return out
    # Strip timezone
    if getattr(out.index, "tz", None) is not None:
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
    st.plotly_chart(fig, width="stretch")


# ---------------------------------------------------------------------------
# Header metric tiles with sparklines
# ---------------------------------------------------------------------------
def _sparkline(series: pd.Series, *, bad_direction: str = "up", height: int = 38) -> go.Figure | None:
    """Tiny trend chart — 60-day history, no axes, color by net change over window."""
    if series is None or series.empty or len(series) < 2:
        return None
    s = _normalize_series_index(series).dropna().tail(60)
    if s.empty:
        return None
    chg = s.iloc[-1] - s.iloc[0]
    if bad_direction == "up":
        rgb = (255, 107, 107) if chg > 0 else (79, 201, 120)
    else:
        rgb = (79, 201, 120) if chg > 0 else (255, 107, 107)
    line_color = f"rgb({rgb[0]},{rgb[1]},{rgb[2]})"
    fill_color = f"rgba({rgb[0]},{rgb[1]},{rgb[2]},0.20)"
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=s.index,
            y=s.values,
            mode="lines",
            line=dict(width=1.6, color=line_color),
            fill="tozeroy",
            fillcolor=fill_color,
            showlegend=False,
            hovertemplate="%{x|%Y-%m-%d}<br>%{y:.2f}<extra></extra>",
        )
    )
    fig.update_layout(
        height=height,
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(visible=False, showgrid=False, showticklabels=False),
        yaxis=dict(visible=False, showgrid=False, showticklabels=False, range=[s.min() - (s.max() - s.min()) * 0.05, s.max() + (s.max() - s.min()) * 0.05] if s.max() != s.min() else None),
    )
    return fig


def _metric_tile(
    label: str,
    series: pd.Series | None,
    *,
    value_fmt: str = "{:,.2f}",
    delta_mode: str = "diff",         # "diff" | "pct" | "none"
    delta_fmt: str | None = None,
    bad_direction: str = "up",         # "up" = rising is bad (VIX, HY); "down" = falling is bad (SPX, F&G)
) -> None:
    """Header metric: label + value + 1-day delta + 60-day sparkline."""
    if series is None or series.empty:
        st.caption(f"{label}: no data")
        return
    last = float(series.iloc[-1])
    prev = float(series.iloc[-2]) if len(series) > 1 else last
    delta_str: str | None = None
    if delta_mode == "diff":
        d = last - prev
        delta_str = (delta_fmt or "{:+.2f}").format(d)
    elif delta_mode == "pct":
        d = (last / prev - 1) * 100 if prev else 0.0
        delta_str = (delta_fmt or "{:+.2f}%").format(d)
    delta_color = "normal"
    if delta_mode != "none" and delta_str is not None:
        # st.metric: "inverse" flips the red/green; we want red/green to map to GOOD vs BAD
        delta_color = "inverse" if bad_direction == "up" else "normal"
    st.metric(label, value_fmt.format(last), delta=delta_str, delta_color=delta_color)
    spark = _sparkline(series, bad_direction=bad_direction, height=36)
    if spark is not None:
        st.plotly_chart(spark, width="stretch", config={"displayModeBar": False})


composite_score = comp["composite"]
label, emoji, color = regime_label(composite_score) if not np.isnan(composite_score) else ("NO DATA", "?", "#555")

# Actionable: map composite + cluster → suggested net exposure + hedge
exp_rec = exposure_recommendation(composite_score, cluster)

# Trend: what was our suggested exposure ~5 business days ago?
prev_exp = None
prev_score = np.nan
try:
    if comp_hist is not None and not comp_hist.empty and len(comp_hist) >= 6:
        prev_score = float(comp_hist.iloc[-6])  # ~1 week back (5 business days)
        if not np.isnan(prev_score):
            prev_exp = exposure_recommendation(prev_score, cluster=None)
except Exception:
    prev_exp = None


# ---------------------------------------------------------------------------
# HEADER — the "daily glance"
# ---------------------------------------------------------------------------
st.markdown(f"## Market Regime — {datetime.now().strftime('%A, %b %d %Y')}")

hc1, hc_exp, hc2, hc3, hc4 = st.columns([1.25, 1.4, 0.95, 0.95, 1.0])

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
    st.plotly_chart(gauge, width="stretch")
    st.markdown(
        f"<div style='text-align:center;'><span class='pill' style='background:{color}'>{emoji}</span>"
        f"<b>{label}</b></div>",
        unsafe_allow_html=True,
    )

with hc_exp:
    st.markdown("**Recommended Exposure**")
    if exp_rec["net_pct"] is None:
        st.markdown(
            "<div style='text-align:center; color:#888; padding:40px 0;'>No data</div>",
            unsafe_allow_html=True,
        )
    else:
        net_pct = exp_rec["net_pct"]
        hedge_pct = exp_rec["hedge_pct"]
        exp_color = exp_rec["color"]
        exp_label = exp_rec["label"]
        conviction = exp_rec["conviction"]
        override = exp_rec["cluster_override"]

        # Hedge text
        hedge_txt = "None" if hedge_pct <= 0 else f"{hedge_pct:.1f}% OTM puts"
        hedge_color = "#888" if hedge_pct <= 0 else "#e67e22"

        # Trend vs 1 week ago
        trend_html = ""
        if prev_exp is not None and prev_exp["net_pct"] is not None:
            delta = net_pct - prev_exp["net_pct"]
            if delta == 0:
                trend_html = f"Unchanged vs 1w ago ({prev_exp['net_pct']}%)"
            elif delta > 0:
                trend_html = f"<span style='color:#4fc978;'>▲ +{delta}pp</span> vs 1w ago (was {prev_exp['net_pct']}%)"
            else:
                trend_html = f"<span style='color:#ff6b6b;'>▼ {delta}pp</span> vs 1w ago (was {prev_exp['net_pct']}%)"
        elif not np.isnan(prev_score):
            trend_html = f"1w-ago composite: {prev_score:.0f}"

        override_badge = (
            "<div style='display:inline-block; background:#c0392b; color:#fff; "
            "font-size:0.7rem; font-weight:700; padding:3px 8px; border-radius:4px; "
            "margin-bottom:8px; letter-spacing:0.5px;'>⚠ CLUSTER OVERRIDE</div>"
            if override else ""
        )

        # Pull rgb for soft glow
        r = int(exp_color[1:3], 16)
        g = int(exp_color[3:5], 16)
        b = int(exp_color[5:7], 16)

        st.markdown(
            f"""
<div style='background: linear-gradient(180deg, #1a1d29 0%, #121520 100%);
            border-radius: 12px;
            padding: 16px 14px;
            border: 2px solid {exp_color};
            text-align: center;
            box-shadow: 0 0 24px rgba({r},{g},{b},0.25);
            min-height: 296px;'>
  {override_badge}
  <div style='font-size: 2.9rem; font-weight: 800; color: {exp_color}; line-height: 1; margin: 4px 0 4px 0; white-space: nowrap;'>
    {net_pct}%
  </div>
  <div style='font-size: 0.72rem; color: #aaa; letter-spacing: 1px; margin-bottom: 10px;'>
    NET EQUITY EXPOSURE
  </div>
  <div style='font-size: 0.95rem; color: #fff; font-weight: 700; margin-bottom: 12px; line-height: 1.2;'>
    {exp_label}
  </div>
  <div style='font-size: 0.8rem; color: #ccc; line-height: 1.6; text-align: left; padding: 0 4px;'>
    <div>Hedge: <b style='color:{hedge_color};'>{hedge_txt}</b></div>
    <div>Conviction: <b>{conviction}</b></div>
    <div style='margin-top: 6px; color: #888; font-size: 0.74rem; line-height: 1.4;'>{trend_html}</div>
  </div>
</div>
""",
            unsafe_allow_html=True,
        )

with hc2:
    _metric_tile(
        "S&P 500",
        raw.series.get("spx"),
        value_fmt="{:,.2f}",
        delta_mode="pct",
        bad_direction="down",
    )
    _metric_tile(
        "VIX",
        raw.series.get("vix"),
        value_fmt="{:.2f}",
        delta_mode="diff",
        bad_direction="up",
    )
    # HY spread is in percent on FRED → render as bps
    hy = raw.series.get("hy_spread", pd.Series(dtype=float))
    hy_bps = (hy * 100.0) if not hy.empty else hy
    _metric_tile(
        "HY Spread (bps)",
        hy_bps,
        value_fmt="{:,.0f}",
        delta_mode="diff",
        delta_fmt="{:+.0f}",
        bad_direction="up",
    )

with hc3:
    _metric_tile(
        "Fear & Greed",
        raw.series.get("fear_greed"),
        value_fmt="{:.0f}",
        delta_mode="diff",
        delta_fmt="{:+.0f}",
        bad_direction="down",
    )
    _metric_tile(
        "NAAIM Exposure",
        raw.series.get("naaim"),
        value_fmt="{:.1f}",
        delta_mode="diff",
        delta_fmt="{:+.1f}",
        bad_direction="up",
    )
    _metric_tile(
        "AAII Bull−Bear",
        raw.series.get("aaii_bull_bear"),
        value_fmt="{:+.1f}",
        delta_mode="diff",
        delta_fmt="{:+.1f}",
        bad_direction="up",
    )

with hc4:
    st.markdown("**Cluster signals**")
    tcnt = cluster["top_cluster_count"]
    bcnt = cluster["bottom_cluster_count"]
    if tcnt >= 4:
        st.error(f"TOP CLUSTER · {tcnt} ≥85")
    elif bcnt >= 4:
        st.success(f"BOTTOM CLUSTER · {bcnt} ≤15")
    else:
        st.info(f"No cluster · {tcnt}↑ / {bcnt}↓")
    st.caption(f"Indicators at extreme: **{tcnt}** top-risk, **{bcnt}** panic. Clusters of 4+ are actionable.")


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
# PILLAR MOMENTUM — rate-of-change per pillar (1w / 1m)
# ---------------------------------------------------------------------------
st.markdown("### Pillar momentum — is the regime shifting?")
st.caption(
    "Δ = change in pillar score. **Positive Δ = moving toward top-risk/complacency** "
    "(sell pressure building). **Negative Δ = moving toward panic/bottom** "
    "(buy setup forming). |Δ| ≥ 10 over 1w = deteriorating/improving fast."
)

def _momentum_color_and_tag(delta: float, window: str) -> tuple[str, str]:
    if np.isnan(delta):
        return ("#7f8c8d", "—")
    if delta >= 10:
        return ("#ff4d4d", f"deteriorating fast ({window})")   # bright red
    if delta >= 5:
        return ("#ff9f1a", f"drifting toward top ({window})")  # bright orange
    if delta <= -10:
        return ("#00c08a", f"improving fast ({window})")       # vivid green
    if delta <= -5:
        return ("#4fc978", f"drifting toward bottom ({window})")
    return ("#6b7a85", f"stable ({window})")

mc = st.columns(4)
for col, bkey in zip(mc, pillar_order):
    m = momentum.get(bkey, {}) if isinstance(momentum, dict) else {}
    today = m.get("today", np.nan)
    d1w = m.get("d_1w", np.nan)
    d1m = m.get("d_1m", np.nan)
    with col:
        st.markdown(f"**{pillar_names[bkey]}**")
        if isinstance(today, float) and np.isnan(today):
            st.caption("no history yet")
            continue
        c1w, _ = _momentum_color_and_tag(d1w, "1w")
        c1m, _ = _momentum_color_and_tag(d1m, "1m")
        arr1w = "▲" if (not np.isnan(d1w) and d1w > 0) else ("▼" if (not np.isnan(d1w) and d1w < 0) else "—")
        arr1m = "▲" if (not np.isnan(d1m) and d1m > 0) else ("▼" if (not np.isnan(d1m) and d1m < 0) else "—")
        d1w_s = "—" if np.isnan(d1w) else f"{d1w:+.1f}"
        d1m_s = "—" if np.isnan(d1m) else f"{d1m:+.1f}"
        st.markdown(
            f"<div class='mom-label'>today <span class='mom-today'>{today:.0f}</span></div>"
            f"<div>"
            f"<span class='mom-pill' style='background:{c1w}'>1w {arr1w} {d1w_s}</span>"
            f"<span class='mom-pill' style='background:{c1m}'>1m {arr1m} {d1m_s}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# INDICATOR TABLE
# ---------------------------------------------------------------------------
st.markdown("### Indicator breakdown")

# Unit + precision per indicator for the "Latest" column
INDICATOR_UNITS: dict[str, str] = {
    "hy_spread": "%",
    "ig_spread": "%",
    "move_index": "pts",
    "net_liquidity": "$B",
    "financial_conditions": "σ",
    "pct_above_200dma": "%",
    "ad_line_slope": "",
    "rsi_spx": "",
    "new_highs_lows": "",
    "aaii_bull_bear": "pp",
    "naaim": "",
    "fear_greed": "/100",
    "put_call": "",
    "vix": "",
    "vvix": "",
    "skew": "",
    "equity_risk_premium": "%",
    "cta_positioning": "%",
    "vix_term_9d_1m": "x",
    "vix_term_1m_3m": "x",
    "curve_2s10s": "pp",
    "curve_3m10y": "pp",
    "curve_resteep_2s10s": "pp",
    "hy_spread_velocity": "bps",
    "dxy": "",
    "real_yield_10y": "%",
    "copper_gold": "x",
}


def _fmt_raw_cell(row: pd.Series) -> str:
    v = row.get("raw")
    if pd.isna(v):
        return "—"
    key = row.get("key", "")
    unit = INDICATOR_UNITS.get(key, "")
    # Per-indicator precision
    if key in ("ad_line_slope", "new_highs_lows", "fear_greed", "dxy", "net_liquidity"):
        val = f"{v:,.0f}"
    elif key in ("copper_gold", "vix_term_9d_1m", "vix_term_1m_3m", "put_call"):
        val = f"{v:.3f}"
    elif key in ("hy_spread_velocity", "aaii_bull_bear", "cta_positioning", "equity_risk_premium"):
        val = f"{v:+,.1f}"
    else:
        val = f"{v:,.2f}"
    return f"{val} {unit}".strip() if unit else val


tbl = scores.copy()
tbl["raw"] = tbl.apply(_fmt_raw_cell, axis=1)
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

st.dataframe(styled, width="stretch", hide_index=True)


# ---------------------------------------------------------------------------
# MAJOR INDICES + HISTORICAL REGIME OVERLAY
# ---------------------------------------------------------------------------
def _regime_color_for_score(v: float) -> str:
    if np.isnan(v):
        return "rgba(127,127,127,0.0)"
    if v >= 85:   return "rgba(192, 57, 43, 0.22)"
    if v >= 65:   return "rgba(230,126, 34, 0.18)"
    if v >= 55:   return "rgba(241,196, 15, 0.14)"
    if v >= 45:   return "rgba(149,165,166, 0.10)"
    if v >= 35:   return "rgba( 52,152,219, 0.14)"
    if v >= 15:   return "rgba( 39,174, 96, 0.18)"
    return            "rgba( 22,160,133, 0.22)"


def _render_index_regime_overlay(
    section_title: str,
    price_full: pd.Series | None,
    trace_name: str,
    yaxis_title: str,
) -> None:
    st.markdown(f"### {section_title}")
    st.caption(
        "Background color = composite regime on that day (same scale as the gauge). "
        "Lets you eyeball every past top/bottom call at once: red bands before drawdowns, "
        "green bands before rallies."
    )
    if price_full is None or price_full.empty or comp_hist is None or comp_hist.empty:
        st.info("Not enough history yet to render regime bands.")
        return

    price_plot = _clip_series_to_chart_window(price_full)
    comp_plot = _normalize_series_index(comp_hist)
    comp_plot = comp_plot.loc[(comp_plot.index >= CHART_START) & (comp_plot.index <= CHART_END)]
    if price_plot.empty or comp_plot.empty:
        st.info("Not enough history in chart window.")
        return

    # Align composite onto this index's trading calendar so bands line up
    comp_aligned = comp_plot.reindex(price_plot.index, method="ffill")

    # Collapse consecutive same-color days into single vrects for performance
    colors = comp_aligned.apply(_regime_color_for_score)
    segments: list[tuple[pd.Timestamp, pd.Timestamp, str]] = []
    if not colors.empty:
        seg_start = colors.index[0]
        seg_color = colors.iloc[0]
        prev_idx = colors.index[0]
        for idx, col in colors.items():
            if col != seg_color:
                segments.append((seg_start, prev_idx, seg_color))
                seg_start = idx
                seg_color = col
            prev_idx = idx
        segments.append((seg_start, prev_idx, seg_color))

    fig = go.Figure()
    for x0, x1, col in segments:
        if "rgba(127" in col:  # skip missing
            continue
        fig.add_vrect(x0=x0, x1=x1, fillcolor=col, line_width=0, layer="below")

    fig.add_trace(go.Scatter(
        x=price_plot.index, y=price_plot.values,
        mode="lines", name=trace_name,
        line=dict(color="#f5f5f5", width=1.8),
        hovertemplate=f"%{{x|%Y-%m-%d}}<br>{trace_name} %{{y:,.2f}}<extra></extra>",
    ))
    if ts_marker is not None:
        _add_chart_date_marker(fig, price_plot, ts_marker)

    fig.update_layout(
        height=380,
        margin=dict(l=10, r=10, t=20, b=10),
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
        yaxis=dict(gridcolor="rgba(255,255,255,0.08)", title=yaxis_title),
    )
    st.plotly_chart(fig, width="stretch")


# ts_marker is defined below in the time-series grid block; we need it here too.
# Grab it from session state so the marker lines up across all charts.
ts_marker = st.session_state.get("ts_marker_date", None)


# Lazy fetch: if the cached RawFrame predates these indices being added,
# pull them directly (cheap yfinance call) so the chart works without
# waiting for the 1-hour load_all cache to expire.
@st.cache_data(ttl=60 * 60, show_spinner=False)
def _lazy_index_series(which: str) -> pd.Series:
    """
    Fetch RUT / IXIC for charts when the cached RawFrame is stale.
    Prefer src.data helpers when present; fall back to yf_series on older
    deploys that do not yet define russell2000() / nasdaq_composite().
    """
    from src import data as _D

    def _rut() -> pd.Series:
        fn = getattr(_D, "russell2000", None)
        if callable(fn):
            return fn()
        yf = getattr(_D, "yf_series", None)
        return yf("^RUT") if callable(yf) else pd.Series(dtype=float)

    def _ixic() -> pd.Series:
        fn = getattr(_D, "nasdaq_composite", None)
        if callable(fn):
            return fn()
        yf = getattr(_D, "yf_series", None)
        return yf("^IXIC") if callable(yf) else pd.Series(dtype=float)

    if which == "russell2000":
        return _rut()
    if which == "nasdaq":
        return _ixic()
    return pd.Series(dtype=float)


def _index_series(key: str) -> pd.Series:
    s = raw.series.get(key, pd.Series(dtype=float))
    if s is None or s.empty:
        s = _lazy_index_series(key)
    return s if s is not None else pd.Series(dtype=float)


_render_index_regime_overlay("SPX with historical regime bands", raw.series.get("spx", pd.Series(dtype=float)), "SPX", "SPX")
_render_index_regime_overlay(
    "Russell 2000 with historical regime bands",
    _index_series("russell2000"),
    "Russell 2000",
    "RUT",
)
_render_index_regime_overlay(
    "Nasdaq Composite with historical regime bands",
    _index_series("nasdaq"),
    "Nasdaq",
    "IXIC",
)


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

st.markdown("**Highlight date** — drop a red marker on every chart below")

_m1, _m2, _m3, _m4 = st.columns([1.1, 0.75, 0.75, 2.0], gap="small", vertical_alignment="center")
with _m1:
    _pick = st.date_input(
        "Highlight date",
        value=date.today(),
        min_value=CHART_START.date(),
        max_value=CHART_END.date(),
        key="ts_marker_date_picker",
        label_visibility="collapsed",
    )
with _m2:
    if st.button("Set", key="ts_marker_set", width="stretch", help="Place a red dot on all charts at this date"):
        st.session_state.ts_marker_date = pd.Timestamp(_pick).normalize()
with _m3:
    if st.button("Clear", key="ts_marker_clear", width="stretch", help="Remove the marker from all charts"):
        st.session_state.ts_marker_date = None
with _m4:
    _md = st.session_state.ts_marker_date
    if _md is not None:
        st.markdown(
            f"<div style='color:#4fc978; font-size:0.85rem; padding:4px 0;'>"
            f"● Marker set: <b>{_md.strftime('%Y-%m-%d')}</b> — red dot on each series. "
            f"Change date + Set again to move.</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            "<div style='color:#888; font-size:0.85rem; padding:4px 0;'>"
            "No marker — choose a date and click <b>Set marker</b>.</div>",
            unsafe_allow_html=True,
        )

ts_marker: pd.Timestamp | None = st.session_state.ts_marker_date

# Paired left<->right by theme so each row reads as a side-by-side comparison.
# 11 charts split 6+5 — the orphan (% SP500 above 200DMA) sits at the bottom
# of the LEFT column so any trailing empty space falls on the right (the eye
# expects the left to extend further and tolerates a right-side gap better).
rc1, rc2 = st.columns(2)
with rc1:
    # Row 1 — credit stress (pairs with Fed liquidity)
    _line(
        raw.series.get("hy_spread"),
        "HY Credit Spread (%)",
        ref_lines=[(4.0, "tight → top risk"), (8.0, "stress")],
        indicator_key="hy_spread",
        marker_ts=ts_marker,
    )
    # Row 2 — equity vol (pairs with tail-risk SKEW)
    _line(
        raw.series.get("vix"),
        "VIX",
        ref_lines=[(13, "complacency"), (30, "fear")],
        indicator_key="vix",
        marker_ts=ts_marker,
    )
    # Row 3 — cross-asset divergence (pairs with cross-asset correlation)
    _line(
        raw.series.get("move_vix_div"),
        "MOVE / VIX  (bond-vol vs equity-vol)",
        ref_lines=[(6, "bonds 'lying' — watch")],
        extra_direction="risk_high_is_top",
        marker_ts=ts_marker,
    )
    # Row 4 — retail sentiment (pairs with manager sentiment)
    _line(
        raw.series.get("fear_greed"),
        "CNN Fear & Greed",
        ref_lines=[(25, "extreme fear"), (75, "extreme greed")],
        indicator_key="fear_greed",
        marker_ts=ts_marker,
    )
    # Row 5 — survey positioning (pairs with futures positioning)
    _line(
        raw.series.get("aaii_bull_bear"),
        "AAII Bull−Bear Spread (%)",
        ref_lines=[(-20, "panic"), (20, "euphoria")],
        indicator_key="aaii_bull_bear",
        marker_ts=ts_marker,
    )
    # Row 6 — breadth (orphan; no right-side pair)
    _line(
        raw.series.get("pct_above_200dma"),
        "% SP500 above 200DMA",
        ref_lines=[(20, "washout"), (80, "extended")],
        indicator_key="pct_above_200dma",
        marker_ts=ts_marker,
    )

with rc2:
    # Row 1 — liquidity (pairs with credit stress)
    nl = raw.series.get("net_liquidity")
    _line(
        nl,
        "Fed Net Liquidity (WALCL − TGA − RRP, $B)",
        indicator_key="net_liquidity",
        marker_ts=ts_marker,
    )
    # Row 2 — tail-risk (pairs with VIX)
    _line(
        raw.series.get("skew"),
        "CBOE SKEW",
        ref_lines=[(130, "tail hedging elevated")],
        indicator_key="skew",
        marker_ts=ts_marker,
    )
    # Row 3 — cross-asset correlation (pairs with MOVE/VIX divergence)
    _line(
        raw.series.get("corr_cluster"),
        "SPY vs (TLT+GLD)/2  20D correlation",
        ref_lines=[(0.6, "liquidity event")],
        extra_direction="risk_high_is_top",
        marker_ts=ts_marker,
    )
    # Row 4 — manager sentiment (pairs with retail F&G)
    _line(
        raw.series.get("naaim"),
        "NAAIM Exposure",
        ref_lines=[(30, "defensive"), (100, "leveraged")],
        indicator_key="naaim",
        marker_ts=ts_marker,
    )
    # Row 5 — futures positioning (pairs with AAII survey)
    _line(
        raw.series.get("cta_positioning"),
        "CTA Net Long — CFTC Leveraged Funds, S&P 500 E-mini (% of OI)",
        ref_lines=[(10, "CTAs fully loaded → top risk"), (-10, "CTAs capitulated → bottom setup"), (0, "neutral")],
        indicator_key="cta_positioning",
        marker_ts=ts_marker,
    )


# ---------------------------------------------------------------------------
# VOLATILITY TERM STRUCTURE — backwardation = bottom signal
# ---------------------------------------------------------------------------
st.markdown("### Volatility term structure")
st.caption(
    "Ratio **> 1.0 = backwardation** (near-term vol > longer-term vol) = acute panic. "
    "Historically marks short-term bottoms within ~3 trading days."
)
vc1, vc2 = st.columns(2)
with vc1:
    _line(
        raw.series.get("vix_term_9d_1m"),
        "VIX9D / VIX  (near-term backwardation)",
        ref_lines=[(1.0, "backwardation → bottom signal")],
        indicator_key="vix_term_9d_1m",
        marker_ts=ts_marker,
    )
with vc2:
    _line(
        raw.series.get("vix_term_1m_3m"),
        "VIX / VIX3M  (full-term backwardation)",
        ref_lines=[(1.0, "backwardation → bottom")],
        indicator_key="vix_term_1m_3m",
        marker_ts=ts_marker,
    )


# ---------------------------------------------------------------------------
# YIELD CURVE — inversion & re-steepening
# ---------------------------------------------------------------------------
st.markdown("### Yield curve — inversion & re-steepening")
st.caption(
    "The signal isn't inversion itself — it's the **re-steepening from inversion**. "
    "Every US recession since 1970 started with the curve un-inverting off its trough."
)
yc1, yc2 = st.columns(2)
with yc1:
    _line(
        raw.series.get("curve_2s10s"),
        "2s10s Yield Curve (T10Y2Y)",
        ref_lines=[(0.0, "inversion"), (0.5, "normal")],
        indicator_key="curve_2s10s",
        marker_ts=ts_marker,
    )
with yc2:
    _line(
        raw.series.get("curve_resteep_2s10s"),
        "2s10s Re-steepening from Inversion",
        ref_lines=[(0.5, "active re-steepening → top warning"), (1.0, "strong re-steepen")],
        indicator_key="curve_resteep_2s10s",
        marker_ts=ts_marker,
    )


# ---------------------------------------------------------------------------
# CREDIT VELOCITY & MACRO CONTEXT
# ---------------------------------------------------------------------------
st.markdown("### Credit velocity & macro context")
st.caption(
    "**HY 4W Δ** is the fastest risk-off trigger. "
    "**DXY / real yields** proxy liquidity drain. "
    "**Copper/Gold** leads HY spreads by 1–2 months."
)
mc1, mc2 = st.columns(2)
with mc1:
    _line(
        raw.series.get("hy_spread_velocity"),
        "HY Spread — 4W change (bps)",
        ref_lines=[(75, "fast widening = risk-off trigger"), (-75, "fast compression"), (0, "flat")],
        indicator_key="hy_spread_velocity",
        marker_ts=ts_marker,
    )
    _line(
        raw.series.get("dxy"),
        "DXY — US Dollar Index",
        indicator_key="dxy",
        marker_ts=ts_marker,
    )
with mc2:
    _line(
        raw.series.get("real_yield_10y"),
        "10Y TIPS Real Yield (%)",
        ref_lines=[(0.0, "zero real"), (2.0, "tight")],
        indicator_key="real_yield_10y",
        marker_ts=ts_marker,
    )
    _line(
        raw.series.get("copper_gold"),
        "Copper / Gold Ratio  (growth proxy, leads HY by 1-2m)",
        indicator_key="copper_gold",
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
    st.dataframe(pd.DataFrame(src_rows), width="stretch", hide_index=True)
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
