"""
Central configuration for the Quant Regime Dashboard.

Four-bucket composite model used by institutional allocators:
    40%  Credit & Liquidity  ("The Truth")
    30%  Breadth & Momentum  ("The Flow")
    20%  Sentiment & Positioning ("The Contrarian")
    10%  Valuation  ("The Anchor")

Each raw indicator is converted to a 0-100 percentile vs a rolling 3y window,
then oriented so that HIGH = complacent/top-risk and LOW = panic/bottom-setup.
"""

from __future__ import annotations
import os
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from dotenv import load_dotenv

load_dotenv()

FRED_API_KEY = os.getenv("FRED_API_KEY", "")
NASDAQ_DATA_LINK_API_KEY = os.getenv("NASDAQ_DATA_LINK_API_KEY", "")

ROLLING_WINDOW_DAYS = 252 * 3

BUCKET_WEIGHTS = {
    "credit_liquidity": 0.40,
    "breadth_momentum": 0.30,
    "sentiment_positioning": 0.20,
    "valuation": 0.10,
}

Direction = Literal["contrarian_high_is_top", "risk_high_is_top"]


@dataclass(frozen=True)
class IndicatorSpec:
    key: str
    label: str
    bucket: str
    source: str
    direction: Direction
    description: str
    weight: float = 1.0
    # Optional pre-percentile transform. Trending / non-stationary macro series
    # (net liquidity, DXY, real yields, copper/gold) saturate a raw rolling
    # percentile at 0/100 for months; a rolling z-score re-centres them on their
    # own recent regime so the percentile measures deviation, not level/drift.
    #   None          -> rank the raw level (default; mean-reverting series)
    #   "zscore"      -> rank a rolling z-score over `transform_window`
    #   "active_only" -> an event flag that sits at exactly 0 when it is off.
    #                    Off scores a neutral 50; on scores 50-100 by its rank
    #                    among past active readings. Ranking the zeros gave an
    #                    arbitrary score equal to the share of zeros in the window.
    #   "stress_only" -> a spread that only carries information when it is high.
    #                    Readings below the window median all score a neutral 50;
    #                    above it the usual rank applies. A funding spread of -3bp
    #                    instead of +5bp is noise, not "extreme complacency".
    transform: str | None = None
    transform_window: int = 252
    # Business-day moving average applied before the percentile. Daily put/call
    # and new-highs prints are mostly noise one day at a time (the raw equity
    # put/call moved its own score 30+ points on about one day in four), and the
    # standard reading of each is its 5- or 10-day average.
    smooth: int = 1
    # Calendar days a reading may lag the newest data before it stops voting.
    # A dead feed used to keep its last score at full weight indefinitely.
    # Daily feeds get 10 days (publication lags and holidays, not just a day);
    # weekly and monthly feeds set their own.
    max_stale_days: int = 10
    # Charted and tabled, never scored.
    display_only: bool = False


# Minimum observations an indicator must have before its percentile score is
# trusted. Thin-history feeds (a 2-point GEX cache, a fresh put/call snapshot)
# otherwise score 0/100 and masquerade as real extremes in the composite and
# the high-conviction cluster override.
MIN_OBS = 60

# Percentile window warm-up, in business days. The window grows from one year of
# history to the full three and then rolls. The engine used to halve the window
# for any series under six years long, so several "3-year percentiles" were
# really 6 to 18 months.
PCT_MIN_PERIODS = 252

# A reading at or beyond these oriented scores counts as an extreme for the cluster.
EXTREME_TOP = 85.0
EXTREME_BOTTOM = 15.0

# A cluster is a LOPSIDED set of extremes: themes at one extreme minus themes at
# the other must reach this share of the live themes (8 of 22), across 2+ pillars.
# The rule used to be "3 themes at one extreme", ignoring the other side. Each
# tail is 15% of an indicator's own history, so with 22 themes about five sit at
# each extreme on an ordinary day: over 2024-2026 that rule had a cluster on 99%
# of sessions and "TOP CLUSTER, cut 20 points, High conviction" on 9 in 10.
CLUSTER_NET_SHARE = 1.0 / 3.0

# Share of a pillar's indicator weight that must be live for the pillar to be
# scored, and share of the whole model that must be live for a composite at all.
# One surviving indicator used to stand in for a whole pillar at full weight.
BUCKET_MIN_COVERAGE = 0.5
MODEL_MIN_COVERAGE = 0.5


INDICATORS: list[IndicatorSpec] = [
    # 1. Credit & Liquidity (40%)
    IndicatorSpec(
        key="hy_spread", label="High Yield Credit Spread",
        bucket="credit_liquidity", source="FRED:BAMLH0A0HYM2",
        direction="contrarian_high_is_top",
        description="ICE BofA US High Yield OAS. Widening = stress (bottom setup → score 0). Tight = complacency (top risk → score 100).",
    ),
    IndicatorSpec(
        key="ig_spread", label="Investment Grade Spread",
        bucket="credit_liquidity", source="FRED:BAMLC0A0CM",
        direction="contrarian_high_is_top",
        description="ICE BofA US Corporate OAS. Wide = credit stress = bottom. Tight = complacency = top.",
    ),
    IndicatorSpec(
        key="move_index", label="Bond Vol (ICE MOVE)",
        bucket="credit_liquidity", source="yfinance:^MOVE (TLT realized-vol proxy if unavailable)",
        direction="contrarian_high_is_top",
        description="ICE BofA MOVE: implied volatility of US Treasuries. Falls back to scaled TLT 20d realized vol if ^MOVE is unavailable. High = liquidity stress = bottom setup (mirrors VIX).",
    ),
    IndicatorSpec(
        key="net_liquidity", label="Fed Net Liquidity (WALCL - TGA - RRP)",
        bucket="credit_liquidity", source="FRED:WALCL/WTREGEN/RRPONTSYD",
        direction="risk_high_is_top",
        description="Systemic USD liquidity, z-scored vs its own 1y regime (it trends secularly with QE/QT, so the raw level pins the percentile). Rising vs trend = liquidity tailwind for tops; draining = bottom regime.",
        weight=1.0, transform="zscore",
    ),
    IndicatorSpec(
        key="financial_conditions", label="Financial Conditions (NFCI)",
        bucket="credit_liquidity", source="FRED:NFCI",
        direction="contrarian_high_is_top",
        description="Chicago Fed NFCI. Above 0 = tight = stress = bottom setup. Below 0 = loose = supportive of tops.",
        max_stale_days=16,  # weekly, published Wednesday for the prior Friday
    ),

    # 2. Breadth & Momentum (30%)
    IndicatorSpec(
        key="pct_above_200dma", label="% SP500 Above 200DMA",
        bucket="breadth_momentum", source="yfinance:SP500 components",
        direction="risk_high_is_top",
        description="Breadth health. >80% = extended top risk, <20% = washout bottom.",
    ),
    IndicatorSpec(
        key="ad_line_slope", label="Advance-Decline Line (vs 1y trend)",
        bucket="breadth_momentum", source="computed: S&P 500 constituents",
        direction="risk_high_is_top",
        description="Cumulative advancers minus decliners across S&P 500 constituents over the full panel, z-scored vs its own 1y regime (a cumulative line trends, so its raw level pins the percentile). High vs trend = mature, broad advance = top risk; low = persistent net declines = bottom setup.",
        transform="zscore",
    ),
    IndicatorSpec(
        key="rsi_spx", label="SPX 14D RSI",
        bucket="breadth_momentum", source="yfinance:^GSPC",
        direction="risk_high_is_top",
        description=">70 overbought (top risk), <30 oversold (bottom setup). Actionable at extremes.",
    ),
    IndicatorSpec(
        key="new_highs_lows", label="NYSE 52W New Highs - Lows",
        bucket="breadth_momentum", source="yfinance:SP500 components",
        direction="risk_high_is_top",
        description="Participation, 10-day average. Many net new highs = extended top. Net new lows = washout bottom.",
        smooth=10,
    ),

    # 3. Sentiment & Positioning (20%)
    IndicatorSpec(
        key="aaii_bull_bear", label="AAII Bull-Bear Spread",
        bucket="sentiment_positioning", source="scrape:aaii.com",
        direction="risk_high_is_top",
        description="Retail sentiment. Extreme bullish spread = contrarian top.",
        max_stale_days=14,  # weekly survey
    ),
    IndicatorSpec(
        key="fear_greed", label="CNN Fear & Greed",
        bucket="sentiment_positioning", source="api:cnn",
        direction="risk_high_is_top",
        description="Composite crowd fear/greed. Extreme greed = top, extreme fear = bottom.",
    ),
    IndicatorSpec(
        key="put_call", label="CBOE Equity Put/Call Ratio (10D avg)",
        bucket="sentiment_positioning", source="CBOE official daily statistics",
        direction="contrarian_high_is_top",
        description="Options hedging demand, official Cboe ratio, 10-day average. High = panic bottom, low = complacency top. INVERTED. Down-weighted: shares the put/call theme with the index ratio.",
        weight=0.5, smooth=10,
    ),
    IndicatorSpec(
        key="vix", label="VIX",
        bucket="sentiment_positioning", source="yfinance:^VIX",
        direction="contrarian_high_is_top",
        description="Equity vol. <13 = complacency top risk. >35 = panic bottom setup. INVERTED. Down-weighted: anchor of the collinear vol complex (VIX/VVIX/SKEW/term-structure).",
        weight=0.4,
    ),
    IndicatorSpec(
        key="vvix", label="VVIX (Vol of Vol)",
        bucket="sentiment_positioning", source="yfinance:^VVIX",
        direction="contrarian_high_is_top",
        description="Hedge demand on VIX itself. Spike = serious tail hedging. Down-weighted: collinear with the VIX complex.",
        weight=0.4,
    ),
    IndicatorSpec(
        key="skew", label="CBOE SKEW",
        bucket="sentiment_positioning", source="yfinance:^SKEW",
        direction="risk_high_is_top",
        description="Crash hedge demand by institutions. High SKEW = smart money hedging tail. Down-weighted: collinear with the VIX complex.",
        weight=0.4,
    ),

    # 4. Valuation (10%)
    IndicatorSpec(
        key="equity_risk_premium", label="Equity Risk Premium",
        bucket="valuation", source="computed: SPX EP - US10Y",
        direction="contrarian_high_is_top",
        description="SPX earnings yield minus 10Y, z-scored vs its own 1y regime (like real yields, it trends with the secular rate regime — the raw level pinned the 3y percentile near the tail through the whole 2023-25 high-rate era, turning the valuation pillar into a constant vote). Negative = expensive. INVERTED: high ERP = cheap.",
        transform="zscore",
    ),

    # 5. CTA / Institutional Positioning (bonus — folded into sentiment pillar)
    IndicatorSpec(
        key="cta_positioning", label="CTA Net Long (CFTC COT, % OI)",
        bucket="sentiment_positioning", source="CFTC:TFF fut_fin_txt",
        direction="risk_high_is_top",
        description=(
            "CFTC Traders in Financial Futures — Leveraged Funds net long in "
            "S&P 500 E-mini as % of open interest. "
            ">10% net long = CTAs fully loaded (top risk). "
            "<-10% net short = CTAs capitulated (bottom setup). Weekly."
        ),
        max_stale_days=21,  # as of Tuesday, published Friday; holidays delay it further
    ),

    # 6. VIX term structure (sentiment pillar)
    IndicatorSpec(
        key="vix_term_9d_1m", label="VIX9D / VIX (9D vs 1M term)",
        bucket="sentiment_positioning", source="yfinance:^VIX9D / ^VIX",
        direction="contrarian_high_is_top",
        description="Ratio > 1 = near-term backwardation = acute panic = bottom within ~3 days historically. Down-weighted: collinear with the VIX complex.",
        weight=0.4,
    ),
    IndicatorSpec(
        key="vix_term_1m_3m", label="VIX / VIX3M (1M vs 3M term)",
        bucket="sentiment_positioning", source="yfinance:^VIX / ^VIX3M",
        direction="contrarian_high_is_top",
        description="Ratio > 1 = full term backwardation = serious stress = bottom regime. Down-weighted: collinear with the VIX complex.",
        weight=0.4,
    ),

    # 7. Yield curve (credit & liquidity pillar)
    IndicatorSpec(
        key="curve_2s10s", label="2s10s Yield Curve",
        bucket="credit_liquidity", source="FRED:T10Y2Y",
        direction="contrarian_high_is_top",
        description="Negative = inverted = late-cycle top-risk warning. Steep positive = early/mid cycle = not extreme.",
    ),
    IndicatorSpec(
        key="curve_3m10y", label="3M10Y Yield Curve",
        bucket="credit_liquidity", source="FRED:T10Y3M",
        direction="contrarian_high_is_top",
        description="Fed's preferred recession curve. Inverted (negative) = top-risk warning.",
    ),
    IndicatorSpec(
        key="curve_resteep_2s10s", label="2s10s Re-steepening from Inversion",
        bucket="credit_liquidity", source="derived:T10Y2Y",
        direction="risk_high_is_top",
        description="Active re-steepening after inversion. Fires when 12m min was inverted and curve is recovering — historically coincides with recession onset and equity tops (2000/2007/2020). A top-risk warning only: it scores a neutral 50 while off and 50-100 while active.",
        transform="active_only",
    ),

    # 8. Credit spread velocity (credit & liquidity pillar)
    IndicatorSpec(
        key="hy_spread_velocity", label="HY Spread 4W Change (bps)",
        bucket="credit_liquidity", source="derived:BAMLH0A0HYM2",
        direction="contrarian_high_is_top",
        description="4-week change in HY OAS in bps. >+75 bps = fast widening = acute stress = bottom trigger. Negative = spreads compressing = supportive.",
    ),

    # 9. Macro context (credit & liquidity pillar)
    IndicatorSpec(
        key="dxy", label="US Dollar Index (DXY)",
        bucket="credit_liquidity", source="yfinance:DX-Y.NYB",
        direction="contrarian_high_is_top",
        description="US dollar, z-scored vs its own 1y regime (the level trends). High vs trend = risk-off / liquidity drain = bottom regime; low = loose conditions supportive of tops.",
        transform="zscore",
    ),
    IndicatorSpec(
        key="real_yield_10y", label="10Y TIPS Real Yield",
        bucket="credit_liquidity", source="FRED:DFII10",
        direction="contrarian_high_is_top",
        description="10Y real yield, z-scored vs its own 1y regime (it regime-shifted from -1% in 2021 to +2% in 2023). High vs trend = tight discount rate = equity pressure = bottom; low = cheap money = top fuel.",
        transform="zscore",
    ),
    IndicatorSpec(
        key="copper_gold", label="Copper / Gold Ratio",
        bucket="credit_liquidity", source="yfinance:HG=F / GC=F",
        direction="risk_high_is_top",
        description="Global growth proxy, z-scored vs its own 1y regime. High vs trend = growth/risk-on euphoria (top); low = growth fears / risk-off (bottom). Leads HY spreads by 1-2 months.",
        transform="zscore",
    ),

    # 10. Interbank Funding Stress (Repo Market / FRA-OIS)
    IndicatorSpec(
        key="fra_ois_spread", label="Funding Stress Spread (3M CP − T-Bill, 5D avg)",
        bucket="credit_liquidity", source="FRED:DCPF3M/DTB3",
        direction="contrarian_high_is_top",
        description="3-month AA financial commercial paper minus the 3-month T-bill, 5-day average: the closest free successor to the TED spread. Both legs look forward three months, so it measures unsecured funding cost, not the expected policy path. Elevated = funding stress = bottom setup. One-sided: a calm reading is neutral, not a top-risk vote.",
        smooth=5, transform="stress_only",
    ),
    IndicatorSpec(
        key="sofr_spread", label="SOFR Spread (vs Effective Fed Funds)",
        bucket="credit_liquidity", source="FRED:SOFR/DFF",
        direction="contrarian_high_is_top",
        description="SOFR vs Fed Funds spread, 5-day average (single prints spike at month-end). Elevated = repo market stress / collateral scarcity (Q4 2019, March 2020). High = liquidity squeeze = bottom regime. One-sided: a calm reading is neutral, not a top-risk vote.",
        smooth=5, transform="stress_only",
    ),

    # 11. Gamma Exposure (GEX) - Options Market Structure
    IndicatorSpec(
        key="gamma_exposure", label="Gamma Exposure (GEX, 5D avg)",
        bucket="sentiment_positioning", source="SqueezeMetrics (SPX dealer gamma)",
        direction="risk_high_is_top",
        description="Estimated SPX dealer gamma, 5-day average, one methodology only (SqueezeMetrics; the SPY chain snapshot is charted but not scored unless SqueezeMetrics has never served data). Deep negative GEX = market makers forced to sell into weakness (crash accelerant). Extreme negative = capitulation = bottom. Positive = stability.",
        weight=0.5, smooth=5,
    ),
    IndicatorSpec(
        key="gamma_flip_zone", label="Gamma Flip Zone Distance (%)",
        bucket="sentiment_positioning", source="CBOE delayed SPY options / GEX proxy",
        direction="contrarian_high_is_top",
        description="Distance to an estimated 'zero gamma' price. Display only: the estimate is the midpoint of gamma/OI-weighted put and call strikes, not a recomputed zero-gamma level, and its history is a handful of sparse snapshots. It is charted for context and does not vote.",
        weight=0.5, display_only=True,
    ),

    # 12. Enhanced Institutional Positioning
    IndicatorSpec(
        key="index_put_call", label="Index Put/Call Ratio (10D avg)",
        bucket="sentiment_positioning", source="CBOE official daily statistics",
        direction="contrarian_high_is_top",
        description="Institutions hedge portfolios with index puts (not equity puts). Official Cboe ratio, 10-day average. High = heavy institutional hedging = bottom setup. Down-weighted: shares the put/call theme with the equity ratio.",
        weight=0.5, smooth=10,
    ),

    # 13. Full-universe participation breadth (breadth & momentum pillar)
    IndicatorSpec(
        key="trend_breadth_10_20", label="% S&P 500 with 10DMA > 20DMA",
        bucket="breadth_momentum", source="yfinance:all S&P 500 constituents",
        direction="risk_high_is_top",
        description=(
            "Short-term trend breadth: share of current S&P 500 constituents whose 10-day "
            ">20-day moving average, equal-weighted. >80% = extended thrust (top risk short-term), "
            "<20% = washed out (bottom setup). Down-weighted: fast oscillator, shares the "
            "breadth-oscillator theme with the 50D OB/OS spread."
        ),
        weight=0.5,
    ),
    IndicatorSpec(
        key="obos_spread_50d", label="50D Overbought − Oversold Breadth (pp)",
        bucket="breadth_momentum", source="yfinance:all S&P 500 constituents",
        direction="risk_high_is_top",
        description=(
            "Share of constituents >1σ above their 50-day average (overbought) minus share >1σ "
            "below (oversold), σ = 50D rolling std of price. Upper-tail spread = stretched breadth "
            "(top risk), lower tail = washout (bottom setup). Down-weighted: shares the "
            "breadth-oscillator theme with 10/20DMA breadth."
        ),
        weight=0.5,
    ),

    # 14. Growth-equity valuation (valuation pillar)
    IndicatorSpec(
        key="ndx_trailing_pe", label="Nasdaq 100 Trailing P/E",
        bucket="valuation", source="worldperatio.com (QQQ basis)",
        direction="risk_high_is_top",
        description=(
            "Nasdaq 100 trailing P/E (QQQ), monthly history since 1990 with a live latest point. "
            "High vs its own 3y history = expensive growth equities (top risk); low = valuation reset "
            "(bottom setup). Adds a direct multiple to the valuation pillar alongside the rates-relative ERP. "
            "Forward P/E is charted but not scored until its free history is long enough."
        ),
        max_stale_days=45,  # monthly history; the live point is daily when the source is up
    ),
]

INDICATORS_BY_KEY = {i.key: i for i in INDICATORS}


# Collinearity themes for cluster de-duplication. The cluster signal counts
# independent *themes* at an extreme, not raw indicators — otherwise a single
# vol event lights up VIX/VVIX/SKEW/term-structure simultaneously and fakes a
# "4+ aligned extremes" confluence from one underlying factor. Indicators not
# listed here are their own theme (keyed by their own name).
INDICATOR_THEMES: dict[str, str] = {
    "vix": "vol_complex",
    "vvix": "vol_complex",
    "skew": "vol_complex",
    "vix_term_9d_1m": "vol_complex",
    "vix_term_1m_3m": "vol_complex",
    "gamma_exposure": "gamma",
    "gamma_flip_zone": "gamma",
    "put_call": "put_call",
    "index_put_call": "put_call",
    "curve_2s10s": "yield_curve",
    "curve_3m10y": "yield_curve",
    "curve_resteep_2s10s": "yield_curve",
    "hy_spread": "credit_spread",
    "ig_spread": "credit_spread",
    "hy_spread_velocity": "credit_spread",
    "fra_ois_spread": "funding_stress",
    "sofr_spread": "funding_stress",
    "trend_breadth_10_20": "breadth_oscillator",
    "obos_spread_50d": "breadth_oscillator",
}


def theme_for(key: str) -> str:
    """Return the collinearity theme for an indicator key (defaults to itself)."""
    return INDICATOR_THEMES.get(key, key)


REGIME_THRESHOLDS = {
    "extreme_complacency": 85,
    "complacent": 65,
    "neutral_high": 55,
    "neutral_low": 45,
    "fearful": 35,
    "extreme_fear": 15,
}

# Default regime colors (gauge steps + regime band backgrounds)
# Each entry: (min_score, max_score, hex_color, regime_name)
DEFAULT_REGIME_COLORS: list[tuple[float, float, str, str]] = [
    (0, 15, "#16a085", "capitulation"),      # Dark teal - aggressive accumulation
    (15, 35, "#27ae60", "extreme_fear"),     # Green - panic/accumulate
    (35, 45, "#3498db", "fearful"),          # Blue - fearful/watch
    (45, 55, "#95a5a6", "neutral"),            # Gray - neutral
    (55, 65, "#f1c40f", "neutral_high"),     # Yellow - neutral/late-cycle
    (65, 85, "#e67e22", "complacent"),       # Orange - complacent
    (85, 100, "#c0392b", "extreme_complacency"),  # Red - extreme complacency
]


def get_regime_color_for_score(
    score: float,
    colors: list[tuple[float, float, str, str]] | None = None,
    alpha: float = 1.0,
) -> str:
    """
    Return hex color (with optional alpha) for a given score.
    
    Args:
        score: 0-100 composite score
        colors: Optional custom color configuration (defaults to DEFAULT_REGIME_COLORS)
        alpha: Opacity 0-1 for rgba output (1.0 = solid hex, <1 = rgba string)
    """
    if np.isnan(score):
        return "rgba(127,127,127,0.0)" if alpha < 1.0 else "#7f8c8d"
    
    palette = colors if colors else DEFAULT_REGIME_COLORS
    for min_s, max_s, hex_c, _ in palette:
        if min_s <= score < max_s or (max_s == 100 and score >= min_s):
            if alpha < 1.0:
                # Convert hex to rgba
                hex_c = hex_c.lstrip("#")
                r = int(hex_c[0:2], 16)
                g = int(hex_c[2:4], 16)
                b = int(hex_c[4:6], 16)
                return f"rgba({r},{g},{b},{alpha})"
            return hex_c
    return "#7f8c8d"


def get_regime_step_config(
    colors: list[tuple[float, float, str, str]] | None = None,
) -> list[dict]:
    """Return Plotly gauge step configuration from regime colors."""
    palette = colors if colors else DEFAULT_REGIME_COLORS
    return [
        {"range": [min_s, max_s], "color": hex_c}
        for min_s, max_s, hex_c, _ in palette
    ]


def get_regime_band_alpha(regime_name: str, is_active: bool = False) -> float:
    """
    Return alpha transparency for regime band backgrounds.
    Active (matching current gauge regime) gets brighter/higher alpha.
    """
    base_alpha = 0.10
    active_alpha = 0.35
    return active_alpha if is_active else base_alpha


def regime_label(score: float) -> tuple[str, str, str]:
    """Return (label, emoji, color) for a 0-100 composite score."""
    if score >= REGIME_THRESHOLDS["extreme_complacency"]:
        return "EXTREME COMPLACENCY — De-risk, buy tail protection", "!!", "#c00"
    if score >= REGIME_THRESHOLDS["complacent"]:
        return "COMPLACENT — Trim risk, tighten stops", "/\\", "#e67e22"
    if score >= REGIME_THRESHOLDS["neutral_high"]:
        return "NEUTRAL / Late-cycle", "~", "#f1c40f"
    if score >= REGIME_THRESHOLDS["neutral_low"]:
        return "NEUTRAL", "=", "#95a5a6"
    if score >= REGIME_THRESHOLDS["fearful"]:
        return "FEARFUL — Watch for stabilization", "v", "#3498db"
    if score >= REGIME_THRESHOLDS["extreme_fear"]:
        return "PANIC — Accumulate quality", "V", "#27ae60"
    return "CAPITULATION — Aggressive accumulation zone", "VV", "#16a085"


def get_regime_name_for_score(score: float) -> str:
    """Return regime name identifier for a given score."""
    for min_s, max_s, _, name in DEFAULT_REGIME_COLORS:
        if min_s <= score < max_s or (max_s == 100 and score >= min_s):
            return name
    return "unknown"
