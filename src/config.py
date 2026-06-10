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
    #   None      -> rank the raw level (default; mean-reverting series)
    #   "zscore"  -> rank a rolling z-score over `transform_window`
    transform: str | None = None
    transform_window: int = 252


# Minimum observations an indicator must have before its percentile score is
# trusted. Thin-history feeds (a 2-point GEX cache, a fresh put/call snapshot)
# otherwise score 0/100 and masquerade as real extremes in the composite and
# the high-conviction cluster override.
MIN_OBS = 60


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
        key="move_index", label="Bond Vol (TLT realized-vol proxy)",
        bucket="credit_liquidity", source="yfinance:TLT realized vol (MOVE proxy)",
        direction="contrarian_high_is_top",
        description="Bond-market volatility proxied by TLT 20d realized vol (true MOVE implied-vol index is not free on yfinance). High = liquidity stress = bottom setup (mirrors VIX).",
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
    ),

    # 2. Breadth & Momentum (30%)
    IndicatorSpec(
        key="pct_above_200dma", label="% SP500 Above 200DMA",
        bucket="breadth_momentum", source="yfinance:SP500 components",
        direction="risk_high_is_top",
        description="Breadth health. >80% = extended top risk, <20% = washout bottom.",
    ),
    IndicatorSpec(
        key="ad_line_slope", label="Advance-Decline Line Momentum",
        bucket="breadth_momentum", source="yfinance:^NYAD / computed",
        direction="risk_high_is_top",
        description="NYSE cumulative A/D. High cumulative A/D = mature bull = top risk.",
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
        description="Participation. Many net new highs = extended top. Net new lows = washout bottom.",
    ),

    # 3. Sentiment & Positioning (20%)
    IndicatorSpec(
        key="aaii_bull_bear", label="AAII Bull-Bear Spread",
        bucket="sentiment_positioning", source="scrape:aaii.com",
        direction="risk_high_is_top",
        description="Retail sentiment. Extreme bullish spread = contrarian top.",
    ),
    IndicatorSpec(
        key="naaim", label="NAAIM Exposure Index",
        bucket="sentiment_positioning", source="csv:naaim.org",
        direction="risk_high_is_top",
        description="Active manager equity exposure. >100 leveraged long = top, <30 = bottom setup.",
    ),
    IndicatorSpec(
        key="fear_greed", label="CNN Fear & Greed",
        bucket="sentiment_positioning", source="api:cnn",
        direction="risk_high_is_top",
        description="Composite crowd fear/greed. Extreme greed = top, extreme fear = bottom.",
    ),
    IndicatorSpec(
        key="put_call", label="CBOE Equity Put/Call Ratio",
        bucket="sentiment_positioning", source="CBOE/YCharts",
        direction="contrarian_high_is_top",
        description="Options hedging demand. Spikes = panic bottom, <0.5 = complacency top. INVERTED. Down-weighted: shares the put/call theme with the index ratio.",
        weight=0.5,
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
        description="Active re-steepening after inversion. Fires when 12m min was inverted and curve is recovering — historically coincides with recession onset and equity tops (2000/2007/2020).",
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
        key="fra_ois_spread", label="Funding Stress Spread (CP-TBill proxy)",
        bucket="credit_liquidity", source="FRED:SOFR90DAYAVG/DTB3/DCPF3M",
        direction="contrarian_high_is_top",
        description="Post-LIBOR interbank stress proxy: SOFR 90D avg vs T-Bill or CP-TBill spread. Replaces FRA-OIS (LIBOR discontinued 2023). Elevated = funding stress = bottom setup.",
    ),
    IndicatorSpec(
        key="sofr_spread", label="SOFR Spread (vs Effective Fed Funds)",
        bucket="credit_liquidity", source="FRED:SOFR/DFF",
        direction="contrarian_high_is_top",
        description="SOFR vs Fed Funds spread. Elevated = repo market stress / collateral scarcity (Q4 2019, March 2020). High = liquidity squeeze = bottom regime.",
    ),

    # 11. Gamma Exposure (GEX) - Options Market Structure
    IndicatorSpec(
        key="gamma_exposure", label="Gamma Exposure (GEX) Proxy",
        bucket="sentiment_positioning", source="CBOE delayed SPY options",
        direction="risk_high_is_top",
        description="Estimated dealer gamma exposure from SPY options. Deep negative GEX = market makers forced to sell into weakness (crash accelerant). Extreme negative = capitulation = bottom. Positive = stability. Down-weighted: shares the gamma theme with the flip-zone metric.",
        weight=0.5,
    ),
    IndicatorSpec(
        key="gamma_flip_zone", label="Gamma Flip Zone Distance (%)",
        bucket="sentiment_positioning", source="CBOE delayed SPY options / GEX proxy",
        direction="contrarian_high_is_top",
        description="Distance to 'zero gamma' price where dealer hedging flips from buy-to-sell to sell-to-buy. Near/past flip = volatility expansion = bottom risk. Down-weighted: shares the gamma theme with GEX.",
        weight=0.5,
    ),

    # 12. Enhanced Institutional Positioning
    IndicatorSpec(
        key="index_put_call", label="Index Put/Call Ratio (Institutional Hedging)",
        bucket="sentiment_positioning", source="CBOE/YCharts",
        direction="contrarian_high_is_top",
        description="Institutions hedge portfolios with index puts (not equity puts). Spike = panic hedging by smart money. Extreme spike then rapid drop = hedges monetized = bottom. Down-weighted: shares the put/call theme with the equity ratio.",
        weight=0.5,
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
