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
        key="move_index", label="MOVE Index (Bond Vol)",
        bucket="credit_liquidity", source="yfinance:^MOVE / proxy",
        direction="contrarian_high_is_top",
        description="Bond market volatility. High MOVE = liquidity stress = bottom setup (mirrors VIX).",
    ),
    IndicatorSpec(
        key="net_liquidity", label="Fed Net Liquidity (WALCL - TGA - RRP)",
        bucket="credit_liquidity", source="FRED:WALCL/WTREGEN/RRPONTSYD",
        direction="risk_high_is_top",
        description="Systemic USD liquidity. Abundant liquidity fuels euphoric tops (Dec 2021 peak); drained liquidity = bottom regime.",
        weight=1.0,
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
        bucket="sentiment_positioning", source="cboe CSV",
        direction="contrarian_high_is_top",
        description="Options hedging demand. Spikes = panic bottom, <0.5 = complacency top. INVERTED.",
    ),
    IndicatorSpec(
        key="vix", label="VIX",
        bucket="sentiment_positioning", source="yfinance:^VIX",
        direction="contrarian_high_is_top",
        description="Equity vol. <13 = complacency top risk. >35 = panic bottom setup. INVERTED.",
    ),
    IndicatorSpec(
        key="vvix", label="VVIX (Vol of Vol)",
        bucket="sentiment_positioning", source="yfinance:^VVIX",
        direction="contrarian_high_is_top",
        description="Hedge demand on VIX itself. Spike = serious tail hedging.",
    ),
    IndicatorSpec(
        key="skew", label="CBOE SKEW",
        bucket="sentiment_positioning", source="yfinance:^SKEW",
        direction="risk_high_is_top",
        description="Crash hedge demand by institutions. High SKEW = smart money hedging tail.",
    ),

    # 4. Valuation (10%)
    IndicatorSpec(
        key="equity_risk_premium", label="Equity Risk Premium",
        bucket="valuation", source="computed: SPX EP - US10Y",
        direction="contrarian_high_is_top",
        description="SPX earnings yield minus 10Y. Negative = expensive. INVERTED: high ERP = cheap.",
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
        description="Ratio > 1 = near-term backwardation = acute panic = bottom within ~3 days historically.",
    ),
    IndicatorSpec(
        key="vix_term_1m_3m", label="VIX / VIX3M (1M vs 3M term)",
        bucket="sentiment_positioning", source="yfinance:^VIX / ^VIX3M",
        direction="contrarian_high_is_top",
        description="Ratio > 1 = full term backwardation = serious stress = bottom regime.",
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
        description="High DXY = risk-off / liquidity drain / tight financial conditions = bottom regime. Low DXY = loose conditions = supportive of risk-asset tops.",
    ),
    IndicatorSpec(
        key="real_yield_10y", label="10Y TIPS Real Yield",
        bucket="credit_liquidity", source="FRED:DFII10",
        direction="contrarian_high_is_top",
        description="High real yields = tight discount rate = equity pressure = bottom regime (2022 drawdown). Low/negative = cheap money = euphoric top fuel (2021).",
    ),
    IndicatorSpec(
        key="copper_gold", label="Copper / Gold Ratio",
        bucket="credit_liquidity", source="yfinance:HG=F / GC=F",
        direction="risk_high_is_top",
        description="Global growth proxy. High ratio = growth strong / risk-on euphoria (top risk). Low ratio = growth fears / risk-off (bottom). Leads HY spreads by 1-2 months.",
    ),
]

INDICATORS_BY_KEY = {i.key: i for i in INDICATORS}


REGIME_THRESHOLDS = {
    "extreme_complacency": 85,
    "complacent": 65,
    "neutral_high": 55,
    "neutral_low": 45,
    "fearful": 35,
    "extreme_fear": 15,
}


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
