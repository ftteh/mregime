"""
Indicator engine: normalize raw data to 0-100 percentile scores,
orient them so HIGH = top-risk / complacency, and produce the composite.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd

from . import data as D
from .config import BUCKET_WEIGHTS, INDICATORS_BY_KEY, ROLLING_WINDOW_DAYS


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------
def rolling_percentile(s: pd.Series, window: int = ROLLING_WINDOW_DAYS) -> pd.Series:
    """Return rolling percentile rank (0-100) of s over `window` observations."""
    if s is None or s.empty:
        return pd.Series(dtype=float)
    win = min(window, max(30, len(s) // 2 if len(s) > 60 else len(s)))
    r = s.rolling(win, min_periods=max(30, win // 4)).apply(
        lambda x: (x.rank(pct=True).iloc[-1] * 100.0) if len(x) > 0 else np.nan,
        raw=False,
    )
    return r


def latest_percentile(s: pd.Series, window: int = ROLLING_WINDOW_DAYS) -> float:
    """Cheap single-value percentile (current value vs last `window` obs)."""
    if s is None or s.empty:
        return np.nan
    tail = s.tail(window).dropna()
    if tail.empty:
        return np.nan
    return float((tail.rank(pct=True).iloc[-1]) * 100.0)


def orient_score(pct: float, direction: str) -> float:
    """
    Convert a 0-100 percentile to a score where HIGH = top-risk / complacency.
      - 'risk_high_is_top'        : raw pct (high value = top risk).
      - 'contrarian_high_is_top'  : invert (a high raw value is SUPPORTIVE, so score = 100 - pct).
    """
    if np.isnan(pct):
        return np.nan
    if direction == "contrarian_high_is_top":
        return 100.0 - pct
    return pct


# ---------------------------------------------------------------------------
# Technicals
# ---------------------------------------------------------------------------
def rsi(series: pd.Series, n: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0).rolling(n).mean()
    down = (-delta.clip(upper=0)).rolling(n).mean()
    rs = up / down.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    line = ema_fast - ema_slow
    sig = line.ewm(span=signal, adjust=False).mean()
    hist = line - sig
    return line, sig, hist


# ---------------------------------------------------------------------------
# Build raw series map
# ---------------------------------------------------------------------------
@dataclass
class RawFrame:
    series: Dict[str, pd.Series]
    meta: Dict[str, dict]


def build_raw() -> RawFrame:
    s: Dict[str, pd.Series] = {}
    meta: Dict[str, dict] = {}

    # ---- Credit & Liquidity
    s["hy_spread"] = D.fred_hy_spread()
    s["ig_spread"] = D.fred_ig_spread()
    s["net_liquidity"] = D.fred_net_liquidity()
    s["financial_conditions"] = D.fred_nfci()
    s["move_index"] = D.move_index()

    # ---- Market / Breadth / Momentum
    spx_px = D.spx()
    s["rsi_spx"] = rsi(spx_px, 14) if not spx_px.empty else pd.Series(dtype=float)
    s["pct_above_200dma"] = D.breadth_pct_above_200dma()
    s["new_highs_lows"] = D.new_highs_minus_lows()
    # A/D line proxy: use sign of daily returns across sample
    try:
        tickers = D.sp500_tickers()[:100]
        import yfinance as yf
        dl = yf.download(tickers, period="1y", progress=False, auto_adjust=False, group_by="ticker", threads=True)
        rets = pd.DataFrame({t: dl[t]["Close"].pct_change() for t in tickers if t in dl.columns.get_level_values(0)})
        ad = (rets > 0).sum(axis=1) - (rets < 0).sum(axis=1)
        s["ad_line_slope"] = ad.cumsum()
    except Exception:
        s["ad_line_slope"] = pd.Series(dtype=float)

    # ---- Sentiment & Positioning
    aaii = D.aaii_sentiment()
    s["aaii_bull_bear"] = aaii["spread"] if "spread" in aaii.columns else pd.Series(dtype=float)
    s["naaim"] = D.naaim_exposure()
    s["fear_greed"] = D.fear_greed_index()
    s["put_call"] = D.put_call_ratio()
    s["vix"] = D.vix()
    s["vvix"] = D.vvix()
    s["skew"] = D.skew()

    # ---- Valuation
    s["equity_risk_premium"] = D.equity_risk_premium()

    # ---- Advanced / bonus
    s["corr_cluster"] = D.correlation_cluster()
    s["move_vix_div"] = D.move_vs_vix_spread()
    s["dix"] = D.dix_proxy()
    s["cta_positioning"] = D.cftc_cta_positioning()
    s["spx"] = spx_px

    # Metadata (current value + freshness)
    for k, v in s.items():
        if v is None or v.empty:
            meta[k] = {"last": np.nan, "as_of": None, "n": 0}
        else:
            meta[k] = {
                "last": float(v.iloc[-1]),
                "as_of": v.index[-1].to_pydatetime(),
                "n": int(len(v)),
            }

    return RawFrame(series=s, meta=meta)


# ---------------------------------------------------------------------------
# Composite scoring
# ---------------------------------------------------------------------------
def score_indicators(raw: RawFrame) -> pd.DataFrame:
    """Return DataFrame: key | label | bucket | raw | percentile | score | direction."""
    rows = []
    for key, spec in INDICATORS_BY_KEY.items():
        series = raw.series.get(key, pd.Series(dtype=float))
        raw_value = float(series.iloc[-1]) if not series.empty else np.nan
        pct = latest_percentile(series)
        score = orient_score(pct, spec.direction)
        rows.append({
            "key": key,
            "label": spec.label,
            "bucket": spec.bucket,
            "direction": spec.direction,
            "raw": raw_value,
            "percentile": pct,
            "score": score,
            "weight": spec.weight,
            "as_of": raw.meta.get(key, {}).get("as_of"),
            "n_obs": raw.meta.get(key, {}).get("n", 0),
        })
    return pd.DataFrame(rows)


def composite(scores_df: pd.DataFrame) -> dict:
    """Compute weighted composite score + per-bucket scores."""
    bucket_scores = {}
    for bucket, weight in BUCKET_WEIGHTS.items():
        sub = scores_df[scores_df["bucket"] == bucket].dropna(subset=["score"])
        if sub.empty:
            bucket_scores[bucket] = {"score": np.nan, "weight": weight, "n": 0}
            continue
        # Indicator-level equal weight within bucket (can use sub['weight'])
        w = sub["weight"].fillna(1.0)
        score = float(np.average(sub["score"].values, weights=w.values))
        bucket_scores[bucket] = {"score": score, "weight": weight, "n": int(len(sub))}

    valid = [b for b in bucket_scores.values() if not np.isnan(b["score"])]
    if not valid:
        return {"composite": np.nan, "buckets": bucket_scores}
    total_w = sum(b["weight"] for b in valid)
    comp = sum(b["score"] * b["weight"] for b in valid) / total_w
    return {"composite": float(comp), "buckets": bucket_scores}


# ---------------------------------------------------------------------------
# Cluster detection — "are we in a confluence top/bottom?"
# ---------------------------------------------------------------------------
def cluster_signal(scores_df: pd.DataFrame) -> dict:
    """Count how many indicators are in extreme territory (both directions)."""
    extreme_top = scores_df[scores_df["score"] >= 85]
    extreme_bot = scores_df[scores_df["score"] <= 15]
    return {
        "top_cluster_count": int(len(extreme_top)),
        "bottom_cluster_count": int(len(extreme_bot)),
        "top_names": extreme_top["label"].tolist(),
        "bottom_names": extreme_bot["label"].tolist(),
    }
