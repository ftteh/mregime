"""
Pure analytics for the participation / rotation / valuation / standardized-signal
panels.

No network and no Streamlit here: every function takes pandas objects and
returns pandas objects or plain dicts, so the math is unit-testable and shared
by `data.py` (which fetches and caches) and `app.py` (which renders).

Methodology notes (calibrated against the reference research charts):
  * Short-term trend breadth = % of eligible constituents whose 10-day SMA is
    above their 20-day SMA. Equal-weighted across eligible securities.
  * Overbought / oversold breadth = per stock z = (close - SMA50) / SD50, where
    SD50 is the 50-day rolling standard deviation of the close. A stock is
    overbought when z > +1 and oversold when z < -1. The spread (OB% - OS%) is
    the headline oscillator.
  * Relative strength = ratio of two price series, rebased to 100 at the first
    observation of the display window, with 50D / 200D moving averages of the
    ratio (computed on full history so they are populated from the left edge).
  * Standardized signal = rolling 1-year (252 business-day) z-score — point in
    time, no look-ahead — plus the 1-year percentile of the latest raw value.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
def _clean_index(obj):
    """Return a copy with a tz-naive, sorted, de-duplicated DatetimeIndex."""
    out = obj.copy()
    if not isinstance(out.index, pd.DatetimeIndex):
        out.index = pd.to_datetime(out.index, errors="coerce")
        out = out[out.index.notna()]
    if getattr(out.index, "tz", None) is not None:
        out.index = out.index.tz_localize(None)
    out = out.sort_index()
    return out[~out.index.duplicated(keep="last")]


def clean_series(s: pd.Series | None) -> pd.Series:
    """Numeric, tz-naive, sorted, de-duplicated series with NaNs dropped."""
    if s is None or len(s) == 0:
        return pd.Series(dtype=float)
    out = _clean_index(pd.to_numeric(s, errors="coerce"))
    return out.dropna()


def _clean_panel(panel: pd.DataFrame | None) -> pd.DataFrame:
    """Numeric close panel (dates x tickers), short gaps forward-filled.

    `ffill(limit=5)` bridges single-name halts / missing prints without
    resurrecting a delisted ticker for months.
    """
    if panel is None or panel.empty:
        return pd.DataFrame()
    p = _clean_index(panel.apply(pd.to_numeric, errors="coerce"))
    p = p.dropna(how="all")
    return p.ffill(limit=5)


def _coverage_ok(eligible: pd.Series, min_frac: float = 0.5) -> pd.Series:
    """Mask out dates where the eligible-stock count collapses.

    A partial bulk download (rate limit, a half-populated intraday row) can
    leave a handful of tickers on one date, which would print a wild breadth
    reading off a tiny sample. Require at least `min_frac` of the typical
    (median) eligible count.
    """
    pos = eligible[eligible > 0]
    if pos.empty:
        return eligible > 0
    floor = max(1.0, min_frac * float(pos.median()))
    return eligible >= floor


def percentile_of_last(s: pd.Series, lookback: int | None = None) -> float:
    """Percentile rank (0-100) of the last value within the trailing window
    (inclusive of itself; ties count as <=)."""
    s = clean_series(s)
    if lookback is not None:
        s = s.iloc[-lookback:]
    if s.empty:
        return float("nan")
    return float((s <= s.iloc[-1]).mean() * 100.0)


def window_by_years(s: pd.Series, years: float, end: pd.Timestamp | None = None) -> pd.Series:
    """Slice the trailing `years` of a series by calendar time (not row count),
    so weekly/monthly/daily series get the same wall-clock lookback."""
    s = clean_series(s)
    if s.empty:
        return s
    end = s.index[-1] if end is None else pd.Timestamp(end)
    start = end - pd.DateOffset(days=int(round(365.25 * years)))
    return s.loc[(s.index > start) & (s.index <= end)]


# ---------------------------------------------------------------------------
# Breadth: short-term trend (10DMA > 20DMA)
# ---------------------------------------------------------------------------
def trend_breadth(panel: pd.DataFrame, fast: int = 10, slow: int = 20) -> pd.DataFrame:
    """% of eligible constituents with SMA(fast) > SMA(slow).

    Returns a frame indexed by date with columns:
      pct       — share of eligible stocks in a short-term uptrend (0-100)
      eligible  — number of stocks with both averages defined that day
    """
    p = _clean_panel(panel)
    if p.empty:
        return pd.DataFrame(columns=["pct", "eligible"], dtype=float)
    f = p.rolling(fast, min_periods=fast).mean()
    s = p.rolling(slow, min_periods=slow).mean()
    valid = f.notna() & s.notna()
    n = valid.sum(axis=1)
    above = ((f > s) & valid).sum(axis=1)
    pct = above / n.replace(0, np.nan) * 100.0
    out = pd.DataFrame({"pct": pct, "eligible": n.astype(float)})
    out = out[_coverage_ok(n)]
    return out.dropna(subset=["pct"])


TREND_BREADTH_BANDS = (20.0, 50.0, 80.0)  # washout / neutral / extended


def trend_breadth_regime(pct: float) -> tuple[str, str]:
    """(label, signal-lens sentence) for a 10/20DMA breadth reading."""
    if pct is None or np.isnan(pct):
        return "NO DATA", "Breadth unavailable."
    if pct >= 80:
        return ("EXTENDED",
                "Participation is stretched — thrusts this strong usually cool; "
                "momentum is healthy but adding here is late.")
    if pct >= 60:
        return "BROAD", "Participation is constructive."
    if pct >= 40:
        return "MIXED", "Participation is balanced — no short-term breadth edge."
    if pct >= 20:
        return "NARROW", "Participation is deteriorating — most stocks are in short-term downtrends."
    return ("WASHED OUT",
            "Short-term breadth is washed out — historically where selling exhausts "
            "and bounces begin.")


def trend_breadth_snapshot(
    breadth: pd.DataFrame,
    index_px: pd.Series | None = None,
    lookback_years: float = 3.0,
    neutral: float = 50.0,
) -> dict:
    """Research-snapshot numbers for the trend-breadth panel."""
    if breadth is None or breadth.empty or "pct" not in breadth.columns:
        return {}
    pct = clean_series(breadth["pct"])
    if pct.empty:
        return {}
    cur = float(pct.iloc[-1])
    label, lens = trend_breadth_regime(cur)
    hist = window_by_years(pct, lookback_years)
    elig = breadth["eligible"].dropna() if "eligible" in breadth.columns else pd.Series(dtype=float)
    out = {
        "current": cur,
        "as_of": pct.index[-1],
        "label": label,
        "lens": lens,
        "pct_rank": percentile_of_last(hist),
        "vs_neutral": cur - neutral,
        "eligible": int(elig.iloc[-1]) if not elig.empty else None,
        "index_vs_200dma": float("nan"),
    }
    px = clean_series(index_px)
    if len(px) >= 200:
        ma = px.rolling(200).mean()
        out["index_vs_200dma"] = float((px.iloc[-1] / ma.iloc[-1] - 1.0) * 100.0)
    return out


# ---------------------------------------------------------------------------
# Breadth: overbought / oversold vs 50-day band
# ---------------------------------------------------------------------------
def overbought_oversold_breadth(
    panel: pd.DataFrame, window: int = 50, k: float = 1.0
) -> pd.DataFrame:
    """Share of constituents overbought / oversold vs their own 50-day band.

    Per stock: z = (close - SMA_w) / SD_w (rolling std of the close). OB when
    z > +k, OS when z < -k. Columns: ob_pct, os_pct, spread (= ob - os, in pp),
    eligible.
    """
    p = _clean_panel(panel)
    cols = ["ob_pct", "os_pct", "spread", "eligible"]
    if p.empty:
        return pd.DataFrame(columns=cols, dtype=float)
    ma = p.rolling(window, min_periods=window).mean()
    sd = p.rolling(window, min_periods=window).std()
    z = (p - ma) / sd.where(sd > 0)
    valid = z.notna()
    n = valid.sum(axis=1)
    denom = n.replace(0, np.nan)
    ob = ((z > k) & valid).sum(axis=1) / denom * 100.0
    os_ = ((z < -k) & valid).sum(axis=1) / denom * 100.0
    out = pd.DataFrame({"ob_pct": ob, "os_pct": os_, "spread": ob - os_, "eligible": n.astype(float)})
    out = out[_coverage_ok(n)]
    return out.dropna(subset=["spread"])


def obos_zones(spread: pd.Series, lo_q: float = 0.15, hi_q: float = 0.85) -> tuple[float, float]:
    """(oversold threshold, overbought threshold) from the spread's own history.

    Adaptive percentile zones instead of fixed ±30: the spread's range depends
    on the universe and regime; the outer 15% tails on each side are the
    statistically stretched readings.
    """
    s = clean_series(spread)
    if len(s) < 30:
        return (-30.0, 30.0)
    return (float(s.quantile(lo_q)), float(s.quantile(hi_q)))


def obos_snapshot(df: pd.DataFrame) -> dict:
    if df is None or df.empty or "spread" not in df.columns:
        return {}
    spread = clean_series(df["spread"])
    if spread.empty:
        return {}
    lo, hi = obos_zones(spread)
    cur = float(spread.iloc[-1])
    if cur >= hi:
        state = "OVERBOUGHT ZONE"
        note = "Breadth is stretched to the upside — near-term mean reversion risk is elevated."
    elif cur <= lo:
        state = "OVERSOLD ZONE"
        note = "Breadth is washed out — selling pressure is historically near exhaustion."
    elif cur > 0:
        state = "POSITIVE"
        note = "More stocks stretched up than down — constructive, not extreme."
    else:
        state = "NEGATIVE"
        note = "More stocks stretched down than up — weak, not yet washed out."
    last = df.loc[spread.index[-1]]
    return {
        "spread": cur,
        "ob": float(last.get("ob_pct", np.nan)),
        "os": float(last.get("os_pct", np.nan)),
        "zone_lo": lo,
        "zone_hi": hi,
        "state": state,
        "note": note,
        "pct_rank": percentile_of_last(spread),
        "as_of": spread.index[-1],
        "eligible": int(last.get("eligible")) if pd.notna(last.get("eligible", np.nan)) else None,
    }


# ---------------------------------------------------------------------------
# Relative strength / style rotation
# ---------------------------------------------------------------------------
def relative_strength(
    num: pd.Series, den: pd.Series, fast: int = 50, slow: int = 200
) -> pd.DataFrame:
    """Ratio num/den on common dates + fast/slow moving averages of the ratio."""
    a, b = clean_series(num), clean_series(den)
    if a.empty or b.empty:
        return pd.DataFrame(columns=["ratio", "ma_fast", "ma_slow"], dtype=float)
    df = pd.concat([a.rename("a"), b.rename("b")], axis=1, join="inner").dropna()
    df = df[(df["a"] > 0) & (df["b"] > 0)]
    ratio = df["a"] / df["b"]
    return pd.DataFrame({
        "ratio": ratio,
        "ma_fast": ratio.rolling(fast, min_periods=fast).mean(),
        "ma_slow": ratio.rolling(slow, min_periods=slow).mean(),
    })


def rebase_to(df: pd.DataFrame, start: pd.Timestamp, base: float = 100.0) -> pd.DataFrame:
    """Slice from `start` and rebase every column by the ratio's first value
    in the window (so the MAs stay on the same scale as the ratio)."""
    if df is None or df.empty:
        return df
    w = df.loc[df.index >= pd.Timestamp(start)]
    first = w["ratio"].dropna()
    if first.empty:
        return w.iloc[0:0]
    return w / float(first.iloc[0]) * base


def relative_strength_status(df: pd.DataFrame, slope_days: int = 20) -> dict:
    """Classify the ratio's trend from price-vs-MA stacking.

      ratio > 50D > 200D  -> RELATIVE STRENGTH  (leading, uptrend)
      ratio < 50D < 200D  -> RELATIVE WEAKNESS  (lagging, downtrend)
      ratio > 50D < 200D  -> IMPROVING          (bounce inside a downtrend)
      ratio < 50D > 200D  -> FADING             (pullback inside an uptrend)
    """
    if df is None or df.empty:
        return {"label": "NO DATA", "tone": "neutral", "detail": ""}
    last = df.dropna(subset=["ratio"]).iloc[-1]
    r, f, s = float(last["ratio"]), float(last.get("ma_fast", np.nan)), float(last.get("ma_slow", np.nan))
    if np.isnan(f) or np.isnan(s):
        return {"label": "INSUFFICIENT HISTORY", "tone": "neutral", "detail": "needs 200 sessions"}
    if r > f and f > s:
        label, tone = "RELATIVE STRENGTH", "pos"
    elif r < f and f < s:
        label, tone = "RELATIVE WEAKNESS", "neg"
    elif r > f and f <= s:
        label, tone = "IMPROVING", "neutral"
    else:
        label, tone = "FADING", "neutral"
    fast_series = df["ma_fast"].dropna()
    slope = float("nan")
    if len(fast_series) > slope_days:
        slope = float((fast_series.iloc[-1] / fast_series.iloc[-1 - slope_days] - 1.0) * 100.0)
    return {
        "label": label,
        "tone": tone,
        "ratio_vs_fast_pct": (r / f - 1.0) * 100.0,
        "ratio_vs_slow_pct": (r / s - 1.0) * 100.0,
        "fast_slope_pct": slope,
        "detail": (
            f"ratio {(r / f - 1) * 100:+.1f}% vs 50D, {(r / s - 1) * 100:+.1f}% vs 200D"
            + ("" if np.isnan(slope) else f" · 50D slope {slope:+.1f}% / {slope_days}d")
        ),
    }


# ---------------------------------------------------------------------------
# Valuation statistics (multiple vs its own history)
# ---------------------------------------------------------------------------
def years_since_at_or_below(s: pd.Series) -> float | None:
    """Years since the series was last at or below its current value.

    None means the current value is the lowest in the available history.
    """
    s = clean_series(s)
    if len(s) < 2:
        return None
    cur = s.iloc[-1]
    prior = s.iloc[:-1]
    hit = prior[prior <= cur]
    if hit.empty:
        return None
    return float((s.index[-1] - hit.index[-1]).days / 365.25)


def years_since_at_or_above(s: pd.Series) -> float | None:
    s = clean_series(s)
    if len(s) < 2:
        return None
    cur = s.iloc[-1]
    prior = s.iloc[:-1]
    hit = prior[prior >= cur]
    if hit.empty:
        return None
    return float((s.index[-1] - hit.index[-1]).days / 365.25)


def valuation_stats(s: pd.Series, years: float = 5.0) -> dict:
    """Current multiple vs its trailing `years` history (by calendar time).

    `span_years` reports the history actually available — when a feed is
    younger than the requested lookback the UI must say so instead of calling
    a 7-month range a "5-year percentile".
    """
    # Time-weight on a business-day grid (same as the scoring engine): a feed
    # that mixes monthly history with daily live points would otherwise let the
    # densely-sampled recent stretch outvote years of monthly readings.
    full = to_business_daily(s)
    if full.empty:
        return {}
    w = window_by_years(full, years)
    cur = float(w.iloc[-1])
    avg = float(w.mean())
    std = float(w.std()) if len(w) > 1 else float("nan")
    span = (w.index[-1] - w.index[0]).days / 365.25 if len(w) > 1 else 0.0
    pct = percentile_of_last(w)
    yrs_low = years_since_at_or_below(w)
    yrs_high = years_since_at_or_above(w)

    def _extreme(kind: str, yrs: float | None) -> str:
        # yrs None = most extreme in the whole window -> the window span is the claim.
        n = int(np.floor(span if yrs is None else yrs))
        return f"VALUATION AT {n}-YEAR {kind}" if n >= 1 else f"VALUATION NEAR RANGE {kind}"

    if pct <= 10:
        headline, tone = _extreme("LOWS", yrs_low), "pos"
    elif pct <= 30:
        headline, tone = "BELOW-AVERAGE VALUATION", "pos"
    elif pct >= 90:
        headline, tone = _extreme("HIGHS", yrs_high), "neg"
    elif pct >= 70:
        headline, tone = "RICH VALUATION", "neg"
    else:
        headline, tone = "VALUATION NEAR HISTORICAL NORMS", "neutral"
    return {
        "current": cur,
        "as_of": w.index[-1],
        "avg": avg,
        "std": std,
        "z": (cur - avg) / std if std and not np.isnan(std) and std > 0 else float("nan"),
        "pct_rank": pct,
        "vs_avg_pct": (cur / avg - 1.0) * 100.0 if avg else float("nan"),
        "min": float(w.min()),
        "max": float(w.max()),
        "span_years": float(span),
        "requested_years": float(years),
        "n": int(len(w)),
        "years_since_lower": yrs_low,
        "years_since_higher": yrs_high,
        "headline": headline,
        "tone": tone,
    }


# ---------------------------------------------------------------------------
# Standardized signals (rolling 1-year z-score)
# ---------------------------------------------------------------------------
def to_business_daily(s: pd.Series) -> pd.Series:
    """Put any-frequency series on a business-day grid (forward-filled), so a
    "1-year" window means 252 sessions for daily, weekly and monthly feeds."""
    s = clean_series(s)
    if s.empty:
        return s
    return s.resample("B").last().ffill()


def rolling_zscore(s: pd.Series, window: int = TRADING_DAYS_PER_YEAR, min_periods: int | None = None) -> pd.Series:
    """Point-in-time rolling z-score (each value vs its own trailing window)."""
    s = clean_series(s)
    if s.empty:
        return s
    mp = min_periods if min_periods is not None else max(20, window // 2)
    mean = s.rolling(window, min_periods=mp).mean()
    std = s.rolling(window, min_periods=mp).std()
    return ((s - mean) / std.where(std > 0)).dropna()


def zscore_label(z: float) -> str:
    if z is None or np.isnan(z):
        return "NO DATA"
    if z <= -2:
        return "AT A NEGATIVE EXTREME"
    if z <= -1:
        return "STRETCHED LOW"
    if z >= 2:
        return "AT A POSITIVE EXTREME"
    if z >= 1:
        return "STRETCHED HIGH"
    return "WITHIN NORMAL RANGE"


def standardize_signal(
    s: pd.Series,
    window: int = TRADING_DAYS_PER_YEAR,
    min_obs: int = 60,
) -> dict:
    """Rolling 1-year z-score of a signal + 1-year percentile of its latest value.

    Returns {} when there isn't enough history to standardize honestly.
    """
    daily = to_business_daily(s)
    if len(daily) < min_obs:
        return {}
    z = rolling_zscore(daily, window=window)
    if z.empty:
        return {}
    last_year = daily.iloc[-window:]
    latest_z = float(z.iloc[-1])
    pct = percentile_of_last(last_year)
    label = zscore_label(latest_z)
    if "NEGATIVE" in label:
        take = (f"The signal has fallen into the bottom {pct:.1f}% of its one-year range — "
                "a deeply negative extreme vs its own recent history.")
    elif "POSITIVE" in label:
        take = (f"The signal has risen into the top {100 - pct:.1f}% of its one-year range — "
                "a deeply positive extreme vs its own recent history.")
    elif "LOW" in label:
        take = f"Stretched below its one-year mean (1Y percentile {pct:.0f}%), not yet extreme."
    elif "HIGH" in label:
        take = f"Stretched above its one-year mean (1Y percentile {pct:.0f}%), not yet extreme."
    else:
        take = f"Inside its normal one-year range (1Y percentile {pct:.0f}%)."
    return {
        "z": z.iloc[-window:],
        "latest_z": latest_z,
        "pct_1y": pct,
        "label": label,
        "takeaway": take,
        "as_of": daily.index[-1],
        "latest_raw": float(daily.iloc[-1]),
    }


def oriented_extreme(z: float, direction: str | None) -> str:
    """Translate a raw z into the dashboard's top/bottom vocabulary."""
    if z is None or np.isnan(z) or direction is None:
        return "—"
    oz = z if direction == "risk_high_is_top" else -z
    if oz >= 2:
        return "top-risk extreme"
    if oz >= 1:
        return "leaning top-risk"
    if oz <= -2:
        return "bottom-setup extreme"
    if oz <= -1:
        return "leaning bottom-setup"
    return "neutral"
