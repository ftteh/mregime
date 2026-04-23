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
    s["russell2000"] = D.russell2000()
    s["nasdaq"] = D.nasdaq_composite()

    # ---- VIX term structure
    s["vix_term_9d_1m"] = D.vix_term_9d_1m()
    s["vix_term_1m_3m"] = D.vix_term_1m_3m()

    # ---- Yield curve + re-steepening
    s["curve_2s10s"] = D.curve_2s10s()
    s["curve_3m10y"] = D.curve_3m10y()
    s["curve_resteep_2s10s"] = D.curve_resteep_2s10s()

    # ---- Credit spread velocity
    s["hy_spread_velocity"] = D.hy_spread_velocity()

    # ---- Macro context
    s["dxy"] = D.dxy()
    s["real_yield_10y"] = D.real_yield_10y()
    s["copper_gold"] = D.copper_gold_ratio()

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


# ---------------------------------------------------------------------------
# Historical composite — reconstruct the regime score over time
# ---------------------------------------------------------------------------
def historical_pillar_scores(raw: RawFrame) -> pd.DataFrame:
    """
    Daily DataFrame with one column per pillar, values = pillar score 0-100 over time.
    Built purely from past data via rolling_percentile (no look-ahead).

    Used by the SPX regime-overlay chart and by pillar_momentum().
    """
    per_indicator: Dict[str, pd.Series] = {}
    for key, spec in INDICATORS_BY_KEY.items():
        s = raw.series.get(key, pd.Series(dtype=float))
        if s is None or s.empty:
            continue
        # Resample to business-day grid so pillar avg aligns across mixed frequencies.
        # Some feeds (AAII/NAAIM/put-call cache) occasionally emit duplicate
        # timestamps; resample() then reindexes and raises on duplicates, so
        # dedupe first.
        s_d = s.copy()
        # Ensure DatetimeIndex — some fallback fetchers can emit string/int indices
        if not isinstance(s_d.index, pd.DatetimeIndex):
            try:
                s_d.index = pd.to_datetime(s_d.index, errors="coerce")
                s_d = s_d[s_d.index.notna()]
                if s_d.empty or not isinstance(s_d.index, pd.DatetimeIndex):
                    continue
            except Exception:
                continue
        # Strip timezone for resample
        if getattr(s_d.index, "tz", None) is not None:
            s_d.index = s_d.index.tz_localize(None)
        s_d = s_d.sort_index()
        s_d = s_d[~s_d.index.duplicated(keep="last")]
        try:
            s_d = s_d.resample("B").ffill()
        except Exception:
            # Last-resort: skip malformed indicator rather than kill the whole panel
            continue
        pct = rolling_percentile(s_d)
        score = pct.apply(lambda p, d=spec.direction: orient_score(p, d))
        per_indicator[key] = score

    if not per_indicator:
        return pd.DataFrame()

    df = pd.DataFrame(per_indicator)
    buckets: Dict[str, pd.Series] = {}
    for bkey in BUCKET_WEIGHTS:
        cols = [k for k, sp in INDICATORS_BY_KEY.items()
                if sp.bucket == bkey and k in df.columns]
        if cols:
            buckets[bkey] = df[cols].mean(axis=1, skipna=True)
    return pd.DataFrame(buckets)


def historical_composite(raw: RawFrame) -> pd.Series:
    """Rolling 0-100 composite regime score over time (weighted pillar average)."""
    pillars = historical_pillar_scores(raw)
    if pillars.empty:
        return pd.Series(dtype=float, name="composite")
    w = pd.Series(BUCKET_WEIGHTS).reindex(pillars.columns).fillna(0.0)
    if w.sum() == 0:
        return pd.Series(dtype=float, name="composite")
    valid = pillars.notna()
    weighted = pillars.fillna(0.0).mul(w, axis=1).sum(axis=1)
    denom = valid.mul(w, axis=1).sum(axis=1).replace(0, np.nan)
    out = (weighted / denom).dropna()
    out.name = "composite"
    return out


# ---------------------------------------------------------------------------
# Exposure recommendation — map composite score + cluster to an actionable position
# ---------------------------------------------------------------------------
def exposure_recommendation(
    composite_score: float,
    cluster: dict | None = None,
) -> dict:
    """
    Translate a 0-100 composite regime score (plus optional cluster state) into
    an explicit net equity exposure + tail-hedge recommendation.

    Returns dict with:
      - net_pct      : int, suggested net long % (can exceed 100 = levered long)
      - hedge_pct    : float, suggested notional in OTM SPX puts as % of NAV
      - label        : short action label ("TRIM", "BACK UP TRUCK", etc.)
      - color        : hex color for UI (matches gauge regime colors)
      - conviction   : "High" / "Medium" / "Low"
      - rationale    : one-line explanation
      - cluster_override : bool — True if cluster forced a deviation from the base mapping

    The rule-of-thumb mapping below is tunable. Cluster overrides:
      - top_cluster ≥ 4    : -20pp to net, +0.5% to hedge minimum
      - bottom_cluster ≥ 4 : +15pp to net, hedge cleared
    """
    if composite_score is None or (isinstance(composite_score, float) and np.isnan(composite_score)):
        return {
            "net_pct": None, "hedge_pct": None,
            "label": "NO DATA", "color": "#555",
            "conviction": "—", "rationale": "Composite unavailable.",
            "cluster_override": False,
        }

    s = float(composite_score)

    # Base mapping from composite regime → exposure
    if s < 15:
        base = {"net": 130, "hedge": 0.0, "label": "BACK UP THE TRUCK",  "color": "#16a085"}
    elif s < 35:
        base = {"net": 115, "hedge": 0.0, "label": "SCALE IN",            "color": "#27ae60"}
    elif s < 45:
        base = {"net": 100, "hedge": 0.0, "label": "FULLY INVESTED",      "color": "#3498db"}
    elif s < 65:
        base = {"net": 90,  "hedge": 0.0, "label": "STANDARD ALLOCATION", "color": "#95a5a6"}
    elif s < 85:
        base = {"net": 50,  "hedge": 0.5, "label": "TRIM",                "color": "#e67e22"}
    else:
        base = {"net": 20,  "hedge": 1.0, "label": "MAX DEFENSIVE",       "color": "#c0392b"}

    net = base["net"]
    hedge = base["hedge"]
    label = base["label"]
    color = base["color"]
    cluster_override = False
    conviction = "Low"

    top_n = int(cluster.get("top_cluster_count", 0)) if cluster else 0
    bot_n = int(cluster.get("bottom_cluster_count", 0)) if cluster else 0

    # Cluster overrides — the whole point of clusters is they trump noise
    if top_n >= 4:
        net = max(0, net - 20)
        hedge = max(hedge, 0.5) + 0.5  # at least 0.5%, plus 0.5 more on top
        cluster_override = True
        label = f"TOP CLUSTER ({top_n}) — CUT"
        color = "#c0392b"
        conviction = "High"
    elif bot_n >= 4:
        net = min(150, net + 15)
        hedge = 0.0
        cluster_override = True
        label = f"BOTTOM CLUSTER ({bot_n}) — ADD"
        color = "#16a085"
        conviction = "High"
    else:
        # Conviction without cluster: extremes = Medium, mid-range = Low
        if s < 25 or s >= 75:
            conviction = "Medium"
        elif 40 <= s <= 60:
            conviction = "Low"
        else:
            conviction = "Medium"

    rationale_bits: list[str] = []
    if cluster_override:
        rationale_bits.append(
            f"{top_n or bot_n} indicators clustered at "
            f"{'top' if top_n >= 4 else 'bottom'} — overrides base reading"
        )
    rationale_bits.append(f"composite {s:.0f}")
    rationale = " · ".join(rationale_bits)

    return {
        "net_pct": int(round(net)),
        "hedge_pct": float(hedge),
        "label": label,
        "color": color,
        "conviction": conviction,
        "rationale": rationale,
        "cluster_override": cluster_override,
    }


def pillar_momentum(raw: RawFrame) -> dict:
    """
    Rate-of-change per pillar: today score, 1 week ago, 1 month ago.
    Returns dict of pillar -> {today, 1w, 1m, d_1w, d_1m} where d_1w / d_1m are
    the change (today - prior). Positive delta = moving toward top-risk/complacency.
    """
    hist = historical_pillar_scores(raw).dropna(how="all")
    if hist.empty:
        return {}
    hist = hist.ffill()

    def _ago(n: int) -> pd.Series:
        if len(hist) <= n:
            return hist.iloc[0]
        return hist.iloc[-(n + 1)]

    today = hist.iloc[-1]
    wk = _ago(5)
    mo = _ago(21)

    out: dict = {}
    for bkey in hist.columns:
        t = float(today.get(bkey, np.nan)) if not pd.isna(today.get(bkey, np.nan)) else np.nan
        w = float(wk.get(bkey, np.nan)) if not pd.isna(wk.get(bkey, np.nan)) else np.nan
        m = float(mo.get(bkey, np.nan)) if not pd.isna(mo.get(bkey, np.nan)) else np.nan
        out[bkey] = {
            "today": t,
            "w_ago": w,
            "m_ago": m,
            "d_1w": (t - w) if not (np.isnan(t) or np.isnan(w)) else np.nan,
            "d_1m": (t - m) if not (np.isnan(t) or np.isnan(m)) else np.nan,
        }
    return out
