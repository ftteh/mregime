"""
Indicator engine: normalize raw data to 0-100 percentile scores,
orient them so HIGH = top-risk / complacency, and produce the composite.
"""
from __future__ import annotations

import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Callable, Dict

import numpy as np
import pandas as pd

from . import data as D
from .config import (
    BUCKET_WEIGHTS,
    INDICATORS_BY_KEY,
    MIN_OBS,
    ROLLING_WINDOW_DAYS,
    theme_for,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------
def rolling_percentile(s: pd.Series, window: int = ROLLING_WINDOW_DAYS) -> pd.Series:
    """Return rolling percentile rank (0-100) of s over `window` observations.

    Equivalent to ranking the current observation against the trailing window via
    ``(x <= x[-1]).mean() * 100``. Implemented with the vectorized C-level
    ``Rolling.rank`` (method="max" counts ties as values <= current, matching the
    ``<=`` comparison) instead of a Python-level apply — same result, ~10x faster.
    """
    if s is None or s.empty:
        return pd.Series(dtype=float)
    win = min(window, max(30, len(s) // 2 if len(s) > 60 else len(s)))
    return s.rolling(win, min_periods=max(30, win // 4)).rank(
        pct=True, method="max"
    ) * 100.0


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


def apply_transform(s: pd.Series, transform: str | None, window: int = 252) -> pd.Series:
    """Pre-percentile transform for non-stationary series.

    "zscore" re-centres a trending series on its own rolling regime so the
    downstream percentile measures deviation-from-trend rather than absolute
    level (which would pin at 0/100 during long drifts). Monotonic within each
    window, so the indicator's direction/orientation is unchanged.
    """
    if transform != "zscore" or s is None or s.empty:
        return s
    win = min(window, max(30, len(s) // 2 if len(s) > 60 else len(s)))
    mean = s.rolling(win, min_periods=max(30, win // 4)).mean()
    std = s.rolling(win, min_periods=max(30, win // 4)).std()
    z = (s - mean) / std.replace(0, np.nan)
    return z.dropna()


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


def _empty_series(name: str = "") -> pd.Series:
    return pd.Series(dtype=float, name=name or None)


def _max_fetch_workers(task_count: int) -> int:
    default_workers = 10
    try:
        configured = int(os.getenv("QUANT_DASH_MAX_WORKERS", str(default_workers)))
    except ValueError:
        configured = default_workers
    return max(1, min(task_count, configured))


def _fetch_parallel(tasks: dict[str, Callable[[], pd.Series]]) -> Dict[str, pd.Series]:
    """
    Fetch independent IO-bound data sources concurrently.

    Most dashboard latency is network/API wait time (FRED, yfinance, CBOE,
    CFTC, CNN, etc.). Keeping each fetcher isolated preserves the existing
    failure behavior: one dead source becomes an empty series, not a dead app.
    """
    if not tasks:
        return {}

    out: Dict[str, pd.Series] = {}
    with ThreadPoolExecutor(max_workers=_max_fetch_workers(len(tasks))) as executor:
        future_to_key = {executor.submit(fn): key for key, fn in tasks.items()}
        for future in as_completed(future_to_key):
            key = future_to_key[future]
            try:
                value = future.result()
                out[key] = value if isinstance(value, pd.Series) else _empty_series(key)
            except Exception as e:
                log.info("raw fetch failed for %s: %s", key, str(e)[:120])
                out[key] = _empty_series(key)
    return out


def _ad_line_proxy() -> pd.Series:
    """A/D line proxy: sign of daily returns across a liquid S&P sample.

    Reuses the shared S&P close panel (top 100, last ~1y) instead of issuing its
    own bulk download. The 1-year tail reproduces the original period="1y" input
    to the cumulative advance-decline calculation.
    """
    try:
        panel, wanted = D._panel_columns(100)
        if not wanted:
            return _empty_series("ad_line_slope")
        closes = panel[wanted].tail(252)  # ~1 trading year
        rets = closes.pct_change()
        ad = (rets > 0).sum(axis=1) - (rets < 0).sum(axis=1)
        ad.name = "ad_line_slope"
        return ad.cumsum()
    except Exception as e:
        log.info("A/D line proxy failed: %s", str(e)[:120])
        return _empty_series("ad_line_slope")


def _aaii_bull_bear_spread() -> pd.Series:
    try:
        aaii = D.aaii_sentiment()
        return aaii["spread"] if "spread" in aaii.columns else _empty_series("aaii_bull_bear")
    except Exception as e:
        log.info("AAII fetch failed: %s", str(e)[:120])
        return _empty_series("aaii_bull_bear")


def build_raw() -> RawFrame:
    s: Dict[str, pd.Series] = {}
    meta: Dict[str, dict] = {}

    # Fresh build → drop any memoized fetch results from a prior build so shared
    # sources (^VIX, S&P close panel, GEX proxy, …) are fetched once per build
    # but still refreshed live on the next cache miss / "Refresh data".
    D.reset_fetch_memo()

    tasks: dict[str, Callable[[], pd.Series]] = {
        # ---- Credit & Liquidity
        "hy_spread": D.fred_hy_spread,
        "ig_spread": D.fred_ig_spread,
        "net_liquidity": D.fred_net_liquidity,
        "financial_conditions": D.fred_nfci,
        "move_index": D.move_index,

        # ---- Market / Breadth / Momentum
        "spx": D.spx,
        "pct_above_200dma": D.breadth_pct_above_200dma,
        "new_highs_lows": D.new_highs_minus_lows,
        "ad_line_slope": _ad_line_proxy,

        # ---- Sentiment & Positioning
        "naaim": D.naaim_exposure,
        "aaii_bull_bear": _aaii_bull_bear_spread,
        "fear_greed": D.fear_greed_index,
        "put_call": D.put_call_ratio,
        "vix": D.vix,
        "vvix": D.vvix,
        "skew": D.skew,

        # ---- Valuation
        "equity_risk_premium": D.equity_risk_premium,

        # ---- Advanced / bonus
        "corr_cluster": D.correlation_cluster,
        "move_vix_div": D.move_vs_vix_spread,
        "dix": D.dix_proxy,
        "cta_positioning": D.cftc_cta_positioning,
        "russell2000": D.russell2000,
        "nasdaq": D.nasdaq_composite,

        # ---- VIX term structure
        "vix_term_9d_1m": D.vix_term_9d_1m,
        "vix_term_1m_3m": D.vix_term_1m_3m,

        # ---- Yield curve + re-steepening
        "curve_2s10s": D.curve_2s10s,
        "curve_3m10y": D.curve_3m10y,
        "curve_resteep_2s10s": D.curve_resteep_2s10s,

        # ---- Credit spread velocity
        "hy_spread_velocity": D.hy_spread_velocity,

        # ---- Macro context
        "dxy": D.dxy,
        "real_yield_10y": D.real_yield_10y,
        "copper_gold": D.copper_gold_ratio,

        # ---- Interbank Funding / Repo Market Stress
        "fra_ois_spread": D.fra_ois_spread,
        "sofr_spread": D.sofr_spread,

        # ---- Gamma Exposure / Options Market Structure
        "gamma_exposure": D.gamma_exposure_proxy,
        "gamma_flip_zone": D.gamma_flip_zone_distance,
        "index_put_call": D.index_put_call_ratio,
    }
    s.update(_fetch_parallel(tasks))

    spx_px = s.get("spx", _empty_series("spx"))
    s["rsi_spx"] = rsi(spx_px, 14) if not spx_px.empty else pd.Series(dtype=float)

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
def score_indicators(raw: RawFrame, per_indicator: pd.DataFrame | None = None) -> pd.DataFrame:
    """Return DataFrame: key | label | bucket | raw | percentile | score | direction.

    The live score is the most recent point of per_indicator_history (same engine
    as the regime-band chart), so the gauge endpoint matches the chart endpoint
    and weekly/daily indicators share one calendar window. Indicators below
    MIN_OBS observations — or whose history is too short for the percentile's
    min_periods — resolve to NaN and are dropped from the composite and cluster.
    """
    df = per_indicator_history(raw) if per_indicator is None else per_indicator
    rows = []
    for key, spec in INDICATORS_BY_KEY.items():
        series = raw.series.get(key, pd.Series(dtype=float))
        raw_value = float(series.iloc[-1]) if series is not None and not series.empty else np.nan
        n_obs = int(raw.meta.get(key, {}).get("n", 0))

        score = np.nan
        if df is not None and not df.empty and key in df.columns and n_obs >= MIN_OBS:
            col = df[key].dropna()
            if not col.empty:
                score = float(col.iloc[-1])
        # Back-derive the displayed percentile from the oriented score.
        if np.isnan(score):
            pct = np.nan
        elif spec.direction == "contrarian_high_is_top":
            pct = 100.0 - score
        else:
            pct = score

        rows.append({
            "key": key,
            "label": spec.label,
            "bucket": spec.bucket,
            "direction": spec.direction,
            "raw": raw_value,
            "percentile": pct,
            "score": score,
            "weight": spec.weight,
            "theme": theme_for(key),
            "as_of": raw.meta.get(key, {}).get("as_of"),
            "n_obs": n_obs,
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
    all_weight = sum(BUCKET_WEIGHTS.values())
    covered_weight = (sum(b["weight"] for b in valid) / all_weight) if all_weight else 0.0
    if not valid:
        return {"composite": np.nan, "buckets": bucket_scores, "covered_weight": 0.0}
    total_w = sum(b["weight"] for b in valid)
    comp = sum(b["score"] * b["weight"] for b in valid) / total_w
    return {
        "composite": float(comp),
        "buckets": bucket_scores,
        # Fraction of the model's total bucket weight that actually has data.
        # < 1.0 means the composite was renormalised over a partial model (e.g.
        # no FRED key → the 40% Credit pillar is dark) — surfaced as confidence.
        "covered_weight": float(covered_weight),
    }


# ---------------------------------------------------------------------------
# Cluster detection — "are we in a confluence top/bottom?"
# ---------------------------------------------------------------------------
def cluster_signal(scores_df: pd.DataFrame) -> dict:
    """Detect a confluence of extremes — counting independent THEMES, not raw
    indicators, so a single vol event (VIX/VVIX/SKEW/term-structure all firing)
    counts once rather than four times. Also reports how many buckets the
    extremes span, so the override can require cross-pillar confirmation.
    """
    extreme_top = scores_df[scores_df["score"] >= 85]
    extreme_bot = scores_df[scores_df["score"] <= 15]

    def _themes(df: pd.DataFrame) -> int:
        if df.empty or "theme" not in df.columns:
            return int(len(df))
        return int(df["theme"].nunique())

    return {
        # Raw indicator counts (kept for display continuity)
        "top_cluster_count": int(len(extreme_top)),
        "bottom_cluster_count": int(len(extreme_bot)),
        # De-duplicated independent-theme counts (drive the override)
        "top_theme_count": _themes(extreme_top),
        "bottom_theme_count": _themes(extreme_bot),
        # Cross-pillar breadth of the extremes
        "top_buckets": sorted(extreme_top["bucket"].unique().tolist()),
        "bottom_buckets": sorted(extreme_bot["bucket"].unique().tolist()),
        "top_names": extreme_top["label"].tolist(),
        "bottom_names": extreme_bot["label"].tolist(),
    }


# ---------------------------------------------------------------------------
# Historical composite — reconstruct the regime score over time
# ---------------------------------------------------------------------------
def per_indicator_history(raw: RawFrame) -> pd.DataFrame:
    """
    Daily (business-day) DataFrame, one column per indicator, values = the
    oriented 0-100 top-risk score over time. Built purely from past data via
    rolling_percentile (point-in-time, no look-ahead).

    This is the single source of truth for scoring: the live gauge reads the
    last row (see score_indicators) and the regime-band chart / pillar momentum
    aggregate the full history — so the gauge can never disagree with the chart
    endpoint. Indicators with too little history (min_periods) stay NaN and are
    excluded everywhere downstream.
    """
    per_indicator: Dict[str, pd.Series] = {}
    for key, spec in INDICATORS_BY_KEY.items():
        s = raw.series.get(key, pd.Series(dtype=float))
        if s is None or s.empty:
            continue
        # Single min-observation gate, enforced here so the live scores (last row)
        # and the historical chart exclude exactly the same thin indicators —
        # otherwise a 30-59-obs series would be gated live but counted on the
        # chart, breaking the gauge==chart-endpoint guarantee.
        if int(raw.meta.get(key, {}).get("n", 0)) < MIN_OBS:
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
        s_d = apply_transform(s_d, spec.transform, spec.transform_window)
        if s_d.empty:
            continue
        pct = rolling_percentile(s_d)
        score = pct.apply(lambda p, d=spec.direction: orient_score(p, d))
        per_indicator[key] = score

    if not per_indicator:
        return pd.DataFrame()
    return pd.DataFrame(per_indicator)


def _weighted_bucket_mean(per_ind: pd.DataFrame, cols: list[str]) -> pd.Series:
    """Per-row weighted mean of indicator scores in `cols`, skipping NaNs.

    Mirrors composite()'s within-bucket weighting so the historical pillar path
    and the live composite use one consistent aggregation.
    """
    w = np.array([INDICATORS_BY_KEY[c].weight for c in cols], dtype=float)
    sub = per_ind[cols]
    mask = sub.notna()
    num = (sub.fillna(0.0) * w).sum(axis=1)
    den = (mask * w).sum(axis=1).replace(0, np.nan)
    return num / den


def historical_pillar_scores(raw: RawFrame, per_indicator: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Daily DataFrame with one column per pillar, values = pillar score 0-100 over
    time (weighted mean of that pillar's indicator scores). Accepts a precomputed
    per_indicator_history frame to avoid recomputing the rolling-percentile pass.
    """
    df = per_indicator_history(raw) if per_indicator is None else per_indicator
    if df is None or df.empty:
        return pd.DataFrame()

    buckets: Dict[str, pd.Series] = {}
    for bkey in BUCKET_WEIGHTS:
        cols = [k for k, sp in INDICATORS_BY_KEY.items()
                if sp.bucket == bkey and k in df.columns]
        if cols:
            buckets[bkey] = _weighted_bucket_mean(df, cols)
    return pd.DataFrame(buckets)


def historical_composite(raw: RawFrame, pillars: pd.DataFrame | None = None) -> pd.Series:
    """Rolling 0-100 composite regime score over time (weighted pillar average).

    Accepts an optional precomputed `pillars` frame (from historical_pillar_scores)
    so callers that also need pillar momentum don't pay for the rolling-percentile
    pass twice.
    """
    if pillars is None:
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
CLUSTER_MIN_THEMES = 3   # distinct independent themes needed to fire an override
CLUSTER_MIN_BUCKETS = 2  # spanning at least this many pillars (cross-confirmation)


def exposure_recommendation(
    composite_score: float,
    cluster: dict | None = None,
    covered_weight: float = 1.0,
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

    Cluster overrides require a confluence of independent THEMES spanning multiple
    pillars (not a raw indicator count), so one collinear vol event can't fake a
    high-conviction signal. `covered_weight` (fraction of the model with data)
    caps conviction: a partial-model reading can never be "High".
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

    # Theme-deduplicated counts + cross-pillar breadth drive the override.
    top_themes = int(cluster.get("top_theme_count", cluster.get("top_cluster_count", 0))) if cluster else 0
    bot_themes = int(cluster.get("bottom_theme_count", cluster.get("bottom_cluster_count", 0))) if cluster else 0
    top_buckets = len(cluster.get("top_buckets", [])) if cluster else 0
    bot_buckets = len(cluster.get("bottom_buckets", [])) if cluster else 0

    # Fire on a cross-pillar confluence (≥3 themes spanning ≥2 buckets) OR a
    # strong single-pillar one (≥4 distinct themes) — the latter so a genuine
    # credit-led blowout, whose themes all live in the credit_liquidity pillar,
    # is not silently blocked by the cross-pillar requirement.
    def _fires(themes: int, buckets: int) -> bool:
        return themes >= CLUSTER_MIN_THEMES and (
            buckets >= CLUSTER_MIN_BUCKETS or themes >= CLUSTER_MIN_THEMES + 1
        )

    top_fire = _fires(top_themes, top_buckets)
    bot_fire = _fires(bot_themes, bot_buckets)

    # Cluster overrides — the whole point of clusters is they trump noise
    if top_fire:
        net = max(0, net - 20)
        hedge = max(hedge, 0.5) + 0.5  # at least 0.5%, plus 0.5 more on top
        cluster_override = True
        label = f"TOP CLUSTER ({top_themes} themes) — CUT"
        color = "#c0392b"
        conviction = "High"
    elif bot_fire:
        net = min(150, net + 15)
        hedge = 0.0
        cluster_override = True
        label = f"BOTTOM CLUSTER ({bot_themes} themes) — ADD"
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

    # Partial-model honesty gate: a reading off an incomplete model (e.g. the
    # 40%-weight Credit pillar offline) is capped at Low conviction.
    low_coverage = covered_weight < 0.8
    if low_coverage:
        conviction = "Low"

    rationale_bits: list[str] = []
    if cluster_override:
        themes = top_themes if top_fire else bot_themes
        buckets = top_buckets if top_fire else bot_buckets
        rationale_bits.append(
            f"{themes} independent themes across {buckets} pillars clustered at "
            f"{'top' if top_fire else 'bottom'} — overrides base reading"
        )
    rationale_bits.append(f"composite {s:.0f}")
    if low_coverage:
        rationale_bits.append(f"only {covered_weight*100:.0f}% of model has data")
    rationale = " · ".join(rationale_bits)

    return {
        "net_pct": int(round(net)),
        "hedge_pct": float(hedge),
        "label": label,
        "color": color,
        "conviction": conviction,
        "rationale": rationale,
        "cluster_override": cluster_override,
        "covered_weight": float(covered_weight),
    }


def pillar_momentum(raw: RawFrame, pillars: pd.DataFrame | None = None) -> dict:
    """
    Rate-of-change per pillar: today score, 1 week ago, 1 month ago.
    Returns dict of pillar -> {today, 1w, 1m, d_1w, d_1m} where d_1w / d_1m are
    the change (today - prior). Positive delta = moving toward top-risk/complacency.

    Accepts an optional precomputed `pillars` frame so it can share the
    rolling-percentile pass with historical_composite().
    """
    if pillars is None:
        pillars = historical_pillar_scores(raw)
    hist = pillars.dropna(how="all")
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
