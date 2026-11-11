"""
Indicator engine: normalize raw data to 0-100 percentile scores,
orient them so HIGH = top-risk / complacency, and produce the composite.
"""
from __future__ import annotations

import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from dataclasses import dataclass, field
from typing import Callable, Dict

import numpy as np
import pandas as pd

from . import data as D
from .config import (
    BUCKET_MIN_COVERAGE,
    BUCKET_WEIGHTS,
    CLUSTER_NET_SHARE,
    EXTREME_BOTTOM,
    EXTREME_TOP,
    INDICATORS_BY_KEY,
    MIN_OBS,
    MODEL_MIN_COVERAGE,
    PCT_MIN_PERIODS,
    ROLLING_WINDOW_DAYS,
    IndicatorSpec,
    theme_for,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------
def rolling_percentile(
    s: pd.Series, window: int = ROLLING_WINDOW_DAYS, min_periods: int = PCT_MIN_PERIODS
) -> pd.Series:
    """Percentile rank (0-100) of each observation within its trailing `window`.

    Equivalent to ``(x <= x[-1]).mean() * 100`` over the window, via the
    vectorized ``Rolling.rank`` (method="max" counts ties as values <= current).

    The window grows from `min_periods` to `window` observations and then rolls,
    so a series with three years of history gets a true three-year rank. The
    engine used to halve the window for anything under six years long: HY and IG
    spreads were ranked over 18 months, and Fear & Greed over six. Less than
    `min_periods` of history gives no score at all.
    """
    if s is None or s.empty:
        return pd.Series(dtype=float)
    return s.rolling(window, min_periods=min(min_periods, window)).rank(
        pct=True, method="max"
    ) * 100.0


def latest_percentile(s: pd.Series, window: int = ROLLING_WINDOW_DAYS) -> float:
    """Single-value percentile (current value vs the last `window` obs), for
    charts of series that aren't scored. Ties count as <=, like the engine."""
    if s is None or s.empty:
        return np.nan
    tail = s.tail(window).dropna()
    if tail.empty:
        return np.nan
    return float((tail <= tail.iloc[-1]).mean() * 100.0)


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
    # Keep NaNs (warm-up window, zero-std stretches) instead of deleting rows:
    # dropping them silently shifted rolling-percentile window membership and
    # could report a stale earlier row as the "current" value.
    return z


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
    # Display-only context series (synthetic backfills, legacy mixed-method
    # cache rows). Charts may render them visibly distinct; they never enter
    # scoring/MIN_OBS/cluster. Default keeps 2-arg construction (tests) valid.
    display: Dict[str, pd.Series] = field(default_factory=dict)
    # Quotes from a session still in progress, keyed by ticker. Shown as live
    # context; never scored (see src/session.py).
    live: Dict[str, dict] = field(default_factory=dict)


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
    """Cumulative advance-decline line across all S&P 500 constituents.

    Built over the whole shared close panel (about five years). It used to be
    restarted from zero over the last 252 sessions, which left six months of
    usable history and made the level depend on where the year happened to
    start. The config z-scores it against its own one-year trend.
    """
    try:
        panel, wanted = D._panel_columns()
        if not wanted:
            D.record_provenance("ad_line_slope", "", kind="unavailable")
            return _empty_series("ad_line_slope")
        rets = panel[wanted].pct_change(fill_method=None).iloc[1:]
        ad = ((rets > 0).sum(axis=1) - (rets < 0).sum(axis=1)).astype(float)
        D.record_provenance(
            "ad_line_slope", f"sp500_panel({len(wanted)} constituents)", kind="computed",
            note="A/D proxy from current S&P 500 membership, not NYSE official",
        )
        return ad.cumsum().rename("ad_line_slope")
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
        # Full-universe participation (one shared panel, memoized frame)
        "trend_breadth_10_20": D.trend_breadth_10_20,
        "trend_breadth_eligible": D.trend_breadth_eligible,
        "obos_spread_50d": D.obos_spread_50d,
        "obos_ob_pct": D.obos_overbought_pct,
        "obos_os_pct": D.obos_oversold_pct,

        # ---- Sentiment & Positioning
        "aaii_bull_bear": _aaii_bull_bear_spread,
        "fear_greed": D.fear_greed_index,
        "put_call": D.put_call_ratio,
        "vix": D.vix,
        "vvix": D.vvix,
        "skew": D.skew,

        # ---- Valuation
        "equity_risk_premium": D.equity_risk_premium,
        "ndx_trailing_pe": D.ndx_trailing_pe,
        "ndx_forward_pe": D.ndx_forward_pe,

        # ---- Advanced / bonus
        "corr_cluster": D.correlation_cluster,
        "move_vix_div": D.move_vs_vix_spread,
        "dix": D.dix_proxy,
        "cta_positioning": D.cftc_cta_positioning,
        "margin_debt": D.finra_margin_debt,
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

    # One clock for every feed. Yahoo's unfinished bar is already dropped at the
    # fetch; this also covers a same-day print from any other source or a saved
    # row, so "scores through <date>" is true of every input, not only prices.
    s = {k: D._closed_only(v) for k, v in s.items()}

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

    # Which source ACTUALLY served each indicator this build (primary /
    # fallback / proxy / cache / unavailable) — surfaced in the UI so degraded
    # feeds can't masquerade as their configured source.
    prov = D.get_provenance()
    for k in s:
        meta.setdefault(k, {})["provenance"] = prov.get(k)

    return RawFrame(series=s, meta=meta, display=D.get_display_overlays(), live=D.get_live_quotes())


# ---------------------------------------------------------------------------
# Scoring engine
# ---------------------------------------------------------------------------
# One prepared, business-daily history per indicator (per_indicator_history) and
# one aggregator (aggregate_scores). The live gauge reads the last row of the
# same frame the regime-band chart is drawn from, so the two cannot disagree.
ACTIVE_MIN_OBS = 20  # active readings needed before an event flag is ranked


def _scored_specs() -> Dict[str, IndicatorSpec]:
    return {k: sp for k, sp in INDICATORS_BY_KEY.items() if not sp.display_only}


def _ffill_limit(spec: IndicatorSpec) -> int:
    """Business days a print may be carried forward: its staleness allowance."""
    return int(np.ceil(spec.max_stale_days * 5.0 / 7.0)) + 1


def _business_daily(s: pd.Series, limit: int | None = None) -> pd.Series:
    """One value per business day, the last print carried forward at most `limit` days.

    A weekend-dated print (a month-end P/E, say) first shows on the next business
    day. Gaps longer than `limit` stay empty instead of becoming a flat line of
    one old value.
    """
    out = s.copy()
    if not isinstance(out.index, pd.DatetimeIndex):
        out.index = pd.to_datetime(out.index, errors="coerce")
        out = out[out.index.notna()]
    if out.empty:
        return pd.Series(dtype=float)
    if getattr(out.index, "tz", None) is not None:
        out.index = out.index.tz_localize(None)
    out.index = out.index.normalize()
    out = pd.to_numeric(out, errors="coerce").dropna().sort_index()
    out = out[~out.index.duplicated(keep="last")]
    if out.empty:
        return pd.Series(dtype=float)
    grid = pd.bdate_range(out.index[0], pd.offsets.BDay().rollforward(out.index[-1]))
    return out.reindex(grid, method="ffill", limit=limit)


def _prepared(key: str, spec: IndicatorSpec, raw: RawFrame) -> pd.Series | None:
    """The series the percentile is taken of: business-daily and smoothed."""
    s = raw.series.get(key)
    if s is None or s.empty:
        return None
    # Single min-observation gate, enforced here so the live score (last row)
    # and the historical chart exclude exactly the same thin indicators.
    if int(raw.meta.get(key, {}).get("n", 0)) < MIN_OBS:
        return None
    try:
        x = _business_daily(s, _ffill_limit(spec))
    except Exception:
        return None  # a malformed feed drops out; it doesn't take the panel down
    if x.empty:
        return None
    if spec.smooth > 1:
        x = x.rolling(spec.smooth, min_periods=max(1, spec.smooth // 2)).mean().where(x.notna())
    return x


def _active_only_score(x: pd.Series, direction: str) -> pd.Series:
    """Score for an event flag that reads exactly 0 while it is off.

    Off = 50 (no vote either way). On = 50 to 100 by rank among the active
    readings of the trailing window, or 50 down to 0 for a contrarian flag.
    Ranking the raw series instead scored "off" as the share of zeros in the
    window: an arbitrary number that could reach an extreme with the flag off.
    """
    active = x.where(x > 0)
    pct = active.rolling(ROLLING_WINDOW_DAYS, min_periods=ACTIVE_MIN_OBS).rank(pct=True, method="max") * 100.0
    on = 50.0 + pct / 2.0 if direction == "risk_high_is_top" else 50.0 - pct / 2.0
    score = on.where(x > 0, 50.0).where(x.notna())
    return score.where(x.notna().cumsum() >= PCT_MIN_PERIODS)


def _score_history(x: pd.Series, spec: IndicatorSpec) -> pd.Series:
    if spec.transform == "active_only":
        return _active_only_score(x, spec.direction)
    pct = rolling_percentile(apply_transform(x, spec.transform, spec.transform_window))
    if spec.transform == "stress_only":
        # Below the window median every reading ties at the median: no vote.
        pct = pct.clip(lower=50.0)
    return 100.0 - pct if spec.direction == "contrarian_high_is_top" else pct


def per_indicator_history(raw: RawFrame) -> pd.DataFrame:
    """
    Daily (business-day) DataFrame, one column per scored indicator, values = the
    oriented 0-100 top-risk score over time. Built only from past data via
    rolling_percentile (point-in-time, no look-ahead).

    This is the single source of truth for scoring: the live gauge reads the
    last row (see score_indicators) and the regime-band chart / pillar momentum
    aggregate the full history. Indicators with too little history stay NaN and
    are excluded everywhere downstream; so does one whose last print is older
    than its staleness allowance.
    """
    cols: Dict[str, pd.Series] = {}
    for key, spec in _scored_specs().items():
        x = _prepared(key, spec, raw)
        if x is not None:
            cols[key] = _score_history(x, spec)
    if not cols:
        return pd.DataFrame()
    return _carry_scores_to_edge(
        pd.DataFrame(cols), {k: INDICATORS_BY_KEY[k].max_stale_days for k in cols}
    )


def _carry_scores_to_edge(df: pd.DataFrame, max_stale_days: Dict[str, int] | None = None) -> pd.DataFrame:
    """Carry each indicator's LAST valid score forward to the frame's end date.

    Feeds publish on different lags (FRED a day behind yfinance, weekly
    AAII/CTA/NFCI up to a week behind), so the final rows of the joint frame are
    ragged. Carrying the last known reading forward is point-in-time correct (it
    was the latest information on those days) and keeps the chart endpoint,
    pillar momentum and the gauge in agreement. Only the trailing edge is
    filled; interior gaps are left untouched.

    With `max_stale_days`, a reading older than its allowance is NOT carried: a
    feed that died weeks ago stops voting instead of repeating its last score
    at full weight.
    """
    if df.empty:
        return df
    out = df.copy()
    edge = out.index[-1]
    for col in out.columns:
        last = out[col].last_valid_index()
        if last is None or last == edge:
            continue
        allowed = (max_stale_days or {}).get(col)
        if allowed is not None and (edge - last).days > allowed:
            continue
        out.loc[out.index > last, col] = out.at[last, col]
    return out


def aggregate_scores(per_ind: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """(pillar scores, composite, model coverage) by date.

    Within a pillar, indicators are averaged by their config weight. A pillar is
    scored only when at least BUCKET_MIN_COVERAGE of its indicator weight is
    live, so one surviving feed can't stand in for the whole pillar. Coverage is
    the share of the model's weight that is live, counted indicator by indicator;
    below MODEL_MIN_COVERAGE there is no composite.
    """
    empty = (pd.DataFrame(), pd.Series(dtype=float, name="composite"), pd.Series(dtype=float, name="coverage"))
    if per_ind is None or per_ind.empty:
        return empty
    pillars: Dict[str, pd.Series] = {}
    cover: Dict[str, pd.Series] = {}
    specs = _scored_specs()
    for bkey in BUCKET_WEIGHTS:
        members = {k: sp for k, sp in specs.items() if sp.bucket == bkey}
        total = float(sum(sp.weight for sp in members.values()))
        cols = [k for k in members if k in per_ind.columns]
        if not cols or total <= 0:
            continue
        w = np.array([members[c].weight for c in cols], dtype=float)
        sub = per_ind[cols]
        live_w = (sub.notna() * w).sum(axis=1)
        cov = live_w / total
        score = (sub.fillna(0.0) * w).sum(axis=1) / live_w.replace(0, np.nan)
        pillars[bkey] = score.where(cov >= BUCKET_MIN_COVERAGE)
        cover[bkey] = cov
    if not pillars:
        return empty
    P, C = pd.DataFrame(pillars), pd.DataFrame(cover)
    bw = pd.Series(BUCKET_WEIGHTS, dtype=float)
    w = bw.reindex(P.columns)
    valid = P.notna()
    comp = P.fillna(0.0).mul(w, axis=1).sum(axis=1) / valid.mul(w, axis=1).sum(axis=1).replace(0, np.nan)
    coverage = C.where(valid, 0.0).mul(w, axis=1).sum(axis=1) / float(bw.sum())
    comp = comp.where(coverage >= MODEL_MIN_COVERAGE)
    return P, comp.rename("composite"), coverage.rename("coverage")


def score_indicators(raw: RawFrame, per_indicator: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per indicator: raw value, percentile, oriented score and why it
    is or isn't voting.

    The live score is the last row of per_indicator_history (same engine as the
    regime-band chart). `status` says why a score is missing:
      display  charted only, by design
      offline  no data
      thin     fewer than MIN_OBS observations
      warmup   under a year of history, so no percentile yet
      stale    last print older than the indicator's allowance
    """
    df = per_indicator_history(raw) if per_indicator is None else per_indicator
    have = df is not None and not df.empty
    edge = df.index[-1] if have else None
    rows = []
    for key, spec in INDICATORS_BY_KEY.items():
        series = raw.series.get(key, pd.Series(dtype=float))
        has = series is not None and not series.empty
        raw_value = float(series.iloc[-1]) if has else np.nan
        meta = raw.meta.get(key, {})
        n_obs = int(meta.get("n", 0))
        as_of = meta.get("as_of")

        score, status = np.nan, "ok"
        if spec.display_only:
            status = "display"
        elif not has:
            status = "offline"
        elif n_obs < MIN_OBS or not have or key not in df.columns:
            status = "thin"
        else:
            v = df[key].iloc[-1]
            if pd.notna(v):
                score = float(v)
            elif as_of is not None and (edge - pd.Timestamp(as_of).normalize()).days > spec.max_stale_days:
                status = "stale"
            else:
                status = "warmup"

        # Back-derive the displayed percentile from the oriented score.
        if np.isnan(score) or spec.transform == "active_only":
            pct = np.nan
        elif spec.direction == "contrarian_high_is_top":
            pct = 100.0 - score
        else:
            pct = score

        window_years = np.nan
        if has and isinstance(series.index, pd.DatetimeIndex):
            n_b = int(np.busday_count(series.index[0].date(), series.index[-1].date())) + 1
            window_years = min(ROLLING_WINDOW_DAYS, n_b) / 252.0

        prov = meta.get("provenance") or {}
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
            "as_of": as_of,
            "n_obs": n_obs,
            "status": status,
            "window_years": window_years,
            "max_stale_days": spec.max_stale_days,
            "source_used": prov.get("source"),
            "source_kind": prov.get("kind"),
            "source_note": prov.get("note"),
        })
    return pd.DataFrame(rows)


def composite(scores_df: pd.DataFrame) -> dict:
    """Weighted composite + per-pillar scores from a scores table.

    Runs the same aggregator as the history, on a one-row frame, so the gauge and
    the chart endpoint are the same arithmetic. `covered_weight` is the share of
    the model's indicator weight that is live (not just the share of pillars
    with at least one indicator).
    """
    buckets = {b: {"score": np.nan, "weight": w, "n": 0, "coverage": 0.0} for b, w in BUCKET_WEIGHTS.items()}
    out = {"composite": np.nan, "buckets": buckets, "covered_weight": 0.0}
    if scores_df is None or scores_df.empty or "score" not in scores_df.columns:
        return out
    row = scores_df.drop_duplicates("key").set_index("key")["score"].astype(float)
    frame = pd.DataFrame([row.values], columns=row.index)
    pillars, comp, coverage = aggregate_scores(frame)
    specs = _scored_specs()
    for b, info in buckets.items():
        members = {k: sp for k, sp in specs.items() if sp.bucket == b}
        live = [k for k in members if k in row.index and pd.notna(row[k])]
        total = float(sum(sp.weight for sp in members.values()))
        info["n"] = len(live)
        info["coverage"] = float(sum(members[k].weight for k in live) / total) if total else 0.0
        if b in pillars.columns and pd.notna(pillars[b].iloc[0]):
            info["score"] = float(pillars[b].iloc[0])
    out["covered_weight"] = float(coverage.iloc[0]) if len(coverage) else 0.0
    if len(comp) and pd.notna(comp.iloc[0]):
        out["composite"] = float(comp.iloc[0])
    return out


# ---------------------------------------------------------------------------
# Cluster detection — "are we in a confluence top/bottom?"
# ---------------------------------------------------------------------------
def cluster_need(themes_live: int) -> int:
    """Net themes at one extreme that make a cluster: a third of the live themes."""
    return max(CLUSTER_MIN_THEMES, int(np.ceil(themes_live * CLUSTER_NET_SHARE)))


def cluster_side(cluster: dict | None) -> str | None:
    """'top', 'bottom' or None: is one extreme lopsided enough to be a cluster?

    Themes at one extreme minus themes at the other must reach `need`, and the
    winning side must span CLUSTER_MIN_BUCKETS pillars. Opposing extremes
    cancel: nine themes washed out next to nine stretched is a divided market,
    not a signal in either direction.
    """
    if not cluster:
        return None
    top = int(cluster.get("top_theme_count", cluster.get("top_cluster_count", 0)))
    bot = int(cluster.get("bottom_theme_count", cluster.get("bottom_cluster_count", 0)))
    # A dict from an older caller has no theme count: assume the whole model is live.
    live = int(cluster.get("themes_live") or len({theme_for(k) for k in _scored_specs()}))
    need = int(cluster.get("need") or cluster_need(live))
    if top - bot >= need and len(cluster.get("top_buckets", [])) >= CLUSTER_MIN_BUCKETS:
        return "top"
    if bot - top >= need and len(cluster.get("bottom_buckets", [])) >= CLUSTER_MIN_BUCKETS:
        return "bottom"
    return None


def cluster_signal(scores_df: pd.DataFrame) -> dict:
    """Count the independent THEMES at each extreme, and say whether the
    extremes are lopsided enough to call a cluster.

    Themes, not raw indicators, so a single vol event (VIX/VVIX/SKEW/term
    structure all firing) counts once rather than four times. "Extreme" is
    relative: a score beyond EXTREME_TOP / EXTREME_BOTTOM against the
    indicator's own last three years, not an absolute stress level.
    """
    extreme_top = scores_df[scores_df["score"] >= EXTREME_TOP]
    extreme_bot = scores_df[scores_df["score"] <= EXTREME_BOTTOM]
    scored = scores_df.dropna(subset=["score"])

    def _themes(df: pd.DataFrame) -> int:
        if df.empty or "theme" not in df.columns:
            return int(len(df))
        return int(df["theme"].nunique())

    live = _themes(scored)
    out = {
        # Raw indicator counts (kept for display continuity)
        "top_cluster_count": int(len(extreme_top)),
        "bottom_cluster_count": int(len(extreme_bot)),
        # De-duplicated independent-theme counts
        "top_theme_count": _themes(extreme_top),
        "bottom_theme_count": _themes(extreme_bot),
        # Cross-pillar breadth of the extremes
        "top_buckets": sorted(extreme_top["bucket"].unique().tolist()),
        "bottom_buckets": sorted(extreme_bot["bucket"].unique().tolist()),
        "top_names": extreme_top["label"].tolist(),
        "bottom_names": extreme_bot["label"].tolist(),
        "themes_live": live,
        "need": cluster_need(live),
    }
    out["net_themes"] = out["top_theme_count"] - out["bottom_theme_count"]
    out["side"] = cluster_side(out)
    return out


# ---------------------------------------------------------------------------
# Historical pillars and composite
# ---------------------------------------------------------------------------
def historical_pillar_scores(raw: RawFrame, per_indicator: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Daily DataFrame with one column per pillar, values = pillar score 0-100 over
    time (weighted mean of that pillar's indicator scores). Accepts a precomputed
    per_indicator_history frame to avoid recomputing the rolling-percentile pass.
    """
    df = per_indicator_history(raw) if per_indicator is None else per_indicator
    return aggregate_scores(df)[0]


def historical_composite(
    raw: RawFrame,
    pillars: pd.DataFrame | None = None,
    per_indicator: pd.DataFrame | None = None,
) -> pd.Series:
    """Composite regime score over time, on dates with enough of the model live.

    Pass `per_indicator` to reuse the rolling-percentile pass. `pillars` is
    accepted for older callers and ignored: coverage is counted indicator by
    indicator, which pillar scores alone can't provide. Before the model is at
    least MODEL_MIN_COVERAGE populated there is no value: the history used to
    start in 1988 as a "composite" of one survey.
    """
    df = per_indicator_history(raw) if per_indicator is None else per_indicator
    return aggregate_scores(df)[1].dropna()


def composite_context(comp_hist: pd.Series, window: int = ROLLING_WINDOW_DAYS) -> dict:
    """Where today's composite sits in its own recent range.

    An average of 30-odd percentile scores is compressed toward 50: over two
    years of the full model it stayed between about 22 and 68, so the gauge's
    outer bands are almost never reached. The rank against its own history says
    how unusual a reading is.
    """
    if comp_hist is None or comp_hist.empty:
        return {}
    s = comp_hist.dropna().iloc[-window:]
    if len(s) < 60:
        return {}
    cur = float(s.iloc[-1])
    return {
        "pct": float((s <= cur).mean() * 100.0),
        "low": float(s.min()), "high": float(s.max()), "median": float(s.median()),
        "years": len(s) / 252.0,
        "d_1w": float(cur - s.iloc[-6]) if len(s) > 5 else float("nan"),
    }


# ---------------------------------------------------------------------------
# Exposure recommendation — map composite score + cluster to an actionable position
# ---------------------------------------------------------------------------
CLUSTER_MIN_THEMES = 3   # floor on the net themes a cluster needs (thin models)
CLUSTER_MIN_BUCKETS = 2  # the winning side must span this many pillars

# What a cluster does to the card, in points of net exposure.
BOTTOM_CLUSTER_ADD = 15
# A top cluster is a flag, not a cut. Over Nov 2022 - Nov 2026, the only window
# with the full model, it was on about one session in seven and was followed by
# ordinary returns, while a 20-point cut held the card near 42% net through a
# rising market. Set these above zero to restore a cut once it has evidence.
TOP_CLUSTER_CUT = 0
TOP_CLUSTER_HEDGE = 0.0

# Net exposure by composite score: the six regime bands' levels, placed at each
# band's midpoint and joined by straight lines. The mapping used to be a step
# function with a 40-point drop at 65 (90% at 64.9, 50% at 65.1), so a one-point
# wobble in the score could move the recommendation by 40 points of exposure.
EXPOSURE_KNOTS = ((7.5, 130.0), (25.0, 115.0), (40.0, 100.0), (55.0, 90.0), (75.0, 50.0), (92.5, 20.0))
HEDGE_KNOTS = ((65.0, 0.0), (75.0, 0.5), (92.5, 1.0))
# Label follows the exposure it describes: (floor of net %, label, colour).
EXPOSURE_LABELS = (
    (125, "BACK UP THE TRUCK", "#16a085"),
    (105, "SCALE IN", "#27ae60"),
    (95, "FULLY INVESTED", "#3498db"),
    (80, "STANDARD ALLOCATION", "#95a5a6"),
    (45, "TRIM", "#e67e22"),
    (-1, "MAX DEFENSIVE", "#c0392b"),
)


def base_exposure(score: float) -> tuple[int, float]:
    """(net %, hedge %) for a composite score, before any cluster override.
    Net is rounded to 5 points and the hedge to a quarter point."""
    xs, ys = zip(*EXPOSURE_KNOTS)
    hx, hy = zip(*HEDGE_KNOTS)
    net = int(round(float(np.interp(score, xs, ys)) / 5.0) * 5)
    hedge = round(float(np.interp(score, hx, hy)) * 4.0) / 4.0
    return net, hedge


def exposure_label(net: float) -> tuple[str, str]:
    for floor, label, color in EXPOSURE_LABELS:
        if net >= floor:
            return label, color
    return EXPOSURE_LABELS[-1][1], EXPOSURE_LABELS[-1][2]


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
      - base_pct     : int, the net % from the composite alone
      - hedge_pct    : float, suggested notional in OTM SPX puts as % of NAV
      - label        : short action label ("TRIM", "SCALE IN", etc.)
      - color        : hex color for UI (matches gauge regime colors)
      - conviction   : "High" / "Medium" / "Low"
      - rationale    : one-line explanation
      - cluster_override : bool — True if cluster forced a deviation from the base mapping
      - cluster_side : "top" / "bottom" / None — which way the extremes are lopsided
      - cluster_flag : bool — a top cluster is on but, with TOP_CLUSTER_CUT at 0,
                       it is reported and changes nothing

    A cluster needs lopsided extremes (see cluster_side): a third of the model's
    independent themes, net, at one extreme across 2+ pillars. A bottom cluster
    adds exposure. A top cluster is a risk flag only until it has evidence.
    `covered_weight` (fraction of the model with data) caps conviction: a
    partial-model reading can never be "High".
    """
    if composite_score is None or (isinstance(composite_score, float) and np.isnan(composite_score)):
        return {
            "net_pct": None, "base_pct": None, "hedge_pct": None,
            "label": "NO DATA", "color": "#555",
            "conviction": "—", "rationale": "Composite unavailable.",
            "cluster_override": False, "cluster_side": None, "cluster_flag": False,
        }

    s = float(composite_score)
    net, hedge = base_exposure(s)
    base_net = net
    label, color = exposure_label(net)
    cluster_override = False
    conviction = "Low"

    top_themes = int(cluster.get("top_theme_count", cluster.get("top_cluster_count", 0))) if cluster else 0
    bot_themes = int(cluster.get("bottom_theme_count", cluster.get("bottom_cluster_count", 0))) if cluster else 0
    top_buckets = len(cluster.get("top_buckets", [])) if cluster else 0
    bot_buckets = len(cluster.get("bottom_buckets", [])) if cluster else 0
    side = cluster_side(cluster)
    top_fire, bot_fire = side == "top", side == "bottom"

    # A cluster is rare by construction, which is what earns it an override.
    top_cuts = top_fire and (TOP_CLUSTER_CUT > 0 or TOP_CLUSTER_HEDGE > 0)
    cluster_flag = top_fire and not top_cuts
    if top_cuts:
        net = max(0, net - TOP_CLUSTER_CUT)
        hedge = max(hedge, 0.5) + TOP_CLUSTER_HEDGE
        cluster_override = True
        label = f"TOP CLUSTER ({top_themes} vs {bot_themes} themes) — CUT"
        color = "#c0392b"
        conviction = "High"
    elif bot_fire:
        net = min(150, net + BOTTOM_CLUSTER_ADD)
        hedge = 0.0
        cluster_override = True
        label = f"BOTTOM CLUSTER ({bot_themes} vs {top_themes} themes) — ADD"
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
        other = bot_themes if top_fire else top_themes
        rationale_bits.append(
            f"{themes} independent themes at a {'top' if top_fire else 'bottom'} extreme across {buckets} pillars, "
            f"{other} at the other — overrides base reading"
        )
    if cluster_flag:
        rationale_bits.append(
            f"top-cluster flag ({top_themes} themes at a top extreme vs {bot_themes}): no exposure change, untested signal"
        )
    rationale_bits.append(f"composite {s:.0f}")
    if low_coverage:
        rationale_bits.append(f"only {covered_weight*100:.0f}% of model has data")
    rationale = " · ".join(rationale_bits)

    return {
        "net_pct": int(round(net)),
        "base_pct": int(base_net),
        "hedge_pct": float(hedge),
        "label": label,
        "color": color,
        "conviction": conviction,
        "rationale": rationale,
        "cluster_override": cluster_override,
        "cluster_side": side,
        "cluster_flag": bool(cluster_flag),
        "covered_weight": float(covered_weight),
    }


def scores_from_row(row: pd.Series) -> pd.DataFrame:
    """A minimal scores table for one date of per_indicator_history."""
    return pd.DataFrame([
        {"key": k, "label": sp.label, "bucket": sp.bucket, "weight": sp.weight, "theme": theme_for(k),
         "score": float(row[k]) if k in row.index and pd.notna(row[k]) else np.nan}
        for k, sp in _scored_specs().items()
    ])


def recommendation_at(per_ind: pd.DataFrame, sessions_back: int = 0) -> dict | None:
    """The exposure card as it would have read `sessions_back` sessions ago,
    rebuilt from that day's own scores: same composite, same cluster test."""
    if per_ind is None or per_ind.empty or len(per_ind) <= sessions_back:
        return None
    pos = -1 - sessions_back
    sc = scores_from_row(per_ind.iloc[pos])
    comp = composite(sc)
    rec = exposure_recommendation(comp["composite"], cluster_signal(sc), comp["covered_weight"])
    rec["composite"] = comp["composite"]
    rec["as_of"] = per_ind.index[pos]
    return rec


def contributions(scores_df: pd.DataFrame) -> pd.DataFrame:
    """Each live indicator's pull on the composite, in points away from 50.

    `share` is the indicator's actual weight in the composite (pillar weight x
    its weight within the pillar, over what is live). The pulls add up to
    composite - 50, so the list explains the number instead of approximating it.
    """
    cols = ["key", "label", "bucket", "score", "share", "contrib"]
    comp = composite(scores_df)
    if np.isnan(comp["composite"]):
        return pd.DataFrame(columns=cols)
    valid = {b: v for b, v in comp["buckets"].items() if not np.isnan(v["score"])}
    total = sum(v["weight"] for v in valid.values())
    live = scores_df.dropna(subset=["score"])
    live = live[live["key"].isin(_scored_specs())]
    rows = []
    for b, info in valid.items():
        sub = live[live["bucket"] == b]
        wsum = float(sub["weight"].sum())
        for r in sub.itertuples():
            share = info["weight"] / total * float(r.weight) / wsum
            rows.append({"key": r.key, "label": r.label, "bucket": b, "score": float(r.score),
                         "share": share, "contrib": share * (float(r.score) - 50.0)})
    return pd.DataFrame(rows, columns=cols).sort_values("contrib").reset_index(drop=True)


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


# ---------------------------------------------------------------------------
# Decision-trust helpers (pure logic — UI renders the result)
# ---------------------------------------------------------------------------
def gauge_state(composite_score: float | None) -> dict:
    """Render-state for the composite gauge: {'status': 'ok'|'offline', 'value'}.

    A NaN composite must NEVER be drawn as 0.0 — on the 0-100 regime scale,
    0 reads as 'capitulation / aggressive accumulation', the exact opposite of
    'model offline'. The UI shows an explicit offline state instead.
    """
    if composite_score is None or (
        isinstance(composite_score, (int, float)) and np.isnan(composite_score)
    ):
        return {"status": "offline", "value": None}
    return {"status": "ok", "value": float(composite_score)}


def read_confidence(
    scores_df: pd.DataFrame, covered_weight: float, now: datetime | None = None
) -> dict:
    """How much to trust today's composite: model coverage, indicator agreement
    (score dispersion) and data freshness by the wall clock.

    Freshness is judged per indicator against its own publication cadence. A
    single 10-day limit for everything meant the weekly CFTC and NFCI prints
    ruled out "High" on most days for no reason. `staleness` is still the age of
    the stalest contributing feed; `freshest` is the age of the newest, which is
    what shows a whole pipeline that has stopped updating. Unknown freshness is
    treated as NOT fresh (Low).
    """
    scored = scores_df.dropna(subset=["score"])
    n_contrib, total = int(len(scored)), int(len(scores_df))
    dispersion = float(scored["score"].std()) if len(scored) > 1 else np.nan
    now = datetime.now() if now is None else now
    allow = scored["max_stale_days"].tolist() if "max_stale_days" in scored.columns else [10] * len(scored)
    ages, n_overdue = [], 0
    for a, limit in zip(scored["as_of"].tolist(), allow):
        if a is None or (isinstance(a, float) and np.isnan(a)):
            continue
        age = (now - pd.Timestamp(a).to_pydatetime().replace(tzinfo=None)).days
        ages.append(age)
        n_overdue += int(age > limit)
    staleness = max(ages) if ages else None
    freshest = min(ages) if ages else None
    n_missing_asof = n_contrib - len(ages)
    n_stale = int((scores_df["status"] == "stale").sum()) if "status" in scores_df.columns else 0

    n_proxy = 0
    if "source_kind" in scored.columns:
        n_proxy = int(scored["source_kind"].isin(["proxy", "fallback"]).sum())

    aligned = np.isnan(dispersion) or dispersion <= 25
    if staleness is None:
        level, color = "Low", "#c0392b"
    elif covered_weight >= 0.8 and n_overdue == 0 and freshest <= 4 and aligned:
        level, color = "High", "#16a085"
    elif covered_weight >= 0.6 and freshest <= 7:
        level, color = "Medium", "#e67e22"
    else:
        level, color = "Low", "#c0392b"
    return {
        "level": level, "color": color,
        "coverage_pct": covered_weight * 100.0,
        "n_contrib": n_contrib, "total": total,
        "dispersion": dispersion, "staleness": staleness, "freshest": freshest,
        "n_missing_asof": n_missing_asof, "n_proxy": n_proxy,
        "n_overdue": n_overdue, "n_stale": n_stale,
    }
