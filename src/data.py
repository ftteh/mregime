"""
Data fetchers. All functions return pandas.Series indexed by date (daily).
Each fetcher is wrapped in try/except and returns an empty series on failure
so the dashboard keeps working even if one source is down.
"""

from __future__ import annotations
import contextlib
import functools
import io
import json
import logging
import os
import re
import sys
import threading
import time
from datetime import datetime, timedelta
from functools import lru_cache

import numpy as np
import pandas as pd
import requests
import yfinance as yf


@contextlib.contextmanager
def _silence_stderr():
    """Swallow stderr inside this block (yfinance prints '$TICKER: possibly
    delisted' directly, bypassing logging)."""
    try:
        with open(os.devnull, "w") as devnull:
            old = sys.stderr
            sys.stderr = devnull
            try:
                yield
            finally:
                sys.stderr = old
    except Exception:
        yield

from . import analytics as A
from . import session as SES
from .config import FRED_API_KEY, NASDAQ_DATA_LINK_API_KEY

log = logging.getLogger(__name__)

# Silence yfinance's built-in stderr chatter for individual bad/delisted tickers
# (e.g. "$FI: possibly delisted"). We handle failures gracefully per-ticker
# downstream; those prints just spam the dashboard terminal.
for _lname in ("yfinance", "yfinance.utils", "yfinance.data"):
    _lg = logging.getLogger(_lname)
    _lg.setLevel(logging.CRITICAL)
    _lg.propagate = False
try:
    yf.set_tz_cache_location(".yfinance_cache")  # silence tz-cache warnings on Windows
except Exception:
    pass

UA = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0 Safari/537.36"
    )
}


# ---------------------------------------------------------------------------
# Per-build fetch memoization
# ---------------------------------------------------------------------------
# Several building-block fetchers are consumed by more than one indicator in a
# single dashboard build: ^VIX feeds the VIX tile plus three term/divergence
# series; the S&P close panel feeds breadth, new-highs and the A/D proxy; the
# (expensive) gamma-exposure proxy feeds both GEX and the flip-zone history.
# Without memoization each is re-fetched on every call, multiplying network and
# CPU cost on the cold path.
#
# The cache is scoped to a single build: `reset_fetch_memo()` is called at the
# top of build_raw(), so each fresh build (cache miss / "Refresh data") still
# re-fetches live, while concurrent consumers within one build share one result.
_FETCH_MEMO: dict[str, object] = {}
_FETCH_MEMO_LOCKS: dict[str, threading.Lock] = {}
_FETCH_MEMO_GUARD = threading.Lock()
_MEMO_MISSING = object()  # sentinel: distinguishes "absent" from a cached None/NaN


def reset_fetch_memo() -> None:
    """Drop all memoized fetch results. Call once at the start of a build."""
    with _FETCH_MEMO_GUARD:
        _FETCH_MEMO.clear()
        _FETCH_MEMO_LOCKS.clear()
        _PROVENANCE.clear()
        _DISPLAY_OVERLAYS.clear()
        _LIVE_QUOTES.clear()


# ---------------------------------------------------------------------------
# Per-build provenance: which source ACTUALLY served each indicator
# ---------------------------------------------------------------------------
# Most fetchers have multi-level fallback chains (yfinance → CBOE CSV,
# FRED api → FRED CSV, official put/call → YCharts → live ETF-chain proxy).
# The config's static `source` string describes the design, not the build —
# without this registry a proxy or fallback silently masquerades as the
# primary feed in the UI. Keys are the indicator keys from build_raw's task
# dict so the dashboard can join provenance onto scores/meta.
_PROVENANCE: dict[str, dict] = {}

# Series that are shown on charts for context but are NOT real measurements
# (synthetic backfills, legacy mixed-methodology cache rows). They must never
# enter scoring/MIN_OBS/cluster; the UI renders them visibly distinct.
_DISPLAY_OVERLAYS: dict[str, pd.Series] = {}


def record_provenance(key: str, source: str, *, kind: str = "primary", note: str = "") -> None:
    """kind: 'primary' | 'fallback' | 'proxy' | 'cache' | 'computed' | 'unavailable'."""
    with _FETCH_MEMO_GUARD:
        _PROVENANCE[key] = {"source": source, "kind": kind, "note": note}


def get_provenance() -> dict[str, dict]:
    """Snapshot of this build's provenance registry."""
    with _FETCH_MEMO_GUARD:
        return {k: dict(v) for k, v in _PROVENANCE.items()}


def _set_display_overlay(key: str, s: pd.Series) -> None:
    if s is None or s.empty:
        return
    with _FETCH_MEMO_GUARD:
        _DISPLAY_OVERLAYS[key] = s


def get_display_overlays() -> dict[str, pd.Series]:
    """Snapshot of this build's display-only (non-scored) overlay series."""
    with _FETCH_MEMO_GUARD:
        return dict(_DISPLAY_OVERLAYS)


# ---------------------------------------------------------------------------
# Closed sessions only
# ---------------------------------------------------------------------------
# Quotes from a session that is still printing. They are shown on the page as
# live context and never scored: every rule and backtest here is close-based.
_LIVE_QUOTES: dict[str, dict] = {}


def _closed_only(obj, label: str | None = None):
    """Drop rows from a session still in progress; remember the dropped quote."""
    closed, live = SES.split_live(obj)
    if live is not None and len(live) and label and isinstance(live, pd.Series):
        last = live.dropna()
        if len(last):
            prev = closed.dropna()
            with _FETCH_MEMO_GUARD:
                _LIVE_QUOTES[label] = {
                    "date": pd.Timestamp(last.index[-1]),
                    "value": float(last.iloc[-1]),
                    "prev_close": float(prev.iloc[-1]) if len(prev) else float("nan"),
                }
    return closed


def get_live_quotes() -> dict[str, dict]:
    """In-progress quotes dropped from this build, keyed by ticker."""
    with _FETCH_MEMO_GUARD:
        return {k: dict(v) for k, v in _LIVE_QUOTES.items()}


def _memo_by_key(key: str, fn):
    """Per-build memo for fetchers that take arguments (`key` names the call)."""
    cached = _FETCH_MEMO.get(key, _MEMO_MISSING)
    if cached is not _MEMO_MISSING:
        return cached
    with _FETCH_MEMO_GUARD:
        lock = _FETCH_MEMO_LOCKS.setdefault(key, threading.Lock())
    with lock:
        cached = _FETCH_MEMO.get(key, _MEMO_MISSING)
        if cached is _MEMO_MISSING:
            cached = fn()
            _FETCH_MEMO[key] = cached
        return cached


def _memoized_fetch(fn):
    """Memoize a zero-argument fetcher for the lifetime of one build.

    Only no-arg calls are memoized (all decorated fetchers use their defaults);
    any call passing arguments bypasses the cache. A per-key lock prevents two
    threads from racing to compute the same source while still letting distinct
    sources fetch concurrently.
    """
    key = fn.__qualname__

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        if args or kwargs:
            return fn(*args, **kwargs)
        # Single-read fast path: .get() is atomic, so a concurrent
        # reset_fetch_memo() can only make us miss into the locked slow path —
        # never raise KeyError on a key that was cleared between check and read.
        cached = _FETCH_MEMO.get(key, _MEMO_MISSING)
        if cached is not _MEMO_MISSING:
            return cached
        with _FETCH_MEMO_GUARD:
            lock = _FETCH_MEMO_LOCKS.setdefault(key, threading.Lock())
        with lock:
            cached = _FETCH_MEMO.get(key, _MEMO_MISSING)
            if cached is _MEMO_MISSING:
                cached = fn()
                _FETCH_MEMO[key] = cached
            return cached

    return wrapper


# ---------------------------------------------------------------------------
# FRED
# ---------------------------------------------------------------------------
def _fred(series_id: str, start: str = "2015-01-01", prov_key: str | None = None) -> pd.Series:
    """
    Pull a FRED series.
    - If FRED_API_KEY is set, use fredapi (fast, reliable).
    - Otherwise try CSV fallback with a VERY short timeout — if the user's
      network can't reach fred.stlouisfed.org (common on residential ISPs
      behind CloudFront), we fail instantly rather than hanging the dashboard.

    `prov_key`: indicator key to record provenance under (only single-series
    indicators pass it; composite fetchers record at their own level).
    """
    if FRED_API_KEY:
        try:
            from fredapi import Fred
            f = Fred(api_key=FRED_API_KEY)
            s = f.get_series(series_id, observation_start=start)
            s.name = series_id
            if prov_key:
                record_provenance(prov_key, f"fred_api:{series_id}", kind="primary")
            return s.dropna()
        except Exception as e:
            # FRED intermittently 500s on certain series (esp. RRPONTSYD).
            # We transparently fall back to their CSV endpoint, so demote to info.
            log.info("fredapi transient %s (%s) — using CSV fallback", series_id, str(e)[:60])

    # CSV fallback — 4s timeout so a missing key doesn't stall the dashboard
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}&cosd={start}"
    try:
        r = requests.get(url, headers=UA, timeout=4)
        r.raise_for_status()
        if len(r.text) < 20 or "," not in r.text[:500]:
            return pd.Series(dtype=float, name=series_id)
        df = pd.read_csv(io.StringIO(r.text))
        df.columns = [str(c).strip() for c in df.columns]
        date_col = df.columns[0]
        val_col = df.columns[1] if len(df.columns) > 1 else df.columns[0]
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df[val_col] = pd.to_numeric(df[val_col], errors="coerce")
        s = df.dropna(subset=[date_col]).set_index(date_col)[val_col].dropna()
        s.name = series_id
        s.index.name = "date"
        if not s.empty:
            if prov_key:
                record_provenance(
                    prov_key, f"fred_csv:{series_id}", kind="fallback",
                    note="" if FRED_API_KEY else "no FRED_API_KEY",
                )
            return s.loc[start:]
    except Exception as e:
        log.info("FRED CSV %s unavailable (%s) — set FRED_API_KEY", series_id, str(e)[:60])
    if prov_key:
        record_provenance(prov_key, "", kind="unavailable")
    return pd.Series(dtype=float, name=series_id)


@_memoized_fetch
def fred_hy_spread() -> pd.Series:
    return _fred("BAMLH0A0HYM2", prov_key="hy_spread")


def fred_ig_spread() -> pd.Series:
    return _fred("BAMLC0A0CM", prov_key="ig_spread")


def fred_nfci() -> pd.Series:
    """Chicago Fed National Financial Conditions Index (weekly)."""
    return _fred("NFCI", prov_key="financial_conditions")


def fred_ted() -> pd.Series:
    """TED discontinued 2022 — we fall back to SOFR-T10Y proxy if empty."""
    ted = _fred("TEDRATE")
    return ted


def fred_net_liquidity() -> pd.Series:
    """
    Fed Net Liquidity = WALCL - TGA - RRP, expressed in $ BILLIONS.

    FRED units:
      WALCL       : Millions of U.S. Dollars
      WTREGEN     : Millions of Dollars (TGA)
      RRPONTSYD   : Billions of U.S. Dollars
    """
    walcl = _fred("WALCL")
    tga = _fred("WTREGEN")
    rrp = _fred("RRPONTSYD")
    if walcl.empty:
        return pd.Series(dtype=float, name="net_liquidity")
    walcl_b = walcl / 1000.0        # millions -> billions
    tga_b = tga / 1000.0 if not tga.empty else tga   # millions -> billions
    df = pd.concat(
        [walcl_b.rename("walcl"), tga_b.rename("tga"), rrp.rename("rrp")], axis=1
    ).ffill()
    df["net_liq"] = df["walcl"] - df["tga"].fillna(0) - df["rrp"].fillna(0)
    s = df["net_liq"].dropna()
    s.name = "net_liquidity"
    if not s.empty:
        record_provenance(
            "net_liquidity", "fred:WALCL-WTREGEN-RRPONTSYD",
            kind="primary" if FRED_API_KEY else "fallback",
            note="" if FRED_API_KEY else "no FRED_API_KEY (CSV endpoint)",
        )
    else:
        record_provenance("net_liquidity", "", kind="unavailable")
    return s


def fred_dgs10() -> pd.Series:
    """10Y Treasury yield. FRED primary; yfinance ^TNX as fallback (in %, /10)."""
    s = _fred("DGS10")
    if not s.empty:
        return s
    tnx = yf_series("^TNX", period="10y")
    if tnx.empty:
        return pd.Series(dtype=float, name="DGS10")
    s = tnx / 10.0  # ^TNX is 10x yield in percent
    s.name = "DGS10"
    return s


# ---------------------------------------------------------------------------
# yfinance (prices, vol indices)
# ---------------------------------------------------------------------------
# Yahoo rate-limits burst traffic hard on shared-IP hosts (Streamlit Cloud).
# build_raw fans ~15 yf calls out in parallel; gate them so only a couple hit
# Yahoo at once, and retry once after a short backoff — 429s are bursty.
_YF_GATE = threading.Semaphore(2)


def yf_series(ticker: str, period: str = "5y", field: str = "Close") -> pd.Series:
    for attempt in (1, 2):
        try:
            with _YF_GATE:
                with _silence_stderr():
                    df = yf.Ticker(ticker).history(period=period, auto_adjust=False)
            if not df.empty:
                s = df[field].copy()
                s.index = s.index.tz_localize(None) if s.index.tz is not None else s.index
                s.name = ticker
                # Yahoo serves today's bar while the session is open. It is a
                # quote, not a close, so it is kept out of the scored series.
                return _closed_only(s.dropna(), ticker)
        except Exception as e:
            log.info("yfinance unavailable for %s (%s)", ticker, str(e)[:80])
        if attempt == 1:
            time.sleep(1.5)
    return pd.Series(dtype=float, name=ticker)


def _cboe_index_history(symbol: str) -> pd.Series:
    """
    Official CBOE daily index history CSV (full archive since inception, no
    key). Fallback for the vol complex when Yahoo rate-limits shared-IP hosts.
    OHLC files (VIX/VIX9D/VIX3M) use the CLOSE column; single-value files
    (VVIX/SKEW) use their value column.
    """
    url = f"https://cdn.cboe.com/api/global/us_indices/daily_prices/{symbol}_History.csv"
    try:
        r = requests.get(url, headers=UA, timeout=15)
        if not r.ok:
            return pd.Series(dtype=float, name=symbol)
        df = pd.read_csv(io.StringIO(r.text))
        df.columns = [str(c).strip().upper() for c in df.columns]
        if "DATE" not in df.columns:
            return pd.Series(dtype=float, name=symbol)
        col = "CLOSE" if "CLOSE" in df.columns else df.columns[-1]
        s = pd.Series(
            pd.to_numeric(df[col], errors="coerce").values,
            index=pd.to_datetime(df["DATE"], errors="coerce"),
            name=symbol,
        )
        s = s[s.index.notna()].dropna().sort_index()
        return s
    except Exception as e:
        log.info("CBOE history unavailable for %s (%s)", symbol, str(e)[:60])
        return pd.Series(dtype=float, name=symbol)


# Scored series need more than the 3-year percentile window, or the window is
# cut short and the "3-year percentile" is really a 2.5-year one.
SCORED_PERIOD = "10y"


def _yf_or_cboe(prov_key: str, ticker: str, cboe_symbol: str) -> pd.Series:
    """yfinance primary, official CBOE index-history CSV fallback — recorded."""
    s = yf_series(ticker, period=SCORED_PERIOD)
    if not s.empty:
        record_provenance(prov_key, f"yfinance:{ticker}", kind="primary")
        return s
    s = _closed_only(_cboe_index_history(cboe_symbol))
    if not s.empty:
        record_provenance(prov_key, f"cboe_csv:{cboe_symbol}", kind="fallback")
    else:
        record_provenance(prov_key, "", kind="unavailable")
    return s


@_memoized_fetch
def vix() -> pd.Series:
    return _yf_or_cboe("vix", "^VIX", "VIX")


def vvix() -> pd.Series:
    return _yf_or_cboe("vvix", "^VVIX", "VVIX")


def skew() -> pd.Series:
    return _yf_or_cboe("skew", "^SKEW", "SKEW")


@_memoized_fetch
def spx() -> pd.Series:
    s = yf_series("^GSPC", period=SCORED_PERIOD)
    if not s.empty:
        record_provenance("spx", "yfinance:^GSPC", kind="primary")
        return s
    # FRED carries the S&P 500 daily close (trailing ~10y) — enough for the
    # regime-band chart, RSI and percentile scoring when Yahoo is blocked.
    fb = _fred("SP500")
    fb.name = "^GSPC"
    record_provenance(
        "spx", "fred:SP500" if not fb.empty else "",
        kind="fallback" if not fb.empty else "unavailable",
    )
    return fb


def russell2000() -> pd.Series:
    """Russell 2000 index (^RUT)."""
    return yf_series("^RUT")


def nasdaq_composite() -> pd.Series:
    """Nasdaq Composite index (^IXIC)."""
    s = yf_series("^IXIC")
    if not s.empty:
        return s
    fb = _fred("NASDAQCOM")
    fb.name = "^IXIC"
    return fb


def vix9d() -> pd.Series:
    """CBOE 9-day volatility index. Not on yfinance for all ranges; falls back empty."""
    s = yf_series("^VIX9D", period=SCORED_PERIOD)
    return s if not s.empty else _closed_only(_cboe_index_history("VIX9D"))


def vix3m() -> pd.Series:
    """CBOE 3-month volatility index."""
    s = yf_series("^VIX3M", period=SCORED_PERIOD)
    return s if not s.empty else _closed_only(_cboe_index_history("VIX3M"))


def vix_term_9d_1m() -> pd.Series:
    """
    VIX9D / VIX ratio. >1.0 = near-term backwardation = acute panic.
    Historically bottoms within ~3 days of backwardation spikes.
    """
    a, b = vix9d(), vix()
    if a.empty or b.empty:
        return pd.Series(dtype=float, name="vix_term_9d_1m")
    df = pd.concat([a.rename("n"), b.rename("d")], axis=1).dropna()
    if df.empty:
        return pd.Series(dtype=float, name="vix_term_9d_1m")
    s = (df["n"] / df["d"]).replace([np.inf, -np.inf], np.nan).dropna()
    s.name = "vix_term_9d_1m"
    return s


def vix_term_1m_3m() -> pd.Series:
    """VIX / VIX3M ratio. >1.0 = full-term backwardation = serious stress."""
    a, b = vix(), vix3m()
    if a.empty or b.empty:
        return pd.Series(dtype=float, name="vix_term_1m_3m")
    df = pd.concat([a.rename("n"), b.rename("d")], axis=1).dropna()
    if df.empty:
        return pd.Series(dtype=float, name="vix_term_1m_3m")
    s = (df["n"] / df["d"]).replace([np.inf, -np.inf], np.nan).dropna()
    s.name = "vix_term_1m_3m"
    return s


@_memoized_fetch
def move_index() -> pd.Series:
    """
    ICE BofA MOVE index from Yahoo (^MOVE). If it's unavailable, proxy with TLT
    20-trading-day annualized realized vol scaled to the MOVE range.
    """
    move = yf_series("^MOVE", period=SCORED_PERIOD)
    if len(move) >= 252:
        move.name = "move_index"
        record_provenance("move_index", "yfinance:^MOVE", kind="primary")
        return move
    tlt = yf_series("TLT", period="5y")
    if tlt.empty:
        record_provenance("move_index", "", kind="unavailable")
        return pd.Series(dtype=float, name="move_proxy")
    log_ret = np.log(tlt / tlt.shift(1))
    vol_ann = log_ret.rolling(20).std() * np.sqrt(252) * 100
    # Scale roughly into MOVE units (MOVE ~ 80-200 range, TLT rv ~ 8-25%)
    proxy = vol_ann * 8
    proxy.name = "move_proxy"
    record_provenance("move_index", "tlt_realized_vol_proxy", kind="proxy",
                      note="TLT 20d realized vol scaled; not the ICE MOVE index")
    return proxy.dropna()


# ---------------------------------------------------------------------------
# S&P 500 constituents + breadth
# ---------------------------------------------------------------------------
_SP500_TICKERS: list[str] = []  # last successful scrape; the fallback list is never cached here


def sp500_tickers() -> list[str]:
    """
    Return current SP500 tickers. Primary: Wikipedia (requires browser UA —
    their WAF blocks anonymous requests). Fallback: vetted SP100 hard-coded list
    (liquid megacaps only, no delisted/renamed tickers).

    Only a successful scrape is kept for the life of the process. Caching the
    fallback too meant one failed request at startup left breadth running on
    80 megacaps until the next restart.
    """
    if _SP500_TICKERS:
        return list(_SP500_TICKERS)
    return list(_memo_by_key("sp500_tickers", _fetch_sp500_tickers))


def _fetch_sp500_tickers() -> list[str]:
    # Primary: Wikipedia with proper User-Agent (they 403 missing-UA requests)
    try:
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        r = requests.get(url, headers=UA, timeout=10)
        r.raise_for_status()
        tables = pd.read_html(io.StringIO(r.text))
        tickers = (
            tables[0]["Symbol"]
            .astype(str)
            .str.replace(".", "-", regex=False)
            .str.strip()
            .tolist()
        )
        if len(tickers) >= 400:  # sanity
            _SP500_TICKERS[:] = tickers
            return tickers
    except Exception as e:
        log.info("SP500 wiki scrape unavailable (%s) — using megacap fallback", str(e)[:60])

    # Fallback: 80 megacaps that (a) are on yfinance today, (b) capture >70% of
    # SP500 cap so breadth proxies remain meaningful. No delisted / renamed
    # tickers (FI, MMC removed — FISV->FI ticker migration confused yfinance cache).
    return [
        "AAPL","MSFT","NVDA","AMZN","META","GOOGL","GOOG","BRK-B","TSLA","LLY",
        "JPM","V","WMT","XOM","UNH","MA","PG","JNJ","HD","AVGO","ORCL","COST",
        "ABBV","BAC","CVX","KO","PEP","MRK","ADBE","CRM","AMD","NFLX","TMO",
        "PFE","LIN","ABT","CSCO","DIS","WFC","ACN","MCD","TXN","DHR","INTC",
        "VZ","NKE","PM","INTU","NEE","CAT","IBM","COP","UPS","HON","AMGN",
        "QCOM","UNP","GS","RTX","LOW","BA","AMAT","MS","T","SBUX","BLK","SPGI",
        "AXP","DE","PLD","BKNG","GE","NOW","MDT","ELV","LMT","SYK","ISRG","ADP",
        "GILD","CVS","TJX",
    ]


@_memoized_fetch
def _sp500_close_panel() -> pd.DataFrame:
    """
    Daily closes for ALL current S&P 500 constituents (~503) over 5 years,
    downloaded once per build and shared by every breadth consumer.

    Breadth is equal-weighted across eligible securities, so the full
    constituent list matters: the old panel used the first 150 rows of the
    Wikipedia table — which is ordered by company NAME, not market cap — over
    2 years, so "% above 200DMA" really had ~1y of history and a sample biased
    to names starting A–E. The 5y window gives the 200DMA / 52-week-high
    measures a genuine 3-year percentile history.

    One bulk `yf.download` (~15s for 503 x 5y) feeds % above 200DMA, net new
    highs, the A/D proxy, 10/20DMA trend breadth and the 50D OB/OS spread.
    Columns are tickers (sp500_tickers order); index is tz-naive daily.
    Returns an empty frame if the download fails.
    """
    tickers = sp500_tickers()
    try:
        with _silence_stderr():
            df = yf.download(
                tickers,
                period="5y",
                auto_adjust=False,
                progress=False,
                group_by="ticker",
                threads=True,
            )
    except Exception as e:
        log.error("yf.download S&P panel failed: %s", e)
        return pd.DataFrame()
    if df is None or df.empty:
        return pd.DataFrame()

    closes = {}
    lvl0 = set(df.columns.get_level_values(0)) if isinstance(df.columns, pd.MultiIndex) else set()
    for t in tickers:
        if t not in lvl0:
            continue
        try:
            closes[t] = df[t]["Close"]
        except Exception:
            continue
    if not closes:
        return pd.DataFrame()

    panel = pd.DataFrame(closes)
    if getattr(panel.index, "tz", None) is not None:
        panel.index = panel.index.tz_localize(None)
    panel = panel.dropna(how="all", axis=1).dropna(how="all", axis=0)
    # Breadth from a half-printed session is not a closing reading.
    return _closed_only(panel)


def _panel_columns(sample_size: int | None = None) -> tuple[pd.DataFrame, list[str]]:
    """Return the shared close panel and the tickers to use.

    `sample_size=None` (default) = every constituent that downloaded. An int
    keeps the first N in sp500_tickers() order (legacy/testing only).
    """
    panel = _sp500_close_panel()
    if panel.empty:
        return panel, []
    universe = sp500_tickers() if sample_size is None else sp500_tickers()[:sample_size]
    wanted = [t for t in universe if t in panel.columns]
    return panel, wanted


def breadth_pct_above_200dma(sample_size: int | None = None) -> pd.Series:
    """
    % of S&P 500 constituents above their 200-day MA over time (daily),
    equal-weighted across every constituent with a defined 200DMA.
    """
    panel, wanted = _panel_columns(sample_size)
    if not wanted:
        record_provenance("pct_above_200dma", "", kind="unavailable")
        return pd.Series(dtype=float, name="pct_above_200dma")
    record_provenance(
        "pct_above_200dma", f"sp500_panel({len(wanted)} constituents)", kind="computed",
        note="current membership (survivorship bias)",
    )

    close_df = panel[wanted].ffill()
    ma200 = close_df.rolling(200).mean()
    above = (close_df > ma200).sum(axis=1)
    valid = (close_df.notna() & ma200.notna()).sum(axis=1)
    pct = (above / valid.replace(0, np.nan)) * 100
    pct.name = "pct_above_200dma"
    pct.index = pct.index.tz_localize(None) if pct.index.tz is not None else pct.index
    return pct.dropna()


def new_highs_minus_lows(sample_size: int | None = None) -> pd.Series:
    """52-week new highs minus new lows across S&P 500 constituents."""
    panel, wanted = _panel_columns(sample_size)
    if not wanted:
        record_provenance("new_highs_lows", "", kind="unavailable")
        return pd.Series(dtype=float, name="new_highs_minus_lows")
    record_provenance(
        "new_highs_lows", f"sp500_panel({len(wanted)} constituents)", kind="computed",
        note="current membership (survivorship bias)",
    )
    cdf = panel[wanted].ffill()
    hi = cdf.rolling(252).max()
    lo = cdf.rolling(252).min()
    new_hi = (cdf >= hi).sum(axis=1)
    new_lo = (cdf <= lo).sum(axis=1)
    s = (new_hi - new_lo).astype(float)
    s.name = "new_highs_minus_lows"
    s.index = s.index.tz_localize(None) if s.index.tz is not None else s.index
    return s.dropna()


# ---------------------------------------------------------------------------
# Sentiment scrapers (graceful fallbacks)
# ---------------------------------------------------------------------------
def _aaii_live() -> tuple[pd.DataFrame, str]:
    """
    AAII weekly sentiment. Returns DataFrame with bullish, bearish, neutral, spread.
    Values are in PERCENT (0-100). Spread = bullish - bearish (also in percent points).

    Sources (in order of preference):
      1. Nasdaq Data Link (requires free API key in NASDAQ_DATA_LINK_API_KEY)
      2. AAII public XLS — usually 403'd in 2026 (Incapsula)
      3. AAII HTML page scrape — usually 403'd
    """
    cols = ["bullish", "neutral", "bearish", "spread"]

    # --- 1. Nasdaq Data Link (most reliable — free key from data.nasdaq.com)
    if NASDAQ_DATA_LINK_API_KEY:
        try:
            url = (
                f"https://data.nasdaq.com/api/v3/datasets/AAII/AAII_SENTIMENT/data.csv"
                f"?api_key={NASDAQ_DATA_LINK_API_KEY}"
            )
            r = requests.get(url, headers=UA, timeout=20)
            r.raise_for_status()
            if "," in r.text[:200] and "<html" not in r.text[:200].lower():
                df = pd.read_csv(io.StringIO(r.text))
                df.columns = [str(c).strip().lower() for c in df.columns]
                date_col = next((c for c in df.columns if "date" in c), df.columns[0])
                df["date"] = pd.to_datetime(df[date_col], errors="coerce")
                df = df.dropna(subset=["date"]).set_index("date").sort_index()
                for want in ["bullish", "neutral", "bearish"]:
                    match = next((c for c in df.columns if want in c), None)
                    if match:
                        df[want] = pd.to_numeric(df[match], errors="coerce")
                if {"bullish", "bearish"}.issubset(df.columns):
                    scale = 100.0 if df["bullish"].dropna().max() <= 1.5 else 1.0
                    df["bullish"] = df["bullish"] * scale
                    df["bearish"] = df["bearish"] * scale
                    if "neutral" in df.columns:
                        df["neutral"] = df["neutral"] * scale
                    df["spread"] = df["bullish"] - df["bearish"]
                    out = df[[c for c in cols if c in df.columns]].dropna(subset=["spread"])
                    if not out.empty:
                        return out, "nasdaq_data_link:AAII_SENTIMENT"
        except Exception as e:
            log.info("Nasdaq Data Link AAII unavailable (%s)", str(e)[:80])

    # --- Primary: historical XLS (parse with xlrd directly to avoid pandas version checks)
    try:
        aaii_headers = {
            **UA,
            "Accept": "application/vnd.ms-excel,application/octet-stream,*/*",
            "Referer": "https://www.aaii.com/sentimentsurvey",
        }
        sess = requests.Session()
        sess.get("https://www.aaii.com/sentimentsurvey", headers=aaii_headers, timeout=15)
        r = sess.get(
            "https://www.aaii.com/files/surveys/sentiment.xls",
            headers=aaii_headers, timeout=45,
        )
        r.raise_for_status()
        # Sanity: must be XLS binary, not HTML error page
        if r.content[:4] not in (b"\xd0\xcf\x11\xe0", b"PK\x03\x04"):
            raise RuntimeError(f"AAII returned non-XLS (first bytes {r.content[:8]!r})")
        import xlrd as _xlrd
        book = _xlrd.open_workbook(file_contents=r.content)
        sheet = book.sheet_by_index(0)

        # Find header row — look for "Bullish"
        header_row = None
        for i in range(min(10, sheet.nrows)):
            row = [str(c.value).strip().lower() for c in sheet.row(i)]
            if any("bullish" in c for c in row) and any("bearish" in c for c in row):
                header_row = i
                break
        if header_row is None:
            raise RuntimeError("AAII sheet: can't find header row")

        header = [str(c.value).strip().lower() for c in sheet.row(header_row)]
        def idx_of(substrs):
            for j, h in enumerate(header):
                if all(s in h for s in substrs):
                    return j
            for j, h in enumerate(header):
                if substrs[0] in h:
                    return j
            return None

        i_date = idx_of(["date"]) or 0
        i_bull = idx_of(["bullish"])
        i_neut = idx_of(["neutral"])
        i_bear = idx_of(["bearish"])
        if i_bull is None or i_bear is None:
            raise RuntimeError("AAII sheet: missing columns")

        rows = []
        for i in range(header_row + 1, sheet.nrows):
            row = sheet.row(i)
            dval = row[i_date].value
            if not dval:
                continue
            try:
                if isinstance(dval, float):
                    dt = pd.Timestamp(_xlrd.xldate_as_datetime(dval, book.datemode))
                else:
                    dt = pd.to_datetime(str(dval), errors="coerce")
            except Exception:
                continue
            if pd.isna(dt):
                continue
            def num(j):
                if j is None: return None
                v = row[j].value
                try: return float(v)
                except (TypeError, ValueError): return None
            rows.append((dt, num(i_bull), num(i_neut), num(i_bear)))

        if rows:
            df = pd.DataFrame(rows, columns=["date", "bullish", "neutral", "bearish"]).set_index("date").sort_index()
            df = df.dropna(subset=["bullish", "bearish"])
            # Rescale if fractional
            if df["bullish"].iloc[-1] <= 1.0:
                df[["bullish", "neutral", "bearish"]] = df[["bullish", "neutral", "bearish"]] * 100
            df["spread"] = df["bullish"] - df["bearish"]
            return df[cols], "aaii.com:sentiment.xls"
    except Exception as e:
        log.info("AAII XLS unavailable (%s)", str(e)[:80])

    # --- Fallback: scrape AAII's public results page. The page renders a
    # 4-column table (Reported Date | Bullish | Neutral | Bearish) with the
    # latest ~21 weekly readings. Enough history for short-term percentile
    # scoring; full-history XLS is blocked by Incapsula so this is the best
    # free/no-key path in 2026.
    try:
        import re
        from bs4 import BeautifulSoup
        aaii_ua = {
            **UA,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Referer": "https://www.aaii.com/",
        }
        r = requests.get(
            "https://www.aaii.com/sentimentsurvey/sent_results",
            headers=aaii_ua, timeout=30,
        )
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "lxml")

        pct_re = re.compile(r"([0-9]{1,2}\.[0-9])\s*%")
        today = pd.Timestamp.today().normalize()

        def _parse_row_date(date_txt: str, anchor_year: int) -> pd.Timestamp | None:
            """Parse 'Apr 15' style dates; infer year by stepping back from anchor."""
            try:
                return pd.to_datetime(f"{date_txt} {anchor_year}", errors="raise")
            except Exception:
                return None

        rows: list[tuple[pd.Timestamp, float, float, float]] = []
        for table in soup.find_all("table"):
            header_cells = [c.get_text(" ", strip=True).lower() for c in table.find_all(["th", "td"])[:4]]
            if not ("bullish" in " ".join(header_cells) and "bearish" in " ".join(header_cells)):
                continue
            year_cursor = today.year
            prev_month: int | None = None
            for tr in table.find_all("tr"):
                cells = [c.get_text(" ", strip=True) for c in tr.find_all(["td", "th"])]
                if len(cells) < 4:
                    continue
                date_raw = cells[0].strip()
                if not date_raw or "date" in date_raw.lower():
                    continue
                bull_m = pct_re.search(cells[1])
                neut_m = pct_re.search(cells[2])
                bear_m = pct_re.search(cells[3])
                if not (bull_m and bear_m):
                    continue
                dt = _parse_row_date(date_raw, year_cursor)
                if dt is None:
                    continue
                # If the month advanced (e.g., going from Jan -> Dec), we crossed
                # a year boundary backwards; decrement the year cursor.
                if prev_month is not None and dt.month > prev_month:
                    year_cursor -= 1
                    dt = _parse_row_date(date_raw, year_cursor) or dt
                prev_month = dt.month
                rows.append((
                    dt,
                    float(bull_m.group(1)),
                    float(neut_m.group(1)) if neut_m else float("nan"),
                    float(bear_m.group(1)),
                ))
            if rows:
                break

        if rows:
            df = (
                pd.DataFrame(rows, columns=["date", "bullish", "neutral", "bearish"])
                .drop_duplicates(subset=["date"])
                .set_index("date")
                .sort_index()
            )
            df["spread"] = df["bullish"] - df["bearish"]
            return df[cols], "aaii.com:results page (recent weeks only)"
    except Exception as e:
        log.info("AAII HTML fallback unavailable (%s)", str(e)[:80])

    return pd.DataFrame(columns=cols), ""


AAII_SEED = "aaii_sentiment_seed.csv"


def aaii_sentiment() -> pd.DataFrame:
    """AAII weekly sentiment (bullish, neutral, bearish, spread, in percent):
    live rows over the local cache over a committed seed.

    The full-history download succeeds on some builds and is blocked on others.
    When it was blocked, the only live source is a page with about 21 recent
    weeks, which is too thin to score, so the indicator dropped out of the
    composite and came back from one build to the next. Live rows are now
    merged over saved history, so a blocked download costs nothing.

    The seed holds the bull-bear spread only (the one column the model scores)
    and is what a fresh cloud deploy starts from, since `cache/` is local. The
    results page then extends it, as long as the seed is under about 20 weeks
    old: refresh it with `python -m scripts.refresh_seeds`.
    """
    cols = ["bullish", "neutral", "bearish", "spread"]
    live, source = _aaii_live()
    if not live.empty:
        live = live[[c for c in cols if c in live.columns]].copy()
        live.index = pd.DatetimeIndex(live.index).normalize()
        live.index.name = "date"
    cache = _load_frame_cache("aaii_sentiment_history.csv")
    seed = _load_seed_frame(AAII_SEED)
    merged = _merge_frames(seed, cache, live)
    if merged.empty:
        record_provenance("aaii_bull_bear", "", kind="unavailable")
        return pd.DataFrame(columns=cols)
    if not live.empty:
        _save_frame_cache("aaii_sentiment_history.csv", merged)
        partial = len(live) < len(merged)
        record_provenance("aaii_bull_bear", source, kind="primary",
                          note="recent weeks live, older history from the local cache or seed" if partial else "")
    elif not cache.empty:
        record_provenance("aaii_bull_bear", "aaii (saved history)", kind="cache",
                          note=f"live fetch failed; cached history through {cache.index.max():%Y-%m-%d}")
    else:
        record_provenance("aaii_bull_bear", "bundled_seed", kind="fallback",
                          note=f"live source unreachable and no local cache; seed through {seed.index.max():%Y-%m-%d}")
    return merged[[c for c in cols if c in merged.columns]].dropna(subset=["spread"])


def fear_greed_index() -> pd.Series:
    """
    CNN Fear & Greed Index. Uses the public JSON endpoint backing the CNN widget.
    Returns a daily series (0-100).
    """
    base = "https://production.dataviz.cnn.io/index/fearandgreed/graphdata"
    # The bare endpoint serves one year, which would make the "3-year percentile"
    # a 6-month one. A start date in the path returns up to about five years.
    start = (pd.Timestamp.today() - pd.DateOffset(years=5) + pd.Timedelta(days=14)).strftime("%Y-%m-%d")
    for url, note in ((f"{base}/{start}", ""), (base, "1 year of history only")):
        try:
            r = requests.get(url, headers=UA, timeout=20)
            r.raise_for_status()
            s = _parse_fear_greed(r.json())
        except Exception as e:
            log.info("Fear&Greed fetch failed (%s): %s", url[-10:], str(e)[:80])
            continue
        if not s.empty:
            record_provenance("fear_greed", "cnn:fearandgreed", kind="primary", note=note)
            return s
    record_provenance("fear_greed", "", kind="unavailable")
    return pd.Series(dtype=float, name="fear_greed")


def _parse_fear_greed(payload: dict) -> pd.Series:
    """Daily closes from CNN's graph payload.

    The feed stamps today's reading with a time of day and can also carry a
    midnight row for the same date; keep one row per date (the latest), and
    leave out a session that is still printing.
    """
    hist = (payload or {}).get("fear_and_greed_historical", {}).get("data", [])
    if not hist:
        return pd.Series(dtype=float, name="fear_greed")
    df = pd.DataFrame(hist)
    stamp = pd.to_datetime(df["x"], unit="ms")
    s = pd.Series(pd.to_numeric(df["y"], errors="coerce").values, index=stamp).dropna().sort_index()
    s.index = s.index.normalize()
    s = s[~s.index.duplicated(keep="last")]
    s.name = "fear_greed"
    return _closed_only(s)


def _series_cache_path(filename: str) -> str:
    cache_dir = os.path.join(os.path.dirname(__file__), "..", "cache")
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, filename)


def _load_series_cache(filename: str, col_name: str) -> pd.Series:
    path = _series_cache_path(filename)
    if not os.path.exists(path):
        return pd.Series(dtype=float, name=col_name)
    try:
        df = pd.read_csv(path, parse_dates=["date"])
        s = pd.to_numeric(df.set_index("date")[col_name], errors="coerce").dropna()
        s.name = col_name
        return s.sort_index()
    except Exception:
        return pd.Series(dtype=float, name=col_name)


def _save_series_cache(filename: str, series: pd.Series, col_name: str) -> None:
    path = _series_cache_path(filename)
    try:
        out = pd.to_numeric(series, errors="coerce").dropna()
        out = out[~out.index.duplicated(keep="last")].sort_index()
        out.to_frame(col_name).reset_index().rename(columns={"index": "date"}).to_csv(path, index=False)
    except Exception as e:
        log.warning("Cache write failed for %s: %s", filename, e)


# ---------------------------------------------------------------------------
# Provenance-aware series cache (schema: date,<col>,source)
# ---------------------------------------------------------------------------
# Used where rows of DIFFERENT methodologies/units could otherwise end up in
# one persisted series (the gamma caches mixed SqueezeMetrics GEX, CBOE
# snapshot GEX, an ad-hoc yfinance heuristic and a scaled-GEX synthetic
# backfill with no marker). The `source` column makes every row attributable
# so scoring can use measured rows only.
GAMMA_SCORED_SOURCES = {"squeezemetrics", "cboe_snapshot"}  # flip-zone cache; GEX picks ONE of these per build


def _empty_sourced_frame(col_name: str) -> pd.DataFrame:
    return pd.DataFrame(
        {col_name: pd.Series(dtype=float), "source": pd.Series(dtype=str)},
        index=pd.DatetimeIndex([], name="date"),
    )


def _load_series_cache_with_source(filename: str, col_name: str) -> pd.DataFrame:
    """Load a cached series WITH per-row provenance.

    MIGRATION: old-schema files (date,<col>) are rewritten once with every
    pre-existing row marked source='legacy_mixed' (display-only). Rows written
    before provenance tracking interleave different methodologies/units and
    cannot be attributed post-hoc — quarantining them wholesale is the only
    rule that never promotes fabricated data back into scoring. Idempotent.
    """
    path = _series_cache_path(filename)
    if not os.path.exists(path):
        return _empty_sourced_frame(col_name)
    try:
        df = pd.read_csv(path, parse_dates=["date"])
        if "date" not in df.columns or col_name not in df.columns:
            return _empty_sourced_frame(col_name)
        df[col_name] = pd.to_numeric(df[col_name], errors="coerce")
        df = df.dropna(subset=["date", col_name]).set_index("date").sort_index()
        migrated = "source" not in df.columns
        if migrated:
            df["source"] = "legacy_mixed"
        else:
            df["source"] = df["source"].fillna("legacy_mixed").astype(str)
        out = df[[col_name, "source"]]
        out = out[~out.index.duplicated(keep="last")]
        out.index.name = "date"
        if migrated:
            _save_series_cache_with_source(filename, out, col_name)
        return out
    except Exception as e:
        log.warning("Sourced cache read failed for %s: %s", filename, e)
        return _empty_sourced_frame(col_name)


def _save_series_cache_with_source(filename: str, df: pd.DataFrame, col_name: str) -> None:
    path = _series_cache_path(filename)
    try:
        out = df[[col_name, "source"]].copy()
        out[col_name] = pd.to_numeric(out[col_name], errors="coerce")
        out = out.dropna(subset=[col_name])
        out = out[~out.index.duplicated(keep="last")].sort_index()
        out.index.name = "date"
        out.reset_index().to_csv(path, index=False)
    except Exception as e:
        log.warning("Cache write failed for %s: %s", filename, e)


def _upsert_sourced_rows(
    df: pd.DataFrame, s: pd.Series, col_name: str, source: str
) -> pd.DataFrame:
    """Insert/overwrite rows from `s` labeled with `source` (new rows win on
    duplicate dates — measured data overrides legacy rows for the same day)."""
    if s is None or s.empty:
        return df
    vals = pd.to_numeric(pd.Series(s.values, index=pd.DatetimeIndex(s.index)), errors="coerce")
    if getattr(vals.index, "tz", None) is not None:
        vals.index = vals.index.tz_localize(None)
    vals = vals.dropna()
    if vals.empty:
        return df
    add = pd.DataFrame({col_name: vals.values, "source": source}, index=vals.index)
    out = pd.concat([df, add])
    out = out[~out.index.duplicated(keep="last")].sort_index()
    out.index.name = "date"
    return out


def _merge_series(name: str, *series: pd.Series) -> pd.Series:
    parts = []
    for s in series:
        if s is None or s.empty:
            continue
        out = pd.to_numeric(s.copy(), errors="coerce").dropna()
        if out.empty:
            continue
        if not isinstance(out.index, pd.DatetimeIndex):
            out.index = pd.to_datetime(out.index, errors="coerce")
            out = out[out.index.notna()]
        if out.empty:
            continue
        if getattr(out.index, "tz", None) is not None:
            out.index = out.index.tz_localize(None)
        parts.append(out.sort_index())
    if not parts:
        return pd.Series(dtype=float, name=name)
    merged = pd.concat(parts)
    merged = pd.to_numeric(merged, errors="coerce").dropna()
    merged = merged[~merged.index.duplicated(keep="last")].sort_index()
    merged.name = name
    return merged


# ---------------------------------------------------------------------------
# US trading-day calendar — honest stamping for live snapshot values
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def _us_busday_calendar() -> np.busdaycalendar:
    """Weekend + US-federal-holiday calendar. Approximation of the NYSE
    calendar (lacks Good Friday, includes Columbus/Veterans Day) — only the
    fallback path uses it; live runs prefer the real SPX market calendar."""
    from pandas.tseries.holiday import USFederalHolidayCalendar

    hols = USFederalHolidayCalendar().holidays(start="2000-01-01", end="2040-12-31")
    return np.busdaycalendar(holidays=hols.values.astype("datetime64[D]"))


def _trading_day_mask(idx: pd.DatetimeIndex) -> np.ndarray:
    """Boolean mask: which index dates fall on a US trading day."""
    if len(idx) == 0:
        return np.array([], dtype=bool)
    days = idx.normalize().values.astype("datetime64[D]")
    return np.is_busday(days, busdaycal=_us_busday_calendar())


def last_completed_trading_day(now: pd.Timestamp | None = None) -> pd.Timestamp:
    """Last completed US equity trading day — the only honest stamp for a live
    snapshot value. `pd.Timestamp.today()` stamped weekend/holiday phantom
    dates into the caches and made `as_of` read "today" for stale data.

    Authority order:
    1. The fetched SPX series' last index date (real market calendar; memoized
       per build, so this adds no network cost) when within 7 days of `now`.
    2. Fallback: previous day per the US-federal-holiday business calendar —
       conservative: prefers a 1-day-stale label over a phantom today-stamp,
       since without market data we cannot know whether today's session closed.
    """
    if now is None:
        now = pd.Timestamp.now()
    now = pd.Timestamp(now)
    try:
        s = spx()
        if s is not None and not s.empty:
            last = pd.Timestamp(s.index[-1])
            if getattr(last, "tz", None) is not None:
                last = last.tz_localize(None)
            last = last.normalize()
            age = now.normalize() - last
            if pd.Timedelta(0) <= age <= pd.Timedelta(days=7):
                return last
    except Exception:
        pass
    d = (now.normalize() - pd.Timedelta(days=1)).to_datetime64().astype("datetime64[D]")
    prev = np.busday_offset(d, 0, roll="backward", busdaycal=_us_busday_calendar())
    return pd.Timestamp(prev)


def _read_cboe_put_call_csv(url: str, name: str) -> pd.Series:
    try:
        r = requests.get(url, headers=UA, timeout=12)
        if not r.ok or "," not in r.text[:1000]:
            return pd.Series(dtype=float, name=name)
        lines = r.text.splitlines()
        header = next(
            (
                i for i, line in enumerate(lines)
                if ("DATE" in line.upper() or "TRADE_DATE" in line.upper())
                and ("P/C" in line.upper().replace(" ", "") or "RATIO" in line.upper())
            ),
            0,
        )
        df = pd.read_csv(io.StringIO(r.text), skiprows=header)
        df.columns = [str(c).strip() for c in df.columns]
        date_col = next((c for c in df.columns if "date" in c.lower()), None)
        ratio_col = next(
            (
                c for c in df.columns
                if "p/c" in c.lower().replace(" ", "") or "ratio" in c.lower()
            ),
            None,
        )
        if date_col is None or ratio_col is None:
            return pd.Series(dtype=float, name=name)
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        s = pd.to_numeric(
            df.dropna(subset=[date_col]).set_index(date_col)[ratio_col],
            errors="coerce",
        ).dropna()
        s.name = name
        return s.sort_index()
    except Exception as e:
        log.info("CBOE put/call CSV failed for %s (%s)", url, str(e)[:80])
        return pd.Series(dtype=float, name=name)


@lru_cache(maxsize=8)
def _cboe_put_call_history(kind: str, name: str) -> pd.Series:
    """CBOE's no-key CSVs cover the official series through Oct 2019."""
    urls = [
        f"https://cdn.cboe.com/resources/options/volume_and_call_put_ratios/{kind}pcarchive.csv",
        f"https://cdn.cboe.com/resources/options/volume_and_call_put_ratios/{kind}pc.csv",
    ]
    return _merge_series(name, *(_read_cboe_put_call_csv(url, name) for url in urls))


@lru_cache(maxsize=8)
def _ycharts_recent_indicator(slug: str, name: str) -> pd.Series:
    """
    YCharts exposes the most recent public rows for CBOE daily statistics without
    auth. It is not a full archive, but it fills the current chart window with
    official CBOE values instead of a single local snapshot.
    """
    url = f"https://ycharts.com/indicators/{slug}"
    try:
        r = requests.get(url, headers=UA, timeout=12)
        if not r.ok:
            return pd.Series(dtype=float, name=name)
        tables = pd.read_html(io.StringIO(r.text))
        parts = []
        for df in tables:
            cols = [str(c).strip() for c in df.columns]
            if "Date" not in cols or "Value" not in cols:
                continue
            date_col = cols[cols.index("Date")]
            value_col = cols[cols.index("Value")]
            tmp = df[[date_col, value_col]].copy()
            tmp[date_col] = pd.to_datetime(tmp[date_col], errors="coerce")
            tmp[value_col] = pd.to_numeric(tmp[value_col], errors="coerce")
            s = tmp.dropna(subset=[date_col, value_col]).set_index(date_col)[value_col]
            if not s.empty:
                parts.append(s)
        return _merge_series(name, *parts)
    except Exception as e:
        log.info("YCharts recent %s unavailable (%s)", slug, str(e)[:80])
        return pd.Series(dtype=float, name=name)


def _cboe_delayed_options(symbol: str) -> tuple[pd.DataFrame, float]:
    """Delayed CBOE option chain JSON with greeks, OI and volume. No API key.

    Memoized per build, not per process: a process-lifetime cache meant a server
    that stayed up kept re-stamping the chain it fetched at startup as "today".
    """
    return _memo_by_key(f"cboe_delayed_options:{symbol.upper()}", lambda: _fetch_cboe_delayed_options(symbol))


def _fetch_cboe_delayed_options(symbol: str) -> tuple[pd.DataFrame, float]:
    url = f"https://cdn.cboe.com/api/global/delayed_quotes/options/{symbol.upper()}.json"
    try:
        r = requests.get(url, headers=UA, timeout=15)
        if not r.ok:
            return pd.DataFrame(), np.nan
        payload = r.json()
        data = payload.get("data", {})
        options = data.get("options", [])
        if not options:
            return pd.DataFrame(), np.nan
        df = pd.DataFrame(options)
        parsed = df["option"].astype(str).str.extract(
            rf"^{re.escape(symbol.upper())}(?P<expiry>\d{{6}})(?P<type>[CP])(?P<strike>\d{{8}})$"
        )
        df = pd.concat([df, parsed], axis=1).dropna(subset=["expiry", "type", "strike"])
        if df.empty:
            return pd.DataFrame(), np.nan
        df["expiry"] = pd.to_datetime("20" + df["expiry"], format="%Y%m%d", errors="coerce")
        df["strike"] = pd.to_numeric(df["strike"], errors="coerce") / 1000.0
        for col in ("volume", "open_interest", "gamma", "last_trade_price", "bid", "ask"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        spot = pd.to_numeric(pd.Series([data.get("current_price")]), errors="coerce").iloc[0]
        return df.dropna(subset=["expiry", "strike"]), float(spot)
    except Exception as e:
        log.info("CBOE delayed options %s unavailable (%s)", symbol, str(e)[:80])
        return pd.DataFrame(), np.nan


def _near_expiry_options(df: pd.DataFrame, max_expiries: int = 3) -> pd.DataFrame:
    if df.empty or "expiry" not in df.columns:
        return df
    expiries = pd.Series(df["expiry"].dropna().unique()).sort_values().tolist()
    if not expiries:
        return df.iloc[0:0]
    return df[df["expiry"].isin(expiries[:max_expiries])].copy()


def _cboe_daily_ratio(kind: str, name: str) -> pd.Series:
    """Official put/call from Cboe's daily statistics files (Oct 2019 onward).

    Cboe's CSV archive stops in Oct 2019 and YCharts is unreliable from here, so
    without this leg the "official" series has years of gaps and its latest row
    is an SPY/QQQ/IWM chain proxy that can disagree badly with the real ratio.
    """
    from src import research_data as RD  # local import: research_data imports this module

    try:
        df = RD.cboe_volumes()
    except Exception as e:
        log.info("Cboe daily volumes unavailable: %s", str(e)[:100])
        return pd.Series(dtype=float, name=name)
    put, call = f"{kind}_put", f"{kind}_call"
    if df is None or df.empty or put not in df or call not in df:
        return pd.Series(dtype=float, name=name)
    s = (df[put] / df[call].replace(0, np.nan)).dropna()
    s.name = name
    return s


def _official_put_call_series(*, kind: str, ycharts_slug: str, name: str) -> pd.Series:
    """Official Cboe put/call ratio, and nothing else.

    A day with no official row yet used to be filled with an SPY/QQQ/IWM chain
    ratio. That is a different population: ETF chains run near 1.2 while the
    equity-only ratio runs near 0.6, so the fill ranked as a 99th-percentile
    panic against the official history and put a false extreme into the
    composite and the cluster. The same number also went into the index ratio.
    The series now ends at Cboe's last published day and says so in its date.
    The old put/call cache files held those fills and are no longer read.
    """
    hist = _cboe_put_call_history(kind, name)
    recent = _ycharts_recent_indicator(ycharts_slug, name)
    daily = _cboe_daily_ratio(kind, name)
    # Later legs win: Cboe's own daily files override a YCharts copy of the same day.
    merged = _closed_only(_merge_series(name, hist, recent, daily))
    legs = [
        lbl for lbl, leg in (("cboe_archive", hist), ("ycharts", recent), ("cboe_daily", daily))
        if not leg.empty
    ]
    if merged.empty:
        record_provenance(name, "", kind="unavailable")
    else:
        record_provenance(name, "+".join(legs), kind="primary", note="official Cboe ratio")
    return merged


def put_call_ratio() -> pd.Series:
    """
    CBOE Equity Put/Call ratio (official values only).

    1. CBOE official CSV archive through Oct 2019
    2. CBOE daily market statistics files from Oct 2019
    3. YCharts public copy of recent CBOE rows, when reachable
    """
    return _official_put_call_series(
        kind="equity", ycharts_slug="cboe_equity_put_call_ratio", name="put_call",
    )


# ---------------------------------------------------------------------------
# Valuation: Equity Risk Premium
# ---------------------------------------------------------------------------
@_memoized_fetch
def sp500_pe_ratio() -> pd.Series:
    """
    S&P 500 trailing P/E (monthly) from multpl.com — no API key. Used to derive
    a *real* earnings yield for the ERP. Returns empty on failure so the caller
    can be honestly absent rather than fabricate a constant.
    """
    url = "https://www.multpl.com/s-p-500-pe-ratio/table/by-month"
    try:
        r = requests.get(url, headers=UA, timeout=12)
        r.raise_for_status()
        tables = pd.read_html(io.StringIO(r.text))
        df = tables[0]
        df.columns = [str(c).strip() for c in df.columns]
        if "Date" not in df.columns or "Value" not in df.columns:
            return pd.Series(dtype=float, name="sp500_pe")
        dt = pd.to_datetime(df["Date"], errors="coerce")
        # The Value cell can carry a stray prefix byte ("� 32.69") — extract the number.
        val = pd.to_numeric(
            df["Value"].astype(str).str.extract(r"([0-9]+\.?[0-9]*)")[0],
            errors="coerce",
        )
        s = pd.Series(val.values, index=dt).dropna()
        s = s[s > 0]
        s = s[~s.index.duplicated(keep="last")].sort_index()
        s.name = "sp500_pe"
        return s
    except Exception as e:
        log.info("multpl S&P PE unavailable (%s)", str(e)[:80])
        return pd.Series(dtype=float, name="sp500_pe")


def equity_risk_premium() -> pd.Series:
    """
    ERP = S&P 500 trailing earnings yield (100 / trailing P/E) − 10Y Treasury yield.

    Earnings yield comes from multpl.com's monthly S&P 500 P/E, forward-filled
    onto the daily yield calendar — a *real* valuation signal that moves with P/E
    expansion/compression. If the P/E source is unavailable the series is empty
    (the valuation pillar is then honestly absent and the coverage gate reflects
    it), rather than the previous constant earnings yield which made ERP a mere
    sign-flipped duplicate of the 10Y rate.
    """
    pe = sp500_pe_ratio()
    ten = fred_dgs10()
    if pe.empty or ten.empty:
        record_provenance("equity_risk_premium", "", kind="unavailable",
                          note="multpl P/E or 10Y yield feed missing")
        return pd.Series(dtype=float, name="erp")
    ey = 100.0 / pe  # earnings yield in percent
    ey_daily = ey.sort_index().reindex(ten.index, method="ffill")
    erp = (ey_daily - ten).dropna()
    erp.name = "erp"
    pe_age_days = int((ten.index[-1] - pe.index[-1]).days) if len(pe) else -1
    record_provenance(
        "equity_risk_premium", "multpl_pe+fred_dgs10", kind="primary",
        note=f"monthly P/E ffilled to daily; P/E print {pe_age_days}d old",
    )
    return erp


# ---------------------------------------------------------------------------
# Advanced: correlation cluster + DIX + MOVE/VIX divergence
# ---------------------------------------------------------------------------
def correlation_cluster() -> pd.Series:
    """20-day rolling correlation of SPY vs (TLT+GLD)/2. Near 1.0 = liquidity event."""
    spy = yf_series("SPY", period="3y")
    tlt = yf_series("TLT", period="3y")
    gld = yf_series("GLD", period="3y")
    if spy.empty or tlt.empty or gld.empty:
        return pd.Series(dtype=float, name="corr_cluster")
    df = pd.concat([spy.rename("spy"), tlt.rename("tlt"), gld.rename("gld")], axis=1).dropna()
    rets = df.pct_change()
    hedge = (rets["tlt"] + rets["gld"]) / 2
    corr = rets["spy"].rolling(20).corr(hedge)
    corr.name = "corr_cluster"
    return corr.dropna()


def move_vs_vix_spread() -> pd.Series:
    """MOVE (or proxy) divided by VIX, z-scored. >0 means bond vol elevated vs equity vol."""
    v = vix()
    m = move_index()
    if v.empty or m.empty:
        return pd.Series(dtype=float, name="move_vix_div")
    df = pd.concat([m.rename("move"), v.rename("vix")], axis=1).dropna()
    ratio = df["move"] / df["vix"]
    ratio.name = "move_vix_div"
    return ratio


@_memoized_fetch
def _squeezemetrics_csv() -> pd.DataFrame:
    """
    Fetch the SqueezeMetrics public CSV. Returns DataFrame with date index
    and columns for dix, gex (if present). Cached to avoid repeat calls.

    The endpoint was paywalled in late 2024 but currently serves the full
    history again — treat it as flaky, not gone. The ~200KB download can
    exceed a short timeout while ~30 fetchers share the line on a cold
    build, so retry once before giving up.
    """
    urls = [
        "https://squeezemetrics.com/monitor/static/DIX.csv",
    ]
    for url in urls:
        for attempt in (1, 2):
            try:
                r = requests.get(url, headers=UA, timeout=20)
                if r.ok and "," in r.text and len(r.text) > 100:
                    df = pd.read_csv(io.StringIO(r.text))
                    df.columns = [c.strip().lower() for c in df.columns]
                    if "date" in df.columns:
                        df["date"] = pd.to_datetime(df["date"], errors="coerce")
                        df = df.dropna(subset=["date"]).set_index("date").sort_index()
                        return df
                break  # got a response but not parseable data — retry won't help
            except Exception as e:
                log.info("SqueezeMetrics CSV fetch failed (attempt %d): %s", attempt, e)
    return pd.DataFrame()


def _dix_percent(s: pd.Series) -> pd.Series:
    """DIX in percent. The source publishes a fraction (0.44); the chart's
    35 / 45 / 55 reference lines and the watchlist's "1 point above its 60-day
    average" test are in percent, so on the raw fraction the lines were off the
    chart and the alert could never fire."""
    out = pd.to_numeric(s, errors="coerce").dropna()
    if not out.empty and float(out.abs().median()) <= 1.5:
        out = out * 100.0
    out.name = "dix"
    return out


def dix_proxy() -> pd.Series:
    """
    Dark Index (DIX) — share of dark pool volume that is buying, in percent.
    Rising DIX during a sell-off = institutional accumulation.

    Source: SqueezeMetrics public CSV. Last-good copy persisted to
    cache/dix_history.csv so a transient fetch failure degrades to
    stale-but-real history instead of an empty chart (the empty result
    would otherwise be pinned for an hour by the app-level cache).
    """
    df = _squeezemetrics_csv()
    if not df.empty and "dix" in df.columns:
        s = df["dix"].astype(float).dropna()
        s.name = "dix"
        _save_series_cache("dix_history.csv", s, "dix")
        record_provenance("dix", "squeezemetrics", kind="primary")
        return _closed_only(_dix_percent(s))
    cached = _load_series_cache("dix_history.csv", "dix")
    if not cached.empty:
        record_provenance(
            "dix", "squeezemetrics", kind="cache",
            note=f"live fetch failed; cached history through {cached.index.max():%Y-%m-%d}",
        )
        return _closed_only(_dix_percent(cached))
    record_provenance("dix", "", kind="unavailable",
                      note="SqueezeMetrics fetch failed and no cached history")
    return pd.Series(dtype=float, name="dix")


def _normalize_gex_units(s: pd.Series) -> pd.Series:
    """Normalize a GEX series to $B. Decided per-SERIES (median |value|), not
    per-row: the old per-row mask (abs > 1e6) left mid-range raw-dollar values
    unconverted and could split one feed across two unit regimes."""
    out = pd.to_numeric(s.copy(), errors="coerce").dropna()
    if out.empty:
        out.name = "gamma_exposure"
        return out
    if float(out.abs().median()) > 1e4:  # raw dollars, not $B
        out = out / 1e9
    out.name = "gamma_exposure"
    return out


def squeezemetrics_gex() -> pd.Series:
    """
    GEX (Gamma Exposure) from SqueezeMetrics — historical daily series.
    Much richer history than our options-chain proxy (goes back to 2011).

    Falls back to our computed proxy if SqueezeMetrics is unavailable.
    """
    df = _squeezemetrics_csv()
    if not df.empty and "gex" in df.columns:
        return _normalize_gex_units(df["gex"])
    return pd.Series(dtype=float, name="gamma_exposure")


# ---------------------------------------------------------------------------
# CTA / Leveraged-Fund Equity Positioning (CFTC COT TFF)
# ---------------------------------------------------------------------------
ES_CONTRACT_CODE = "13874A"  # E-mini S&P 500, CME


def cftc_cta_positioning() -> pd.Series:
    """
    CFTC Traders in Financial Futures (TFF) — Leveraged Funds net position
    in S&P 500 E-mini futures as % of open interest.

    Interpretation (contrarian):
      High net-long %  (>10% OI)  → CTAs heavily long → top risk
      High net-short % (<-10% OI) → CTAs capitulated  → bottom setup

    Read from the same full-history CFTC feed as the research desk (Socrata,
    filtered by contract code 13874A, since 2006). The old path downloaded three
    yearly ZIPs, so the series was about 1.5 years deep after warm-up, had no
    cache or seed behind it, and could disagree with the positioning table.
    """
    from src import research_data as RD  # local import: research_data imports this module

    name = "cta_positioning"
    try:
        cot = RD.cot_positioning()
    except Exception as e:
        log.info("CFTC positioning unavailable: %s", str(e)[:100])
        cot = pd.DataFrame()
    if cot is None or cot.empty or "ES" not in cot.columns:
        record_provenance(name, "", kind="unavailable")
        return pd.Series(dtype=float, name=name)
    s = pd.to_numeric(cot["ES"], errors="coerce").dropna()
    s.name = name
    feed = get_provenance().get("cot_positioning") or {}
    record_provenance(
        name, feed.get("source") or "cftc:publicreporting (TFF)",
        kind=feed.get("kind", "primary"), note=feed.get("note", ""),
    )
    return s


# ---------------------------------------------------------------------------
# FINRA margin debt (leverage / positioning)
# ---------------------------------------------------------------------------
# Monthly "Debit Balances in Customers' Securities Margin Accounts" — the
# market's aggregate leverage. Rapid YoY expansion has marked euphoric tops
# (2000, 2007, 2021); YoY contraction accompanies deleveraging washouts. FINRA
# publishes it only as a Cloudflare-gated page/XLSX ("data feeds are not
# available"), which blocks a direct runtime fetch from datacenter/shared IPs,
# so the live feed is YCharts' public indicator page. Values there carry
# magnitude suffixes ("1.416T", "920.96B") that the generic put/call parser
# reads as NaN, so parse them here and normalise everything to $ billions.

# Bundled fallback so a fresh cloud deploy (empty local cache) still renders the
# chart even when YCharts is blocked on a shared IP. Real FINRA values (obtained
# via YCharts), $ billions, month-end. Refreshed opportunistically by the live
# feed; extend it by re-running the generator when it drifts too far behind.
_MARGIN_DEBT_SEED: list[tuple[str, float]] = [
    ("2022-04-30", 772.94), ("2022-05-31", 752.94), ("2022-06-30", 683.44),
    ("2022-07-31", 696.78), ("2022-08-31", 687.79), ("2022-09-30", 664.01),
    ("2022-10-31", 649.62), ("2022-11-30", 643.78), ("2022-12-31", 606.66),
    ("2023-01-31", 641.23), ("2023-02-28", 624.38), ("2023-03-31", 645.43),
    ("2023-04-30", 631.95), ("2023-05-31", 644.17), ("2023-06-30", 681.23),
    ("2023-07-31", 709.83), ("2023-08-31", 689.18), ("2023-09-30", 680.85),
    ("2023-10-31", 635.28), ("2023-11-30", 660.89), ("2023-12-31", 700.77),
    ("2024-01-31", 701.98), ("2024-02-29", 742.96), ("2024-03-31", 784.14),
    ("2024-04-30", 775.46), ("2024-05-31", 809.43), ("2024-06-30", 809.32),
    ("2024-07-31", 810.84), ("2024-08-31", 797.16), ("2024-09-30", 813.21),
    ("2024-10-31", 815.37), ("2024-11-30", 890.85), ("2024-12-31", 899.17),
    ("2025-01-31", 937.25), ("2025-02-28", 918.14), ("2025-03-31", 880.32),
    ("2025-04-30", 850.56), ("2025-05-31", 920.96), ("2025-06-30", 1008.0),
    ("2025-07-31", 1023.0), ("2025-08-31", 1060.0), ("2025-09-30", 1126.0),
    ("2025-10-31", 1184.0), ("2025-11-30", 1214.0), ("2025-12-31", 1226.0),
    ("2026-01-31", 1279.0), ("2026-02-28", 1253.0), ("2026-03-31", 1221.0),
    ("2026-04-30", 1304.0), ("2026-05-31", 1416.0),
]


def _parse_ycharts_magnitude(text) -> float | None:
    """Parse a YCharts abbreviated magnitude to $ billions.

    "1.416T" -> 1416.0, "920.96B" -> 920.96, "1,234M" -> 1.234. A bare number
    is assumed to already be in billions. Returns None when no number is present.
    """
    m = re.search(r"(-?[0-9][0-9,]*\.?[0-9]*)\s*([TBMK])?", str(text).strip().upper())
    if not m:
        return None
    try:
        num = float(m.group(1).replace(",", ""))
    except ValueError:
        return None
    return num * {"T": 1000.0, "B": 1.0, "M": 1e-3, "K": 1e-6, None: 1.0}[m.group(2)]


def _margin_debt_seed_series() -> pd.Series:
    """The bundled seed as a tz-naive, sorted monthly Series in $ billions."""
    idx = pd.to_datetime([d for d, _ in _MARGIN_DEBT_SEED])
    s = pd.Series([v for _, v in _MARGIN_DEBT_SEED], index=idx, name="margin_debt")
    return s.sort_index()


def _ycharts_margin_debt() -> pd.Series:
    """FINRA margin debt ($B) from YCharts' public indicator page (no key).

    YCharts renders ~50 recent months across two Date/Value tables; the values
    use magnitude suffixes, so parse with _parse_ycharts_magnitude rather than the
    plain-decimal path used by the put/call slugs.
    """
    url = "https://ycharts.com/indicators/finra_margin_debt"
    try:
        r = requests.get(url, headers=UA, timeout=12)
        if not r.ok:
            return pd.Series(dtype=float, name="margin_debt")
        tables = pd.read_html(io.StringIO(r.text))
    except Exception as e:
        log.info("YCharts margin debt unavailable (%s)", str(e)[:80])
        return pd.Series(dtype=float, name="margin_debt")
    parts = []
    for df in tables:
        cols = [str(c).strip() for c in df.columns]
        if "Date" not in cols or "Value" not in cols:
            continue
        tmp = df[[cols[cols.index("Date")], cols[cols.index("Value")]]].copy()
        tmp.columns = ["date", "value"]
        tmp["date"] = pd.to_datetime(tmp["date"], errors="coerce")
        tmp["value"] = tmp["value"].map(_parse_ycharts_magnitude)
        s = tmp.dropna().set_index("date")["value"]
        if not s.empty:
            parts.append(s)
    return _merge_series("margin_debt", *parts)


def finra_margin_debt() -> pd.Series:
    """
    FINRA margin debt — monthly "Debit Balances in Customers' Securities Margin
    Accounts", in $ billions.

    Source chain (FINRA offers no API/data-feed and Cloudflare-gates its page and
    XLSX, so the series cannot be read from FINRA directly at runtime):
      1. YCharts public indicator page — primary live feed (~50 recent months)
      2. local disk cache — self-updating; survives a transient YCharts failure
      3. bundled seed constant — guarantees the chart renders on a fresh cloud
         deploy where the cache is empty and YCharts is blocked on a shared IP
    Live rows override cached/seed rows for a shared month.
    """
    live = _ycharts_margin_debt()
    cache = _load_series_cache("margin_debt_history.csv", "margin_debt")
    seed = _margin_debt_seed_series()
    merged = _merge_series("margin_debt", seed, cache, live)
    if merged.empty:
        record_provenance("margin_debt", "", kind="unavailable")
        return merged
    if not live.empty:
        _save_series_cache("margin_debt_history.csv", merged, "margin_debt")
        record_provenance("margin_debt", "ycharts:finra_margin_debt", kind="primary")
    elif not cache.empty:
        record_provenance(
            "margin_debt", "ycharts:finra_margin_debt", kind="cache",
            note=f"live fetch failed; cached history through {cache.index.max():%Y-%m-%d}",
        )
    else:
        record_provenance(
            "margin_debt", "bundled_seed", kind="fallback",
            note=f"YCharts unreachable and no local cache; seed through {seed.index.max():%Y-%m-%d}",
        )
    return merged


# ---------------------------------------------------------------------------
# Yield curve
# ---------------------------------------------------------------------------
@_memoized_fetch
def curve_2s10s() -> pd.Series:
    """10Y - 2Y Treasury yield spread (FRED T10Y2Y). Negative = inverted."""
    s = _fred("T10Y2Y", prov_key="curve_2s10s")
    s.name = "curve_2s10s"
    return s


def curve_3m10y() -> pd.Series:
    """10Y - 3M Treasury yield spread (FRED T10Y3M). Fed's preferred recession curve."""
    s = _fred("T10Y3M", prov_key="curve_3m10y")
    s.name = "curve_3m10y"
    return s


def curve_resteep_2s10s(window: int = 252) -> pd.Series:
    """
    Re-steepening-from-inversion signal on 2s10s.

    = max(0, curve_today - rolling_min(curve, 252d))  only when rolling_min < 0.

    Fires only when the curve was inverted at some point in the past year and has
    since recovered — historically coincides with recession onset and equity tops
    (2000, 2007, 2020). Large positive values = active un-inversion = top-risk signal.
    Flat zero values mean either "never inverted in past year" or "still at the trough".
    """
    c = curve_2s10s()
    if c.empty:
        return pd.Series(dtype=float, name="curve_resteep_2s10s")
    rmin = c.rolling(window, min_periods=60).min()
    raw = (c - rmin).clip(lower=0)
    sig = raw.where(rmin < 0, 0.0)
    sig.name = "curve_resteep_2s10s"
    return sig.dropna()


# ---------------------------------------------------------------------------
# Credit spread velocity
# ---------------------------------------------------------------------------
def hy_spread_velocity(lookback_days: int = 20) -> pd.Series:
    """
    4-week (≈20 business day) change in HY OAS, expressed in BPS.

    FRED BAMLH0A0HYM2 is in percent, so we convert to bps first (×100), then diff.
    Positive values = spreads widening (stress building). >+75 bps in 4w is the
    cleanest "risk-off now" trigger in the data (2020, 2022, 2023 regional banks).
    """
    s = fred_hy_spread()
    if s.empty:
        return pd.Series(dtype=float, name="hy_spread_velocity")
    bps = s * 100.0
    vel = bps - bps.shift(lookback_days)
    vel.name = "hy_spread_velocity"
    return vel.dropna()


# ---------------------------------------------------------------------------
# Interbank Funding / Repo Market (FRA-OIS, SOFR)
# ---------------------------------------------------------------------------
def fra_ois_spread() -> pd.Series:
    """
    Funding stress: 3-month AA financial commercial paper minus the 3-month
    T-bill, in basis points. The closest free successor to the TED spread: what
    banks pay for unsecured 3-month money over the risk-free bill.

    Both legs are forward-looking 3-month rates on a discount basis. The series
    this replaces compared the SOFR 90-day AVERAGE, which looks back over the
    last three months, with the bill, which looks forward. That gap is the
    expected policy path, not funding stress: it read -96bp while the Fed hiked
    in mid-2022 and +64bp as it cut in Sep 2024, with funding markets calm both
    times.

    Normal is about 10bp. March 2020 reached 257bp, March 2023 41bp. Daily CP
    prints are noisy, so the scoring config smooths this over 5 sessions.
    """
    name = "fra_ois_spread"
    tbill3m = _fred("DTB3")
    cp3m = _fred("DCPF3M")  # 3M AA Financial Commercial Paper rate
    if not cp3m.empty and not tbill3m.empty:
        # CP is not published every day: carry it a few sessions, no further.
        df = pd.concat([cp3m.rename("cp"), tbill3m.rename("tbill")], axis=1, sort=True)
        df["cp"] = df["cp"].ffill(limit=3)
        df = df.dropna()
        if len(df) > 30:
            spread = ((df["cp"] - df["tbill"]) * 100.0).dropna()  # percent -> bps
            spread.name = name
            record_provenance(name, "fred:DCPF3M-DTB3", kind="primary", note="3M AA financial CP minus 3M T-bill")
            return spread

    # Fallback: A2/P2 minus AA commercial paper (credit-tier spread). A different
    # measure, so it is labelled a proxy and never spliced onto the series above.
    cp_a2p2 = _fred("DCPN3M")
    if not cp_a2p2.empty and not cp3m.empty:
        df = pd.concat([cp_a2p2.rename("low"), cp3m.rename("high")], axis=1, sort=True).ffill(limit=3).dropna()
        if len(df) > 30:
            spread = ((df["low"] - df["high"]) * 100.0).dropna()
            spread.name = name
            record_provenance(name, "fred:DCPN3M-DCPF3M", kind="proxy", note="A2/P2 minus AA CP; T-bill leg unavailable")
            return spread

    record_provenance(name, "", kind="unavailable")
    return pd.Series(dtype=float, name=name)


def sofr_spread() -> pd.Series:
    """
    SOFR (Secured Overnight Financing Rate) spread vs Effective Fed Funds.
    Elevated SOFR = repo market stress / collateral scarcity.

    Notable spikes:
    - Sept 2019: Repo crisis (SOFR ~5% above Fed Funds)
    - March 2020: COVID liquidity freeze
    """
    sofr = _fred("SOFR")  # Overnight secured rate
    effr = _fred("EFFR")  # Effective Fed Funds Rate

    if sofr.empty or effr.empty:
        return pd.Series(dtype=float, name="sofr_spread")

    df = pd.concat([sofr.rename("sofr"), effr.rename("effr")], axis=1).dropna()
    if df.empty:
        return pd.Series(dtype=float, name="sofr_spread")

    # Spread in basis points
    spread = (df["sofr"] - df["effr"]) * 100
    spread.name = "sofr_spread"
    return spread.dropna()


# ---------------------------------------------------------------------------
# Gamma Exposure (GEX) - Options Market Structure
# ---------------------------------------------------------------------------
def _cboe_gamma_snapshot(symbol: str = "SPY") -> tuple[float | None, float | None]:
    """
    Current GEX and gamma-flip distance from CBOE's delayed option-chain JSON.

    GEX is expressed as $B per 1% underlying move. The flip distance is still an
    estimate: without recomputing greeks across hypothetical spot prices, the
    most stable no-key proxy is the midpoint between gamma/OI-weighted put and
    call strikes across the front expiries.
    """
    df, spot = _cboe_delayed_options(symbol)
    sub = _near_expiry_options(df, max_expiries=4)
    if sub.empty or not np.isfinite(spot) or spot <= 0:
        return None, None

    oi = sub.get("open_interest", pd.Series(0.0, index=sub.index)).fillna(0.0)
    gamma = sub.get("gamma", pd.Series(0.0, index=sub.index)).fillna(0.0)
    strikes = sub["strike"].fillna(0.0)
    sign = np.where(sub["type"].eq("C"), 1.0, -1.0)
    gamma_dollars = pd.Series(sign * gamma * oi * 100.0 * (spot ** 2) * 0.01, index=sub.index)
    gex = float(gamma_dollars.sum() / 1e9)

    call_w = (gamma.abs() * oi).where(sub["type"].eq("C"), 0.0)
    put_w = (gamma.abs() * oi).where(sub["type"].eq("P"), 0.0)
    flip_distance = None
    if call_w.sum() > 0 and put_w.sum() > 0:
        call_strike = float((strikes * call_w).sum() / call_w.sum())
        put_strike = float((strikes * put_w).sum() / put_w.sum())
        flip_zone = (call_strike + put_strike) / 2.0
        if flip_zone > 0:
            flip_distance = float((spot - flip_zone) / spot * 100.0)

    return gex, flip_distance


def _gex_implied_gamma_flip_history(gex: pd.Series) -> pd.Series:
    """
    Backfill a display/history proxy for gamma-flip distance from historical GEX.

    True historical zero-gamma requires point-in-time option chains, which are
    not publicly available without a paid feed. When we have historical GEX but
    only a live flip snapshot, this keeps the chart useful by preserving the
    sign and intensity of the gamma regime instead of showing an empty panel.
    """
    if gex is None or gex.empty or len(gex.dropna()) < 30:
        return pd.Series(dtype=float, name="gamma_flip")
    s = pd.to_numeric(gex.copy(), errors="coerce").dropna()
    scale = s.abs().rolling(252, min_periods=30).quantile(0.85)
    fallback_scale = float(s.abs().quantile(0.85)) if not s.empty else np.nan
    if not np.isfinite(fallback_scale) or fallback_scale <= 0:
        return pd.Series(dtype=float, name="gamma_flip")
    scale = scale.replace(0, np.nan).ffill().fillna(fallback_scale)
    proxy = (s / scale).clip(-2.5, 2.5) * 1.25
    proxy.name = "gamma_flip"
    return proxy.dropna()


def _session_in_progress() -> bool:
    return SES.in_trading_hours() and not SES.include_live()


def _snapshot_stamp() -> pd.Timestamp | None:
    """The session a delayed-chain snapshot belongs to, or None while one is printing.

    Greeks in the chain move with spot all day, so a snapshot taken at 11am is
    not that day's reading and must not be saved as one.
    """
    if _session_in_progress():
        return None
    return last_completed_trading_day()


@_memoized_fetch
def gamma_exposure_proxy() -> pd.Series:
    """
    Gamma Exposure (GEX) — estimated dealer gamma, in $B per 1% move.

    Two measured sources exist and they are NOT comparable:
      - 'squeezemetrics' : SPX dealer gamma, daily history since 2011
      - 'cboe_snapshot'  : naive call-minus-put gamma summed over SPY's front
                           expiries from CBOE's delayed chain

    The scored series is one methodology only. SqueezeMetrics wins whenever it
    has any rows; the SPY snapshot is then shown for context and not scored.
    Stamping the snapshot over SqueezeMetrics' latest day, as this used to,
    ranked a -3.9 SPY reading against a history whose 1st percentile is -1.0:
    a false "dealers are short gamma" extreme in the composite, the cluster and
    the leverage gate. The snapshot is scored only where SqueezeMetrics has
    never served data, so it is still a single methodology.

    Rows written before provenance tracking stay 'legacy_mixed' and display-only.

    When GEX is deeply negative, dealers must sell into weakness (crash
    accelerant); positive GEX = dealers buy dips, sell rips (stabilizing).
    """
    df = _load_series_cache_with_source("gamma_exposure_history.csv", "gamma_exposure")

    today_val, _ = _cboe_gamma_snapshot("SPY")
    stamp = _snapshot_stamp() if today_val is not None else None
    if stamp is not None:
        live = pd.Series([float(today_val)], index=[stamp])
        df = _upsert_sourced_rows(df, live, "gamma_exposure", "cboe_snapshot")

    # Applied last, so the measured SPX backbone keeps every date it covers.
    sqz = _closed_only(squeezemetrics_gex())
    if not sqz.empty:
        df = _upsert_sourced_rows(df, _normalize_gex_units(sqz), "gamma_exposure", "squeezemetrics")

    if stamp is not None or not sqz.empty:
        _save_series_cache_with_source("gamma_exposure_history.csv", df, "gamma_exposure")

    has_sqz = bool((df["source"] == "squeezemetrics").any())
    scored_mask = df["source"] == ("squeezemetrics" if has_sqz else "cboe_snapshot")
    display_only = df.loc[~scored_mask, "gamma_exposure"]
    if not display_only.empty:
        _set_display_overlay("gamma_exposure", display_only.rename("gamma_exposure"))

    out = df.loc[scored_mask, "gamma_exposure"].copy()
    out.name = "gamma_exposure"
    if out.empty:
        record_provenance("gamma_exposure", "", kind="unavailable",
                          note="no measured GEX source; legacy rows are display-only")
    elif has_sqz and not sqz.empty:
        record_provenance("gamma_exposure", "squeezemetrics", kind="primary", note="SPX dealer gamma")
    elif has_sqz:
        record_provenance("gamma_exposure", "squeezemetrics", kind="cache",
                          note=f"live fetch failed; cached history through {out.index.max():%Y-%m-%d}")
    elif stamp is not None:
        record_provenance("gamma_exposure", "cboe_delayed_options", kind="primary",
                          note="naive GEX from SPY chain greeks; building history")
    else:
        record_provenance("gamma_exposure", "cache(measured rows)", kind="cache",
                          note="no snapshot saved this build")
    return out


def gamma_flip_zone_distance() -> pd.Series:
    """
    Distance (in % terms) to the estimated 'gamma flip' price level.

    The gamma flip zone is where net gamma exposure crosses zero.
    Above flip = positive gamma (dealers sell highs, buy lows) = stable.
    Below flip = negative gamma (dealers sell lows, buy highs) = unstable.

    Scored/persisted rows are MEASURED CBOE-snapshot estimates only (uniform
    % distance units). The GEX-scaled synthetic backfill that used to be merged
    and saved here is now display-only (_gex_implied_gamma_flip_history): it is
    in scaled-GEX units, not % distance — persisting it polluted the cache,
    satisfied MIN_OBS and fed the composite/cluster with fabricated extremes.
    The old yfinance OI-weighted flip estimate is deleted for the same reason:
    a second methodology (nearest expiry, OI-only, no gamma weighting) writing
    into the same column.
    """
    df = _load_series_cache_with_source("gamma_flip_history.csv", "gamma_flip")

    _, today_val = _cboe_gamma_snapshot("SPY")
    stamp = _snapshot_stamp() if today_val is not None else None
    if stamp is not None:
        live = pd.Series([float(today_val)], index=[stamp])
        df = _upsert_sourced_rows(df, live, "gamma_flip", "cboe_snapshot")
        _save_series_cache_with_source("gamma_flip_history.csv", df, "gamma_flip")

    # Display-only context for the chart: legacy mixed rows + synthetic proxy.
    scored_mask = df["source"].isin(GAMMA_SCORED_SOURCES)
    legacy = df.loc[~scored_mask, "gamma_flip"]
    proxy_hist = _gex_implied_gamma_flip_history(gamma_exposure_proxy())
    overlay = _merge_series("gamma_flip", legacy, proxy_hist)
    if not overlay.empty:
        _set_display_overlay("gamma_flip_zone", overlay)

    out = df.loc[scored_mask, "gamma_flip"].copy()
    out.name = "gamma_flip"
    if out.empty:
        record_provenance("gamma_flip_zone", "", kind="unavailable",
                          note="no measured flip estimate; backfill is display-only")
    elif stamp is not None:
        record_provenance("gamma_flip_zone", "cboe_delayed_options", kind="primary",
                          note="gamma/OI-weighted strike midpoint estimate; display only")
    else:
        record_provenance("gamma_flip_zone", "cache(measured rows)", kind="cache",
                          note="no snapshot saved this build (session in progress or chain unavailable)")
    return out


def index_put_call_ratio() -> pd.Series:
    """
    Put/Call ratio specifically for INDEX options (SPX, NDX, RUT) vs equity options.

    Retail hedges single stocks (equity puts). Institutions hedge portfolios (index puts).
    High index P/C = institutional panic. Extreme readings followed by sharp drops
    indicate institutions monetizing hedges and buying underlying.

    Official CBOE index P/C only: the archive, the daily statistics files and,
    when reachable, YCharts' copy of recent rows.
    """
    return _official_put_call_series(
        kind="index", ycharts_slug="cboe_index_put_call_ratio", name="index_put_call",
    )


# ---------------------------------------------------------------------------
# Macro context
# ---------------------------------------------------------------------------
def dxy() -> pd.Series:
    """
    US Dollar Index. yfinance symbols vary — try DX-Y.NYB first (ICE cash index),
    then ^DXY, then DX=F (futures) as fallback.
    """
    for sym in ("DX-Y.NYB", "^DXY", "DX=F"):
        s = yf_series(sym, period="10y")
        if not s.empty:
            s.name = "dxy"
            record_provenance(
                "dxy", f"yfinance:{sym}",
                kind="primary" if sym == "DX-Y.NYB" else "fallback",
            )
            return s
    record_provenance("dxy", "", kind="unavailable")
    return pd.Series(dtype=float, name="dxy")


def real_yield_10y() -> pd.Series:
    """10Y TIPS real yield (FRED DFII10), in percent."""
    s = _fred("DFII10", prov_key="real_yield_10y")
    s.name = "real_yield_10y"
    return s


def copper_gold_ratio() -> pd.Series:
    """
    Copper / Gold ratio (front-month futures). Growth proxy — leads HY spreads
    by 1-2 months. High ratio = growth strong / risk-on.
    Low ratio = growth fears / risk-off.
    """
    cu = yf_series("HG=F", period="10y")
    au = yf_series("GC=F", period="10y")
    if cu.empty or au.empty:
        return pd.Series(dtype=float, name="copper_gold")
    df = pd.concat([cu.rename("cu"), au.rename("au")], axis=1).dropna()
    if df.empty:
        return pd.Series(dtype=float, name="copper_gold")
    s = (df["cu"] / df["au"]).replace([np.inf, -np.inf], np.nan).dropna()
    s.name = "copper_gold"
    return s



# ---------------------------------------------------------------------------
# Committed seeds + frame caches (robust fallback chain for new panels)
# ---------------------------------------------------------------------------
# `cache/` is gitignored and ephemeral on Streamlit Cloud, so a disk cache only
# helps a warm container. Every new chart panel therefore follows:
#     live fetch  ->  local disk cache  ->  committed seed (src/seeds/*.csv)
# The seed guarantees a fresh cloud deploy still renders when Yahoo or a
# scraped page is blocked on a shared IP. Live rows win on shared dates.
# Regenerate seeds with:  python -m scripts.refresh_seeds
SEED_DIR = os.path.join(os.path.dirname(__file__), "seeds")


def _seed_path(filename: str) -> str:
    return os.path.join(SEED_DIR, filename)


def _read_dated_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        df = pd.read_csv(path, parse_dates=["date"])
        df = df.dropna(subset=["date"]).set_index("date").sort_index()
        df = df[~df.index.duplicated(keep="last")]
        return df.apply(pd.to_numeric, errors="coerce")
    except Exception as e:
        log.warning("Dated CSV read failed for %s: %s", path, e)
        return pd.DataFrame()


def _load_frame_cache(filename: str) -> pd.DataFrame:
    return _read_dated_csv(_series_cache_path(filename))


def _save_frame_cache(filename: str, df: pd.DataFrame) -> None:
    try:
        out = df[~df.index.duplicated(keep="last")].sort_index()
        out.index.name = "date"
        out.reset_index().to_csv(_series_cache_path(filename), index=False)
    except Exception as e:
        log.warning("Frame cache write failed for %s: %s", filename, e)


def _load_seed_frame(filename: str) -> pd.DataFrame:
    return _read_dated_csv(_seed_path(filename))


def _merge_frames(*frames: pd.DataFrame) -> pd.DataFrame:
    """Union of dated frames; later frames win on duplicate dates."""
    parts = [f for f in frames if f is not None and not f.empty]
    if not parts:
        return pd.DataFrame()
    out = pd.concat(parts)
    out = out[~out.index.duplicated(keep="last")].sort_index()
    out.index.name = "date"
    return out


def _with_fallback_chain(
    *,
    keys: tuple[str, ...],
    live: pd.DataFrame,
    cache_file: str,
    seed_file: str,
    source: str,
) -> pd.DataFrame:
    """live -> disk cache -> committed seed, with provenance for every key."""
    # Closed sessions only, on every leg: a cache written while a session was
    # printing would otherwise hand back that half-day row as a close.
    cache = _closed_only(_load_frame_cache(cache_file))
    seed = _load_seed_frame(seed_file)
    live = _closed_only(live) if live is not None else live
    if live is not None and not live.empty:
        merged = _merge_frames(seed, cache, live)
        _save_frame_cache(cache_file, merged)
        for k in keys:
            record_provenance(k, source, kind="primary")
        return merged
    if not cache.empty:
        merged = _merge_frames(seed, cache)
        for k in keys:
            record_provenance(
                k, source, kind="cache",
                note=f"live fetch failed; cached history through {cache.index.max():%Y-%m-%d}",
            )
        return merged
    if not seed.empty:
        for k in keys:
            record_provenance(
                k, "bundled_seed", kind="fallback",
                note=f"live source unreachable and no local cache; seed through {seed.index.max():%Y-%m-%d}",
            )
        return seed
    for k in keys:
        record_provenance(k, "", kind="unavailable")
    return pd.DataFrame()


def _frame_col(df: pd.DataFrame, col: str, name: str) -> pd.Series:
    if df is None or df.empty or col not in df.columns:
        return pd.Series(dtype=float, name=name)
    s = pd.to_numeric(df[col], errors="coerce").dropna()
    s.name = name
    return s


# ---------------------------------------------------------------------------
# Full-universe S&P 500 participation: 10/20DMA trend breadth + 50D OB/OS
# ---------------------------------------------------------------------------
PARTICIPATION_KEYS = ("trend_breadth_10_20", "obos_spread_50d", "obos_ob_pct", "obos_os_pct")


@_memoized_fetch
def sp500_participation_frame() -> pd.DataFrame:
    """
    Daily participation panel computed from the shared full-constituent closes:
      trend_10_20     % of constituents with 10DMA > 20DMA
      trend_eligible  constituents with both averages defined
      ob_pct / os_pct % overbought / oversold vs own 50D band (z > +1 / < -1)
      obos_spread     ob_pct - os_pct (pp)
      obos_eligible   constituents with a defined 50D band
    """
    panel, wanted = _panel_columns()
    live = pd.DataFrame()
    if wanted:
        p = panel[wanted]
        tb = A.trend_breadth(p, 10, 20).rename(
            columns={"pct": "trend_10_20", "eligible": "trend_eligible"})
        ob = A.overbought_oversold_breadth(p, 50, 1.0).rename(
            columns={"spread": "obos_spread", "eligible": "obos_eligible"})
        live = pd.concat([tb, ob], axis=1).sort_index()
    return _with_fallback_chain(
        keys=PARTICIPATION_KEYS,
        live=live,
        cache_file="sp500_participation_history.csv",
        seed_file="sp500_participation_seed.csv",
        source=f"sp500_panel({len(wanted)} constituents)" if wanted else "sp500_panel",
    )


def trend_breadth_10_20() -> pd.Series:
    return _frame_col(sp500_participation_frame(), "trend_10_20", "trend_breadth_10_20")


def trend_breadth_eligible() -> pd.Series:
    return _frame_col(sp500_participation_frame(), "trend_eligible", "trend_breadth_eligible")


def obos_spread_50d() -> pd.Series:
    return _frame_col(sp500_participation_frame(), "obos_spread", "obos_spread_50d")


def obos_overbought_pct() -> pd.Series:
    return _frame_col(sp500_participation_frame(), "ob_pct", "obos_ob_pct")


def obos_oversold_pct() -> pd.Series:
    return _frame_col(sp500_participation_frame(), "os_pct", "obos_os_pct")


def obos_eligible() -> pd.Series:
    return _frame_col(sp500_participation_frame(), "obos_eligible", "obos_eligible")


# ---------------------------------------------------------------------------
# Style rotation: price legs for relative-strength ratios
# ---------------------------------------------------------------------------
def iwm_etf() -> pd.Series:
    """iShares Russell 2000 ETF — fallback numerator if ^RUT is unavailable."""
    return yf_series("IWM")


# ---------------------------------------------------------------------------
# Nasdaq 100 valuation — forward P/E (VCP Scanner) + trailing P/E (WorldPERatio)
# ---------------------------------------------------------------------------
# Bloomberg's BEst forward P/E history is paid. The free, keyless sources:
#   * VCP Scanner (vcpscanner.com/market-valuation/nasdaq-100) embeds a daily
#     bottom-up NDX valuation history (consensus forward P/E since 2026-03) in
#     its Next.js flight payload. Its trailing-P/E history is NOT used — it has
#     methodology breaks (e.g. 67-74x prints in H2 2022).
#   * WorldPERatio (worldperatio.com/index/nasdaq-100/) publishes a monthly
#     trailing P/E for the Nasdaq 100 (computed on QQQ) since 1990, stamped at
#     the 1st of each month but holding the END-of-month value (verified: the
#     monthly change correlates 0.76 with month-end QQQ returns vs -0.07 for
#     start-of-month). The final point is the live value "calculated on" the
#     date printed on the page.
# Forward history accumulates in the disk cache each build; the trailing
# series supplies the genuine 5-year perspective and drives the scored
# valuation indicator.
_PE_SANE = (3.0, 200.0)


def _next_flight_text(html: str) -> str:
    """Concatenate + JSON-unescape a Next.js app-router flight payload."""
    chunks = re.findall(r'self\.__next_f\.push\(\[1,"(.*?)"\]\)', html, re.S)
    out = []
    for c in chunks:
        try:
            out.append(json.loads('"' + c + '"'))
        except Exception:
            continue
    return "".join(out)


def _parse_vcp_history(html: str) -> pd.DataFrame:
    text = _next_flight_text(html)
    i = text.find('"initialHistory"')
    if i < 0:
        return pd.DataFrame()
    start = text.find("[", i)
    if start < 0:
        return pd.DataFrame()
    try:
        arr, _ = json.JSONDecoder().raw_decode(text[start:])
    except Exception:
        return pd.DataFrame()
    if not isinstance(arr, list) or not arr:
        return pd.DataFrame()
    df = pd.DataFrame(arr)
    if "snapshot_date" not in df.columns:
        return pd.DataFrame()
    df["date"] = pd.to_datetime(df["snapshot_date"], errors="coerce")
    df = df.dropna(subset=["date"]).set_index("date").sort_index()
    df = df[~df.index.duplicated(keep="last")]
    out = pd.DataFrame(index=df.index)
    for col in ("forward_pe", "trailing_pe"):
        if col in df.columns:
            v = pd.to_numeric(df[col], errors="coerce")
            out[col] = v.where((v > _PE_SANE[0]) & (v < _PE_SANE[1]))
    out.index.name = "date"
    return out


def _vcp_ndx_history() -> pd.DataFrame:
    url = "https://vcpscanner.com/market-valuation/nasdaq-100"
    try:
        r = requests.get(url, headers=UA, timeout=15)
        if not r.ok:
            return pd.DataFrame()
        return _parse_vcp_history(r.text)
    except Exception as e:
        log.info("VCP Scanner NDX valuation unavailable (%s)", str(e)[:80])
        return pd.DataFrame()


def ndx_forward_pe() -> pd.Series:
    """Nasdaq 100 forward (next-12M consensus) P/E, daily. live -> cache -> seed."""
    hist = _vcp_ndx_history()
    live = pd.DataFrame()
    if not hist.empty and "forward_pe" in hist.columns:
        live = hist[["forward_pe"]].dropna()
    df = _with_fallback_chain(
        keys=("ndx_forward_pe",),
        live=live,
        cache_file="ndx_forward_pe_history.csv",
        seed_file="ndx_forward_pe_seed.csv",
        source="vcpscanner:nasdaq100 forward_pe",
    )
    return _closed_only(_frame_col(df, "forward_pe", "ndx_forward_pe"))


def _parse_worldperatio(html: str) -> pd.Series:
    empty = pd.Series(dtype=float, name="ndx_trailing_pe")
    m = re.search(r"detailPE_data\s*=\s*\[(.*?)\];", html, re.S)
    if not m:
        return empty
    pts = re.findall(r"Date\.UTC\((\d+),\s*(\d+),\s*(\d+)\)\s*,\s*([-\d\.]+)\]", m.group(1))
    if not pts:
        return empty
    idx, vals = [], []
    for y, mo, _d, v in pts:
        try:
            # JS months are 0-based; the value is the month-END reading.
            idx.append(pd.Timestamp(int(y), int(mo) + 1, 1) + pd.offsets.MonthEnd(0))
            vals.append(float(v))
        except Exception:
            continue
    if not idx:
        return empty
    s = pd.Series(vals, index=pd.DatetimeIndex(idx), name="ndx_trailing_pe").sort_index()
    s = s[~s.index.duplicated(keep="last")]
    s = s[(s > _PE_SANE[0]) & (s < _PE_SANE[1])]
    # The last point is the live, month-to-date value: stamp it with the page's
    # "calculated on" date instead of the (future) month-end.
    plain = re.sub(r"<[^>]+>", " ", html)
    plain = re.sub(r"\s+", " ", plain)
    calc = re.search(r"calculated on\s*(\d{1,2}\s+[A-Za-z]+\s+\d{4})", plain)
    if calc and len(s) >= 2:
        try:
            calc_dt = pd.Timestamp(pd.to_datetime(calc.group(1), errors="raise")).normalize()
            if s.index[-2] < calc_dt <= s.index[-1]:
                cur = re.search(r"Ratio for [^.]*? is\s*([\d\.]+)\s*,?\s*calculated on", plain)
                last_val = float(cur.group(1)) if cur else float(s.iloc[-1])
                s = pd.concat([s.iloc[:-1], pd.Series([last_val], index=[calc_dt])])
                s.name = "ndx_trailing_pe"
        except Exception:
            pass
    return s


def _worldperatio_ndx() -> pd.Series:
    url = "https://worldperatio.com/index/nasdaq-100/"
    try:
        r = requests.get(url, headers=UA, timeout=15)
        if not r.ok:
            return pd.Series(dtype=float, name="ndx_trailing_pe")
        return _parse_worldperatio(r.text)
    except Exception as e:
        log.info("WorldPERatio NDX P/E unavailable (%s)", str(e)[:80])
        return pd.Series(dtype=float, name="ndx_trailing_pe")


def ndx_trailing_pe() -> pd.Series:
    """Nasdaq 100 trailing P/E (QQQ basis): monthly history + live latest point."""
    s = _worldperatio_ndx()
    live = s.to_frame("trailing_pe") if not s.empty else pd.DataFrame()
    df = _with_fallback_chain(
        keys=("ndx_trailing_pe",),
        live=live,
        cache_file="ndx_trailing_pe_history.csv",
        seed_file="ndx_trailing_pe_seed.csv",
        source="worldperatio:nasdaq-100 (QQQ)",
    )
    out = _frame_col(df, "trailing_pe", "ndx_trailing_pe")
    if out.empty:
        return out
    # A future month-end stamp (e.g. a page served without the "calculated on"
    # line) would fake freshness for staleness checks, and a value computed
    # during the session is not a close. Drop both.
    return _closed_only(out[out.index <= pd.Timestamp.now().normalize()])
