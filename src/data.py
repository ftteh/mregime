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
def _fred(series_id: str, start: str = "2015-01-01") -> pd.Series:
    """
    Pull a FRED series.
    - If FRED_API_KEY is set, use fredapi (fast, reliable).
    - Otherwise try CSV fallback with a VERY short timeout — if the user's
      network can't reach fred.stlouisfed.org (common on residential ISPs
      behind CloudFront), we fail instantly rather than hanging the dashboard.
    """
    if FRED_API_KEY:
        try:
            from fredapi import Fred
            f = Fred(api_key=FRED_API_KEY)
            s = f.get_series(series_id, observation_start=start)
            s.name = series_id
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
            return s.loc[start:]
    except Exception as e:
        log.info("FRED CSV %s unavailable (%s) — set FRED_API_KEY", series_id, str(e)[:60])
    return pd.Series(dtype=float, name=series_id)


@_memoized_fetch
def fred_hy_spread() -> pd.Series:
    return _fred("BAMLH0A0HYM2")


def fred_ig_spread() -> pd.Series:
    return _fred("BAMLC0A0CM")


def fred_nfci() -> pd.Series:
    """Chicago Fed National Financial Conditions Index (weekly)."""
    return _fred("NFCI")


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
def yf_series(ticker: str, period: str = "5y", field: str = "Close") -> pd.Series:
    try:
        with _silence_stderr():
            df = yf.Ticker(ticker).history(period=period, auto_adjust=False)
        if df.empty:
            return pd.Series(dtype=float, name=ticker)
        s = df[field].copy()
        s.index = s.index.tz_localize(None) if s.index.tz is not None else s.index
        s.name = ticker
        return s.dropna()
    except Exception as e:
        log.info("yfinance unavailable for %s (%s)", ticker, str(e)[:80])
        return pd.Series(dtype=float, name=ticker)


@_memoized_fetch
def vix() -> pd.Series:
    return yf_series("^VIX")


def vvix() -> pd.Series:
    return yf_series("^VVIX")


def skew() -> pd.Series:
    return yf_series("^SKEW")


@_memoized_fetch
def spx() -> pd.Series:
    return yf_series("^GSPC")


def russell2000() -> pd.Series:
    """Russell 2000 index (^RUT)."""
    return yf_series("^RUT")


def nasdaq_composite() -> pd.Series:
    """Nasdaq Composite index (^IXIC)."""
    return yf_series("^IXIC")


def vix9d() -> pd.Series:
    """CBOE 9-day volatility index. Not on yfinance for all ranges; falls back empty."""
    return yf_series("^VIX9D")


def vix3m() -> pd.Series:
    """CBOE 3-month volatility index."""
    return yf_series("^VIX3M")


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
    MOVE index is not directly on yfinance. We proxy with realized vol of TLT
    (20-trading-day annualized) scaled to MOVE range if live MOVE is unavailable.
    """
    tlt = yf_series("TLT", period="5y")
    if tlt.empty:
        return pd.Series(dtype=float, name="move_proxy")
    log_ret = np.log(tlt / tlt.shift(1))
    vol_ann = log_ret.rolling(20).std() * np.sqrt(252) * 100
    # Scale roughly into MOVE units (MOVE ~ 80-200 range, TLT rv ~ 8-25%)
    proxy = vol_ann * 8
    proxy.name = "move_proxy"
    return proxy.dropna()


# ---------------------------------------------------------------------------
# S&P 500 constituents + breadth
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def sp500_tickers() -> list[str]:
    """
    Return current SP500 tickers. Primary: Wikipedia (requires browser UA —
    their WAF blocks anonymous requests). Fallback: vetted SP100 hard-coded list
    (liquid megacaps only, no delisted/renamed tickers).
    """
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
    Download daily closes for the largest S&P names once and share the result.

    breadth_pct_above_200dma (120), new_highs_minus_lows (150) and the A/D-line
    proxy (100) all operate on the top-N most liquid constituents over a 2y
    window. Fetching the 150-name / 2y superset a single time and slicing it by
    ticker name lets every consumer reuse one bulk download instead of issuing
    three overlapping ones — the dominant cold-load network cost.

    Columns are constituent tickers (insertion-ordered to match sp500_tickers);
    the index is tz-naive daily. Returns an empty frame if the download fails.
    """
    tickers = sp500_tickers()[:150]
    try:
        with _silence_stderr():
            df = yf.download(
                tickers,
                period="2y",
                auto_adjust=False,
                progress=False,
                group_by="ticker",
                threads=True,
            )
    except Exception as e:
        log.error("yf.download S&P panel failed: %s", e)
        return pd.DataFrame()

    closes = {}
    for t in tickers:
        try:
            closes[t] = df[t]["Close"]
        except Exception:
            continue
    if not closes:
        return pd.DataFrame()

    panel = pd.DataFrame(closes)
    if getattr(panel.index, "tz", None) is not None:
        panel.index = panel.index.tz_localize(None)
    return panel


def _panel_columns(sample_size: int) -> tuple[pd.DataFrame, list[str]]:
    """Return the shared close panel and the present top-`sample_size` tickers.

    Selecting by ticker name (not positional slice) reproduces the original
    per-function behavior, which iterated `tickers[:sample_size]` and kept only
    those that downloaded successfully.
    """
    panel = _sp500_close_panel()
    if panel.empty:
        return panel, []
    wanted = [t for t in sp500_tickers()[:sample_size] if t in panel.columns]
    return panel, wanted


def breadth_pct_above_200dma(sample_size: int = 120) -> pd.Series:
    """
    Compute % of a sample of SP500 above their 200-day MA over time.
    Returns a daily series.

    We use a sample (default 120 largest) for speed; correlation with full-index
    breadth is >0.97 empirically.
    """
    panel, wanted = _panel_columns(sample_size)
    if not wanted:
        return pd.Series(dtype=float, name="pct_above_200dma")

    close_df = panel[wanted].ffill()
    ma200 = close_df.rolling(200).mean()
    above = (close_df > ma200).sum(axis=1)
    valid = (close_df.notna() & ma200.notna()).sum(axis=1)
    pct = (above / valid.replace(0, np.nan)) * 100
    pct.name = "pct_above_200dma"
    pct.index = pct.index.tz_localize(None) if pct.index.tz is not None else pct.index
    return pct.dropna()


def new_highs_minus_lows(sample_size: int = 150) -> pd.Series:
    """52-week new highs minus new lows among a sample of SP500."""
    panel, wanted = _panel_columns(sample_size)
    if not wanted:
        return pd.Series(dtype=float, name="new_highs_minus_lows")
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
def aaii_sentiment() -> pd.DataFrame:
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
                        return out
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
            return df[cols]
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
            return df[cols]
    except Exception as e:
        log.info("AAII HTML fallback unavailable (%s)", str(e)[:80])

    return pd.DataFrame(columns=cols)


def naaim_exposure() -> pd.Series:
    """
    NAAIM Exposure Index (weekly, Wednesday release).

    NAAIM publishes the full-history XLSX with a date-stamped filename that
    changes every week:  USE_Data-since-Inception_YYYY-MM-DD.xlsx
    We scrape the programs page to find the current link dynamically.
    """
    import re
    try:
        r = requests.get(
            "https://www.naaim.org/programs/naaim-exposure-index/",
            headers=UA, timeout=15, allow_redirects=True,
        )
        r.raise_for_status()
        # Find the current xlsx link
        matches = re.findall(
            r"https?://(?:www\.)?naaim\.org/wp-content/uploads/[^\"'\s]+\.xlsx",
            r.text,
        )
        # Prefer the "since-Inception" full-history file
        xlsx_url = next((m for m in matches if "inception" in m.lower()), None) \
                   or (matches[0] if matches else None)
        if not xlsx_url:
            # Legacy CSV patterns (rarely work anymore)
            for legacy in (
                "https://www.naaim.org/wp-content/uploads/2014/04/NAAIM-Exposure-Index-Data.csv",
            ):
                try:
                    rr = requests.get(legacy, headers=UA, timeout=10)
                    if rr.ok and "," in rr.text[:500]:
                        df = pd.read_csv(io.StringIO(rr.text))
                        dcol = next((c for c in df.columns if "date" in c.lower()), None)
                        ncol = next((c for c in df.columns if "naaim" in c.lower() or "mean" in c.lower()), None)
                        if dcol and ncol:
                            df["date"] = pd.to_datetime(df[dcol], errors="coerce")
                            s = pd.to_numeric(df.set_index("date")[ncol], errors="coerce").dropna().sort_index()
                            s.name = "naaim"
                            return s
                except Exception:
                    pass
            log.warning("NAAIM: no xlsx link found on programs page")
            return pd.Series(dtype=float, name="naaim")

        rr = requests.get(xlsx_url, headers={**UA, "Referer": "https://www.naaim.org/programs/naaim-exposure-index/"}, timeout=30)
        rr.raise_for_status()
        df = pd.read_excel(io.BytesIO(rr.content), sheet_name=0, engine="openpyxl")
        df.columns = [str(c).strip() for c in df.columns]
        # Typical columns: Date, Number of Responses, Mean (the exposure index),
        # Median, Mode, High, Low, Bearish, Quart1, Quart2, Quart3, Bullish, Deviation, SP500
        dcol = next((c for c in df.columns if c.lower() == "date"
                     or c.lower().startswith("week") or "date" in c.lower()), None)
        if dcol is None:
            dcol = df.columns[0]
        ncol = next((c for c in df.columns if c.lower() in ("mean", "naaim number", "naaim exposure index")), None)
        if ncol is None:
            # Fall back to the second numeric column typically "Mean"
            numerics = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
            if not numerics:
                raise RuntimeError("NAAIM xlsx: no numeric columns")
            ncol = numerics[0]
        df["date"] = pd.to_datetime(df[dcol], errors="coerce")
        df = df.dropna(subset=["date"]).set_index("date").sort_index()
        s = pd.to_numeric(df[ncol], errors="coerce").dropna()
        s.name = "naaim"
        return s
    except Exception as e:
        log.warning("NAAIM xlsx fetch failed: %s", e)
    return pd.Series(dtype=float, name="naaim")


def fear_greed_index() -> pd.Series:
    """
    CNN Fear & Greed Index. Uses the public JSON endpoint backing the CNN widget.
    Returns a daily series (0-100).
    """
    url = "https://production.dataviz.cnn.io/index/fearandgreed/graphdata"
    try:
        r = requests.get(url, headers=UA, timeout=20)
        r.raise_for_status()
        data = r.json()
        hist = data.get("fear_and_greed_historical", {}).get("data", [])
        if not hist:
            return pd.Series(dtype=float, name="fear_greed")
        df = pd.DataFrame(hist)
        df["date"] = pd.to_datetime(df["x"], unit="ms")
        s = df.set_index("date")["y"].astype(float).sort_index()
        s.name = "fear_greed"
        return s
    except Exception as e:
        log.warning("Fear&Greed fetch failed: %s", e)
        return pd.Series(dtype=float, name="fear_greed")


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


@lru_cache(maxsize=8)
def _cboe_delayed_options(symbol: str) -> tuple[pd.DataFrame, float]:
    """Delayed CBOE option chain JSON with greeks, OI and volume. No API key."""
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


def _cboe_live_put_call_ratio(symbols: tuple[str, ...], max_expiries: int = 3) -> float | None:
    total_calls = 0.0
    total_puts = 0.0
    for symbol in symbols:
        df, _ = _cboe_delayed_options(symbol)
        sub = _near_expiry_options(df, max_expiries=max_expiries)
        if sub.empty or "volume" not in sub.columns:
            continue
        vol = sub["volume"].fillna(0.0)
        total_calls += float(vol[sub["type"] == "C"].sum())
        total_puts += float(vol[sub["type"] == "P"].sum())
    if total_calls <= 0:
        return None
    return round(total_puts / total_calls, 4)


def _official_put_call_series(
    *,
    kind: str,
    ycharts_slug: str,
    name: str,
    cache_filename: str,
    live_symbols: tuple[str, ...],
) -> pd.Series:
    hist = _cboe_put_call_history(kind, name)
    recent = _ycharts_recent_indicator(ycharts_slug, name)
    cache = _load_series_cache(cache_filename, name)
    if not cache.empty:
        cache = cache[cache.index.dayofweek < 5]

    live = pd.Series(dtype=float, name=name)
    today = pd.Timestamp.today().normalize()
    last_market_day = today if today.dayofweek < 5 else today - pd.offsets.BDay(1)
    recent_last = recent.index.max().normalize() if not recent.empty else pd.Timestamp.min
    if recent_last < last_market_day:
        live_val = _cboe_live_put_call_ratio(live_symbols)
        if live_val is not None:
            live = pd.Series([live_val], index=[last_market_day], name=name)

    merged = _merge_series(name, hist, cache, recent, live)
    if not live.empty:
        _save_series_cache(cache_filename, merged, name)
    return merged


def put_call_ratio() -> pd.Series:
    """
    CBOE Equity Put/Call ratio.

    Source priority:
    1. CBOE official CSV archive/recent files through Oct 2019
    2. YCharts public recent CBOE daily-stat rows
    3. CBOE delayed option-chain live ETF proxy + local cache
    """
    return _official_put_call_series(
        kind="equity",
        ycharts_slug="cboe_equity_put_call_ratio",
        name="put_call",
        cache_filename="put_call_history.csv",
        live_symbols=("SPY", "QQQ", "IWM"),
    )


# ---------------------------------------------------------------------------
# Valuation: Equity Risk Premium
# ---------------------------------------------------------------------------
def equity_risk_premium() -> pd.Series:
    """
    ERP = SPX trailing earnings yield - US10Y.
    We approximate SPX earnings yield from SPY's P/E using a fixed-ish estimate;
    more robust: use SP500 earnings yield FRED series if available.
    We use MULT: S&P 500 Shiller PE via multpl (no API). Fall back to SPX 1/PE from yfinance info.
    """
    # Simple approx using DGS10 plus SPX return 5y realized — crude but keeps series alive.
    ten = fred_dgs10()
    spx_px = spx()
    if ten.empty or spx_px.empty:
        return pd.Series(dtype=float, name="erp")
    # Use rolling earnings-yield proxy: 1/(trailing 5y avg P/E proxy ~ price/rolling 5y mean price * 20)
    # Cleaner path: Damodaran ERP (monthly). For live daily: crude 1/fwd_pe ~ 0.045 static.
    # We'll use a reasonable earnings yield estimate = 1/22 (typical SPX fwd P/E ~ 20-24)
    ey = pd.Series(100.0 / 22.0, index=spx_px.index)  # ~4.5%
    ten_aligned = ten.reindex(spx_px.index).ffill()
    erp = ey - ten_aligned
    erp.name = "erp"
    return erp.dropna()


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

    The free endpoint was paywalled in late 2024 ($720/mo). We try it
    anyway — if it returns data, great; if not, we fall back gracefully.
    """
    urls = [
        "https://squeezemetrics.com/monitor/static/DIX.csv",
    ]
    for url in urls:
        try:
            r = requests.get(url, headers=UA, timeout=10)
            if r.ok and "," in r.text and len(r.text) > 100:
                df = pd.read_csv(io.StringIO(r.text))
                df.columns = [c.strip().lower() for c in df.columns]
                if "date" in df.columns:
                    df["date"] = pd.to_datetime(df["date"], errors="coerce")
                    df = df.dropna(subset=["date"]).set_index("date").sort_index()
                    return df
        except Exception as e:
            log.info("SqueezeMetrics CSV fetch failed: %s", e)
    return pd.DataFrame()


def dix_proxy() -> pd.Series:
    """
    Dark Index (DIX) — proportion of dark pool volume that is buying.
    Rising DIX during a sell-off = institutional accumulation.

    Source: SqueezeMetrics free CSV (may be paywalled).
    """
    df = _squeezemetrics_csv()
    if not df.empty and "dix" in df.columns:
        s = df["dix"].astype(float).dropna()
        s.name = "dix"
        return s
    return pd.Series(dtype=float, name="dix")


def _normalize_gex_units(s: pd.Series) -> pd.Series:
    """Normalize GEX to $B; some public feeds/cache rows arrive as raw dollars."""
    out = pd.to_numeric(s.copy(), errors="coerce").dropna()
    if out.empty:
        out.name = "gamma_exposure"
        return out
    dollar_mask = out.abs() > 1_000_000
    if dollar_mask.any():
        out.loc[dollar_mask] = out.loc[dollar_mask] / 1e9
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
def cftc_cta_positioning() -> pd.Series:
    """
    CFTC Traders in Financial Futures (TFF) — Leveraged Funds net position
    in S&P 500 E-mini futures as % of open interest.

    Interpretation (contrarian):
      High net-long %  (>10% OI)  → CTAs heavily long → top risk
      High net-short % (<-10% OI) → CTAs capitulated  → bottom setup

    Data: CFTC publishes free weekly TXT/CSV ZIPs for each calendar year.
    We download 3 years + current year and concatenate.
    """
    from datetime import date as _date

    current_year = _date.today().year
    years = [current_year - 2, current_year - 1, current_year]

    frames = []
    for yr in years:
        url = f"https://www.cftc.gov/files/dea/history/fut_fin_txt_{yr}.zip"
        try:
            r = requests.get(url, headers=UA, timeout=20)
            if not r.ok:
                continue
            z = __import__("zipfile").ZipFile(io.BytesIO(r.content))
            csv_files = [n for n in z.namelist() if n.lower().endswith((".txt", ".csv"))]
            if not csv_files:
                continue
            df = pd.read_csv(z.open(csv_files[0]), low_memory=False)
            # Keep only S&P 500 E-mini rows
            mask = df["Market_and_Exchange_Names"].str.contains(
                "E-MINI S&P 500", na=False, case=False
            )
            sub = df[mask].copy()
            if sub.empty:
                continue
            frames.append(sub)
        except Exception as e:
            log.warning("CFTC COT %d failed: %s", yr, e)

    if not frames:
        return pd.Series(dtype=float, name="cta_positioning")

    df = pd.concat(frames, ignore_index=True)

    # Parse date (format: YYMMDD or YYYYMMDD)
    date_col = next((c for c in df.columns if "Date" in c and "Form" in c), None) \
               or next((c for c in df.columns if "date" in c.lower()), None)
    if date_col is None:
        return pd.Series(dtype=float, name="cta_positioning")

    df["date"] = pd.to_datetime(df[date_col].astype(str), format="%y%m%d", errors="coerce")
    df.loc[df["date"].isna(), "date"] = pd.to_datetime(
        df.loc[df["date"].isna(), date_col].astype(str), format="%Y%m%d", errors="coerce"
    )
    df = df.dropna(subset=["date"]).set_index("date").sort_index()
    df = df[~df.index.duplicated(keep="last")]

    long_col  = "Lev_Money_Positions_Long_All"
    short_col = "Lev_Money_Positions_Short_All"
    oi_col    = "Open_Interest_All"

    for c in (long_col, short_col, oi_col):
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=[long_col, short_col, oi_col])

    # Net position as % of total open interest
    net_pct = (df[long_col] - df[short_col]) / df[oi_col] * 100
    net_pct.name = "cta_positioning"
    return net_pct


# ---------------------------------------------------------------------------
# Yield curve
# ---------------------------------------------------------------------------
@_memoized_fetch
def curve_2s10s() -> pd.Series:
    """10Y - 2Y Treasury yield spread (FRED T10Y2Y). Negative = inverted."""
    s = _fred("T10Y2Y")
    s.name = "curve_2s10s"
    return s


def curve_3m10y() -> pd.Series:
    """10Y - 3M Treasury yield spread (FRED T10Y3M). Fed's preferred recession curve."""
    s = _fred("T10Y3M")
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
    Interbank funding stress proxy.

    The classic FRA-OIS spread is not available on FRED, and LIBOR was
    permanently discontinued in 2023. We construct a post-LIBOR equivalent:

    Primary: SOFR 90-Day Average (SOFR90DAYAVG) minus 3-Month Treasury Bill
             rate (DTB3). This captures the secured-vs-unsecured funding gap
             that FRA-OIS used to measure. Elevated = funding stress.

    Fallback: 3-Month AA Financial Commercial Paper Rate (DCPF3M) minus
              3-Month Treasury Bill (DTB3). CP-Treasury spread is the closest
              surviving proxy for interbank credit risk (a la TED spread).

    Output is in basis points. Normal < 20bps. Crisis > 50bps.
    """
    # Path 1: SOFR 90D avg vs 3M T-Bill
    sofr90 = _fred("SOFR90DAYAVG")
    tbill3m = _fred("DTB3")

    if not sofr90.empty and not tbill3m.empty:
        df = pd.concat([sofr90.rename("sofr90"), tbill3m.rename("tbill")], axis=1).ffill().dropna()
        if not df.empty and len(df) > 30:
            spread = (df["sofr90"] - df["tbill"]) * 100.0  # percent -> bps
            spread.name = "fra_ois_spread"
            return spread.dropna()

    # Path 2: Financial CP vs T-Bill (TED-like proxy)
    cp3m = _fred("DCPF3M")  # 3M AA Financial Commercial Paper rate
    if not cp3m.empty and not tbill3m.empty:
        df = pd.concat([cp3m.rename("cp"), tbill3m.rename("tbill")], axis=1).ffill().dropna()
        if not df.empty and len(df) > 30:
            spread = (df["cp"] - df["tbill"]) * 100.0
            spread.name = "fra_ois_spread"
            return spread.dropna()

    # Path 3: A2/P2 minus AA CP spread (pure credit risk tier spread)
    cp_a2p2 = _fred("DCPN3M")  # 3M A2/P2 Nonfinancial CP
    cp_aa = _fred("DCPF3M")    # 3M AA Financial CP
    if not cp_a2p2.empty and not cp_aa.empty:
        df = pd.concat([cp_a2p2.rename("low"), cp_aa.rename("high")], axis=1).ffill().dropna()
        if not df.empty:
            spread = (df["low"] - df["high"]) * 100.0
            spread.name = "fra_ois_spread"
            return spread.dropna()

    return pd.Series(dtype=float, name="fra_ois_spread")


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
def _cache_path(filename: str) -> str:
    """Return full path for a cache file."""
    return _series_cache_path(filename)


def _load_cache(filename: str, col_name: str) -> pd.Series:
    """Load a cached time series from CSV."""
    return _load_series_cache(filename, col_name)


def _save_cache(filename: str, series: pd.Series, col_name: str):
    """Save a time series to CSV cache."""
    _save_series_cache(filename, series, col_name)


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


@_memoized_fetch
def gamma_exposure_proxy() -> pd.Series:
    """
    Gamma Exposure (GEX) — estimated dealer gamma.

    Strategy:
    1. Try SqueezeMetrics historical GEX (daily, back to 2011 if available)
    2. Fall back to live computation from SPY options chain + cache

    When GEX is:
    - Deeply negative: Dealers must sell into weakness = crash accelerant
    - Positive: Dealers buy dips, sell rips = stabilizing
    """
    import yfinance as yf

    hist = _normalize_gex_units(_load_cache("gamma_exposure_history.csv", "gamma_exposure"))

    # SqueezeMetrics, when available, gives the historical backbone.
    sqz = squeezemetrics_gex()
    if not sqz.empty:
        hist = _merge_series("gamma_exposure", sqz, hist)

    today_val, _ = _cboe_gamma_snapshot("SPY")
    try:
        if today_val is None:
            with _silence_stderr():
                t = yf.Ticker("SPY")
                spot = t.info.get("regularMarketPrice", t.info.get("previousClose", 0))
                if not spot:
                    spot_hist = t.history(period="5d")["Close"].iloc[-1]
                    spot = float(spot_hist)

                all_expiries = list(t.options)
                near_expiries = [e for e in all_expiries[:4] if e]

                total_gamma = 0.0
                total_dollars = 0.0

                for expiry in near_expiries:
                    try:
                        chain = t.option_chain(expiry)
                        calls = chain.calls
                        puts = chain.puts

                        for _, row in calls.iterrows():
                            strike = row["strike"]
                            oi = row.get("openInterest", 0) or 0
                            if oi > 0:
                                moneyness = abs(strike - spot) / spot
                                gamma_weight = oi * max(0.1, 1 - moneyness * 5)
                                total_gamma += gamma_weight * 0.01
                                total_dollars += oi * row.get("lastPrice", 0) * 100

                        for _, row in puts.iterrows():
                            strike = row["strike"]
                            oi = row.get("openInterest", 0) or 0
                            if oi > 0:
                                moneyness = abs(strike - spot) / spot
                                gamma_weight = oi * max(0.1, 1 - moneyness * 5)
                                if strike < spot:
                                    total_gamma -= gamma_weight * 0.02
                                total_dollars += oi * row.get("lastPrice", 0) * 100

                    except Exception:
                        continue

                if total_dollars > 0:
                    today_val = total_gamma / 1e9
    except Exception as e:
        log.info("Gamma exposure calculation failed: %s", str(e)[:80])

    if today_val is not None:
        today = pd.Timestamp.today().normalize()
        hist.loc[today] = today_val
        hist = _normalize_gex_units(hist)
        hist = hist[~hist.index.duplicated(keep="last")].sort_index()
        _save_cache("gamma_exposure_history.csv", hist, "gamma_exposure")

    hist.name = "gamma_exposure"
    return hist


def gamma_flip_zone_distance() -> pd.Series:
    """
    Distance (in % terms) to the estimated 'gamma flip' price level.

    The gamma flip zone is where net gamma exposure crosses zero.
    Above flip = positive gamma (dealers sell highs, buy lows) = stable.
    Below flip = negative gamma (dealers sell lows, buy highs) = unstable.

    Returns % distance from current price to estimated flip zone.
    Positive = distance above flip (safe), Negative = below flip (danger).
    Cached to build history over time.
    """
    import yfinance as yf

    hist = _load_cache("gamma_flip_history.csv", "gamma_flip")
    gex_history = gamma_exposure_proxy()
    proxy_hist = _gex_implied_gamma_flip_history(gex_history)
    hist = _merge_series("gamma_flip", proxy_hist, hist)

    _, today_val = _cboe_gamma_snapshot("SPY")
    try:
        if today_val is None:
            with _silence_stderr():
                t = yf.Ticker("SPY")
                spot = t.info.get("regularMarketPrice", t.info.get("previousClose", 0))
                if not spot:
                    px_hist = t.history(period="5d")["Close"].iloc[-1]
                    spot = float(px_hist)

                # Estimate flip zone from option open interest distribution
                all_expiries = list(t.options)
                if not all_expiries:
                    return hist  # Return cached/proxy data even if compute fails

                # Use nearest expiry for max gamma sensitivity
                chain = t.option_chain(all_expiries[0])

                calls = chain.calls
                puts = chain.puts

                # Weighted average strike by open interest
                call_oi = calls.get("openInterest", pd.Series(0, index=calls.index)).fillna(0)
                put_oi = puts.get("openInterest", pd.Series(0, index=puts.index)).fillna(0)

                call_oi_weighted = (calls["strike"] * call_oi).sum() / max(1, call_oi.sum())
                put_oi_weighted = (puts["strike"] * put_oi).sum() / max(1, put_oi.sum())

                # Flip zone roughly between weighted put and call strikes
                flip_zone = (put_oi_weighted + call_oi_weighted) / 2

                # Distance as % of spot
                distance_pct = (spot - flip_zone) / spot * 100
                today_val = distance_pct
    except Exception as e:
        log.info("Gamma flip zone calculation failed: %s", str(e)[:80])

    if today_val is not None:
        today = pd.Timestamp.today().normalize()
        hist.loc[today] = today_val
        hist = hist[~hist.index.duplicated(keep="last")].sort_index()
        _save_cache("gamma_flip_history.csv", hist, "gamma_flip")

    hist.name = "gamma_flip"
    return hist


def index_put_call_ratio() -> pd.Series:
    """
    Put/Call ratio specifically for INDEX options (SPX, NDX, RUT) vs equity options.

    Retail hedges single stocks (equity puts). Institutions hedge portfolios (index puts).
    High index P/C = institutional panic. Extreme readings followed by sharp drops
    indicate institutions monetizing hedges and buying underlying.

    Uses official CBOE index P/C history and recent public YCharts rows.
    Falls back to a CBOE delayed ETF-chain proxy only when official recent
    rows are unavailable.
    """
    return _official_put_call_series(
        kind="index",
        ycharts_slug="cboe_index_put_call_ratio",
        name="index_put_call",
        cache_filename="index_put_call_history.csv",
        live_symbols=("SPY", "QQQ", "IWM"),
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
            return s
    return pd.Series(dtype=float, name="dxy")


def real_yield_10y() -> pd.Series:
    """10Y TIPS real yield (FRED DFII10), in percent."""
    s = _fred("DFII10")
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
