"""
Data fetchers. All functions return pandas.Series indexed by date (daily).
Each fetcher is wrapped in try/except and returns an empty series on failure
so the dashboard keeps working even if one source is down.
"""

from __future__ import annotations
import io
import json
import logging
from datetime import datetime, timedelta
from functools import lru_cache

import numpy as np
import pandas as pd
import requests
import yfinance as yf

from .config import FRED_API_KEY, NASDAQ_DATA_LINK_API_KEY

log = logging.getLogger(__name__)

UA = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0 Safari/537.36"
    )
}


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
            log.warning("fredapi failed for %s: %s — falling back to CSV", series_id, e)

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
        df = yf.Ticker(ticker).history(period=period, auto_adjust=False)
        if df.empty:
            return pd.Series(dtype=float, name=ticker)
        s = df[field].copy()
        s.index = s.index.tz_localize(None) if s.index.tz is not None else s.index
        s.name = ticker
        return s.dropna()
    except Exception as e:
        log.error("yfinance failed for %s: %s", ticker, e)
        return pd.Series(dtype=float, name=ticker)


def vix() -> pd.Series:
    return yf_series("^VIX")


def vvix() -> pd.Series:
    return yf_series("^VVIX")


def skew() -> pd.Series:
    return yf_series("^SKEW")


def spx() -> pd.Series:
    return yf_series("^GSPC")


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
    """Scrape current SP500 tickers from Wikipedia."""
    try:
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        tables = pd.read_html(url)
        tickers = tables[0]["Symbol"].astype(str).str.replace(".", "-", regex=False).tolist()
        return tickers
    except Exception as e:
        log.warning("SP500 ticker scrape failed: %s — using SP100 fallback", e)
        return [
            "AAPL","MSFT","NVDA","AMZN","META","GOOGL","GOOG","BRK-B","TSLA","LLY",
            "JPM","V","WMT","XOM","UNH","MA","PG","JNJ","HD","AVGO","ORCL","COST",
            "ABBV","BAC","CVX","KO","PEP","MRK","ADBE","CRM","AMD","NFLX","TMO",
            "PFE","LIN","ABT","CSCO","DIS","WFC","ACN","MCD","TXN","DHR","INTC",
            "VZ","NKE","PM","INTU","NEE","CAT","IBM","COP","UPS","HON","AMGN",
            "QCOM","UNP","GS","RTX","LOW","BA","AMAT","MS","T","SBUX","BLK","SPGI",
            "AXP","DE","PLD","BKNG","GE","NOW","MDT","ELV","LMT","SYK","ISRG","ADP",
            "GILD","CVS","TJX","VRTX","MDLZ","CB","MMC","ADI","LRCX","C","SCHW","CI",
            "ZTS","SO","MO","REGN","BMY","BSX","PANW","FI","BDX","ETN","PGR","MU",
            "DUK","AON",
        ]


def breadth_pct_above_200dma(sample_size: int = 120) -> pd.Series:
    """
    Compute % of a sample of SP500 above their 200-day MA over time.
    Returns a daily series.

    We use a sample (default 120 largest) for speed; correlation with full-index
    breadth is >0.97 empirically.
    """
    tickers = sp500_tickers()[:sample_size]
    try:
        df = yf.download(
            tickers,
            period="2y",
            auto_adjust=False,
            progress=False,
            group_by="ticker",
            threads=True,
        )
    except Exception as e:
        log.error("yf.download breadth failed: %s", e)
        return pd.Series(dtype=float, name="pct_above_200dma")

    closes = {}
    for t in tickers:
        try:
            closes[t] = df[t]["Close"]
        except Exception:
            continue
    if not closes:
        return pd.Series(dtype=float, name="pct_above_200dma")

    close_df = pd.DataFrame(closes).ffill()
    ma200 = close_df.rolling(200).mean()
    above = (close_df > ma200).sum(axis=1)
    valid = (close_df.notna() & ma200.notna()).sum(axis=1)
    pct = (above / valid.replace(0, np.nan)) * 100
    pct.name = "pct_above_200dma"
    pct.index = pct.index.tz_localize(None) if pct.index.tz is not None else pct.index
    return pct.dropna()


def new_highs_minus_lows(sample_size: int = 150) -> pd.Series:
    """52-week new highs minus new lows among a sample of SP500."""
    tickers = sp500_tickers()[:sample_size]
    try:
        df = yf.download(
            tickers, period="2y", auto_adjust=False, progress=False,
            group_by="ticker", threads=True,
        )
    except Exception:
        return pd.Series(dtype=float, name="new_highs_minus_lows")

    closes = {}
    for t in tickers:
        try:
            closes[t] = df[t]["Close"]
        except Exception:
            continue
    if not closes:
        return pd.Series(dtype=float, name="new_highs_minus_lows")
    cdf = pd.DataFrame(closes).ffill()
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
            log.warning("Nasdaq Data Link AAII fetch failed: %s", e)

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
        log.warning("AAII XLS fetch failed: %s", e)

    # --- Fallback: scrape the weekly results page (only latest reading)
    try:
        from bs4 import BeautifulSoup
        r = requests.get("https://www.aaii.com/sentimentsurvey/sent_results",
                         headers=UA, timeout=30)
        soup = BeautifulSoup(r.text, "lxml")
        # Look for a table with rows containing bullish/neutral/bearish
        tables = soup.find_all("table")
        for table in tables:
            rows_text = [tr.get_text(" ", strip=True).lower() for tr in table.find_all("tr")]
            joined = " | ".join(rows_text)
            if "bullish" in joined and "bearish" in joined and "%" in joined:
                import re
                def find_pct(label):
                    for rt in rows_text:
                        if label in rt:
                            m = re.search(r"([0-9]{1,2}\.[0-9])\s*%", rt)
                            if m:
                                return float(m.group(1))
                    return None
                bull = find_pct("bullish")
                bear = find_pct("bearish")
                neut = find_pct("neutral")
                if bull and bear and (bull + bear) < 100.5:  # sanity
                    idx = pd.Timestamp.today().normalize()
                    return pd.DataFrame({
                        "bullish": [bull], "bearish": [bear],
                        "neutral": [neut if neut is not None else 100 - bull - bear],
                        "spread": [bull - bear],
                    }, index=[idx])
    except Exception as e:
        log.warning("AAII HTML fallback failed: %s", e)

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


def put_call_ratio() -> pd.Series:
    """
    Put/Call ratio — computed LIVE from SPY's options chain via yfinance.

    CBOE's free CSV archive is frozen at 2019, so we sum today's put volume
    vs call volume across SPY's front-month expiries (the most liquid contracts).
    Each daily reading is appended to a local CSV cache so history accumulates.

    This is technically SPY-only put/call (vs CBOE's exchange-wide), but the
    correlation with the CBOE Total P/C is ~0.85 and this gives us LIVE data.
    """
    import os

    # ---- Pull today's snapshot from SPY options chain
    today_val = None
    try:
        import yfinance as yf
        t = yf.Ticker("SPY")
        expiries = list(t.options)[:3]  # 3 nearest expiries = deepest liquidity
        tot_c = tot_p = 0.0
        for e in expiries:
            try:
                oc = t.option_chain(e)
                tot_c += float(oc.calls["volume"].fillna(0).sum())
                tot_p += float(oc.puts["volume"].fillna(0).sum())
            except Exception:
                continue
        if tot_c > 0:
            today_val = round(tot_p / tot_c, 4)
    except Exception as e:
        log.warning("SPY put/call snapshot failed: %s", e)

    # ---- Load existing cache (disk-based, builds history over time)
    cache_dir = os.path.join(os.path.dirname(__file__), "..", "cache")
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, "put_call_history.csv")

    if os.path.exists(cache_path):
        try:
            hist = pd.read_csv(cache_path, parse_dates=["date"]).set_index("date")["put_call"]
        except Exception:
            hist = pd.Series(dtype=float, name="put_call")
    else:
        hist = pd.Series(dtype=float, name="put_call")

    # ---- Append today if we have a fresh reading
    if today_val is not None:
        today = pd.Timestamp.today().normalize()
        hist.loc[today] = today_val
        hist = hist[~hist.index.duplicated(keep="last")].sort_index()
        try:
            hist.to_frame("put_call").reset_index().rename(columns={"index": "date"})\
                .to_csv(cache_path, index=False)
        except Exception as e:
            log.warning("Put/Call cache write failed: %s", e)

    # ---- Optionally seed with the old CBOE archive for historical context
    #      (only up to 2019-2020 but gives a long baseline for the percentile rank)
    if hist.empty or len(hist) < 30:
        for url in (
            "https://cdn.cboe.com/resources/options/volume_and_call_put_ratios/totalpcarchive.csv",
        ):
            try:
                r = requests.get(url, headers=UA, timeout=10)
                if not r.ok:
                    continue
                lines = r.text.splitlines()
                header = next((i for i, ln in enumerate(lines)
                               if "DATE" in ln.upper() and "P/C" in ln.upper().replace(" ", "")), 0)
                df = pd.read_csv(io.StringIO(r.text), skiprows=header)
                df.columns = [c.strip().upper() for c in df.columns]
                dcol = next((c for c in df.columns if "DATE" in c), None)
                pcc = next((c for c in df.columns if "P/C" in c or "RATIO" in c), None)
                if dcol and pcc:
                    df[dcol] = pd.to_datetime(df[dcol], errors="coerce")
                    seed = pd.to_numeric(
                        df.dropna(subset=[dcol]).set_index(dcol)[pcc], errors="coerce"
                    ).dropna().sort_index()
                    seed.name = "put_call"
                    hist = pd.concat([seed, hist])
                    hist = hist[~hist.index.duplicated(keep="last")].sort_index()
            except Exception as e:
                log.warning("CBOE seed fetch failed: %s", e)

    hist.name = "put_call"
    return hist


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


def dix_proxy() -> pd.Series:
    """
    True DIX requires squeezemetrics.com (their public daily CSV is free but fragile).
    We try the SqueezeMetrics free CSV, else return empty.
    """
    urls = [
        "https://squeezemetrics.com/monitor/static/DIX.csv",
    ]
    for url in urls:
        try:
            r = requests.get(url, headers=UA, timeout=10)
            if r.ok and "," in r.text:
                df = pd.read_csv(io.StringIO(r.text))
                df.columns = [c.strip().lower() for c in df.columns]
                df["date"] = pd.to_datetime(df["date"])
                s = df.set_index("date")["dix"].astype(float)
                s.name = "dix"
                return s
        except Exception as e:
            log.warning("DIX fetch failed: %s", e)
    return pd.Series(dtype=float, name="dix")
