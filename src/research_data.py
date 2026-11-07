"""
Data layer for the research desk.

Every feed here is free and keyless except the FRED series, and each follows the
dashboard's chain:  live fetch -> local disk cache -> committed seed. `cache/` is
gitignored and ephemeral on Streamlit Cloud, so without the seed a fresh deploy
behind a blocked shared IP would render nothing.

Fetch failures never raise: a dead source yields an empty frame, its provenance
says so, and the panels that need it show "unavailable" instead of a stale or
synthetic number.
"""
from __future__ import annotations

import logging
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import pandas as pd
import requests
import yfinance as yf

from src import data as D
from src import research as R

log = logging.getLogger(__name__)

UA = D.UA
SOCRATA = "https://publicreporting.cftc.gov/resource/{}.json"
CBOE_DAILY = "https://cdn.cboe.com/data/us/options/market_statistics/daily/{}_daily_options"
TFF = "gpe5-46if"     # Traders in Financial Futures, futures only
DISAGG = "72hh-3qpy"  # Disaggregated, futures only

# group: which trader category the market's crowding is measured on.
COT_MARKETS: list[dict] = [
    {"key": "ES", "label": "S&P 500 e-mini", "dataset": TFF, "code": "13874A", "side": "lev", "group": "Leveraged funds", "price": "^GSPC"},
    {"key": "NQ", "label": "Nasdaq 100 e-mini", "dataset": TFF, "code": "209742", "side": "lev", "group": "Leveraged funds", "price": "QQQ"},
    {"key": "RTY", "label": "Russell 2000 e-mini", "dataset": TFF, "code": "239742", "side": "lev", "group": "Leveraged funds", "price": "IWM"},
    {"key": "TY", "label": "UST 10Y note", "dataset": TFF, "code": "043602", "side": "lev", "group": "Leveraged funds", "price": "IEF"},
    {"key": "VIX", "label": "VIX futures", "dataset": TFF, "code": "1170E1", "side": "am", "group": "Asset managers", "price": "^VIX"},
    {"key": "GOLD", "label": "Gold", "dataset": DISAGG, "code": "088691", "side": "mm", "group": "Managed money", "price": "GC=F"},
    {"key": "SILVER", "label": "Silver", "dataset": DISAGG, "code": "084691", "side": "mm", "group": "Managed money", "price": "SI=F"},
    {"key": "COPPER", "label": "Copper", "dataset": DISAGG, "code": "085692", "side": "mm", "group": "Managed money", "price": "HG=F"},
    {"key": "GASOLINE", "label": "RBOB gasoline", "dataset": DISAGG, "code": "111659", "side": "mm", "group": "Managed money", "price": "RB=F"},
    {"key": "WTI", "label": "WTI crude", "dataset": DISAGG, "code": "067651", "side": "mm", "group": "Managed money", "price": "CL=F"},
]
_COT_SIDES = {
    "lev": ("lev_money_positions_long", "lev_money_positions_short"),
    "am": ("asset_mgr_positions_long", "asset_mgr_positions_short"),
    "mm": ("m_money_positions_long_all", "m_money_positions_short_all"),
}

AI_GROUPS: dict[str, list[str]] = {
    "AI chips": ["NVDA", "AMD", "AVGO", "ARM", "INTC"],
    "Memory & storage": ["MU", "WDC", "STX", "SNDK"],
    "Semi equipment": ["ASML", "AMAT", "LRCX", "KLAC"],
    "Neocloud & compute": ["CRWV", "NBIS", "IREN", "APLD"],
    "Optical & networking": ["ANET", "ALAB", "COHR", "LITE", "CIEN", "NOK"],
}
AI_BASKET: list[str] = [t for g in AI_GROUPS.values() for t in g]
HYPERSCALERS: list[str] = ["GOOGL", "MSFT", "AMZN", "META"]

INDEX_TICKERS = ["^GSPC", "^SOX", "^VIX", "^VXN", "^SKEW", "^MOVE", "^TNX", "^TYX", "^KS11"]
ETF_TICKERS = [
    "SPY", "QQQ", "IWM", "RSP", "SMH", "IGV", "MTUM", "QUAL", "USMV", "GVIP", "XLK", "XLV", "XLF", "XLU",
    "XLE", "XLI", "XLB", "XLP", "XLY", "XLC", "XLRE", "ITB", "XHB", "JETS", "GDX", "HYG", "LQD", "TLT",
    "IEF", "EEM", "UUP", "CPER", "DBC", "USO", "GLD",
]
FUTURE_TICKERS = ["GC=F", "SI=F", "HG=F", "CL=F", "RB=F", "BTC-USD"]
RESEARCH_TICKERS = INDEX_TICKERS + ETF_TICKERS + FUTURE_TICKERS + AI_BASKET + HYPERSCALERS
PCA_UNIVERSE = ["SPY", "QQQ", "IWM", "EEM", "XLE", "XLF", "HYG", "LQD", "IEF", "TLT", "UUP", "CPER", "USO", "GLD"]

SECTOR_NAMES = {
    "XLK": "Technology", "XLC": "Communication", "XLY": "Discretionary", "XLF": "Financials", "XLV": "Health care",
    "XLI": "Industrials", "XLB": "Materials", "XLP": "Staples", "XLE": "Energy", "XLU": "Utilities", "XLRE": "Real estate",
}
ROTATION_NAMES = {
    "ITB": "Homebuilders", "XLV": "Health care", "TLT": "Long Treasuries", "XLU": "Utilities", "XLF": "Financials",
    "RSP": "Equal-weight S&P", "IWM": "Small caps", "JETS": "Airlines", "GDX": "Gold miners", "IGV": "Software",
    "SPY": "S&P 500", "QQQ": "Nasdaq 100", "XLK": "Technology", "SMH": "Semiconductors",
}
TECH_BOARD = {
    "SPY": "S&P 500", "QQQ": "Nasdaq 100", "IWM": "Russell 2000", "RSP": "Equal-weight S&P", "^SOX": "Semis (SOX)",
    "IGV": "Software", "^KS11": "KOSPI", "GC=F": "Gold", "SI=F": "Silver", "GDX": "Gold miners", "BTC-USD": "Bitcoin",
    "TLT": "Long Treasuries", "^TNX": "10Y yield", "CL=F": "WTI crude",
}


# ---------------------------------------------------------------------------
# Chain helper: cell-wise merge for frames that may have partial live coverage
# ---------------------------------------------------------------------------
def _cellwise_chain(*, key: str, live: pd.DataFrame, cache_file: str, seed_file: str, source: str) -> pd.DataFrame:
    """live -> cache -> seed where a missing live CELL falls back, not the whole row.

    The shared `_with_fallback_chain` lets the newest frame win per date, which
    is right for one-source series but would blank a ticker's history whenever
    Yahoo drops just that ticker from a bulk download.
    """
    cache = D._load_frame_cache(cache_file)
    seed = D._load_seed_frame(seed_file)
    older = cache.combine_first(seed) if not cache.empty else seed
    if live is not None and not live.empty:
        merged = live.combine_first(older) if not older.empty else live
        merged = merged.sort_index()
        merged.index.name = "date"
        D._save_frame_cache(cache_file, merged)
        D.record_provenance(key, source, kind="primary")
        return merged
    if not cache.empty:
        D.record_provenance(key, source, kind="cache", note=f"live fetch failed; cached through {cache.index.max():%Y-%m-%d}")
        return older.sort_index()
    if not seed.empty:
        D.record_provenance(key, "bundled_seed", kind="fallback", note=f"live source unreachable; seed through {seed.index.max():%Y-%m-%d}")
        return seed.sort_index()
    D.record_provenance(key, "", kind="unavailable")
    return pd.DataFrame()


# ---------------------------------------------------------------------------
# Prices
# ---------------------------------------------------------------------------
def fetch_prices_live(tickers: list[str] | None = None, start: str = "1990-01-01") -> pd.DataFrame:
    tickers = tickers or RESEARCH_TICKERS
    try:
        with D._silence_stderr():
            df = yf.download(tickers, start=start, auto_adjust=True, progress=False, threads=True)
    except Exception as e:
        log.warning("research price download failed: %s", str(e)[:120])
        return pd.DataFrame()
    if df is None or df.empty:
        return pd.DataFrame()
    close = df["Close"] if isinstance(df.columns, pd.MultiIndex) else df
    close = close.dropna(how="all", axis=1).dropna(how="all", axis=0)
    if getattr(close.index, "tz", None) is not None:
        close.index = close.index.tz_localize(None)
    close.index.name = "date"
    close.columns.name = None
    return close


def on_us_calendar(prices: pd.DataFrame) -> pd.DataFrame:
    """Keep US equity sessions only.

    Bitcoin prints on weekends and Korea on US holidays. Those rows are blank for
    every US asset, so left in the shared frame they would erase the next
    session's return in `pct_change(fill_method=None)` and add fake zero-return
    days to correlation windows.
    """
    if prices is None or prices.empty:
        return prices
    keys = [c for c in ("^GSPC", "SPY") if c in prices.columns]
    if not keys:
        return prices
    return prices[prices[keys].notna().any(axis=1)]


def research_prices() -> pd.DataFrame:
    return _cellwise_chain(
        key="research_prices", live=fetch_prices_live(), cache_file="research_prices_history.csv",
        seed_file="research_prices_seed.csv", source="yfinance",
    )


def spx_long_history() -> pd.Series:
    """S&P 500 month-end closes since 1950 for the seasonality table."""
    s = D.yf_series("^GSPC", period="max")
    live = pd.DataFrame()
    if not s.empty:
        m = s.loc["1950-01-01":].resample("ME").last()
        live = m.rename("close").to_frame()
        live.index.name = "date"
    df = _cellwise_chain(key="spx_monthly", live=live, cache_file="spx_monthly_history.csv",
                         seed_file="spx_monthly_seed.csv", source="yfinance:^GSPC")
    if df.empty:
        return pd.Series(dtype=float)
    return df["close"].dropna()


# ---------------------------------------------------------------------------
# CFTC positioning, net as % of open interest
# ---------------------------------------------------------------------------
def _fetch_cot_dataset(dataset: str, markets: list[dict]) -> pd.DataFrame:
    frames = {}
    for m in markets:
        long_col, short_col = _COT_SIDES[m["side"]]
        params = {
            "$select": f"report_date_as_yyyy_mm_dd as d,{long_col} as l,{short_col} as s,open_interest_all as oi",
            "$where": f"cftc_contract_market_code='{m['code']}'",
            "$order": "d", "$limit": "6000",
        }
        r = requests.get(SOCRATA.format(dataset), params=params, headers=UA, timeout=30)
        r.raise_for_status()
        df = pd.DataFrame(r.json())
        if df.empty:
            continue
        df["d"] = pd.to_datetime(df["d"], errors="coerce")
        df = df.dropna(subset=["d"]).set_index("d").apply(pd.to_numeric, errors="coerce")
        df = df[~df.index.duplicated(keep="last")].sort_index()
        frames[m["key"]] = (df["l"] - df["s"]) / df["oi"] * 100
    return pd.DataFrame(frames)


def fetch_cot_live() -> pd.DataFrame:
    try:
        parts = []
        for ds in (TFF, DISAGG):
            parts.append(_fetch_cot_dataset(ds, [m for m in COT_MARKETS if m["dataset"] == ds]))
        out = pd.concat(parts, axis=1).sort_index()
        out.index.name = "date"
        return out
    except Exception as e:
        log.warning("CFTC Socrata fetch failed: %s", str(e)[:120])
        return pd.DataFrame()


def cot_positioning() -> pd.DataFrame:
    return _cellwise_chain(key="cot_positioning", live=fetch_cot_live(), cache_file="cot_positioning_history.csv",
                           seed_file="cot_positioning_seed.csv", source="cftc:publicreporting (TFF + Disaggregated)")


# ---------------------------------------------------------------------------
# Cboe daily option volumes
# ---------------------------------------------------------------------------
_CBOE_PRODUCTS = {
    "total": "SUM OF ALL PRODUCTS", "index": "INDEX OPTIONS", "equity": "EQUITY OPTIONS",
    "vix": "CBOE VOLATILITY INDEX (VIX)", "spx": "SPX + SPXW",
}


def parse_cboe_daily(payload: dict) -> dict:
    """Pick call/put volumes per product from one Cboe daily-statistics JSON."""
    out: dict[str, float] = {}
    for short, name in _CBOE_PRODUCTS.items():
        rows = payload.get(name) or []
        vol = next((r for r in rows if str(r.get("name", "")).upper() == "VOLUME"), None)
        if vol:
            out[f"{short}_call"] = float(vol.get("call") or 0)
            out[f"{short}_put"] = float(vol.get("put") or 0)
    return out


def fetch_cboe_day(day: pd.Timestamp) -> dict | None:
    """One day's volumes, or None for holidays / not-yet-published dates."""
    url = CBOE_DAILY.format(f"{day:%Y-%m-%d}")
    try:
        r = requests.get(url, headers=UA, timeout=15)
        if r.status_code != 200:
            return None
        row = parse_cboe_daily(r.json())
        return row or None
    except Exception as e:
        log.info("Cboe daily %s unavailable: %s", day.date(), str(e)[:80])
        return None


def fetch_cboe_range(start: pd.Timestamp, end: pd.Timestamp, workers: int = 8) -> pd.DataFrame:
    days = list(pd.bdate_range(start, end))
    if not days:
        return pd.DataFrame()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        rows = list(ex.map(fetch_cboe_day, days))
    data = {d: r for d, r in zip(days, rows) if r}
    if not data:
        return pd.DataFrame()
    out = pd.DataFrame.from_dict(data, orient="index").sort_index()
    out.index.name = "date"
    return out


@D._memoized_fetch
def cboe_volumes(max_backfill: int = 25) -> pd.DataFrame:
    """Daily volumes: only the days the cache and seed don't already hold are fetched."""
    cache = D._load_frame_cache("cboe_volume_history.csv")
    seed = D._load_seed_frame("cboe_volume_seed.csv")
    known = cache.combine_first(seed) if not cache.empty else seed
    end = pd.Timestamp.today().normalize()
    start = end - pd.tseries.offsets.BDay(max_backfill)
    if not known.empty:
        start = max(start, known.index.max() + pd.Timedelta(days=1))
    live = fetch_cboe_range(start, end)
    # Nothing new is normal on weekends and before tonight's file is posted, so
    # don't report it as a failed fetch.
    if live.empty and not known.empty and known.index.max() >= D.last_completed_trading_day() - pd.tseries.offsets.BDay(1):
        D.record_provenance("cboe_volumes", "cboe:daily market statistics", kind="primary",
                            note=f"up to date through {known.index.max():%Y-%m-%d}")
        return known.sort_index()
    return _cellwise_chain(key="cboe_volumes", live=live, cache_file="cboe_volume_history.csv",
                           seed_file="cboe_volume_seed.csv", source="cboe:daily market statistics")


# ---------------------------------------------------------------------------
# Short interest (Nasdaq-listed only: QQQ yes, NYSE Arca SPY no)
# ---------------------------------------------------------------------------
def parse_nasdaq_short_interest(payload: dict) -> pd.Series:
    rows = ((payload or {}).get("data") or {}).get("shortInterestTable", {}) or {}
    rows = rows.get("rows") or []
    data = {}
    for r in rows:
        d = pd.to_datetime(r.get("settlementDate"), format="%m/%d/%Y", errors="coerce")
        v = pd.to_numeric(str(r.get("interest", "")).replace(",", ""), errors="coerce")
        if pd.notna(d) and pd.notna(v):
            data[d] = float(v)
    return pd.Series(data, dtype=float).sort_index()


def qqq_short_interest() -> pd.Series:
    live = pd.DataFrame()
    try:
        r = requests.get("https://api.nasdaq.com/api/quote/QQQ/short-interest?assetClass=etf", headers=UA, timeout=20)
        r.raise_for_status()
        s = parse_nasdaq_short_interest(r.json())
        if not s.empty:
            live = (s / 1e6).rename("qqq_si_m").to_frame()
            live.index.name = "date"
    except Exception as e:
        log.info("Nasdaq short interest unavailable: %s", str(e)[:100])
    df = _cellwise_chain(key="qqq_short_interest", live=live, cache_file="qqq_short_interest_history.csv",
                         seed_file="qqq_short_interest_seed.csv", source="nasdaq:short-interest")
    return df["qqq_si_m"].dropna() if not df.empty else pd.Series(dtype=float)


# ---------------------------------------------------------------------------
# Credit: CCC & lower OAS
# ---------------------------------------------------------------------------
def ccc_oas() -> pd.Series:
    """ICE BofA CCC & lower OAS, in percent. FRED now serves only ~3 years of ICE
    data, so older history lives in the cache and seed."""
    s = D._fred("BAMLH0A3HYC")
    live = pd.DataFrame()
    if not s.empty:
        live = s.rename("ccc_oas").to_frame()
        live.index.name = "date"
    df = _cellwise_chain(key="ccc_oas", live=live, cache_file="ccc_oas_history.csv", seed_file="ccc_oas_seed.csv",
                         source="fred:BAMLH0A3HYC")
    return df["ccc_oas"].dropna() if not df.empty else pd.Series(dtype=float)


# ---------------------------------------------------------------------------
# Excess CAPE yield
# ---------------------------------------------------------------------------
def parse_multpl_table(html: str) -> pd.Series:
    """multpl.com monthly table: <td>Nov 6, 2026</td><td> 38.91 </td> rows."""
    rows = re.findall(r"<td>\s*([A-Z][a-z]{2} \d{1,2}, \d{4})\s*</td>\s*<td[^>]*>([^<]*)</td>", html)
    data = {}
    for d, v in rows:
        num = re.sub(r"[^\d.\-]", "", v.replace("&#x2002;", ""))
        ts = pd.to_datetime(d, format="%b %d, %Y", errors="coerce")
        val = pd.to_numeric(num, errors="coerce")
        if pd.notna(ts) and pd.notna(val):
            data[ts] = float(val)
    return pd.Series(data, dtype=float).sort_index()


def excess_cape_frame() -> pd.DataFrame:
    """Shiller's Excess CAPE Yield (seed), extended live with multpl CAPE,
    FRED 10Y and CPI using the same construction."""
    live = pd.DataFrame()
    try:
        r = requests.get("https://www.multpl.com/shiller-pe/table/by-month", headers=UA, timeout=20)
        r.raise_for_status()
        cape = parse_multpl_table(r.text)
        gs10, cpi = D._fred("DGS10", start="1990-01-01"), D._fred("CPIAUCSL", start="1990-01-01")
        ecy = R.excess_cape_yield(cape, gs10, cpi)
        if not ecy.empty:
            live = pd.DataFrame({"ecy": ecy, "cape": cape.resample("MS").last()}).dropna().iloc[-48:]
            live.index.name = "date"
    except Exception as e:
        log.info("Excess CAPE live extension unavailable: %s", str(e)[:100])
    return _cellwise_chain(key="excess_cape", live=live, cache_file="excess_cape_history.csv",
                           seed_file="excess_cape_seed.csv", source="multpl + fred (extension of Shiller)")


# ---------------------------------------------------------------------------
# S&P 500 forward P/E (consensus, daily since 2026-03)
# ---------------------------------------------------------------------------
def sp500_forward_pe() -> pd.Series:
    live = pd.DataFrame()
    try:
        r = requests.get("https://www.vcpscanner.com/market-valuation/sp-500", headers=UA, timeout=25)
        r.raise_for_status()
        df = D._parse_vcp_history(r.text)
        if df is not None and not df.empty and "forward_pe" in df.columns:
            live = df[["forward_pe"]].dropna()
            live.index.name = "date"
    except Exception as e:
        log.info("VCP S&P P/E unavailable: %s", str(e)[:100])
    out = _cellwise_chain(key="sp500_forward_pe", live=live, cache_file="sp500_forward_pe_history.csv",
                          seed_file="sp500_forward_pe_seed.csv", source="vcpscanner.com")
    return out["forward_pe"].dropna() if not out.empty else pd.Series(dtype=float)


# ---------------------------------------------------------------------------
# CNN Fear & Greed components
# ---------------------------------------------------------------------------
_FNG_PARTS = {
    "market_momentum_sp500": "Market momentum", "stock_price_strength": "Price strength",
    "stock_price_breadth": "Price breadth", "put_call_options": "Put/call", "market_volatility_vix": "Volatility",
    "junk_bond_demand": "Junk bond demand", "safe_haven_demand": "Safe-haven demand",
}


def parse_fng_components(payload: dict) -> list[dict]:
    out = []
    for key, label in _FNG_PARTS.items():
        part = (payload or {}).get(key) or {}
        if "score" in part:
            out.append({"name": label, "score": float(part["score"]), "rating": str(part.get("rating", "")).lower()})
    return out


def fear_greed_components() -> list[dict]:
    try:
        r = requests.get("https://production.dataviz.cnn.io/index/fearandgreed/graphdata",
                         headers={**UA, "Referer": "https://www.cnn.com/markets/fear-and-greed"}, timeout=20)
        r.raise_for_status()
        return parse_fng_components(r.json())
    except Exception as e:
        log.info("CNN components unavailable: %s", str(e)[:100])
        return []


# ---------------------------------------------------------------------------
# S&P internals: advancers, participation, realized correlation
# ---------------------------------------------------------------------------
def sp500_internals() -> pd.DataFrame:
    panel, wanted = D._panel_columns()
    live = pd.DataFrame()
    if wanted:
        p = panel[wanted]
        live = R.internals_frame(p)
        for w in (60, 126):
            live[f"corr_{w}"] = R.avg_pairwise_corr(p, w)
        live.index.name = "date"
    return _cellwise_chain(key="sp500_internals", live=live, cache_file="sp500_internals_history.csv",
                           seed_file="sp500_internals_seed.csv", source=f"sp500_panel({len(wanted)} constituents)")


# ---------------------------------------------------------------------------
# Bundle
# ---------------------------------------------------------------------------
@dataclass
class ResearchBundle:
    prices: pd.DataFrame
    spx_monthly: pd.Series
    cot: pd.DataFrame
    cboe: pd.DataFrame
    qqq_si: pd.Series
    ccc: pd.Series
    cape: pd.DataFrame
    sp_fwd_pe: pd.Series
    internals: pd.DataFrame
    btc: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))  # 7-day calendar
    fng_parts: list[dict] = field(default_factory=list)
    provenance: dict = field(default_factory=dict)


def _safe(fn, empty):
    try:
        return fn()
    except Exception as e:
        log.warning("%s failed: %s", getattr(fn, "__name__", "research fetch"), str(e)[:120])
        return empty


def build_research() -> ResearchBundle:
    """Fetch every research feed. Independent feeds run concurrently; the Yahoo
    bulk download and constituent panel run first because both are rate-limited."""
    all_prices = _safe(research_prices, pd.DataFrame())
    btc = R.px(all_prices, "BTC-USD")
    prices = on_us_calendar(all_prices)
    internals = _safe(sp500_internals, pd.DataFrame())
    with ThreadPoolExecutor(max_workers=6) as ex:
        fut = {
            "spx_monthly": ex.submit(_safe, spx_long_history, pd.Series(dtype=float)),
            "cot": ex.submit(_safe, cot_positioning, pd.DataFrame()),
            "cboe": ex.submit(_safe, cboe_volumes, pd.DataFrame()),
            "qqq_si": ex.submit(_safe, qqq_short_interest, pd.Series(dtype=float)),
            "ccc": ex.submit(_safe, ccc_oas, pd.Series(dtype=float)),
            "cape": ex.submit(_safe, excess_cape_frame, pd.DataFrame()),
            "sp_fwd_pe": ex.submit(_safe, sp500_forward_pe, pd.Series(dtype=float)),
            "fng_parts": ex.submit(_safe, fear_greed_components, []),
        }
        got = {k: f.result() for k, f in fut.items()}
    return ResearchBundle(prices=prices, internals=internals, btc=btc, provenance=D.get_provenance(), **got)


