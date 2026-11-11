"""
Pure analytics for the research desk: positioning crowding, realized correlation,
cross-asset growth pricing, technical / rotation boards, event studies and the
decision rules (capitulation checklist, system guardrails, leverage gate,
rotation stage).

No Streamlit and no network: every function takes pandas data and returns pandas
data or plain dicts, so the logic is deterministic and unit-testable offline.

Two conventions are enforced throughout because the desk's job is to support a
decision, not to tell a story:

* No look-ahead. Rolling/expanding statistics only use data up to the signal
  date, and weekly COT flags are lagged to their Friday publication.
* Every historical claim carries its base rate. An event study reports the
  unconditional hit rate over the same span next to the signal's, because the
  S&P 500 is up over most horizons anyway: "up 80% of the time" is only
  interesting when the base rate is well below 80%.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

TRADING_DAYS = 252


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def clean(s: pd.Series | None) -> pd.Series:
    """Numeric, sorted, tz-naive, de-duplicated series (empty-safe)."""
    if s is None or len(s) == 0:
        return pd.Series(dtype=float)
    out = pd.to_numeric(s, errors="coerce").dropna()
    if out.empty:
        return pd.Series(dtype=float)
    out.index = pd.to_datetime(out.index)
    if getattr(out.index, "tz", None) is not None:
        out.index = out.index.tz_localize(None)
    out = out[~out.index.duplicated(keep="last")].sort_index()
    return out


def px(prices: pd.DataFrame | None, ticker: str) -> pd.Series:
    """One cleaned price column, empty if the ticker is absent."""
    if prices is None or prices.empty or ticker not in prices.columns:
        return pd.Series(dtype=float, name=ticker)
    return clean(prices[ticker]).rename(ticker)


def pct_rank(s: pd.Series, value: float | None = None) -> float:
    """Share of observations <= value (latest by default), 0-100."""
    s = clean(s)
    if s.empty:
        return float("nan")
    v = float(s.iloc[-1]) if value is None else float(value)
    return float((s <= v).mean() * 100.0)


def ordinal(p: float) -> str:
    """80 -> '80th', 2 -> '2nd', 91.4 -> '91st'. Rounds first, so labels read naturally."""
    n = int(round(p))
    suffix = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def rolling_rank(s: pd.Series, window: int, min_periods: int | None = None) -> pd.Series:
    """Rolling percentile rank of the last point within its window, 0-100."""
    s = clean(s)
    if s.empty:
        return s
    mp = min_periods or max(20, window // 4)
    return s.rolling(window, min_periods=mp).apply(lambda a: (a <= a[-1]).mean() * 100.0, raw=True)


def expanding_rank(s: pd.Series, min_periods: int = 252) -> pd.Series:
    """Percentile rank against all history up to each date (no look-ahead)."""
    s = clean(s)
    if s.empty:
        return s
    vals = s.to_numpy(dtype=float)
    out = np.full(len(vals), np.nan)
    # Insertion into a sorted list keeps this O(n log n) over ~10k points.
    from bisect import bisect_right, insort
    seen: list[float] = []
    for i, v in enumerate(vals):
        insort(seen, v)
        if i + 1 >= min_periods:
            out[i] = bisect_right(seen, v) / (i + 1) * 100.0
    return pd.Series(out, index=s.index, name=s.name)


def rsi_wilder(s: pd.Series, n: int = 14) -> pd.Series:
    """Wilder RSI (exponential smoothing), the standard charting definition."""
    s = clean(s)
    if len(s) <= n:
        return pd.Series(dtype=float)
    d = s.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    rs = up / dn.replace(0, np.nan)
    out = 100 - 100 / (1 + rs)
    return out.where(dn != 0, 100.0).dropna()


def weekly(s: pd.Series) -> pd.Series:
    """Friday-close weekly series. The in-progress week keeps its latest print."""
    s = clean(s)
    return s if s.empty else s.resample("W-FRI").last().dropna()


def zscore_last(s: pd.Series, window: int | None = None) -> float:
    s = clean(s)
    if window:
        s = s.iloc[-window:]
    if len(s) < 20 or s.std() == 0:
        return float("nan")
    return float((s.iloc[-1] - s.mean()) / s.std())


# ---------------------------------------------------------------------------
# Positioning (CFTC)
# ---------------------------------------------------------------------------
def positioning_label(p: float) -> str:
    if np.isnan(p):
        return "NO DATA"
    if p <= 5:
        return "VERY LOW"
    if p <= 20:
        return "LOW"
    if p >= 95:
        return "VERY HIGH"
    if p >= 80:
        return "HIGH"
    return "NEUTRAL"


def positioning_snapshot(net_oi: pd.Series) -> dict:
    """Crowding read for one market's net position as % of open interest.

    Reports the percentile against ALL history and against the last 3 years.
    The two can disagree sharply and the gap is the point: leveraged funds' UST
    10Y short has been a near-record for years because it is largely the
    cash-futures basis trade, so "6th percentile all-time" can be the 40th
    percentile of the current regime. `structural` flags that case so a
    long-running structural position isn't read as a fresh crowded trade.
    """
    s = clean(net_oi)
    if s.empty:
        return {}
    cur = float(s.iloc[-1])
    all_pct = pct_rank(s)
    tail = s.iloc[-156:]
    p3 = pct_rank(tail)
    z3 = zscore_last(s, 156)
    return {
        "as_of": s.index[-1],
        "net_oi": cur,
        "pct_all": all_pct,
        "pct_3y": p3,
        "z_3y": z3,
        "flow_1w": float(s.iloc[-1] - s.iloc[-2]) if len(s) > 1 else float("nan"),
        "flow_4w": float(s.iloc[-1] - s.iloc[-5]) if len(s) > 4 else float("nan"),
        "dist_median": all_pct - 50.0,
        "state": positioning_label(all_pct),
        "state_3y": positioning_label(p3),
        "structural": abs(all_pct - p3) >= 25,
        "years": (s.index[-1] - s.index[0]).days / 365.25,
    }


def positioning_table(frame: pd.DataFrame, markets: list[dict]) -> pd.DataFrame:
    """One row per market in `markets` order, from a wide net/OI frame."""
    rows = []
    for m in markets:
        snap = positioning_snapshot(frame[m["key"]]) if m["key"] in frame.columns else {}
        if not snap:
            continue
        rows.append({"key": m["key"], "market": m["label"], "group": m["group"], **snap})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Realized correlation
# ---------------------------------------------------------------------------
def avg_pairwise_corr(
    closes: pd.DataFrame, window: int = 60, min_coverage: float = 0.8, step: int = 1
) -> pd.Series:
    """Exact equal-weighted average pairwise correlation of daily returns.

    For standardized returns z_i, Var(sum z_i) = sum over all i,j of rho_ij, so
    the mean off-diagonal rho = (Var(sum z) - N) / (N(N-1)). That gives the exact
    answer in O(T x N) per window instead of building a 500 x 500 matrix.
    """
    if closes is None or closes.empty:
        return pd.Series(dtype=float)
    rets = closes.ffill(limit=3).pct_change(fill_method=None).to_numpy(dtype=float)
    idx = closes.index
    out = np.full(len(idx), np.nan)
    for end in range(window, len(idx), step):
        w = rets[end - window + 1: end + 1]
        keep = ~np.isnan(w).any(axis=0)
        n = int(keep.sum())
        if n < 2 or n < min_coverage * closes.shape[1]:
            continue
        x = w[:, keep]
        sd = x.std(axis=0, ddof=1)
        ok = sd > 0
        if ok.sum() < 2:
            continue
        z = (x[:, ok] - x[:, ok].mean(axis=0)) / sd[ok]
        n = z.shape[1]
        out[end] = (z.sum(axis=1).var(ddof=1) - n) / (n * (n - 1))
    res = pd.Series(out, index=idx, name=f"corr_{window}d").dropna()
    if step > 1 and not res.empty:
        res = res.reindex(idx[idx >= res.index[0]]).ffill()
    return res


def rolling_corr(a: pd.Series, b: pd.Series, window: int = 20, *, diff: bool = False) -> pd.Series:
    """Rolling correlation of daily % changes (or level changes when diff=True).

    Use diff=True for yields: a bp move is the meaningful daily change.
    """
    df = pd.concat([clean(a).rename("a"), clean(b).rename("b")], axis=1, join="inner").dropna()
    if len(df) < window + 2:
        return pd.Series(dtype=float)
    ch = df.diff() if diff else df.pct_change(fill_method=None)
    return ch["a"].rolling(window).corr(ch["b"]).dropna()


# ---------------------------------------------------------------------------
# Cross-asset growth pricing (dynamic PCA)
# ---------------------------------------------------------------------------
def cross_asset_pca(
    closes: pd.DataFrame,
    *,
    anchor: str = "SPY",
    halflife: int = 60,
    horizon: int = 20,
    z_window: int = 126,
    warmup: int = 120,
) -> pd.DataFrame:
    """First principal component of cross-asset returns, read as growth pricing.

    Each day: returns are scaled by their own lagged EW volatility, projected on
    the first eigenvector of the EW correlation matrix estimated from data up to
    the PRIOR close (so there is no look-ahead), and signed so the equity anchor
    loads positively. `factor` is the trailing `horizon`-day factor return scaled
    to unit variance. `z_6m` places the CUMULATIVE factor within its own trailing
    `z_window` range: how far growth pricing sits from where it has been over six
    months, which is what makes a fast repricing stand out.
    """
    if closes is None or closes.empty:
        return pd.DataFrame()
    r = np.log(closes.dropna(how="any")).diff().dropna()
    if len(r) < warmup + z_window or anchor not in r.columns:
        return pd.DataFrame()
    vol = r.ewm(halflife=halflife, min_periods=warmup // 2).std().shift(1)
    z = (r / vol).clip(-5, 5).dropna()
    if len(z) < warmup + z_window:
        return pd.DataFrame()
    arr = z.to_numpy(dtype=float)
    k = arr.shape[1]
    a_idx = list(z.columns).index(anchor)
    lam = 0.5 ** (1 / halflife)
    cov = np.eye(k)
    f = np.full(len(arr), np.nan)
    ev = np.full(len(arr), np.nan)
    for t in range(len(arr)):
        if t >= warmup:
            d = np.sqrt(np.clip(np.diag(cov), 1e-9, None))
            corr = cov / np.outer(d, d)
            vals, vecs = np.linalg.eigh(corr)
            pc = vecs[:, -1]
            if pc[a_idx] < 0:
                pc = -pc
            f[t] = float(arr[t] @ pc) / np.sqrt(max(vals[-1], 1e-9))
            ev[t] = vals[-1] / k
        cov = lam * cov + (1 - lam) * np.outer(arr[t], arr[t])
    fs = pd.Series(f, index=z.index).dropna()
    level = fs.rolling(horizon).sum() / np.sqrt(horizon)
    cum = fs.cumsum()
    mean = cum.rolling(z_window, min_periods=z_window // 2).mean()
    std = cum.rolling(z_window, min_periods=z_window // 2).std()
    out = pd.DataFrame({
        "factor": level,
        "z_6m": (cum - mean) / std.replace(0, np.nan),
        "explained": pd.Series(ev, index=z.index),
    }).dropna(subset=["factor"])
    return out


def growth_snapshot(pca: pd.DataFrame) -> dict:
    if pca is None or pca.empty:
        return {}
    last = pca.dropna(subset=["z_6m"]).iloc[-1]
    z = float(last["z_6m"])
    z_hist = pca["z_6m"].dropna()
    rising = bool(z_hist.iloc[-1] > z_hist.iloc[-6]) if len(z_hist) > 5 else False
    if z <= -2:
        state = "EXTREME DOWN" + (" · TURNING" if rising else "")
    elif z >= 2:
        state = "EXTREME UP"
    elif z < 0:
        state = "RECOVERING" if rising else "WEAKENING"
    else:
        state = "IMPROVING" if rising else "ROLLING OVER"
    return {
        "as_of": pca.dropna(subset=["z_6m"]).index[-1],
        "factor": float(last["factor"]),
        "z_6m": z,
        "explained": float(last["explained"]) * 100,
        "pct_1y": pct_rank(pca["factor"].iloc[-252:]),
        "state": state,
        "rising": rising,
    }


# ---------------------------------------------------------------------------
# Boards: technical extremes and rotation scoreboards
# ---------------------------------------------------------------------------
def technical_row(s: pd.Series) -> dict:
    s = clean(s)
    if len(s) < 60:
        return {}
    last = float(s.iloc[-1])
    ma = {n: s.rolling(n).mean().iloc[-1] if len(s) >= n else float("nan") for n in (20, 50, 200)}
    rsi_d = rsi_wilder(s, 14)
    roc5 = s.pct_change(5, fill_method=None) * 100
    hi = float(s.iloc[-252:].max())
    ma50, ma200 = s.rolling(50).mean(), s.rolling(200).mean()
    cross = ""
    if len(s) >= 220:
        spread = (ma50 - ma200).dropna().iloc[-21:]
        flips = np.sign(spread).diff().fillna(0)
        if (flips > 0).any():
            cross = "Golden cross"
        elif (flips < 0).any():
            cross = "Death cross"
    r = float(rsi_d.iloc[-1]) if not rsi_d.empty else float("nan")
    flag = "OVERSOLD" if r <= 30 else "OVERBOUGHT" if r >= 70 else ""
    return {
        "last": last,
        "d1": float(s.iloc[-1] / s.iloc[-2] - 1) * 100,
        "d5": float(roc5.iloc[-1]),
        "d21": float(s.iloc[-1] / s.iloc[-22] - 1) * 100,
        "rsi": r,
        "vs20": (last / ma[20] - 1) * 100,
        "vs50": (last / ma[50] - 1) * 100,
        "vs200": (last / ma[200] - 1) * 100 if not np.isnan(ma[200]) else float("nan"),
        "dd_52w": (last / hi - 1) * 100,
        "roc5_pct": pct_rank(roc5.dropna()),
        "flag": flag,
        "cross": cross,
    }


def technical_board(prices: pd.DataFrame, names: dict[str, str], overrides: dict[str, pd.Series] | None = None) -> pd.DataFrame:
    """`overrides` supplies a series on its own calendar (Bitcoin trades 7 days)."""
    rows = []
    for t, label in names.items():
        over = (overrides or {}).get(t)
        row = technical_row(over if over is not None and len(over) else px(prices, t))
        if row:
            rows.append({"ticker": t, "name": label, **row})
    return pd.DataFrame(rows)


def scoreboard(prices: pd.DataFrame, names: dict[str, str], *, days: int = 10, bench: str = "SPY") -> pd.DataFrame:
    """Return over `days` sessions and vs the benchmark, best to worst."""
    b = px(prices, bench)
    if len(b) <= days:
        return pd.DataFrame()
    end = b.index[-1]
    start = b.index[-1 - days]
    br = float(b.iloc[-1] / b.iloc[-1 - days] - 1) * 100
    rows = []
    for t, label in names.items():
        s = px(prices, t)
        if s.empty:
            continue
        a, z = s.asof(start), s.asof(end)
        if pd.isna(a) or pd.isna(z) or s.index[0] > start:
            continue
        ret = float(z / a - 1) * 100
        rows.append({"ticker": t, "name": label, "ret": ret, "vs_bench": ret - br})
    return pd.DataFrame(rows).sort_values("ret", ascending=False).reset_index(drop=True) if rows else pd.DataFrame()


def since_date_returns(prices: pd.DataFrame, names: list[str], start: pd.Timestamp) -> pd.DataFrame:
    """Each name rebased to 100 at `start` (for the rotation line chart)."""
    cols = [c for c in names if c in prices.columns]
    if not cols:
        return pd.DataFrame()
    p = prices[cols].ffill().loc[prices.index >= pd.Timestamp(start)].dropna(how="all", axis=1)
    if p.empty:
        return p
    return p / p.iloc[0] * 100.0


def basket_index(prices: pd.DataFrame, tickers: list[str]) -> pd.Series:
    """Equal-weight, daily-rebalanced basket index (100 at the first date).

    Names join when they start trading, so a recent IPO cannot distort the level.
    """
    cols = [c for c in tickers if c in prices.columns]
    if not cols:
        return pd.Series(dtype=float)
    rets = prices[cols].pct_change(fill_method=None)
    avg = rets.mean(axis=1, skipna=True).dropna()
    return (100.0 * (1 + avg).cumprod()).rename("basket")


def relative_extension(num: pd.Series, den: pd.Series, days: int = 100) -> dict:
    """Relative return over `days` in pp and its z-score vs the pair's own history."""
    df = pd.concat([clean(num).rename("n"), clean(den).rename("d")], axis=1, join="inner").dropna()
    if len(df) < days + 252:
        return {}
    rel = (df["n"].pct_change(days, fill_method=None) - df["d"].pct_change(days, fill_method=None)).dropna() * 100
    return {
        "as_of": rel.index[-1], "rel_pp": float(rel.iloc[-1]),
        "z": float((rel.iloc[-1] - rel.mean()) / rel.std()), "pct": pct_rank(rel), "series": rel,
    }


def ratio_regime(num: pd.Series, den: pd.Series) -> dict:
    """Ratio level, full-history and 3Y percentile, distance from its 200D average."""
    df = pd.concat([clean(num).rename("n"), clean(den).rename("d")], axis=1, join="inner").dropna()
    if len(df) < 260:
        return {}
    r = df["n"] / df["d"]
    ma20, ma50, ma200 = r.rolling(20).mean(), r.rolling(50).mean(), r.rolling(200).mean()
    tail = r.iloc[-756:]
    return {
        "as_of": r.index[-1], "ratio": float(r.iloc[-1]), "series": r, "ma200": ma200,
        "pct_all": pct_rank(r), "pct_3y": pct_rank(tail),
        "vs_200": float(r.iloc[-1] / ma200.iloc[-1] - 1) * 100,
        "chg_20d": float(r.iloc[-1] / r.iloc[-21] - 1) * 100,
        "above_20": bool(r.iloc[-1] > ma20.iloc[-1]),
        "above_50": bool(r.iloc[-1] > ma50.iloc[-1]),
    }


def monthly_spread(a: pd.Series, b: pd.Series) -> pd.DataFrame:
    """Calendar-month total return of a minus b, in pp (RSP - XLK style)."""
    df = pd.concat([clean(a).rename("a"), clean(b).rename("b")], axis=1, join="inner").dropna()
    if df.empty:
        return pd.DataFrame()
    m = df.resample("ME").last().pct_change(fill_method=None).dropna() * 100
    m["spread"] = m["a"] - m["b"]
    return m


def monthly_spread_snapshot(m: pd.DataFrame) -> dict:
    """Latest COMPLETE month plus the month in progress, with rank and tails."""
    if m is None or m.empty:
        return {}
    s = m["spread"]
    now = s.index[-1]
    today_me = pd.Timestamp.today().normalize() + pd.offsets.MonthEnd(0)
    complete = s[s.index < today_me] if now >= today_me else s
    if complete.empty:
        complete = s
    v = float(complete.iloc[-1])
    return {
        "month": complete.index[-1], "spread": v,
        "pct": pct_rank(complete), "rank": int((complete > v).sum() + 1), "n": int(len(complete)),
        "p95": float(complete.quantile(0.95)), "p99": float(complete.quantile(0.99)),
        "mtd": float(s.iloc[-1]) if now >= today_me else float("nan"),
    }


# ---------------------------------------------------------------------------
# Index internals from the constituent panel
# ---------------------------------------------------------------------------
def internals_frame(panel: pd.DataFrame) -> pd.DataFrame:
    """Daily advancers/decliners and % above 20/50/200DMA from constituent closes."""
    if panel is None or panel.empty:
        return pd.DataFrame()
    c = panel.ffill(limit=5)
    rets = c.pct_change(fill_method=None)
    out = pd.DataFrame({
        "adv": (rets > 0).sum(axis=1).astype(float),
        "dec": (rets < 0).sum(axis=1).astype(float),
    })
    for n in (20, 50, 200):
        ma = c.rolling(n, min_periods=n).mean()
        valid = (c.notna() & ma.notna()).sum(axis=1)
        pct = (c > ma).sum(axis=1) / valid.replace(0, np.nan) * 100
        out[f"pct_{n}"] = pct.where(valid >= 0.5 * c.shape[1])
    return out.iloc[1:]


def divergence_days(internals: pd.DataFrame, index_px: pd.Series, *, min_adv: int = 300, max_ret: float = -0.25) -> pd.DataFrame:
    """Days the index fell at least `max_ret`% while most constituents rose."""
    if internals is None or internals.empty:
        return pd.DataFrame()
    ret = clean(index_px).pct_change(fill_method=None) * 100
    df = internals[["adv"]].join(ret.rename("ret"), how="inner").dropna()
    return df[(df["adv"] >= min_adv) & (df["ret"] <= max_ret)]


def disconnect_snapshot(internals: pd.DataFrame, index_px: pd.Series, days: int = 20) -> dict:
    """Index vs participation over `days`: rising breadth under a flat index, etc."""
    if internals is None or internals.empty or "pct_50" not in internals:
        return {}
    p50 = internals["pct_50"].dropna()
    idx = clean(index_px)
    if len(p50) <= days or len(idx) <= days:
        return {}
    d_b = float(p50.iloc[-1] - p50.iloc[-1 - days])
    d_i = float(idx.iloc[-1] / idx.iloc[-1 - days] - 1) * 100
    if d_b >= 5 and d_i <= 1:
        read = "Participation broadening under a weak index"
    elif d_b <= -5 and d_i >= 1:
        read = "Index rising on narrowing participation"
    elif d_b >= 5:
        read = "Rally is broadening"
    elif d_b <= -5:
        read = "Selling is broadening"
    else:
        read = "Index and participation agree"
    return {"pct_50": float(p50.iloc[-1]), "chg_pp": d_b, "index_ret": d_i, "read": read, "days": days}


# ---------------------------------------------------------------------------
# Rates
# ---------------------------------------------------------------------------
def bond_composite(prices: pd.DataFrame, tickers: tuple[str, ...] = ("HYG", "LQD", "TLT")) -> pd.DataFrame:
    """Equal-weight bond ETF composite with daily and weekly Wilder RSI."""
    cols = [t for t in tickers if t in prices.columns]
    if len(cols) < len(tickers):
        return pd.DataFrame()
    p = prices[cols].dropna(how="any")
    if len(p) < 300:
        return pd.DataFrame()
    comp = 100.0 * (1 + p.pct_change(fill_method=None).mean(axis=1).fillna(0)).cumprod()
    wk = weekly(comp)
    return pd.DataFrame({"composite": comp, "rsi_d": rsi_wilder(comp, 14), "rsi_w": rsi_wilder(wk, 14)}).assign(
        rsi_w=lambda d: d["rsi_w"].reindex(d.index).ffill(limit=5)
    )


def yield_rsi_episodes(y: pd.Series, *, level: float = 70.0, gap_weeks: int = 13) -> pd.DataFrame:
    """Weekly RSI peaks above `level` on a yield and the fall that followed.

    Separate overbought runs are one episode each; `peak_date` is the yield high
    of the run and the fall is measured to the low of the next 26 weeks.
    """
    w = weekly(y)
    r = rsi_wilder(w, 14)
    if r.empty:
        return pd.DataFrame()
    hot = r[r >= level]
    if hot.empty:
        return pd.DataFrame()
    rows, run = [], [hot.index[0]]
    for d in hot.index[1:]:
        if (d - run[-1]).days <= gap_weeks * 7:
            run.append(d)
        else:
            rows.append(run)
            run = [d]
    rows.append(run)
    out = []
    for run in rows:
        seg = w.loc[run[0]:run[-1]]
        peak_d = seg.idxmax()
        peak = float(seg.max())
        after = w.loc[peak_d:].iloc[1:27]
        complete = len(after) >= 8
        low_d = after.idxmin() if not after.empty else pd.NaT
        out.append({
            "peak_date": peak_d, "peak_yield": peak, "rsi": float(r.loc[run].max()),
            "fall_pct": float(after.min() / peak - 1) * 100 if not after.empty else float("nan"),
            "weeks_to_low": int((low_d - peak_d).days / 7) if not after.empty else None,
            "complete": complete,
        })
    return pd.DataFrame(out)


def change_shock(y: pd.Series, days: int = 10) -> dict:
    """How extreme the latest `days`-session yield change is, in bps."""
    s = clean(y)
    if len(s) <= days + 60:
        return {}
    ch = s.diff(days).dropna() * 100
    return {
        "as_of": ch.index[-1], "bps": float(ch.iloc[-1]), "pct": pct_rank(ch),
        "z": float((ch.iloc[-1] - ch.mean()) / ch.std()), "series": ch,
    }


def excess_cape_yield(cape: pd.Series, gs10: pd.Series, cpi: pd.Series) -> pd.Series:
    """100/CAPE minus the 10Y real yield (nominal less trailing 10Y CPI inflation)."""
    cape, gs10, cpi = clean(cape), clean(gs10), clean(cpi)
    if cape.empty or gs10.empty or cpi.empty:
        return pd.Series(dtype=float)
    infl = ((cpi / cpi.shift(120)) ** 0.1 - 1) * 100
    m_gs = gs10.resample("MS").mean()
    m_cape = cape.resample("MS").last()
    # CPI publishes about 6 weeks late: carry the last print into recent months.
    idx = m_cape.index.union(m_gs.index)
    m_infl = infl.resample("MS").last().reindex(idx).ffill(limit=3)
    df = pd.concat([m_cape.rename("c"), m_gs.rename("g"), m_infl.rename("i")], axis=1).dropna()
    return (100 / df["c"] - (df["g"] - df["i"])).rename("excess_cape_yield")


# ---------------------------------------------------------------------------
# Event studies
# ---------------------------------------------------------------------------
def fresh(flag: pd.Series, gap: int = 20) -> pd.DatetimeIndex:
    """Event dates from a boolean series: first True after `gap` sessions of quiet."""
    f = flag.fillna(False).astype(bool)
    dates, last = [], None
    idx = list(f.index)
    pos = np.flatnonzero(f.to_numpy())
    for p in pos:
        if last is None or p - last >= gap:
            dates.append(idx[p])
            last = p
    return pd.DatetimeIndex(dates)


def event_study(
    price: pd.Series,
    events: pd.DatetimeIndex,
    horizons: tuple[int, ...] = (5, 10, 21, 63, 126),
    *,
    n_perm: int = 2000,
    seed: int = 7,
) -> pd.DataFrame:
    """Forward returns after each event, next to the unconditional base rate.

    * `n` counts only events whose full horizon has elapsed.
    * `base_mean` / `base_hit` are the unconditional stats from the first event
      to the end of the sample, so the comparison shares the signal's era.
    * `p_mean` is a permutation p-value: the share of random same-size date sets
      with a mean return at least as high. Overlapping windows make a t-test
      overconfident; a permutation against the same series is more honest.
    * `mae` is the average worst close-to-close drawdown inside the horizon: how
      much pain was typically taken before the outcome.
    """
    p = clean(price)
    if p.empty or len(events) == 0:
        return pd.DataFrame()
    locs = p.index.searchsorted(pd.DatetimeIndex(events))
    locs = locs[locs < len(p)]
    if len(locs) == 0:
        return pd.DataFrame()
    arr = p.to_numpy(dtype=float)
    start = int(locs.min())
    rng = np.random.default_rng(seed)
    rows = []
    for h in horizons:
        fwd_all = np.full(len(arr), np.nan)
        fwd_all[:-h] = arr[h:] / arr[:-h] - 1
        valid = locs[locs + h < len(arr)]
        base = fwd_all[start:]
        base = base[~np.isnan(base)] * 100
        if len(valid) == 0 or len(base) == 0:
            rows.append({"horizon": h, "n": 0})
            continue
        r = fwd_all[valid] * 100
        mae = np.array([(arr[i + 1: i + h + 1].min() / arr[i] - 1) * 100 for i in valid])
        perm = rng.choice(base, size=(n_perm, len(valid)), replace=True).mean(axis=1)
        rows.append({
            "horizon": h, "n": int(len(valid)),
            "mean": float(r.mean()), "median": float(np.median(r)),
            "hit": float((r > 0).mean() * 100), "worst": float(r.min()),
            "mae": float(mae.mean()),
            "base_mean": float(base.mean()), "base_hit": float((base > 0).mean() * 100),
            "edge_hit": float((r > 0).mean() * 100 - (base > 0).mean() * 100),
            "p_mean": float((perm >= r.mean()).mean()),
        })
    return pd.DataFrame(rows)


def strength_label(row: pd.Series | dict) -> str:
    """Plain-language evidence grade so a thin sample can't read as a strong one."""
    n = row.get("n", 0) or 0
    if n < 8:
        return "Too few events"
    p = row.get("p_mean", 1.0)
    edge = row.get("edge_hit", 0.0)
    if p <= 0.05 and edge >= 5:
        return "Real edge"
    if p <= 0.20 and edge > 0:
        return "Weak edge"
    return "No edge vs base rate"


@dataclass(frozen=True)
class Signal:
    key: str
    label: str
    target: str
    flag: pd.Series
    gap: int = 20
    note: str = ""


def signal_library(
    prices: pd.DataFrame,
    *,
    internals: pd.DataFrame | None = None,
    trend_breadth: pd.Series | None = None,
    cot: pd.DataFrame | None = None,
) -> list[Signal]:
    """Named historical signals, each a no-look-ahead boolean series."""
    out: list[Signal] = []
    spx = px(prices, "^GSPC")
    qqq = px(prices, "QQQ")

    if len(spx) > 300:
        ma = spx.rolling(50).mean()
        below = spx < ma
        # First close under the 50DMA after at least 20 sessions above it.
        above_run = (~below).astype(int).rolling(20).sum().shift(1) == 20
        out.append(Signal("spx_50dma", "S&P 500 first close below 50DMA", "^GSPC", below & above_run, 5,
                          "After 20+ sessions above the average."))

    skew = px(prices, "^SKEW")
    if len(skew) > 2600:
        thr = skew.rolling(2520, min_periods=1260).quantile(0.985).shift(1)
        out.append(Signal("skew_extreme", "SKEW above its rolling 10Y 98.5th pct", "^GSPC", skew >= thr, 20,
                          "Tail-hedge demand at an extreme."))

    if len(qqq) > 300:
        r = rsi_wilder(qqq, 14)
        cross = (r <= 30) & (r.shift(1) > 30)
        out.append(Signal("qqq_rsi30", "QQQ RSI falls through 30", "QQQ", cross, 10, "Wilder 14-day RSI."))

    vxn = px(prices, "^VXN")
    if len(vxn) > 300:
        out.append(Signal("vxn_30", "VXN crosses above 30", "QQQ", (vxn >= 30) & (vxn.shift(1) < 30), 20,
                          "Fresh entry, after 20 quiet sessions."))

    vix = px(prices, "^VIX")
    if len(vix) > 300:
        out.append(Signal("vix_spike", "VIX up 40%+ in a day", "^GSPC", vix.pct_change(fill_method=None) >= 0.40, 10,
                          "Close-to-close."))

    sox = px(prices, "^SOX")
    if len(sox) > 300:
        out.append(Signal("sox_flush", "SOX down 17.5%+ in 5 days", "^SOX",
                          sox.pct_change(5, fill_method=None) <= -0.175, 20, "Semiconductor index flush."))

    gvip, spy = px(prices, "GVIP"), px(prices, "SPY")
    if len(gvip) > 500 and len(spy) > 500:
        rel = (gvip.pct_change(5, fill_method=None) - spy.pct_change(5, fill_method=None)).dropna()
        thr = rel.expanding(252).quantile(0.02).shift(1)
        out.append(Signal("hf_unwind", "Hedge-fund crowded longs unwind (proxy)", "^GSPC", rel <= thr, 20,
                          "GVIP minus SPY 5D at a 2nd-percentile low. Long-only proxy, not prime-book de-grossing."))

    mv, tnx = px(prices, "^MOVE"), px(prices, "^TNX")
    if len(mv) > 300 and len(tnx) > 300:
        hi = mv.rolling(60).max()
        tnx_a = tnx.reindex(mv.index).ffill()
        fading = (hi >= 110) & (mv <= hi * 0.8) & (tnx_a < tnx_a.rolling(20).mean())
        out.append(Signal("move_fade", "MOVE fades 20% from a spike, yields falling", "^GSPC", fading, 20,
                          "Spike high of at least 110."))

    tb = clean(trend_breadth)
    if len(tb) > 100:
        recent_low = tb.rolling(20).min().shift(1) <= 25
        out.append(Signal("breadth_turn", "Trend breadth back above 50 from a washout", "^GSPC",
                          recent_low & (tb >= 50), 20, "10/20DMA breadth at or below 25 within 20 sessions."))

    if internals is not None and not internals.empty:
        ret = spx.pct_change(fill_method=None) * 100
        df = internals[["adv"]].join(ret.rename("ret"), how="inner").dropna()
        out.append(Signal("divergence", "Index falls while 300+ members rise", "^GSPC",
                          (df["ret"] <= -0.25) & (df["adv"] >= 300), 3, "Current constituents; 5 years of history."))

    rsp = px(prices, "RSP")
    if len(rsp) > 1000 and len(spy) > 1000:
        ratio = (rsp / spy.reindex(rsp.index).ffill()).dropna().rename("RSP/SPY")
        rank = rolling_rank(ratio, 756, 252)
        out.append(Signal("rsp_turn", "RSP/SPY at a 3Y low and back above its 20DMA", "RSP/SPY",
                          (rank.rolling(20).min() <= 5) & (ratio > ratio.rolling(20).mean()), 40,
                          "Target is the ratio itself."))

    if cot is not None and not cot.empty:
        def weekly_flag(key: str, fn: Callable[[pd.Series], pd.Series]) -> pd.Series:
            s = clean(cot[key]) if key in cot.columns else pd.Series(dtype=float)
            if len(s) < 200:
                return pd.Series(dtype=bool)
            flag = fn(expanding_rank(s, 104))
            # COT is as of Tuesday and published Friday: act from the next close.
            flag.index = flag.index + pd.Timedelta(days=3)
            return flag
        for key, target, fn, label, gap in (
            ("VIX", "^VIX", lambda r: r <= 5, "VIX asset-manager net short at a 5th-pct extreme", 13),
            ("TY", "TLT", lambda r: r <= 10, "UST 10Y leveraged net short at a 10th-pct extreme", 13),
            ("RTY", "IWM", lambda r: r <= 10, "Russell 2000 leveraged net short at a 10th-pct extreme", 13),
            ("GASOLINE", "RB=F", lambda r: r >= 95, "Gasoline managed-money long at a 95th-pct extreme", 13),
        ):
            f = weekly_flag(key, fn)
            if not f.empty:
                out.append(Signal(f"cot_{key.lower()}", label, target, f, gap, "COT lagged to publication."))
    return out


def run_playbook(prices: pd.DataFrame, signals: list[Signal], horizons=(5, 10, 21, 63, 126)) -> pd.DataFrame:
    """Long-format event-study results for every signal, plus recency."""
    rows = []
    for sig in signals:
        if sig.target == "RSP/SPY":
            a, b = px(prices, "RSP"), px(prices, "SPY")
            target = (a / b.reindex(a.index).ffill()).dropna()
        else:
            target = px(prices, sig.target)
        if target.empty:
            continue
        events = fresh(sig.flag, sig.gap)
        if len(events) == 0:
            continue
        study = event_study(target, events, horizons)
        if study.empty:
            continue
        last = events[-1]
        days_ago = int((target.index[-1] - last).days)
        for r in study.to_dict("records"):
            r.update({"key": sig.key, "signal": sig.label, "target": sig.target, "note": sig.note,
                      "events": len(events), "last_event": last, "days_ago": days_ago,
                      "first_event": events[0], "evidence": strength_label(r)})
            rows.append(r)
    return pd.DataFrame(rows)


def largest_moves(s: pd.Series, n: int = 10) -> pd.DataFrame:
    """Top `n` close-to-close percentage gains."""
    p = clean(s)
    if len(p) < 2:
        return pd.DataFrame()
    ch = p.pct_change(fill_method=None).dropna() * 100
    top = ch.sort_values(ascending=False).head(n)
    return pd.DataFrame({"date": top.index, "gain": top.values, "close": p.reindex(top.index).values})


# ---------------------------------------------------------------------------
# Seasonality
# ---------------------------------------------------------------------------
def midterm_seasonality(spx: pd.Series) -> pd.DataFrame:
    """Nov-Apr returns for each midterm-year cycle, and the 6-month total."""
    p = clean(spx)
    if p.empty:
        return pd.DataFrame()
    m = p.resample("ME").last()
    r = m.pct_change(fill_method=None) * 100
    rows = []
    for y in sorted({d.year for d in m.index if d.year % 4 == 2}):
        months = [(y, 11), (y, 12), (y + 1, 1), (y + 1, 2), (y + 1, 3), (y + 1, 4)]
        vals = {}
        for yy, mm in months:
            hit = r[(r.index.year == yy) & (r.index.month == mm)]
            vals[pd.Timestamp(year=yy, month=mm, day=1).strftime("%b")] = float(hit.iloc[0]) if len(hit) else np.nan
        oct_close = m[(m.index.year == y) & (m.index.month == 10)]
        apr_close = m[(m.index.year == y + 1) & (m.index.month == 4)]
        six = float(apr_close.iloc[0] / oct_close.iloc[0] - 1) * 100 if len(oct_close) and len(apr_close) else np.nan
        rows.append({"cycle": f"{y}-{str(y + 1)[-2:]}", **vals, "six_m": six})
    return pd.DataFrame(rows)


def non_midterm_base(spx: pd.Series) -> dict:
    """The same Oct-to-Apr window in all OTHER years, as the base rate."""
    p = clean(spx)
    if p.empty:
        return {}
    m = p.resample("ME").last()
    rets = []
    for y in sorted({d.year for d in m.index}):
        if y % 4 == 2:
            continue
        a = m[(m.index.year == y) & (m.index.month == 10)]
        b = m[(m.index.year == y + 1) & (m.index.month == 4)]
        if len(a) and len(b):
            rets.append(float(b.iloc[0] / a.iloc[0] - 1) * 100)
    if not rets:
        return {}
    arr = np.array(rets)
    return {"n": len(arr), "mean": float(arr.mean()), "hit": float((arr > 0).mean() * 100)}


# ---------------------------------------------------------------------------
# Decision rules
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Rule:
    key: str
    label: str
    value: float
    threshold: str
    fired: bool
    near: bool = False
    core: bool = False
    detail: str = ""
    available: bool = True


def _last(s: pd.Series | None) -> float:
    s = clean(s)
    return float(s.iloc[-1]) if not s.empty else float("nan")


def _unavailable(key: str, label: str, threshold: str, core: bool = False) -> Rule:
    return Rule(key, label, float("nan"), threshold, False, core=core, available=False)


# Rules that count toward the verdict. SKEW stays on the checklist as context but
# isn't counted: its edge was negative in both 2003-2016 and 2017-2026, so a high
# reading isn't bullish evidence. See .claude/plan/rule-backtest-results.md.
CORE_RULES = ("qqq_rsi", "vxn", "sox_flush", "vix_spike", "momentum", "hf_unwind")


def capitulation_history(prices: pd.DataFrame) -> pd.DataFrame:
    """Daily booleans for the capitulation rules (long, price-derived history).

    Includes the context-only SKEW rule; count only `CORE_RULES`.

    Each rule stays on for a short memory window because capitulation signals
    arrive in sequence over a few sessions, not all on one close. Percentile
    thresholds use expanding ranks, so a date only knows its own past.
    """
    qqq, sox = px(prices, "QQQ"), px(prices, "^SOX")
    vxn, vix, skew = px(prices, "^VXN"), px(prices, "^VIX"), px(prices, "^SKEW")
    mtum, spy, gvip = px(prices, "MTUM"), px(prices, "SPY"), px(prices, "GVIP")
    if qqq.empty or spy.empty:
        return pd.DataFrame()
    idx = qqq.index
    held = lambda f, n: f.reindex(idx).fillna(False).astype(bool).rolling(n, min_periods=1).max().astype(bool)  # noqa: E731
    out = pd.DataFrame(index=idx)
    out["qqq_rsi"] = held(rsi_wilder(qqq, 14) <= 32, 5)
    if not vxn.empty:
        out["vxn"] = held(vxn >= 30, 5)
    if not sox.empty:
        out["sox_flush"] = held(sox.pct_change(5, fill_method=None) <= -0.15, 5)
    if not vix.empty:
        out["vix_spike"] = held(vix.pct_change(fill_method=None) >= 0.30, 5)
    if len(skew) > 1500:
        thr = skew.rolling(2520, min_periods=1260).quantile(0.985).shift(1)
        out["skew"] = held(skew >= thr, 10)
    if len(mtum) > 500:
        rel = (mtum.pct_change(5, fill_method=None) - spy.pct_change(5, fill_method=None)).dropna() * 100
        out["momentum"] = held(expanding_rank(rel, 252) <= 2, 5)
    if len(gvip) > 500:
        rel = (gvip.pct_change(5, fill_method=None) - spy.pct_change(5, fill_method=None)).dropna() * 100
        out["hf_unwind"] = held(expanding_rank(rel, 252) <= 2, 5)
    return out


def capitulation_rules(
    prices: pd.DataFrame,
    *,
    put_call: pd.Series | None = None,
    fear_greed: pd.Series | None = None,
    aaii: pd.Series | None = None,
    ndx_pe: pd.Series | None = None,
    put_volume: pd.Series | None = None,
    history: pd.DataFrame | None = None,
) -> list[Rule]:
    """The tech-capitulation checklist, current state. `core` rules are backtested."""
    qqq, sox = px(prices, "QQQ"), px(prices, "^SOX")
    vxn, vix, skew = px(prices, "^VXN"), px(prices, "^VIX"), px(prices, "^SKEW")
    mtum, spy, gvip = px(prices, "MTUM"), px(prices, "SPY"), px(prices, "GVIP")
    h = history if history is not None else capitulation_history(prices)
    on = lambda k: bool(h[k].iloc[-1]) if h is not None and not h.empty and k in h.columns else False  # noqa: E731
    rules: list[Rule] = []

    r = rsi_wilder(qqq, 14)
    if r.empty:
        rules.append(_unavailable("qqq_rsi", "QQQ RSI washed out", "≤ 32 in last 5 sessions", True))
    else:
        low = float(r.iloc[-5:].min())
        rules.append(Rule("qqq_rsi", "QQQ RSI washed out", float(r.iloc[-1]), "≤ 32 in last 5 sessions", on("qqq_rsi"),
                          near=low <= 38, core=True, detail=f"5-session low {low:.1f}"))
    if vxn.empty:
        rules.append(_unavailable("vxn", "Nasdaq vol panic (VXN)", "≥ 30 in last 5 sessions", True))
    else:
        v = float(vxn.iloc[-1])
        rules.append(Rule("vxn", "Nasdaq vol panic (VXN)", v, "≥ 30 in last 5 sessions", on("vxn"),
                          near=float(vxn.iloc[-5:].max()) >= 26, core=True,
                          detail=f"{ordinal(pct_rank(vxn.iloc[-1260:]))} pct of 5Y"))
    if sox.empty:
        rules.append(_unavailable("sox_flush", "Semis flush (SOX 5D)", "≤ −15% in last 5 sessions", True))
    else:
        roc = sox.pct_change(5, fill_method=None) * 100
        low = float(roc.iloc[-5:].min())
        rules.append(Rule("sox_flush", "Semis flush (SOX 5D)", float(roc.iloc[-1]), "≤ −15% in last 5 sessions",
                          on("sox_flush"), near=low <= -10, core=True, detail=f"5-session low {low:+.1f}%"))
    if vix.empty:
        rules.append(_unavailable("vix_spike", "VIX one-day spike", "+30% in last 5 sessions", True))
    else:
        ch = vix.pct_change(fill_method=None) * 100
        top = float(ch.iloc[-5:].max())
        rules.append(Rule("vix_spike", "VIX one-day spike", top, "+30% in last 5 sessions", on("vix_spike"),
                          near=top >= 20, core=True, detail=f"largest close-to-close rise, VIX {vix.iloc[-1]:.1f}"))
    if len(skew) <= 1500:
        rules.append(_unavailable("skew", "Tail-hedge demand (SKEW)", "≥ rolling 10Y 98.5th pct in 10 sessions"))
    else:
        p = pct_rank(skew.iloc[-2520:])
        recent = pct_rank(skew.iloc[-2520:], float(skew.iloc[-10:].max()))
        rules.append(Rule("skew", "Tail-hedge demand (SKEW)", float(skew.iloc[-1]),
                          "≥ rolling 10Y 98.5th pct in 10 sessions", on("skew"), near=recent >= 90,
                          detail=f"now {ordinal(p)} pct of 10Y, 10-session high {ordinal(recent)}"))
    for key, label, a in (("momentum", "Momentum unwind (MTUM−SPY 5D)", mtum), ("hf_unwind", "Crowded HF longs unwind (GVIP−SPY 5D)", gvip)):
        if len(a) <= 500 or spy.empty:
            rules.append(_unavailable(key, label, "≤ 2nd pct of history", True))
            continue
        rel = (a.pct_change(5, fill_method=None) - spy.pct_change(5, fill_method=None)).dropna() * 100
        ranks = expanding_rank(rel, 252).iloc[-5:]
        rules.append(Rule(key, label, float(rel.iloc[-1]), "≤ 2nd pct of history", on(key),
                          near=float(ranks.min()) <= 8, core=True,
                          detail=f"{rel.iloc[-1]:+.1f}pp now; 5-session low ranks {ordinal(ranks.min())} pct"))

    pc = clean(put_call)
    if len(pc) < 5:
        rules.append(_unavailable("put_call", "Equity put/call", "≥ 0.85 in last 5 sessions"))
    else:
        high = float(pc.iloc[-5:].max())
        rules.append(Rule("put_call", "Equity put/call", float(pc.iloc[-1]), "≥ 0.85 in last 5 sessions",
                          high >= 0.85, near=high >= 0.75, detail=f"5-session high {high:.2f}, Cboe official"))
    s = clean(put_volume)
    if len(s) < 60:
        rules.append(_unavailable("put_volume", "Total put volume", "≥ 95th pct of 1Y"))
    else:
        p = pct_rank(s.iloc[-252:], float(s.iloc[-5:].max()))
        rules.append(Rule("put_volume", "Total put volume", float(s.iloc[-1]) / 1e6, "≥ 95th pct of 1Y",
                          p >= 95, near=p >= 85, detail=f"5-session high ranks {ordinal(p)} pct"))
    v = _last(fear_greed)
    if np.isnan(v):
        rules.append(_unavailable("fear_greed", "CNN Fear & Greed", "≤ 25"))
    else:
        rules.append(Rule("fear_greed", "CNN Fear & Greed", v, "≤ 25", v <= 25, near=v <= 35))
    v = _last(aaii)
    if np.isnan(v):
        rules.append(_unavailable("aaii", "AAII bull−bear", "≤ −20"))
    else:
        rules.append(Rule("aaii", "AAII bull−bear", v, "≤ −20", v <= -20, near=v <= -10))
    pe = clean(ndx_pe)
    if len(pe) < 24:
        rules.append(_unavailable("ndx_pe", "Nasdaq 100 valuation reset", "≤ 10th pct of 5Y"))
    else:
        tail = pe[pe.index >= pe.index[-1] - pd.DateOffset(years=5)]
        p = pct_rank(tail)
        rules.append(Rule("ndx_pe", "Nasdaq 100 valuation reset", float(pe.iloc[-1]), "≤ 10th pct of 5Y",
                          p <= 10, near=p <= 25, detail=f"{ordinal(p)} pct of 5Y"))
    return rules


def capitulation_verdict(rules: list[Rule]) -> dict:
    core = [r for r in rules if r.core and r.available]
    live = [r for r in rules if r.available]
    fired_core = sum(r.fired for r in core)
    fired_all = sum(r.fired for r in live)
    if fired_core >= 5:
        state, tone = "CAPITULATION", "bull"
    elif fired_core >= 3:
        state, tone = "ARMED", "warm"
    elif fired_core >= 1:
        state, tone = "BUILDING", "neutral"
    else:
        state, tone = "QUIET", "neutral"
    return {"state": state, "tone": tone, "core_fired": fired_core, "core_total": len(core),
            "all_fired": fired_all, "all_total": len(live)}


def capitulation_backtest(history: pd.DataFrame, target: pd.Series, horizons=(10, 21, 63)) -> pd.DataFrame:
    """Forward returns by how many core rules were on, with the base rate."""
    if history is None or history.empty:
        return pd.DataFrame()
    # Only dates where every rule had enough history to be able to fire.
    count = history[[c for c in CORE_RULES if c in history.columns]].sum(axis=1)
    count = count.reindex(history.index[history.index >= _rules_live_from(history)])
    p = clean(target)
    rows = []
    for lo, hi, name in ((0, 0, "0"), (1, 2, "1–2"), (3, 4, "3–4"), (5, 99, "5+")):
        flag = (count >= lo) & (count <= hi)
        events = fresh(flag, 1 if lo == 0 else 10)
        if len(events) == 0:
            continue
        for rec in event_study(p, events, horizons).to_dict("records"):
            rec["bucket"] = name
            rows.append(rec)
    return pd.DataFrame(rows)


def _rules_live_from(history: pd.DataFrame) -> pd.Timestamp:
    """First date on which every core rule could plausibly have fired."""
    firsts = [history[c][history[c]].index[0] for c in CORE_RULES if c in history.columns and history[c].any()]
    return max(firsts) if firsts else history.index[0]


def reversal_confirmation(prices: pd.DataFrame, ai_corr: pd.Series | None = None) -> list[Rule]:
    """Price evidence that the flush is ending: the panic gauges roll over first."""
    qqq, vxn, skew = px(prices, "QQQ"), px(prices, "^VXN"), px(prices, "^SKEW")
    mv, tnx = px(prices, "^MOVE"), px(prices, "^TNX")
    out: list[Rule] = []
    if vxn.empty:
        out.append(_unavailable("vxn_fade", "VXN off its peak", "≥ 10% below 10-session high"))
    else:
        hi = float(vxn.iloc[-10:].max())
        off = (float(vxn.iloc[-1]) / hi - 1) * 100
        out.append(Rule("vxn_fade", "VXN off its peak", off, "≥ 10% below 10-session high", off <= -10,
                        near=off <= -5, detail=f"high {hi:.1f}, now {vxn.iloc[-1]:.1f}"))
    if len(qqq) < 25:
        out.append(_unavailable("qqq_5dma", "QQQ reclaims 5DMA and 20DMA", "close above both"))
    else:
        ma5, ma20 = float(qqq.rolling(5).mean().iloc[-1]), float(qqq.rolling(20).mean().iloc[-1])
        last = float(qqq.iloc[-1])
        out.append(Rule("qqq_5dma", "QQQ reclaims 5DMA and 20DMA", (last / ma20 - 1) * 100, "close above both",
                        last > ma5 and last > ma20, near=last > ma5, detail=f"{(last / ma5 - 1) * 100:+.1f}% vs 5DMA"))
    if mv.empty or tnx.empty:
        out.append(_unavailable("move_fade", "MOVE falling with yields", "MOVE < 20DMA and 10Y < 20DMA"))
    else:
        a, b = float(mv.iloc[-1]), float(mv.rolling(20).mean().iloc[-1])
        c, d = float(tnx.iloc[-1]), float(tnx.rolling(20).mean().iloc[-1])
        out.append(Rule("move_fade", "MOVE falling with yields", a, "MOVE < 20DMA and 10Y < 20DMA", a < b and c < d,
                        near=a < b or c < d, detail=f"MOVE 20DMA {b:.0f}, 10Y {c:.2f}% vs {d:.2f}%"))
    if len(skew) < 25:
        out.append(_unavailable("skew_fade", "SKEW rolling over", "< 10-session high by 5 pts"))
    else:
        off = float(skew.iloc[-1] - skew.iloc[-10:].max())
        out.append(Rule("skew_fade", "SKEW rolling over", off, "< 10-session high by 5 pts", off <= -5, near=off <= -2))
    c = clean(ai_corr)
    if len(c) < 30:
        out.append(_unavailable("ai_corr", "AI-basket correlation falling", "below 20 sessions ago"))
    else:
        d = float(c.iloc[-1] - c.iloc[-21])
        out.append(Rule("ai_corr", "AI-basket correlation falling", float(c.iloc[-1]), "below 20 sessions ago", d < 0,
                        near=d < 0.03, detail=f"{d:+.2f} vs 20 sessions ago"))
    return out


def _passthrough_limit(ccc: pd.Series, ig: pd.Series, default: float = 0.135) -> float:
    """90th percentile of IG's 20D move per bp of CCC widening, in past widenings.

    IG normally moves about 0.08x CCC, so a fixed "IG barely moved" test would
    misjudge an ordinary 11bp move. FRED serves only ~3 years of ICE data, so the
    default (measured Dec 2023 - Nov 2026) covers thin histories.
    """
    df = pd.concat([ig.diff(20).rename("i"), ccc.diff(20).rename("c")], axis=1).dropna() * 100
    wide = df[df["c"] >= 50]
    if len(wide) < 60:
        return default
    return float((wide["i"] / wide["c"]).iloc[:-1].quantile(0.90))


# SOFR−EFFR, 5-day average, below which funding is calm whatever its rank. Well
# under any real squeeze (Sept 2019 was about +300bp); it only removes the case
# where a spread near zero ranks high because the past year was even lower.
FUNDING_FLOOR_BP = 5.0


def system_guardrails(
    *,
    trend_breadth: pd.Series | None,
    pct_200: pd.Series | None,
    ig: pd.Series | None,
    hy: pd.Series | None,
    ccc: pd.Series | None,
    sofr_spread: pd.Series | None,
    spx_corr: pd.Series | None,
    nfci: pd.Series | None,
) -> list[Rule]:
    """Is this a contained rotation or a system problem? `fired` = guardrail HOLDS."""
    out: list[Rule] = []
    v = _last(trend_breadth)
    if np.isnan(v):
        out.append(_unavailable("breadth", "Short-term breadth holds", "10/20DMA breadth ≥ 50%"))
    else:
        out.append(Rule("breadth", "Short-term breadth holds", v, "10/20DMA breadth ≥ 50%", v >= 50, near=v >= 40))
    v = _last(pct_200)
    if np.isnan(v):
        out.append(_unavailable("pct200", "Long-term participation holds", "% above 200DMA ≥ 40%"))
    else:
        out.append(Rule("pct200", "Long-term participation holds", v, "% above 200DMA ≥ 40%", v >= 40, near=v >= 30))
    s = clean(ig)
    if len(s) < 25:
        out.append(_unavailable("ig", "Investment-grade credit calm", "IG OAS 20D change < +15bp"))
    else:
        ch = float(s.iloc[-1] - s.iloc[-21]) * 100
        rank = pct_rank(s.diff(20).dropna())
        out.append(Rule("ig", "Investment-grade credit calm", float(s.iloc[-1]) * 100, "IG OAS 20D change < +15bp",
                        ch < 15, near=ch < 25, detail=f"{ch:+.0f}bp in 20 sessions, {ordinal(rank)} pct of available history"))
    s = clean(hy)
    if len(s) < 25:
        out.append(_unavailable("hy", "High-yield credit calm", "HY OAS 20D change < +75bp"))
    else:
        ch = float(s.iloc[-1] - s.iloc[-21]) * 100
        out.append(Rule("hy", "High-yield credit calm", float(s.iloc[-1]) * 100, "HY OAS 20D change < +75bp",
                        ch < 75, near=ch < 110, detail=f"{ch:+.0f}bp in 20 sessions"))
    a, b = clean(ccc), clean(ig)
    if len(a) < 60 or len(b) < 60:
        out.append(_unavailable("contain", "Credit stress contained below IG", "CCC widening not passing through to IG"))
    else:
        ch_c = float(a.iloc[-1] - a.iloc[-21]) * 100
        ch_i = float(b.iloc[-1] - b.iloc[-21]) * 100
        beta = ch_i / ch_c if ch_c >= 50 else 0.0
        limit = _passthrough_limit(a, b)
        out.append(Rule("contain", "Credit stress contained below IG", float(a.iloc[-1]) * 100,
                        f"IG/CCC pass-through below {limit:.2f}x", beta < limit, near=beta < limit * 1.5,
                        detail=f"CCC {ch_c:+.0f}bp vs IG {ch_i:+.0f}bp in 20 sessions ({beta:.2f}x)"))
    s = clean(sofr_spread)
    label = f"SOFR−EFFR 5D avg below its 1Y 95th pct, or under +{FUNDING_FLOOR_BP:.0f}bp"
    if len(s) < 60:
        out.append(_unavailable("funding", "Funding markets calm", label))
    else:
        avg = s.iloc[-252:].rolling(5).mean().dropna()
        p, cur = pct_rank(avg), float(avg.iloc[-1])
        # A rank alone fails 5% of the time by construction: in a year when SOFR
        # sat below fed funds, a spread of zero is "the 95th percentile". Stress
        # also has to be a positive spread of some size.
        calm = cur < FUNDING_FLOOR_BP
        out.append(Rule("funding", "Funding markets calm", float(s.iloc[-1]), label,
                        p < 95 or calm, near=p < 99 or calm,
                        detail=f"5D average {cur:+.1f}bp, {ordinal(p)} pct of 1Y"))
    s = clean(spx_corr)
    if len(s) < 40:
        out.append(_unavailable("corr", "No correlation shock", "S&P 60D avg correlation 20D rise < 0.15"))
    else:
        ch = float(s.iloc[-1] - s.iloc[-21])
        out.append(Rule("corr", "No correlation shock", float(s.iloc[-1]), "S&P 60D avg correlation 20D rise < 0.15",
                        ch < 0.15, near=ch < 0.22, detail=f"{ch:+.2f} in 20 sessions"))
    v = _last(nfci)
    if np.isnan(v):
        out.append(_unavailable("nfci", "Financial conditions loose", "NFCI < 0"))
    else:
        out.append(Rule("nfci", "Financial conditions loose", v, "NFCI < 0", v < 0, near=v < 0.25))
    return out


def guardrail_verdict(rules: list[Rule]) -> dict:
    live = [r for r in rules if r.available]
    broken = [r for r in live if not r.fired]
    # Breadth alone can wobble; credit breaking is what turns a drawdown systemic.
    credit_broken = any(r.key in ("ig", "contain", "funding") for r in broken)
    if len(broken) >= 3 or (credit_broken and len(broken) >= 2):
        state, tone = "BREAKING", "bear"
        read = "Stress has left the sector. Treat as systemic until credit stabilises."
    elif broken:
        state, tone = "WATCH", "warm"
        read = "Check the failed ones before calling this contained."
    else:
        state, tone = "INTACT", "bull"
        read = "Stress is contained. The damage is sector or factor specific, not systemic."
    return {"state": state, "tone": tone, "read": read, "held": len(live) - len(broken), "total": len(live)}


def leverage_gate(
    *,
    vix_cot: dict | None,
    spx_corr: pd.Series | None,
    gamma: pd.Series | None,
    vix: pd.Series | None,
) -> dict:
    """Should leverage be on at all? Independent of direction.

    Crowded short-vol positioning and very low index correlation both mean index
    vol can reprice fast from a calm level; negative dealer gamma amplifies moves
    already under way. Any of them makes leverage expensive to hold even if the
    directional view proves right.
    """
    reasons, checks = [], []
    if vix_cot:
        p = vix_cot["pct_all"]
        hit = p <= 10
        checks.append(("VIX futures positioning", f"{ordinal(p)} pct, {vix_cot['net_oi']:+.1f}% OI", hit))
        if hit:
            reasons.append("short-vol positioning is still crowded")
    c = clean(spx_corr)
    if len(c) >= 252:
        p = pct_rank(c.iloc[-1260:])
        hit = p <= 10
        checks.append(("S&P 60D avg correlation", f"{c.iloc[-1]:.2f}, {ordinal(p)} pct of 5Y", hit))
        if hit:
            reasons.append("index correlation is near its lows, so index vol understates single-name risk")
    g = _last(gamma)
    if not np.isnan(g):
        hit = g < 0
        checks.append(("Dealer gamma (est.)", "negative" if hit else "positive", hit))
        if hit:
            reasons.append("dealers are short gamma and will amplify moves")
    v = _last(vix)
    if not np.isnan(v):
        hit = v >= 30
        checks.append(("VIX level", f"{v:.1f}", hit))
        if hit:
            reasons.append("implied vol is already high, so option buyers pay up")
    n = sum(1 for _, _, h in checks if h)
    state, tone = ("NO LEVERAGE", "bear") if n >= 2 else ("REDUCED", "warm") if n == 1 else ("OPEN", "bull")
    return {"state": state, "tone": tone, "checks": checks, "reasons": reasons, "hits": n}


def rotation_stage(prices: pd.DataFrame) -> dict:
    """Where the rate-sensitive laggards trade sits versus tech leadership."""
    rs = ratio_regime(px(prices, "RSP"), px(prices, "SPY"))
    if not rs:
        return {}
    laggards = [t for t in ("IWM", "RSP", "XLU", "XLF", "XLV", "ITB") if t in prices.columns]
    leaders = [t for t in ("XLK", "SMH") if t in prices.columns]
    lag, lead = basket_index(prices, laggards), basket_index(prices, leaders)
    df = pd.concat([lag.rename("lag"), lead.rename("lead")], axis=1).dropna()
    if len(df) < 300:
        return {}
    spread = (df["lag"].pct_change(20, fill_method=None) - df["lead"].pct_change(20, fill_method=None)).dropna() * 100
    z = float((spread.iloc[-1] - spread.mean()) / spread.std())
    sp_5 = float(spread.iloc[-1] - spread.iloc[-6])
    xlv = relative_extension(px(prices, "XLV"), px(prices, "SPY"), 100)
    stretched = z >= 2.0 or (xlv and xlv["z"] >= 2.5)
    if stretched and sp_5 < 0:
        stage, tone, read = "ROLLING OVER", "warm", "Laggards are stretched and starting to give back. Leadership may be turning."
    elif stretched:
        stage, tone, read = "STRETCHED", "warm", "The rotation trade is extended. New entries carry mean-reversion risk."
    elif spread.iloc[-1] > 0 and rs["above_20"]:
        stage, tone, read = "ACTIVE", "bull", "Laggards lead and the equal-weight ratio is above trend."
    elif rs["pct_3y"] <= 20 and rs["above_20"]:
        stage, tone, read = "EARLY", "bull", "Equal-weight is turning up from a depressed level."
    elif spread.iloc[-1] < 0 and not rs["above_50"]:
        stage, tone, read = "TECH-LED", "neutral", "No rotation: leadership stays narrow."
    else:
        stage, tone, read = "MIXED", "neutral", "Signals disagree."
    return {
        "stage": stage, "tone": tone, "read": read, "spread_20d": float(spread.iloc[-1]), "spread_z": z,
        "rsp_spy": rs, "xlv": xlv, "spread": spread,
    }
