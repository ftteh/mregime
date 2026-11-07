"""
Historical reconstruction of the decision rules, and the tools to judge them.

The live rules in `research.py` answer "what is true today". This module rebuilds
the same rules as daily series so they can be tested. Three things keep the test
honest:

* Parity: `tests/test_backtest.py` checks the last row of each history against
  the live function, so what gets tested is the rule on the page.
* No look-ahead: a position set from data through close t earns the return of
  t+1, percentiles only see their own past, and COT flags wait for publication.
* A control: every leverage rule is compared with "never lever", because a rule
  that merely levers less always looks safer without adding any information.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src import research as R

TRADING_DAYS = 252
FINANCING_SPREAD = 0.005  # annual, over T-bills, on borrowed exposure


# ---------------------------------------------------------------------------
# Rule histories
# ---------------------------------------------------------------------------
def confirmation_history(prices: pd.DataFrame, ai_corr: pd.Series | None = None) -> pd.DataFrame:
    """Daily booleans for the five reversal confirmations (see reversal_confirmation)."""
    qqq, vxn, skew = R.px(prices, "QQQ"), R.px(prices, "^VXN"), R.px(prices, "^SKEW")
    mv, tnx = R.px(prices, "^MOVE"), R.px(prices, "^TNX")
    idx = qqq.index
    out = pd.DataFrame(index=idx)
    if not vxn.empty:
        out["vxn_fade"] = (vxn / vxn.rolling(10).max() - 1 <= -0.10).reindex(idx).fillna(False)
    out["qqq_5dma"] = (qqq > qqq.rolling(5).mean()) & (qqq > qqq.rolling(20).mean())
    if not mv.empty and not tnx.empty:
        a = mv < mv.rolling(20).mean()
        b = tnx < tnx.rolling(20).mean()
        out["move_fade"] = (a.reindex(idx).fillna(False)) & (b.reindex(idx).fillna(False))
    if not skew.empty:
        out["skew_fade"] = (skew - skew.rolling(10).max() <= -5).reindex(idx).fillna(False)
    c = R.clean(ai_corr)
    if len(c) > 30:
        out["ai_corr"] = (c - c.shift(20) < 0).reindex(idx).fillna(False)
    return out.astype(bool)


def published_cot(net_oi: pd.Series) -> pd.Series:
    """Move Tuesday as-of dates to their Friday publication."""
    s = R.clean(net_oi)
    s = s.copy()
    s.index = s.index + pd.Timedelta(days=3)
    return s


def gate_history(
    *,
    index: pd.DatetimeIndex,
    vix_cot: pd.Series | None,
    spx_corr: pd.Series | None,
    gamma: pd.Series | None,
    vix: pd.Series | None,
) -> pd.DataFrame:
    """Daily booleans for the leverage gate's four checks, plus `hits`."""
    out = pd.DataFrame(index=index)
    cot = R.clean(vix_cot)
    if len(cot) > 104:
        rank = R.expanding_rank(cot, 104)
        rank = published_cot(rank)
        out["vix_cot"] = (rank <= 10).reindex(index, method="ffill").fillna(False)
    corr = R.clean(spx_corr)
    if len(corr) > 252:
        rank = R.rolling_rank(corr, 1260, 252)
        out["corr"] = (rank <= 10).reindex(index, method="ffill").fillna(False)
    g = R.clean(gamma)
    if len(g):
        out["gamma"] = (g < 0).reindex(index, method="ffill", limit=5).fillna(False)
    v = R.clean(vix)
    if len(v):
        out["vix"] = (v >= 30).reindex(index, method="ffill", limit=5).fillna(False)
    out = out.astype(bool)
    out["hits"] = out.sum(axis=1)
    return out


def permission_history(hits: pd.Series, confirmations: pd.Series, *, gate_hits: int = 2, min_confirm: int = 3) -> pd.Series:
    """True where leverage is allowed: gate not closed and price has confirmed."""
    df = pd.concat([hits.rename("h"), confirmations.rename("c")], axis=1).dropna()
    return (df["h"] < gate_hits) & (df["c"] >= min_confirm)


def model_exposure_history(per_indicator: pd.DataFrame, meta: pd.DataFrame, composite: pd.Series, recommend, cluster_fn, *, min_coverage: float = 0.8) -> pd.DataFrame:
    """Rebuild the card's exposure by date from the model's own score history.

    `recommend` and `cluster_fn` are the live functions, injected so the history
    can't drift from what the page does. Dates with too little of the model
    populated are skipped.
    """
    cols = [c for c in per_indicator.columns if c in meta.index]
    weights = meta.loc[cols, "weight"].astype(float)
    rows = {}
    for d, row in per_indicator[cols].iterrows():
        if d not in composite.index or np.isnan(composite.loc[d]):
            continue
        live = row.notna()
        covered = float(weights[live].sum() / weights.sum())
        if covered < min_coverage:
            continue
        scored = meta.loc[cols].assign(score=row.values)
        rec = recommend(float(composite.loc[d]), cluster_fn(scored), covered)
        rows[d] = {"net_pct": rec["net_pct"], "override": rec["cluster_override"], "label": rec["label"], "covered": covered}
    return pd.DataFrame.from_dict(rows, orient="index").sort_index()


# ---------------------------------------------------------------------------
# Simple parameterized rules, for threshold sensitivity
# ---------------------------------------------------------------------------
def rule_flag(prices: pd.DataFrame, rule: str, threshold: float) -> pd.Series:
    """The raw condition for one core rule at a chosen threshold (no memory)."""
    qqq, spy = R.px(prices, "QQQ"), R.px(prices, "SPY")
    if rule == "qqq_rsi":
        return R.rsi_wilder(qqq, 14) <= threshold
    if rule == "vxn":
        return R.px(prices, "^VXN") >= threshold
    if rule == "sox_flush":
        return R.px(prices, "^SOX").pct_change(5, fill_method=None) * 100 <= threshold
    if rule == "vix_spike":
        return R.px(prices, "^VIX").pct_change(fill_method=None) * 100 >= threshold
    if rule == "skew":
        s = R.px(prices, "^SKEW")
        return s >= s.rolling(2520, min_periods=1260).quantile(threshold).shift(1)
    if rule in ("momentum", "hf_unwind"):
        a = R.px(prices, "MTUM" if rule == "momentum" else "GVIP")
        rel = (a.pct_change(5, fill_method=None) - spy.pct_change(5, fill_method=None)).dropna() * 100
        return R.expanding_rank(rel, 252) <= threshold
    raise KeyError(rule)


# ---------------------------------------------------------------------------
# Performance
# ---------------------------------------------------------------------------
def strategy_returns(weight: pd.Series, asset: pd.Series, cash_rate: pd.Series) -> pd.Series:
    """Daily return of holding `weight` of the asset, set at the PRIOR close.

    Idle cash earns the T-bill rate; borrowed exposure pays it plus a spread.
    `cash_rate` is annual, in percent.
    """
    ret = R.clean(asset).pct_change(fill_method=None)
    w = weight.reindex(ret.index).ffill().shift(1)
    cash = (cash_rate.reindex(ret.index).ffill().fillna(0.0) / 100.0) / TRADING_DAYS
    borrow = cash + FINANCING_SPREAD / TRADING_DAYS
    out = w * ret + (1 - w).clip(lower=0) * cash - (w - 1).clip(lower=0) * borrow
    return out.dropna()


def performance(r: pd.Series, cash_rate: pd.Series | None = None) -> dict:
    r = r.dropna()
    if len(r) < 20:
        return {}
    eq = (1 + r).cumprod()
    years = len(r) / TRADING_DAYS
    rf = 0.0
    if cash_rate is not None:
        rf = float((cash_rate.reindex(r.index).ffill().fillna(0.0) / 100.0).mean())
    vol = float(r.std() * np.sqrt(TRADING_DAYS))
    ann = float(eq.iloc[-1] ** (1 / years) - 1)
    return {
        "cagr": ann * 100,
        "vol": vol * 100,
        "sharpe": (float(r.mean()) * TRADING_DAYS - rf) / vol if vol else float("nan"),
        "max_dd": float((eq / eq.cummax() - 1).min()) * 100,
        "worst_21d": float(((1 + r).rolling(21).apply(np.prod, raw=True) - 1).min()) * 100,
    }


def bootstrap_mean_diff(a: pd.Series, b: pd.Series, *, block: int = 21, n: int = 2000, seed: int = 11) -> dict:
    """Annualized mean return difference a - b with a circular block bootstrap CI.

    Blocks keep the volatility clustering that an IID resample would destroy, so
    the interval doesn't come out falsely narrow.
    """
    df = pd.concat([a.rename("a"), b.rename("b")], axis=1, join="inner").dropna()
    if len(df) < block * 4:
        return {}
    d = (df["a"] - df["b"]).to_numpy()
    size = len(d)
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, size, size=(n, size // block + 1))
    offsets = np.arange(block)
    sampled = d[(starts[:, :, None] + offsets) % size].reshape(n, -1)[:, :size]
    means = sampled.mean(axis=1) * TRADING_DAYS * 100
    return {"diff": float(d.mean() * TRADING_DAYS * 100), "lo": float(np.quantile(means, 0.05)), "hi": float(np.quantile(means, 0.95))}


def group_difference(forward: pd.Series, group: pd.Series, episodes: pd.Series, *, n: int = 4000, seed: int = 13) -> dict:
    """Mean forward return where `group` is True minus False, resampling whole episodes.

    Days inside one sell-off are not independent, so the unit of resampling is the
    episode, not the day.
    """
    df = pd.concat([forward.rename("f"), group.rename("g"), episodes.rename("e")], axis=1).dropna()
    if df.empty or df["g"].nunique() < 2:
        return {}
    df["g"] = df["g"].astype(bool)
    rng = np.random.default_rng(seed)
    ids = df["e"].unique()
    frames = {e: x for e, x in df.groupby("e")}
    stats = np.empty(n)
    for i in range(n):
        pick = rng.choice(ids, size=len(ids), replace=True)
        s = pd.concat([frames[e] for e in pick])
        y, z = s.loc[s["g"], "f"], s.loc[~s["g"], "f"]
        stats[i] = (y.mean() - z.mean()) if len(y) and len(z) else np.nan
    return {
        "yes_mean": float(df.loc[df["g"], "f"].mean()), "no_mean": float(df.loc[~df["g"], "f"].mean()),
        "yes_days": int(df["g"].sum()), "no_days": int((~df["g"]).sum()), "episodes": int(len(ids)),
        "diff": float(df.loc[df["g"], "f"].mean() - df.loc[~df["g"], "f"].mean()),
        "lo": float(np.nanquantile(stats, 0.05)), "hi": float(np.nanquantile(stats, 0.95)),
    }


def episode_ids(flag: pd.Series, gap: int = 10) -> pd.Series:
    """Label contiguous runs of True, merging runs separated by under `gap` sessions."""
    f = flag.fillna(False).astype(bool)
    ids = pd.Series(np.nan, index=f.index)
    current, last = 0, None
    for i, on in enumerate(f.to_numpy()):
        if on:
            if last is None or i - last > gap:
                current += 1
            last = i
            ids.iloc[i] = current
    return ids


@dataclass(frozen=True)
class Period:
    name: str
    start: str
    end: str

    def clip(self, obj):
        return obj.loc[self.start:self.end]


OUT_OF_SAMPLE = Period("2003–2016 (out of sample)", "2003-01-01", "2016-12-31")
IN_SAMPLE = Period("2017–2026 (rule-building era)", "2017-01-01", "2026-12-31")
