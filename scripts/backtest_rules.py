"""
Test the decision rules on history. Run from the repo root:

    python -m scripts.backtest_rules

Writes .claude/plan/rule-backtest-results.md. The stock panel is cached in cache/.

PASS / FAIL CRITERIA, FIXED BEFORE ANY RESULT WAS SEEN
------------------------------------------------------
The thresholds were chosen from 2026 commentary and reviewed on 2017-2026 data,
so 2003-2016 is genuinely out of sample for the five long-history core rules.

Cap rule ("washed" = 3+ of 5 core rules; lever to 130% only if allowed):
  C1 Risk       Capped max drawdown is shallower than uncapped in both periods.
  C2 Efficiency Capped Sharpe is no more than 0.02 below uncapped, out of sample.
  C3 Skill      On washed days, forward 21-day SPY return is lower when the cap
                blocks leverage than when it allows it. Without this, the cap is
                only "less leverage", which any rule could deliver.
  The cap passes on C1 and C2. C3 decides if it adds information.

Thresholds:
  T1 Edge       3+ of 5 core rules beat the base rate on QQQ out of sample at 21 or
                63 sessions, with p <= 0.10.
  T2 Plateau    A rule's edge at its threshold keeps its sign at both neighbouring
                grid values, and isn't more than double their average.
  T3 Stability  A rule's 63-session edge has the same sign in both periods.

Nothing is re-tuned from these results here. A failure is reported as a failure.
"""
from __future__ import annotations

import os
import pickle
import sys

import numpy as np
import pandas as pd
import yfinance as yf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import analytics as A  # noqa: E402
from src import backtest as B  # noqa: E402
from src import data as D  # noqa: E402
from src import indicators as I  # noqa: E402
from src import research as R  # noqa: E402
from src import research_data as RD  # noqa: E402

# The count as it stood when the criteria were fixed. SKEW has since left the live
# core count, but the pre-set verdict stays on the rules it was set for.
CORE5 = ["qqq_rsi", "vxn", "sox_flush", "vix_spike", "skew"]
LIVE_LONG = [r for r in R.CORE_RULES if r in CORE5]
PANEL_CACHE = os.path.join("cache", "backtest_panel.pkl")
REPORT = os.path.join(".claude", "plan", "rule-backtest-results.md")
GRID = {
    "qqq_rsi": ([24, 26, 28, 30, 32, 34, 36, 38], 32),
    "vxn": ([25, 27.5, 30, 32.5, 35, 40], 30),
    "sox_flush": ([-10, -12.5, -15, -17.5, -20], -15),
    "vix_spike": ([20, 25, 30, 35, 40, 50], 30),
    "skew": ([0.90, 0.95, 0.97, 0.985, 0.995], 0.985),
}


def long_panel() -> pd.DataFrame:
    if os.path.exists(PANEL_CACHE):
        return pd.read_pickle(PANEL_CACHE)
    tickers = D.sp500_tickers()
    df = yf.download(tickers, start="2000-01-01", auto_adjust=True, progress=False, threads=True)["Close"]
    df = df.dropna(how="all", axis=1).dropna(how="all", axis=0)
    df.to_pickle(PANEL_CACHE)
    return df


def md_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_no data_\n"
    head = "| " + " | ".join(map(str, df.columns)) + " |\n|" + "---|" * len(df.columns) + "\n"
    rows = "".join("| " + " | ".join(map(str, r)) + " |\n" for r in df.itertuples(index=False))
    return head + rows


def yn(v: bool) -> str:
    return "PASS" if v else "FAIL"


class Report:
    def __init__(self) -> None:
        self.parts: list[str] = []

    def add(self, text: str) -> None:
        self.parts.append(text)
        print(text)


def load() -> dict:
    raw = I.build_raw()
    b = RD.build_research()
    panel = long_panel()
    P = b.prices.loc["1999-01-01":]
    ai_cols = [c for c in RD.AI_BASKET if c in P.columns]
    return {
        "raw": raw, "b": b, "panel": panel, "P": P,
        "corr60": R.avg_pairwise_corr(panel.loc["2000-01-01":], 60, min_coverage=0.6),
        "ai_corr": R.avg_pairwise_corr(P[ai_cols].loc["2003-01-01":], 60, min_coverage=0.4),
        "cash": D._fred("DTB3", start="1990-01-01"),
    }


def cap_strategies(washed: pd.Series, hits: pd.Series, conf: pd.Series, *, gate_hits: int = 2, min_confirm: int = 3) -> dict:
    idx = washed.index
    ok_gate = hits.reindex(idx).fillna(0) < gate_hits
    ok_conf = conf.reindex(idx).fillna(0) >= min_confirm
    lever = lambda mask: pd.Series(np.where(mask, 1.3, 1.0), index=idx)  # noqa: E731
    return {
        "Never lever": lever(pd.Series(False, index=idx)),
        "Uncapped": lever(washed),
        "Capped (live rule)": lever(washed & ok_gate & ok_conf),
        "Gate only": lever(washed & ok_gate),
        "Confirmation only": lever(washed & ok_conf),
    }


def perf_rows(strategies: dict, asset: pd.Series, cash: pd.Series, period: B.Period) -> tuple[pd.DataFrame, dict]:
    rows, rets = [], {}
    for name, w in strategies.items():
        r = period.clip(B.strategy_returns(w, asset, cash))
        rets[name] = r
        p = B.performance(r, cash)
        lev = float((period.clip(w) > 1).mean() * 100)
        rows.append({"Strategy": name, "CAGR": f"{p['cagr']:.2f}%", "Vol": f"{p['vol']:.1f}%", "Sharpe": f"{p['sharpe']:.3f}",
                     "Max DD": f"{p['max_dd']:.1f}%", "Worst 21D": f"{p['worst_21d']:.1f}%", "Days levered": f"{lev:.1f}%"})
    return pd.DataFrame(rows), rets


def test_cap(d: dict, rep: Report) -> None:
    P, cash, b, raw = d["P"], d["cash"], d["b"], d["raw"]
    hist = R.capitulation_history(P)
    count = hist[CORE5].sum(axis=1)
    conf = B.confirmation_history(P, d["ai_corr"])
    conf_n = conf.sum(axis=1)
    gate = B.gate_history(index=hist.index, vix_cot=b.cot["VIX"], spx_corr=d["corr60"],
                          gamma=raw.series.get("gamma_exposure"), vix=R.px(P, "^VIX"))
    washed = count >= 3
    d.update(count=count, conf_n=conf_n, gate=gate, hist=hist)

    rep.add("\n## Test 1: the leverage cap\n")
    rep.add("Washed = 3+ of the 5 long-history core rules. Normal exposure is 100%, levered exposure 130%, "
            "with borrowing at T-bills + 0.5%. Positions are set at the close and earn the next day.\n")
    rep.add(f"Gate inputs by start date: VIX COT from {b.cot['VIX'].dropna().index[0]:%Y} (usable after 2 years), "
            f"S&P correlation from {d['corr60'].index[0]:%Y}, gamma from {R.clean(raw.series.get('gamma_exposure')).index[0]:%Y}, "
            "so the gate is weaker early in the out-of-sample window.\n")
    strategies = cap_strategies(washed, gate["hits"], conf_n)
    verdict = {}
    spy = R.px(P, "SPY")
    for period in (B.OUT_OF_SAMPLE, B.IN_SAMPLE):
        table, rets = perf_rows(strategies, spy, cash, period)
        rep.add(f"### SPY, {period.name}\n\n{md_table(table)}")
        verdict[period.name] = {k: B.performance(v, cash) for k, v in rets.items()}
        for x, y in (("Capped (live rule)", "Uncapped"), ("Capped (live rule)", "Never lever")):
            bt = B.bootstrap_mean_diff(rets[x], rets[y])
            rep.add(f"- {x} minus {y}: {bt['diff']:+.2f}% a year, 90% CI {bt['lo']:+.2f} to {bt['hi']:+.2f}")
        rep.add("")
    table, _ = perf_rows(strategies, R.px(P, "QQQ"), cash, B.OUT_OF_SAMPLE)
    rep.add(f"### QQQ, {B.OUT_OF_SAMPLE.name}\n\n{md_table(table)}")

    fwd = (spy.shift(-21) / spy - 1) * 100
    perm = B.permission_history(gate["hits"], conf_n)
    rows = []
    c3 = {}
    for period in (B.OUT_OF_SAMPLE, B.IN_SAMPLE):
        w = period.clip(washed[washed])
        eps = B.episode_ids(washed).reindex(w.index)
        g = B.group_difference(fwd.reindex(w.index), perm.reindex(w.index), eps)
        c3[period.name] = g
        if g:
            rows.append({"Period": period.name, "Allowed days": g["yes_days"], "Blocked days": g["no_days"], "Episodes": g["episodes"],
                         "Allowed 21D": f"{g['yes_mean']:+.2f}%", "Blocked 21D": f"{g['no_mean']:+.2f}%",
                         "Blocked minus allowed": f"{-g['diff']:+.2f}pp", "90% CI": f"{-g['hi']:+.2f} to {-g['lo']:+.2f}"})
    rep.add(f"### Does the cap block the right days?\n\nForward 21-day SPY return on washed days only.\n\n{md_table(pd.DataFrame(rows))}")

    oos, ins = verdict[B.OUT_OF_SAMPLE.name], verdict[B.IN_SAMPLE.name]
    cap, unc = "Capped (live rule)", "Uncapped"
    c1 = oos[cap]["max_dd"] > oos[unc]["max_dd"] and ins[cap]["max_dd"] > ins[unc]["max_dd"]
    c2 = oos[cap]["sharpe"] >= oos[unc]["sharpe"] - 0.02
    g = c3[B.OUT_OF_SAMPLE.name]
    c3_ok = bool(g) and g["no_mean"] < g["yes_mean"]
    c3_strong = bool(g) and g["lo"] > 0
    rep.add("### Verdict on the pre-set criteria\n")
    rep.add(f"- C1 risk: **{yn(c1)}** (max DD {oos[cap]['max_dd']:.1f}% vs {oos[unc]['max_dd']:.1f}% out of sample, {ins[cap]['max_dd']:.1f}% vs {ins[unc]['max_dd']:.1f}% in sample)")
    rep.add(f"- C2 efficiency: **{yn(c2)}** (Sharpe {oos[cap]['sharpe']:.3f} vs {oos[unc]['sharpe']:.3f})")
    rep.add(f"- C3 skill: **{yn(c3_ok)}**, {'with' if c3_strong else 'without'} statistical support\n")

    # Sensitivity of the cutoffs
    rows = []
    for period in (B.OUT_OF_SAMPLE, B.IN_SAMPLE):
        base = B.performance(period.clip(B.strategy_returns(strategies["Uncapped"], spy, cash)), cash)
        for label, kw in (("Confirm 2+", dict(min_confirm=2)), ("Confirm 3+ (live)", dict(min_confirm=3)), ("Confirm 4+", dict(min_confirm=4)),
                          ("Gate closes at 1 hit", dict(gate_hits=1)), ("Gate closes at 2 (live)", dict(gate_hits=2)), ("Gate closes at 3", dict(gate_hits=3))):
            w = cap_strategies(washed, gate["hits"], conf_n, **kw)["Capped (live rule)"]
            p = B.performance(period.clip(B.strategy_returns(w, spy, cash)), cash)
            rows.append({"Period": period.name.split(" ")[0], "Setting": label, "Sharpe vs uncapped": f"{p['sharpe'] - base['sharpe']:+.3f}",
                         "Max DD vs uncapped": f"{p['max_dd'] - base['max_dd']:+.1f}pp", "Days levered": f"{(period.clip(w) > 1).mean() * 100:.1f}%"})
    rep.add(f"### Cutoff sensitivity\n\n{md_table(pd.DataFrame(rows))}")

    # Gate as a risk control, regardless of direction
    ret = spy.pct_change(fill_method=None)
    fwd_vol = ret.rolling(21).std().shift(-21) * np.sqrt(252) * 100
    fwd_min = pd.concat([(spy.shift(-i) / spy - 1) * 100 for i in range(1, 22)], axis=1).min(axis=1)
    rows = []
    for label, mask in (("Open (0 hits)", gate["hits"] == 0), ("Partly closed (1)", gate["hits"] == 1), ("Closed (2+)", gate["hits"] >= 2)):
        for period in (B.OUT_OF_SAMPLE, B.IN_SAMPLE):
            m = period.clip(mask.reindex(spy.index).fillna(False))
            days = m[m].index
            if len(days):
                rows.append({"Period": period.name.split(" ")[0], "Gate": label, "Days": len(days), "Next 21D vol": f"{fwd_vol.reindex(days).mean():.1f}%",
                             "Next 21D return": f"{fwd.reindex(days).mean():+.2f}%", "Avg worst close": f"{fwd_min.reindex(days).mean():+.2f}%"})
    rep.add(f"### Does a closed gate mark riskier conditions?\n\n{md_table(pd.DataFrame(rows))}")


def test_real_model(d: dict, rep: Report) -> None:
    raw, P, cash = d["raw"], d["P"], d["cash"]
    per = I.per_indicator_history(raw)
    sc = I.score_indicators(raw, per_indicator=per)
    pillars = I.historical_pillar_scores(raw, per_indicator=per)
    comp = I.historical_composite(raw, per_indicator=per)
    meta = sc.set_index("key").drop(columns=[c for c in ("score",) if c in sc.columns])
    hist = B.model_exposure_history(per, meta, comp, I.exposure_recommendation, I.cluster_signal)
    rep.add("\n## Test 2: the real composite model\n")
    if hist.empty:
        rep.add("_No dates with 80% of the model populated._\n")
        return
    perm = B.permission_history(d["gate"]["hits"], d["conf_n"]).reindex(hist.index).fillna(False)
    unc = (hist["net_pct"] / 100.0)
    cap = unc.where(perm | (unc <= 1.0), 1.0)
    spy = R.px(P, "SPY")
    period = B.Period(f"{hist.index[0]:%b %Y}–{hist.index[-1]:%b %Y}", str(hist.index[0].date()), str(hist.index[-1].date()))
    table, rets = perf_rows({"Card as it was": unc, "Card with cap": cap, "Flat 100%": pd.Series(1.0, index=unc.index)}, spy, cash, period)
    rep.add(f"The window starts once 80% of the model's indicator weight has a year of history. The ICE credit spreads "
            f"join later still, because FRED serves only three years of them. All of it overlaps the rule-building era, "
            f"so treat it as a check that the logic runs on real scores, not as evidence.\n\n### SPY, {period.name}\n\n{md_table(table)}")
    above = unc > 1.0
    rep.add(f"- Sessions above 100%: {int(above.sum())} of {len(unc)}; capped on {int((above & (cap <= 1.0)).sum())}. "
            f"Bottom-cluster sessions: {int(hist['label'].str.startswith('BOTTOM').sum())}.")
    bt = B.bootstrap_mean_diff(rets["Card with cap"], rets["Card as it was"])
    if bt:
        rep.add(f"- Cap minus no cap: {bt['diff']:+.2f}% a year, 90% CI {bt['lo']:+.2f} to {bt['hi']:+.2f}\n")

    # How often the cluster override is on, and what followed. Not a test: the
    # window is short, the days overlap, and it sits inside the rule-building era.
    side = hist["side"].str.upper().replace("NONE", "none")
    rows = []
    for name in ("TOP", "BOTTOM", "none"):
        on = side == name
        if not on.any():
            rows.append({"Cluster": name, "Sessions": 0, "Share": "0.0%", "Episodes": 0, "Avg net": "n/a", "Next 21D": "n/a", "Next 63D": "n/a"})
            continue
        days = on[on].index
        f21, f63 = (spy.shift(-21) / spy - 1) * 100, (spy.shift(-63) / spy - 1) * 100
        rows.append({"Cluster": name, "Sessions": int(on.sum()), "Share": f"{on.mean() * 100:.1f}%",
                     "Episodes": int((on.astype(int).diff() == 1).sum() + int(on.iloc[0])),
                     "Avg net": f"{hist.loc[days, 'net_pct'].mean():.0f}%",
                     "Next 21D": f"{f21.reindex(days).mean():+.2f}%", "Next 63D": f"{f63.reindex(days).mean():+.2f}%"})
    rep.add("### How often the cluster override is on\n\n"
            "A cluster needs the themes at one extreme to lead the other side by a third of the live themes. "
            "Under the earlier rule (3 themes at one extreme, the other side ignored) a cluster was on almost every "
            "session, most of them TOP CLUSTER at about 20 points less exposure. A bottom cluster adds "
            f"{I.BOTTOM_CLUSTER_ADD} points. A top cluster "
            + (f"cuts {I.TOP_CLUSTER_CUT} points" if I.TOP_CLUSTER_CUT else
               "is a flag and changes nothing, because nothing here shows it is followed by weak returns")
            + ". Forward SPY returns are descriptive only: overlapping days in a short window inside the "
            "rule-building era.\n\n" + md_table(pd.DataFrame(rows)))


def edge_row(flag: pd.Series, target: pd.Series, period: B.Period, gap: int = 10) -> dict:
    t = period.clip(target)
    ev = R.fresh(period.clip(flag.reindex(target.index).fillna(False)), gap)
    if len(ev) == 0:
        return {"n": 0}
    s = R.event_study(t, ev, (21, 63))
    if s.empty or "mean" not in s.columns:
        return {"n": 0}
    s = s.set_index("horizon")
    return {"n": int(s.loc[63, "n"]),
            "e21": float(s.loc[21, "mean"] - s.loc[21, "base_mean"]), "p21": float(s.loc[21, "p_mean"]),
            "e63": float(s.loc[63, "mean"] - s.loc[63, "base_mean"]), "p63": float(s.loc[63, "p_mean"]),
            "h63": float(s.loc[63, "hit"]), "b63": float(s.loc[63, "base_hit"]), "mae": float(s.loc[21, "mae"])}


def test_thresholds(d: dict, rep: Report) -> None:
    P, count = d["P"], d["count"]
    qqq = R.px(P, "QQQ")
    rep.add("\n## Test 3: the capitulation thresholds\n### How many rules? QQQ after the count is reached\n")
    rows, t1 = [], False
    for period in (B.OUT_OF_SAMPLE, B.IN_SAMPLE):
        for k in (1, 2, 3, 4, 5):
            e = edge_row(count >= k, qqq, period)
            if e.get("n"):
                rows.append({"Period": period.name.split(" ")[0], "Rules on": f"{k}+", "Episodes": e["n"], "21D vs base": f"{e['e21']:+.2f}pp", "p": f"{e['p21']:.2f}",
                             "63D vs base": f"{e['e63']:+.2f}pp", "p ": f"{e['p63']:.2f}", "63D hit": f"{e['h63']:.0f}% vs {e['b63']:.0f}%", "Avg drawdown 21D": f"{e['mae']:+.1f}%"})
                if period is B.OUT_OF_SAMPLE and k == 3:
                    t1 = (e["e21"] > 0 and e["p21"] <= 0.10) or (e["e63"] > 0 and e["p63"] <= 0.10)
    rep.add(md_table(pd.DataFrame(rows)))
    rep.add(f"- T1 edge at 3+ rules out of sample: **{yn(t1)}**\n")

    rep.add("### Each threshold and its neighbours, QQQ 63-session edge vs base\n")
    summary = []
    for rule, (grid, chosen) in GRID.items():
        rows, edges = [], {}
        for thr in grid:
            flag = B.rule_flag(P, rule, thr)
            a, c = edge_row(flag, qqq, B.OUT_OF_SAMPLE), edge_row(flag, qqq, B.IN_SAMPLE)
            edges[thr] = (a.get("e63", np.nan), c.get("e63", np.nan))
            rows.append({"Threshold": f"{thr}{' ←' if thr == chosen else ''}", "OOS events": a.get("n", 0), "OOS edge": f"{a.get('e63', np.nan):+.2f}pp", "OOS p": f"{a.get('p63', np.nan):.2f}",
                         "In-sample events": c.get("n", 0), "In-sample edge": f"{c.get('e63', np.nan):+.2f}pp", "In-sample p": f"{c.get('p63', np.nan):.2f}"})
        i = grid.index(chosen)
        near = [edges[grid[j]][0] for j in (i - 1, i + 1) if 0 <= j < len(grid)]
        me = edges[chosen][0]
        same = all(np.sign(x) == np.sign(me) for x in near if not np.isnan(x))
        avg = float(np.nanmean(near)) if near else np.nan
        not_peak = abs(me) <= 2 * abs(avg) if avg else False
        t2 = bool(same and not_peak)
        t3 = bool(np.sign(edges[chosen][0]) == np.sign(edges[chosen][1]))
        rep.add(f"**{rule}**\n\n{md_table(pd.DataFrame(rows))}")
        summary.append({"Rule": rule, "Threshold": chosen, "T2 plateau": yn(t2), "T3 stable": yn(t3), "OOS edge": f"{edges[chosen][0]:+.2f}pp", "In-sample edge": f"{edges[chosen][1]:+.2f}pp"})
    rep.add(f"### Threshold verdict\n\n{md_table(pd.DataFrame(summary))}")
    vxn = R.px(P, "^VXN")
    share = {y: float((vxn.loc[str(y)] >= 30).mean() * 100) for y in (2002, 2003, 2008, 2009, 2017, 2019, 2023, 2025)}
    rep.add("Share of days with VXN at 30 or above: " + ", ".join(f"{y} {v:.0f}%" for y, v in share.items()) + "\n")

    rep.add("### Rules that lack out-of-sample history (2017–2026 only)\n")
    rows = []
    for rule, grid, chosen in (("momentum", [1, 2, 5, 10], 2), ("hf_unwind", [1, 2, 5, 10], 2)):
        for thr in grid:
            e = edge_row(B.rule_flag(P, rule, thr), qqq, B.IN_SAMPLE)
            rows.append({"Rule": rule, "Pct": f"{thr}{' ←' if thr == chosen else ''}", "Events": e.get("n", 0), "21D edge": f"{e.get('e21', np.nan):+.2f}pp", "63D edge": f"{e.get('e63', np.nan):+.2f}pp", "p": f"{e.get('p63', np.nan):.2f}"})
    rep.add(md_table(pd.DataFrame(rows)))


def test_guardrails(d: dict, rep: Report) -> None:
    P, panel, raw = d["P"], d["panel"], d["raw"]
    spy = R.px(P, "SPY")
    tb = A.trend_breadth(panel, 10, 20)["pct"]
    pct200 = R.internals_frame(panel)["pct_200"]
    baa = D._fred("BAA10Y", start="1990-01-01")
    nfci = D._fred("NFCI", start="1990-01-01")
    idx = spy.index
    fill = lambda s: R.clean(s).reindex(idx, method="ffill", limit=7)  # noqa: E731
    broken = pd.DataFrame({
        "breadth": fill(tb) < 50, "pct200": fill(pct200) < 40,
        "credit": (fill(baa) - fill(baa).shift(20)) * 100 >= 15,
        "corr": (fill(d["corr60"]) - fill(d["corr60"]).shift(20)) >= 0.15,
        "nfci": fill(nfci) >= 0,
    })
    n = broken.sum(axis=1)
    state = pd.Series("WATCH", index=idx)
    state[n == 0] = "INTACT"
    state[(n >= 3) | (broken["credit"] & (n >= 2))] = "BREAKING"
    fwd = (spy.shift(-63) / spy - 1) * 100
    fwd_min = pd.concat([(spy.shift(-i) / spy - 1) * 100 for i in range(1, 64, 3)], axis=1).min(axis=1)
    rep.add("\n## Test 4: the guardrails\n")
    rep.add("Long-history stand-ins: Baa–10Y spread for credit, and a stock panel of current S&P members for breadth "
            "and correlation (survivorship bias flatters breadth). CCC and funding checks have too little history to test.\n")
    rows = []
    for period in (B.OUT_OF_SAMPLE, B.IN_SAMPLE):
        for s in ("INTACT", "WATCH", "BREAKING"):
            days = period.clip(state[state == s]).index
            if len(days):
                rows.append({"Period": period.name.split(" ")[0], "State": s, "Days": len(days), "Next 63D return": f"{fwd.reindex(days).mean():+.2f}%",
                             "Hit": f"{(fwd.reindex(days).dropna() > 0).mean() * 100:.0f}%", "Avg worst close": f"{fwd_min.reindex(days).mean():+.2f}%"})
    rep.add(md_table(pd.DataFrame(rows)))
    washed = (d["count"] >= 3).reindex(idx).fillna(False)
    rows = []
    for period in (B.OUT_OF_SAMPLE, B.IN_SAMPLE):
        for s, mask in (("Washed + not breaking", state != "BREAKING"), ("Washed + breaking", state == "BREAKING")):
            days = period.clip((washed & mask).pipe(lambda x: x[x])).index
            if len(days):
                rows.append({"Period": period.name.split(" ")[0], "State": s, "Days": len(days), "Episodes": int(R.fresh(washed & mask, 10).isin(days).sum()),
                             "Next 63D return": f"{fwd.reindex(days).mean():+.2f}%", "Avg worst close": f"{fwd_min.reindex(days).mean():+.2f}%"})
    rep.add(f"### Does a broken system spoil a washout?\n\n{md_table(pd.DataFrame(rows))}")


def test_rotation(d: dict, rep: Report) -> None:
    P = d["P"]
    lag = R.basket_index(P, ["IWM", "RSP", "XLU", "XLF", "XLV", "ITB"])
    lead = R.basket_index(P, ["XLK", "SMH"])
    df = pd.concat([lag.rename("a"), lead.rename("b")], axis=1).dropna()
    spread = (df["a"].pct_change(20, fill_method=None) - df["b"].pct_change(20, fill_method=None)).dropna() * 100
    z = (spread - spread.expanding(252).mean()) / spread.expanding(252).std()
    ratio = (df["a"] / df["b"]).rename("Laggards/Tech")
    rep.add("\n## Test 5: the rotation STRETCHED flag\n\nIf laggards are stretched against tech, does the trade mean-revert? Target is the laggards/tech ratio.\n")
    rows = []
    for period in (B.OUT_OF_SAMPLE, B.IN_SAMPLE):
        e = edge_row(z >= 2.0, ratio, period, gap=20)
        if e.get("n"):
            rows.append({"Period": period.name.split(" ")[0], "Events": e["n"], "21D vs base": f"{e['e21']:+.2f}pp", "p": f"{e['p21']:.2f}", "63D vs base": f"{e['e63']:+.2f}pp", "p ": f"{e['p63']:.2f}"})
    rep.add(md_table(pd.DataFrame(rows)))


def test_post_hoc(d: dict, rep: Report) -> None:
    """Questions raised BY the results above. These can't change a pre-set verdict."""
    P, cash = d["P"], d["cash"]
    spy, qqq = R.px(P, "SPY"), R.px(P, "QQQ")
    count, gate, conf_n = d["count"], d["gate"], d["conf_n"]
    rep.add("\n## Post-hoc checks\n\nAdded after seeing the results, so treat these as explanation, not as tests.\n")

    rows = []
    fwd21, fwd63 = (qqq.shift(-21) / qqq - 1) * 100, (qqq.shift(-63) / qqq - 1) * 100
    for period in (B.OUT_OF_SAMPLE, B.IN_SAMPLE):
        for ev in R.fresh(period.clip(count >= 3), 10):
            rows.append({"Episode": f"{ev:%d %b %Y}", "Rules on": int(count.loc[ev]), "QQQ next 21D": f"{fwd21.get(ev, np.nan):+.1f}%", "QQQ next 63D": f"{fwd63.get(ev, np.nan):+.1f}%"})
    rep.add(f"### Every 3+ episode\n\n{md_table(pd.DataFrame(rows))}")

    gfc = (qqq.index >= "2008-09-01") & (qqq.index <= "2009-03-31")
    no_gfc = qqq[~gfc]
    e = edge_row((count >= 3).reindex(no_gfc.index), no_gfc, B.OUT_OF_SAMPLE)
    rep.add(f"- 3+ rules out of sample without Sep 2008–Mar 2009: {e.get('n', 0)} episodes, 21D {e.get('e21', float('nan')):+.2f}pp (p {e.get('p21', float('nan')):.2f}), "
            f"63D {e.get('e63', float('nan')):+.2f}pp (p {e.get('p63', float('nan')):.2f})\n")

    # SKEW left the core count BECAUSE of these results, so this can't be a clean test.
    hist = d["hist"]
    count4 = hist[LIVE_LONG].sum(axis=1)
    rep.add("### The live rule set, without SKEW\n")
    rep.add("Not a clean test: SKEW was removed after seeing 2003–2016, so that window now informed this rule set.\n")
    rows = []
    for period in (B.OUT_OF_SAMPLE, B.IN_SAMPLE):
        for k in (2, 3, 4):
            e = edge_row(count4 >= k, qqq, period)
            if e.get("n"):
                rows.append({"Period": period.name.split(" ")[0], "Rules on": f"{k}+ of {len(LIVE_LONG)}", "Episodes": e["n"], "21D vs base": f"{e['e21']:+.2f}pp", "p": f"{e['p21']:.2f}",
                             "63D vs base": f"{e['e63']:+.2f}pp", "p ": f"{e['p63']:.2f}", "63D hit": f"{e['h63']:.0f}% vs {e['b63']:.0f}%", "Avg drawdown 21D": f"{e['mae']:+.1f}%"})
    rep.add(md_table(pd.DataFrame(rows)))

    # The card's cluster persists for weeks; a five-day washout flag doesn't.
    rep.add("### Cap with a signal that lasts 21 sessions, like the card's cluster\n")
    washed = (count >= 3).astype(float).rolling(21, min_periods=1).max().astype(bool)
    strategies = cap_strategies(washed, gate["hits"], conf_n)
    perm = B.permission_history(gate["hits"], conf_n)
    fwd = (spy.shift(-21) / spy - 1) * 100
    rows = []
    for period in (B.OUT_OF_SAMPLE, B.IN_SAMPLE):
        table, rets = perf_rows(strategies, spy, cash, period)
        rep.add(f"**{period.name}**\n\n{md_table(table)}")
        for x, y in (("Capped (live rule)", "Uncapped"), ("Capped (live rule)", "Never lever")):
            bt = B.bootstrap_mean_diff(rets[x], rets[y])
            rep.add(f"- {x} minus {y}: {bt['diff']:+.2f}% a year, 90% CI {bt['lo']:+.2f} to {bt['hi']:+.2f}")
        rep.add("")
        w = period.clip(washed[washed])
        g = B.group_difference(fwd.reindex(w.index), perm.reindex(w.index), B.episode_ids(washed).reindex(w.index))
        if g:
            rows.append({"Period": period.name.split(" ")[0], "Allowed days": g["yes_days"], "Blocked days": g["no_days"], "Episodes": g["episodes"],
                         "Allowed 21D": f"{g['yes_mean']:+.2f}%", "Blocked 21D": f"{g['no_mean']:+.2f}%", "Blocked minus allowed": f"{-g['diff']:+.2f}pp",
                         "90% CI": f"{-g['hi']:+.2f} to {-g['lo']:+.2f}"})
    rep.add(f"**Does it block the right days?**\n\n{md_table(pd.DataFrame(rows))}")


def main() -> int:
    rep = Report()
    rep.add("# Decision rule backtest\n\nGenerated by `python -m scripts.backtest_rules`. Pass/fail criteria are in that file's "
            "header and were fixed before any result was seen. Historical results, not forecasts.\n")
    d = load()
    print("data loaded", d["P"].shape, d["panel"].shape, flush=True)
    test_cap(d, rep)
    test_real_model(d, rep)
    test_thresholds(d, rep)
    test_guardrails(d, rep)
    test_rotation(d, rep)
    test_post_hoc(d, rep)
    with open(REPORT, "w", encoding="utf-8") as fh:
        fh.write("\n".join(rep.parts))
    print("wrote", REPORT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
