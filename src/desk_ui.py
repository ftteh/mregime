"""
Streamlit rendering for the decision desk and research views.

`render_decision_desk` is the at-a-glance block. `render_research` is a view
switcher wrapped in a fragment: only the selected view builds its charts, and
changing view reruns just this section rather than the whole dashboard.
"""
from __future__ import annotations

import html

import numpy as np
import pandas as pd
import streamlit as st

from src import research as R
from src import research_charts as RC
from src import research_data as RD
from src.charts import GREEN_ZONE, RED_ZONE
from src.research import Rule

TONE = {"bull": "#2ecc8f", "bear": "#e8615f", "warm": "#e0a84a", "neutral": "#9aa4ad"}
MARK = {"on": "●", "near": "◐", "off": "○", "na": "–"}
MARK_COL = {"on": "#f2f4f5", "near": "#c9d1d9", "off": "#6b7580", "na": "#555e68"}

TARGET_NAMES = {"^GSPC": "S&P 500", "^SOX": "SOX", "^VIX": "VIX", "RB=F": "Gasoline"}

VALUE_FMT = {
    "qqq_rsi": "{:.1f}", "vxn": "{:.1f}", "sox_flush": "{:+.1f}%", "vix_spike": "{:+.0f}%", "skew": "{:.0f}",
    "momentum": "{:+.1f}pp", "hf_unwind": "{:+.1f}pp", "put_call": "{:.2f}", "put_volume": "{:.1f}M",
    "fear_greed": "{:.0f}", "aaii": "{:+.1f}", "ndx_pe": "{:.1f}x", "vxn_fade": "{:+.1f}%", "qqq_5dma": "{:+.1f}%",
    "move_fade": "{:.0f}", "skew_fade": "{:+.1f}", "ai_corr": "{:.2f}", "breadth": "{:.1f}%", "pct200": "{:.1f}%",
    "ig": "{:.0f}bp", "hy": "{:.0f}bp", "contain": "{:.0f}bp", "funding": "{:+.0f}bp", "corr": "{:.2f}", "nfci": "{:+.2f}",
}


def _rule_status(r: Rule) -> str:
    if not r.available:
        return "na"
    return "on" if r.fired else "near" if r.near else "off"


def _rule_rows(rules: list[Rule]) -> str:
    rows = []
    for r in rules:
        status = _rule_status(r)
        val = "n/a" if not r.available else VALUE_FMT.get(r.key, "{:.2f}").format(r.value)
        star = "<span style='color:#6b7580;'> ·core</span>" if r.core else ""
        tip = html.escape(f"{r.threshold}. {r.detail}".strip(". "), quote=True)
        rows.append(
            f"<div title='{tip}' style='display:flex; justify-content:space-between; gap:8px; padding:4px 0; "
            "border-bottom:1px solid rgba(255,255,255,0.05);'>"
            f"<span style='font-size:0.8rem; color:#c9d1d9;'><span style='color:{MARK_COL[status]};'>{MARK[status]}</span> "
            f"{html.escape(r.label)}{star}</span>"
            f"<b style='font-size:0.82rem; color:#f2f4f5; white-space:nowrap;'>{val}</b></div>"
        )
    return "".join(rows)


def _card(title: str, state: str, tone: str, sub: str, body: str) -> str:
    col = TONE.get(tone, TONE["neutral"])
    return (
        f"<div class='card' style='min-height:660px; border-top:3px solid {col};'>"
        f"<div style='color:#9aa4ad; font-size:0.72rem; letter-spacing:1px; font-weight:700;'>{title}</div>"
        f"<div style='color:{col}; font-size:1.3rem; font-weight:800; margin:4px 0 2px 0;'>{state}</div>"
        f"<div class='sub' style='margin-bottom:8px; line-height:1.35;'>{sub}</div>{body}</div>"
    )


def render_decision_desk(d: dict) -> None:
    p, cap, guard, gate = d["posture"], d["cap"], d["guard"], d["gate"]
    col = TONE.get(p["tone"], TONE["neutral"])
    st.markdown("### Decision desk")
    fresh = d["freshness"]
    if fresh["issues"]:
        st.warning("Read this desk with care: " + "; ".join(fresh["issues"]) + ".")
    st.markdown(
        f"<div class='card' style='border-left:4px solid {col};'>"
        f"<div style='font-size:1.1rem; font-weight:800; color:#fff;'>{html.escape(p['head'])}</div>"
        f"<div class='sub' style='font-size:0.92rem; line-height:1.5; margin-top:4px;'>{html.escape(p['body'])} "
        f"<b style='color:#ddd;'>{html.escape(p['gate'])}</b></div></div>",
        unsafe_allow_html=True,
    )
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(_card("1 · IS THE SYSTEM INTACT?", guard["state"], guard["tone"],
                          f"{guard['held']} of {guard['total']} guardrails hold. {guard['read']}",
                          _rule_rows(d["guards"])), unsafe_allow_html=True)
    with c2:
        st.markdown(_card("2 · IS FEAR WASHED OUT?", cap["state"], cap["tone"],
                          f"{cap['core_fired']} of {cap['core_total']} core rules fired, "
                          f"{cap['all_fired']} of {cap['all_total']} overall. Not reliable alone: read with card 1.",
                          _rule_rows(d["rules"])), unsafe_allow_html=True)
    live = [r for r in d["confirm"] if r.available]
    fired = sum(r.fired for r in live)
    with c3:
        tone = "bull" if fired >= 3 else "warm" if fired else "neutral"
        st.markdown(_card("3 · HAS PRICE CONFIRMED?", f"{fired} OF {len(live)}", tone,
                          "The list needs 3 before a washout counts as a turn.",
                          _rule_rows(d["confirm"])), unsafe_allow_html=True)
    with c4:
        rows = "".join(
            "<div style='display:flex; justify-content:space-between; gap:8px; padding:4px 0; "
            "border-bottom:1px solid rgba(255,255,255,0.05);'>"
            f"<span style='font-size:0.8rem; color:#c9d1d9;'><span style='color:{MARK_COL['on' if hit else 'off']};'>"
            f"{MARK['on' if hit else 'off']}</span> {html.escape(name)}</span>"
            f"<b style='font-size:0.82rem; color:#f2f4f5; text-align:right;'>{html.escape(val)}</b></div>"
            for name, val, hit in gate["checks"]
        )
        rot = d["rotation"]
        extra = ""
        if rot:
            xlv = rot.get("xlv") or {}
            xlv_txt = f" XLV 100D relative {xlv['z']:+.1f}σ." if xlv else ""
            extra = (
                "<div style='color:#9aa4ad; font-size:0.72rem; letter-spacing:1px; font-weight:700; margin-top:14px;'>ROTATION</div>"
                f"<div style='color:{TONE.get(rot['tone'], TONE['neutral'])}; font-size:1.05rem; font-weight:800;'>{rot['stage']}</div>"
                f"<div class='sub' style='line-height:1.35;'>{html.escape(rot['read'])} Laggards minus tech, 20D: "
                f"{rot['spread_20d']:+.1f}pp ({rot['spread_z']:+.1f}σ).{xlv_txt}</div>"
            )
        st.markdown(_card("4 · IS LEVERAGE SAFE?", gate["state"], gate["tone"],
                          "Risk control, not a forecast: it needs 2 hits to close.", rows + extra), unsafe_allow_html=True)
    dates = " · ".join(f"{k} {v:%d %b}" for k, v in fresh["as_of"].items())
    st.caption(
        f"Closing data through: {dates}. ● fired or holds · ◐ close · ○ not met · – no data. Hover a row for its threshold. "
        "Rule output from public data, not investment advice. These rules don't change the composite score, but they "
        "limit the exposure card: it stays at or below 100% unless the leverage gate isn't closed and 3 of 5 confirmations are met."
    )


# ---------------------------------------------------------------------------
# Research views
# ---------------------------------------------------------------------------
def _source(b: RD.ResearchBundle, *keys: str) -> str:
    parts = []
    for k in keys:
        p = b.provenance.get(k) or {}
        kind, src = p.get("kind"), p.get("source") or ""
        if kind == "unavailable" or not src:
            parts.append(f"{k}: offline")
            continue
        flag = {"cache": " (cached, live fetch failed)", "fallback": " (⚠ bundled seed, live source unreachable)"}.get(kind, "")
        parts.append(f"{src}{flag}")
    return "Source: " + " · ".join(dict.fromkeys(parts)) + "."


def _pct(v: float) -> str:
    return "—" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.0f}"


def _view_positioning(d: dict, b: RD.ResearchBundle) -> None:
    t = d["cot_table"]
    if t.empty:
        st.info("CFTC positioning unavailable: no live, cached or seed data.")
        return
    st.markdown("**Crowding board.** Where each futures market sits against all history and against the last 3 years.")
    st.plotly_chart(RC.crowding_dumbbell(t), width="stretch")
    show = pd.DataFrame({
        "Market": t["market"], "Trader group": t["group"], "Net / OI": t["net_oi"].map("{:+.1f}%".format),
        "All-time %ile": t["pct_all"].map(_pct), "3Y %ile": t["pct_3y"].map(_pct),
        "3Y z": t["z_3y"].map("{:+.1f}".format), "1W flow": t["flow_1w"].map("{:+.1f}".format),
        "4W flow": t["flow_4w"].map("{:+.1f}".format), "State": t["state"],
        "Note": np.where(t["structural"], "Horizons disagree: likely structural", ""),
    })
    st.dataframe(show, hide_index=True, width="stretch")
    st.caption(
        f"As of {t['as_of'].max():%d %b %Y}; published the following Friday. Flows are changes in net % of OI. "
        "A market marked structural looks extreme against one horizon only. Leveraged funds' UST 10Y short is the main "
        "case: much of it is the cash-futures basis trade, so read the 3Y percentile first. " + _source(b, "cot_positioning")
    )
    labels = {m["label"]: m for m in RD.COT_MARKETS if m["key"] in b.cot.columns}
    pick = st.selectbox("Market", list(labels), index=list(labels).index("VIX futures"), key="desk_cot_market")
    m = labels[pick]
    st.plotly_chart(RC.cot_figure(R.px(b.prices, m["price"]), b.cot[m["key"]], m["label"],
                                  pd.Timestamp.today() - pd.DateOffset(years=10)), width="stretch")
    if not b.qqq_si.empty:
        st.markdown("**QQQ short interest**, millions of shares (Nasdaq-listed only; SPY is not published free).")
        st.plotly_chart(RC.lines([("QQQ short interest", b.qqq_si)], fmt="{:.1f}M", fill_first=True, height=260), width="stretch")
        st.caption("Exchange-reported, twice monthly, about 10 days late. " + _source(b, "qqq_short_interest"))


def _view_rotation(d: dict, b: RD.ResearchBundle, series: dict) -> None:
    P, I = b.prices, b.internals
    rot = d["rotation"]
    if rot:
        rs = rot["rsp_spy"]
        st.markdown(f"**Equal-weight vs cap-weight (RSP/SPY)**: {R.ordinal(rs['pct_all'])} percentile of full history, "
                    f"{R.ordinal(rs['pct_3y'])} of 3 years, {rs['vs_200']:+.1f}% vs its 200-day average.")
        st.plotly_chart(RC.stacked(
            [("RSP / SPY", rs["series"]), ("200D average", rs["ma200"])],
            [("3Y percentile", R.rolling_rank(rs["series"], 756, 252))],
            start=pd.Timestamp.today() - pd.DateOffset(years=8), bottom_title="Rolling 3-year percentile",
            bottom_zones=[(0, 10, GREEN_ZONE), (90, 100, RED_ZONE)], bottom_range=(0, 100), fmt="{:.4f}", bottom_fmt="{:.0f}",
        ), width="stretch")
    c1, c2 = st.columns([1, 1])
    with c1:
        days = st.radio("Scoreboard window, sessions", [5, 10, 21, 63], index=1, horizontal=True, key="desk_rot_days")
        sb = R.scoreboard(P, RD.ROTATION_NAMES, days=days)
        if not sb.empty:
            st.plotly_chart(RC.hbars(sb, "name", "ret", title=f"Cross-asset return, last {days} sessions"), width="stretch")
    with c2:
        st.write("")
        st.write("")
        st.write("")
        sec = R.scoreboard(P, RD.SECTOR_NAMES, days=days)
        if not sec.empty:
            up = int((sec["ret"] > 0).sum())
            beat = int((sec["vs_bench"] > 0).sum())
            st.plotly_chart(RC.hbars(sec, "name", "ret", title=f"S&P sectors: {up} of {len(sec)} up, {beat} beat SPY"), width="stretch")
    m = R.monthly_spread(R.px(P, "RSP"), R.px(P, "XLK"))
    snap = R.monthly_spread_snapshot(m)
    if snap:
        st.markdown(f"**RSP minus XLK by month.** {snap['month']:%b %Y}: {snap['spread']:+.2f}pp, rank {snap['rank']} of {snap['n']}.")
        st.plotly_chart(RC.signed_bars(m["spread"], start=pd.Timestamp.today() - pd.DateOffset(years=10),
                                       levels=[(snap["p95"], "95th pct"), (snap["p99"], "99th pct")]), width="stretch")
        st.caption("The latest bar is the month to date until the month closes.")
    if not I.empty:
        parts = [(f"% above {n}DMA", I[f"pct_{n}"]) for n in (20, 50, 200) if f"pct_{n}" in I]
        dis = R.disconnect_snapshot(I, d["spx"])
        if dis:
            st.markdown(f"**Participation monitor.** {dis['read']}: {dis['chg_pp']:+.1f}pp on the 50DMA measure "
                        f"vs S&P {dis['index_ret']:+.1f}% over {dis['days']} sessions.")
        st.plotly_chart(RC.lines(parts, start=pd.Timestamp.today() - pd.DateOffset(years=2), fmt="{:.0f}%",
                                 levels=[(50, "50%")]), width="stretch")
        div = R.divergence_days(I, d["spx"])
        st.markdown(f"**Index down, members up.** {len(div)} sessions in 5 years with the index down 0.25%+ and 300+ advancers"
                    + (f", latest {div.index[-1]:%d %b %Y}." if not div.empty else "."))
        st.plotly_chart(RC.divergence_scatter(I, d["spx"]), width="stretch")
        st.caption("Uses current members, so history carries survivorship bias. " + _source(b, "sp500_internals"))


def _view_ai(d: dict, b: RD.ResearchBundle) -> None:
    P = b.prices
    today = P.index[-1] if not P.empty else pd.Timestamp.today()
    default = (today - pd.Timedelta(days=30)).date()
    since = pd.Timestamp(st.date_input("AI groups since", value=default, key="desk_ai_since"))
    groups = {g: R.basket_index(P.loc[since:], t) for g, t in RD.AI_GROUPS.items()}
    groups["Hyperscalers"] = R.basket_index(P.loc[since:], RD.HYPERSCALERS)
    rows = [{"group": g, "ret": float(s.iloc[-1] / 100 - 1) * 100} for g, s in groups.items() if not s.empty]
    if rows:
        df = pd.DataFrame(rows).sort_values("ret", ascending=False)
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(RC.hbars(df, "group", "ret", title=f"Equal-weight group return since {since:%d %b %Y}"), width="stretch")
        with c2:
            names = R.since_date_returns(P, RD.AI_BASKET + RD.HYPERSCALERS, since)
            if not names.empty:
                last = (names.ffill().iloc[-1] - 100).dropna().sort_values().rename("ret").reset_index()
                last.columns = ["name", "ret"]
                avg = float(last["ret"].mean())
                st.plotly_chart(RC.hbars(last, "name", "ret", title="Single names", ref=avg, ref_label=f"avg {avg:+.1f}%", height=600), width="stretch")
    st.markdown("**Correlation.** Low S&P correlation means index vol understates stock risk; a falling AI-basket reading after "
                "a peak marks forced selling fading.")
    a, c = d["ai_corr"], d["spx_corr"]
    spread = (a - c).dropna()
    st.plotly_chart(RC.stacked(
        [("AI basket 60D", a), ("S&P 500 60D", c)], [("AI minus S&P", spread)],
        start=pd.Timestamp.today() - pd.DateOffset(years=4), bottom_title="Correlation spread", fmt="{:.2f}", bottom_fmt="{:+.2f}",
    ), width="stretch")
    st.caption(f"Latest: S&P {R.ordinal(R.pct_rank(c.iloc[-1260:]))} pct of 5Y; AI basket {R.ordinal(R.pct_rank(a))} pct since 2021. "
               "Average pairwise correlation of daily returns. The AI basket is equal-weight across 23 names.")
    mtum, spy, gvip = R.px(P, "MTUM"), R.px(P, "SPY"), R.px(P, "GVIP")
    if not mtum.empty and not spy.empty:
        wk = (R.weekly(mtum).pct_change(fill_method=None) - R.weekly(spy).pct_change(fill_method=None)).dropna() * 100
        rank = int((wk < wk.iloc[-1]).sum() + 1)
        st.markdown(f"**Momentum factor proxy (MTUM − SPY, weekly).** Latest {wk.iloc[-1]:+.2f}pp, rank {rank} from worst of {len(wk)}.")
        st.plotly_chart(RC.signed_bars(wk, start=pd.Timestamp.today() - pd.DateOffset(years=6)), width="stretch")
        st.caption("A long-only ETF proxy. Bank long/short momentum baskets fall further and aren't public.")
    if not gvip.empty and not spy.empty:
        rel = (gvip.pct_change(5, fill_method=None) - spy.pct_change(5, fill_method=None)).dropna() * 100
        st.markdown("**Crowded hedge-fund longs (GVIP − SPY, 5 days).** The free stand-in for prime-book de-grossing.")
        st.plotly_chart(RC.lines([("GVIP minus SPY, 5D", rel)], start=pd.Timestamp.today() - pd.DateOffset(years=6), fmt="{:+.1f}pp",
                                 levels=[(float(rel.quantile(0.02)), "2nd pct")]), width="stretch")
        st.caption("Measures how crowded longs traded, not actual gross exposure. Prime-brokerage data is not public.")


def _view_vol(d: dict, b: RD.ResearchBundle, series: dict) -> None:
    P = b.prices
    vxn, vix = R.px(P, "^VXN"), R.px(P, "^VIX")
    if not vxn.empty and not vix.empty:
        tail = vxn.iloc[-1260:]
        st.markdown(f"**Nasdaq vol (VXN)** at {vxn.iloc[-1]:.1f}, {R.ordinal(R.pct_rank(tail))} percentile of 5 years. "
                    "VIX can stay calm when the stress is only in tech.")
        gap = (vxn - vix.reindex(vxn.index).ffill()).dropna()
        st.plotly_chart(RC.stacked(
            [("VXN", vxn), ("VIX", vix)], [("VXN minus VIX", gap)],
            start=pd.Timestamp.today() - pd.DateOffset(years=3), bottom_title="Tech vol premium over index vol, pts",
            bottom_fmt="{:+.1f}",
        ), width="stretch")
    sox = R.px(P, "^SOX")
    if not sox.empty:
        roc = sox.pct_change(5, fill_method=None).dropna() * 100
        st.markdown(f"**SOX 5-day change.** Latest {roc.iloc[-1]:+.1f}%, 5-session low {roc.iloc[-5:].min():+.1f}%.")
        st.plotly_chart(RC.lines([("SOX 5-day change", roc)], start=pd.Timestamp.today() - pd.DateOffset(years=10), fmt="{:+.1f}%",
                                 levels=[(-17.5, "−17.5%")], height=280), width="stretch")
    skew = R.px(P, "^SKEW")
    if len(skew) > 2600:
        rk = R.rolling_rank(skew, 2520, 1260)
        st.markdown("**SKEW, rolling 10-year percentile**")
        st.plotly_chart(RC.lines([("SKEW percentile", rk)], start=pd.Timestamp.today() - pd.DateOffset(years=5), fmt="{:.0f}",
                                 levels=[(98.5, "98.5th")], height=260), width="stretch")
    if not b.cboe.empty:
        c = b.cboe
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Total put volume**, contracts")
            st.plotly_chart(RC.lines([("Total puts (M)", c["total_put"] / 1e6), ("20D average", (c["total_put"] / 1e6).rolling(20).mean())],
                                     start=pd.Timestamp.today() - pd.DateOffset(years=3), fmt="{:.1f}M", height=280), width="stretch")
        with col2:
            st.markdown("**VIX call volume**, contracts")
            st.plotly_chart(RC.lines([("VIX calls (M)", c["vix_call"] / 1e6), ("20D average", (c["vix_call"] / 1e6).rolling(20).mean())],
                                     start=pd.Timestamp.today() - pd.DateOffset(years=3), fmt="{:.2f}M", height=280), width="stretch")
        st.caption(_source(b, "cboe_volumes"))
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Largest one-day VIX gains since 1990**")
        top = R.largest_moves(vix, 10)
        if not top.empty:
            st.dataframe(pd.DataFrame({"Date": top["date"].dt.strftime("%d %b %Y"), "Gain": top["gain"].map("{:+.1f}%".format),
                                       "Close": top["close"].map("{:.1f}".format)}), hide_index=True, width="stretch")
    with c2:
        st.markdown("**CNN Fear & Greed components**")
        if b.fng_parts:
            ext = sum(p["rating"] == "extreme fear" for p in b.fng_parts)
            st.dataframe(pd.DataFrame([{"Component": p["name"], "Score": f"{p['score']:.0f}", "Rating": p["rating"].title()} for p in b.fng_parts]),
                         hide_index=True, width="stretch")
            st.caption(f"{ext} of {len(b.fng_parts)} in extreme fear.")
        else:
            st.info("CNN components unavailable.")


def _view_rates(d: dict, b: RD.ResearchBundle, series: dict) -> None:
    P = b.prices
    g, pca = d["growth"], d["pca"]
    if g:
        st.markdown(f"**Cross-asset growth pricing.** {g['state']}: factor {g['factor']:+.2f}σ, 6-month z {g['z_6m']:+.2f}σ. "
                    f"PC1 explains {g['explained']:.0f}% of variance.")
        st.plotly_chart(RC.lines([("6M z-score", pca["z_6m"]), ("Growth factor", pca["factor"])],
                                 start=pd.Timestamp.today() - pd.DateOffset(years=3), fmt="{:+.2f}",
                                 levels=[(2, "+2σ"), (-2, "−2σ")], height=320), width="stretch")
        st.caption("First principal component of 14 cross-asset ETFs (equities, credit, Treasuries, FX, commodities) on a "
                   "correlation matrix estimated only from prior data. The 6-month z places the cumulative factor in its own trailing range.")
    tnx = R.px(P, "^TNX")
    if not tnx.empty:
        w = R.weekly(tnx)
        rw = R.rsi_wilder(w, 14)
        st.markdown(f"**10Y yield and weekly RSI.** {tnx.iloc[-1]:.2f}%, weekly RSI {rw.iloc[-1]:.1f}.")
        st.plotly_chart(RC.stacked([("10Y yield", w)], [("Weekly RSI", rw)], start=pd.Timestamp.today() - pd.DateOffset(years=6),
                                   fmt="{:.2f}%", bottom_lines=[(70, "70")], bottom_range=(10, 95)), width="stretch")
        ep = R.yield_rsi_episodes(tnx.loc["2019-01-01":])
        if not ep.empty:
            st.dataframe(pd.DataFrame({
                "Peak week": ep["peak_date"].dt.strftime("%d %b %Y"), "Yield": ep["peak_yield"].map("{:.2f}%".format),
                "Peak RSI": ep["rsi"].map("{:.0f}".format), "Yield fall within 26W": ep["fall_pct"].map("{:+.1f}%".format),
                "Weeks to low": ep["weeks_to_low"], "Complete": np.where(ep["complete"], "Yes", "In progress"),
            }), hide_index=True, width="stretch")
    bc = R.bond_composite(P.loc["2015-01-01":])
    if not bc.empty:
        st.markdown(f"**Bond composite (HYG / LQD / TLT).** Daily RSI {bc['rsi_d'].iloc[-1]:.1f}, "
                    f"{R.ordinal(R.pct_rank(bc['rsi_d'].dropna()))} pct since 2015.")
        st.plotly_chart(RC.stacked([("Composite", bc["composite"])], [("Daily RSI", bc["rsi_d"])],
                                   start=pd.Timestamp.today() - pd.DateOffset(years=4), fmt="{:.1f}",
                                   bottom_lines=[(30, "30"), (70, "70")], bottom_range=(10, 90)), width="stretch")
    tyx = R.px(P, "^TYX")
    shock = R.change_shock(tyx.loc["2015-01-01":], 10) if not tyx.empty else {}
    if shock:
        st.markdown(f"**30Y yield, 10-session change.** {shock['bps']:+.1f}bp, {R.ordinal(shock['pct'])} percentile since 2015.")
        st.plotly_chart(RC.lines([("30Y 10-session change, bp", shock["series"])], start=pd.Timestamp.today() - pd.DateOffset(years=4),
                                 fmt="{:+.0f}", height=260), width="stretch")
    mv, spx = R.px(P, "^MOVE"), d["spx"]
    if not mv.empty:
        st.markdown(f"**MOVE vs the S&P 500.** MOVE {mv.iloc[-1]:.0f}, {R.ordinal(R.pct_rank(mv.iloc[-1260:]))} pct of 5Y.")
        st.plotly_chart(RC.stacked([("S&P 500", spx)], [("MOVE", mv)], start=pd.Timestamp.today() - pd.DateOffset(years=5),
                                   fmt="{:,.0f}", bottom_fmt="{:.0f}", bottom_lines=[(120, "120")]), width="stretch")
    ig, ccc = R.clean(series.get("ig_spread")), b.ccc
    if not ccc.empty and not ig.empty:
        st.markdown(f"**Credit divide.** IG {ig.iloc[-1] * 100:.0f}bp, CCC & lower {ccc.iloc[-1] * 100:,.0f}bp.")
        st.plotly_chart(RC.stacked([("CCC & lower OAS, bp", ccc * 100)], [("IG OAS, bp", ig * 100)], fmt="{:,.0f}", bottom_fmt="{:.0f}",
                                   start=pd.Timestamp.today() - pd.DateOffset(years=3)), width="stretch")
        st.caption("FRED serves about 3 years of ICE spreads. " + _source(b, "ccc_oas"))
    if not b.cape.empty:
        e = b.cape["ecy"].dropna()
        st.markdown(f"**Excess CAPE yield.** {e.iloc[-1]:.2f}%, {R.ordinal(R.pct_rank(e))} percentile since 1881, mean {e.mean():.2f}%.")
        st.plotly_chart(RC.lines([("Excess CAPE yield", e)], fmt="{:.2f}%", levels=[(float(e.mean()), "mean"), (0, "")], height=280), width="stretch")
        st.caption("Earnings yield on CAPE minus the 10Y real yield. Slow-moving context, not a timing tool. " + _source(b, "excess_cape"))
    fpe = b.sp_fwd_pe
    if not fpe.empty:
        st.markdown(f"**S&P 500 forward P/E.** {fpe.iloc[-1]:.1f}x, {R.ordinal(R.pct_rank(fpe))} percentile of its short free history.")
        st.plotly_chart(RC.lines([("Forward P/E", fpe)], fmt="{:.1f}x", height=240), width="stretch")
        st.caption("Bottom-up consensus, daily since March 2026. Levels run above Bloomberg's estimate by methodology, "
                   "so use the direction rather than the level. " + _source(b, "sp500_forward_pe"))
    st.markdown("**What is driving equities?** 20-day rolling correlation of daily changes.")
    spy, oil = R.px(P, "SPY"), R.px(P, "CL=F")
    corr = [("SPY vs 10Y yield", R.rolling_corr(spy, tnx))]
    if not oil.empty:
        corr.append(("SPY vs WTI", R.rolling_corr(spy, oil)))
    if not d["ai_index"].empty:
        corr.append(("AI basket vs TLT", R.rolling_corr(d["ai_index"], R.px(P, "TLT"))))
    st.plotly_chart(RC.lines(corr, start=pd.Timestamp.today() - pd.DateOffset(years=2), fmt="{:+.2f}", levels=[(0, "")], height=300), width="stretch")


def _view_playbook(d: dict, b: RD.ResearchBundle) -> None:
    pb = d["playbook"]
    if pb.empty:
        st.info("Event studies unavailable: no price history.")
        return
    st.markdown("**Does the signal beat doing nothing?** Each row compares returns after the signal with the unconditional "
                "return of the same asset over the same era.")
    h = st.radio("Horizon, sessions", [5, 10, 21, 63, 126], index=2, horizontal=True, key="desk_pb_h")
    t = pb[pb["horizon"] == h].copy()
    t = t.sort_values(["evidence", "p_mean"], key=lambda c: c.map({"Real edge": 0, "Weak edge": 1, "No edge vs base rate": 2, "Too few events": 3}) if c.name == "evidence" else c)
    show = pd.DataFrame({
        "Signal": t["signal"], "Target": t["target"].map(lambda x: TARGET_NAMES.get(x, x)), "Evidence": t["evidence"], "Events": t["n"],
        "Mean": t["mean"].map("{:+.2f}%".format), "Base mean": t["base_mean"].map("{:+.2f}%".format),
        "Hit rate": t["hit"].map("{:.0f}%".format), "Base hit": t["base_hit"].map("{:.0f}%".format),
        "Avg drawdown": t["mae"].map("{:+.1f}%".format), "p": t["p_mean"].map("{:.2f}".format),
        "Last fired": t["last_event"].dt.strftime("%d %b %Y"),
    })
    st.dataframe(show, hide_index=True, width="stretch")
    st.caption(
        "Real edge: p ≤ 0.05 and hit rate 5+ points above base, on 8+ events. p = share of random same-size date sets with "
        "as high a mean. Avg drawdown = worst close inside the window. With 15 signals and 5 horizons, some rows clear "
        "p ≤ 0.05 by chance, so trust ones that hold across horizons. Events after a rule fires overlap in crises."
    )
    bt = d["backtest"]
    if not bt.empty:
        st.markdown("**Capitulation checklist: QQQ returns by core rules active**")
        hz = st.radio("Checklist horizon", [10, 21, 63], index=1, horizontal=True, key="desk_bt_h")
        x = bt[bt["horizon"] == hz]
        st.dataframe(pd.DataFrame({
            "Core rules on": x["bucket"], "Episodes": x["n"], "Mean": x["mean"].map("{:+.2f}%".format),
            "Base mean": x["base_mean"].map("{:+.2f}%".format), "Hit rate": x["hit"].map("{:.0f}%".format),
            "Base hit": x["base_hit"].map("{:.0f}%".format), "Avg drawdown": x["mae"].map("{:+.1f}%".format),
            "p": x["p_mean"].map("{:.2f}".format),
        }), hide_index=True, width="stretch")
        st.caption(f"Since {R._rules_live_from(d['cap_history']):%b %Y}, when every core rule could fire. Episodes need 10 quiet "
                   "sessions apart, except the 0 bucket. The 5+ bucket is very thin, so don't read it as a reliable rate.")
        st.warning("This table covers the era the rules were built on. On untouched 2003–2016 data, 3+ of 5 long-history "
                   "rules lagged the base rate: −3.2pp at 21 sessions over 11 episodes, 5 of them in late 2008. Without late "
                   "2008 the result was flat at 21 sessions and not significant at 63. SKEW was one of those 5 and was then "
                   "removed from the core count, so today's 6-rule count has no clean test of its own. Treat a washout as a "
                   "condition to watch, not as evidence of a bottom.")
    sm = R.midterm_seasonality(b.spx_monthly)
    if not sm.empty:
        done = sm["six_m"].dropna()
        base = R.non_midterm_base(b.spx_monthly)
        txt = f"{int((done > 0).sum())} of {len(done)} positive, mean {done.mean():+.1f}%"
        if base:
            txt += f". All other years: {base['hit']:.0f}% positive, mean {base['mean']:+.1f}% (n={base['n']})"
        st.markdown(f"**Midterm-year seasonality, Oct close to Apr close.** {txt}.")
        st.dataframe(sm.iloc[::-1].rename(columns={"cycle": "Cycle", "six_m": "6M"}).round(1), hide_index=True, width="stretch")
        st.caption("S&P 500 price return, monthly closes since 1950. Nineteen cycles is a small sample with one year each. " + _source(b, "spx_monthly"))


def _view_board(d: dict, b: RD.ResearchBundle) -> None:
    t = R.technical_board(b.prices, RD.TECH_BOARD, overrides={"BTC-USD": b.btc})
    if t.empty:
        st.info("Technical board unavailable.")
        return
    show = pd.DataFrame({
        "Asset": t["name"], "Last": t["last"].map("{:,.2f}".format), "1D": t["d1"].map("{:+.1f}%".format), "5D": t["d5"].map("{:+.1f}%".format),
        "21D": t["d21"].map("{:+.1f}%".format), "RSI": t["rsi"].map("{:.1f}".format), "vs 50D": t["vs50"].map("{:+.1f}%".format),
        "vs 200D": t["vs200"].map("{:+.1f}%".format), "From 52W high": t["dd_52w"].map("{:+.1f}%".format),
        "5D %ile": t["roc5_pct"].map("{:.0f}".format), "Flag": t["flag"], "MA cross, 21D": t["cross"],
    })
    st.dataframe(show, hide_index=True, width="stretch")
    st.caption("RSI is Wilder 14-day. 5D %ile ranks the 5-day change against the asset's history since 1990 or its listing.")


VIEWS = ["Positioning", "Rotation", "AI & momentum", "Vol & options", "Rates & credit", "Technical board", "Playbook"]


@st.fragment
def render_research(d: dict, b: RD.ResearchBundle, series: dict) -> None:
    st.markdown("### Research desk")
    view = st.segmented_control("View", VIEWS, default="Playbook", key="desk_view", label_visibility="collapsed") or "Playbook"
    if view == "Positioning":
        _view_positioning(d, b)
    elif view == "Rotation":
        _view_rotation(d, b, series)
    elif view == "AI & momentum":
        _view_ai(d, b)
    elif view == "Vol & options":
        _view_vol(d, b, series)
    elif view == "Rates & credit":
        _view_rates(d, b, series)
    elif view == "Technical board":
        _view_board(d, b)
    else:
        _view_playbook(d, b)
