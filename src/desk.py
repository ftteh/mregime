"""
Assemble the research desk: combine the research bundle with the dashboard's own
scored series into one dict of tables, rule results and a model posture.

No Streamlit here, so the whole decision layer can be exercised in a test. The
composite regime score is deliberately NOT changed by any of this: the desk sits
beside the composite as a separate read, so a new feed outage can't shift a score
that has its own history and regression snapshot.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import indicators as I
from src import research as R
from src import research_data as RD
from src.config import regime_label


def model_posture(guard: dict, cap: dict, confirm_fired: int, confirm_total: int, gate: dict, rotation: dict) -> dict:
    """Combine the four independent reads into one conditional sentence.

    Order matters and mirrors how the checks depend on each other: a broken
    system overrides a washed-out tape, and a washout without price confirmation
    is a watch state, not a green light.
    """
    state = cap.get("state", "QUIET")
    washed = state in ("ARMED", "CAPITULATION")
    enough = confirm_total > 0 and confirm_fired >= 3
    if guard.get("state") == "BREAKING":
        head, tone = "Defensive: stress is spreading beyond the sector", "bear"
        body = "Credit or participation has failed. Washout signals are less reliable until the guardrails recover."
    elif washed and enough:
        head, tone = "Washed out with confirmation", "bull"
        held = "the guardrails hold" if guard.get("state") == "INTACT" else "most guardrails hold"
        body = f"Panic gauges fired and are rolling over while {held}. This is the setup the checklist waits for."
    elif washed:
        head, tone = "Washed out, turn not confirmed", "warm"
        held = "the guardrails hold" if guard.get("state") == "INTACT" else "most guardrails hold"
        body = f"Panic gauges fired and {held}, but price has not confirmed. Watch the confirmation list."
    elif rotation.get("stage") in ("ACTIVE", "EARLY", "STRETCHED", "ROLLING OVER"):
        head, tone = f"Rotation regime: {rotation['stage'].lower()}", "neutral"
        body = rotation.get("read", "")
    else:
        head, tone = "No extreme: the rules have nothing to add", "neutral"
        body = "Neither a washout nor a rotation signal is active."
    suffix = {"NO LEVERAGE": "Leverage gate closed.", "REDUCED": "Leverage gate partly closed.", "OPEN": "Leverage gate open."}
    return {"head": head, "tone": tone, "body": body, "gate": suffix.get(gate.get("state"), "")}


# Feeds the rules can't do without. Anything else degrading only affects a panel.
KEY_FEEDS = {
    "research_prices": "Prices", "cot_positioning": "CFTC", "cboe_volumes": "Cboe volumes",
    "sp500_internals": "S&P internals", "ccc_oas": "CCC spread",
}


# Calendar days each feed may trail today before the desk says so. CFTC is as of
# Tuesday and published Friday, so it is 3 to 10 days old in normal running.
FEED_MAX_AGE = {"Prices": 4, "Credit": 7, "CFTC": 13, "Cboe": 6, "Gamma": 10}


def _age_days(ts: pd.Timestamp | None, today: pd.Timestamp) -> int | None:
    return None if ts is None else int((today.normalize() - pd.Timestamp(ts).normalize()).days)


def desk_freshness(provenance: dict, as_of: dict[str, pd.Timestamp | None], today: pd.Timestamp, max_age_days: int = 4) -> dict:
    """What the desk is actually reading, and whether it should be trusted today.

    A rule output looks equally confident on two-week-old seed data, so anything
    not served live, or older than its limit, is surfaced next to the verdict.
    Every dated feed is checked, not only prices: a CFTC report that stopped
    publishing would otherwise keep feeding the leverage gate as if current.
    """
    issues = []
    for key, label in KEY_FEEDS.items():
        p = provenance.get(key) or {}
        if p.get("kind") in ("cache", "fallback"):
            issues.append(f"{label} from saved data ({p.get('note', '')})")
        elif p.get("kind") == "unavailable" or not p:
            issues.append(f"{label} unavailable")
    limits = {**FEED_MAX_AGE, "Prices": max_age_days}
    if as_of.get("Prices") is None:
        issues.append("no price data")
    for name, ts in as_of.items():
        age = _age_days(ts, today)
        if age is not None and age > limits.get(name, max_age_days):
            issues.append(f"{name.lower() if name == 'Prices' else name} data is stale, last {ts:%d %b %Y}")
    return {"as_of": {k: v for k, v in as_of.items() if v is not None}, "issues": issues}


MIN_CONFIRMATIONS = 3  # same bar the desk uses before a washout counts as a turn


def leverage_permission(desk: dict | None) -> dict:
    """Whether the exposure card may go above 100%, from the desk's own reads.

    Leverage has to be earned: the gate must not be closed AND price must have
    confirmed. If the desk couldn't run, nothing has been checked, so the answer
    is no rather than quietly allowing leverage.
    """
    if not desk:
        return {"allowed": False, "confirmed": False, "reason": "decision desk unavailable"}
    live = [r for r in desk.get("confirm", []) if r.available]
    fired = sum(r.fired for r in live)
    confirmed = fired >= MIN_CONFIRMATIONS
    closed = (desk.get("gate") or {}).get("state") == "NO LEVERAGE"
    if closed:
        reason = "leverage gate closed"
    elif not confirmed:
        reason = f"price confirmation {fired} of {len(live)}, needs {MIN_CONFIRMATIONS}"
    else:
        reason = ""
    return {"allowed": confirmed and not closed, "confirmed": confirmed, "reason": reason}


def apply_leverage_cap(rec: dict, permission: dict, limit: int = 100) -> dict:
    """Apply the desk's permission to an exposure recommendation.

    Only limits how far above 100% the card goes: it never raises a lower
    recommendation or cuts below `limit`, so the composite's read on being
    invested is untouched. A bottom cluster also stops reading as High conviction
    until price confirms. A top cluster keeps its conviction, because the
    confirmation list only describes turns up from a washout.
    """
    out = dict(rec)
    net = out.get("net_pct")
    if net is None:
        return out
    if net > limit and not permission.get("allowed", False):
        out["uncapped_pct"] = net
        out["net_pct"] = limit
        out["cap_reason"] = permission.get("reason", "")
        out["rationale"] = f"{out.get('rationale', '')} · capped from {net}% to {limit}%: {out['cap_reason']}"
        # The label describes the exposure, so it follows the capped figure. A
        # cluster label stays: it names the signal, and the cap line explains the number.
        if out.get("cluster_override") is False:
            out["label"], out["color"] = I.exposure_label(limit)
    bottom = str(out.get("label", "")).startswith("BOTTOM CLUSTER")
    if bottom and out.get("conviction") == "High" and not permission.get("confirmed", False):
        out["conviction"] = "Medium"
    return out


def assemble(b: RD.ResearchBundle, series: dict[str, pd.Series]) -> dict:
    """`series` is the dashboard's raw.series map."""
    P, I = b.prices, b.internals
    get = lambda k: series.get(k, pd.Series(dtype=float))  # noqa: E731
    spx = R.px(P, "^GSPC")
    if spx.empty:
        spx = R.clean(get("spx"))
    spx_corr = I["corr_60"].dropna() if "corr_60" in I else pd.Series(dtype=float)

    ai_cols = [c for c in RD.AI_BASKET if c in P.columns]
    ai_corr = R.avg_pairwise_corr(P[ai_cols].loc["2021-01-01":], 60, min_coverage=0.7) if ai_cols else pd.Series(dtype=float)
    ai_idx = R.basket_index(P.loc["2021-01-01":], ai_cols)
    pca = R.cross_asset_pca(P[RD.PCA_UNIVERSE].loc["2012-01-01":]) if set(RD.PCA_UNIVERSE) <= set(P.columns) else pd.DataFrame()

    # Decision rules
    hist = R.capitulation_history(P)
    put_vol = b.cboe["total_put"].dropna() if "total_put" in b.cboe else pd.Series(dtype=float)
    # Cboe's own files, not the scored series, whose newest row can be an ETF-chain proxy.
    equity_pc = (b.cboe["equity_put"] / b.cboe["equity_call"]).dropna() if "equity_put" in b.cboe else get("put_call")
    rules = R.capitulation_rules(
        P, put_call=equity_pc, fear_greed=get("fear_greed"), aaii=get("aaii_bull_bear"),
        ndx_pe=get("ndx_trailing_pe"), put_volume=put_vol, history=hist,
    )
    cap = R.capitulation_verdict(rules)
    backtest = R.capitulation_backtest(hist, R.px(P, "QQQ"))
    confirm = R.reversal_confirmation(P, ai_corr)
    guards = R.system_guardrails(
        trend_breadth=get("trend_breadth_10_20"), pct_200=get("pct_above_200dma"), ig=get("ig_spread"),
        hy=get("hy_spread"), ccc=b.ccc, sofr_spread=get("sofr_spread"), spx_corr=spx_corr, nfci=get("financial_conditions"),
    )
    guard = R.guardrail_verdict(guards)

    # Positioning
    cot_table = R.positioning_table(b.cot, RD.COT_MARKETS)
    vix_row = cot_table[cot_table["key"] == "VIX"] if not cot_table.empty else pd.DataFrame()
    vix_cot = vix_row.iloc[0].to_dict() if not vix_row.empty else None
    # The gate reads only current inputs. An old gamma or positioning print is
    # left out and reported, not passed off as today's.
    today = pd.Timestamp.today()
    gamma = R.clean(get("gamma_exposure"))
    gamma_asof = gamma.index[-1] if len(gamma) else None
    cot_asof = b.cot.index.max() if not b.cot.empty else None
    dropped = []
    if gamma_asof is not None and _age_days(gamma_asof, today) > FEED_MAX_AGE["Gamma"]:
        gamma, dropped = None, dropped + ["dealer gamma"]
    if vix_cot is not None and cot_asof is not None and _age_days(cot_asof, today) > FEED_MAX_AGE["CFTC"]:
        vix_cot, dropped = None, dropped + ["VIX futures positioning"]
    gate = R.leverage_gate(vix_cot=vix_cot, spx_corr=spx_corr, gamma=gamma, vix=R.px(P, "^VIX"))
    rotation = R.rotation_stage(P)
    live_confirm = [r for r in confirm if r.available]
    posture = model_posture(guard, cap, sum(r.fired for r in live_confirm), len(live_confirm), gate, rotation)

    # Playbook
    signals = R.signal_library(P, internals=I, trend_breadth=get("trend_breadth_10_20"), cot=b.cot)
    playbook = R.run_playbook(P, signals)

    last = lambda x: R.clean(x).index[-1] if len(R.clean(x)) else None  # noqa: E731
    freshness = desk_freshness(
        b.provenance,
        {"Prices": last(spx), "Credit": last(get("ig_spread")), "CFTC": cot_asof,
         "Cboe": b.cboe.index.max() if not b.cboe.empty else None, "Gamma": gamma_asof},
        today,
    )
    if dropped:
        freshness["issues"].append("leverage gate ran without " + " and ".join(dropped) + " (stale)")

    return {
        "freshness": freshness, "posture": posture, "guards": guards, "guard": guard, "rules": rules, "cap": cap, "confirm": confirm,
        "gate": gate, "rotation": rotation, "backtest": backtest, "cap_history": hist, "cot_table": cot_table,
        "playbook": playbook, "pca": pca, "growth": R.growth_snapshot(pca), "ai_corr": ai_corr, "ai_index": ai_idx,
        "spx_corr": spx_corr, "spx": spx,
    }



# ---------------------------------------------------------------------------
# One reconciled read
# ---------------------------------------------------------------------------
def _label_bounds(score: float) -> tuple[float | None, float | None]:
    """Nearest composite scores below and above `score` where the exposure
    label changes, read off the live mapping so the text can't drift from it."""
    here = I.exposure_label(I.base_exposure(score)[0])[0]
    lo = next((s for s in np.arange(np.floor(score), -1, -1) if I.exposure_label(I.base_exposure(s)[0])[0] != here), None)
    hi = next((s for s in np.arange(np.ceil(score), 101) if I.exposure_label(I.base_exposure(s)[0])[0] != here), None)
    return lo, hi


def decision(
    rec: dict,
    comp: dict,
    cluster: dict,
    confidence: dict,
    desk: dict | None = None,
    *,
    context: dict | None = None,
    session: dict | None = None,
) -> dict:
    """Reconcile the composite card with the desk into one read.

    The composite is relative (each indicator against its own last three years)
    and the desk rules are absolute thresholds. They can disagree, and the page
    used to show both headlines without saying so. This returns what the card
    says, why, where the two reads conflict, what would change the call, and
    what to discount in the data. Pure: no thresholds of its own.
    """
    why: list[str] = []
    conflicts: list[str] = []
    triggers: list[str] = []
    caveats: list[str] = []
    score, net = comp.get("composite"), rec.get("net_pct")
    session = session or {}
    if session.get("live"):
        caveats.append(f"US session in progress. Scores use closes through {session['as_of']:%a %d %b}; live quotes are not scored.")
    elif session.get("provisional"):
        caveats.append("Live-session mode is on: today's unfinished bar is being scored. Treat every reading as provisional.")

    if net is None or score is None or np.isnan(score):
        cov = comp.get("covered_weight", 0.0) * 100
        return {"headline": "No call: the composite is offline", "sub": f"Only {cov:.0f}% of the model's weight has live data.",
                "tone": "#7f8c8d", "why": [], "conflicts": [], "triggers": [], "caveats": caveats}

    regime = regime_label(score)[0].split("—")[0].strip().title()
    line = f"Composite {score:.0f}, {regime}"
    if context:
        line += (f": {R.ordinal(context['pct'])} percentile of its own last {context['years']:.1f} years "
                 f"(range {context['low']:.0f} to {context['high']:.0f})")
    why.append(line + ".")
    top, bot = cluster.get("top_theme_count", 0), cluster.get("bottom_theme_count", 0)
    need = cluster.get("need", I.cluster_need(cluster.get("themes_live", 0)))
    if rec.get("cluster_override"):
        side = "top" if str(rec.get("label", "")).startswith("TOP") else "bottom"
        n, other = (top, bot) if side == "top" else (bot, top)
        why.append(f"Cluster: {n} independent themes at a {side} extreme against {other} at the other. "
                   "Extreme means against each indicator's own last 3 years, not an absolute stress level.")
    elif rec.get("cluster_flag"):
        why.append(f"Top-cluster flag: {top} themes at a top extreme against {bot}. A risk flag only. It does not "
                   "cut exposure, because in the history available it was followed by ordinary returns.")
    else:
        why.append(f"No cluster: {bot} themes at a bottom extreme and {top} at a top one. "
                   f"One side has to lead by {need}.")
    if rec.get("uncapped_pct") is not None:
        why.append(f"The composite alone asks for {rec['uncapped_pct']}%. Held at {net}%: {rec.get('cap_reason', '')}.")

    washed = False
    if desk:
        guard, cap, gate = desk["guard"], desk["cap"], desk["gate"]
        live = [r for r in desk.get("confirm", []) if r.available]
        fired = sum(r.fired for r in live)
        washed = cap["state"] in ("ARMED", "CAPITULATION")
        why.append(f"Desk: guardrails {guard['state'].lower()} ({guard['held']} of {guard['total']} hold), washout rules "
                   f"{cap['state'].lower()} ({cap['core_fired']} of {cap['core_total']} core), price confirmation "
                   f"{fired} of {len(live)}, leverage gate {gate['state'].lower()}.")

        fear = score < 35 or (rec.get("cluster_override") and str(rec.get("label", "")).startswith("BOTTOM"))
        greed = score >= 65 or rec.get("cluster_side") == "top" or str(rec.get("label", "")).startswith("TOP")
        if fear and not washed:
            conflicts.append(f"The composite reads fear against the last 3 years, but only {cap['core_fired']} of "
                             f"{cap['core_total']} absolute washout rules fired. A relative extreme, not a capitulation.")
        if washed and not fear and score >= 45:
            conflicts.append(f"Washout rules fired ({cap['core_fired']} of {cap['core_total']}) while the composite is "
                             "neutral or higher: the stress is concentrated in one part of the market, not broad.")
        if washed and greed:
            conflicts.append("The composite reads complacent while washout rules fired. Read the desk first.")
        if guard["state"] == "BREAKING" and net >= 95:
            conflicts.append("Guardrails are breaking while the card is near fully invested. In the out-of-sample test, "
                             "washouts with broken guardrails had much deeper drawdowns. The card does not cut for this.")

        if rec.get("uncapped_pct") is not None or net > 100:
            hits = gate.get("hits", 0)
            if net > 100:
                lever = (f"Leverage is on. It comes off if the gate closes (now {hits} of 2 hits) or confirmations "
                         f"fall below {MIN_CONFIRMATIONS} (now {fired}).")
            else:
                lever = (f"Above 100% needs the leverage gate not closed (now {hits} of the 2 hits that close it) and "
                         f"{MIN_CONFIRMATIONS} of {len(live)} price confirmations (now {fired}).")
                missing = [r.label for r in live if not r.fired]
                if missing:
                    lever += " Not yet met: " + "; ".join(missing) + "."
            triggers.append(lever)
        if guard["state"] == "WATCH":
            failed = [r.label for r in desk.get("guards", []) if r.available and not r.fired]
            triggers.append("Guardrails on watch (" + "; ".join(failed) + "). A second failure that involves credit, "
                            "or a third of any kind, reads as breaking.")
        caveats += [i[0].upper() + i[1:] + "." for i in desk.get("freshness", {}).get("issues", [])]
    else:
        caveats.append("Decision desk unavailable this run, so nothing has checked leverage: the card stays at or below 100%.")

    lo, hi = _label_bounds(score)
    base_label = I.exposure_label(I.base_exposure(score)[0])[0]
    parts = []
    if lo is not None:
        parts.append(f"at {lo:.0f} or below it becomes {I.exposure_label(I.base_exposure(lo)[0])[0]}")
    if hi is not None:
        parts.append(f"at {hi:.0f} or above {I.exposure_label(I.base_exposure(hi)[0])[0]}")
    if parts:
        triggers.append(f"Composite alone maps to {base_label} ({rec.get('base_pct', net)}%): " + ", ".join(parts) + ".")
    gap = need - abs(top - bot)
    if not rec.get("cluster_override") and not rec.get("cluster_flag") and 0 < gap <= 2 and top != bot:
        triggers.append(f"{gap} more theme{'s' if gap != 1 else ''} at a {'bottom' if bot > top else 'top'} extreme "
                        "would make it a cluster.")

    cf = confidence or {}
    notes = []
    if cf.get("coverage_pct", 100) < 80:
        notes.append(f"{cf['coverage_pct']:.0f}% of model weight is live")
    if cf.get("n_stale"):
        notes.append(f"{cf['n_stale']} indicator{'s' if cf['n_stale'] != 1 else ''} dropped as stale")
    if cf.get("n_overdue"):
        notes.append(f"{cf['n_overdue']} feed{'s' if cf['n_overdue'] != 1 else ''} overdue")
    if cf.get("n_proxy"):
        notes.append(f"{cf['n_proxy']} proxy or fallback feed{'s' if cf['n_proxy'] != 1 else ''}")
    disp = cf.get("dispersion")
    if disp is not None and not np.isnan(disp) and disp > 25:
        notes.append(f"indicators disagree (score spread {disp:.0f})")
    if cf.get("level") and cf["level"] != "High" and notes:
        caveats.append(f"{cf['level']} confidence: " + ", ".join(notes) + ".")

    sub = desk["posture"]["head"] if desk else "Composite read only"
    return {"headline": f"{net}% net equity · {rec['label']}", "sub": sub, "tone": rec.get("color", "#95a5a6"),
            "why": why, "conflicts": conflicts, "triggers": triggers, "caveats": caveats}
