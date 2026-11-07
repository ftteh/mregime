"""
Assemble the research desk: combine the research bundle with the dashboard's own
scored series into one dict of tables, rule results and a model posture.

No Streamlit here, so the whole decision layer can be exercised in a test. The
composite regime score is deliberately NOT changed by any of this: the desk sits
beside the composite as a separate read, so a new feed outage can't shift a score
that has its own history and regression snapshot.
"""
from __future__ import annotations

import pandas as pd

from src import research as R
from src import research_data as RD


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


def desk_freshness(provenance: dict, as_of: dict[str, pd.Timestamp | None], today: pd.Timestamp, max_age_days: int = 4) -> dict:
    """What the desk is actually reading, and whether it should be trusted today.

    A rule output looks equally confident on two-week-old seed data, so anything
    not served live, or older than `max_age_days`, is surfaced next to the verdict.
    """
    issues = []
    for key, label in KEY_FEEDS.items():
        p = provenance.get(key) or {}
        if p.get("kind") in ("cache", "fallback"):
            issues.append(f"{label} from saved data ({p.get('note', '')})")
        elif p.get("kind") == "unavailable" or not p:
            issues.append(f"{label} unavailable")
    prices = as_of.get("Prices")
    if prices is None:
        issues.append("no price data")
    elif (today.normalize() - prices.normalize()).days > max_age_days:
        issues.append(f"prices are stale, last {prices:%d %b %Y}")
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
    gate = R.leverage_gate(vix_cot=vix_cot, spx_corr=spx_corr, gamma=get("gamma_exposure"), vix=R.px(P, "^VIX"))
    rotation = R.rotation_stage(P)
    live_confirm = [r for r in confirm if r.available]
    posture = model_posture(guard, cap, sum(r.fired for r in live_confirm), len(live_confirm), gate, rotation)

    # Playbook
    signals = R.signal_library(P, internals=I, trend_breadth=get("trend_breadth_10_20"), cot=b.cot)
    playbook = R.run_playbook(P, signals)

    last = lambda x: R.clean(x).index[-1] if len(R.clean(x)) else None  # noqa: E731
    freshness = desk_freshness(
        b.provenance,
        {"Prices": last(spx), "Credit": last(get("ig_spread")), "CFTC": b.cot.index.max() if not b.cot.empty else None,
         "Cboe": b.cboe.index.max() if not b.cboe.empty else None},
        pd.Timestamp.today(),
    )

    return {
        "freshness": freshness, "posture": posture, "guards": guards, "guard": guard, "rules": rules, "cap": cap, "confirm": confirm,
        "gate": gate, "rotation": rotation, "backtest": backtest, "cap_history": hist, "cot_table": cot_table,
        "playbook": playbook, "pca": pca, "growth": R.growth_snapshot(pca), "ai_corr": ai_corr, "ai_index": ai_idx,
        "spx_corr": spx_corr, "spx": spx,
    }
