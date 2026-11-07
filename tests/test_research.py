"""Research desk engine: deterministic, no network."""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import desk as K
from src import research as R
from src import research_charts as RC
from src import research_data as RD
from src.research import Rule


def _walk(n: int = 900, seed: int = 1, drift: float = 0.0004, vol: float = 0.01, start: str = "2015-01-01") -> pd.Series:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, periods=n)
    return pd.Series(100 * np.exp(np.cumsum(rng.normal(drift, vol, n))), index=idx)


def _prices(n: int = 1800, tickers=None) -> pd.DataFrame:
    tickers = tickers or ["^GSPC", "SPY", "QQQ", "RSP", "IWM", "XLK", "SMH", "XLU", "XLF", "XLV", "ITB", "^SOX",
                          "^VIX", "^VXN", "^SKEW", "MTUM", "GVIP", "^MOVE", "^TNX"]
    return pd.DataFrame({t: _walk(n, seed=i + 1) for i, t in enumerate(tickers)})


# --- basics ----------------------------------------------------------------
def test_rsi_wilder_bounds_and_direction():
    up = pd.Series(np.arange(1.0, 60.0), index=pd.bdate_range("2024-01-01", periods=59))
    assert R.rsi_wilder(up).iloc[-1] == pytest.approx(100.0)
    assert R.rsi_wilder(up[::-1].reset_index(drop=True).set_axis(up.index)).iloc[-1] == pytest.approx(0.0, abs=1e-9)


def test_pct_rank_counts_ties_as_at_or_below():
    s = pd.Series([1.0, 2.0, 3.0, 4.0], index=pd.bdate_range("2024-01-01", periods=4))
    assert R.pct_rank(s) == 100.0
    assert R.pct_rank(s, 2.0) == 50.0


@pytest.mark.parametrize("p,txt", [(1, "1st"), (2, "2nd"), (3, "3rd"), (4, "4th"), (11, "11th"), (12, "12th"), (13, "13th"),
                                   (21, "21st"), (22.4, "22nd"), (91.4, "91st"), (100, "100th")])
def test_ordinal(p, txt):
    assert R.ordinal(p) == txt


def test_expanding_rank_has_no_lookahead():
    s = _walk(400)
    full = R.expanding_rank(s, 100)
    cut = R.expanding_rank(s.iloc[:300], 100)
    pd.testing.assert_series_equal(full.iloc[:300], cut, check_names=False)


def test_on_us_calendar_keeps_monday_returns():
    idx = pd.date_range("2024-01-01", periods=21, freq="D")
    p = pd.DataFrame({"SPY": 100.0 + np.arange(21.0), "BTC-USD": 50.0 + np.arange(21.0)}, index=idx)
    p.loc[idx.dayofweek >= 5, "SPY"] = np.nan
    out = RD.on_us_calendar(p)
    assert (out.index.dayofweek < 5).all()
    # Monday's return survives once the weekend rows are gone.
    assert out["SPY"].pct_change(fill_method=None).notna().sum() == len(out) - 1


# --- positioning -----------------------------------------------------------
def test_positioning_snapshot_flags_structural_gap():
    # Near an all-time low but mid-range for the last 3 years.
    idx = pd.date_range("2010-01-05", periods=800, freq="W-TUE")
    vals = np.r_[np.linspace(40, 0, 640), -10 + 3 * np.sin(np.arange(160) / 3.0)]
    snap = R.positioning_snapshot(pd.Series(vals, index=idx))
    assert snap["pct_all"] < 20 and snap["pct_3y"] > 40
    assert snap["structural"] is True
    assert snap["dist_median"] == pytest.approx(snap["pct_all"] - 50)


def test_positioning_snapshot_flows_are_changes_in_net():
    idx = pd.date_range("2020-01-07", periods=10, freq="W-TUE")
    snap = R.positioning_snapshot(pd.Series(np.arange(10.0), index=idx))
    assert snap["flow_1w"] == 1.0 and snap["flow_4w"] == 4.0


def test_positioning_labels():
    assert R.positioning_label(2) == "VERY LOW"
    assert R.positioning_label(15) == "LOW"
    assert R.positioning_label(50) == "NEUTRAL"
    assert R.positioning_label(85) == "HIGH"
    assert R.positioning_label(99) == "VERY HIGH"


# --- correlation -----------------------------------------------------------
def test_avg_pairwise_corr_matches_brute_force():
    rng = np.random.default_rng(3)
    common = rng.normal(0, 0.01, 200)
    data = {f"S{i}": 100 * np.exp(np.cumsum(common + rng.normal(0, 0.01, 200))) for i in range(6)}
    panel = pd.DataFrame(data, index=pd.bdate_range("2024-01-01", periods=200))
    fast = R.avg_pairwise_corr(panel, 60).iloc[-1]
    cm = panel.pct_change(fill_method=None).iloc[-60:].corr().to_numpy()
    brute = cm[~np.eye(6, dtype=bool)].mean()
    assert fast == pytest.approx(brute, abs=1e-9)


def test_avg_pairwise_corr_skips_thin_windows():
    p = pd.DataFrame({"A": _walk(200), "B": _walk(200, 2)})
    p.iloc[-70:, 1] = np.nan
    out = R.avg_pairwise_corr(p, 60, min_coverage=0.9)
    assert not out.empty and out.index.max() < p.index[-1]


# --- cross-asset PCA -------------------------------------------------------
def _cross_asset(n: int = 900) -> pd.DataFrame:
    rng = np.random.default_rng(5)
    g = rng.normal(0, 0.008, n)
    idx = pd.bdate_range("2018-01-01", periods=n)
    out = {}
    for i, (t, b) in enumerate({"SPY": 1.0, "HYG": 0.6, "CPER": 0.7, "TLT": -0.5, "UUP": -0.4}.items()):
        out[t] = 100 * np.exp(np.cumsum(b * g + rng.normal(0, 0.004, n)))
    return pd.DataFrame(out, index=idx)


def test_pca_signs_factor_with_equities():
    c = _cross_asset()
    out = R.cross_asset_pca(c)
    spy20 = np.log(c["SPY"]).diff(20).reindex(out.index)
    assert out["factor"].corr(spy20) > 0.7


def test_pca_has_no_lookahead():
    c = _cross_asset()
    full = R.cross_asset_pca(c)
    cut = R.cross_asset_pca(c.iloc[:700])
    join = full.index.intersection(cut.index)
    pd.testing.assert_series_equal(full.loc[join, "factor"], cut.loc[join, "factor"], check_names=False)


def test_growth_snapshot_states():
    idx = pd.bdate_range("2024-01-01", periods=300)
    z = pd.Series(np.r_[np.zeros(290), np.linspace(-3, -2.5, 10)], index=idx)
    out = R.growth_snapshot(pd.DataFrame({"factor": z, "z_6m": z, "explained": 0.4}))
    assert out["state"] == "EXTREME DOWN · TURNING" and out["explained"] == pytest.approx(40)


# --- boards ----------------------------------------------------------------
def test_scoreboard_sorted_with_benchmark_spread():
    p = pd.DataFrame({"SPY": [100.0] * 20 + [110.0], "A": [100.0] * 20 + [130.0], "B": [100.0] * 20 + [90.0]},
                     index=pd.bdate_range("2024-01-01", periods=21))
    t = R.scoreboard(p, {"A": "a", "B": "b"}, days=10)
    assert list(t["ticker"]) == ["A", "B"]
    assert t["vs_bench"].iloc[0] == pytest.approx(20.0)


def test_scoreboard_skips_assets_without_history():
    p = _prices(100, ["SPY", "A"])
    p.loc[p.index[:-5], "A"] = np.nan
    assert R.scoreboard(p, {"A": "a"}, days=10).empty


def test_basket_index_handles_late_listing():
    idx = pd.bdate_range("2024-01-01", periods=5)
    p = pd.DataFrame({"A": [100.0, 101.0, 102.0, 103.0, 104.0], "B": [np.nan, np.nan, 50.0, 100.0, 100.0]}, index=idx)
    out = R.basket_index(p, ["A", "B"])
    # Day 3 uses A only (B has no return yet); day 4 averages A's +0.98% and B's +100%.
    assert out[idx[2]] / out[idx[1]] - 1 == pytest.approx(102 / 101 - 1)
    assert out[idx[3]] / out[idx[2]] - 1 == pytest.approx(((103 / 102 - 1) + 1.0) / 2)


def test_technical_row_flags_oversold():
    s = pd.Series(np.r_[np.full(250, 100.0) + np.sin(np.arange(250)), np.linspace(100, 60, 20)],
                  index=pd.bdate_range("2023-01-01", periods=270))
    row = R.technical_row(s)
    assert row["flag"] == "OVERSOLD" and row["dd_52w"] < -30


def test_technical_board_override_uses_own_calendar():
    p = _prices(300, ["SPY"])
    btc = _walk(400, 9)
    t = R.technical_board(p, {"BTC-USD": "Bitcoin"}, overrides={"BTC-USD": btc})
    assert t["last"].iloc[0] == pytest.approx(btc.iloc[-1])


def test_monthly_spread_rank():
    idx = pd.date_range("2020-01-01", "2021-06-30", freq="B")
    a = pd.Series(np.linspace(100, 200, len(idx)), index=idx)
    m = R.monthly_spread(a, pd.Series(100.0, index=idx))
    snap = R.monthly_spread_snapshot(m)
    assert snap["n"] == len(m) and 1 <= snap["rank"] <= snap["n"]


# --- internals -------------------------------------------------------------
def test_internals_counts_and_divergence():
    idx = pd.bdate_range("2024-01-01", periods=260)
    up = {f"U{i}": 100 + np.arange(260.0) for i in range(4)}
    dn = {"D": 400 - np.arange(260.0)}
    i = R.internals_frame(pd.DataFrame({**up, **dn}, index=idx))
    assert i["adv"].iloc[-1] == 4 and i["dec"].iloc[-1] == 1
    assert i["pct_200"].iloc[-1] == pytest.approx(80.0)
    spx = pd.Series(100.0, index=idx)
    spx.iloc[-1] = 99.0
    d = R.divergence_days(i, spx, min_adv=4, max_ret=-0.5)
    assert list(d.index) == [idx[-1]]


def test_disconnect_read():
    idx = pd.bdate_range("2024-01-01", periods=40)
    i = pd.DataFrame({"pct_50": np.linspace(30, 70, 40)}, index=idx)
    out = R.disconnect_snapshot(i, pd.Series(100.0, index=idx))
    assert out["read"] == "Participation broadening under a weak index"


# --- rates -----------------------------------------------------------------
def test_yield_rsi_episodes_measures_fall_after_peak():
    w = pd.date_range("2022-01-07", periods=120, freq="W-FRI")
    y = pd.Series(np.r_[np.linspace(1, 5, 80), np.linspace(5, 3.5, 40)], index=w)
    ep = R.yield_rsi_episodes(y)
    assert len(ep) == 1
    assert ep["peak_yield"].iloc[0] == pytest.approx(5.0)
    assert ep["fall_pct"].iloc[0] < -10


def test_change_shock_in_bps():
    y = pd.Series(np.r_[np.full(100, 5.0), np.full(10, 4.7)], index=pd.bdate_range("2024-01-01", periods=110))
    assert R.change_shock(y, 10)["bps"] == pytest.approx(-30.0)


def test_excess_cape_yield_identity():
    idx = pd.date_range("2000-01-01", "2025-12-01", freq="MS")
    cpi = pd.Series(100 * 1.02 ** (np.arange(len(idx)) / 12), index=idx)
    out = R.excess_cape_yield(pd.Series(25.0, index=idx), pd.Series(5.0, index=idx), cpi)
    # 4% earnings yield less a 3% real yield (5% nominal, 2% inflation).
    assert out.iloc[-1] == pytest.approx(1.0, abs=0.01)


def test_excess_cape_carries_late_cpi_forward():
    idx = pd.date_range("2000-01-01", "2025-12-01", freq="MS")
    cpi = pd.Series(100 * 1.02 ** (np.arange(len(idx)) / 12), index=idx).iloc[:-2]
    out = R.excess_cape_yield(pd.Series(25.0, index=idx), pd.Series(5.0, index=idx), cpi)
    assert out.index[-1] == idx[-1]


# --- event studies ---------------------------------------------------------
def test_fresh_applies_gap():
    idx = pd.bdate_range("2024-01-01", periods=30)
    f = pd.Series(False, index=idx)
    f.iloc[[2, 3, 10, 25]] = True
    assert list(R.fresh(f, 10)) == [idx[2], idx[25]]
    assert list(R.fresh(f, 8)) == [idx[2], idx[10], idx[25]]


def test_event_study_returns_and_excludes_incomplete():
    idx = pd.bdate_range("2024-01-01", periods=100)
    p = pd.Series(100 * 1.01 ** np.arange(100.0), index=idx)
    out = R.event_study(p, pd.DatetimeIndex([idx[10], idx[98]]), (5,))
    row = out.iloc[0]
    assert row["n"] == 1  # the second event has no complete 5-day window
    assert row["mean"] == pytest.approx((1.01 ** 5 - 1) * 100)
    assert row["hit"] == 100.0 and row["mae"] > 0


def test_event_study_base_rate_starts_at_first_event():
    idx = pd.bdate_range("2024-01-01", periods=200)
    p = pd.Series(np.r_[100 * 0.99 ** np.arange(100.0), 37 * 1.01 ** np.arange(100.0)], index=idx)
    out = R.event_study(p, pd.DatetimeIndex([idx[120]]), (5,)).iloc[0]
    assert out["base_hit"] == 100.0  # the falling era before the event is excluded


def test_strength_label_needs_sample_and_edge():
    assert R.strength_label({"n": 5, "p_mean": 0.01, "edge_hit": 20}) == "Too few events"
    assert R.strength_label({"n": 30, "p_mean": 0.01, "edge_hit": 20}) == "Real edge"
    assert R.strength_label({"n": 30, "p_mean": 0.15, "edge_hit": 3}) == "Weak edge"
    assert R.strength_label({"n": 30, "p_mean": 0.6, "edge_hit": 10}) == "No edge vs base rate"


def test_signal_library_cot_flags_lag_publication():
    p = _prices(1800)
    idx = pd.date_range("2015-01-06", periods=300, freq="W-TUE")
    cot = pd.DataFrame({"VIX": np.r_[np.arange(299, 0, -1.0), -5.0]}, index=idx)
    sig = {s.key: s for s in R.signal_library(p.assign(**{"^VIX": p["^VIX"]}), cot=cot)}["cot_vix"]
    fired = sig.flag[sig.flag].index
    assert (fired.dayofweek == 4).all()  # Tuesday as-of date moved to Friday publication


def test_run_playbook_shape():
    p = _prices(1800)
    pb = R.run_playbook(p, R.signal_library(p))
    assert {"signal", "evidence", "base_hit", "last_event"} <= set(pb.columns)


# --- seasonality -----------------------------------------------------------
def test_midterm_seasonality_uses_oct_to_apr():
    idx = pd.date_range("2017-01-31", "2024-12-31", freq="ME")
    p = pd.Series(100.0, index=idx)
    p.loc["2023-04-30":] = 110.0
    t = R.midterm_seasonality(p)
    row = t[t["cycle"] == "2022-23"].iloc[0]
    assert row["six_m"] == pytest.approx(10.0)
    assert list(t["cycle"]) == ["2018-19", "2022-23"]


def test_non_midterm_base_excludes_midterm_years():
    idx = pd.date_range("2018-01-31", "2024-12-31", freq="ME")
    assert R.non_midterm_base(pd.Series(100.0, index=idx))["n"] == 4  # 2019, 2020, 2021, 2023


# --- decision rules --------------------------------------------------------
def _rule(key: str, fired: bool, core: bool = False, available: bool = True) -> Rule:
    return Rule(key, key, 1.0, "x", fired, core=core, available=available)


def test_capitulation_verdict_thresholds():
    rules = [_rule(f"c{i}", i < 3, core=True) for i in range(7)] + [_rule("x", True)]
    v = R.capitulation_verdict(rules)
    assert v["state"] == "ARMED" and v["core_fired"] == 3 and v["all_fired"] == 4
    rules = [_rule(f"c{i}", i < 5, core=True) for i in range(7)]
    assert R.capitulation_verdict(rules)["state"] == "CAPITULATION"


def test_capitulation_verdict_ignores_unavailable():
    rules = [_rule("a", True, core=True), _rule("b", False, core=True, available=False)]
    assert R.capitulation_verdict(rules)["core_total"] == 1


def test_capitulation_rules_fire_on_a_crash():
    p = _prices(1800)
    p.iloc[-4:, p.columns.get_loc("QQQ")] = p["QQQ"].iloc[-5] * np.array([0.9, 0.82, 0.75, 0.7])
    p.iloc[-1, p.columns.get_loc("^VXN")] = 40.0
    p.iloc[-2, p.columns.get_loc("^VIX")] = p["^VIX"].iloc[-3] * 1.6
    rules = {r.key: r for r in R.capitulation_rules(p, put_call=pd.Series(0.9, index=p.index[-5:]))}
    assert rules["qqq_rsi"].fired and rules["vxn"].fired and rules["vix_spike"].fired and rules["put_call"].fired
    assert rules["fear_greed"].available is False


def test_guardrail_verdict_credit_breaks_faster():
    good = [_rule(k, True) for k in ("breadth", "pct200", "hy", "corr")]
    assert R.guardrail_verdict(good)["state"] == "INTACT"
    assert R.guardrail_verdict(good + [_rule("contain", False)])["state"] == "WATCH"
    two = good + [_rule("contain", False), _rule("ig", False)]
    assert R.guardrail_verdict(two)["state"] == "BREAKING"


def test_passthrough_limit_defaults_on_thin_history():
    s = pd.Series(np.linspace(1, 2, 30), index=pd.bdate_range("2024-01-01", periods=30))
    assert R._passthrough_limit(s, s) == pytest.approx(0.135)


def test_contain_rule_fails_on_abnormal_passthrough():
    idx = pd.bdate_range("2024-01-01", periods=120)
    ccc, ig = pd.Series(10.0, index=idx), pd.Series(1.0, index=idx)
    ccc.iloc[-1] = 11.0   # +100bp
    ig.iloc[-1] = 1.30    # +30bp, 0.30x
    rules = {r.key: r for r in R.system_guardrails(trend_breadth=None, pct_200=None, ig=ig, hy=None, ccc=ccc,
                                                   sofr_spread=None, spx_corr=None, nfci=None)}
    assert rules["contain"].fired is False
    assert rules["breadth"].available is False


def test_leverage_gate_counts_hits():
    idx = pd.bdate_range("2024-01-01", periods=300)
    calm = pd.Series(np.linspace(0.6, 0.1, 300), index=idx)
    out = R.leverage_gate(vix_cot={"pct_all": 2.0, "net_oi": -12.0}, spx_corr=calm,
                          gamma=pd.Series([-1.0], index=[idx[-1]]), vix=pd.Series([18.0], index=[idx[-1]]))
    assert out["state"] == "NO LEVERAGE" and out["hits"] == 3
    out = R.leverage_gate(vix_cot={"pct_all": 60.0, "net_oi": 2.0}, spx_corr=None, gamma=None, vix=None)
    assert out["state"] == "OPEN"


def test_rotation_stage_flags_stretched_laggards():
    p = _prices(1200, ["SPY", "RSP", "IWM", "XLU", "XLF", "XLV", "ITB", "XLK", "SMH"])
    p.iloc[-20:, [p.columns.get_loc(c) for c in ("XLK", "SMH")]] *= np.linspace(1, 0.6, 20).reshape(-1, 1)
    out = R.rotation_stage(p)
    assert out["spread_z"] >= 2 and out["stage"] in ("STRETCHED", "ROLLING OVER")


# --- posture precedence ----------------------------------------------------
@pytest.mark.parametrize("guard,cap,fired,expect", [
    ("BREAKING", "CAPITULATION", 5, "Defensive"),   # a broken system overrides a washout
    ("INTACT", "CAPITULATION", 3, "Washed out with confirmation"),
    ("WATCH", "ARMED", 1, "Washed out, turn not confirmed"),
    ("INTACT", "QUIET", 0, "No extreme"),
])
def test_model_posture_precedence(guard, cap, fired, expect):
    out = K.model_posture({"state": guard}, {"state": cap}, fired, 5, {"state": "OPEN"}, {})
    assert out["head"].startswith(expect)


def test_model_posture_reports_rotation_when_no_washout():
    out = K.model_posture({"state": "INTACT"}, {"state": "BUILDING"}, 0, 5, {"state": "REDUCED"},
                          {"stage": "ACTIVE", "read": "r"})
    assert out["head"] == "Rotation regime: active" and out["gate"] == "Leverage gate partly closed."


# --- parsers ---------------------------------------------------------------
def test_parse_cboe_daily():
    payload = {"SUM OF ALL PRODUCTS": [{"name": "VOLUME", "call": 10, "put": 20, "total": 30}, {"name": "OPEN INTEREST", "call": 1, "put": 2}],
               "CBOE VOLATILITY INDEX (VIX)": [{"name": "VOLUME", "call": 5, "put": 1, "total": 6}]}
    out = RD.parse_cboe_daily(payload)
    assert out["total_put"] == 20 and out["vix_call"] == 5 and "spx_put" not in out


def test_parse_nasdaq_short_interest():
    payload = {"data": {"shortInterestTable": {"rows": [{"settlementDate": "10/15/2026", "interest": "82,001,618"}]}}}
    s = RD.parse_nasdaq_short_interest(payload)
    assert s.iloc[0] == 82001618 and s.index[0] == pd.Timestamp("2026-10-15")
    assert RD.parse_nasdaq_short_interest({"data": None}).empty


def test_parse_multpl_table():
    html = "<tr><td>Nov 6, 2026</td><td> &#x2002; 38.91 </td></tr><tr><td>Nov 1, 2026</td><td>39.82</td></tr>"
    s = RD.parse_multpl_table(html)
    assert s.loc["2026-11-06"] == pytest.approx(38.91) and len(s) == 2


def test_parse_fng_components():
    out = RD.parse_fng_components({"put_call_options": {"score": 13.4, "rating": "extreme fear", "data": []}})
    assert out == [{"name": "Put/call", "score": 13.4, "rating": "extreme fear"}]


# --- fallback chain --------------------------------------------------------
def test_cellwise_chain_keeps_history_for_a_ticker_live_dropped(tmp_path, monkeypatch):
    from src import data as D
    idx = pd.bdate_range("2024-01-01", periods=3)
    cache = pd.DataFrame({"A": [1.0, 2.0, 3.0], "B": [4.0, 5.0, 6.0]}, index=idx)
    monkeypatch.setattr(D, "_load_frame_cache", lambda f: cache)
    monkeypatch.setattr(D, "_load_seed_frame", lambda f: pd.DataFrame())
    monkeypatch.setattr(D, "_save_frame_cache", lambda f, df: None)
    live = pd.DataFrame({"A": [9.0]}, index=idx[-1:])  # Yahoo dropped B from the bulk download
    out = RD._cellwise_chain(key="t", live=live, cache_file="x", seed_file="y", source="s")
    assert out["A"].iloc[-1] == 9.0 and out["B"].iloc[-1] == 6.0


def test_cellwise_chain_reports_unavailable(monkeypatch):
    from src import data as D
    monkeypatch.setattr(D, "_load_frame_cache", lambda f: pd.DataFrame())
    monkeypatch.setattr(D, "_load_seed_frame", lambda f: pd.DataFrame())
    assert RD._cellwise_chain(key="t2", live=pd.DataFrame(), cache_file="x", seed_file="y", source="s").empty
    assert D.get_provenance()["t2"]["kind"] == "unavailable"


# --- figures ---------------------------------------------------------------
def test_figures_build_without_network():
    s = _walk(400)
    assert RC.stacked([("a", s)], [("b", s / 2)]).data
    assert RC.lines([("a", s), ("b", s * 2)], levels=[(100, "x")], zones=[(0, 1, "rgba(0,0,0,0.1)")]).data
    assert RC.signed_bars(s.pct_change().dropna()).data
    assert RC.cot_figure(s, s.resample("W-TUE").last(), "X").data
    t = pd.DataFrame({"market": ["A", "B"], "pct_all": [5.0, 90.0], "pct_3y": [40.0, 88.0]})
    assert RC.crowding_dumbbell(t).data
    assert RC.hbars(pd.DataFrame({"n": ["a", "b"], "v": [1.0, -2.0]}), "n", "v", ref=0.5, ref_label="avg").data
    i = pd.DataFrame({"adv": np.arange(400.0)}, index=s.index)
    assert RC.divergence_scatter(i, s).data


def test_figures_tolerate_empty_series():
    e = pd.Series(dtype=float)
    assert RC.lines([("a", e)]) is not None
    assert RC.stacked([("a", e)], [("b", e)]) is not None


# --- reconciliation regressions -------------------------------------------
def test_cta_positioning_uses_the_e_mini_contract_not_micro(monkeypatch):
    """The name filter also matched the Micro contract, which sat on the other side."""
    import io
    import zipfile
    from src import data as D

    csv = (
        "Market_and_Exchange_Names,As_of_Date_In_Form_YYMMDD,CFTC_Contract_Market_Code,"
        "Lev_Money_Positions_Long_All,Lev_Money_Positions_Short_All,Open_Interest_All\n"
        '"E-MINI S&P 500 - CHICAGO MERCANTILE EXCHANGE",261103,13874A,100,500,1000\n'
        '"MICRO E-MINI S&P 500 INDEX - CHICAGO MERCANTILE EXCHANGE",261103,13874U,300,100,1000\n'
    )
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("FinFutYY.txt", csv)

    class Resp:
        ok = True
        content = buf.getvalue()

    monkeypatch.setattr(D.requests, "get", lambda *a, **k: Resp())
    s = D.cftc_cta_positioning()
    assert s.iloc[-1] == pytest.approx(-40.0)


def _stub_put_call(monkeypatch, *, cache, daily, live=9.9):
    from src import data as D
    empty = pd.Series(dtype=float)
    monkeypatch.setattr(D, "_cboe_put_call_history", lambda k, n: empty)
    monkeypatch.setattr(D, "_ycharts_recent_indicator", lambda s, n: empty)
    monkeypatch.setattr(D, "_load_series_cache", lambda f, n: cache)
    monkeypatch.setattr(D, "_save_series_cache", lambda *a: None)
    monkeypatch.setattr(D, "_cboe_daily_ratio", lambda k, n: daily)
    monkeypatch.setattr(D, "_cboe_live_put_call_ratio", lambda s: live)
    return D


def test_official_daily_ratio_replaces_cached_proxy_rows(monkeypatch):
    day = pd.Timestamp("2026-11-06")
    D = _stub_put_call(monkeypatch, cache=pd.Series([1.126], index=[day]), daily=pd.Series([0.59], index=[day]))
    monkeypatch.setattr(D, "last_completed_trading_day", lambda: day)
    out = D._official_put_call_series(kind="equity", ycharts_slug="x", name="put_call", cache_filename="f", live_symbols=("SPY",))
    assert out.iloc[-1] == pytest.approx(0.59)


def test_proxy_only_fills_a_day_with_no_official_row(monkeypatch):
    day = pd.Timestamp("2026-11-06")
    D = _stub_put_call(monkeypatch, cache=pd.Series(dtype=float), daily=pd.Series([0.6], index=[day - pd.Timedelta(days=1)]))
    monkeypatch.setattr(D, "last_completed_trading_day", lambda: day)
    out = D._official_put_call_series(kind="equity", ycharts_slug="x", name="pc_test", cache_filename="f", live_symbols=("SPY",))
    assert out.iloc[-1] == pytest.approx(9.9)
    assert D.get_provenance()["pc_test"]["kind"] == "fallback"


def test_cboe_daily_ratio_by_kind(monkeypatch):
    from src import data as D
    idx = pd.bdate_range("2026-11-02", periods=2)
    monkeypatch.setattr(RD, "cboe_volumes", lambda: pd.DataFrame({"index_put": [10.0, 20.0], "index_call": [5.0, 0.0]}, index=idx))
    assert D._cboe_daily_ratio("index", "x").to_dict() == {idx[0]: 2.0}  # a zero denominator is dropped
    assert D._cboe_daily_ratio("equity", "x").empty


def test_cboe_up_to_date_is_not_reported_as_failure(monkeypatch):
    from src import data as D
    last = pd.Timestamp("2026-11-06")
    known = pd.DataFrame({"total_put": [1.0]}, index=[last])
    monkeypatch.setattr(D, "_load_frame_cache", lambda f: pd.DataFrame())
    monkeypatch.setattr(D, "_load_seed_frame", lambda f: known)
    monkeypatch.setattr(D, "last_completed_trading_day", lambda: last)
    monkeypatch.setattr(RD, "fetch_cboe_range", lambda *a: pd.DataFrame())
    D.reset_fetch_memo()
    assert not RD.cboe_volumes().empty
    assert D.get_provenance()["cboe_volumes"]["kind"] == "primary"
    D.reset_fetch_memo()


def test_move_index_prefers_real_move_and_falls_back_to_proxy(monkeypatch):
    from src import data as D
    idx = pd.bdate_range("2023-01-02", periods=600)
    real = pd.Series(100.0, index=idx)
    tlt = _walk(600, 4, start="2023-01-02")
    monkeypatch.setattr(D, "yf_series", lambda t, **k: real if t == "^MOVE" else tlt)
    D.reset_fetch_memo()
    assert D.move_index().iloc[-1] == 100.0
    assert D.get_provenance()["move_index"]["kind"] == "primary"
    monkeypatch.setattr(D, "yf_series", lambda t, **k: pd.Series(dtype=float) if t == "^MOVE" else tlt)
    D.reset_fetch_memo()
    assert D.move_index().iloc[-1] != 100.0
    assert D.get_provenance()["move_index"]["kind"] == "proxy"
    D.reset_fetch_memo()


def test_put_call_rule_keeps_a_five_session_memory():
    p = _prices(1800)
    pc = pd.Series([0.6, 0.6, 0.89, 0.6, 0.59], index=p.index[-5:])
    rule = {r.key: r for r in R.capitulation_rules(p, put_call=pc)}["put_call"]
    assert rule.fired and rule.value == pytest.approx(0.59)


def test_desk_freshness_flags_saved_and_stale_data():
    today = pd.Timestamp("2026-11-07")
    live = {k: {"kind": "primary"} for k in K.KEY_FEEDS}
    ok = K.desk_freshness(live, {"Prices": pd.Timestamp("2026-11-06")}, today)
    assert ok["issues"] == []
    seed = {**live, "research_prices": {"kind": "fallback", "note": "seed through 2026-10-01"}}
    assert "saved data" in K.desk_freshness(seed, {"Prices": pd.Timestamp("2026-11-06")}, today)["issues"][0]
    assert "stale" in K.desk_freshness(live, {"Prices": pd.Timestamp("2026-10-20")}, today)["issues"][0]
    assert "no price data" in K.desk_freshness(live, {"Prices": None}, today)["issues"]


# --- exposure leverage cap -------------------------------------------------
def _desk_state(gate: str, fired: int, total: int = 5) -> dict:
    confirm = [_rule(f"c{i}", i < fired) for i in range(total)]
    return {"gate": {"state": gate}, "confirm": confirm}


def _rec(net: int, label: str = "SCALE IN", conviction: str = "Medium") -> dict:
    return {"net_pct": net, "hedge_pct": 0.0, "label": label, "conviction": conviction, "rationale": "composite 32"}


@pytest.mark.parametrize("gate,fired,expect", [
    ("NO LEVERAGE", 0, 100),
    ("NO LEVERAGE", 5, 100),   # confirmation never overrides a closed gate
    ("REDUCED", 2, 100),
    ("REDUCED", 3, 130),
    ("OPEN", 2, 100),
    ("OPEN", 3, 130),
])
def test_leverage_cap_table(gate, fired, expect):
    out = K.apply_leverage_cap(_rec(130), K.leverage_permission(_desk_state(gate, fired)))
    assert out["net_pct"] == expect


def test_cap_records_why_and_the_original_figure():
    out = K.apply_leverage_cap(_rec(130), K.leverage_permission(_desk_state("NO LEVERAGE", 2)))
    assert out["uncapped_pct"] == 130
    assert out["cap_reason"] == "leverage gate closed"
    assert "capped from 130% to 100%" in out["rationale"]


def test_cap_explains_missing_confirmation():
    perm = K.leverage_permission(_desk_state("OPEN", 2))
    assert perm["reason"] == "price confirmation 2 of 5, needs 3"


def test_cap_covers_the_base_mapping_not_just_the_cluster():
    out = K.apply_leverage_cap(_rec(115), K.leverage_permission(_desk_state("REDUCED", 0)))
    assert out["net_pct"] == 100 and out["uncapped_pct"] == 115


@pytest.mark.parametrize("net", [20, 50, 90, 100])
def test_cap_never_changes_an_unlevered_recommendation(net):
    out = K.apply_leverage_cap(_rec(net), K.leverage_permission(_desk_state("NO LEVERAGE", 0)))
    assert out["net_pct"] == net and "uncapped_pct" not in out


def test_cap_leaves_the_input_untouched():
    rec = _rec(130)
    K.apply_leverage_cap(rec, K.leverage_permission(None))
    assert rec["net_pct"] == 130


def test_no_desk_means_no_leverage():
    perm = K.leverage_permission(None)
    assert perm["allowed"] is False and perm["reason"] == "decision desk unavailable"
    assert K.apply_leverage_cap(_rec(130), perm)["net_pct"] == 100


def test_unavailable_confirmations_do_not_count():
    desk = _desk_state("OPEN", 3)
    desk["confirm"] = [Rule(f"c{i}", "x", 1.0, "x", True, available=False) for i in range(3)] + desk["confirm"][3:]
    assert K.leverage_permission(desk)["allowed"] is False


def test_no_data_recommendation_passes_through():
    rec = {"net_pct": None, "label": "NO DATA", "conviction": "—"}
    assert K.apply_leverage_cap(rec, K.leverage_permission(None)) == rec


def test_bottom_cluster_conviction_waits_for_confirmation():
    rec = _rec(130, "BOTTOM CLUSTER (8 themes) — ADD", "High")
    assert K.apply_leverage_cap(rec, K.leverage_permission(_desk_state("NO LEVERAGE", 2)))["conviction"] == "Medium"
    assert K.apply_leverage_cap(rec, K.leverage_permission(_desk_state("OPEN", 3)))["conviction"] == "High"


def test_gate_closed_alone_does_not_lower_conviction_after_confirmation():
    rec = _rec(130, "BOTTOM CLUSTER (8 themes) — ADD", "High")
    out = K.apply_leverage_cap(rec, K.leverage_permission(_desk_state("NO LEVERAGE", 4)))
    assert out["net_pct"] == 100 and out["conviction"] == "High"


def test_top_cluster_keeps_its_conviction():
    rec = _rec(30, "TOP CLUSTER (5 themes) — CUT", "High")
    assert K.apply_leverage_cap(rec, K.leverage_permission(_desk_state("OPEN", 0)))["conviction"] == "High"


# --- SKEW is context, not core ---------------------------------------------
def test_skew_is_not_a_core_rule():
    assert "skew" not in R.CORE_RULES
    rules = {r.key: r for r in R.capitulation_rules(_prices(1800))}
    assert rules["skew"].core is False
    assert {k for k, r in rules.items() if r.core} == set(R.CORE_RULES)


def test_skew_firing_does_not_change_the_verdict():
    p = _prices(1800)
    hist = R.capitulation_history(p)
    calm = hist.assign(**{c: False for c in hist.columns})
    fired = calm.assign(skew=True)
    a = R.capitulation_verdict(R.capitulation_rules(p, history=calm))
    b = R.capitulation_verdict(R.capitulation_rules(p, history=fired))
    assert a["core_fired"] == b["core_fired"] == 0
    assert a["state"] == b["state"] == "QUIET"
    assert b["all_fired"] == a["all_fired"] + 1  # still visible as context


def test_backtest_buckets_ignore_skew():
    p = _prices(1800)
    hist = R.capitulation_history(p).astype(bool)
    with_skew = R.capitulation_backtest(hist.assign(skew=True), R.px(p, "QQQ"))
    without = R.capitulation_backtest(hist.assign(skew=False), R.px(p, "QQQ"))
    pd.testing.assert_frame_equal(with_skew, without)
