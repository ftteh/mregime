"""Scoring engine: true windows, noise, stale feeds, coverage, the cluster and
the exposure mapping. Deterministic, no network."""
from __future__ import annotations

import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import data as D
from src import indicators as I
from src.config import BUCKET_WEIGHTS, INDICATORS_BY_KEY, theme_for

END = "2026-11-06"


def _raw(series: dict[str, pd.Series]) -> I.RawFrame:
    meta = {k: {"last": float(v.iloc[-1]), "as_of": v.index[-1].to_pydatetime(), "n": len(v)} for k, v in series.items()}
    return I.RawFrame(series=series, meta=meta)


def _full_raw(years: int = 5, seed: int = 3, skip: tuple[str, ...] = ()) -> I.RawFrame:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(end=END, periods=252 * years)
    return _raw({k: pd.Series(np.cumsum(rng.normal(0, 1, len(idx))), index=idx)
                 for k in INDICATORS_BY_KEY if k not in skip})


# --- percentile window -----------------------------------------------------
def test_percentile_uses_the_full_three_years_when_they_exist():
    """A 3-year series used to be ranked over 18 months (the window was halved)."""
    idx = pd.bdate_range(end=END, periods=780)
    # First 300 days high, the rest low and rising: only a full window sees the highs.
    s = pd.Series(np.r_[np.full(300, 10.0), np.linspace(0.0, 5.0, 480)], index=idx)
    pct = I.rolling_percentile(s)
    full = (s.iloc[-756:] <= s.iloc[-1]).mean() * 100
    assert pct.iloc[-1] == pytest.approx(full)
    assert pct.iloc[-1] < 70                      # the old half-window said 100


def test_percentile_needs_a_year_and_grows_to_three():
    idx = pd.bdate_range(end=END, periods=400)
    s = pd.Series(np.arange(400.0), index=idx)
    pct = I.rolling_percentile(s)
    assert pct.iloc[:251].isna().all() and pct.iloc[251] == 100.0
    assert I.rolling_percentile(s.iloc[:200]).isna().all()


def test_latest_percentile_counts_ties_like_the_engine():
    s = pd.Series([1.0, 2.0, 2.0, 2.0])
    assert I.latest_percentile(s) == 100.0


def test_weekend_print_shows_on_the_next_business_day_and_long_gaps_stay_empty():
    s = pd.Series([1.0, 2.0, 9.0], index=pd.DatetimeIndex(["2026-01-05", "2026-01-31", "2026-06-01"]))  # Jan 31 = Saturday
    out = I._business_daily(s, limit=10)
    assert out.loc["2026-02-02"] == 2.0                    # Monday carries Saturday's print
    assert out.loc["2026-01-30"] != 2.0 or np.isnan(out.loc["2026-01-30"])
    assert np.isnan(out.loc["2026-03-16"])                 # months later: no flat line of one old value
    assert out.iloc[-1] == 9.0
    last_sat = pd.Series([1.0, 2.0], index=pd.DatetimeIndex(["2026-01-05", "2026-01-31"]))
    assert I._business_daily(last_sat).iloc[-1] == 2.0     # a final weekend print is not lost


# --- smoothing -------------------------------------------------------------
def test_one_noisy_print_no_longer_swings_the_put_call_score():
    """Daily put/call is mostly noise: 0.44 one day, 0.89 the next. Unsmoothed,
    its score moved 30+ points on about one day in four."""
    from dataclasses import replace

    rng = np.random.default_rng(5)
    idx = pd.bdate_range(end=END, periods=900)
    level = 0.6 + 0.1 * np.sin(np.arange(len(idx)) / 60.0)          # slow regime
    base = pd.Series(level + rng.normal(0, 0.05, len(idx)), index=idx)
    spiked = base.copy()
    spiked.iloc[-1] += 0.15                                          # one noisy print
    spec = INDICATORS_BY_KEY["put_call"]
    assert spec.smooth == 10 and INDICATORS_BY_KEY["index_put_call"].smooth == 10

    def swing(sp):
        score = lambda s: I._score_history(I._prepared("put_call", sp, _raw({"put_call": s})), sp).iloc[-1]
        return abs(score(spiked) - score(base))

    raw_swing, smooth_swing = swing(replace(spec, smooth=1)), swing(spec)
    assert raw_swing > 25
    assert smooth_swing < raw_swing / 2


# --- stale feeds -----------------------------------------------------------
def test_a_dead_feed_stops_voting():
    raw = _full_raw()
    dead = raw.series["gamma_exposure"].iloc[:-60]          # last print ~12 weeks before the rest
    raw.series["gamma_exposure"] = dead
    raw.meta["gamma_exposure"] = {"last": float(dead.iloc[-1]), "as_of": dead.index[-1].to_pydatetime(), "n": len(dead)}
    per = I.per_indicator_history(raw)
    assert np.isnan(per["gamma_exposure"].iloc[-1])
    assert per["gamma_exposure"].notna().sum() > 0          # its history is still there
    scores = I.score_indicators(raw, per).set_index("key")
    assert scores.loc["gamma_exposure", "status"] == "stale" and np.isnan(scores.loc["gamma_exposure", "score"])
    assert scores.loc["vix", "status"] == "ok"
    assert "gamma_exposure" not in I.contributions(scores.reset_index())["key"].tolist()


def test_a_weekly_feed_inside_its_allowance_still_votes():
    raw = _full_raw()
    weekly = raw.series["cta_positioning"].iloc[:-8]        # 8 business days old: normal for CFTC
    raw.series["cta_positioning"] = weekly
    raw.meta["cta_positioning"] = {"last": float(weekly.iloc[-1]), "as_of": weekly.index[-1].to_pydatetime(), "n": len(weekly)}
    per = I.per_indicator_history(raw)
    assert not np.isnan(per["cta_positioning"].iloc[-1])
    daily = raw.series["vix"].iloc[:-12]                    # the same lag on a daily feed is a dead feed
    raw.series["vix"] = daily
    raw.meta["vix"] = {"last": float(daily.iloc[-1]), "as_of": daily.index[-1].to_pydatetime(), "n": len(daily)}
    assert np.isnan(I.per_indicator_history(raw)["vix"].iloc[-1])


def test_short_history_is_warmup_not_a_score():
    raw = _full_raw()
    short = raw.series["fear_greed"].iloc[-200:]
    raw.series["fear_greed"] = short
    raw.meta["fear_greed"] = {"last": float(short.iloc[-1]), "as_of": short.index[-1].to_pydatetime(), "n": len(short)}
    row = I.score_indicators(raw).set_index("key").loc["fear_greed"]
    assert row["status"] == "warmup" and np.isnan(row["score"])
    assert row["window_years"] == pytest.approx(200 / 252, abs=0.02)


def test_display_only_indicator_never_scores():
    raw = _full_raw()
    assert INDICATORS_BY_KEY["gamma_flip_zone"].display_only
    per = I.per_indicator_history(raw)
    assert "gamma_flip_zone" not in per.columns
    row = I.score_indicators(raw, per).set_index("key").loc["gamma_flip_zone"]
    assert row["status"] == "display" and np.isnan(row["score"])


# --- one-sided indicators --------------------------------------------------
def test_event_flag_is_neutral_while_off():
    """Ranking the zeros scored 'off' as the share of zeros in the window."""
    idx = pd.bdate_range(end=END, periods=1000)
    x = pd.Series(0.0, index=idx)
    x.iloc[300:500] = np.linspace(0.1, 1.5, 200)            # one active episode, then off again
    spec = INDICATORS_BY_KEY["curve_resteep_2s10s"]
    assert spec.transform == "active_only"
    score = I._score_history(x, spec)
    assert score.iloc[-1] == 50.0
    assert score.iloc[499] == 100.0 and score.iloc[320] > 50.0
    assert (score.dropna() >= 50.0).all()                    # a top-risk warning never votes "bottom"
    x2 = x.copy()
    x2.iloc[-1] = 0.2                                        # small and active: mildly above neutral
    assert 50.0 < I._score_history(x2, spec).iloc[-1] < 70.0


def test_calm_funding_is_neutral_not_a_top_risk_vote():
    idx = pd.bdate_range(end=END, periods=900)
    rng = np.random.default_rng(2)
    x = pd.Series(10.0 + rng.normal(0, 3, len(idx)), index=idx)
    spec = INDICATORS_BY_KEY["fra_ois_spread"]
    assert spec.transform == "stress_only" and INDICATORS_BY_KEY["sofr_spread"].transform == "stress_only"
    calm, stressed = x.copy(), x.copy()
    calm.iloc[-5:] = -3.0
    stressed.iloc[-5:] = 60.0
    assert I._score_history(calm, spec).iloc[-1] == 50.0     # used to be ~95: "extreme complacency"
    assert I._score_history(stressed, spec).iloc[-1] < 5.0
    assert I._score_history(x, spec).dropna().max() <= 50.0


# --- coverage --------------------------------------------------------------
def test_gauge_matches_chart_endpoint_and_pillars():
    raw = _full_raw()
    per = I.per_indicator_history(raw)
    comp = I.composite(I.score_indicators(raw, per))
    pillars, hist, cov = I.aggregate_scores(per)
    assert comp["composite"] == pytest.approx(float(hist.iloc[-1]), abs=1e-9)
    assert comp["covered_weight"] == pytest.approx(float(cov.iloc[-1]))
    for b, info in comp["buckets"].items():
        assert info["score"] == pytest.approx(float(pillars[b].iloc[-1]), abs=1e-9)


def test_one_survivor_cannot_stand_in_for_a_pillar():
    credit = [k for k, sp in INDICATORS_BY_KEY.items() if sp.bucket == "credit_liquidity"]
    raw = _full_raw(skip=tuple(credit[1:]))                  # 1 of 14 credit indicators alive
    comp = I.composite(I.score_indicators(raw))
    assert np.isnan(comp["buckets"]["credit_liquidity"]["score"])
    assert comp["buckets"]["credit_liquidity"]["coverage"] == pytest.approx(1 / 14)
    assert comp["covered_weight"] == pytest.approx(0.6)      # the other three pillars, fully live
    assert not np.isnan(comp["composite"])
    assert I.exposure_recommendation(comp["composite"], None, comp["covered_weight"])["conviction"] == "Low"


def test_coverage_counts_indicators_not_pillars():
    sent = [k for k, sp in INDICATORS_BY_KEY.items() if sp.bucket == "sentiment_positioning" and not sp.display_only]
    drop = sent[:3]
    comp = I.composite(I.score_indicators(_full_raw(skip=tuple(drop))))
    total = sum(INDICATORS_BY_KEY[k].weight for k in sent)
    lost = sum(INDICATORS_BY_KEY[k].weight for k in drop)
    assert comp["covered_weight"] == pytest.approx(1 - BUCKET_WEIGHTS["sentiment_positioning"] * lost / total)


def test_no_composite_when_under_half_the_model_is_live():
    keep = [k for k, sp in INDICATORS_BY_KEY.items() if sp.bucket in ("valuation", "sentiment_positioning")]
    raw = _full_raw(skip=tuple(k for k in INDICATORS_BY_KEY if k not in keep))
    comp = I.composite(I.score_indicators(raw))
    assert comp["covered_weight"] == pytest.approx(0.3) and np.isnan(comp["composite"])
    assert I.gauge_state(comp["composite"])["status"] == "offline"
    assert I.historical_composite(raw).empty


def test_history_has_no_one_indicator_composite():
    """The composite history used to start in 1988 as a single survey's score."""
    rng = np.random.default_rng(9)
    long_idx = pd.bdate_range(end=END, periods=252 * 12)
    short_idx = long_idx[-252 * 4:]
    series = {k: pd.Series(np.cumsum(rng.normal(0, 1, len(short_idx))), index=short_idx) for k in INDICATORS_BY_KEY}
    series["aaii_bull_bear"] = pd.Series(np.cumsum(rng.normal(0, 1, len(long_idx))), index=long_idx)
    hist = I.historical_composite(_raw(series))
    assert hist.index[0] >= short_idx[251]                    # not before the rest of the model has a year of data


# --- cluster ---------------------------------------------------------------
def _scores_with(top: int, bottom: int) -> pd.DataFrame:
    themes: dict[str, list[str]] = {}
    for k, sp in INDICATORS_BY_KEY.items():
        if not sp.display_only:
            themes.setdefault(theme_for(k), []).append(k)
    names = list(themes)
    score = {}
    for i, t in enumerate(names):
        v = 95.0 if i < top else 5.0 if i < top + bottom else 50.0
        for k in themes[t]:
            score[k] = v
    return pd.DataFrame([{"key": k, "label": k, "bucket": INDICATORS_BY_KEY[k].bucket, "weight": INDICATORS_BY_KEY[k].weight,
                          "theme": theme_for(k), "score": v} for k, v in score.items()])


def test_cluster_needs_lopsided_extremes():
    """'3 themes at an extreme' was true on 99% of days: about five themes sit
    at each extreme on an ordinary day."""
    c = I.cluster_signal(_scores_with(top=5, bottom=5))
    assert c["themes_live"] == 22 and c["need"] == 8
    assert c["side"] is None and c["net_themes"] == 0
    assert I.cluster_signal(_scores_with(top=7, bottom=0))["side"] is None
    assert I.cluster_signal(_scores_with(top=9, bottom=1))["side"] == "top"
    assert I.cluster_signal(_scores_with(top=1, bottom=9))["side"] == "bottom"
    assert I.cluster_signal(_scores_with(top=10, bottom=3))["side"] is None      # opposing extremes cancel


def test_cluster_override_and_labels():
    base = I.exposure_recommendation(40.0, I.cluster_signal(_scores_with(5, 5)))
    assert base["cluster_override"] is False and base["net_pct"] == base["base_pct"] == 100
    bot = I.exposure_recommendation(40.0, I.cluster_signal(_scores_with(1, 9)))
    assert bot["label"].startswith("BOTTOM CLUSTER") and bot["net_pct"] == 115 and bot["base_pct"] == 100
    assert bot["cluster_side"] == "bottom" and bot["cluster_flag"] is False


def test_top_cluster_is_a_flag_and_changes_nothing():
    """It was followed by ordinary returns in the only history there is, so it
    is reported and does not cut exposure, add a hedge or raise conviction."""
    plain = I.exposure_recommendation(60.0, I.cluster_signal(_scores_with(5, 5)))
    top = I.exposure_recommendation(60.0, I.cluster_signal(_scores_with(9, 1)))
    assert top["cluster_side"] == "top" and top["cluster_flag"] is True and top["cluster_override"] is False
    for k in ("net_pct", "base_pct", "hedge_pct", "label", "color", "conviction"):
        assert top[k] == plain[k], k
    assert "top-cluster flag (9 themes at a top extreme vs 1)" in top["rationale"]
    assert plain["cluster_flag"] is False and plain["cluster_side"] is None


def test_top_cluster_cut_can_be_restored_from_one_constant(monkeypatch):
    monkeypatch.setattr(I, "TOP_CLUSTER_CUT", 20)
    monkeypatch.setattr(I, "TOP_CLUSTER_HEDGE", 0.5)
    top = I.exposure_recommendation(60.0, I.cluster_signal(_scores_with(9, 1)))
    assert top["label"].startswith("TOP CLUSTER") and top["cluster_override"] is True and top["cluster_flag"] is False
    assert top["net_pct"] == top["base_pct"] - 20 and top["hedge_pct"] >= 1.0 and top["conviction"] == "High"


def test_cluster_must_span_two_pillars():
    one_pillar = {"top_theme_count": 9, "bottom_theme_count": 0, "top_buckets": ["credit_liquidity"], "bottom_buckets": [], "need": 8}
    assert I.cluster_side(one_pillar) is None
    assert I.cluster_side({**one_pillar, "top_buckets": ["credit_liquidity", "valuation"]}) == "top"
    assert I.cluster_side(None) is None


def test_cluster_threshold_scales_with_live_themes():
    assert I.cluster_need(22) == 8 and I.cluster_need(12) == 4 and I.cluster_need(3) == 3


# --- exposure mapping ------------------------------------------------------
def test_exposure_moves_in_small_steps():
    """90% at 64.9 and 50% at 65.1: one point of score used to be 40 points of exposure."""
    xs = np.arange(0.0, 100.01, 0.5)
    net = np.array([I.base_exposure(x)[0] for x in xs])
    assert (np.diff(net) <= 0).all()                          # more top-risk never adds exposure
    assert np.abs(np.diff(net)).max() <= 5
    assert I.base_exposure(64.9)[0] - I.base_exposure(65.1)[0] <= 5


def test_exposure_keeps_each_band_level_at_its_centre():
    for score, level in ((7.5, 130), (25, 115), (40, 100), (55, 90), (75, 50), (92.5, 20)):
        assert I.base_exposure(score)[0] == level
    assert I.base_exposure(0)[0] == 130 and I.base_exposure(100)[0] == 20
    assert I.base_exposure(60)[1] == 0.0 and I.base_exposure(75)[1] == 0.5 and I.base_exposure(95)[1] == 1.0


@pytest.mark.parametrize("net,label", [(130, "BACK UP THE TRUCK"), (115, "SCALE IN"), (100, "FULLY INVESTED"),
                                         (90, "STANDARD ALLOCATION"), (70, "TRIM"), (30, "MAX DEFENSIVE")])
def test_label_follows_the_exposure(net, label):
    assert I.exposure_label(net)[0] == label


def test_no_data_recommendation():
    out = I.exposure_recommendation(float("nan"))
    assert out["net_pct"] is None and out["label"] == "NO DATA"


# --- explaining the number -------------------------------------------------
def test_contributions_add_up_to_the_composite():
    scores = I.score_indicators(_full_raw())
    comp = I.composite(scores)
    pulls = I.contributions(scores)
    assert pulls["contrib"].sum() == pytest.approx(comp["composite"] - 50.0, abs=1e-9)
    assert pulls["share"].sum() == pytest.approx(1.0)
    # Two valuation indicators share a 10% pillar: 5% each, more than one of 14 credit indicators.
    share = pulls.set_index("key")["share"]
    assert share["ndx_trailing_pe"] == pytest.approx(0.05) and share["hy_spread"] == pytest.approx(0.4 / 14)


def test_last_weeks_card_is_rebuilt_from_that_days_scores():
    raw = _full_raw()
    per = I.per_indicator_history(raw)
    scores = I.score_indicators(raw, per)
    comp = I.composite(scores)
    now = I.recommendation_at(per, 0)
    live = I.exposure_recommendation(comp["composite"], I.cluster_signal(scores), comp["covered_weight"])
    assert now["net_pct"] == live["net_pct"] and now["label"] == live["label"]
    week = I.recommendation_at(per, 5)
    assert week["as_of"] == per.index[-6]
    assert week["composite"] == pytest.approx(float(I.aggregate_scores(per)[1].iloc[-6]))
    assert I.recommendation_at(per.iloc[:3], 5) is None


def test_composite_context_ranks_today_in_its_own_range():
    idx = pd.bdate_range(end=END, periods=800)
    hist = pd.Series(np.r_[np.linspace(30, 70, 799), 32.0], index=idx)
    ctx = I.composite_context(hist)
    assert ctx["low"] == pytest.approx(32.0, abs=2.3) and ctx["high"] == pytest.approx(70.0, abs=0.1)
    assert ctx["pct"] < 5 and ctx["years"] == pytest.approx(3.0)
    assert I.composite_context(hist.iloc[:30]) == {}


# --- confidence ------------------------------------------------------------
NOW = datetime(2026, 11, 10, 12, 0, 0)


def _conf_rows(ages_limits):
    return pd.DataFrame([{"score": 50.0 + i, "as_of": NOW - pd.Timedelta(days=a), "source_kind": "primary",
                          "max_stale_days": lim, "status": "ok"} for i, (a, lim) in enumerate(ages_limits)])


def test_weekly_feeds_no_longer_rule_out_high_confidence():
    """One 10-day limit for every feed meant CFTC's normal lag capped confidence."""
    cf = I.read_confidence(_conf_rows([(1, 10), (2, 10), (9, 21), (12, 16)]), covered_weight=1.0, now=NOW)
    assert cf["level"] == "High" and cf["staleness"] == 12 and cf["freshest"] == 1 and cf["n_overdue"] == 0


def test_an_overdue_feed_lowers_confidence():
    cf = I.read_confidence(_conf_rows([(1, 10), (15, 10)]), covered_weight=1.0, now=NOW)
    assert cf["level"] == "Medium" and cf["n_overdue"] == 1


def test_a_pipeline_that_stopped_updating_is_low_confidence():
    cf = I.read_confidence(_conf_rows([(9, 21), (9, 21)]), covered_weight=1.0, now=NOW)
    assert cf["level"] == "Low" and cf["freshest"] == 9


def test_confidence_counts_indicators_dropped_as_stale():
    df = _conf_rows([(1, 10), (2, 10)])
    df.loc[len(df)] = {"score": np.nan, "as_of": NOW - pd.Timedelta(days=60), "source_kind": "cache", "max_stale_days": 10, "status": "stale"}
    assert I.read_confidence(df, 0.95, now=NOW)["n_stale"] == 1


# --- feeds -----------------------------------------------------------------
def test_funding_spread_is_cp_minus_bill_not_a_backward_average(monkeypatch):
    idx = pd.bdate_range("2026-01-01", periods=60)
    calls = []

    def fake_fred(sid, start="2015-01-01", prov_key=None):
        calls.append(sid)
        return {"DCPF3M": pd.Series(4.10, index=idx[::2]),      # CP isn't published every day
                "DTB3": pd.Series(4.00, index=idx)}.get(sid, pd.Series(dtype=float))

    monkeypatch.setattr(D, "_fred", fake_fred)
    D.reset_fetch_memo()
    s = D.fra_ois_spread()
    assert "SOFR90DAYAVG" not in calls
    assert s.iloc[-1] == pytest.approx(10.0) and len(s) >= 59   # short CP gaps are bridged
    assert D.get_provenance()["fra_ois_spread"]["source"] == "fred:DCPF3M-DTB3"
    D.reset_fetch_memo()


def test_funding_spread_unavailable_without_its_legs(monkeypatch):
    monkeypatch.setattr(D, "_fred", lambda *a, **k: pd.Series(dtype=float))
    D.reset_fetch_memo()
    assert D.fra_ois_spread().empty and D.get_provenance()["fra_ois_spread"]["kind"] == "unavailable"
    D.reset_fetch_memo()


def test_ticker_fallback_is_retried_next_build(monkeypatch):
    """One failed Wikipedia request at startup used to pin breadth to 80 megacaps until restart."""
    monkeypatch.setattr(D, "_SP500_TICKERS", [])
    state = {"fail": True}
    full = [f"T{i}" for i in range(500)]

    def fake_fetch():
        if state["fail"]:
            return ["AAPL", "MSFT"]
        D._SP500_TICKERS[:] = full
        return full

    monkeypatch.setattr(D, "_fetch_sp500_tickers", fake_fetch)
    D.reset_fetch_memo()
    assert D.sp500_tickers() == ["AAPL", "MSFT"]
    state["fail"] = False
    assert D.sp500_tickers() == ["AAPL", "MSFT"]              # same build: one answer
    D.reset_fetch_memo()
    assert len(D.sp500_tickers()) == 500                      # next build retries and recovers
    D.reset_fetch_memo()


def test_option_chain_is_refetched_each_build(monkeypatch):
    """A process-lifetime cache re-stamped the startup chain as 'today' for days."""
    n = {"calls": 0}

    def fake(symbol):
        n["calls"] += 1
        return pd.DataFrame(), float(n["calls"])

    monkeypatch.setattr(D, "_fetch_cboe_delayed_options", fake)
    D.reset_fetch_memo()
    assert D._cboe_delayed_options("SPY")[1] == 1.0 and D._cboe_delayed_options("SPY")[1] == 1.0
    D.reset_fetch_memo()
    assert D._cboe_delayed_options("SPY")[1] == 2.0
    D.reset_fetch_memo()


def test_ad_line_uses_the_whole_panel(monkeypatch):
    idx = pd.bdate_range(end=END, periods=600)
    rng = np.random.default_rng(1)
    panel = pd.DataFrame({f"T{i}": 100 * np.exp(np.cumsum(rng.normal(0, 0.01, len(idx)))) for i in range(20)}, index=idx)
    monkeypatch.setattr(D, "_panel_columns", lambda sample_size=None: (panel, list(panel.columns)))
    D.reset_fetch_memo()
    s = I._ad_line_proxy()
    assert len(s) == 599                                       # not the last 252 sessions only
    assert INDICATORS_BY_KEY["ad_line_slope"].transform == "zscore"
    D.reset_fetch_memo()


def test_aaii_history_survives_a_blocked_download(monkeypatch, tmp_path):
    """The full-history file is blocked on some builds, leaving a 21-week page
    that is too thin to score: the indicator used to drop in and out of the composite."""
    monkeypatch.setattr(D, "_series_cache_path", lambda fn: str(tmp_path / fn))
    monkeypatch.setattr(D, "SEED_DIR", str(tmp_path / "seeds"))
    cols = ["bullish", "neutral", "bearish", "spread"]
    full_idx = pd.date_range("2024-01-04", periods=120, freq="W-THU")
    full = pd.DataFrame({"bullish": 40.0, "neutral": 30.0, "bearish": 30.0, "spread": 10.0}, index=full_idx)[cols]
    recent = full.iloc[-21:].copy()
    recent.loc[full_idx[-1] + pd.Timedelta(days=7)] = [35.0, 30.0, 35.0, 0.0]     # one new week

    D.reset_fetch_memo()
    monkeypatch.setattr(D, "_aaii_live", lambda: (full, "aaii.com:sentiment.xls"))
    assert len(D.aaii_sentiment()) == 120
    monkeypatch.setattr(D, "_aaii_live", lambda: (recent, "aaii.com:results page (recent weeks only)"))
    out = D.aaii_sentiment()
    assert len(out) == 121 and out["spread"].iloc[-1] == 0.0                       # history kept, new week added
    assert "local cache" in D.get_provenance()["aaii_bull_bear"]["note"]
    monkeypatch.setattr(D, "_aaii_live", lambda: (pd.DataFrame(columns=cols), ""))
    assert len(D.aaii_sentiment()) == 121 and D.get_provenance()["aaii_bull_bear"]["kind"] == "cache"
    D.reset_fetch_memo()


def test_aaii_unavailable_with_no_source_and_no_history(monkeypatch, tmp_path):
    monkeypatch.setattr(D, "_series_cache_path", lambda fn: str(tmp_path / fn))
    monkeypatch.setattr(D, "SEED_DIR", str(tmp_path / "seeds"))
    monkeypatch.setattr(D, "_aaii_live", lambda: (pd.DataFrame(columns=["bullish", "neutral", "bearish", "spread"]), ""))
    D.reset_fetch_memo()
    assert D.aaii_sentiment().empty and D.get_provenance()["aaii_bull_bear"]["kind"] == "unavailable"
    D.reset_fetch_memo()


def test_aaii_seed_covers_a_fresh_deploy(monkeypatch, tmp_path):
    """No local cache and the history file blocked: the committed seed plus the
    recent-weeks page must still give a scoreable series."""
    monkeypatch.setattr(D, "_series_cache_path", lambda fn: str(tmp_path / fn))
    seed = D._load_seed_frame(D.AAII_SEED)
    assert len(seed) > 1500 and list(seed.columns) == ["spread"]
    assert seed.index.is_monotonic_increasing and not seed.index.duplicated().any()
    assert seed["spread"].between(-60, 70).all()

    cols = ["bullish", "neutral", "bearish", "spread"]
    D.reset_fetch_memo()
    monkeypatch.setattr(D, "_aaii_live", lambda: (pd.DataFrame(columns=cols), ""))
    out = D.aaii_sentiment()
    assert len(out) == len(seed) and D.get_provenance()["aaii_bull_bear"]["kind"] == "fallback"

    new_week = seed.index[-1] + pd.Timedelta(days=7)
    recent = pd.DataFrame({"bullish": [30.0], "neutral": [30.0], "bearish": [40.0], "spread": [-10.0]}, index=[new_week])
    monkeypatch.setattr(D, "_aaii_live", lambda: (recent, "aaii.com:results page (recent weeks only)"))
    out = D.aaii_sentiment()
    assert len(out) == len(seed) + 1 and out["spread"].iloc[-1] == -10.0
    assert D.get_provenance()["aaii_bull_bear"]["kind"] == "primary"
    D.reset_fetch_memo()

