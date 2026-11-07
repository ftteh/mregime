"""Backtest machinery: parity with the live rules, and no look-ahead."""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import backtest as B
from src import research as R


def _walk(n: int = 900, seed: int = 1, drift: float = 0.0004, vol: float = 0.012, start: str = "2015-01-01") -> pd.Series:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, periods=n)
    return pd.Series(100 * np.exp(np.cumsum(rng.normal(drift, vol, n))), index=idx)


def _prices(n: int = 1800) -> pd.DataFrame:
    tickers = ["SPY", "QQQ", "^SOX", "^VIX", "^VXN", "^SKEW", "MTUM", "GVIP", "^MOVE", "^TNX"]
    return pd.DataFrame({t: _walk(n, seed=i + 1) for i, t in enumerate(tickers)})


# --- parity: the history's last row must equal the live rule ---------------
@pytest.mark.parametrize("seed_shift", [0, 3, 7, 11, 19])
def test_confirmation_history_matches_live_rules(seed_shift):
    p = pd.DataFrame({t: _walk(600, seed=i + 1 + seed_shift) for i, t in enumerate(["QQQ", "^VXN", "^SKEW", "^MOVE", "^TNX"])})
    ai = _walk(600, 50 + seed_shift) / 200
    live = {r.key: r.fired for r in R.reversal_confirmation(p, ai)}
    hist = B.confirmation_history(p, ai).iloc[-1]
    assert hist.to_dict() == live


@pytest.mark.parametrize("cot,gamma,v", [(-30.0, -1.0, 35.0), (5.0, 2.0, 15.0), (-30.0, 2.0, 15.0)])
def test_gate_history_matches_live_gate(cot, gamma, v):
    idx = pd.bdate_range("2019-01-01", periods=1500)
    weekly = pd.date_range("2015-01-06", "2024-06-25", freq="W-TUE")
    net = pd.Series(np.sin(np.arange(len(weekly)) / 9.0), index=weekly)
    net.iloc[-1] = cot
    corr = pd.Series(0.3 + 0.1 * np.sin(np.arange(1500) / 40.0), index=idx)
    g = pd.Series(gamma, index=idx)
    vix = pd.Series(v, index=idx)
    snap = R.positioning_snapshot(net)
    live = R.leverage_gate(vix_cot=snap, spx_corr=corr, gamma=g, vix=vix)
    hist = B.gate_history(index=idx[idx >= weekly[-1] + pd.Timedelta(days=3)], vix_cot=net, spx_corr=corr, gamma=g, vix=vix)
    assert int(hist["hits"].iloc[-1]) == live["hits"]


def test_permission_needs_gate_and_confirmation():
    idx = pd.bdate_range("2024-01-01", periods=4)
    hits = pd.Series([2, 2, 1, 1], index=idx)
    conf = pd.Series([3, 2, 3, 2], index=idx)
    assert B.permission_history(hits, conf).tolist() == [False, False, True, False]


# --- no look-ahead ---------------------------------------------------------
def test_confirmation_history_has_no_lookahead():
    p = _prices(700)
    full = B.confirmation_history(p)
    cut = B.confirmation_history(p.iloc[:500])
    pd.testing.assert_frame_equal(full.iloc[:500], cut)


def test_gate_history_has_no_lookahead():
    idx = pd.bdate_range("2018-01-01", periods=900)
    weekly = pd.date_range("2013-01-01", periods=600, freq="W-TUE")
    cot = pd.Series(np.sin(np.arange(600) / 7.0), index=weekly)
    corr = pd.Series(np.cos(np.arange(900) / 30.0), index=idx)
    full = B.gate_history(index=idx, vix_cot=cot, spx_corr=corr, gamma=None, vix=None)
    cut_t = idx[500]
    cut = B.gate_history(index=idx[:501], vix_cot=cot[cot.index <= cut_t], spx_corr=corr.loc[:cut_t], gamma=None, vix=None)
    pd.testing.assert_frame_equal(full.iloc[:501], cut, check_freq=False)


def test_cot_waits_for_friday_publication():
    idx = pd.date_range("2012-01-03", periods=200, freq="W-TUE")
    s = B.published_cot(pd.Series(1.0, index=idx))
    assert (s.index.dayofweek == 4).all()


def test_strategy_earns_the_day_after_the_signal():
    idx = pd.bdate_range("2024-01-01", periods=5)
    asset = pd.Series([100.0, 110.0, 110.0, 121.0, 121.0], index=idx)
    # Signal on day 1 (the +10% day itself) must not capture that day's return.
    w = pd.Series([0.0, 1.0, 0.0, 0.0, 0.0], index=idx)
    r = B.strategy_returns(w, asset, pd.Series(0.0, index=idx))
    assert r.loc[idx[1]] == 0.0 and r.loc[idx[2]] == pytest.approx(0.0)
    assert r.loc[idx[3]] == 0.0


def test_leverage_pays_financing_and_cash_earns_bills():
    idx = pd.bdate_range("2024-01-01", periods=3)
    asset = pd.Series([100.0, 100.0, 100.0], index=idx)
    rate = pd.Series(5.0, index=idx)
    lev = B.strategy_returns(pd.Series(1.3, index=idx), asset, rate)
    assert lev.iloc[-1] == pytest.approx(-0.3 * (0.05 + B.FINANCING_SPREAD) / 252)
    half = B.strategy_returns(pd.Series(0.5, index=idx), asset, rate)
    assert half.iloc[-1] == pytest.approx(0.5 * 0.05 / 252)


# --- statistics ------------------------------------------------------------
def test_performance_on_known_path():
    idx = pd.bdate_range("2024-01-01", periods=252)
    r = pd.Series(0.0, index=idx)
    r.iloc[10] = 0.10
    r.iloc[20] = -0.20
    out = B.performance(r)
    assert out["max_dd"] == pytest.approx(-20.0)
    assert out["cagr"] == pytest.approx((1.1 * 0.8 - 1) * 100, abs=0.01)


def test_bootstrap_diff_centres_on_the_true_gap():
    a = _walk(1500, 1).pct_change().dropna()
    out = B.bootstrap_mean_diff(a + 0.001, a)
    assert out["diff"] == pytest.approx(25.2)
    assert out["lo"] == pytest.approx(25.2) and out["hi"] == pytest.approx(25.2)


def test_bootstrap_interval_spans_zero_for_noise():
    a, b = _walk(1500, 1).pct_change().dropna(), _walk(1500, 2).pct_change().dropna()
    out = B.bootstrap_mean_diff(a, b)
    assert out["lo"] < out["diff"] < out["hi"]
    assert out["hi"] - out["lo"] > 1.0  # real noise gives a real interval


def test_episode_ids_merge_close_runs():
    idx = pd.bdate_range("2024-01-01", periods=30)
    f = pd.Series(False, index=idx)
    f.iloc[[1, 2, 5, 25]] = True
    ids = B.episode_ids(f, gap=5)
    assert ids.dropna().tolist() == [1, 1, 1, 2]


def test_group_difference_resamples_episodes():
    idx = pd.bdate_range("2024-01-01", periods=40)
    f = pd.Series(np.r_[np.full(20, 1.0), np.full(20, -1.0)], index=idx)
    g = pd.Series(np.r_[np.full(20, True), np.full(20, False)], index=idx)
    e = pd.Series(np.arange(40) // 5, index=idx).astype(float)
    out = B.group_difference(f, g, e, n=300)
    assert out["diff"] == pytest.approx(2.0) and out["episodes"] == 8


def test_rule_flag_parity_with_capitulation_history():
    p = _prices(1800)
    hist = R.capitulation_history(p)
    held = B.rule_flag(p, "qqq_rsi", 32).reindex(hist.index).fillna(False).astype(float).rolling(5, min_periods=1).max().astype(bool)
    pd.testing.assert_series_equal(held, hist["qqq_rsi"], check_names=False)
    with pytest.raises(KeyError):
        B.rule_flag(p, "nope", 1)


def test_periods_do_not_overlap():
    assert pd.Timestamp(B.OUT_OF_SAMPLE.end) < pd.Timestamp(B.IN_SAMPLE.start)
