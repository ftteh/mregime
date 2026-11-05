"""Pure analytics behind the participation / rotation / valuation /
standardized-signal panels (src/analytics.py). Deterministic, no network."""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import analytics as A


def _panel(n_days: int = 120) -> pd.DataFrame:
    """3 rising + 1 falling + 1 flat stock, business-day index."""
    idx = pd.bdate_range("2025-01-01", periods=n_days)
    t = np.arange(n_days, dtype=float)
    return pd.DataFrame({
        "UP1": 100 + t,
        "UP2": 50 + 0.5 * t,
        "UP3": 10 + 0.1 * t,
        "DN1": 300 - t,
        "FLAT": np.full(n_days, 20.0),
    }, index=idx)


# --- trend breadth -----------------------------------------------------------
def test_trend_breadth_counts_fast_above_slow():
    out = A.trend_breadth(_panel(), fast=10, slow=20)
    assert not out.empty
    # Rising names: 10DMA > 20DMA (3); falling and flat: not above (2) -> 60%.
    assert out["pct"].iloc[-1] == pytest.approx(60.0)
    assert out["eligible"].iloc[-1] == 5
    # Warm-up rows (slow MA undefined) are excluded, not reported as 0%.
    assert out.index[0] == _panel().index[19]


def test_trend_breadth_drops_dates_with_collapsed_coverage():
    p = _panel()
    p.iloc[-1, 1:] = np.nan  # last row: only 1 of 5 names printed
    p.iloc[-1, 0] = 500.0
    # ffill(limit=5) bridges a one-day gap, so blank a long stretch instead.
    p.iloc[-30:, 1:] = np.nan
    out = A.trend_breadth(p, 10, 20)
    assert out["eligible"].min() >= 3  # thin dates masked out


def test_trend_breadth_empty_panel():
    assert A.trend_breadth(pd.DataFrame()).empty


@pytest.mark.parametrize("pct,label", [
    (85, "EXTENDED"), (70.8, "BROAD"), (50, "MIXED"), (30, "NARROW"), (10, "WASHED OUT"),
])
def test_trend_breadth_regime_labels(pct, label):
    assert A.trend_breadth_regime(pct)[0] == label


def test_trend_breadth_snapshot_fields():
    b = A.trend_breadth(_panel(260), 10, 20)
    spx = pd.Series(np.linspace(100, 130, 260), index=_panel(260).index)
    snap = A.trend_breadth_snapshot(b, spx, lookback_years=3)
    assert snap["current"] == pytest.approx(60.0)
    assert snap["vs_neutral"] == pytest.approx(10.0)
    assert snap["eligible"] == 5
    assert snap["index_vs_200dma"] > 0  # steadily rising index sits above its 200DMA
    assert 0 <= snap["pct_rank"] <= 100


# --- overbought / oversold -------------------------------------------------
def test_obos_classifies_against_own_50d_band():
    idx = pd.bdate_range("2025-01-01", periods=80)
    rng = np.random.default_rng(7)
    base = 100 + rng.normal(0, 1, 80).cumsum() * 0.1
    p = pd.DataFrame({"A": base.copy(), "B": base.copy(), "C": base.copy()}, index=idx)
    p.iloc[-1, 0] = base[-1] + 50   # spike far above band -> overbought
    p.iloc[-1, 1] = base[-1] - 50   # crash far below band -> oversold
    out = A.overbought_oversold_breadth(p, window=50, k=1.0)
    last = out.iloc[-1]
    assert last["ob_pct"] == pytest.approx(100 / 3)
    assert last["os_pct"] == pytest.approx(100 / 3)
    assert last["spread"] == pytest.approx(0.0)
    assert last["eligible"] == 3


def test_obos_zones_and_snapshot_state():
    spread = pd.Series(np.linspace(-50, 50, 500), index=pd.bdate_range("2022-01-03", periods=500))
    lo, hi = A.obos_zones(spread)
    assert lo < 0 < hi
    df = pd.DataFrame({"spread": spread, "ob_pct": 60.0, "os_pct": 10.0})
    snap = A.obos_snapshot(df)
    assert snap["state"] == "OVERBOUGHT ZONE"  # last value is the series max
    assert snap["pct_rank"] == pytest.approx(100.0)


# --- relative strength -----------------------------------------------------
def _rs_inputs(trend: float) -> tuple[pd.Series, pd.Series]:
    idx = pd.bdate_range("2024-01-01", periods=400)
    den = pd.Series(np.full(400, 100.0), index=idx)
    num = pd.Series(100 * np.exp(trend * np.arange(400)), index=idx)
    return num, den


def test_relative_strength_status_strength_and_weakness():
    up = A.relative_strength(*_rs_inputs(+0.001))
    dn = A.relative_strength(*_rs_inputs(-0.001))
    assert A.relative_strength_status(up)["label"] == "RELATIVE STRENGTH"
    assert A.relative_strength_status(dn)["label"] == "RELATIVE WEAKNESS"


def test_relative_strength_needs_200_sessions():
    idx = pd.bdate_range("2025-01-01", periods=120)
    s = pd.Series(np.linspace(1, 2, 120), index=idx)
    rs = A.relative_strength(s, s * 0 + 1)
    assert A.relative_strength_status(rs)["label"] == "INSUFFICIENT HISTORY"


def test_rebase_to_starts_at_100_and_keeps_ma_scale():
    rs = A.relative_strength(*_rs_inputs(+0.001))
    start = rs.index[300]
    rb = A.rebase_to(rs, start)
    assert rb["ratio"].iloc[0] == pytest.approx(100.0)
    # MAs rebased by the same factor -> ratio/MA relationship is preserved.
    raw_rel = rs.loc[start:, "ratio"] / rs.loc[start:, "ma_fast"]
    rb_rel = rb["ratio"] / rb["ma_fast"]
    assert np.allclose(raw_rel.values, rb_rel.values)


# --- valuation stats -------------------------------------------------------
def test_valuation_stats_flags_multi_year_low():
    idx = pd.bdate_range("2021-11-01", "2026-11-04")
    vals = 25 + 3 * np.sin(np.arange(len(idx)) / 120.0)
    vals[-1] = 18.0  # new low for the window
    st = A.valuation_stats(pd.Series(vals, index=idx), years=5)
    assert st["pct_rank"] < 1
    assert st["headline"].startswith("VALUATION AT ") and st["headline"].endswith("-YEAR LOWS")
    assert st["vs_avg_pct"] < 0
    assert st["span_years"] == pytest.approx(5, abs=0.05)


def test_valuation_stats_reports_short_history_honestly():
    idx = pd.bdate_range("2026-03-27", "2026-11-04")
    s = pd.Series(np.linspace(25, 21, len(idx)), index=idx)
    st = A.valuation_stats(s, years=5)
    assert st["span_years"] < 1             # 7 months, not "5 years"
    assert "YEAR" not in st["headline"]     # no multi-year claim off 7 months


# --- standardized signals ----------------------------------------------------
def test_rolling_zscore_point_in_time():
    idx = pd.bdate_range("2024-01-01", periods=300)
    s = pd.Series(np.r_[np.zeros(299) + np.tile([1.0, -1.0], 150)[:299], -10.0], index=idx)
    z = A.rolling_zscore(s, window=252)
    assert z.iloc[-1] < -5   # the final shock is extreme vs its own trailing window


def test_standardize_signal_negative_extreme():
    idx = pd.bdate_range("2024-01-01", periods=400)
    rng = np.random.default_rng(1)
    v = rng.normal(0, 1, 400)
    v[-1] = -6.0
    r = A.standardize_signal(pd.Series(v, index=idx))
    assert r["label"] == "AT A NEGATIVE EXTREME"
    assert r["pct_1y"] < 1
    assert len(r["z"]) <= 252
    assert "bottom" in r["takeaway"]


def test_standardize_signal_weekly_series_uses_calendar_year():
    idx = pd.date_range("2022-01-05", periods=200, freq="W-WED")
    s = pd.Series(np.arange(200, dtype=float), index=idx)
    r = A.standardize_signal(s)
    # Business-day resampling: the window is ~1 calendar year, not 252 weeks.
    span_days = (r["z"].index[-1] - r["z"].index[0]).days
    assert 330 <= span_days <= 380


def test_standardize_signal_requires_history():
    s = pd.Series([1.0, 2.0, 3.0], index=pd.bdate_range("2026-01-01", periods=3))
    assert A.standardize_signal(s) == {}


@pytest.mark.parametrize("z,direction,expected", [
    (3.0, "contrarian_high_is_top", "bottom-setup extreme"),  # VIX spike
    (3.0, "risk_high_is_top", "top-risk extreme"),
    (-2.5, "risk_high_is_top", "bottom-setup extreme"),
    (0.3, "risk_high_is_top", "neutral"),
    (2.5, None, "—"),
])
def test_oriented_extreme(z, direction, expected):
    assert A.oriented_extreme(z, direction) == expected


def test_valuation_stats_time_weights_mixed_frequency():
    """Monthly history + a dense run of recent daily points (the trailing-P/E
    cache accumulates one live point per build) must not let the recent days
    outvote years of monthly readings."""
    monthly = pd.Series(30.0, index=pd.date_range("2021-11-30", "2025-10-31", freq="ME"))
    daily = pd.Series(20.0, index=pd.bdate_range("2025-11-03", "2026-11-04"))
    st = A.valuation_stats(pd.concat([monthly, daily]), years=5)
    # ~1 of 5 years at 20x -> ~20th percentile by time, not ~80% by row count.
    assert st["pct_rank"] < 30
    assert st["avg"] > 27
