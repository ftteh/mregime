"""Smoke tests: every new panel figure builds from synthetic data, and degrades
to an empty-but-valid figure instead of raising when inputs are missing."""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import analytics as A
from src import charts as C

IDX = pd.bdate_range("2023-01-02", periods=900)
RNG = np.random.default_rng(3)


def _walk(base=100.0, scale=1.0):
    return pd.Series(base + RNG.normal(0, scale, len(IDX)).cumsum(), index=IDX)


def test_obos_figure_builds():
    df = pd.DataFrame({"spread": _walk(0, 3).clip(-60, 60), "ob_pct": _walk(30, 1).clip(0, 100),
                       "os_pct": _walk(20, 1).clip(0, 100)})
    lo, hi = A.obos_zones(df["spread"])
    fig = C.obos_figure(df, IDX[-252], lo, hi, marker_ts=IDX[-50])
    names = {t.name for t in fig.data}
    assert {"OB − OS spread", "Overbought %", "Oversold %"} <= names


def test_trend_breadth_figure_builds():
    fig = C.trend_breadth_figure(_walk(50, 2).clip(0, 100), _walk(5000, 20), IDX[100])
    assert {"S&P 500", "200DMA", "10/20DMA breadth"} <= {t.name for t in fig.data}


def test_relative_strength_figure_builds():
    rs = A.relative_strength(_walk(200, 1), _walk(500, 1))
    fig = C.relative_strength_figure(A.rebase_to(rs, IDX[-252]), "IWM / SPX")
    assert len(fig.data) >= 3


def test_valuation_figure_builds_with_and_without_forward():
    trl = _walk(30, 0.2).resample("ME").last()
    fwd = _walk(22, 0.1).iloc[-150:]
    assert len(C.valuation_figure(trl, fwd, IDX[0], 30.0, 22.0).data) >= 3
    assert len(C.valuation_figure(trl, pd.Series(dtype=float), IDX[0], 30.0).data) >= 1


def test_zscore_figure_builds():
    r = A.standardize_signal(_walk(0, 1))
    fig = C.zscore_figure(r["z"], "test signal")
    assert any(t.name == "test signal" for t in fig.data)


def test_figures_tolerate_empty_inputs():
    empty = pd.Series(dtype=float)
    C.obos_figure(pd.DataFrame(), None, -30, 30)
    C.trend_breadth_figure(empty, empty, None)
    C.relative_strength_figure(pd.DataFrame(), "x")
    C.valuation_figure(empty, empty, None, None)
    C.zscore_figure(empty, "x")


def test_zone_shapes_survive_subplots():
    """Plotly drops row/col shapes on still-empty subplots by default
    (exclude_empty_subplots=True) — the OB/OS zones and the 50% neutral line
    silently vanished until the figure builders opted out."""
    df = pd.DataFrame({"spread": _walk(0, 3).clip(-60, 60), "ob_pct": 40.0, "os_pct": 10.0})
    fig = C.obos_figure(df, IDX[-252], -30, 35)
    assert len(fig.layout.shapes) >= 3   # OB zone, OS zone, zero line
    fig2 = C.trend_breadth_figure(_walk(50, 2).clip(0, 100), _walk(5000, 20), IDX[100])
    assert len(fig2.layout.shapes) >= 3  # two extreme bands + neutral line
