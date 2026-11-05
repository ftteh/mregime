"""Gauge == chart endpoint must hold on REAL-shaped data, where feeds end on
different days (FRED a day behind yfinance, weekly surveys up to a week behind).
The synthetic snapshot fixture has every series ending on the same date, so it
cannot catch a ragged-edge divergence on its own."""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import INDICATORS_BY_KEY
from src import indicators as I


def _ragged_raw() -> I.RawFrame:
    rng = np.random.default_rng(11)
    idx = pd.bdate_range("2020-01-01", "2026-11-05")
    series = {}
    for n, key in enumerate(INDICATORS_BY_KEY):
        lag = n % 6  # 0..5 business days of publication lag per feed
        s = pd.Series(np.cumsum(rng.normal(0, 1, len(idx))), index=idx)
        series[key] = s.iloc[: len(idx) - lag] if lag else s
    meta = {k: {"last": float(v.iloc[-1]), "as_of": v.index[-1].to_pydatetime(), "n": len(v)}
            for k, v in series.items()}
    return I.RawFrame(series=series, meta=meta)


def test_gauge_matches_chart_endpoint_with_ragged_feeds():
    raw = _ragged_raw()
    per = I.per_indicator_history(raw)
    comp = I.composite(I.score_indicators(raw, per))
    pillars = I.historical_pillar_scores(raw, per)
    hist = I.historical_composite(raw, pillars)
    assert abs(comp["composite"] - float(hist.iloc[-1])) < 1e-9
    for bucket, info in comp["buckets"].items():
        if not np.isnan(info["score"]):
            assert abs(info["score"] - float(pillars[bucket].iloc[-1])) < 1e-9, bucket


def test_edge_carry_does_not_fill_interior_gaps():
    df = pd.DataFrame({"a": [1.0, np.nan, 3.0, np.nan], "b": [1.0, 2.0, 3.0, 4.0]},
                      index=pd.bdate_range("2026-01-01", periods=4))
    out = I._carry_scores_to_edge(df)
    assert np.isnan(out["a"].iloc[1])        # interior gap untouched
    assert out["a"].iloc[3] == 3.0           # trailing edge carried
