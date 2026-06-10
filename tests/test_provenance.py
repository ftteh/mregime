"""Provenance registry + plumbing: which source ACTUALLY served each indicator
must survive from the fetcher layer through build_raw meta into the scores frame.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import data as D
from src import indicators as I


def test_registry_lifecycle():
    D.reset_fetch_memo()
    assert D.get_provenance() == {}
    D.record_provenance("vix", "cboe_csv:VIX", kind="fallback", note="rate-limited")
    prov = D.get_provenance()
    assert prov["vix"] == {"source": "cboe_csv:VIX", "kind": "fallback", "note": "rate-limited"}
    # Snapshot is a copy — mutating it must not touch the registry.
    prov["vix"]["kind"] = "primary"
    assert D.get_provenance()["vix"]["kind"] == "fallback"
    D.reset_fetch_memo()
    assert D.get_provenance() == {}


def test_display_overlay_lifecycle():
    D.reset_fetch_memo()
    assert D.get_display_overlays() == {}
    s = pd.Series([1.0, 2.0], index=pd.bdate_range("2026-01-05", periods=2))
    D._set_display_overlay("gamma_flip_zone", s)
    assert "gamma_flip_zone" in D.get_display_overlays()
    # Empty series are ignored rather than registered.
    D._set_display_overlay("other", pd.Series(dtype=float))
    assert "other" not in D.get_display_overlays()
    D.reset_fetch_memo()
    assert D.get_display_overlays() == {}


def test_rawframe_two_arg_backcompat():
    rf = I.RawFrame(series={}, meta={})
    assert rf.display == {}


def test_score_indicators_passes_source_through():
    idx = pd.bdate_range("2021-01-04", periods=1300)
    rng = np.random.default_rng(7)
    series = {"vix": pd.Series(np.abs(np.cumsum(rng.normal(0, 1, len(idx)))) + 12,
                               index=idx, name="vix")}
    meta = {
        "vix": {
            "last": float(series["vix"].iloc[-1]),
            "as_of": idx[-1].to_pydatetime(),
            "n": len(idx),
            "provenance": {"source": "cboe_csv:VIX", "kind": "fallback", "note": ""},
        }
    }
    raw = I.RawFrame(series=series, meta=meta)
    scores = I.score_indicators(raw)
    row = scores.set_index("key").loc["vix"]
    assert row["source_used"] == "cboe_csv:VIX"
    assert row["source_kind"] == "fallback"
    # Keys without provenance stay missing — None or NaN depending on how
    # pandas types the column (UI treats both as "show static config source").
    other = scores.set_index("key").loc["hy_spread"]
    assert other["source_used"] is None or pd.isna(other["source_used"])
