"""gauge_state + read_confidence: a NaN composite must never render as 0.0
('capitulation'), and unknown freshness must never pass as fresh.
"""
from __future__ import annotations

import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import indicators as I


def test_gauge_state_offline_on_nan():
    assert I.gauge_state(np.nan) == {"status": "offline", "value": None}
    assert I.gauge_state(None) == {"status": "offline", "value": None}


def test_gauge_state_ok():
    assert I.gauge_state(42.0) == {"status": "ok", "value": 42.0}
    assert I.gauge_state(0.0) == {"status": "ok", "value": 0.0}  # a real 0 is valid


def _scores(rows):
    return pd.DataFrame(rows)


NOW = datetime(2026, 6, 10, 12, 0, 0)


def test_confidence_high_when_fresh_aligned_covered():
    df = _scores([
        {"score": 50.0, "as_of": datetime(2026, 6, 9), "source_kind": "primary"},
        {"score": 55.0, "as_of": datetime(2026, 6, 8), "source_kind": "primary"},
        {"score": 52.0, "as_of": datetime(2026, 6, 9), "source_kind": "primary"},
    ])
    cf = I.read_confidence(df, covered_weight=1.0, now=NOW)
    assert cf["level"] == "High"
    assert cf["staleness"] == 2  # the STALEST feed (max gap), not the freshest


def test_confidence_low_when_no_asof_at_all():
    """Regression: `staleness is None` used to pass the High-tier freshness
    check — an unverifiable reading was rated trustworthy."""
    df = _scores([
        {"score": 50.0, "as_of": None, "source_kind": "primary"},
        {"score": 55.0, "as_of": None, "source_kind": "primary"},
    ])
    cf = I.read_confidence(df, covered_weight=1.0, now=NOW)
    assert cf["staleness"] is None
    assert cf["level"] == "Low"
    assert cf["n_missing_asof"] == 2


def test_confidence_counts_proxy_and_fallback_feeds():
    df = _scores([
        {"score": 50.0, "as_of": datetime(2026, 6, 9), "source_kind": "proxy"},
        {"score": 55.0, "as_of": datetime(2026, 6, 9), "source_kind": "fallback"},
        {"score": 52.0, "as_of": datetime(2026, 6, 9), "source_kind": "primary"},
        {"score": np.nan, "as_of": None, "source_kind": "proxy"},  # not scored
    ])
    cf = I.read_confidence(df, covered_weight=1.0, now=NOW)
    assert cf["n_proxy"] == 2  # only contributing (scored) rows counted


def test_confidence_low_coverage_caps_level():
    df = _scores([
        {"score": 50.0, "as_of": datetime(2026, 6, 9), "source_kind": "primary"},
    ])
    cf = I.read_confidence(df, covered_weight=0.5, now=NOW)
    assert cf["level"] == "Low"
