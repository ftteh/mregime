"""FINRA margin debt: YCharts magnitude parsing, and the
live -> disk-cache -> bundled-seed fallback chain (so the chart never renders
empty, even on a fresh cloud deploy with YCharts blocked).
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import data as D


def _patch_cache_dir(monkeypatch, tmp_path):
    monkeypatch.setattr(D, "_series_cache_path", lambda fn: str(tmp_path / fn))


def _fake_live(n=6):
    idx = pd.to_datetime(
        ["2026-01-31", "2026-02-28", "2026-03-31",
         "2026-04-30", "2026-05-31", "2026-06-30"][:n]
    )
    return pd.Series(np.linspace(1279.0, 1450.0, n), index=idx, name="margin_debt")


def test_parse_ycharts_magnitude_units():
    p = D._parse_ycharts_magnitude
    assert p("1.416T") == 1416.0          # trillions -> billions
    assert p("920.96B") == 920.96          # billions stay
    assert abs(p("1,234M") - 1.234) < 1e-9  # millions -> billions, commas ok
    assert p("998.5") == 998.5             # bare number assumed billions
    assert p("") is None                    # no number
    assert p("N/A") is None


def test_seed_is_clean_monthly_series():
    s = D._margin_debt_seed_series()
    assert not s.empty
    assert s.index.is_monotonic_increasing
    assert s.index.duplicated().sum() == 0
    # Plausible FINRA range ($ billions), never the raw-millions figure.
    assert s.min() > 100 and s.max() < 5000


def test_successful_fetch_persists_and_is_primary(monkeypatch, tmp_path):
    _patch_cache_dir(monkeypatch, tmp_path)
    monkeypatch.setattr(D, "_ycharts_margin_debt", _fake_live)

    s = D.finra_margin_debt()
    assert not s.empty
    assert float(s.iloc[-1]) == 1450.0            # live tail present
    assert D.get_provenance()["margin_debt"]["kind"] == "primary"
    on_disk = pd.read_csv(tmp_path / "margin_debt_history.csv")
    assert "margin_debt" in on_disk.columns


def test_fetch_failure_falls_back_to_cache(monkeypatch, tmp_path):
    _patch_cache_dir(monkeypatch, tmp_path)
    # Seed the cache from one good fetch, then kill the live source.
    monkeypatch.setattr(D, "_ycharts_margin_debt", _fake_live)
    D.finra_margin_debt()
    monkeypatch.setattr(D, "_ycharts_margin_debt",
                        lambda: pd.Series(dtype=float, name="margin_debt"))

    s = D.finra_margin_debt()
    assert float(s.iloc[-1]) == 1450.0            # served from cache
    assert D.get_provenance()["margin_debt"]["kind"] == "cache"


def test_no_live_no_cache_falls_back_to_seed(monkeypatch, tmp_path):
    _patch_cache_dir(monkeypatch, tmp_path)  # empty temp dir => no cache file
    monkeypatch.setattr(D, "_ycharts_margin_debt",
                        lambda: pd.Series(dtype=float, name="margin_debt"))

    s = D.finra_margin_debt()
    assert not s.empty                            # seed guarantees a chart
    assert D.get_provenance()["margin_debt"]["kind"] == "fallback"
    seed = D._margin_debt_seed_series()
    assert float(s.iloc[-1]) == float(seed.iloc[-1])
