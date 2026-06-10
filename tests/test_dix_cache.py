"""DIX last-good cache: a transient SqueezeMetrics failure serves cached
history (provenance kind='cache'), not an empty series; a successful fetch
persists the series for the next outage.
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


def _fake_squeeze_df(n=5):
    idx = pd.bdate_range("2026-06-01", periods=n, name="date")
    return pd.DataFrame(
        {"price": np.linspace(7000, 7100, n),
         "dix": np.linspace(0.40, 0.44, n),
         "gex": np.linspace(1e9, 2e9, n)},
        index=idx,
    )


def test_successful_fetch_persists_last_good(monkeypatch, tmp_path):
    _patch_cache_dir(monkeypatch, tmp_path)
    monkeypatch.setattr(D, "_squeezemetrics_csv", lambda: _fake_squeeze_df())

    s = D.dix_proxy()
    assert len(s) == 5
    assert D.get_provenance()["dix"]["kind"] == "primary"
    on_disk = pd.read_csv(tmp_path / "dix_history.csv")
    assert "dix" in on_disk.columns and len(on_disk) == 5


def test_fetch_failure_falls_back_to_cache(monkeypatch, tmp_path):
    _patch_cache_dir(monkeypatch, tmp_path)
    # Seed the cache from one good fetch, then kill the live source.
    monkeypatch.setattr(D, "_squeezemetrics_csv", lambda: _fake_squeeze_df())
    D.dix_proxy()
    monkeypatch.setattr(D, "_squeezemetrics_csv", lambda: pd.DataFrame())

    s = D.dix_proxy()
    assert len(s) == 5
    assert float(s.iloc[-1]) == 0.44
    prov = D.get_provenance()["dix"]
    assert prov["kind"] == "cache"
    assert "2026-06-05" in prov["note"]


def test_fetch_failure_with_no_cache_is_unavailable(monkeypatch, tmp_path):
    _patch_cache_dir(monkeypatch, tmp_path)
    monkeypatch.setattr(D, "_squeezemetrics_csv", lambda: pd.DataFrame())

    s = D.dix_proxy()
    assert s.empty
    assert D.get_provenance()["dix"]["kind"] == "unavailable"
