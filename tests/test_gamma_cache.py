"""Gamma cache separation: synthetic/legacy rows are quarantined display-only,
only measured rows are scored, and no methodology can pollute another's cache.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import data as D
from src import indicators as I
from src.config import MIN_OBS


def _patch_cache_dir(monkeypatch, tmp_path):
    monkeypatch.setattr(D, "_series_cache_path", lambda fn: str(tmp_path / fn))


def _write_old_schema(tmp_path, filename, col, n=5):
    idx = pd.bdate_range("2026-05-25", periods=n)
    pd.DataFrame({"date": idx, col: np.linspace(-1.0, 1.0, n)}).to_csv(
        tmp_path / filename, index=False
    )


def test_old_schema_migrates_to_legacy_mixed(monkeypatch, tmp_path):
    _patch_cache_dir(monkeypatch, tmp_path)
    _write_old_schema(tmp_path, "gamma_flip_history.csv", "gamma_flip")

    df = D._load_series_cache_with_source("gamma_flip_history.csv", "gamma_flip")
    assert len(df) == 5
    assert (df["source"] == "legacy_mixed").all()
    # File was rewritten with the source column (migration is persisted)...
    on_disk = pd.read_csv(tmp_path / "gamma_flip_history.csv")
    assert "source" in on_disk.columns
    # ...and a second load is idempotent.
    df2 = D._load_series_cache_with_source("gamma_flip_history.csv", "gamma_flip")
    pd.testing.assert_frame_equal(df, df2)


def test_sourced_roundtrip_preserves_source(monkeypatch, tmp_path):
    _patch_cache_dir(monkeypatch, tmp_path)
    idx = pd.bdate_range("2026-06-01", periods=3)
    df = pd.DataFrame(
        {"gamma_exposure": [1.0, 2.0, 3.0],
         "source": ["legacy_mixed", "cboe_snapshot", "squeezemetrics"]},
        index=idx,
    )
    D._save_series_cache_with_source("gamma_exposure_history.csv", df, "gamma_exposure")
    out = D._load_series_cache_with_source("gamma_exposure_history.csv", "gamma_exposure")
    assert list(out["source"]) == ["legacy_mixed", "cboe_snapshot", "squeezemetrics"]


def test_upsert_new_rows_override_legacy_same_date():
    idx = pd.DatetimeIndex(["2026-06-04", "2026-06-05"])
    df = pd.DataFrame({"gamma_flip": [9.9, 9.9], "source": ["legacy_mixed"] * 2}, index=idx)
    live = pd.Series([1.5], index=pd.DatetimeIndex(["2026-06-05"]))
    out = D._upsert_sourced_rows(df, live, "gamma_flip", "cboe_snapshot")
    assert out.loc[pd.Timestamp("2026-06-05"), "source"] == "cboe_snapshot"
    assert out.loc[pd.Timestamp("2026-06-05"), "gamma_flip"] == 1.5
    assert out.loc[pd.Timestamp("2026-06-04"), "source"] == "legacy_mixed"


def test_flip_fetcher_scores_measured_rows_only(monkeypatch, tmp_path):
    _patch_cache_dir(monkeypatch, tmp_path)
    _write_old_schema(tmp_path, "gamma_flip_history.csv", "gamma_flip", n=4)

    D.reset_fetch_memo()
    monkeypatch.setattr(D, "_cboe_gamma_snapshot", lambda symbol="SPY": (3.0, 1.5))
    monkeypatch.setattr(D, "last_completed_trading_day", lambda now=None: pd.Timestamp("2026-06-05"))
    monkeypatch.setattr(D, "gamma_exposure_proxy", lambda: pd.Series(dtype=float))

    out = D.gamma_flip_zone_distance()
    # Scored series = the single measured snapshot, not the 4 legacy rows.
    assert len(out) == 1
    assert out.index[0] == pd.Timestamp("2026-06-05")
    assert float(out.iloc[0]) == 1.5
    # Legacy rows surface as a display-only overlay, clearly out of scoring.
    overlays = D.get_display_overlays()
    assert "gamma_flip_zone" in overlays
    assert len(overlays["gamma_flip_zone"]) == 4
    # Persisted cache keeps both, attributably.
    on_disk = pd.read_csv(tmp_path / "gamma_flip_history.csv")
    assert set(on_disk["source"]) == {"legacy_mixed", "cboe_snapshot"}
    # Provenance recorded for the UI badge.
    assert D.get_provenance()["gamma_flip_zone"]["kind"] == "primary"


def test_flip_fetcher_honest_absence_when_no_snapshot(monkeypatch, tmp_path):
    _patch_cache_dir(monkeypatch, tmp_path)
    _write_old_schema(tmp_path, "gamma_flip_history.csv", "gamma_flip", n=4)

    D.reset_fetch_memo()
    monkeypatch.setattr(D, "_cboe_gamma_snapshot", lambda symbol="SPY": (None, None))
    monkeypatch.setattr(D, "gamma_exposure_proxy", lambda: pd.Series(dtype=float))

    out = D.gamma_flip_zone_distance()
    assert out.empty  # no yfinance heuristic resurrects a fake value
    assert D.get_provenance()["gamma_flip_zone"]["kind"] == "unavailable"


def test_gex_fetcher_separates_sources(monkeypatch, tmp_path):
    _patch_cache_dir(monkeypatch, tmp_path)
    _write_old_schema(tmp_path, "gamma_exposure_history.csv", "gamma_exposure", n=6)

    D.reset_fetch_memo()
    monkeypatch.setattr(D, "_cboe_gamma_snapshot", lambda symbol="SPY": (4.2, None))
    monkeypatch.setattr(D, "last_completed_trading_day", lambda now=None: pd.Timestamp("2026-06-05"))
    monkeypatch.setattr(D, "squeezemetrics_gex", lambda: pd.Series(dtype=float))

    out = D.gamma_exposure_proxy()
    assert len(out) == 1 and float(out.iloc[0]) == 4.2
    overlays = D.get_display_overlays()
    assert len(overlays["gamma_exposure"]) == 6


def test_thin_measured_history_stays_below_min_obs():
    """A freshly rebuilt measured gamma series (< MIN_OBS rows) must not enter
    the scored panel — honest absence instead of a 2-point '100th percentile'."""
    n = MIN_OBS - 1
    idx = pd.bdate_range("2026-03-02", periods=n)
    series = {"gamma_flip_zone": pd.Series(np.linspace(-1, 1, n), index=idx)}
    meta = {"gamma_flip_zone": {"last": 1.0, "as_of": idx[-1].to_pydatetime(), "n": n}}
    raw = I.RawFrame(series=series, meta=meta)
    panel = I.per_indicator_history(raw)
    assert "gamma_flip_zone" not in panel.columns
