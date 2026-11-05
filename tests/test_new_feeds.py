"""New feeds: full-universe participation breadth and Nasdaq 100 P/E.

Covers page parsers (VCP Scanner flight payload, WorldPERatio inline series),
the live -> disk cache -> committed seed fallback chain, and seed integrity.
No network: every live leg is monkeypatched.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import data as D


def _isolate(monkeypatch, tmp_path, seed_dir=None):
    monkeypatch.setattr(D, "_series_cache_path", lambda fn: str(tmp_path / fn))
    seeds = seed_dir if seed_dir is not None else tmp_path / "no_seeds"
    monkeypatch.setattr(D, "_seed_path", lambda fn: str(seeds / fn))
    D.reset_fetch_memo()


# --- VCP Scanner -------------------------------------------------------------
def _vcp_html(rows: list[dict]) -> str:
    payload = '"initialIndex":"nasdaq100","initialHistory":' + json.dumps(rows) + ',"x":1'
    escaped = json.dumps(payload)[1:-1]  # as embedded in the Next.js flight push
    return f'<script>self.__next_f.push([1,"{escaped}"])</script>'


def test_parse_vcp_history_extracts_forward_pe():
    rows = [
        {"snapshot_date": "2026-11-04", "trailing_pe": 37.2, "forward_pe": 22.17},
        {"snapshot_date": "2026-11-03", "trailing_pe": 36.6, "forward_pe": 21.78},
        {"snapshot_date": "2021-01-04", "trailing_pe": 22.1, "forward_pe": None},
        {"snapshot_date": "2026-05-01", "trailing_pe": 1e6, "forward_pe": -4},  # garbage
    ]
    df = D._parse_vcp_history(_vcp_html(rows))
    fwd = df["forward_pe"].dropna()
    assert list(fwd.index.strftime("%Y-%m-%d")) == ["2026-11-03", "2026-11-04"]
    assert fwd.iloc[-1] == 22.17
    assert df.index.is_monotonic_increasing


def test_parse_vcp_history_layout_change_is_empty_not_crash():
    assert D._parse_vcp_history("<html>redesigned</html>").empty


# --- WorldPERatio -----------------------------------------------------------
_WPR = """
<p>The estimated Price-to-Earnings (P/E) Ratio for Nasdaq 100 Index is <b>26.21</b> ,
calculated on <b>04 November 2026</b> .</p>
<script> detailPE_data = [[Date.UTC(2026, 8, 1),30.0555],[Date.UTC(2026, 9, 1),26.1573],
[Date.UTC(2026, 10, 1),26.2126]]; </script>
"""


def test_parse_worldperatio_month_end_and_live_restamp():
    s = D._parse_worldperatio(_WPR)
    assert list(s.index.strftime("%Y-%m-%d")) == ["2026-09-30", "2026-10-31", "2026-11-04"]
    assert s.iloc[-1] == 26.21          # live value from the page text
    assert s.iloc[1] == 26.1573         # October month-end reading


def test_parse_worldperatio_missing_series():
    assert D._parse_worldperatio("<html></html>").empty


# --- fallback chain ---------------------------------------------------------
def test_forward_pe_live_then_cache(monkeypatch, tmp_path):
    _isolate(monkeypatch, tmp_path)
    live = pd.DataFrame({"forward_pe": [21.5, 22.0]},
                        index=pd.to_datetime(["2026-11-03", "2026-11-04"]))
    monkeypatch.setattr(D, "_vcp_ndx_history", lambda: live)
    s = D.ndx_forward_pe()
    assert s.iloc[-1] == 22.0
    assert D.get_provenance()["ndx_forward_pe"]["kind"] == "primary"

    D.reset_fetch_memo()
    monkeypatch.setattr(D, "_vcp_ndx_history", lambda: pd.DataFrame())
    s2 = D.ndx_forward_pe()
    assert s2.iloc[-1] == 22.0                       # served from disk cache
    assert D.get_provenance()["ndx_forward_pe"]["kind"] == "cache"


def test_forward_pe_falls_back_to_seed(monkeypatch, tmp_path):
    seed_dir = tmp_path / "seeds"
    seed_dir.mkdir()
    pd.DataFrame({"date": ["2026-10-30"], "forward_pe": [21.9]}).to_csv(
        seed_dir / "ndx_forward_pe_seed.csv", index=False)
    _isolate(monkeypatch, tmp_path, seed_dir)
    monkeypatch.setattr(D, "_vcp_ndx_history", lambda: pd.DataFrame())
    s = D.ndx_forward_pe()
    assert s.iloc[-1] == 21.9
    assert D.get_provenance()["ndx_forward_pe"]["kind"] == "fallback"


def test_forward_pe_all_sources_down(monkeypatch, tmp_path):
    _isolate(monkeypatch, tmp_path)
    monkeypatch.setattr(D, "_vcp_ndx_history", lambda: pd.DataFrame())
    assert D.ndx_forward_pe().empty
    assert D.get_provenance()["ndx_forward_pe"]["kind"] == "unavailable"


def test_trailing_pe_drops_future_month_end(monkeypatch, tmp_path):
    _isolate(monkeypatch, tmp_path)
    future = pd.Timestamp.now().normalize() + pd.offsets.MonthEnd(1)
    s = pd.Series([30.0, 29.0], index=[pd.Timestamp("2026-01-31"), future], name="ndx_trailing_pe")
    monkeypatch.setattr(D, "_worldperatio_ndx", lambda: s)
    out = D.ndx_trailing_pe()
    assert out.index.max() <= pd.Timestamp.now().normalize()


def test_participation_frame_from_panel_and_seed_fallback(monkeypatch, tmp_path):
    _isolate(monkeypatch, tmp_path)
    idx = pd.bdate_range("2025-01-01", periods=120)
    t = np.arange(120, dtype=float)
    panel = pd.DataFrame({"A": 100 + t, "B": 200 - t, "C": 50 + 0.2 * t}, index=idx)
    monkeypatch.setattr(D, "_panel_columns", lambda sample_size=None: (panel, list(panel.columns)))
    df = D.sp500_participation_frame()
    assert {"trend_10_20", "obos_spread", "ob_pct", "os_pct"}.issubset(df.columns)
    assert df["trend_10_20"].iloc[-1] == pytest.approx(200 / 3)
    assert D.get_provenance()["trend_breadth_10_20"]["kind"] == "primary"
    assert not D.trend_breadth_10_20().empty      # memoized frame reused

    # Panel download fails -> cache from the first build serves the series.
    D.reset_fetch_memo()
    monkeypatch.setattr(D, "_panel_columns", lambda sample_size=None: (pd.DataFrame(), []))
    s = D.obos_spread_50d()
    assert not s.empty
    assert D.get_provenance()["obos_spread_50d"]["kind"] == "cache"


# --- committed seeds ---------------------------------------------------------
def test_committed_seeds_are_sane():
    seeds = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src", "seeds")
    part = D._read_dated_csv(os.path.join(seeds, "sp500_participation_seed.csv"))
    fwd = D._read_dated_csv(os.path.join(seeds, "ndx_forward_pe_seed.csv"))
    trl = D._read_dated_csv(os.path.join(seeds, "ndx_trailing_pe_seed.csv"))
    for df in (part, fwd, trl):
        assert not df.empty and df.index.is_monotonic_increasing and not df.index.duplicated().any()
    assert part["trend_10_20"].between(0, 100).all()
    assert part["obos_spread"].dropna().between(-100, 100).all()
    assert fwd["forward_pe"].between(5, 80).all()
    assert trl["trailing_pe"].between(5, 120).all()
    assert len(trl) > 300  # decades of monthly history for the 5Y perspective
