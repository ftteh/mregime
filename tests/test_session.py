"""Closing basis: a session that is still printing is shown, never scored."""
from __future__ import annotations

import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import data as D
from src import session as SES


@pytest.fixture(autouse=True)
def _no_live_env(monkeypatch):
    monkeypatch.delenv("QUANT_INCLUDE_LIVE", raising=False)


@pytest.mark.parametrize("now,expect", [
    ("2026-11-10 11:12", "2026-11-09"),   # Tuesday, mid-session: Monday is the last close
    ("2026-11-10 16:05", "2026-11-09"),   # bell has rung, closing prints not settled yet
    ("2026-11-10 16:15", "2026-11-10"),
    ("2026-11-10 23:30", "2026-11-10"),
    ("2026-11-11 08:00", "2026-11-10"),   # next morning, before the open
    ("2026-11-14 12:00", "2026-11-13"),   # Saturday
])
def test_closed_cutoff(now, expect):
    assert SES.closed_cutoff(pd.Timestamp(now)) == pd.Timestamp(expect)


def test_cutoff_reads_other_timezones_in_new_york_time():
    # 00:12 in Kuala Lumpur on the 11th is 11:12 in New York on the 10th.
    now = pd.Timestamp("2026-11-11 00:12", tz="Asia/Kuala_Lumpur")
    assert SES.closed_cutoff(now) == pd.Timestamp("2026-11-09")
    assert SES.in_trading_hours(now)


def test_in_trading_hours():
    assert SES.in_trading_hours(pd.Timestamp("2026-11-10 09:30"))
    assert not SES.in_trading_hours(pd.Timestamp("2026-11-10 09:29"))
    assert not SES.in_trading_hours(pd.Timestamp("2026-11-10 16:15"))
    assert not SES.in_trading_hours(pd.Timestamp("2026-11-14 12:00"))   # Saturday


def test_split_live_separates_the_unfinished_bar():
    s = pd.Series([1.0, 2.0, 3.0], index=pd.DatetimeIndex(["2026-11-06", "2026-11-09", "2026-11-10"]))
    closed, live = SES.split_live(s, pd.Timestamp("2026-11-10 11:00"))
    assert closed.index[-1] == pd.Timestamp("2026-11-09") and live.tolist() == [3.0]
    closed, live = SES.split_live(s, pd.Timestamp("2026-11-10 17:00"))
    assert len(closed) == 3 and live is None
    frame = pd.DataFrame({"a": [1.0, 2.0]}, index=pd.DatetimeIndex(["2026-11-09", "2026-11-10"]))
    assert len(SES.drop_live(frame, pd.Timestamp("2026-11-10 11:00"))) == 1


def test_split_live_handles_empty_and_timestamped_rows():
    assert SES.drop_live(pd.Series(dtype=float)).empty
    s = pd.Series([1.0, 2.0], index=pd.DatetimeIndex(["2026-11-09 00:00", "2026-11-10 16:06:38"]))
    assert len(SES.drop_live(s, pd.Timestamp("2026-11-10 12:00"))) == 1


def test_include_live_env_scores_the_open_bar(monkeypatch):
    monkeypatch.setenv("QUANT_INCLUDE_LIVE", "1")
    assert SES.closed_cutoff(pd.Timestamp("2026-11-10 11:00")) == pd.Timestamp("2026-11-10")


def test_fetch_boundary_keeps_the_live_quote_for_display(monkeypatch):
    monkeypatch.setattr(SES, "closed_cutoff", lambda now=None: pd.Timestamp("2026-11-09"))
    D.reset_fetch_memo()
    s = pd.Series([7376.9, 7549.9, 7501.5], index=pd.DatetimeIndex(["2026-11-06", "2026-11-09", "2026-11-10"]))
    closed = D._closed_only(s, "^GSPC")
    assert closed.iloc[-1] == 7549.9
    q = D.get_live_quotes()["^GSPC"]
    assert q["value"] == 7501.5 and q["prev_close"] == 7549.9 and q["date"] == pd.Timestamp("2026-11-10")
    D.reset_fetch_memo()
    assert D.get_live_quotes() == {}


def test_last_completed_trading_day_ignores_a_session_in_progress(monkeypatch):
    """SPX's last index date was treated as a completed day even at 11am."""
    monkeypatch.setattr(SES, "closed_cutoff", lambda now=None: pd.Timestamp("2026-11-09"))
    raw = pd.Series([1.0, 2.0], index=pd.DatetimeIndex(["2026-11-09", "2026-11-10"]))
    monkeypatch.setattr(D, "spx", lambda: D._closed_only(raw))
    assert D.last_completed_trading_day(pd.Timestamp("2026-11-10 11:00")) == pd.Timestamp("2026-11-09")


def test_fear_greed_is_one_closed_row_per_day(monkeypatch):
    """CNN stamps today's value with a time of day next to a midnight row."""
    monkeypatch.setattr(SES, "closed_cutoff", lambda now=None: pd.Timestamp("2026-11-09"))
    ms = lambda t: int(pd.Timestamp(t).value // 10**6)
    payload = {"fear_and_greed_historical": {"data": [
        {"x": ms("2026-11-06"), "y": 30.0}, {"x": ms("2026-11-09"), "y": 28.0},
        {"x": ms("2026-11-10"), "y": 31.0}, {"x": ms("2026-11-10 16:06:38"), "y": 33.6},
    ]}}
    s = D._parse_fear_greed(payload)
    assert s.index.tolist() == [pd.Timestamp("2026-11-06"), pd.Timestamp("2026-11-09")]
    monkeypatch.setattr(SES, "closed_cutoff", lambda now=None: pd.Timestamp("2026-11-10"))
    s = D._parse_fear_greed(payload)
    assert s.index.is_unique and s.iloc[-1] == 33.6
    assert D._parse_fear_greed({}).empty


def test_fallback_chain_drops_a_half_day_row_saved_earlier(monkeypatch, tmp_path):
    """A cache written at 11am must not hand that row back as a close."""
    monkeypatch.setattr(D, "_series_cache_path", lambda fn: str(tmp_path / fn))
    monkeypatch.setattr(D, "SEED_DIR", str(tmp_path / "seeds"))
    monkeypatch.setattr(SES, "closed_cutoff", lambda now=None: pd.Timestamp("2026-11-09"))
    idx = pd.DatetimeIndex(["2026-11-06", "2026-11-09", "2026-11-10"], name="date")
    pd.DataFrame({"forward_pe": [25.0, 25.5, 24.9]}, index=idx).reset_index().to_csv(tmp_path / "f.csv", index=False)
    D.reset_fetch_memo()
    out = D._with_fallback_chain(keys=("k",), live=pd.DataFrame(), cache_file="f.csv", seed_file="none.csv", source="x")
    assert out.index[-1] == pd.Timestamp("2026-11-09")
    D.reset_fetch_memo()
