"""last_completed_trading_day: live snapshot values must be stamped on a real,
completed US trading session — never a weekend/holiday phantom date.
"""
from __future__ import annotations

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import data as D


def _no_spx(monkeypatch):
    monkeypatch.setattr(D, "spx", lambda: pd.Series(dtype=float))


def test_saturday_rolls_to_friday(monkeypatch):
    _no_spx(monkeypatch)
    out = D.last_completed_trading_day(pd.Timestamp("2026-06-06"))  # Saturday
    assert out == pd.Timestamp("2026-06-05")


def test_sunday_rolls_to_friday(monkeypatch):
    _no_spx(monkeypatch)
    out = D.last_completed_trading_day(pd.Timestamp("2026-06-07"))  # Sunday
    assert out == pd.Timestamp("2026-06-05")


def test_holiday_rollback(monkeypatch):
    """July 4 2026 is a Saturday → observed holiday Friday July 3. From Monday
    July 6 the last *completed* session must skip both weekend and holiday."""
    _no_spx(monkeypatch)
    out = D.last_completed_trading_day(pd.Timestamp("2026-07-06"))  # Monday
    assert out == pd.Timestamp("2026-07-02")  # Thursday


def test_spx_calendar_is_authoritative(monkeypatch):
    s = pd.Series([1.0, 2.0], index=pd.DatetimeIndex(["2026-06-04", "2026-06-05"]))
    monkeypatch.setattr(D, "spx", lambda: s)
    out = D.last_completed_trading_day(pd.Timestamp("2026-06-06"))
    assert out == pd.Timestamp("2026-06-05")


def test_stale_spx_falls_back_to_calendar(monkeypatch):
    s = pd.Series([1.0], index=pd.DatetimeIndex(["2026-05-01"]))  # > 7 days old
    monkeypatch.setattr(D, "spx", lambda: s)
    out = D.last_completed_trading_day(pd.Timestamp("2026-06-08"))  # Monday
    assert out == pd.Timestamp("2026-06-05")  # prior Friday, not stale May date


def test_trading_day_mask_filters_weekends_and_holidays():
    idx = pd.DatetimeIndex([
        "2026-06-05",  # Friday        -> keep
        "2026-06-06",  # Saturday      -> drop
        "2026-07-03",  # observed July-4 holiday -> drop
        "2026-07-06",  # Monday        -> keep
    ])
    mask = D._trading_day_mask(idx)
    assert list(mask) == [True, False, False, True]
