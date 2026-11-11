"""
US equity session clock: which daily bars are finished, and which are still printing.

Yahoo returns today's bar while the session is open, with the latest trade in the
Close column. Every rule and backtest in this project is defined on closes, so a
bar that is still printing must not be scored as one: a VIX that is up 30% at
11am is not a "+30% close". Fetchers drop rows dated after `closed_cutoff()`;
the dropped quote is kept separately so the page can still show it, labelled live.

Set QUANT_INCLUDE_LIVE=1 to score the in-progress bar anyway (the page then
labels the read as provisional).
"""
from __future__ import annotations

import os
from datetime import time

import pandas as pd

NY_TZ = "America/New_York"
OPEN_TIME = time(9, 30)
# 15 minutes after the bell, so index and ETF closing prints have settled.
CLOSE_TIME = time(16, 15)


def include_live() -> bool:
    return os.getenv("QUANT_INCLUDE_LIVE", "").strip().lower() in ("1", "true", "yes")


def now_et(now: pd.Timestamp | None = None) -> pd.Timestamp:
    """Current time in New York. A naive `now` is read as New York wall-clock time."""
    if now is None:
        return pd.Timestamp.now(tz=NY_TZ)
    ts = pd.Timestamp(now)
    return ts.tz_localize(NY_TZ) if ts.tzinfo is None else ts.tz_convert(NY_TZ)


def closed_cutoff(now: pd.Timestamp | None = None) -> pd.Timestamp:
    """Latest calendar date whose US session, if it had one, has closed.

    No holiday calendar is needed: a row dated after this can only exist if a
    session is printing right now.
    """
    n = now_et(now)
    today = n.tz_localize(None).normalize()
    if include_live() or n.time() >= CLOSE_TIME:
        return today
    return today - pd.Timedelta(days=1)


def in_trading_hours(now: pd.Timestamp | None = None) -> bool:
    """Weekday between the open and the settled close. Holidays read as open,
    which only makes callers a little more cautious."""
    n = now_et(now)
    return n.weekday() < 5 and OPEN_TIME <= n.time() < CLOSE_TIME


def _dates(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    idx = index.tz_localize(None) if index.tz is not None else index
    return idx.normalize()


def split_live(obj, now: pd.Timestamp | None = None):
    """(closed rows, in-progress rows) of a date-indexed Series or DataFrame."""
    if obj is None or len(obj) == 0 or not isinstance(obj.index, pd.DatetimeIndex):
        return obj, None
    mask = _dates(obj.index) <= closed_cutoff(now)
    if mask.all():
        return obj, None
    return obj[mask], obj[~mask]


def drop_live(obj, now: pd.Timestamp | None = None):
    """Rows up to the last closed session."""
    return split_live(obj, now)[0]
