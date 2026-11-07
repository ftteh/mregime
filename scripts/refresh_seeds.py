"""
Regenerate the committed fallback seeds in src/seeds/.

The dashboard's new panels (S&P 500 participation breadth, Nasdaq 100 forward /
trailing P/E) follow live -> local cache -> committed seed. `cache/` is
gitignored (ephemeral on Streamlit Cloud), so these seeds are what a fresh
cloud deploy shows when Yahoo / a scraped page is blocked on a shared IP.

Run from the repo root when the seeds drift too far behind:

    python -m scripts.refresh_seeds

Each file is only overwritten when the live pull succeeds, and existing seed
rows outside the live range are preserved (append-only history).
"""
from __future__ import annotations

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import analytics as A  # noqa: E402
from src import data as D  # noqa: E402
from src import research_data as RD  # noqa: E402

SHILLER_XLS = (
    "https://img1.wsimg.com/blobby/go/e5e77e0b-59d1-44d9-ab25-4763ac982e53/downloads/"
    "02d69a38-97f2-45f8-941d-4e4c5b50dea7/ie_data.xls"
)


def _shiller_seed() -> pd.DataFrame:
    """Shiller's monthly CAPE and Excess CAPE Yield (% terms) from ie_data.xls."""
    import requests
    import xlrd

    content = requests.get(SHILLER_XLS, headers=D.UA, timeout=60).content
    sheet = xlrd.open_workbook(file_contents=content).sheet_by_name("Data")
    rows = {}
    for i in range(8, sheet.nrows):
        date, cape, ecy = sheet.cell_value(i, 0), sheet.cell_value(i, 12), sheet.cell_value(i, 16)
        if not isinstance(date, float) or not isinstance(cape, float) or not isinstance(ecy, float):
            continue
        year, month = int(date), int(round((date - int(date)) * 100))
        rows[pd.Timestamp(year=year, month=max(month, 1), day=1)] = {"ecy": ecy * 100, "cape": cape}
    return pd.DataFrame.from_dict(rows, orient="index").sort_index()


def _research_seeds() -> bool:
    """Seeds for the research desk. Long option-volume history is backfilled here
    rather than at runtime: Cboe serves one file per day."""
    ok = True
    print("Research prices (daily closes since 2019)…")
    px = RD.fetch_prices_live()
    if not px.empty:
        trimmed = px.loc["2019-01-01":].apply(lambda c: c.map(lambda v: float(f"{v:.6g}") if pd.notna(v) else v))
        trimmed = trimmed[trimmed.index.dayofweek < 5]
        _write("research_prices_seed.csv", trimmed, 6)
    else:
        print("  SKIPPED: price download failed")
        ok = False

    print("S&P 500 month-end closes since 1950…")
    long_spx = D.yf_series("^GSPC", period="max")
    if not long_spx.empty:
        _write("spx_monthly_seed.csv", long_spx.loc["1950-01-01":].resample("ME").last().rename("close").to_frame(), 2)
    else:
        ok = False

    print("CFTC positioning (full history)…")
    cot = RD.fetch_cot_live()
    if not cot.empty:
        _write("cot_positioning_seed.csv", cot, 2)
    else:
        ok = False

    print("Cboe option volumes (from 2019-10, about 1,800 daily files)…")
    prev = D._load_seed_frame("cboe_volume_seed.csv")
    start = prev.index.max() + pd.Timedelta(days=1) if not prev.empty else pd.Timestamp("2019-10-07")
    cb = RD.fetch_cboe_range(start, pd.Timestamp.today().normalize())
    if not cb.empty:
        _write("cboe_volume_seed.csv", cb, 0)
    elif prev.empty:
        ok = False

    print("Credit, short interest, S&P forward P/E, S&P internals…")
    ccc = RD.ccc_oas()
    if not ccc.empty:
        _write("ccc_oas_seed.csv", ccc.rename("ccc_oas").to_frame(), 2)
    si = RD.qqq_short_interest()
    if not si.empty:
        _write("qqq_short_interest_seed.csv", si.rename("qqq_si_m").to_frame(), 3)
    fpe = RD.sp500_forward_pe()
    if not fpe.empty:
        _write("sp500_forward_pe_seed.csv", fpe.rename("forward_pe").to_frame(), 4)
    internals = RD.sp500_internals()
    if not internals.empty:
        _write("sp500_internals_seed.csv", internals, 3)

    print("Excess CAPE (Shiller history + live extension)…")
    try:
        seed = _shiller_seed()
        ext = RD.excess_cape_frame()
        cape = D._merge_frames(seed, ext) if not ext.empty else seed
        _write("excess_cape_seed.csv", cape, 3)
    except Exception as e:  # noqa: BLE001
        print(f"  SKIPPED: {e}")
        ok = False
    return ok


def _write(filename: str, df: pd.DataFrame, decimals: int = 3) -> None:
    path = D._seed_path(filename)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    prev = D._load_seed_frame(filename)
    merged = D._merge_frames(prev, df.round(decimals))
    merged.index.name = "date"
    merged.reset_index().to_csv(path, index=False, date_format="%Y-%m-%d")
    print(f"  wrote {filename}: {len(merged)} rows, "
          f"{merged.index.min():%Y-%m-%d} -> {merged.index.max():%Y-%m-%d}")


def main() -> int:
    ok = True
    print("S&P 500 participation (full constituent panel)…")
    panel, wanted = D._panel_columns()
    if wanted:
        p = panel[wanted]
        tb = A.trend_breadth(p, 10, 20).rename(columns={"pct": "trend_10_20", "eligible": "trend_eligible"})
        ob = A.overbought_oversold_breadth(p, 50, 1.0).rename(
            columns={"spread": "obos_spread", "eligible": "obos_eligible"})
        _write("sp500_participation_seed.csv", pd.concat([tb, ob], axis=1).sort_index())
    else:
        print("  SKIPPED: panel download failed")
        ok = False

    print("Nasdaq 100 forward P/E (VCP Scanner)…")
    v = D._vcp_ndx_history()
    if not v.empty and "forward_pe" in v.columns and v["forward_pe"].notna().any():
        _write("ndx_forward_pe_seed.csv", v[["forward_pe"]].dropna(), 4)
    else:
        print("  SKIPPED: VCP Scanner unreachable / layout changed")
        ok = False

    print("Nasdaq 100 trailing P/E (WorldPERatio)…")
    w = D._worldperatio_ndx()
    if not w.empty:
        _write("ndx_trailing_pe_seed.csv", w.to_frame("trailing_pe"), 4)
    else:
        print("  SKIPPED: WorldPERatio unreachable / layout changed")
        ok = False

    ok = _research_seeds() and ok
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
