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
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
