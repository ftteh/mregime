"""
Deterministic regression guardrail for the regime scoring pipeline.

Builds a fixed synthetic RawFrame (no network) and asserts the scoring math
(per-indicator scores, composite, cluster, historical composite) stays stable.
When the methodology is *intentionally* changed, regenerate the fixture with:

    python -m tests.test_regime_snapshot --update

and review the printed deltas — they document the intended accuracy change.
Unintended drift between edits fails the test.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import INDICATORS_BY_KEY
from src import indicators as I

FIXTURE = os.path.join(os.path.dirname(__file__), "regime_snapshot.json")


def build_synthetic_raw() -> I.RawFrame:
    """Deterministic RawFrame spanning every indicator key with realistic shapes."""
    rng = np.random.default_rng(20260605)
    idx = pd.bdate_range("2019-01-01", periods=252 * 5)
    series = {}
    for n, key in enumerate(INDICATORS_BY_KEY):
        # Vary length per key so min-obs / coverage paths are exercised. Keep a
        # few keys deliberately thin: 5 and 8 obs (below rolling min_periods) and
        # 45 obs (in the 30-59 band) — the last specifically guards the keystone
        # gauge==chart invariant across the MIN_OBS gate.
        length = {0: 5, 1: 8, 2: 45}.get(n, int(rng.integers(300, len(idx))))
        vals = np.cumsum(rng.normal(0, 1, length)) + rng.integers(0, 4, length)
        series[key] = pd.Series(vals, index=idx[-length:], name=key)
    meta = {
        k: {"last": float(v.iloc[-1]), "as_of": v.index[-1].to_pydatetime(), "n": int(len(v))}
        for k, v in series.items()
    }
    return I.RawFrame(series=series, meta=meta)


def compute_snapshot() -> dict:
    raw = build_synthetic_raw()
    per_ind = I.per_indicator_history(raw)
    scores = I.score_indicators(raw, per_indicator=per_ind)
    comp = I.composite(scores)
    cluster = I.cluster_signal(scores)
    pillars = I.historical_pillar_scores(raw, per_indicator=per_ind)
    comp_hist = I.historical_composite(raw, pillars=pillars)

    def _round(x):
        return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), 4)

    return {
        "scores": {r["key"]: _round(r["score"]) for _, r in scores.iterrows()},
        "composite": _round(comp.get("composite")),
        "covered_weight": _round(comp.get("covered_weight")),
        "buckets": {b: _round(v["score"]) for b, v in comp["buckets"].items()},
        "cluster": {
            "top_cluster_count": cluster["top_cluster_count"],
            "bottom_cluster_count": cluster["bottom_cluster_count"],
            "top_theme_count": cluster.get("top_theme_count"),
            "bottom_theme_count": cluster.get("bottom_theme_count"),
        },
        "comp_hist_last": _round(comp_hist.iloc[-1]) if not comp_hist.empty else None,
    }


def test_gauge_matches_chart_endpoint():
    """Keystone invariant: live composite == last point of the historical composite."""
    snap = compute_snapshot()
    assert snap["composite"] is not None
    assert snap["comp_hist_last"] is not None
    assert abs(snap["composite"] - snap["comp_hist_last"]) < 1e-6, (
        f"gauge {snap['composite']} != chart endpoint {snap['comp_hist_last']}"
    )


def test_snapshot_matches_fixture():
    snap = compute_snapshot()
    if not os.path.exists(FIXTURE):
        raise AssertionError("Fixture missing — run with --update to generate.")
    with open(FIXTURE) as fh:
        expected = json.load(fh)
    assert snap == expected, "Scoring output drifted from fixture (see diff)."


def _update():
    snap = compute_snapshot()
    prev = {}
    if os.path.exists(FIXTURE):
        with open(FIXTURE) as fh:
            prev = json.load(fh)
    with open(FIXTURE, "w") as fh:
        json.dump(snap, fh, indent=2, sort_keys=True)
    print("Fixture written:", FIXTURE)
    if prev:
        for k in ("composite", "covered_weight", "comp_hist_last"):
            if prev.get(k) != snap.get(k):
                print(f"  {k}: {prev.get(k)} -> {snap.get(k)}")
        for key, val in snap["scores"].items():
            if prev.get("scores", {}).get(key) != val:
                print(f"  score[{key}]: {prev.get('scores', {}).get(key)} -> {val}")


if __name__ == "__main__":
    if "--update" in sys.argv:
        _update()
    else:
        test_gauge_matches_chart_endpoint()
        test_snapshot_matches_fixture()
        print("OK")
