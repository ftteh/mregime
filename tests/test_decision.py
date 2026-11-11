"""The reconciled read: what the card says, where the two reads disagree, and
what would change the call."""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import desk as K
from src import indicators as I
from src.research import Rule

HIGH = {"level": "High", "coverage_pct": 100.0, "dispersion": 22.0, "n_stale": 0, "n_overdue": 0, "n_proxy": 0}


def _rule(key: str, fired: bool, label: str | None = None) -> Rule:
    return Rule(key, label or key, 1.0, "x", fired)


def _desk(*, guard="INTACT", cap="QUIET", core=0, confirm=0, gate="OPEN", hits=0, issues=()) -> dict:
    return {
        "guard": {"state": guard, "held": 8 if guard == "INTACT" else 6, "total": 8},
        "cap": {"state": cap, "core_fired": core, "core_total": 6},
        "gate": {"state": gate, "hits": hits},
        "confirm": [_rule(f"c{i}", i < confirm, f"Confirmation {i}") for i in range(5)],
        "guards": [_rule("breadth", guard == "INTACT", "Short-term breadth holds"), _rule("ig", True, "IG calm")],
        "posture": {"head": "Posture line"},
        "freshness": {"issues": list(issues)},
    }


def _cluster(top=0, bot=0) -> dict:
    return {"top_theme_count": top, "bottom_theme_count": bot, "top_buckets": ["a", "b"], "bottom_buckets": ["a", "b"],
            "themes_live": 22, "need": 8}


def _read(score, desk, *, cluster=None, confidence=HIGH, session=None):
    cluster = cluster or _cluster()
    comp = {"composite": score, "covered_weight": 1.0}
    rec = K.apply_leverage_cap(I.exposure_recommendation(score, cluster, 1.0), K.leverage_permission(desk))
    return K.decision(rec, comp, cluster, confidence, desk, session=session), rec


def test_offline_composite_is_no_call():
    d = K.decision(I.exposure_recommendation(float("nan")), {"composite": float("nan"), "covered_weight": 0.3}, _cluster(), HIGH, None)
    assert d["headline"].startswith("No call") and "30%" in d["sub"]


def test_relative_fear_without_a_washout_is_flagged():
    """The composite is relative, the desk rules absolute. When they disagree, say so."""
    d, _ = _read(30.0, _desk(cap="BUILDING", core=1))
    assert len(d["conflicts"]) == 1 and "relative extreme" in d["conflicts"][0] and "1 of 6" in d["conflicts"][0]
    agree, _ = _read(30.0, _desk(cap="CAPITULATION", core=6))
    assert agree["conflicts"] == []


def test_washout_with_a_neutral_composite_is_flagged():
    d, _ = _read(55.0, _desk(cap="ARMED", core=3))
    assert any("concentrated" in c for c in d["conflicts"])


def test_breaking_guardrails_against_a_full_position_is_flagged():
    d, rec = _read(40.0, _desk(guard="BREAKING"))
    assert rec["net_pct"] == 100
    assert any("Guardrails are breaking" in c for c in d["conflicts"])
    light, _ = _read(80.0, _desk(guard="BREAKING"))
    assert not any("Guardrails are breaking" in c for c in light["conflicts"])


def test_capped_card_says_what_is_missing():
    d, rec = _read(25.0, _desk(cap="CAPITULATION", core=6, confirm=2, gate="REDUCED", hits=1))
    assert rec["net_pct"] == 100 and rec["uncapped_pct"] == 115
    assert rec["label"] == "FULLY INVESTED"                     # the label follows the capped figure
    lever = next(t for t in d["triggers"] if t.startswith("Above 100%"))
    assert "now 2" in lever and "Confirmation 2; Confirmation 3; Confirmation 4" in lever
    assert any("asks for 115%" in w for w in d["why"])


def test_leverage_on_says_what_takes_it_off():
    d, rec = _read(25.0, _desk(cap="CAPITULATION", core=6, confirm=4, gate="REDUCED", hits=1))
    assert rec["net_pct"] == 115
    assert any(t.startswith("Leverage is on") for t in d["triggers"])


def test_cluster_label_survives_the_cap():
    d, rec = _read(30.0, _desk(cap="CAPITULATION", core=6, confirm=1), cluster=_cluster(top=1, bot=9))
    assert rec["label"].startswith("BOTTOM CLUSTER") and rec["net_pct"] == 100
    assert any(w.startswith("Cluster: 9 independent themes at a bottom extreme against 1") for w in d["why"])


def test_top_cluster_flag_is_explained_and_does_not_move_the_card():
    d, rec = _read(60.0, _desk(), cluster=_cluster(top=9, bot=1))
    plain, plain_rec = _read(60.0, _desk())
    assert rec["net_pct"] == plain_rec["net_pct"] and rec["label"] == plain_rec["label"]
    assert any(w.startswith("Top-cluster flag: 9 themes at a top extreme against 1") for w in d["why"])
    assert d["headline"] == plain["headline"]


def test_no_desk_is_a_caveat_and_stays_capped():
    d, rec = _read(20.0, None)
    assert rec["net_pct"] == 100
    assert any("Decision desk unavailable" in c for c in d["caveats"]) and d["sub"] == "Composite read only"


def test_session_in_progress_and_data_issues_are_caveats():
    live = {"live": True, "as_of": pd.Timestamp("2026-11-09")}
    d, _ = _read(50.0, _desk(issues=("CFTC data is stale, last 29 Sep 2026",)), session=live)
    assert d["caveats"][0].startswith("US session in progress") and "Mon 09 Nov" in d["caveats"][0]
    assert "CFTC data is stale, last 29 Sep 2026." in d["caveats"]
    low = {"level": "Low", "coverage_pct": 62.0, "dispersion": 31.0, "n_stale": 2, "n_overdue": 0, "n_proxy": 1}
    d, _ = _read(50.0, _desk(), confidence=low)
    note = next(c for c in d["caveats"] if c.startswith("Low confidence"))
    assert "62% of model weight" in note and "2 indicators dropped as stale" in note and "1 proxy or fallback feed" in note


def test_label_bounds_come_from_the_live_mapping():
    lo, hi = K._label_bounds(50.0)
    here = I.exposure_label(I.base_exposure(50.0)[0])[0]
    assert I.exposure_label(I.base_exposure(lo)[0])[0] != here and I.exposure_label(I.base_exposure(lo + 1)[0])[0] == here
    assert I.exposure_label(I.base_exposure(hi)[0])[0] != here and I.exposure_label(I.base_exposure(hi - 1)[0])[0] == here
    assert K._label_bounds(2.0)[0] is None and K._label_bounds(99.0)[1] is None


def test_near_cluster_is_a_trigger():
    d, _ = _read(40.0, _desk(), cluster=_cluster(top=0, bot=7))
    assert any("1 more theme at a bottom extreme" in t for t in d["triggers"])
    far, _ = _read(40.0, _desk(), cluster=_cluster(top=2, bot=3))
    assert not any("would make it a cluster" in t for t in far["triggers"])


def test_watch_state_names_the_failed_guardrail():
    d, _ = _read(50.0, _desk(guard="WATCH"))
    assert any("Short-term breadth holds" in t for t in d["triggers"])
