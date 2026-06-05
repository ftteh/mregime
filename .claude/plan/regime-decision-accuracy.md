# Implementation Plan: Make Regime Dashboard Decisions More Accurate After Reading the Chart

> Planning-only output. No production code was modified. External models
> (Codex/Gemini/ace-tool) are **not installed** on this machine — the mandated
> dual-model analysis was substituted with two independent read-only review
> agents (quant-methodology + chart/decision-UX). All findings below are
> verified against source, not assumed.

### Task Type
- [ ] Frontend (→ Gemini)
- [ ] Backend (→ Codex)
- [x] **Fullstack** — signal-integrity (backend logic) **+** decision-support presentation (frontend)

---

## Problem framing

"More accurate decision-making after reading the chart" is **not** a speed problem (that was the prior session). It has two root causes:

1. **Signal integrity** — the number the trader acts on is computed by a weaker, inconsistent engine, and several inputs are degenerate, double-counted, non-stationary, or fabricated from thin history.
2. **Decision legibility** — a single false-precision number (`67.3`) is shown with no confidence, no "why", and charts that visually imply things they don't encode, so the trader mis-reads or over-trusts the dashboard.

Fix integrity first (so the number is right), then legibility (so it's read right).

---

## Evidence (confirmed from code, this session)

| # | Finding | Proof | Impact |
|---|---------|-------|--------|
| A | **Gauge and regime-band chart use two different scoring engines.** Live `score_indicators` uses `latest_percentile` = `tail(756)` in **observations**; historical path resamples to business-day then `rolling_percentile`. Weekly indicators get **14.5y** context live vs **3y** on the chart; `rank()` default `average` vs `method="max"`. | `src/indicators.py:44-51` vs `:25-41`,`:317-367`; measured 14.5y vs 2.9y window | Gauge endpoint ≠ chart endpoint; the *weaker* engine drives the exposure %. Can flip FULLY INVESTED↔TRIM (net 100%↔50%). |
| B | **No minimum-observation gate.** A 2-point series scores `100`. | measured `latest_percentile([1,2]) == 100.0`; `src/indicators.py:255-275` | Thin-history indicators (GEX, gamma_flip, index/equity P/C cache) fabricate extremes that feed composite **and** the high-conviction cluster override. |
| C | **Cluster counts raw indicators, not independent themes**, thresholds (≥85/≤15, count≥4) untuned. | `src/indicators.py:302-311`,`:444-461` | One vol event makes 4–6 collinear sentiment indicators hit extreme at once → spurious "TOP/BOTTOM CLUSTER — High conviction" → forced −20pp/+15pp exposure. |
| D | **Entire 10% Valuation pillar is a sign-flipped duplicate of 10Y rates.** ERP uses a **constant** earnings yield `100/22`. | confirmed `100.0/22.0` in `equity_risk_premium`; valuation bucket = `['equity_risk_premium']` only | Valuation carries zero real valuation info; rates factor double-counted (also in credit bucket via `real_yield_10y`). |
| E | **Sentiment bucket dominated by one collinear vol/options theme** (VIX, VVIX, SKEW, 2× term-structure, GEX, gamma_flip, 2× put/call), equal-weighted. | `src/config.py` bucket members; `index_put_call` even shares the `put_call` live proxy | Orthogonal positioning signals (NAAIM/AAII/CTA) outvoted ~6:3; bucket ≈ "is implied vol low today". |
| F | **`composite()` silently renormalizes when a bucket is missing.** No FRED key → 40% Credit pillar dark, but gauge/exposure still look authoritative. | `src/indicators.py:278-296`; sidebar warns at `app.py:128-140` | A sentiment-heavy `67` is presented as a full-model reading. |
| G | **Percentile-ranking non-stationary/trending macro series** (net_liquidity, dxy, real_yield_10y, copper_gold, ERP). | `src/config.py:71-76`,`218-235`; `rolling_percentile` assumes mean reversion | Series pin at 0/100 for months → permanent non-actionable "extreme" votes & cluster members. `net_liquidity` direction also reads backwards during QT. |
| H | Labeling/robustness: `move_index` is a **TLT realized-vol proxy mislabeled "MOVE"**; breadth/A-D use **current** S&P membership (survivorship) with an 80-megacap fallback; `_line()` colors the **whole** history by today's scalar; reference lines vs percentile score = two mental models; pillar-momentum `_ago(n)` positional offsets break on ffilled weekly data. | `src/data.py:313-327`,`333-414`; `app.py:284-351`,`1057-1065`; `src/indicators.py:499-514` | Over-trust of proxies; misleading chart reads; "stable" momentum that's just no new weekly print. |

---

## Technical Solution (synthesized)

**Keystone refactor (enables most of the plan at ~zero added cost):** have `historical_pillar_scores()` *also* return the per-indicator daily oriented-score frame it already builds internally, plus a daily **coverage** series. Then:

- the **live composite/cluster reads the last row** of that frame → one engine, gauge endpoint == chart endpoint (fixes **A**, and aligns with the already-once-per-load `pillars` computation from the perf work);
- `_line()` can color each point by **that day's** score (fixes **H** line-color);
- regime bands can modulate opacity by **coverage-over-time** (fixes **P4**).

Layer signal-integrity gates (B/C/F) on top, fix the degenerate/collinear/non-stationary inputs (D/E/G) — most via the existing `weight` field and existing velocity/z-score patterns already in the codebase (`hy_spread_velocity`, `curve_resteep_2s10s`) — then add the two decision-support UI elements (confidence chip + "Today's read" card).

**Trust split:** backend/signal items follow the methodology review; presentation items follow the UX review. No conflicts surfaced — they share the same keystone.

---

## Implementation Steps (phased; validate each before next)

### Phase 0 — Safety net (do first)
1. Capture a **golden snapshot** of today's `scores`, `comp`, `cluster`, `comp_hist`, `momentum` to a fixture, and add a tiny harness that recomputes and diffs. Every later step reports its score delta vs golden so "accuracy" changes are intentional, never silent. — *Deliverable: `tests/test_regime_snapshot.py` + saved fixture.*

### Phase 1 — Signal integrity (backend; highest decision-accuracy ROI)
2. **Unify the engine (A).** Widen `historical_pillar_scores` to return `(buckets_df, per_indicator_df, coverage)`. Add `score_indicators(raw, per_indicator_df=None)` that, when given the frame, takes each indicator's **last valid** oriented score instead of `latest_percentile`. Update `composite`/`cluster_signal` to consume it. Keep `latest_percentile` only as a documented fast fallback (resampled + `method="max"` so it matches). — *Gauge == chart endpoint; weekly windows consistent.*
3. **Min-observation gate (B).** Add `MIN_OBS` (≈60 daily / 24 weekly). In scoring, if `n_obs < MIN_OBS` → `score = NaN` (already dropped by composite; also exclude from cluster). Show a "partial/insufficient history" badge in the table. — *No fabricated extremes.*
4. **Coverage/confidence gate (F).** Compute `covered_weight = Σ valid bucket weights`. Cap `exposure_recommendation` conviction at "Low" when `covered_weight < 0.8`; return `composite = NaN` ("INSUFFICIENT DATA") when missing weight ≥ 0.4. — *No confident calls off a half-blind model.*
5. **Theme-diverse cluster (C).** Map collinear indicators to themes (vol-complex, put/call, positioning, credit, curve…); count **one vote per theme** and/or require extremes to span ≥2 buckets before `cluster_override` fires. — *Cluster reflects independent confirmation.*
6. **De-collinearize sentiment (E).** Config-only: set `weight` on the 6 vol/options members (e.g. `1/3` each) so the vol complex ≈ 2 effective slots; NAAIM/AAII/CTA keep `1.0`. `composite` already honors `weight`. — *Positioning signals regain voice.*
7. **Real valuation (D).** Replace constant EY with a real earnings yield (no-key: scrape `multpl.com` S&P PE/CAPE; or NASDAQ Data Link `MULTPL/SP500_PE_RATIO_MONTH` using the key infra already present). `ERP = EY − DGS10`, same orientation, ffilled to daily. — *Valuation pillar carries independent info.*
8. **Stationarize trending macro (G).** For net_liquidity/dxy/real_yield_10y/copper_gold (and post-D, revisit ERP), percentile-rank a **change/velocity or rolling z-score** (reuse the existing `hy_spread_velocity`/`curve_resteep` pattern) instead of the raw level. Re-examine `net_liquidity` orientation. — *No permanent saturation.*

### Phase 2 — Decision legibility (frontend)
9. **Confidence chip under the gauge (P1).** Render coverage (`n/total`, % weight), agreement (σ of indicator scores / pillar scores), and staleness (oldest `as_of`) as a High/Med/Low chip; visually demote the point value on low-confidence days. Data already exists. — *Stops acting on thin/conflicted/stale `67`.*
10. **"Today's read" decision card (P5)** directly under the header: confidence-qualified verdict + top-3 contributors pulling up / down (rank by `|score−50|`) + biggest 1-week pillar mover (from `momentum`) + active divergences hoisted from the Pro Watchlist. — *Replaces 20-chart manual synthesis with a ranked, weighted read.*
11. **Honest line color (H/P2).** In `_line()` (one helper, ~25 charts): default to a neutral line + a single endpoint dot colored by today's score; (better, once Phase 1 exposes per-indicator history) color each point by that day's oriented score. Fix the caption at `app.py:~1000`. — *Color stops implying historical risk it doesn't show.*
12. **Coverage-aware regime bands (P4).** Scale band opacity by that day's coverage; add a faint coverage strip + caption ("bands before ~DATE use fewer indicators"). — *No over-fitting to under-powered historical calls.*
13. **Reconcile percentile vs absolute (P3).** Annotate each reference line with its percentile ("VIX 13 ≈ 8th pct"); state in the subtitle that the composite uses the percentile score, not the absolute lines. — *One load-bearing model.*
14. **Tile legibility (P6).** Merge pillar score + momentum into one tile (level + direction), show weight prominence and partial-coverage badges, state cluster agreement ("Aligned 4↑/0↓" vs "Conflicted 2↑/3↓"), and spell out cluster-override conflicts on the exposure card.

### Phase 3 — Labeling & robustness (H)
15. Relabel `move_index` as "Bond Vol (TLT realized-vol proxy)"; disclose megacap-breadth survivorship in labels and stamp live snapshots with their true quote timestamp (don't override `as_of` with `today()`); fix `pillar_momentum` to measure deltas on calendar offsets / last-changed values rather than positional rows over ffilled data.

---

## Key Files

| File | Operation | Description |
|------|-----------|-------------|
| `src/indicators.py:317-367` | Modify | `historical_pillar_scores` → return `(buckets, per_indicator_df, coverage)` (keystone). |
| `src/indicators.py:255-311` | Modify | `score_indicators`/`composite`/`cluster_signal` consume last-row scores; min-obs gate; theme-diverse clusters. |
| `src/indicators.py:392-488` | Modify | `exposure_recommendation` conviction capped by coverage. |
| `src/indicators.py:44-51` | Modify | `latest_percentile` → resample + `method="max"` (fallback parity). |
| `src/config.py` | Modify | Per-indicator `weight` for vol complex (E); valuation/macro spec & direction updates (D/G). |
| `src/data.py:1015-1034` | Modify | Real earnings yield for ERP (D). |
| `src/data.py:313-327`,`429-466`,`1480-1567` | Modify | MOVE relabel, breadth disclosure, true `as_of` timestamps (H/B). |
| `app.py:110-124` | Modify | `load_all` passes the exposed per-indicator frame + coverage downstream. |
| `app.py:449-489` | Modify | Confidence chip (P1). |
| `app.py:443-647` | Add | "Today's read" decision card (P5). |
| `app.py:279-351` | Modify | `_line` color encoding + caption (H/P2/P3). |
| `app.py:845-925` | Modify | Coverage-aware regime bands (P4). |
| `app.py:648-722` | Modify | Pillar/momentum tile merge + legibility (P6). |
| `tests/test_regime_snapshot.py` | Add | Golden-snapshot guardrail (Phase 0). |

---

## Risks and Mitigation

| Risk | Mitigation |
|------|------------|
| "Improving accuracy" silently changes today's reading in unintended ways | Phase 0 golden-snapshot harness; every step reports intended score deltas vs golden. |
| Unified engine shifts historical regime bands / composite levels | Diff `comp_hist` before/after on a fixed window; document expected shift; the per-indicator frame is already computed once (cheap). |
| Real earnings-yield source (multpl/NDL) flaky or rate-limited | Wrap in the existing try/except + cache pattern; fall back to last-good cache; never crash the pillar. |
| New `weight`s / theme grouping mis-tuned | Make weights & cluster themes config constants; sanity-check that no single theme can alone trigger an override; keep prior behavior one git revert away. |
| Min-obs / coverage gates blank out indicators users expect to see | Show them in the table with an explicit "insufficient history" badge rather than hiding; gate only their *contribution*, not their display. |
| Scope creep across 15 steps | Phases are independently shippable; Phase 1 (steps 2–4) delivers most of the accuracy gain and can ship alone. |

---

## Validation strategy (accuracy must be guarded, not assumed)
- **Equivalence/no-op proof** for refactors that should not change values (engine unification on daily-only indicators).
- **Intended-delta report** for steps that *should* change values (D/E/G, gates) — reviewed, not silent.
- **Coverage/edge cases**: no FRED key (Credit pillar dark), cold caches (thin GEX/PC), all-stale weekly data — assert confidence demotes and no spurious cluster override.
- **Visual QA**: run `streamlit run app.py`, confirm gauge endpoint == regime-band endpoint, confidence chip and decision card reflect coverage/dispersion/staleness.

---

### SESSION_ID (for /ccg:execute use)
- CODEX_SESSION: N/A — external models unavailable on this machine (substituted with local read-only review agents).
- GEMINI_SESSION: N/A — same.
