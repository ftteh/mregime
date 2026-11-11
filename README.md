# Quant Regime Dashboard

A daily, institutional-grade market-position monitor. One 0–100 **composite regime score**
built from the same stack hedge funds watch: credit & liquidity, breadth, sentiment,
positioning, vol, and valuation — all from **free data sources**.

> Tops are processes, bottoms are events. This dashboard is designed to spot **clusters**
> of aligned extremes (3–5 indicators together), which is how professionals actually de-risk
> or accumulate.

---

## What you get

**Header** — one glance, one number:
- Composite Regime Score gauge (0 = capitulation, 100 = euphoria), with its rank in its own 3-year range
- SPX / VIX / HY Spread / F&G / AAII tiles
- **Cluster detector**: on when the extremes are lopsided, with one side ahead by a third of the model's
  independent themes (8 of 22) across at least two pillars. A bottom cluster adds exposure; a top cluster
  is a risk flag and changes nothing
- **Today's read**: the exposure call, why, where the composite and the decision desk disagree, what would
  change the call, and what to discount in the data

Everything is scored on **closing data**. While a US session is open, the page shows live quotes for context
and scores through the last close.

**The Four Pillars** (institutional weights):
| Pillar | Weight | What it measures |
|---|---|---|
| Credit & Liquidity | 40% | HY/IG spreads, MOVE, Fed Net Liquidity (WALCL−TGA−RRP), NFCI, yield curve, funding spreads |
| Breadth & Momentum | 30% | % above 200DMA, new highs−lows, A/D line, SPX RSI, 10/20DMA trend breadth, 50D overbought−oversold spread |
| Sentiment & Positioning | 20% | AAII, F&G, Put/Call, VIX, VVIX, SKEW |
| Valuation | 10% | Equity Risk Premium, Nasdaq 100 trailing P/E |

**Pro Watchlist** — the divergences that matter:
- **MOVE / VIX** — when the bond market disagrees with equity vol, bonds win
- **SPY vs (TLT+GLD) 20D correlation** — spots liquidity events ("everything selling together")
- **DIX** — dark-pool institutional flow (when available from SqueezeMetrics)
- **FINRA margin debt** — aggregate customer leverage; YoY growth flags euphoric tops (2000/2007/2021)

**Market internals** — research-grade panels:
- **Short-term trend breadth** — % of all S&P 500 constituents with 10DMA > 20DMA, under the index and its
  200DMA, with a research snapshot (3Y percentile, vs neutral, SPX vs 200DMA, eligible stocks, signal lens)
- **Overbought / oversold breadth** — share of constituents >1σ above / below their own 50-day band
  (σ = 50D rolling std of price) and the OB−OS spread, with adaptive extreme zones
- **Style rotation** — relative strength (IWM/SPX default; RSP/SPY, QQQ/SPY, XLY/XLP) rebased to 100 with
  50D / 200D ratio MAs and a strength / weakness / improving / fading read
- **Nasdaq 100 valuation** — forward P/E (consensus, daily) and trailing P/E (5-year perspective):
  percentile, vs 5Y average, "N-year lows/highs" headline
- **Standardized signals** — every series as a rolling 1-year z-score with 1Y percentile, a signed-fill chart
  and an extremes scanner (|z| ≥ 2), oriented into top-risk / bottom-setup language

**Decision desk** — four independent questions under Today's read, kept separate from the composite:
- **Is the system intact?** Guardrails on breadth, participation, IG / HY / CCC credit, funding, correlation, NFCI
- **Is fear washed out?** A capitulation checklist; 6 core rules count toward the verdict, the rest add context
- **Has price confirmed?** Panic gauges rolling over, QQQ reclaiming its averages, MOVE and yields falling
- **Is leverage safe?** VIX futures crowding, index correlation, dealer gamma, VIX level. Plus the rotation stage

The desk also limits the **Recommended Exposure** card. It can only go above 100% when the leverage gate
isn't closed and at least 3 of 5 confirmations are met; otherwise it's capped at 100% and says why. A bottom cluster
reads Medium conviction until price confirms. If the desk can't run, the card stays capped. The composite score is
not affected.

The composite is a relative read (each indicator against its own last 3 years) and the desk rules are absolute
thresholds, so the two can disagree. **Today's read** says so when they do, for example a composite that reads
fear while the washout rules haven't fired.

**Research desk** — one view at a time:
- **Positioning** — CFTC crowding across 10 futures markets, all-time vs 3-year percentile, flows, QQQ short interest
- **Rotation** — RSP/SPY regime, RSP−XLK by month, sector and cross-asset scoreboards, 20/50/200DMA participation
- **AI & momentum** — AI group and single-name returns, S&P vs AI correlation, momentum and crowded-long proxies
- **Vol & options** — VXN vs VIX, SOX 5-day flushes, SKEW percentile, put and VIX-call volume, F&G components
- **Rates & credit** — cross-asset growth PCA, 10Y weekly RSI episodes, bond composite, MOVE, IG vs CCC, Excess CAPE
- **Technical board** — RSI, moving averages, drawdowns and crosses across assets
- **Playbook** — forward returns for 15 signals next to the base rate, the checklist backtest, midterm seasonality

**Time-series charts** for every key series with annotated reference levels.

**Full indicator table** — raw value, 3Y percentile, top-risk score, freshness.

---

## Quick start

```powershell
# 1. install
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 2. (recommended) add a free FRED API key
#    https://fred.stlouisfed.org/docs/api/api_key.html
copy .env.example .env
#  then edit .env and paste your key

# 3. run
streamlit run app.py
```

First load takes ~30–60s (it downloads SP500 component histories to compute breadth).
After that, everything is cached for 1 hour. Click **Refresh data** in the sidebar to force-refresh.

---

## Deploy free (Streamlit Community Cloud)

[Streamlit Community Cloud](https://streamlit.io/cloud) hosts one app per repo for free. The **free** tier deploys from a **public** GitHub repository (never commit `.env` — it stays gitignored).

1. Push this project to GitHub (see `.gitignore`: `.env` is excluded).
2. Go to [share.streamlit.io](https://share.streamlit.io) → sign in with GitHub → **Create app**.
3. Select your repo, branch **`main`**, main file **`app.py`**, then **Deploy**.
4. After deploy: **App settings** (⚙️) → **Secrets** → paste ([TOML format](https://docs.streamlit.io/streamlit-community-cloud/deploy-your-app/secrets-management)):

   ```toml
   FRED_API_KEY = "your_fred_key_here"
   ```

   Optional (AAII, if your Nasdaq key works from Cloud):

   ```toml
   NASDAQ_DATA_LINK_API_KEY = "optional"
   ```

5. **Save** — the app restarts and picks up keys (same names as `.env` locally).

Cold starts can take **1–2 minutes** (downloads breadth data). The `cache/` put/call file is ephemeral on Cloud and resets between runs; the app still works.

**Private repo + free:** use a public fork for deploy only, or paid Streamlit Team, or host elsewhere (e.g. [Render](https://render.com) free tier with a `Dockerfile` running `streamlit run app.py`).

---

## Data sources (all free)

| Source | What we pull | Method |
|---|---|---|
| **FRED** (St. Louis Fed) | HY spread `BAMLH0A0HYM2`, IG spread `BAMLC0A0CM`, NFCI, WALCL, TGA `WTREGEN`, RRP `RRPONTSYD`, DGS10, 3M commercial paper `DCPF3M` and T-bill `DTB3`, SOFR, EFFR | `fredapi` (API key required) |
| **Yahoo Finance** | ^VIX, ^VVIX, ^SKEW, ^GSPC, ^TNX, SPY, TLT, GLD, SP500 components, SPY options chain (put/call) | `yfinance` |
| **CNN** | Fear & Greed Index (about 5 years of history) | public JSON endpoint |
| **AAII** | Weekly sentiment survey | Nasdaq Data Link (optional key) or aaii.com; last good history kept in the local cache |
| **SqueezeMetrics** | DIX (dark pool index) and SPX dealer gamma (GEX) | public CSV; disk cache |
| **YCharts** | FINRA margin debt (monthly) — FINRA's own page/XLSX is Cloudflare-gated with no data feed | public indicator page (no key); bundled seed + disk cache fallback |
| **Wikipedia** | SP500 constituents | scraped table |
| **VCP Scanner** | Nasdaq 100 forward P/E (bottom-up consensus, daily since 2026-03) | public page, Next.js payload (no key); disk cache + committed seed |
| **WorldPERatio** | Nasdaq 100 trailing P/E (QQQ basis, monthly since 1990 + live point) | public page (no key); disk cache + committed seed |
| **CFTC** | Positioning for 10 futures markets since 2006 (TFF and Disaggregated reports). The scored CTA indicator reads the same feed | Socrata API (no key); disk cache + committed seed |
| **Cboe** | Daily call and put volumes by product since Oct 2019; also the official equity and index put/call ratios | one public JSON per day; committed seed, recent days fetched live |
| **Yahoo Finance** (research) | ^MOVE, ^VXN, ^SOX, ^TYX, ^KS11, sector / factor / AI-basket prices | `yfinance`; disk cache + committed seed |
| **Shiller + multpl** | CAPE and Excess CAPE yield since 1881, extended with FRED | public file and page (no key) |
| **Nasdaq** | QQQ short interest (Nasdaq-listed securities only) | public API (no key) |

All fetchers are wrapped with try/except — if one source is down, the rest of the
dashboard still works. The newer panels (participation breadth, Nasdaq 100 P/E) follow
**live → local disk cache → committed seed** (`src/seeds/*.csv`), because `cache/` is
gitignored and ephemeral on Streamlit Cloud. Refresh the seeds occasionally with
`python -m scripts.refresh_seeds`. The UI flags when a panel is served from cache or seed.

---

## How the composite is built

1. **Closing data** for each indicator, on a business-day grid. A print is carried forward only as far as its
   staleness allowance (10 days for daily feeds, longer for weekly and monthly ones).
2. **Smooth or de-trend** where the indicator's description says so: put/call and new highs use a 10-day average,
   trending macro series a 1-year z-score, funding spreads only vote when above their median.
3. **3-year rolling percentile** (0 = lowest in 3y, 100 = highest). The window grows from 1 year where the feed
   is shorter; under a year of history there is no score.
4. **Orient** so that every score reads the same way:
   - `risk_high_is_top`: raw pct used directly (e.g. AAII bull %)
   - `contrarian_high_is_top`: inverted (e.g. VIX — a high VIX is *bullish*, so score = 100 − pct)
5. **Pillar score** = average of its indicators by config weight (indicators that share a theme are down-weighted).
   A pillar needs half its weight live.
6. **Composite** = weighted average of the four pillars. It needs half the model's weight live, and conviction is
   Low below 80%.
7. **Net exposure** follows the composite along straight lines between the regime bands' levels (130% / 115% /
   100% / 90% / 50% / 20% at each band's centre), so it moves a few points a day instead of jumping 40 points
   at a threshold. A bottom cluster adds 15 points. A top cluster is flagged and does not cut: it has no
   tested edge (`TOP_CLUSTER_CUT` in `src/indicators.py` restores a cut).

An indicator stops voting when it has under 60 observations, under a year of history, or a last print older than
its allowance. The indicator table shows which, and what window each one is ranked over.

### Interpretation

| Score | Label | Action (how pros use it) |
|---|---|---|
| **≥ 85** | Extreme Complacency | De-risk gradually, buy tail-risk protection (puts) |
| 65–85 | Complacent | Trim, tighten stops |
| 45–65 | Neutral | Stay with trend |
| 35–45 | Fearful | Watch for stabilization |
| 15–35 | Panic | Accumulate quality |
| **< 15** | Capitulation | Aggressive accumulation |

> **Never trade on the composite alone.** Use it as context, with the decision desk beside it.
> The composite averages 30-odd percentile scores, so it rarely leaves 25–70: read its rank in its own range.

---

## File map

```
quant/
├── app.py                 # Streamlit dashboard
├── requirements.txt
├── .env.example
├── README.md
└── src/
    ├── config.py          # Indicator specs, bucket weights, regime thresholds
    ├── data.py            # All data fetchers (FRED, yfinance, scrapers) + fallback chains
    ├── session.py         # US session clock: which bars are closed, which are still printing
    ├── indicators.py      # Percentile/z-score engine, composite scorer, cluster, exposure mapping
    ├── analytics.py       # Pure math: breadth, OB/OS, relative strength, valuation stats, z-scores
    ├── charts.py          # Plotly builders for the market-internals / z-score panels
    ├── research.py        # Pure research engine: positioning, correlation, PCA, event studies, decision rules
    ├── research_data.py   # Research feeds with live -> cache -> seed fallback
    ├── research_charts.py # Plotly builders for the research desk
    ├── desk.py            # Assembles rules, tables, the model posture and the reconciled read
    ├── desk_ui.py         # Streamlit rendering for the decision and research desks
    ├── backtest.py        # Daily rule histories and performance tools for scripts/backtest_rules.py
    └── seeds/             # Committed fallback history (regenerate: python -m scripts.refresh_seeds)
```

## Extending

- **Add an indicator**: append an `IndicatorSpec` in `src/config.py`, add a fetcher
  in `src/data.py`, and register it in `build_raw()` in `src/indicators.py`.
- **Change weights**: edit `BUCKET_WEIGHTS` in `src/config.py`.
- **Change thresholds**: edit `REGIME_THRESHOLDS` in `src/config.py`.

## Known caveats

- **MOVE Index**: read from Yahoo (`^MOVE`), with scaled TLT 20-day realized vol as the fallback.
- **Decision desk**: rule output from public data, not investment advice. The rules sit beside the composite and
  never change its score. Proxies are labelled: GVIP−SPY measures how crowded longs traded, not actual
  prime-book gross exposure, and MTUM−SPY is a long-only stand-in for long/short momentum.
- **Rule testing**: `python -m scripts.backtest_rules` tests the decision rules against criteria fixed in advance
  and writes `.claude/plan/rule-backtest-results.md`. The cap passed as risk control. The capitulation count failed
  on untouched 2003–2016 data, so don't read a washout as evidence of a bottom.
- **Composite and cluster are not validated on returns.** The full four-pillar model only exists from late 2022,
  a bull market, and over that window the exposure card did not beat holding 100% (see Test 2 in the results
  file). Bottom clusters were followed by strong returns and top clusters by ordinary ones, on about ten and
  about thirty episodes of overlapping days: descriptive only. That is why a top cluster is a flag and not a cut.
- **Closing basis**: a session still in progress is never scored. Set `QUANT_INCLUDE_LIVE=1` to score the
  unfinished bar; the page then marks every reading provisional.
- **Extremes are relative.** A score near 0 or 100 means extreme against the indicator's own last 3 years. HY
  spreads at 3.3% can score as "wide" after three years of tighter spreads while being tight in absolute terms.
- **Event studies**: permutation p-values on overlapping crisis events, across many signals and horizons. Some rows
  will clear p ≤ 0.05 by chance, and thin buckets are flagged. Treat them as evidence, not forecasts.
- **CFTC**: as of Tuesday, published Friday. Backtests lag flags to publication. Leveraged-fund Treasury shorts are
  largely the basis trade, so read the 3-year percentile alongside the all-time one.
- **ICE credit spreads on FRED**: only about 3 years are served, so older history depends on the local cache and seed.
- **Put/Call**: official Cboe ratios only, through Cboe's last published day. Nothing fills the gap before
  Cboe publishes.
- **Gamma**: the scored series is SqueezeMetrics SPX dealer gamma. The SPY chain snapshot is a different measure
  and is charted, not scored. The flip-zone distance is display only.
- **Funding stress**: 3-month AA commercial paper minus the T-bill. Commercial paper isn't published every day
  and trails by a few sessions.
- **AAII**: the full-history file is blocked on some builds. The last good history is kept in `cache/` and a
  committed seed of the bull-bear spread covers a fresh cloud deploy; the results page then adds recent weeks.
  Refresh the seed with `python -m scripts.refresh_seeds` before it is more than about 20 weeks old.
- **Breadth**: computed across all ~503 current S&P 500 constituents (one 5-year bulk download,
  ~15s), equal-weighted. Uses current membership, so history carries survivorship bias.
- **Nasdaq 100 forward P/E**: Bloomberg's BEst series is paid. The free forward history (VCP Scanner)
  only starts in Mar 2026 and accumulates locally, so its percentile is labelled with its true span;
  the 5-year perspective and the scored valuation input use the trailing P/E (WorldPERatio, QQQ basis).
  Levels differ from Bloomberg by methodology.
- **ERP**: simplified constant earnings-yield proxy. For true Damodaran ERP, ingest
  his monthly spreadsheet from NYU Stern.
