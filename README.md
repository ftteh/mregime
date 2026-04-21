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
- Composite Regime Score gauge (0 = capitulation, 100 = euphoria)
- SPX / VIX / HY Spread / F&G / NAAIM / AAII tiles
- **Cluster detector**: tells you when ≥4 indicators are in extreme territory

**The Four Pillars** (institutional weights):
| Pillar | Weight | What it measures |
|---|---|---|
| Credit & Liquidity | 40% | HY/IG spreads, MOVE, Fed Net Liquidity (WALCL−TGA−RRP), NFCI |
| Breadth & Momentum | 30% | % above 200DMA, new highs−lows, A/D line, SPX RSI |
| Sentiment & Positioning | 20% | AAII, NAAIM, F&G, Put/Call, VIX, VVIX, SKEW |
| Valuation | 10% | Equity Risk Premium |

**Pro Watchlist** — the divergences that matter:
- **MOVE / VIX** — when the bond market disagrees with equity vol, bonds win
- **SPY vs (TLT+GLD) 20D correlation** — spots liquidity events ("everything selling together")
- **DIX** — dark-pool institutional flow (when available from SqueezeMetrics)

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
| **FRED** (St. Louis Fed) | HY spread `BAMLH0A0HYM2`, IG spread `BAMLC0A0CM`, NFCI, WALCL, TGA `WTREGEN`, RRP `RRPONTSYD`, DGS10 | `fredapi` (API key required) |
| **Yahoo Finance** | ^VIX, ^VVIX, ^SKEW, ^GSPC, ^TNX, SPY, TLT, GLD, SP500 components, SPY options chain (put/call) | `yfinance` |
| **CNN** | Fear & Greed Index (historical) | public JSON endpoint |
| **Nasdaq Data Link** | AAII weekly sentiment (`AAII/AAII_SENTIMENT`) | API key — free, optional |
| **NAAIM** | Active manager exposure (weekly) | scrape programs page for current XLSX URL |
| **yfinance options** | Put/Call computed live from SPY options chain (put vol / call vol across 3 front expiries) | daily snapshot, cached to disk |
| **SqueezeMetrics** | DIX (dark pool index) | public CSV |
| **Wikipedia** | SP500 constituents | scraped table |

All fetchers are wrapped with try/except — if one source is down, the rest of the
dashboard still works.

---

## How the composite is built

1. **Raw data** for each indicator (daily, weekly — resampled where needed).
2. **3-year rolling percentile** (0 = lowest in 3y, 100 = highest).
3. **Orient** so that every score reads the same way:
   - `risk_high_is_top`: raw pct used directly (e.g. AAII bull %)
   - `contrarian_high_is_top`: inverted (e.g. VIX — a high VIX is *bullish*, so score = 100 − pct)
4. **Bucket score** = equal-weighted average of indicator scores within a pillar.
5. **Composite** = weighted average of the four pillars.

### Interpretation

| Score | Label | Action (how pros use it) |
|---|---|---|
| **≥ 85** | Extreme Complacency | De-risk gradually, buy tail-risk protection (puts) |
| 65–85 | Complacent | Trim, tighten stops |
| 45–65 | Neutral | Stay with trend |
| 35–45 | Fearful | Watch for stabilization |
| 15–35 | Panic | Accumulate quality |
| **< 15** | Capitulation | Aggressive accumulation |

> **Never trade on the composite alone.** Use it as context. The real signal is a
> cluster of 3–5 pillar-extreme indicators (shown in the "Cluster signals" card).

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
    ├── data.py            # All data fetchers (FRED, yfinance, scrapers)
    └── indicators.py      # Percentile/z-score engine, composite scorer
```

## Extending

- **Add an indicator**: append an `IndicatorSpec` in `src/config.py`, add a fetcher
  in `src/data.py`, and register it in `build_raw()` in `src/indicators.py`.
- **Change weights**: edit `BUCKET_WEIGHTS` in `src/config.py`.
- **Change thresholds**: edit `REGIME_THRESHOLDS` in `src/config.py`.

## Known caveats

- **MOVE Index**: proprietary (ICE BofAML). We proxy with scaled TLT 20-day realized vol.
  For the real series, add an ICE subscription or scrape `markets.ft.com/data/indices/tearsheet/summary?s=MOVE:IOM`.
- **Put/Call**: CBOE rotates their free CSV URLs occasionally — may need a minor patch.
- **Breadth**: computed on a 150-ticker sample of SP500 (by market cap order) for speed.
  Correlates >0.97 with full-index breadth.
- **ERP**: simplified constant earnings-yield proxy. For true Damodaran ERP, ingest
  his monthly spreadsheet from NYU Stern.
