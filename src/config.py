"""Shared configuration: paths, constants, and the knobs the pipeline reads.

Keeping these in one place means the notebooks, the fetch script, the model and
the Streamlit app all agree on horizons, lags, the target column and where data
lives — no magic numbers scattered across modules.
"""
from __future__ import annotations

from pathlib import Path

# --- reproducibility -------------------------------------------------------
SEED = 42

# --- paths -----------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"

DATASET_PARQUET = PROCESSED / "dataset.parquet"
LOAD_PARQUET = PROCESSED / "load_hourly.parquet"
BACKTEST_PARQUET = PROCESSED / "backtest_results.parquet"

# --- time ------------------------------------------------------------------
TZ = "Europe/Berlin"
TARGET = "load_MW"

# How far back to build the dataset. Roughly 4.5 years of hourly data is plenty
# for a rolling-origin backtest and keeps the SMARD fetch (weekly chunks) sane.
DATA_START = "2021-01-01"

# --- SMARD (Bundesnetzagentur) load API ------------------------------------
# chart_data JSON API, no token required.
#   index:  {base}/{filter}/{region}/index_{resolution}.json  -> week-start timestamps
#   series: {base}/{filter}/{region}/{filter}_{region}_{resolution}_{ts}.json
SMARD_BASE = "https://www.smard.de/app/chart_data"
SMARD_LOAD_FILTER = 410          # "Stromverbrauch: Gesamt (Netzlast)"
SMARD_LOAD_REGION = "DE"         # Germany, national grid load
SMARD_RESOLUTION = "hour"

# --- SMARD price + day-ahead forecast series (VERIFIED 2026-07-09) ----------
# Same token-free chart_data API as the load series. IDs come from SMARD's
# market_data_configuration.json (field `data_id`), cross-checked by correlation
# (see PLAN.md / research/02_smard-datenkatalog.md). The day-ahead auction price
# is the headline target; the forecast series are fundamentals known *before* gate
# closure, so they are leakage-free inputs for predicting day D.
SMARD_PRICE_FILTER = 4169            # day-ahead wholesale price DE/LU (history from 2018-09-30)
SMARD_PRICE_REGION = "DE-LU"
SMARD_FORECAST_REGION = "DE"         # the day-ahead forecast series are published for region DE
SMARD_FORECAST_FILTERS = {           # dataset column -> SMARD filter id
    "load_fc_MW": 411,               # forecast consumption / grid load (corr 0.989 vs. actual)
    "wind_on_fc_MW": 123,            # forecast wind onshore
    "wind_off_fc_MW": 3791,          # forecast wind offshore
    "pv_fc_MW": 125,                 # forecast PV  (NB: 125, not 126)
    "resload_fc_MW": 4362,           # forecast residual load, delivered directly (corr 0.991)
}
SMARD_RESLOAD_ACTUAL_FILTER = 4359   # actual residual load — EDA / plausibility cross-check only

# --- price target & gate-closure framing -----------------------------------
# The day-ahead auction fixes all 24 hourly prices of day D at once, at gate
# closure (12:00 Europe/Berlin on day D-1). Every price feature must be known by
# then; price lags are whole-day same-hour shifts (D-1/D-2/D-7). See features.py.
PRICE_TARGET = "price_EUR_MWh"       # SMARD day-ahead price (the pivot's headline target)
GATE_CLOSURE_HOUR = 12               # 12:00 Europe/Berlin on day D-1
PRICE_LAGS = [24, 48, 168]           # same-hour price lags in hours: D-1, D-2, D-7

# --- Energy-Charts (Fraunhofer ISE) ----------------------------------------
ENERGY_CHARTS_BASE = "https://api.energy-charts.info"
PRICE_BIDDING_ZONE = "DE-LU"     # single German-Luxembourg day-ahead zone
GENERATION_COUNTRY = "de"

# --- Open-Meteo weather ----------------------------------------------------
# Population-weighted mean over the largest metropolitan areas is a decent proxy
# for temperature-driven national demand. Weights are population in millions.
OPEN_METEO_ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"
WEATHER_CITIES = [
    {"name": "Berlin", "lat": 52.520, "lon": 13.405, "weight": 3.7},
    {"name": "Hamburg", "lat": 53.551, "lon": 9.993, "weight": 1.9},
    {"name": "Munich", "lat": 48.137, "lon": 11.575, "weight": 1.5},
    {"name": "Cologne", "lat": 50.938, "lon": 6.960, "weight": 1.1},
    {"name": "Frankfurt", "lat": 50.110, "lon": 8.682, "weight": 0.77},
    {"name": "Stuttgart", "lat": 48.775, "lon": 9.182, "weight": 0.63},
]

# --- features & evaluation -------------------------------------------------
HORIZONS = [1, 6, 12, 24, 48]        # hours ahead we forecast and report on
LAGS = [24, 48, 168]                 # lagged-load features (day, 2 days, week)
ROLLING_WINDOWS = [24, 168]          # rolling mean/std windows (all shifted)

# Rolling-origin backtest defaults.
INITIAL_TRAIN_HOURS = 365 * 24       # first fold trains on ~1 year
BACKTEST_STEP_HOURS = 30 * 24        # retrain monthly, score the next month
TRAIN_WINDOW_HOURS = 730 * 24        # rolling 2-year training window (bounds cost, adapts)
