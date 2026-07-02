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
