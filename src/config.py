"""Shared configuration: paths, constants, and the knobs the pipeline reads.

Keeping these in one place means the notebooks, the fetch script, the model and
the Streamlit app all agree on the target column, the gate-closure framing and
where data lives — no magic numbers scattered across modules.
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

# --- time ------------------------------------------------------------------
TZ = "Europe/Berlin"

# How far back to build the dataset. Roughly 5 years of hourly data is plenty for a
# rolling-origin backtest and keeps the SMARD fetch (weekly chunks) sane.
DATA_START = "2021-01-01"

# --- SMARD (Bundesnetzagentur) chart_data API ------------------------------
# Token-free JSON API.
#   index:  {base}/{filter}/{region}/index_{resolution}.json  -> week-start timestamps
#   series: {base}/{filter}/{region}/{filter}_{region}_{resolution}_{ts}.json
SMARD_BASE = "https://www.smard.de/app/chart_data"
SMARD_RESOLUTION = "hour"

# --- SMARD price + day-ahead forecast series (VERIFIED 2026-07-09) ----------
# IDs come from SMARD's market_data_configuration.json (field `data_id`), cross-checked
# by correlation. The day-ahead auction price is the target; the forecast series are
# fundamentals known *before* gate closure, so they are leakage-free inputs for day D.
SMARD_PRICE_FILTER = 4169            # day-ahead wholesale price DE/LU (history from 2018-09-30)
SMARD_PRICE_REGION = "DE-LU"
SMARD_FORECAST_REGION = "DE"         # the day-ahead forecast series are published for region DE
SMARD_FORECAST_FILTERS = {           # dataset column -> SMARD filter id
    "load_fc_MW": 411,               # forecast consumption (corr 0.989 vs. actual)
    "wind_on_fc_MW": 123,            # forecast wind onshore
    "wind_off_fc_MW": 3791,          # forecast wind offshore
    "pv_fc_MW": 125,                 # forecast PV  (NB: 125, not 126)
    "resload_fc_MW": 4362,           # forecast residual load, delivered directly (corr 0.991)
}

# --- price target & gate-closure framing -----------------------------------
# The day-ahead auction fixes all 24 hourly prices of day D at once, at gate closure
# (12:00 Europe/Berlin on day D-1). Every price feature must be known by then; price
# lags are whole-day same-hour shifts (D-1/D-2/D-7). See features.py.
PRICE_TARGET = "price_EUR_MWh"       # SMARD day-ahead price (the headline target)
GATE_CLOSURE_HOUR = 12               # 12:00 Europe/Berlin on day D-1
PRICE_LAGS = [24, 48, 168]           # same-hour price lags in hours: D-1, D-2, D-7

# --- rolling-origin backtest defaults --------------------------------------
INITIAL_TRAIN_HOURS = 365 * 24       # first fold trains on ~1 year
BACKTEST_STEP_HOURS = 30 * 24        # retrain monthly, score the next month
TRAIN_WINDOW_HOURS = 730 * 24        # rolling 2-year training window (bounds cost, adapts)
