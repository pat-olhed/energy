# Short-Term Electricity Load Forecasting (German Grid)

Forecasting German electricity demand **1–48h ahead** from open
[SMARD](https://www.smard.de) / [ENTSO-E](https://transparency.entsoe.eu) data.
A time-series project focused on **honest, leakage-free evaluation** against strong
baselines — not just fitting a model.

## Problem

Grid operators and energy traders need short-term load forecasts. This project
predicts hourly German electricity consumption up to 48h ahead and reports how much
a learned model actually improves over naive seasonal baselines.

## Data

- **SMARD.de** (Bundesnetzagentur) — "Realisierter Stromverbrauch" (actual load),
  hourly, Europe/Berlin. Optional weather features from DWD open data.

## Approach

1. EDA: daily/weekly/yearly seasonality, holidays, missing data.
2. Baselines: naive (t−24h) and seasonal-naive (t−168h).
3. Model: gradient boosting (LightGBM) on lag + calendar (+ weather) features.
4. Evaluation: **rolling-origin backtesting**, MAE/MAPE/RMSE vs. baselines, per horizon.

## Results

_TBD — headline: model MAE/MAPE vs. seasonal-naive baseline, per-horizon error curve._

## How to run

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# 1. fetch data (see src/data.py / notebooks/01_eda.ipynb)
jupyter lab
# 2. the artifact:
streamlit run app/streamlit_app.py
```

## Limitations & next steps

_TBD — e.g. weather-forecast uncertainty, holiday edge cases, probabilistic forecasts._

---
*Part of [DS-Portfolio](../README.md).*
