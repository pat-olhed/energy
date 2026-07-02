# Short-Term Electricity Load Forecasting (German Grid)

Forecasting German electricity demand **1–48h ahead** from open data. The focus is
**honest, leakage-free evaluation against strong baselines** — the model is only
interesting insofar as it beats them, measured on a proper rolling-origin backtest.

## Problem

Grid operators and traders need short-term load forecasts. This project predicts hourly
German grid load (Netzlast) up to 48 hours ahead and reports, per horizon, how much a
learned model actually improves over naive seasonal baselines.

## Data

All sources are open and require no API token; a fresh clone rebuilds everything with
`python -m src.data`.

| Source | Role | Detail |
| --- | --- | --- |
| [SMARD.de](https://www.smard.de) (Bundesnetzagentur) | **Target** | Grid load (Netzlast), hourly, chart_data API (filter 410, region DE) |
| [Open-Meteo](https://open-meteo.com) | Feature | 2 m temperature, population-weighted over the six largest cities |
| [Energy-Charts](https://api.energy-charts.info) (Fraunhofer ISE) | Feature | Day-ahead price (DE-LU) and generation mix |

Everything is aligned to one hourly index in **Europe/Berlin** (DST handled by
converting from UTC), covering **2021-01 to present** (~48k hours). Loading and
assembly live in [`src/data.py`](src/data.py).

## Method

1. **EDA** ([`notebooks/01_eda.ipynb`](notebooks/01_eda.ipynb)) — daily/weekly/yearly
   seasonality, the holiday effect, data quality, autocorrelation, and load–temperature
   structure.
2. **Baselines** — daily persistence (same hour, previous whole day) and **seasonal-naive**
   (same hour, previous week). Seasonal-naive is a genuinely strong baseline for load.
3. **Model** — LightGBM on lag, calendar, weather and (lagged) price/mix features. A
   **direct multi-horizon** setup: one model per horizon predicting `y(t+h)` from
   information available at `t`.
4. **Evaluation** ([`notebooks/02_modeling.ipynb`](notebooks/02_modeling.ipynb)) —
   a **rolling-origin backtest**, never a random split. LightGBM is retrained monthly on
   a rolling two-year window and scored on the following month; baselines are scored on
   exactly the same timestamps. Metrics: MAE, MAPE, RMSE per horizon, always relative to
   the baselines.

**Leakage discipline** is the point of the project and is enforced in code and tests:

- every feature at target time `τ = t+h` uses only data available at the origin `t`;
- lags and rolling statistics are shifted so the current value never enters its own window;
- weather is included as the temperature at the target hour — a **perfect-forecast
  proxy**, called out below as a limitation — while price and generation mix enter only
  as lags known at `t`;
- [`tests/test_pipeline.py`](tests/test_pipeline.py) asserts the lag shifts and the
  no-overlap property of the backtest folds.

## Results

Rolling-origin backtest over 2022–2026 (monthly retrain, two-year window). LightGBM
beats the strong seasonal-naive baseline at **every** horizon — by **82 % at 1 h** and
still **36 % at 48 h** ahead. At the 24 h horizon it forecasts hourly German load to
**2.7 % MAPE (≈1,400 MW)**.

| Horizon | LightGBM MAE | LightGBM MAPE | Seasonal-naive MAE | MAE reduction |
| ---: | ---: | ---: | ---: | ---: |
| 1 h  | 482 MW   | 0.92 % | 2,605 MW | **−81.5 %** |
| 6 h  | 1,229 MW | 2.34 % | 2,604 MW | −52.8 % |
| 12 h | 1,398 MW | 2.66 % | 2,602 MW | −46.3 % |
| 24 h | 1,395 MW | 2.67 % | 2,598 MW | −46.3 % |
| 48 h | 1,665 MW | 3.17 % | 2,595 MW | −35.8 % |

![Backtest MAE by horizon](reports/error_by_horizon.png)

The daily-persistence naive baseline is weaker still (7–12 % MAPE) and is omitted from
the table for brevity; it appears in the figure and the app. Full per-horizon numbers,
error analysis and feature importances are in
[`notebooks/02_modeling.ipynb`](notebooks/02_modeling.ipynb).

## Repo layout

```
src/            data.py · features.py · model.py · evaluate.py · config.py
notebooks/      01_eda.ipynb · 02_modeling.ipynb
app/            streamlit_app.py  (forecast vs. actual + metrics)
tests/          leakage & backtest-integrity tests
data/           raw/ + processed/ (git-ignored; rebuilt by src.data)
```

## How to run

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python -m src.data        # fetch + assemble data/processed/dataset.parquet
python -m src.evaluate    # rolling-origin backtest -> metrics + predictions
pytest                    # leakage / backtest-integrity tests

streamlit run app/streamlit_app.py   # the interactive artifact
```

## Limitations & next steps

- **Weather is a perfect-forecast proxy.** Training on observed temperature is
  optimistic; a production system would feed a numerical weather forecast, whose error
  would widen the load error at longer horizons.
- **Single price zone / national holidays.** Price uses the DE-LU day-ahead zone and the
  calendar uses national holidays only, ignoring Bundesland-specific days.
- **Point forecasts.** No predictive intervals yet — quantile/probabilistic forecasts
  are the natural next step.
- **Minimal tuning.** LightGBM runs on sensible defaults; the emphasis is the evaluation,
  not squeezing the last few percent.

---
*Part of [DS-Portfolio](../README.md).*
