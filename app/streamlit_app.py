"""Streamlit artifact: forecast vs. actual + metrics vs. baselines.

Run: streamlit run app/streamlit_app.py
Requires the cached backtest (data/processed/backtest_results.parquet), produced by
`python -m src.evaluate`.
"""
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import config  # noqa: E402

st.set_page_config(page_title="German Load Forecast", layout="wide")


@st.cache_data
def load_predictions() -> pd.DataFrame | None:
    if not config.BACKTEST_PARQUET.exists():
        return None
    return pd.read_parquet(config.BACKTEST_PARQUET)


@st.cache_data
def load_metrics() -> pd.DataFrame | None:
    path = config.PROCESSED / "backtest_metrics.csv"
    return pd.read_csv(path) if path.exists() else None


st.title("⚡ Short-Term Electricity Load Forecast — German Grid")
st.caption("SMARD / Energy-Charts / Open-Meteo · 1–48h ahead · LightGBM vs. seasonal-naive baseline")

preds = load_predictions()
metrics = load_metrics()

if preds is None or metrics is None:
    st.warning(
        "No backtest results found. Build the dataset and run the backtest first:\n\n"
        "```\npython -m src.data\npython -m src.evaluate\n```"
    )
    st.stop()

horizons = sorted(preds["horizon"].unique())

# --- sidebar controls ------------------------------------------------------
st.sidebar.header("Controls")
horizon = st.sidebar.selectbox("Forecast horizon (hours ahead)", horizons, index=len(horizons) - 1)

hz = preds[preds["horizon"] == horizon].drop(columns="horizon").sort_index()
min_day, max_day = hz.index.min().date(), hz.index.max().date()
default_start = max(min_day, (hz.index.max() - pd.Timedelta(days=14)).date())
start_day, end_day = st.sidebar.slider(
    "Date range",
    min_value=min_day,
    max_value=max_day,
    value=(default_start, max_day),
)

# --- headline metrics ------------------------------------------------------
row = metrics[(metrics.horizon == horizon)].set_index("model")
lgb_mae = row.loc["lightgbm", "MAE"]
sn_mae = row.loc["seasonal_naive", "MAE"]
c1, c2, c3 = st.columns(3)
c1.metric(f"LightGBM MAE @ {horizon}h", f"{lgb_mae:,.0f} MW", f"{row.loc['lightgbm', 'MAPE']:.2f}% MAPE")
c2.metric("Seasonal-naive MAE", f"{sn_mae:,.0f} MW", f"{row.loc['seasonal_naive', 'MAPE']:.2f}% MAPE")
c3.metric("Improvement vs. baseline", f"{row.loc['lightgbm', 'MAE_vs_seasonal_%']:.1f}%", "lower error")

# --- forecast vs actual ----------------------------------------------------
st.subheader(f"Forecast vs. actual — {horizon}h ahead")
window = hz.loc[str(start_day):str(end_day), ["y_true", "lightgbm", "seasonal_naive"]]
window = window.rename(columns={"y_true": "actual", "lightgbm": "LightGBM", "seasonal_naive": "seasonal-naive"})
st.line_chart(window, height=380)

# --- error curve over horizon ---------------------------------------------
st.subheader("Error by horizon")
curve = metrics.pivot(index="horizon", columns="model", values="MAE")[
    ["naive", "seasonal_naive", "lightgbm"]
].rename(columns={"seasonal_naive": "seasonal-naive"})
col_a, col_b = st.columns([2, 3])
col_a.line_chart(curve, height=320, y_label="MAE (MW)")

# --- full metrics table ----------------------------------------------------
table = metrics.copy()
table["MAE"] = table["MAE"].round(0)
table["RMSE"] = table["RMSE"].round(0)
table["MAPE"] = table["MAPE"].round(2)
col_b.dataframe(
    table.set_index(["horizon", "model"]).sort_index(),
    use_container_width=True,
)

st.caption(
    "Rolling-origin backtest: LightGBM retrained monthly on the expanding history; "
    "baselines scored on the same timestamps. Weather enters as a perfect-forecast proxy "
    "(see README)."
)
