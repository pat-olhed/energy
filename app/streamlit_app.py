"""Streamlit artifact: forecast vs. actual + metrics.

Run: streamlit run app/streamlit_app.py

TODO: wire up to the trained model / backtest results once they exist.
"""
import streamlit as st

st.set_page_config(page_title="German Load Forecast", layout="wide")

st.title("⚡ Short-Term Electricity Load Forecast — German Grid")
st.caption("SMARD / ENTSO-E · 1–48h ahead · LightGBM vs. seasonal-naive baseline")

st.info(
    "Scaffold. TODO: load the backtest results, plot forecast vs. actual, "
    "and show the MAE/MAPE table vs. baselines."
)
