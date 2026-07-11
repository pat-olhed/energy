"""Shared helpers for the multipage app: number/date formatting in German and cached
loaders for the backtest and live-forecast artifacts.

Every page imports this; it also puts the repo root on the path so `src.config` (the one
source of truth for paths and the target column) is importable on Streamlit Cloud.
"""
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

APP = Path(__file__).resolve().parent
ROOT = APP.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src import config  # noqa: E402

# one consistent colour per series across every page: Ist dark, model blue, naive orange
COL_IST = "#3f3f3f"
COL_LGBM = "#2a78d6"
COL_NAIVE = "#eb6834"

_WEEKDAYS = ["Montag", "Dienstag", "Mittwoch", "Donnerstag", "Freitag", "Samstag", "Sonntag"]
_MONTHS = ["Januar", "Februar", "März", "April", "Mai", "Juni", "Juli", "August",
           "September", "Oktober", "November", "Dezember"]


def de_date(ts) -> str:
    """Full German weekday-and-date label, e.g. 'Samstag, 11. Juli 2026' (locale-free)."""
    ts = pd.Timestamp(ts)
    return f"{_WEEKDAYS[ts.weekday()]}, {ts.day}. {_MONTHS[ts.month - 1]} {ts.year}"


def eur(value: float, decimals: int = 1) -> str:
    """German-formatted euro-per-MWh string (thousands '.', decimal ',')."""
    s = f"{value:,.{decimals}f}".replace(",", "§").replace(".", ",").replace("§", ".")
    return s + " €/MWh"


def pct(value: float, decimals: int = 1) -> str:
    return f"{value:.{decimals}f}".replace(".", ",") + " %"


def thousands(value) -> str:
    return f"{int(value):,}".replace(",", ".")


@st.cache_data
def load_parquet(name: str):
    p = config.PROCESSED / name
    return pd.read_parquet(p).sort_index() if p.exists() else None


@st.cache_data
def load_csv(name: str):
    p = config.PROCESSED / name
    return pd.read_csv(p) if p.exists() else None


def backtest():
    return load_parquet("price_backtest_results.parquet")


def metrics():
    return load_csv("price_backtest_metrics.csv")


def regime():
    return load_csv("price_backtest_regime.csv")


def latest_forecast():
    return load_parquet("latest_forecast.parquet")


def forecast_history():
    return load_parquet("forecast_history.parquet")


def feature_importance():
    return load_csv("feature_importance.csv")


def model_mae():
    """LightGBM's backtest MAE — the honest error band for the daily reconstruction."""
    m = metrics()
    return None if m is None else m.set_index("model").loc["lightgbm", "MAE"]
