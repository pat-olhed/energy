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

st.set_page_config(page_title="Lastprognose Deutschland", layout="wide")


# --- German number formatting (thousands ".", decimal ",") -----------------
def mw(value: float) -> str:
    return f"{value:,.0f}".replace(",", ".") + " MW"


def pct(value: float, decimals: int = 2) -> str:
    return f"{value:.{decimals}f}".replace(".", ",") + " %"


@st.cache_data
def load_predictions() -> pd.DataFrame | None:
    if not config.BACKTEST_PARQUET.exists():
        return None
    return pd.read_parquet(config.BACKTEST_PARQUET)


@st.cache_data
def load_metrics() -> pd.DataFrame | None:
    path = config.PROCESSED / "backtest_metrics.csv"
    return pd.read_csv(path) if path.exists() else None


st.title("⚡ Kurzfrist-Lastprognose — deutsches Stromnetz")
st.caption("SMARD / Energy-Charts / Open-Meteo · 1–48 h voraus · LightGBM vs. saisonal-naive Baseline")

preds = load_predictions()
metrics = load_metrics()

if preds is None or metrics is None:
    st.warning(
        "Keine Backtest-Ergebnisse gefunden. Zuerst den Datensatz bauen und den Backtest laufen lassen:\n\n"
        "```\npython -m src.data\npython -m src.evaluate\n```"
    )
    st.stop()

horizons = sorted(preds["horizon"].unique())

# --- sidebar controls ------------------------------------------------------
st.sidebar.header("Steuerung")
horizon = st.sidebar.selectbox("Prognosehorizont (Stunden voraus)", horizons, index=len(horizons) - 1)

hz = preds[preds["horizon"] == horizon].drop(columns="horizon").sort_index()
min_day, max_day = hz.index.min().date(), hz.index.max().date()
default_start = max(min_day, (hz.index.max() - pd.Timedelta(days=14)).date())
start_day, end_day = st.sidebar.slider(
    "Zeitraum",
    min_value=min_day,
    max_value=max_day,
    value=(default_start, max_day),
)

# --- headline metrics ------------------------------------------------------
row = metrics[(metrics.horizon == horizon)].set_index("model")
lgb_mae = row.loc["lightgbm", "MAE"]
sn_mae = row.loc["seasonal_naive", "MAE"]
c1, c2, c3 = st.columns(3)
c1.metric(f"LightGBM MAE bei {horizon} h", mw(lgb_mae), f"{pct(row.loc['lightgbm', 'MAPE'])} MAPE")
c2.metric("Saisonal-naiv MAE", mw(sn_mae), f"{pct(row.loc['seasonal_naive', 'MAPE'])} MAPE")
c3.metric("Verbesserung ggü. Baseline", pct(row.loc["lightgbm", "MAE_vs_seasonal_%"], 1), "weniger Fehler")

# --- forecast vs actual ----------------------------------------------------
st.subheader(f"Prognose vs. Ist — {horizon} h voraus")
window = hz.loc[str(start_day):str(end_day), ["y_true", "lightgbm", "seasonal_naive"]]
window = window.rename(columns={"y_true": "Ist-Last", "lightgbm": "LightGBM", "seasonal_naive": "saisonal-naiv"})
st.line_chart(window, height=380)

# --- error curve over horizon ---------------------------------------------
st.subheader("Fehler nach Horizont")
curve = metrics.pivot(index="horizon", columns="model", values="MAE")[
    ["naive", "seasonal_naive", "lightgbm"]
].rename(columns={"naive": "naiv", "seasonal_naive": "saisonal-naiv"})
curve.index.name = "Horizont (h)"
col_a, col_b = st.columns([2, 3])
col_a.line_chart(curve, height=320, y_label="MAE (MW)")

# --- full metrics table ----------------------------------------------------
table = metrics.copy()
table["MAE"] = table["MAE"].round(0)
table["RMSE"] = table["RMSE"].round(0)
table["MAPE"] = table["MAPE"].round(2)
table["model"] = table["model"].map({"naive": "naiv", "seasonal_naive": "saisonal-naiv", "lightgbm": "LightGBM"})
table = table.rename(
    columns={"horizon": "Horizont", "model": "Modell", "MAE_vs_seasonal_%": "MAE-Reduktion (%)"}
)
col_b.dataframe(
    table.set_index(["Horizont", "Modell"]).sort_index(),
    width="stretch",
)

st.caption(
    "Rollierender Backtest: LightGBM monatlich auf der wachsenden Historie neu trainiert; "
    "Baselines auf denselben Zeitstempeln bewertet. Wetter geht als perfekter-Vorhersage-Proxy "
    "ein (siehe README)."
)
