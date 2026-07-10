"""Streamlit artifact: day-ahead electricity-price forecast vs. actual + metrics.

Run: streamlit run app/streamlit_app.py
Loads only the cached price backtest (data/processed/price_backtest_*), produced by
`python -m src.evaluate`. No data fetch, no LightGBM at runtime — just pandas + streamlit.
"""
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import config  # noqa: E402

st.set_page_config(page_title="Day-Ahead-Strompreisprognose — DE/LU", layout="wide")


# --- German number formatting (thousands ".", decimal ",") -----------------
def eur(value: float, decimals: int = 1) -> str:
    s = f"{value:,.{decimals}f}".replace(",", "§").replace(".", ",").replace("§", ".")
    return s + " €/MWh"


def pct(value: float, decimals: int = 1) -> str:
    return f"{value:.{decimals}f}".replace(".", ",") + " %"


def thousands(value: int) -> str:
    return f"{value:,}".replace(",", ".")


@st.cache_data
def load_csv(name: str) -> pd.DataFrame | None:
    path = config.PROCESSED / name
    return pd.read_csv(path) if path.exists() else None


@st.cache_data
def load_predictions() -> pd.DataFrame | None:
    path = config.PROCESSED / "price_backtest_results.parquet"
    return pd.read_parquet(path).sort_index() if path.exists() else None


st.title("⚡ Day-Ahead-Strompreisprognose — DE/LU")
st.caption(
    "SMARD Day-Ahead-Preis + Prognose-Fundamentaldaten · eine Prognose je Tag zum "
    "Gate Closure (12:00 am Vortag) für alle 24 Stunden · LightGBM vs. naive Baseline (Tagespersistenz)"
)

preds = load_predictions()
metrics = load_csv("price_backtest_metrics.csv")
regime = load_csv("price_backtest_regime.csv")

if preds is None or metrics is None:
    st.warning(
        "Keine Backtest-Ergebnisse gefunden. Zuerst den Datensatz bauen und den Backtest "
        "laufen lassen:\n\n```\npython -m src.data\npython -m src.evaluate\n```"
    )
    st.stop()

# --- headline metrics ------------------------------------------------------
m = metrics.set_index("model")
c1, c2, c3 = st.columns(3)
c1.metric("LightGBM MAE", eur(m.loc["lightgbm", "MAE"]))
c2.metric("Naiv (gestern) MAE", eur(m.loc["naive", "MAE"]))
c3.metric("Verbesserung ggü. Baseline", pct(m.loc["lightgbm", "MAE_vs_naive_%"]), "weniger Fehler")

# --- sidebar: time window --------------------------------------------------
st.sidebar.header("Steuerung")
min_day, max_day = preds.index.min().date(), preds.index.max().date()
default_start = max(min_day, (preds.index.max() - pd.Timedelta(days=21)).date())
start_day, end_day = st.sidebar.slider(
    "Zeitraum", min_value=min_day, max_value=max_day, value=(default_start, max_day)
)

# --- forecast vs actual ----------------------------------------------------
st.subheader("Prognose vs. Ist")
window = preds.loc[str(start_day):str(end_day), ["y_true", "lightgbm", "naive"]].rename(
    columns={"y_true": "Ist-Preis", "lightgbm": "LightGBM", "naive": "naiv (gestern)"}
)
st.line_chart(window, height=380, y_label="Preis (€/MWh)")

# --- merit-order scatter (the signature plot) ------------------------------
if "resload_fc_MW" in preds.columns:
    st.subheader("Merit-Order — Preis vs. Residuallast-Prognose")
    st.caption(
        "Jeder Punkt eine Lieferstunde: steigende Residuallast (Last minus Wind/PV) ruft "
        "teurere Kraftwerke ab → höherer Preis. Farbe = EE-Anteil der Prognose."
    )
    scatter = preds.dropna(subset=["resload_fc_MW", "y_true"])[["resload_fc_MW", "y_true", "ee_share"]]
    scatter = scatter.sample(min(len(scatter), 6000), random_state=config.SEED).rename(
        columns={
            "resload_fc_MW": "Residuallast-Prognose (MW)",
            "y_true": "Preis (€/MWh)",
            "ee_share": "EE-Anteil",
        }
    )
    st.scatter_chart(
        scatter,
        x="Residuallast-Prognose (MW)",
        y="Preis (€/MWh)",
        color="EE-Anteil",
        height=380,
    )

# --- regime MAE + negative-price lens --------------------------------------
col_a, col_b = st.columns(2)

if regime is not None:
    col_a.subheader("MAE nach Jahr / Regime")
    reg = regime.rename(
        columns={"year": "Jahr", "MAE_lightgbm": "LightGBM", "MAE_naive": "naiv (gestern)"}
    )
    reg["Jahr"] = reg["Jahr"].astype(int).astype(str)  # categorical -> clean horizontal labels
    reg = reg.set_index("Jahr")
    col_a.bar_chart(reg[["LightGBM", "naiv (gestern)"]], height=300, y_label="MAE (€/MWh)", stack=False)
    col_a.caption("2022 = Gaskrise (schwerstes Jahr) → Normalisierung ab 2023.")

# negative-price lens recomputed from preds (no model needed)
p = preds.dropna(subset=["y_true", "lightgbm"])
actual, flagged = p["y_true"] <= 0, p["lightgbm"] <= 0
tp = int((actual & flagged).sum())
fp = int((~actual & flagged).sum())
fn = int((actual & ~flagged).sum())
precision = tp / (tp + fp) if (tp + fp) else float("nan")
recall = tp / (tp + fn) if (tp + fn) else float("nan")

col_b.subheader("Negativpreis-Erkennung (≤ 0 €/MWh)")
col_b.caption("Entscheidungsrelevant für Speicher / flexible Lasten.")
n1, n2, n3 = col_b.columns(3)
n1.metric("Negativstunden", thousands(int(actual.sum())))
n2.metric("Precision", pct(precision * 100))
n3.metric("Recall", pct(recall * 100))

st.caption(
    "Rollierender Backtest (rolling origin): LightGBM monatlich auf einem 2-Jahres-Fenster neu "
    "trainiert; Baselines auf denselben Zeitstempeln bewertet. Jedes Merkmal ist zum Gate "
    "Closure (12:00 am Vortag) bekannt — Details im README."
)
