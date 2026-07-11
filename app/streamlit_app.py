"""Streamlit entry point: a multipage day-ahead electricity-price app (DE/LU).

Four themed pages, from concrete to technical: the model replayed on the latest settled
day, how good the forecast is, what drives the price, and the methodology. Reads only
cached artifacts (backtest + daily reconstruction) — no data fetch, no model training at runtime.

Run: streamlit run app/streamlit_app.py
"""
import sys
from pathlib import Path

APP = Path(__file__).resolve().parent
ROOT = APP.parent
for _p in (str(APP), str(ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import streamlit as st  # noqa: E402

st.set_page_config(
    page_title="Day-Ahead-Strompreisprognose — DE/LU",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

from views import methodik, morgen, treiber, ueberblick, wie_gut  # noqa: E402

pages = [
    st.Page(ueberblick.render, title="Überblick", icon="🗺️", url_path="ueberblick", default=True),
    st.Page(morgen.render, title="Aktueller Tag", icon="📅", url_path="morgen"),
    st.Page(wie_gut.render, title="Wie gut ist die Prognose?", icon="📊", url_path="guete"),
    st.Page(treiber.render, title="Was treibt den Preis?", icon="⚙️", url_path="treiber"),
    st.Page(methodik.render, title="Methodik", icon="🔬", url_path="methodik"),
]
st.navigation(pages).run()
