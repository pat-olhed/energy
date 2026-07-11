"""'How good is it?' — the rolling-origin backtest translated for non-technical readers:
model vs. the 'same as yesterday' rule of thumb, the regime story, and the significance
result stated in plain words."""
import pandas as pd
import streamlit as st

import _shared as sh


def render():
    st.title("📊 Wie gut ist die Prognose?")

    preds = sh.backtest()
    m = sh.metrics()
    if preds is None or m is None:
        st.info("Keine Backtest-Ergebnisse gefunden. Zuerst `python -m src.evaluate` laufen lassen.")
        return
    m = m.set_index("model")

    st.write(
        "Woher weiß man, ob eine Preisprognose *gut* ist? Man vergleicht sie mit einer "
        "einfachen Daumenregel: **„Morgen wird es wie gestern.“** Dann lässt man beide über "
        "mehrere Jahre gegen die tatsächlich eingetretenen Preise antreten — und zwar "
        "ehrlich: immer nur mit dem Wissen, das am jeweiligen Tag verfügbar war "
        "(*rollierende Rückrechnung*, nie ein zufälliger Ausschnitt)."
    )

    mae_model = m.loc["lightgbm", "MAE"]
    mae_naive = m.loc["naive", "MAE"]
    improve = m.loc["lightgbm", "MAE_vs_naive_%"]
    c1, c2, c3 = st.columns(3)
    c1.metric("Ø-Abweichung des Modells", sh.eur(mae_model))
    c2.metric("Ø-Abweichung der Daumenregel", sh.eur(mae_naive))
    c3.metric("Weniger Fehler als die Daumenregel", sh.pct(improve))
    st.caption(
        "„Ø-Abweichung“ = wie weit die Prognose im Mittel neben dem echten Preis lag "
        "(Fachbegriff: MAE, in €/MWh). Kleiner ist besser."
    )

    st.subheader("Prognose gegen Wirklichkeit")
    st.write(
        "Wähle einen Zeitraum, um Modell (blau), Daumenregel (orange) und den tatsächlich "
        "eingetretenen Preis (grau) nebeneinander zu sehen. Auffällig: Bei ruhigem Verlauf "
        "liegen alle nah beieinander — den Unterschied macht das Modell an den "
        "schwierigen Tagen mit starken Ausschlägen."
    )
    min_day, max_day = preds.index.min().date(), preds.index.max().date()
    default_start = max(min_day, (preds.index.max() - pd.Timedelta(days=21)).date())
    start_day, end_day = st.slider(
        "Welchen Zeitraum möchtest du sehen?",
        min_value=min_day, max_value=max_day, value=(default_start, max_day),
        format="DD.MM.YYYY",
    )
    window = preds.loc[str(start_day):str(end_day), ["y_true", "lightgbm", "naive"]].rename(
        columns={"y_true": "Tatsächlich", "lightgbm": "Prognose (Modell)", "naive": "Daumenregel"}
    )
    st.line_chart(
        window[["Prognose (Modell)", "Daumenregel", "Tatsächlich"]],
        height=360, color=[sh.COL_LGBM, sh.COL_NAIVE, sh.COL_IST],
        y_label="Preis (€/MWh)",
    )

    st.subheader("Jedes Jahr getestet — auch im Krisenjahr")
    reg = sh.regime()
    if reg is not None:
        r = reg.rename(columns={"MAE_lightgbm": "Modell", "MAE_naive": "Daumenregel"})
        r["Jahr"] = r["year"].astype(int).astype(str)
        r = r.set_index("Jahr")
        st.bar_chart(
            r[["Modell", "Daumenregel"]], height=300,
            color=[sh.COL_LGBM, sh.COL_NAIVE], y_label="Ø-Abweichung (€/MWh)", stack=False,
        )
    st.write(
        "**2022 war das mit Abstand schwerste Jahr** — die Gaskrise trieb die Preise "
        "extrem und sprunghaft. Danach normalisieren sich Niveau und Fehler wieder. "
        "Entscheidend: Das Modell schlägt die Daumenregel **in jedem einzelnen Jahr**, "
        "nicht nur im Durchschnitt."
    )

    st.subheader("Ist der Vorsprung echt — oder Zufall?")
    st.write(
        "Ein besserer Durchschnitt allein könnte Glück sein. Ein statistischer Test "
        "(*Diebold-Mariano*) prüft das gezielt und berücksichtigt, dass "
        "aufeinanderfolgende Tage sich ähneln. Ergebnis über mehr als **1.600 Testtage**: "
        "Der Vorsprung des Modells ist **hochsignifikant** — die Wahrscheinlichkeit, dass "
        "er reiner Zufall ist, liegt praktisch bei null. Die technischen Details dazu "
        "stehen unter „Methodik“."
    )
