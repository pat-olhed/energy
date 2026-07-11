"""Forecast quality: the rolling-origin backtest against established EPF baselines, the
regime robustness, the significance result, and the open levers (external features)."""
import pandas as pd
import streamlit as st

import _shared as sh


def render():
    st.title("📊 Wie gut ist die Prognose?")

    preds = sh.backtest()
    m = sh.metrics()
    if preds is None or m is None:
        st.info("Keine Backtest-Ergebnisse vorhanden. Erzeugung über `python -m src.evaluate`.")
        return
    m = m.set_index("model")

    st.write(
        "Bewertungsmaßstab ist die Prognosegüte gegenüber etablierten naiven Baselines der "
        "Strompreisprognose: der **Tagespersistenz** (Preis des Vortags), der "
        "**wochentagsabhängigen Referenz** nach Lago et al. (2021) und der **saisonalen "
        "Baseline** (Preis der Vorwoche). Diese Referenzen sind in der Literatur Standard — "
        "eine gelernte Prognose gilt erst dann als belastbar, wenn sie sie auf einem "
        "rollierenden Backtest schlägt (Zeitraum 2022–2026, jede Prognose ausschließlich mit "
        "dem zum Gate Closure verfügbaren Informationsstand)."
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("MAE LightGBM", sh.eur(m.loc["lightgbm", "MAE"]))
    c2.metric("MAE naive Baseline", sh.eur(m.loc["naive", "MAE"]))
    c3.metric("Fehlerreduktion", sh.pct(m.loc["lightgbm", "MAE_vs_naive_%"]))
    st.caption(
        "MAE = mittlerer absoluter Fehler in €/MWh (Abstand zwischen Prognose und "
        "realisiertem Preis). Referenz ist die stärkste naive Baseline — in diesem "
        "niveaudriftenden Markt die Tagespersistenz, die die wochentagsabhängige und die "
        "saisonale Baseline übertrifft."
    )

    st.subheader("Prognose und Realisierung")
    st.write(
        "Gegenüberstellung von Modell, naiver Baseline und realisiertem Preis über einen "
        "wählbaren Zeitraum. Bei ruhigem Verlauf liegen alle Reihen nah beieinander; der "
        "Modellvorteil entsteht an den volatilen Tagen mit ausgeprägten Ausschlägen."
    )
    min_day, max_day = preds.index.min().date(), preds.index.max().date()
    default_start = max(min_day, (preds.index.max() - pd.Timedelta(days=21)).date())
    start_day, end_day = st.slider(
        "Zeitraum", min_value=min_day, max_value=max_day,
        value=(default_start, max_day), format="DD.MM.YYYY",
    )
    window = preds.loc[str(start_day):str(end_day), ["y_true", "lightgbm", "naive"]].rename(
        columns={"y_true": "realisiert", "lightgbm": "Prognose (LightGBM)", "naive": "naive Baseline"}
    )
    st.line_chart(
        window[["Prognose (LightGBM)", "naive Baseline", "realisiert"]],
        height=360, color=[sh.COL_LGBM, sh.COL_NAIVE, sh.COL_IST], y_label="Preis (€/MWh)",
    )

    st.subheader("Robustheit über die Regime")
    reg = sh.regime()
    if reg is not None:
        r = reg.rename(columns={"MAE_lightgbm": "LightGBM", "MAE_naive": "naive Baseline"})
        r["Jahr"] = r["year"].astype(int).astype(str)
        r = r.set_index("Jahr")
        st.bar_chart(
            r[["LightGBM", "naive Baseline"]], height=300,
            color=[sh.COL_LGBM, sh.COL_NAIVE], y_label="MAE (€/MWh)", stack=False,
        )
    st.write(
        "Das Jahr 2022 (Gaskrise) ist mit Abstand das schwierigste — extreme Niveaus und "
        "sprunghafte Preise; danach normalisieren sich Niveau und Fehler. Der Modellvorteil "
        "besteht in **jedem** Kalenderjahr, nicht nur im Mittel."
    )

    st.subheader("Statistische Signifikanz")
    st.write(
        "Ein Diebold-Mariano-Test auf den täglichen Fehler-Differenzen (HAC-robust, "
        "Tagesaggregation nach Lago et al. 2021) bestätigt den Vorsprung: **DM ≈ −12,5, "
        "p ≈ 3·10⁻³⁴** über 1.621 Testtage. Der Abstand zur Tagespersistenz ist damit "
        "hochsignifikant und kein Stichprobenartefakt. Details unter „Methodik“."
    )

    st.subheader("Grenzen und Ansatzpunkte")
    st.write(
        "Der größte ungenutzte Hebel liegt in externen fundamentalen Treibern, die derzeit "
        "nur indirekt über Preis-Lags erfasst werden:"
    )
    st.markdown(
        "- **Gas- (TTF) und CO₂-Preise (EUA)** — Brennstoff-Floor der Merit Order, Auslöser "
        "des Niveausprungs 2022. Hürde ist der token-freie Datenzugang, nicht die Modellierung.\n"
        "- **Cross-Border-Flows und Kraftwerksausfälle (ENTSO-E)** — Angebotsseite, "
        "token-frei verfügbar.\n"
        "- **Wettergrößen (Open-Meteo)** — Temperatur (Heiz-/Kühllast) als ergänzende "
        "fundamentale Sicht."
    )
