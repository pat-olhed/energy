"""Live forecast for the next auctioned day, plus the growing live track record."""
import pandas as pd
import streamlit as st

import _shared as sh


def _read(fc: pd.DataFrame) -> str:
    """A short, factual read of the day's curve, generated from the numbers."""
    lg = fc["lightgbm"]
    cheap, dear = lg.idxmin(), lg.idxmax()
    parts = [
        f"Preisminimum gegen **{cheap:%H}:00 Uhr** ({sh.eur(lg.min())}), Maximum gegen "
        f"**{dear:%H}:00 Uhr** ({sh.eur(lg.max())})."
    ]
    if lg.min() < 0:
        neg = int((lg < 0).sum())
        parts.append(
            f"In {neg} Stunden liegt der Preis im **negativen** Bereich: hohe "
            "Erneuerbaren-Einspeisung bei niedriger Last drückt ihn unter null — relevant "
            "für Speicher und flexible Lasten, die dann Abnahmeprämien erzielen."
        )
    else:
        parts.append(
            "Typischer Tagesverlauf mit Morgen- und Abendspitze und einer Mittagssenke "
            "durch die Photovoltaik-Einspeisung."
        )
    return " ".join(parts)


def render():
    st.title("🔮 Prognose für den nächsten Tag")

    fc = sh.latest_forecast()
    if fc is None:
        st.info("Es liegt noch keine Prognose vor. Erzeugung lokal über `python scripts/make_forecast.py`.")
        return

    day = fc.index.normalize()[0]
    settled = fc["y_true"].notna().all()
    created = pd.Timestamp(fc["created_at"].iloc[0])
    st.caption(f"Automatisch erzeugt am {created:%d.%m.%Y um %H:%M} UTC · Großhandelspreis DE/LU")

    st.write(
        "Erstellung einer vollständigen 24-Stunden-Preisprognose zum Gate Closure — dem "
        "Zeitpunkt (12:00 Uhr am Vortag), zu dem die Day-Ahead-Auktion alle Stundenpreise "
        "des Folgetags fixiert. Verwendet werden ausschließlich zu diesem Zeitpunkt bekannte "
        "Größen, insbesondere SMARDs Vor-Gate-Prognosen für Last, Wind und PV. Die "
        "Vorhersage bleibt damit leckagefrei, ohne Rückgriff auf erst danach entstehende Daten."
    )

    if settled:
        mae = (fc["lightgbm"] - fc["y_true"]).abs().mean()
        st.success(
            f"**{sh.de_date(day)}** — Auktion geräumt, realisierte Preise liegen vor. "
            f"Mittlerer absoluter Fehler des Modells an diesem Tag: **{sh.eur(mae)}**."
        )
        st.caption(
            "Zur Prognose des Folgetags: Wichtigster Eingang ist die Residuallast-Prognose "
            "(Last minus Wind und PV). SMARD veröffentlicht sie für den Folgetag erst am "
            "Abend des Vortags — bis dahin ist der letzte vollständig prognostizierbare Tag "
            "der hier gezeigte. Mit Vorliegen der Residuallast-Prognose verschiebt sich die "
            "Vorhersage automatisch auf den Folgetag."
        )
    else:
        mm = sh.model_mae()
        band = f" Typische Abweichung laut Backtest: **±{sh.eur(mm)}**." if mm else ""
        st.info(
            f"**{sh.de_date(day)}** — Auktion noch offen; Überprüfung nach Vorliegen des "
            f"realisierten Preises.{band}"
        )

    lg = fc["lightgbm"]
    cheap, dear = lg.idxmin(), lg.idxmax()
    c1, c2, c3 = st.columns(3)
    c1.metric("Tagesmittel", sh.eur(lg.mean()))
    c2.metric(f"Minimum · {cheap:%H}:00", sh.eur(lg.min()))
    c3.metric(f"Maximum · {dear:%H}:00", sh.eur(lg.max()))

    chart = pd.DataFrame(index=fc.index)
    chart["Prognose (LightGBM)"] = fc["lightgbm"]
    chart["naive Baseline (Vortag)"] = fc["naive"]
    colors = [sh.COL_LGBM, sh.COL_NAIVE]
    if settled:
        chart["realisiert"] = fc["y_true"]
        colors = [sh.COL_LGBM, sh.COL_NAIVE, sh.COL_IST]
    chart.index = chart.index.strftime("%H:%M")
    st.line_chart(chart, height=360, color=colors, y_label="Preis (€/MWh)", x_label="Uhrzeit")

    st.markdown(_read(fc))

    st.divider()
    st.subheader("Prognosegüte im Zeitverlauf")
    hist = sh.forecast_history()
    verified = None if hist is None else hist.dropna(subset=["y_true"])
    per_day = None
    if verified is not None and not verified.empty:
        per_day = (
            (verified["lightgbm"] - verified["y_true"]).abs()
            .groupby(verified.index.normalize()).mean()
        )
    if per_day is None or len(per_day) < 2:
        st.caption(
            "Der Live-Track-Record baut sich mit jedem geräumten Tag auf. Die systematische "
            "Güte über mehrere Jahre zeigt der Backtest unter „Wie gut ist die Prognose?“."
        )
        return
    st.write(
        f"Mittlerer absoluter Fehler über die letzten **{len(per_day)}** live erzeugten und "
        f"inzwischen geräumten Tagesprognosen: **{sh.eur(per_day.mean())}**."
    )
    recent = per_day.tail(14).copy()
    recent.index = recent.index.strftime("%d.%m.")
    st.bar_chart(recent.rename("MAE"), height=220, color=sh.COL_LGBM, y_label="MAE (€/MWh)", x_label="Tag")
