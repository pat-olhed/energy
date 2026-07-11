"""Landing page: the live day-ahead forecast for the next auctioned day, in plain
language, plus the growing live track record."""
import pandas as pd
import streamlit as st

import _shared as sh


def _plain_language_read(fc: pd.DataFrame) -> str:
    """A sentence or two, generated from the numbers, that a non-technical reader gets."""
    lg = fc["lightgbm"]
    cheap, dear = lg.idxmin(), lg.idxmax()
    parts = [
        f"Am günstigsten wird Strom voraussichtlich gegen **{cheap:%H}:00 Uhr** "
        f"({sh.eur(lg.min())}), am teuersten gegen **{dear:%H}:00 Uhr** ({sh.eur(lg.max())})."
    ]
    if lg.min() < 0:
        neg = int((lg < 0).sum())
        parts.append(
            f"In {neg} Stunden rutscht der Preis sogar **unter null**: dann liefern Sonne "
            "und Wind mehr Strom, als gebraucht wird — Abnehmer bekommen kurzzeitig Geld "
            "dafür, dass sie verbrauchen. Genau solche Stunden sind für Speicher und "
            "flexible Verbraucher bares Geld wert."
        )
    else:
        parts.append(
            "Der typische Tagesverlauf: morgens und abends teuer (hohe Nachfrage, wenig "
            "Sonne), mittags günstiger, wenn die Photovoltaik einspeist."
        )
    return " ".join(parts)


def render():
    st.title("🔮 Strompreis für morgen")

    fc = sh.latest_forecast()
    if fc is None:
        st.info(
            "Es liegt noch keine aktuelle Prognose vor. Sie wird täglich automatisch "
            "erzeugt — lokal mit `python scripts/make_forecast.py`."
        )
        return

    day = fc.index.normalize()[0]
    settled = fc["y_true"].notna().all()
    created = pd.Timestamp(fc["created_at"].iloc[0])
    st.caption(
        f"Automatisch erzeugt am {created:%d.%m.%Y um %H:%M} UTC · "
        "Großhandelspreis Deutschland/Luxemburg"
    )

    st.write(
        "Was kostet eine Megawattstunde Strom an der Börse — Stunde für Stunde? "
        "Diese Prognose entsteht **automatisch jeden Tag**, genau in dem Moment, in dem "
        "die Strombörse die Preise für den nächsten Tag festlegt (das sogenannte *Gate "
        "Closure*, 12:00 Uhr am Vortag). Das Modell nutzt dabei nur Informationen, die zu "
        "diesem Zeitpunkt schon bekannt sind — vor allem die Wetter- und Verbrauchs­"
        "prognosen für den Zieltag. So bleibt die Vorhersage ehrlich: kein Blick in Daten, "
        "die es zum Prognosezeitpunkt noch gar nicht gab."
    )

    # honest status: is this a real forward forecast, or an already-settled verification?
    if settled:
        mae = (fc["lightgbm"] - fc["y_true"]).abs().mean()
        st.success(
            f"**Prognose für {sh.de_date(day)}** — die Börsenauktion ist bereits geräumt, "
            f"die tatsächlichen Preise liegen vor. Das Modell lag im Schnitt "
            f"**{sh.eur(mae)}** daneben."
        )
    else:
        mm = sh.model_mae()
        band = f" Typische Abweichung laut Langzeittest: **±{sh.eur(mm)}**." if mm else ""
        st.info(
            f"**Prognose für {sh.de_date(day)}** — die Auktion ist noch offen. Die "
            f"Vorhersage lässt sich überprüfen, sobald der tatsächliche Preis vorliegt."
            f"{band}"
        )

    lg = fc["lightgbm"]
    cheap, dear = lg.idxmin(), lg.idxmax()
    c1, c2, c3 = st.columns(3)
    c1.metric("Ø-Preis über den Tag", sh.eur(lg.mean()))
    c2.metric(f"Günstigste Stunde · {cheap:%H}:00", sh.eur(lg.min()))
    c3.metric(f"Teuerste Stunde · {dear:%H}:00", sh.eur(lg.max()))

    # the 24h curve — model vs. the simple rule of thumb, plus the actual once settled
    chart = pd.DataFrame(index=fc.index)
    chart["Prognose (Modell)"] = fc["lightgbm"]
    chart["Daumenregel: wie gestern"] = fc["naive"]
    colors = [sh.COL_LGBM, sh.COL_NAIVE]
    if settled:
        chart["Tatsächlich eingetreten"] = fc["y_true"]
        colors = [sh.COL_LGBM, sh.COL_NAIVE, sh.COL_IST]
    chart.index = chart.index.strftime("%H:%M")
    st.line_chart(chart, height=360, color=colors, y_label="Preis (€/MWh)", x_label="Uhrzeit")

    st.markdown(_plain_language_read(fc))

    # the growing live track record — the real proof, once auctions settle
    st.divider()
    st.subheader("Wie gut lagen die bisherigen Prognosen?")
    hist = sh.forecast_history()
    verified = None if hist is None else hist.dropna(subset=["y_true"])
    per_day = None
    if verified is not None and not verified.empty:
        per_day = (
            (verified["lightgbm"] - verified["y_true"]).abs()
            .groupby(verified.index.normalize()).mean()
        )
    # need at least two settled days before a track record says anything beyond the tile above
    if per_day is None or len(per_day) < 2:
        st.caption(
            "Der Live-Track-Record baut sich täglich auf: Sobald mehrere automatisch "
            "erzeugte Prognosen an der Börse geräumt sind, erscheint hier Prognose gegen "
            "Wirklichkeit über die Zeit. Wie gut das Modell grundsätzlich trifft, zeigt "
            "schon jetzt der Rückrechnungstest unter „Wie gut ist die Prognose?“."
        )
        return
    st.write(
        f"Über die letzten **{len(per_day)}** automatisch erzeugten und inzwischen "
        f"geräumten Tagesprognosen lag das Modell im Schnitt **{sh.eur(per_day.mean())}** "
        "neben dem tatsächlichen Preis."
    )
    recent = per_day.tail(14).copy()
    recent.index = recent.index.strftime("%d.%m.")
    st.bar_chart(
        recent.rename("Abweichung"), height=220, color=sh.COL_LGBM,
        y_label="Ø-Abweichung (€/MWh)", x_label="Tag",
    )
