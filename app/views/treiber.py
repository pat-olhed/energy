"""'What drives the price?' — the merit-order intuition, the renewables effect, and the
negative-price lens, all in plain language."""
import streamlit as st

import _shared as sh


def render():
    st.title("⚙️ Was treibt den Preis?")

    preds = sh.backtest()
    if preds is None:
        st.info("Keine Backtest-Ergebnisse gefunden. Zuerst `python -m src.evaluate` laufen lassen.")
        return

    st.write(
        "Der Börsen-Strompreis entsteht aus Angebot und Nachfrage. Kraftwerke werden nach "
        "Kosten sortiert eingesetzt — die günstigsten zuerst (Sonne und Wind, deren "
        "Brennstoff gratis ist), dann immer teurere (Kohle, zuletzt Gas). Den Preis setzt "
        "das **letzte, teuerste noch benötigte Kraftwerk** (die *Merit Order*). Die "
        "entscheidende Größe ist deshalb die **Residuallast**: der Verbrauch **minus** "
        "Wind und Sonne — also der Teil, den konventionelle Kraftwerke decken müssen."
    )

    st.subheader("Viel Wind und Sonne → niedriger Preis")
    st.write(
        "Jeder Punkt ist eine Lieferstunde aus der Rückrechnung. Klar zu sehen: Je höher "
        "die Residuallast (rechts), desto teurere Kraftwerke laufen und desto höher der "
        "Preis. Die Farbe zeigt den Anteil der Erneuerbaren — bei hohem Anteil (hell) "
        "fällt der Preis, teils **unter null**."
    )
    if "resload_fc_MW" in preds.columns:
        scatter = preds.dropna(subset=["resload_fc_MW", "y_true"])[
            ["resload_fc_MW", "y_true", "ee_share"]
        ]
        scatter = scatter.sample(min(len(scatter), 6000), random_state=42).rename(
            columns={
                "resload_fc_MW": "Residuallast-Prognose (MW)",
                "y_true": "Preis (€/MWh)",
                "ee_share": "Erneuerbaren-Anteil",
            }
        )
        st.scatter_chart(
            scatter, x="Residuallast-Prognose (MW)", y="Preis (€/MWh)",
            color="Erneuerbaren-Anteil", height=380,
        )
    st.caption(
        "Residuallast = prognostizierter Verbrauch minus prognostizierte Wind- und "
        "PV-Einspeisung. Negative Residuallast heißt: Erneuerbare allein liefern schon "
        "mehr, als gebraucht wird."
    )

    st.subheader("Was das Modell am stärksten gewichtet")
    st.write(
        "Wenig überraschend deckt sich die Modell-Ökonomie mit der Theorie: Mit Abstand am "
        "wichtigsten ist die **Residuallast-Prognose** (rund 56 % des Erklärungsgewichts), "
        "gefolgt von der **jüngsten Preisgeschichte** (rund 27 %) — sie trägt indirekt das "
        "Brennstoffkosten-Niveau (Gas, CO₂) mit, das an der Börse sonst nicht kostenlos "
        "verfügbar ist. Der Rest verteilt sich auf Wind, Sonne und den Kalender "
        "(Wochentag, Feiertag, Tageszeit)."
    )

    st.subheader("Negative Preise erkennen")
    p = preds.dropna(subset=["y_true", "lightgbm"])
    actual, flagged = p["y_true"] <= 0, p["lightgbm"] <= 0
    tp = int((actual & flagged).sum())
    fp = int((~actual & flagged).sum())
    fn = int((actual & ~flagged).sum())
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    st.write(
        "Negative Preise (Strom wird mit Zuzahlung abgenommen) sind für Speicher und "
        "flexible Verbraucher der spannendste Fall. Wie zuverlässig kündigt das Modell sie an?"
    )
    n1, n2, n3 = st.columns(3)
    n1.metric("Negativstunden im Test", sh.thousands(int(actual.sum())))
    n2.metric("Treffer, wenn Modell warnt", sh.pct(precision * 100))
    n3.metric("Erkannte Negativstunden", sh.pct(recall * 100))
    st.caption(
        f"Von allen Stunden, in denen das Modell einen negativen Preis vorhersagt, tritt "
        f"er in {sh.pct(precision * 100)} der Fälle wirklich ein; umgekehrt erkennt es "
        f"{sh.pct(recall * 100)} aller tatsächlichen Negativstunden im Voraus."
    )
