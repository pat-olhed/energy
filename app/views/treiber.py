"""Price drivers: the merit-order relationship in the backtest data, the model's real
feature importances, and the negative-price detection lens with its planned extension."""
import altair as alt
import streamlit as st

import _shared as sh

_GROUP_COLORS = {"Fundamentaldaten": sh.COL_LGBM, "Preishistorie": "#1baf7a", "Kalender": "#9aa0a6"}


def render():
    st.title("⚙️ Was treibt den Preis?")

    preds = sh.backtest()
    if preds is None:
        st.info("Keine Backtest-Ergebnisse vorhanden. Erzeugung über `python -m src.evaluate`.")
        return

    st.subheader("Merit Order: die Residuallast als zentraler Treiber")
    st.write(
        "Der Börsenpreis ergibt sich aus dem Kraftwerkseinsatz in aufsteigender "
        "Grenzkosten-Reihenfolge; preissetzend ist das teuerste noch benötigte Kraftwerk. "
        "Maßgeblich ist die **Residuallast** — Last abzüglich Wind- und PV-Einspeisung. Das "
        "Streudiagramm zeigt diese Beziehung über alle Backtest-Stunden: Mit steigender "
        "Residuallast (nach rechts) werden teurere Kraftwerke abgerufen, der Preis steigt "
        "annähernd konvex. Bei hoher Erneuerbaren-Einspeisung (helle Punkte) und negativer "
        "Residuallast fällt der Preis unter null — die Punktwolke unten links."
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

    st.subheader("Was das Modell gewichtet")
    st.write(
        "Aufteilung der Vorhersage auf die Eingangsgrößen (Gain-basierte Feature-Wichtigkeit "
        "des trainierten Modells). Die Rangfolge bestätigt die ökonomische Erwartung: Die "
        "Residuallast-Prognose dominiert, gefolgt von der jüngsten Preishistorie, die das "
        "Brennstoffkosten- und Niveau-Regime (Gas, CO₂) indirekt kodiert; Kalendergrößen "
        "tragen wenig bei."
    )
    fi = sh.feature_importance()
    if fi is not None:
        top = fi.nlargest(11, "importance_pct")
        chart = (
            alt.Chart(top)
            .mark_bar()
            .encode(
                x=alt.X("importance_pct:Q", title="Wichtigkeit (% Gain)"),
                y=alt.Y("label:N", sort="-x", title=None, axis=alt.Axis(labelLimit=180)),
                color=alt.Color(
                    "group:N", title="Gruppe",
                    scale=alt.Scale(domain=list(_GROUP_COLORS), range=list(_GROUP_COLORS.values())),
                ),
                tooltip=[
                    alt.Tooltip("label:N", title="Merkmal"),
                    alt.Tooltip("group:N", title="Gruppe"),
                    alt.Tooltip("importance_pct:Q", title="% Gain", format=".1f"),
                ],
            )
            .properties(height=360)
        )
        st.altair_chart(chart, use_container_width=True)
        g = fi.groupby("group")["importance_pct"].sum()
        st.caption(
            f"Nach Gruppe: Fundamentaldaten {sh.pct(g.get('Fundamentaldaten', 0))}, "
            f"Preishistorie {sh.pct(g.get('Preishistorie', 0))}, "
            f"Kalender {sh.pct(g.get('Kalender', 0))}."
        )

    st.subheader("Negativpreise")
    p = preds.dropna(subset=["y_true", "lightgbm"])
    actual, flagged = p["y_true"] <= 0, p["lightgbm"] <= 0
    tp = int((actual & flagged).sum())
    fp = int((~actual & flagged).sum())
    fn = int((actual & ~flagged).sum())
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    st.write(
        "Negative Preise entstehen bei hoher Erneuerbaren-Einspeisung und niedriger Last und "
        "sind für Speicher und flexible Lasten besonders relevant. Derzeit werden sie als "
        "Nebenprodukt der Regression erkannt (Schwelle ≤ 0 €/MWh):"
    )
    n1, n2, n3 = st.columns(3)
    n1.metric("Negativstunden im Test", sh.thousands(int(actual.sum())))
    n2.metric("Precision", sh.pct(precision * 100))
    n3.metric("Recall", sh.pct(recall * 100))
    st.caption(
        f"Von den als negativ prognostizierten Stunden treffen {sh.pct(precision * 100)} zu; "
        f"{sh.pct(recall * 100)} der realisierten Negativstunden werden vorab erkannt. Eine "
        "gezielte Verbesserung — als eigenständiger Klassifikator mit wahrscheinlichkeits"
        "basiertem Schwellenwert und negativpreis-spezifischen Merkmalen — ist als "
        "Projekterweiterung vorgesehen."
    )
