"""Overview / landing page: a concise project summary, a lean pipeline infographic and the
headline result — the entry point before the live forecast and the analysis pages."""
import streamlit as st

import _shared as sh

ACCENT = "#2a78d6"

_STAGES = [
    ("1", "Offene Marktdaten", ["SMARD: Day-Ahead-Preis", "+ Prognosen (Last/Wind/PV)"]),
    ("2", "Gate Closure", ["12:00 am Vortag —", "Informationsschnitt"]),
    ("3", "LightGBM", ["leckagefrei trainiert,", "rollierend validiert"]),
    ("4", "24-Stunden-Prognose", ["alle Stundenpreise", "des Folgetags"]),
]


def _pipeline_svg() -> str:
    w, x0, gap = 200, 8, 32
    boxes, arrows = [], []
    for i, (num, title, subs) in enumerate(_STAGES):
        x = x0 + i * (w + gap)
        cx = x + w / 2
        emphasis = "0.16" if num == "2" else "0.07"
        stroke_w = 2 if num == "2" else 1.2
        boxes.append(
            f'<rect x="{x}" y="48" width="{w}" height="104" rx="10" '
            f'fill="{ACCENT}" fill-opacity="{emphasis}" stroke="{ACCENT}" stroke-width="{stroke_w}"/>'
            f'<circle cx="{x + 26}" cy="74" r="13" fill="{ACCENT}"/>'
            f'<text x="{x + 26}" y="79" text-anchor="middle" font-size="14" font-weight="700" fill="#fff">{num}</text>'
            f'<text x="{cx}" y="100" text-anchor="middle" font-size="15" font-weight="600" fill="currentColor">{title}</text>'
            f'<text x="{cx}" y="122" text-anchor="middle" font-size="11.5" fill="currentColor" opacity="0.62">{subs[0]}</text>'
            f'<text x="{cx}" y="138" text-anchor="middle" font-size="11.5" fill="currentColor" opacity="0.62">{subs[1]}</text>'
        )
        if i < len(_STAGES) - 1:
            ax = x + w
            arrows.append(
                f'<line x1="{ax + 4}" y1="100" x2="{ax + gap - 9}" y2="100" '
                f'stroke="{ACCENT}" stroke-width="2" marker-end="url(#arw)"/>'
            )
    return (
        '<svg viewBox="0 0 928 176" width="100%" role="img" '
        'aria-label="Projekt-Pipeline: offene Marktdaten, Gate Closure, LightGBM, 24-Stunden-Prognose" '
        'style="max-width:820px;display:block;margin:0.5rem auto 0;">'
        f'<defs><marker id="arw" markerWidth="9" markerHeight="9" refX="6" refY="3" orient="auto">'
        f'<path d="M0,0 L6,3 L0,6 Z" fill="{ACCENT}"/></marker></defs>'
        + "".join(boxes) + "".join(arrows) + "</svg>"
    )


def render():
    st.title("⚡ Day-Ahead-Strompreisprognose")
    st.caption("Deutschland/Luxemburg · offene Daten · leckagefreie Evaluation")

    st.write(
        "Gegenstand ist die Prognose des deutschen Day-Ahead-Großhandelspreises — aller 24 "
        "Stundenpreise eines Tages, erstellt zum täglichen Gate Closure (12:00 Uhr am "
        "Vortag) aus ausschließlich zu diesem Zeitpunkt verfügbaren, offenen Daten. Der "
        "Schwerpunkt liegt nicht auf Modellkomplexität, sondern auf einer **leckagefreien "
        "Evaluation gegen etablierte Baselines**: Ein Modell ist erst dann interessant, "
        "wenn es diese Referenzen auf einem rollierenden Backtest belastbar schlägt."
    )

    st.markdown(_pipeline_svg(), unsafe_allow_html=True)

    st.subheader("Kernergebnis")
    m = sh.metrics()
    if m is not None:
        m = m.set_index("model")
        c1, c2, c3 = st.columns(3)
        c1.metric("MAE LightGBM", sh.eur(m.loc["lightgbm", "MAE"]))
        c2.metric("MAE naive Baseline", sh.eur(m.loc["naive", "MAE"]))
        c3.metric("Fehlerreduktion", sh.pct(m.loc["lightgbm", "MAE_vs_naive_%"]))
    st.write(
        "Über den rollierenden Backtest 2022–2026 reduziert das Gradient-Boosting-Modell "
        "den mittleren absoluten Fehler gegenüber der stärksten naiven Baseline "
        "(Tagespersistenz) deutlich — der Abstand ist per Diebold-Mariano-Test "
        "hochsignifikant und hält in jedem einzelnen Jahr, einschließlich des Krisenjahrs 2022."
    )

    st.subheader("Aufbau der Anwendung")
    st.markdown(
        "- **🔮 Morgen** — die automatisch täglich erzeugte Prognose für den nächsten Tag\n"
        "- **📊 Wie gut ist die Prognose?** — Prognosegüte, Baselines und Signifikanz\n"
        "- **⚙️ Was treibt den Preis?** — Merit Order, Feature-Wichtigkeit, Negativpreise\n"
        "- **🔬 Methodik** — Gate-Closure-Framing, Leckagefreiheit, Backtest, DM-Test"
    )
