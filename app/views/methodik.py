"""Methodology page — the technical layer for data-savvy readers: gate-closure framing,
leakage-freeness, rolling-origin backtest, baselines and the significance test. Jargon is
allowed here and clearly signposted as such."""
import streamlit as st

import _shared as sh

REPO = "https://github.com/pat-olhed/energy"


def render():
    st.title("🔬 Methodik")
    st.caption("Für technisch interessierte Leser · das Fundament, das die Zahlen ehrlich macht.")

    st.write(
        "Der Reiz dieses Projekts liegt nicht im Modell, sondern in der **sauberen "
        "Auswertung**. Vier Punkte, auf die es ankommt:"
    )

    st.subheader("1 · Gate-Closure-Framing & Leckagefreiheit")
    st.write(
        "Die Day-Ahead-Auktion fixiert alle 24 Stundenpreise eines Tages **gleichzeitig** "
        "zum Gate Closure (12:00 Uhr am Vortag). Jedes Merkmal muss zu diesem Zeitpunkt "
        "bekannt sein. Preis-Historie geht deshalb nur in **ganzen Tagen** ein (D-1/D-2/D-7 "
        "sowie Aggregate der Vortageskurve — der Liefertag selbst wird vollständig "
        "ausgeschlossen), und die Fundamentaldaten sind SMARDs eigene **Prognosen** für "
        "Last, Wind und PV, nicht deren Ist-Werte. Ein scharfer Sentinel-Test mit positivem "
        "Kontrollfall sichert das ab."
    )

    st.subheader("2 · Rollierender Backtest, keine Zufallsaufteilung")
    st.write(
        "Bewertet wird per *rolling origin*: trainieren auf der Vergangenheit, den "
        "Folgeblock vorhersagen, weiterrollen. LightGBM wird monatlich auf einem "
        "rollierenden Zwei-Jahres-Fenster neu trainiert; die Baselines werden auf **exakt "
        "denselben** Zeitstempeln gemessen. Ein zufälliger Train/Test-Split würde die "
        "Zukunft in die Vergangenheit lecken und die Güte überzeichnen."
    )

    st.subheader("3 · Baselines zuerst — und die härteste gewinnt")
    m = sh.metrics()
    if m is not None:
        m = m.set_index("model")
        st.markdown(
            f"- **Naiv (gestern):** {sh.eur(m.loc['naive', 'MAE'])} — die stärkste naive "
            "Referenz in diesem niveau-driftenden Markt\n"
            f"- **Lago (wochentagsabhängig):** {sh.eur(m.loc['lago_naive', 'MAE'])} — der "
            "Literaturstandard, hier aber schlechter als reine Tagespersistenz\n"
            f"- **Saisonal-naiv (letzte Woche):** {sh.eur(m.loc['seasonal_naive', 'MAE'])}\n"
            f"- **LightGBM:** {sh.eur(m.loc['lightgbm', 'MAE'])} "
            f"(**{sh.pct(m.loc['lightgbm', 'MAE_vs_naive_%'])}** weniger Fehler als „gestern“)"
        )
    st.write(
        "Jedes gelernte Modell wird **relativ zu** diesen Baselines berichtet — kein "
        "MAPE, weil Preise die Null kreuzen und negativ werden."
    )

    st.subheader("4 · Signifikanz: Diebold-Mariano-Test")
    st.write(
        "Der MAE-Vorsprung wird auf statistische Signifikanz geprüft — auf den "
        "**täglichen** Fehler-Differenzen (alle 24 Stundenpreise eines Tages sind *ein* "
        "Auktionsergebnis, der multivariate DM-Test nach Lago et al. 2021), mit "
        "**HAC-robustem** Standardfehler (Newey-West) gegen die Autokorrelation "
        "aufeinanderfolgender Tage. Ergebnis LightGBM vs. „gestern“: **DM ≈ −12,5, "
        "p ≈ 3·10⁻³⁴** über 1.621 Testtage — hochsignifikant. Ein echter "
        "Terminmarkt-Maßstab bleibt bewusst offen (die EEX-Settlement-Historie ist nicht "
        "frei verfügbar)."
    )

    st.subheader("5 · Backtest vs. Live — und eine offene Annahme")
    st.write(
        "Die belastbaren Zahlen oben stammen aus dem **Backtest**. Ein *echter* "
        "Live-Forecast vor der Auktion ist token-frei nicht möglich: SMARD stellt die "
        "vollständigen Fundamentalprognosen für den Folgetag erst am Abend des Vortags "
        "bereit — nach dem 12-Uhr-Gate-Closure, zu dem die Auktion die Preise längst "
        "fixiert hat. Die Seite „Aktueller Tag“ ist deshalb ein **Nachvollzug** gegen den "
        "bereits realisierten Preis, kein Blick in die Zukunft. Offen bleibt zudem die "
        "**Vintage-Annahme**: SMARD liefert stets die jüngste Fassung einer Prognosereihe; "
        "die Leckagefreiheit hält streng nur, wenn diese der zum Gate Closure bekannten "
        "ex-ante-Fassung entspricht. Ein echter Vorab-Forecast — etwa über die "
        "ENTSO-E-Transparenzdaten mit definierten Veröffentlichungszeiten — ist als "
        "Erweiterung möglich."
    )

    st.divider()
    st.write(
        "Der vollständige Bericht — Problem, Daten, Methode, Ergebnisse, Grenzen — steht "
        f"im README, der Code liegt offen auf GitHub."
    )
    c1, c2 = st.columns(2)
    c1.link_button("📄 README (voller Bericht)", f"{REPO}/blob/main/README.md", use_container_width=True)
    c2.link_button("💻 Quellcode auf GitHub", REPO, use_container_width=True)
