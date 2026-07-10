# Deployment der Streamlit-App

Die App wird kostenlos auf der **Streamlit Community Cloud** gehostet, damit sie über einen
Link ohne lokales Setup erreichbar ist. Sie lädt zur Laufzeit nur die eingecheckten
Preis-Backtest-Ergebnisse (`data/processed/price_backtest_results.parquet`,
`price_backtest_metrics.csv` und `price_backtest_regime.csv`) und importiert lediglich
`pandas`, `streamlit` und `src.config` — kein Datenabruf, kein LightGBM zur Laufzeit.

## Voraussetzungen

- Öffentliches GitHub-Repo (hier `github.com/pat-olhed/energy`).
- Die drei Preis-Backtest-Artefakte sind eingecheckt (die `.gitignore` lässt genau diese
  Dateien zu). Ohne sie zeigt die App den Hinweis, den Backtest erst zu bauen.
- `requirements.txt` liegt im Repo-Wurzelverzeichnis.

## Schritte

1. Auf [share.streamlit.io](https://share.streamlit.io) mit dem GitHub-Konto anmelden.
2. **Create app → Deploy a public app from GitHub** wählen.
3. Felder setzen:
   - **Repository:** `pat-olhed/energy`
   - **Branch:** `main`
   - **Main file path:** `app/streamlit_app.py`
4. Unter **Advanced settings** die **Python-Version 3.12** wählen (passend zur lokalen
   Umgebung; die App selbst ist versionsunkritisch).
5. **Deploy** klicken. Der erste Build installiert `requirements.txt` und dauert ein paar
   Minuten.
6. Nach dem Start die App-URL (Form `https://<name>.streamlit.app`) kopieren und in der
   `README.md` als **Live-Demo** eintragen.

## Aktualisieren

Jeder Push auf `main` deployt automatisch neu. Ändern sich die Backtest-Ergebnisse
(`python -m src.evaluate`), die aktualisierten Artefakte einfach committen und pushen.

## Hinweis

Die App zeigt ausschließlich offene Marktdaten (SMARD Day-Ahead-Preis und -Prognosen); es
werden keine Secrets oder personenbezogenen Daten benötigt oder eingecheckt.
