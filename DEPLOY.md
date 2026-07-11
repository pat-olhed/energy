# Deployment der Streamlit-App

Die App wird kostenlos auf der **Streamlit Community Cloud** gehostet, damit sie über einen
Link ohne lokales Setup erreichbar ist. Sie ist als **Multipage-App** aufgebaut
(`app/streamlit_app.py` + `app/views/`) und lädt zur Laufzeit nur eingecheckte Artefakte —
den Preis-Backtest (`data/processed/price_backtest_*`) sowie die täglich neu erzeugte
Modell-Rekonstruktion (`latest_forecast.parquet`, `forecast_history.parquet`). Sie importiert
lediglich `pandas`, `streamlit` und `src.config` — kein Datenabruf, kein LightGBM zur Laufzeit.

## Voraussetzungen

- Öffentliches GitHub-Repo (hier `github.com/pat-olhed/energy`).
- Die Backtest- und Rekonstruktions-Artefakte sind eingecheckt (die `.gitignore` lässt genau
  diese Dateien zu). Ohne sie zeigt die App den Hinweis, den Backtest bzw. die Rekonstruktion
  erst zu bauen.
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

Jeder Push auf `main` deployt automatisch neu.

Die **tägliche Rekonstruktion aktualisiert sich selbst**: Der Workflow `.github/workflows/forecast.yml`
läuft täglich am Abend (`cron: 0 20 * * *`, nach Gate Closure), erzeugt den Modelllauf für den
aktuellen Tag neu (`python scripts/make_forecast.py`) und committet ihn zurück — was den Redeploy
auslöst. **Hinweis:** das ist ein *Nachvollzug* gegen den bereits realisierten Preis, kein
Vor-Auktions-Forecast (SMARD liefert die D+1-Fundamentalprognosen erst abends — siehe README
„Grenzen"). Manuell über *Actions → Daily forecast → Run workflow*.

Ändern sich die Backtest-Ergebnisse (`python -m src.evaluate`), die aktualisierten Artefakte
einfach committen und pushen.

## Hinweis

Die App zeigt ausschließlich offene Marktdaten (SMARD Day-Ahead-Preis und -Prognosen); es
werden keine Secrets oder personenbezogenen Daten benötigt oder eingecheckt.
