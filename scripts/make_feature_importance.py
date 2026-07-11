"""Extract the model's feature importances as a small artifact for the app.

Reproducible companion to the backtest: fit LightGBM once on the trailing window (the
same window the live forecast uses), read gain-based importances, map the raw feature
names to readable German labels and groups, and write a normalised CSV. The app renders
it as a bar chart, so the 'what the model weights' story rests on real numbers rather
than hard-coded percentages.

Run: python scripts/make_feature_importance.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import config, features, model  # noqa: E402

OUT = config.PROCESSED / "feature_importance.csv"

# raw feature name -> (readable German label, group)
_LABELS: dict[str, tuple[str, str]] = {
    "resload_fc_MW": ("Residuallast-Prognose", "Fundamentaldaten"),
    "load_fc_MW": ("Last-Prognose", "Fundamentaldaten"),
    "wind_fc_MW": ("Wind-Prognose", "Fundamentaldaten"),
    "pv_fc_MW": ("PV-Prognose", "Fundamentaldaten"),
    "price_lag_24": ("Preis Vortag (D-1)", "Preishistorie"),
    "price_lag_48": ("Preis D-2", "Preishistorie"),
    "price_lag_168": ("Preis Vorwoche (D-7)", "Preishistorie"),
    "price_d1_mean": ("Vortag Ø", "Preishistorie"),
    "price_d1_min": ("Vortag Minimum", "Preishistorie"),
    "price_d1_max": ("Vortag Maximum", "Preishistorie"),
    "price_d1_std": ("Vortag Streuung", "Preishistorie"),
    "price_roll7_mean": ("7-Tage-Mittel", "Preishistorie"),
    "hour": ("Stunde", "Kalender"),
    "dayofweek": ("Wochentag", "Kalender"),
    "month": ("Monat", "Kalender"),
    "is_weekend": ("Wochenende", "Kalender"),
    "is_holiday": ("Feiertag", "Kalender"),
    "hour_sin": ("Tageszeit (zyklisch)", "Kalender"),
    "hour_cos": ("Tageszeit (zyklisch)", "Kalender"),
    "dow_sin": ("Wochentag (zyklisch)", "Kalender"),
    "dow_cos": ("Wochentag (zyklisch)", "Kalender"),
}


def main() -> None:
    df = pd.read_parquet(config.DATASET_PARQUET)
    X, y = features.make_supervised_dayahead(df)
    X, y = X.iloc[-config.TRAIN_WINDOW_HOURS:], y.iloc[-config.TRAIN_WINDOW_HOURS:]
    fitted = model.train_lgbm(X, y)

    # gain (total loss reduction) is the informative importance, not raw split counts
    gain = fitted.booster_.feature_importance(importance_type="gain")
    imp = pd.DataFrame({"feature": X.columns, "gain": gain})
    imp["label"] = imp["feature"].map(lambda f: _LABELS.get(f, (f, "Sonstige"))[0])
    imp["group"] = imp["feature"].map(lambda f: _LABELS.get(f, (f, "Sonstige"))[1])

    # collapse the two-column cyclical encodings under one readable label, then to percent
    per_label = imp.groupby(["label", "group"], as_index=False)["gain"].sum()
    per_label["importance_pct"] = 100 * per_label["gain"] / per_label["gain"].sum()
    per_label = per_label.sort_values("importance_pct", ascending=False)

    config.PROCESSED.mkdir(parents=True, exist_ok=True)
    per_label[["label", "group", "importance_pct"]].to_csv(OUT, index=False)

    by_group = per_label.groupby("group")["importance_pct"].sum().sort_values(ascending=False)
    print(f"wrote {OUT}  ({len(per_label)} features)")
    print("\nby group (%):")
    print(by_group.round(1).to_string())
    print("\ntop features (%):")
    print(per_label.head(8)[["label", "importance_pct"]].to_string(index=False))


if __name__ == "__main__":
    main()
