"""Generate the live day-ahead price forecast for the next auctioned day.

The reproducible, production counterpart to the backtest: fetch the current SMARD price
and day-ahead forecast fundamentals, train LightGBM on the trailing window exactly as
the backtest does, and predict all 24 hours of the next day whose fundamentals are
already published — the real gate-closure use case the backtest only simulates. Writes:

  * data/processed/latest_forecast.parquet  — the 24h forecast for the target day
  * data/processed/forecast_history.parquet — every run, appended, so the app can show
    'how did earlier forecasts do' once each auction settles (a live track record).

Accuracy claims stay with the backtest (MAE, Diebold-Mariano); this forward forecast has
no label yet, so the app frames it with the historical error band and verifies it a day
later against the realised price.

Run: python scripts/make_forecast.py   (also driven daily by .github/workflows/forecast.yml)
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import config, data, features, model  # noqa: E402

LATEST = config.PROCESSED / "latest_forecast.parquet"
HISTORY = config.PROCESSED / "forecast_history.parquet"


def current_frame() -> pd.DataFrame:
    """Fetch price + forecast fundamentals on a *union* index.

    `data.build_dataset` joins the fundamentals onto the price index, which drops the
    future day (its price is not auctioned yet). Here we keep every timestamp either
    series covers, so the target day survives with its fundamentals present and its
    price still empty.
    """
    price = data.fetch_smard_price()
    forecasts = data.fetch_smard_forecasts()
    idx = price.index.union(forecasts.index)
    return price.reindex(idx).join(forecasts).rename_axis("timestamp").sort_index()


def target_day(frame: pd.DataFrame) -> pd.Timestamp:
    """The most recent day that has a full set of forecast-fundamental hours.

    That is the next day we can honestly forecast at gate closure. `>= 23` rather than
    `== 24` so a clock-change day (23 hours) still qualifies; a half-fetched trailing day
    is filtered out.
    """
    fc_cols = list(config.SMARD_FORECAST_FILTERS)
    full = frame[fc_cols].dropna()
    hours_per_day = full.groupby(full.index.normalize()).size()
    complete = hours_per_day[hours_per_day >= 23].index
    if len(complete) == 0:
        raise SystemExit("no day has complete forecast fundamentals — cannot forecast.")
    return complete.max()


def make_forecast() -> tuple[pd.DataFrame, pd.Timestamp]:
    """Fit on the trailing window and predict the 24 hours of the target day."""
    frame = current_frame()
    day = target_day(frame)
    price = frame[config.PRICE_TARGET]

    # train exactly as the backtest does: same features and params, trailing 2-year
    # window of fully settled days (the future day carries no target, so it is excluded).
    X, y = features.make_supervised_dayahead(frame)
    X, y = X.iloc[-config.TRAIN_WINDOW_HOURS:], y.iloc[-config.TRAIN_WINDOW_HOURS:]
    fitted = model.train_lgbm(X, y)

    X_day = features.make_features_for_day(frame, day)
    out = pd.DataFrame(index=X_day.index)
    out["lightgbm"] = fitted.predict(X_day)
    out["naive"] = price.shift(24).reindex(out.index)   # 'yesterday', known at gate closure
    out["y_true"] = price.reindex(out.index)            # NaN until the auction settles
    out["resload_fc_MW"] = frame["resload_fc_MW"].reindex(out.index)
    ee = (frame["wind_on_fc_MW"] + frame["wind_off_fc_MW"] + frame["pv_fc_MW"]) / frame["load_fc_MW"]
    out["ee_share"] = ee.reindex(out.index).clip(lower=0, upper=1)
    out["created_at"] = pd.Timestamp.now(tz="UTC")
    out = out.rename_axis("timestamp")

    _save(out, day, price)
    return out, day


def _save(out: pd.DataFrame, day: pd.Timestamp, price: pd.Series) -> None:
    """Cache the latest forecast and append it to the rolling history (idempotent).

    On re-runs the target day's rows are replaced, and every past row's realised price is
    refreshed from the current price series — so once an auction settles, that day's
    forecast-vs-actual becomes visible without a second model run.
    """
    config.PROCESSED.mkdir(parents=True, exist_ok=True)
    out.to_parquet(LATEST)

    if HISTORY.exists():
        hist = pd.read_parquet(HISTORY)
        hist = hist[hist.index.normalize() != day]      # drop a prior run for this day
        combined = pd.concat([hist, out]).sort_index()
    else:
        combined = out.copy()
    combined = combined[~combined.index.duplicated(keep="last")]
    combined["y_true"] = price.reindex(combined.index)  # backfill settled prices
    combined.to_parquet(HISTORY)


def _fmt(v: float) -> str:
    return f"{v:6.2f}".replace(".", ",")


def main() -> None:
    out, day = make_forecast()
    lg, nv = out["lightgbm"], out["naive"]
    settled = out["y_true"].notna().all()
    print(f"Forecast for {day.date()} ({len(out)} hours) — generated {out['created_at'].iloc[0]:%Y-%m-%d %H:%M UTC}")
    print(f"  LightGBM: mean {_fmt(lg.mean())}  min {_fmt(lg.min())}  max {_fmt(lg.max())}  EUR/MWh")
    print(f"  naive   : mean {_fmt(nv.mean())} EUR/MWh (yesterday)")
    cheap, dear = lg.idxmin(), lg.idxmax()
    print(f"  cheapest hour {cheap:%H:%M} ({_fmt(lg.min())})   dearest hour {dear:%H:%M} ({_fmt(lg.max())})")
    if settled:
        mae = (out["lightgbm"] - out["y_true"]).abs().mean()
        print(f"  (already auctioned — realised MAE this day: {_fmt(mae)} EUR/MWh)")
    else:
        print("  (not auctioned yet — verifiable tomorrow against the realised price)")
    print(f"\nwrote {LATEST}\nwrote {HISTORY}")


if __name__ == "__main__":
    main()
