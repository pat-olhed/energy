"""Evaluation — the part that makes this a credible portfolio project.

Time-series evaluation MUST use a rolling-origin backtest, not a random split: train on
the past, predict the next block, roll forward. The day-ahead price is scored against
causal baselines in EUR/MWh and broken down by yearly regime.
"""
from __future__ import annotations

from typing import Iterator

import numpy as np
import pandas as pd

from . import config


def rolling_origin_splits(
    index: pd.DatetimeIndex,
    initial_train_hours: int,
    horizon_hours: int,
    step_hours: int,
    train_window_hours: int | None = None,
) -> Iterator[tuple[pd.DatetimeIndex, pd.DatetimeIndex]]:
    """Yield (train_idx, test_idx) folds for a rolling-origin backtest.

    Test the `horizon_hours` block just after a cut point, then advance the cut by
    `step_hours`. Training is everything before the cut (expanding window) unless
    `train_window_hours` is given, in which case the last `train_window_hours` before
    the cut are used (a rolling window that bounds cost and adapts to regime shifts).
    Train and test never overlap in time. Folds are positional over the hourly `index`.
    """
    n = len(index)
    cut = initial_train_hours
    while cut + horizon_hours <= n:
        train_start = 0 if train_window_hours is None else max(0, cut - train_window_hours)
        yield index[train_start:cut], index[cut:cut + horizon_hours]
        cut += step_hours


def mae(y_true, y_pred) -> float:
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def backtest_price(
    df: pd.DataFrame,
    initial_train_hours: int = config.INITIAL_TRAIN_HOURS,
    step_hours: int = config.BACKTEST_STEP_HOURS,
    train_window_hours: int | None = config.TRAIN_WINDOW_HOURS,
    save: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Rolling-origin backtest of the day-ahead price: LightGBM vs. the baselines.

    One supervised set (not a horizon loop): the gate-closure features from
    `make_supervised_dayahead`. LightGBM is retrained every `step_hours` on a rolling
    2-year window and scores the next block; the baselines (yesterday / last-week same
    hour) are causal shifts valid for the day-ahead frame. Reports MAE and RMSE in
    EUR/MWh (no MAPE — prices cross zero), MAE relative to the seasonal baseline, a
    per-year regime table and a negative-price detection lens. Returns
    (metrics, predictions, regime, negatives) and caches the artefacts if `save`.
    """
    from . import features, model

    X, y = features.make_supervised_dayahead(df)
    price = df[config.PRICE_TARGET]
    naive = model.naive_forecast(price, 24).reindex(y.index)
    seasonal = model.seasonal_naive_forecast(price, 168).reindex(y.index)

    truth, lgbm_pred = [], []
    for train_idx, test_idx in rolling_origin_splits(
        y.index, initial_train_hours, step_hours, step_hours, train_window_hours
    ):
        fitted = model.train_lgbm(X.loc[train_idx], y.loc[train_idx])
        lgbm_pred.append(pd.Series(fitted.predict(X.loc[test_idx]), index=test_idx))
        truth.append(y.loc[test_idx])

    yt = pd.concat(truth)
    preds = pd.DataFrame(
        {
            "y_true": yt,
            "lightgbm": pd.concat(lgbm_pred),
            "naive": naive.reindex(yt.index),
            "seasonal_naive": seasonal.reindex(yt.index),
        }
    ).rename_axis("timestamp")

    # carry two per-hour context columns so the Streamlit app can draw the merit-order
    # view (price vs. residual-load forecast, shaded by renewable share) from this one
    # artefact — presentation only, never used in a metric.
    preds["resload_fc_MW"] = df["resload_fc_MW"].reindex(preds.index)
    ee = (df["wind_on_fc_MW"] + df["wind_off_fc_MW"] + df["pv_fc_MW"]) / df["load_fc_MW"]
    preds["ee_share"] = ee.reindex(preds.index).clip(lower=0, upper=1)

    metric_rows = []
    for name in ("lightgbm", "naive", "seasonal_naive"):
        valid = preds["y_true"].notna() & preds[name].notna()
        sub = preds.loc[valid]
        metric_rows.append(
            {
                "model": name,
                "MAE": mae(sub["y_true"], sub[name]),
                "RMSE": rmse(sub["y_true"], sub[name]),
            }
        )
    metrics = pd.DataFrame(metric_rows)
    base_mae = metrics.set_index("model").loc["seasonal_naive", "MAE"]
    metrics["MAE_vs_seasonal_%"] = (100 * (1 - metrics["MAE"] / base_mae)).round(1)

    regime = _price_regime_table(preds)
    negatives = _negative_price_scores(preds)

    if save:
        config.PROCESSED.mkdir(parents=True, exist_ok=True)
        preds.to_parquet(config.PROCESSED / "price_backtest_results.parquet")
        metrics.to_csv(config.PROCESSED / "price_backtest_metrics.csv", index=False)
        regime.to_csv(config.PROCESSED / "price_backtest_regime.csv", index=False)

    return metrics, preds, regime, negatives


def _price_regime_table(preds: pd.DataFrame) -> pd.DataFrame:
    """MAE per calendar year for each model — the honest regime story a rolling-origin
    backtest captures (2021 calm -> 2022 gas crisis -> 2023+ normalisation)."""
    p = preds.dropna(subset=["y_true"]).copy()
    p["year"] = p.index.year
    rows = []
    for year, g in p.groupby("year"):
        row = {"year": int(year), "n": len(g)}
        for name in ("lightgbm", "naive", "seasonal_naive"):
            row[f"MAE_{name}"] = round(mae(g["y_true"], g[name]), 2)
        rows.append(row)
    return pd.DataFrame(rows)


def _negative_price_scores(preds: pd.DataFrame, model_col: str = "lightgbm") -> dict:
    """Precision/recall of hours the model flags as negative-priced (price <= 0).

    A secondary, decision-relevant lens (storage / flexible load), not a second model.
    """
    p = preds.dropna(subset=["y_true", model_col])
    actual = p["y_true"] <= 0
    pred = p[model_col] <= 0
    tp = int((actual & pred).sum())
    fp = int((~actual & pred).sum())
    fn = int((actual & ~pred).sum())
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    return {
        "n_negative_hours": int(actual.sum()),
        "precision": round(precision, 3),
        "recall": round(recall, 3),
    }


if __name__ == "__main__":
    frame = pd.read_parquet(config.DATASET_PARQUET)
    metrics, _, regime, negatives = backtest_price(frame)
    print(metrics.to_string(index=False))
    print()
    print(regime.to_string(index=False))
    print("\nnegative-price lens:", negatives)
