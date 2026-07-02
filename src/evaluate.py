"""Evaluation — the part that makes this a credible portfolio project.

Time-series evaluation MUST use a rolling-origin (expanding-window) backtest, not a
random split: train on the past, predict the next block, roll forward. Metrics are
reported per horizon and always against the baselines.
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


def mape(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def backtest(
    df: pd.DataFrame,
    horizons: list[int] = config.HORIZONS,
    initial_train_hours: int = config.INITIAL_TRAIN_HOURS,
    step_hours: int = config.BACKTEST_STEP_HOURS,
    train_window_hours: int | None = config.TRAIN_WINDOW_HOURS,
    save: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rolling-origin backtest of LightGBM vs. the baselines, per horizon.

    For each horizon we build the direct supervised set, retrain LightGBM every
    `step_hours` on the expanding history and score the next block. Baselines are
    evaluated on exactly the same target timestamps. Returns (metrics, predictions)
    and, if `save`, caches both under data/processed/.
    """
    from . import features, model  # local import keeps the module import-cycle free

    y_full = df[config.TARGET]
    metric_rows, pred_frames = [], []

    for horizon in horizons:
        X, y = features.make_supervised(df, horizon)
        naive = model.naive_forecast(y_full, horizon).reindex(y.index)
        seasonal = model.seasonal_naive_forecast(y_full).reindex(y.index)

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
                "horizon": horizon,
                "y_true": yt,
                "lightgbm": pd.concat(lgbm_pred),
                "naive": naive.reindex(yt.index),
                "seasonal_naive": seasonal.reindex(yt.index),
            }
        )
        pred_frames.append(preds)

        for name in ("lightgbm", "naive", "seasonal_naive"):
            valid = preds["y_true"].notna() & preds[name].notna()
            sub = preds.loc[valid]
            metric_rows.append(
                {
                    "horizon": horizon,
                    "model": name,
                    "MAE": mae(sub["y_true"], sub[name]),
                    "MAPE": mape(sub["y_true"], sub[name]),
                    "RMSE": rmse(sub["y_true"], sub[name]),
                }
            )

    metrics = pd.DataFrame(metric_rows)
    base_mae = metrics[metrics.model == "seasonal_naive"].set_index("horizon")["MAE"]
    metrics["MAE_vs_seasonal_%"] = metrics.apply(
        lambda r: 100 * (1 - r["MAE"] / base_mae[r["horizon"]]), axis=1
    ).round(1)

    predictions = pd.concat(pred_frames).rename_axis("timestamp")

    if save:
        config.PROCESSED.mkdir(parents=True, exist_ok=True)
        predictions.to_parquet(config.BACKTEST_PARQUET)
        metrics.to_csv(config.PROCESSED / "backtest_metrics.csv", index=False)

    return metrics, predictions


if __name__ == "__main__":
    frame = pd.read_parquet(config.DATASET_PARQUET)
    scores, _ = backtest(frame)
    cols = ["horizon", "model", "MAE", "MAPE", "RMSE", "MAE_vs_seasonal_%"]
    print(scores[cols].to_string(index=False))
