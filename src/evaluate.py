"""Evaluation — the part that makes this a credible portfolio project.

Time-series evaluation MUST use a rolling-origin (expanding-window) backtest, not a
random split: train on [0, t), predict [t, t+h), roll forward. Report metrics per
horizon and always against the baselines.

TODO: implement.
"""
from __future__ import annotations

from typing import Iterator

import numpy as np
import pandas as pd


def rolling_origin_splits(
    index: pd.DatetimeIndex,
    initial_train_hours: int,
    horizon_hours: int,
    step_hours: int,
) -> Iterator[tuple[pd.DatetimeIndex, pd.DatetimeIndex]]:
    """Yield (train_idx, test_idx) folds for an expanding-window backtest. TODO."""
    raise NotImplementedError


def mae(y_true, y_pred) -> float:
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def mape(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))
