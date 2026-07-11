"""Feature engineering for the day-ahead electricity price.

Rule: every feature for a delivery hour τ on day D must be known at gate closure
(12:00 Europe/Berlin on day D-1), when the day-ahead auction fixes all 24 prices at
once. Nothing published after gate closure may leak in — so the price history is shifted
by *whole days*, and the fundamentals are SMARD's own pre-gate forecasts.
"""
from __future__ import annotations

import holidays
import numpy as np
import pandas as pd

from . import config

# National German public holidays (state-specific days are intentionally ignored).
_DE_HOLIDAYS = holidays.Germany()


def _is_holiday(index: pd.DatetimeIndex) -> np.ndarray:
    return np.fromiter((d.date() in _DE_HOLIDAYS for d in index), dtype=int, count=len(index))


def _cyclical(values, period: int):
    radians = 2 * np.pi * np.asarray(values) / period
    return np.sin(radians), np.cos(radians)


def _build_features(df: pd.DataFrame, target: str) -> pd.DataFrame:
    """Assemble the full gate-closure feature matrix indexed by delivery hour τ.

    This is the shared core of the supervised builder and the single-day forecaster: it
    only *builds* features (no target join, no row dropping), so the exact same columns
    and leakage rules serve both training and prediction. Feature groups:

      * fundamentals at τ — SMARD's own day-ahead forecasts (residual load, load, wind,
        PV), published the morning of D-1 (before gate closure), so taking their value
        *at the delivery hour* is leakage-free (this is why we use forecast, not actual,
        series here);
      * price history ≤ D-1 — same-hour lags (D-1/D-2/D-7) and aggregates of the whole
        previous day's cleared curve (mean/min/max/std) plus a 7-day rolling daily mean,
        every one shifted by entire days so day D never enters;
      * calendar of τ (deterministic).
    """
    price = df[target]
    tau = df.index
    feat = pd.DataFrame(index=tau)

    # fundamentals: SMARD day-ahead forecasts for the delivery hour τ, known at gate
    # closure because they are forecasts published the morning of D-1.
    feat["resload_fc_MW"] = df["resload_fc_MW"]
    feat["load_fc_MW"] = df["load_fc_MW"]
    feat["wind_fc_MW"] = df["wind_on_fc_MW"] + df["wind_off_fc_MW"]
    feat["pv_fc_MW"] = df["pv_fc_MW"]

    # price history strictly <= gate closure: same-hour lags land on D-1/D-2/D-7, whose
    # full curves were cleared at earlier auctions (known by 12:00 on D-1).
    for lag in config.PRICE_LAGS:
        feat[f"price_lag_{lag}"] = price.shift(lag)

    # aggregates of the *previous day's* cleared curve. Group by calendar day and shift
    # by whole days so day D is fully excluded — the central correctness point of the
    # gate-closure framing (a single-hour shift would leak day D's early hours).
    day = tau.normalize()
    daily = price.groupby(day).agg(["mean", "min", "max", "std"])
    prev = daily.shift(1)  # the row for day D now holds day D-1's stats
    prev["roll7_mean"] = daily["mean"].shift(1).rolling(7).mean()  # 7 prior days, D excluded
    prev = prev.reindex(day)
    prev.index = tau
    feat["price_d1_mean"] = prev["mean"]
    feat["price_d1_min"] = prev["min"]
    feat["price_d1_max"] = prev["max"]
    feat["price_d1_std"] = prev["std"]
    feat["price_roll7_mean"] = prev["roll7_mean"]

    # calendar of the delivery hour τ (deterministic, never leaks)
    feat["hour"] = tau.hour
    feat["dayofweek"] = tau.dayofweek
    feat["month"] = tau.month
    feat["is_weekend"] = (tau.dayofweek >= 5).astype(int)
    feat["is_holiday"] = _is_holiday(tau)
    feat["hour_sin"], feat["hour_cos"] = _cyclical(tau.hour, 24)
    feat["dow_sin"], feat["dow_cos"] = _cyclical(tau.dayofweek, 7)
    return feat


def make_supervised_dayahead(df: pd.DataFrame, target: str = config.PRICE_TARGET):
    """Build (X, y) for the day-ahead price, indexed by delivery hour τ on day D.

    The day-ahead auction fixes all 24 prices of day D at once, at gate closure
    (12:00 Europe/Berlin on D-1), so every feature must be known by then (see
    `_build_features` for the feature groups and the whole-day shift that keeps this
    leakage-free). Rows with any missing value — target or feature, at the series edges
    — are dropped, leaving the aligned training pair.
    """
    feat = _build_features(df, target)
    price = df[target]
    data = feat.join(price.rename("__y__")).dropna()
    return data.drop(columns="__y__"), data["__y__"].rename(target)


def make_features_for_day(
    df: pd.DataFrame, day, target: str = config.PRICE_TARGET
) -> pd.DataFrame:
    """Leakage-free feature rows for the 24 delivery hours of a single target day D.

    The forecasting counterpart to `make_supervised_dayahead`: it returns X only, so it
    works for a *future* day whose price has not been auctioned yet (that whole-day
    target column is still empty). `df` must carry enough history before D for the price
    lags/aggregates (D-1/D-2/D-7) and SMARD's day-ahead forecast fundamentals for D
    itself. Raises if any feature is missing for D — that means the fundamentals for the
    day are not published yet, so no honest gate-closure forecast can be made.
    """
    day = pd.Timestamp(day)
    if day.tzinfo is None:
        day = day.tz_localize(config.TZ)
    day = day.normalize()

    feat = _build_features(df, target)
    rows = feat.loc[feat.index.normalize() == day]
    if rows.empty:
        raise ValueError(f"no rows for target day {day.date()} — is it in the dataset index?")
    if rows.isna().any().any():
        missing = rows.columns[rows.isna().any()].tolist()
        raise ValueError(
            f"missing features for {day.date()} ({missing}); the day-ahead fundamentals "
            "or price history for this day are not fully available yet."
        )
    return rows
