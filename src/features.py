"""Feature engineering for load forecasting.

Rule: every feature used to predict target time τ = t + h must be known at the
forecast origin t. Nothing from the interval (t, τ] may leak in. Lags and rolling
stats are shifted accordingly.

Two building blocks (`add_calendar_features`, `add_lag_features`) operate on a
frame's own index and are handy in the EDA and the tests. `make_supervised` assembles
the actual (X, y) matrix for a direct h-ahead forecast, indexed by the target time.
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


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Hour-of-day, day-of-week, month, is_weekend, German holidays + cyclical encodings.

    Deterministic given the timestamp, so these never leak regardless of horizon.
    """
    out = df.copy()
    idx = out.index
    out["hour"] = idx.hour
    out["dayofweek"] = idx.dayofweek
    out["month"] = idx.month
    out["is_weekend"] = (idx.dayofweek >= 5).astype(int)
    out["is_holiday"] = _is_holiday(idx)
    out["hour_sin"], out["hour_cos"] = _cyclical(idx.hour, 24)
    out["dow_sin"], out["dow_cos"] = _cyclical(idx.dayofweek, 7)
    return out


def add_lag_features(
    df: pd.DataFrame,
    lags_hours: list[int] = config.LAGS,
    target: str = config.TARGET,
    windows: list[int] = config.ROLLING_WINDOWS,
) -> pd.DataFrame:
    """Lagged load and shifted rolling mean/std — all shifted so no future value leaks.

    Rolling stats are shifted by one hour so the current value never enters its own
    window.
    """
    out = df.copy()
    y = out[target]
    for lag in lags_hours:
        out[f"{target}_lag_{lag}"] = y.shift(lag)
    for window in windows:
        out[f"{target}_rmean_{window}"] = y.shift(1).rolling(window).mean()
        out[f"{target}_rstd_{window}"] = y.shift(1).rolling(window).std()
    return out


def make_supervised(df: pd.DataFrame, horizon: int, target: str = config.TARGET):
    """Build (X, y) for a direct h-ahead forecast, indexed by target time τ = t + h.

    Features are computed at the origin t and relabelled to τ, so every column is known
    by the origin:
      * load history — current value, lags and shifted rolling stats (known at t);
      * price / generation mix — 24h lags (known at t);
      * weather — temperature at the *target* hour τ, a documented perfect-forecast
        proxy (a production system would substitute a numerical weather forecast);
      * calendar of the predicted hour τ (deterministic).
    Rows with any missing value (series edges) are dropped.
    """
    origin = df.index
    y_series = df[target]
    feat = pd.DataFrame(index=origin)

    # load history known at the origin t
    feat["load_last"] = y_series
    for lag in config.LAGS:
        feat[f"load_lag_{lag}"] = y_series.shift(lag)
    for window in config.ROLLING_WINDOWS:
        feat[f"load_rmean_{window}"] = y_series.shift(1).rolling(window).mean()
        feat[f"load_rstd_{window}"] = y_series.shift(1).rolling(window).std()

    # exogenous drivers known at t (lagged one day)
    if "price_EUR_MWh" in df:
        feat["price_lag_24"] = df["price_EUR_MWh"].shift(24)
    for col in ("solar_MW", "wind_MW", "renewable_share"):
        if col in df:
            feat[f"{col}_lag_24"] = df[col].shift(24)

    # weather at the target hour τ (perfect-forecast proxy)
    if "temp_DE" in df:
        feat["temp_target"] = df["temp_DE"].shift(-horizon)

    # calendar of the predicted hour τ
    target_time = origin + pd.Timedelta(hours=horizon)
    feat["hour"] = target_time.hour
    feat["dayofweek"] = target_time.dayofweek
    feat["month"] = target_time.month
    feat["is_weekend"] = (target_time.dayofweek >= 5).astype(int)
    feat["is_holiday"] = _is_holiday(target_time)
    feat["hour_sin"], feat["hour_cos"] = _cyclical(target_time.hour, 24)
    feat["dow_sin"], feat["dow_cos"] = _cyclical(target_time.dayofweek, 7)

    # relabel from origin t to target time τ and attach the target
    feat.index = target_time
    y = pd.Series(y_series.shift(-horizon).to_numpy(), index=target_time, name=target)

    data = feat.join(y.rename("__y__")).dropna()
    return data.drop(columns="__y__"), data["__y__"].rename(target)


def make_supervised_dayahead(df: pd.DataFrame, target: str = config.PRICE_TARGET):
    """Build (X, y) for the day-ahead price, indexed by delivery hour τ on day D.

    The day-ahead auction fixes all 24 prices of day D at once, at gate closure
    (12:00 Europe/Berlin on D-1), so every feature must be known by then. Unlike the
    generic `make_supervised`, the price history is shifted by *whole days*: the
    reference always lands on D-1 or earlier, whose price curves were cleared at
    previous auctions and are known at gate closure. Feature groups:

      * fundamentals at τ — SMARD's own day-ahead forecasts (residual load, load, wind,
        PV), published the morning of D-1 (before gate closure), so taking their value
        *at the delivery hour* is leakage-free (this is why we use forecast, not actual,
        series here);
      * price history ≤ D-1 — same-hour lags (D-1/D-2/D-7) and aggregates of the whole
        previous day's cleared curve (mean/min/max/std) plus a 7-day rolling daily mean,
        every one shifted by entire days so day D never enters;
      * calendar of τ (deterministic).
    Rows with any missing value (series edges) are dropped.
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
    # by whole days so day D is fully excluded — the central correctness point versus the
    # load builder (which may shift by a single hour).
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

    data = feat.join(price.rename("__y__")).dropna()
    return data.drop(columns="__y__"), data["__y__"].rename(target)
