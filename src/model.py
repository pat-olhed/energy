"""Models: baselines + a gradient-boosting forecaster.

The baselines are not optional — every learned model is reported relative to them.
Both baselines are expressed in *target time* τ (the hour being predicted), so they
line up directly with the supervised targets built in `features.make_supervised_dayahead`.
"""
from __future__ import annotations

import math

import lightgbm as lgb
import pandas as pd

from . import config


def naive_forecast(y: pd.Series, horizon_hours: int = 24) -> pd.Series:
    """Daily-persistence baseline, kept causal for multi-day horizons.

    In target time, ŷ(τ) = y(τ − 24·d) with d = ceil(h/24): the fewest whole days
    that keep the reference point at or before the forecast origin. For h ≤ 24 this is
    the classic 'same hour, yesterday'.
    """
    days = max(1, math.ceil(horizon_hours / 24))
    return y.shift(24 * days)


def seasonal_naive_forecast(y: pd.Series, season_hours: int = 168) -> pd.Series:
    """Weekly seasonal baseline: ŷ(τ) = y(τ − 168h), i.e. same hour last week.

    A strong baseline for the day-ahead price: a pure causal shift, so the reference is
    always in the past.
    """
    return y.shift(season_hours)


def lago_naive_forecast(y: pd.Series) -> pd.Series:
    """Weekday-aware naive baseline (Lago et al. 2021) — the canonical EPF benchmark.

    The standard reference in day-ahead price forecasting predicts the same hour of the
    most representative recent settled day: on Tue–Fri that is *yesterday* (shift 24h);
    on Mon/Sat/Sun — the days that break the weekly rhythm, where yesterday is a poor
    guide — it is the *same hour last week* (shift 168h). Combining the two shifts on a
    weekday switch beats either single shift alone, so it is the hardest fair reference
    a learned model must clear. Both shifts are whole days, so the reference is always a
    fully settled day-ahead price vector, known before gate closure.
    """
    # target-time weekday: Mon=0 … Sun=6; Tue–Fri (1–4) take yesterday, the rest last week
    use_yesterday = pd.Series(y.index.dayofweek, index=y.index).isin([1, 2, 3, 4])
    return y.shift(24).where(use_yesterday, y.shift(168))


def default_lgbm_params() -> dict:
    """Sensible, unfussy defaults. Tuning is not the point of this project."""
    return dict(
        n_estimators=400,
        learning_rate=0.05,
        num_leaves=64,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        min_child_samples=50,
        random_state=config.SEED,
        n_jobs=-1,
        verbose=-1,
    )


def train_lgbm(X_train, y_train, **params) -> lgb.LGBMRegressor:
    """Fit a LightGBM regressor on engineered features."""
    p = default_lgbm_params()
    p.update(params)
    model = lgb.LGBMRegressor(**p)
    model.fit(X_train, y_train)
    return model
