"""Models: baselines + a gradient-boosting forecaster.

The baselines are not optional — every learned model is reported relative to them.
Both baselines are expressed in *target time* τ (the hour being predicted), so they
line up directly with the supervised targets built in `features.make_supervised`.
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

    A strong baseline for electricity load, and valid for every horizon we forecast
    (all ≤ 48h < 168h, so the reference is always in the past).
    """
    return y.shift(season_hours)


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
