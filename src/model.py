"""Models: baselines + a gradient-boosting forecaster.

The baselines are not optional — every learned model is reported relative to them.

TODO: implement.
"""
from __future__ import annotations

import pandas as pd


def naive_forecast(y: pd.Series, horizon_hours: int = 24) -> pd.Series:
    """ŷ(t) = y(t - horizon). The floor every model must beat. TODO."""
    raise NotImplementedError


def seasonal_naive_forecast(y: pd.Series, season_hours: int = 168) -> pd.Series:
    """ŷ(t) = y(t - one week). Strong baseline for load. TODO."""
    raise NotImplementedError


def train_lgbm(X_train, y_train, **params):
    """Fit a LightGBM regressor on engineered features. TODO."""
    raise NotImplementedError
