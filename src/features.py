"""Feature engineering for load forecasting.

Rule: every feature at time t must use only information available at or before t.
Lags and rolling stats must be shifted so no future value leaks in.

TODO: implement calendar + lag + (weather) features.
"""
from __future__ import annotations

import pandas as pd


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Hour-of-day, day-of-week, month, is_weekend, German public holidays.

    TODO: use the `holidays` package (DE + relevant Bundesland).
    """
    raise NotImplementedError


def add_lag_features(df: pd.DataFrame, lags_hours: list[int]) -> pd.DataFrame:
    """Lagged load (e.g. 24h, 48h, 168h) and rolling means — all shifted, no leakage."""
    raise NotImplementedError
