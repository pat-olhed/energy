"""Rigor tests: the leakage-free guarantees the whole project rests on.

These run on small synthetic series (no network, fast) and assert the two things a
time-series pipeline must get right: features/lags are shifted correctly, and the
backtest folds never let the future into the past.
"""
import numpy as np
import pandas as pd

from src import config, evaluate, features, model


def _ramp(n: int = 500) -> pd.DataFrame:
    """A strictly increasing hourly series, so shifts are trivial to check by hand."""
    idx = pd.date_range("2022-01-01", periods=n, freq="h", tz=config.TZ)
    return pd.DataFrame({config.TARGET: np.arange(n, dtype=float)}, index=idx)


def test_add_lag_features_shift_and_rolling():
    df = _ramp()
    out = features.add_lag_features(df, lags_hours=[24], windows=[24])
    # lag-24 at row i is the target at row i-24
    assert out[f"{config.TARGET}_lag_24"].iloc[100] == df[config.TARGET].iloc[76]
    # rolling mean is shifted by one, so the current value is excluded from its window
    expected = df[config.TARGET].iloc[76:100].mean()
    assert np.isclose(out[f"{config.TARGET}_rmean_24"].iloc[100], expected)


def test_make_supervised_target_and_no_future_leak():
    df = _ramp()
    df["temp_DE"] = 10.0
    df["price_EUR_MWh"] = 50.0
    X, y = features.make_supervised(df, horizon=6)

    tau = y.index[100]
    origin = tau - pd.Timedelta(hours=6)
    # target at τ is the load at τ
    assert np.isclose(y.loc[tau], df[config.TARGET].loc[tau])
    # load_lag_24 at τ is the load 24h before the origin t = τ-6
    assert np.isclose(
        X.loc[tau, "load_lag_24"],
        df[config.TARGET].loc[origin - pd.Timedelta(hours=24)],
    )
    # every feature must be knowable at the origin: none may equal the target value
    assert not (X.eq(y, axis=0).any().any())
    assert not X.isna().any().any() and not y.isna().any()


def test_baselines_are_plain_shifts():
    y = _ramp()[config.TARGET]
    assert model.seasonal_naive_forecast(y, 168).iloc[200] == y.iloc[200 - 168]
    assert model.naive_forecast(y, 24).iloc[200] == y.iloc[200 - 24]
    # multi-day horizon stays causal: 48h ahead -> two whole days back
    assert model.naive_forecast(y, 48).iloc[200] == y.iloc[200 - 48]


def test_rolling_origin_splits_no_overlap_and_expanding():
    idx = pd.date_range("2022-01-01", periods=1000, freq="h", tz=config.TZ)
    folds = list(
        evaluate.rolling_origin_splits(
            idx, initial_train_hours=200, horizon_hours=48, step_hours=100
        )
    )
    assert len(folds) > 1
    prev_train_len = 0
    for train, test in folds:
        assert train[-1] < test[0]                      # no temporal overlap
        assert train.intersection(test).empty           # disjoint
        assert len(train) > prev_train_len              # expanding window
        prev_train_len = len(train)


def test_metrics_zero_on_perfect_prediction():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    assert evaluate.mae(y, y) == 0
    assert evaluate.rmse(y, y) == 0
    assert evaluate.mape(y, y) == 0
