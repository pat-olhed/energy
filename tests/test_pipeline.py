"""Rigor tests: the leakage-free guarantees the whole project rests on.

These run on small synthetic series (no network, fast) and assert the two things a
time-series pipeline must get right: the gate-closure price features never see day D,
and the backtest folds never let the future into the past.
"""
import numpy as np
import pandas as pd

from src import config, evaluate, features, model


def _price_frame(days: int = 30, start: str = "2022-05-01") -> pd.DataFrame:
    """Synthetic hourly price + SMARD forecast columns for the day-ahead builder.

    May 2022 has no DST switch, so every day is a clean 24 hours. The price is a unique
    per-hour counter, which makes it trivial to see which source hour a lag pulled from.
    Forecast fundamentals are constants well outside the price range, so no feature can
    accidentally equal another.
    """
    n = days * 24
    idx = pd.date_range(start, periods=n, freq="h", tz=config.TZ)
    df = pd.DataFrame(index=idx)
    df[config.PRICE_TARGET] = np.arange(n, dtype=float)  # unique value per hour
    df["load_fc_MW"] = 88_888.0
    df["wind_on_fc_MW"] = 7_777.0
    df["wind_off_fc_MW"] = 3_333.0
    df["pv_fc_MW"] = 5_555.0
    df["resload_fc_MW"] = (
        df["load_fc_MW"] - df["wind_on_fc_MW"] - df["wind_off_fc_MW"] - df["pv_fc_MW"]
    )
    return df


def test_make_supervised_dayahead_lags_and_target():
    df = _price_frame()
    X, y = features.make_supervised_dayahead(df)

    tau = y.index[400]
    price = df[config.PRICE_TARGET]
    # same-hour lags map to D-1 / D-2 / D-7 exactly
    assert np.isclose(X.loc[tau, "price_lag_24"], price.loc[tau - pd.Timedelta(hours=24)])
    assert np.isclose(X.loc[tau, "price_lag_48"], price.loc[tau - pd.Timedelta(hours=48)])
    assert np.isclose(X.loc[tau, "price_lag_168"], price.loc[tau - pd.Timedelta(hours=168)])
    # target at τ is the price at τ
    assert np.isclose(y.loc[tau], price.loc[tau])
    # nothing missing, and no feature equals the target (a crude leak tripwire)
    assert not X.isna().any().any() and not y.isna().any()
    assert not X.eq(y, axis=0).any().any()


def test_make_supervised_dayahead_no_gate_closure_leak():
    df = _price_frame()
    sentinel = -1e6
    day_d = pd.Timestamp("2022-05-20", tz=config.TZ)
    df.loc[df.index.normalize() == day_d, config.PRICE_TARGET] = sentinel

    X, y = features.make_supervised_dayahead(df)
    price_cols = [c for c in X.columns if c.startswith("price_")]

    # delivery hours ON day D must not see any price from day D itself — that is the
    # auction outcome we predict, so no price feature may carry the sentinel.
    tau_d = y.index[y.index.normalize() == day_d]
    assert len(tau_d) == 24
    assert not (X.loc[tau_d, price_cols] == sentinel).any().any()

    # positive control: on day D+1 the same-hour lag_24 *must* pull day D's prices, so
    # the sentinel has to appear — proving the test is sharp, not trivially true.
    tau_d1 = y.index[y.index.normalize() == (day_d + pd.Timedelta(days=1))]
    assert (X.loc[tau_d1, "price_lag_24"] == sentinel).all()


def test_baselines_are_plain_shifts():
    idx = pd.date_range("2022-01-01", periods=500, freq="h", tz=config.TZ)
    y = pd.Series(np.arange(500, dtype=float), index=idx)
    # same hour last week / yesterday are pure causal shifts
    assert model.seasonal_naive_forecast(y, 168).iloc[200] == y.iloc[200 - 168]
    assert model.naive_forecast(y, 24).iloc[200] == y.iloc[200 - 24]
    # multi-day horizon stays causal: 48h ahead -> two whole days back
    assert model.naive_forecast(y, 48).iloc[200] == y.iloc[200 - 48]


def test_lago_naive_switches_on_weekday():
    # two full weeks starting Monday 2022-05-02 (Mon=0 … Sun=6 in dayofweek)
    idx = pd.date_range("2022-05-02", periods=14 * 24, freq="h", tz=config.TZ)
    y = pd.Series(np.arange(len(idx), dtype=float), index=idx)
    lago = model.lago_naive_forecast(y)

    # Tue–Fri pull yesterday (shift 24h)
    wed = pd.Timestamp("2022-05-11 10:00", tz=config.TZ)  # Wednesday, week 2
    assert lago.loc[wed] == y.loc[wed - pd.Timedelta(hours=24)]
    # Mon/Sat/Sun pull the same hour last week (shift 168h)
    for day in ("2022-05-09", "2022-05-14", "2022-05-15"):  # Mon / Sat / Sun, week 2
        tau = pd.Timestamp(f"{day} 10:00", tz=config.TZ)
        assert lago.loc[tau] == y.loc[tau - pd.Timedelta(hours=168)]


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
