"""Data loading for the day-ahead electricity-price forecast (DE/LU).

Target: the German day-ahead wholesale price from SMARD.de, the Bundesnetzagentur's
open market-data platform — hourly, no API token. Alongside the target we pull SMARD's
own day-ahead forecast fundamentals (forecast load, wind onshore/offshore, PV, and the
directly delivered residual-load forecast). Because these are forecasts published the
morning of D-1 — before the 12:00 gate closure — they are leakage-free inputs for
predicting the next day's prices.

Everything is aligned onto one hourly index in Europe/Berlin and is reproducible from a
clean clone: `python -m src.data` fetches and caches the lot.
"""
from __future__ import annotations

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from . import config

PROCESSED = config.PROCESSED


def _session(retries: int = 3, backoff: float = 0.5) -> requests.Session:
    """A requests session that retries transient errors with backoff."""
    s = requests.Session()
    retry = Retry(
        total=retries,
        backoff_factor=backoff,
        status_forcelist=(429, 500, 502, 503, 504),
    )
    s.mount("https://", HTTPAdapter(max_retries=retry))
    s.headers.update({"User-Agent": "energy-forecast/1.0 (portfolio project)"})
    return s


def fetch_smard_series(
    filter_id: int,
    region: str,
    resolution: str = config.SMARD_RESOLUTION,
    start: str = config.DATA_START,
) -> pd.DataFrame:
    """Fetch a full SMARD chart_data series as a [value] frame indexed by timestamp.

    SMARD serves data in weekly chunks. We read the index of week-start timestamps,
    keep the chunk containing `start` plus everything after it, and concatenate.
    Timestamps are epoch-ms in UTC and are converted to Europe/Berlin so DST
    transitions land on the correct wall-clock hour.
    """
    session = _session()
    index_url = f"{config.SMARD_BASE}/{filter_id}/{region}/index_{resolution}.json"
    stamps = sorted(session.get(index_url, timeout=30).json()["timestamps"])

    start_ms = int(pd.Timestamp(start, tz=config.TZ).timestamp() * 1000)
    kept = [t for t in stamps if t >= start_ms]
    earlier = [t for t in stamps if t < start_ms]
    if earlier:  # include the chunk straddling `start` so we keep its early hours
        kept = [earlier[-1]] + kept

    frames = []
    for ts in kept:
        url = f"{config.SMARD_BASE}/{filter_id}/{region}/{filter_id}_{region}_{resolution}_{ts}.json"
        series = session.get(url, timeout=30).json()["series"]
        frames.append(pd.DataFrame(series, columns=["ts_ms", "value"]))

    raw = pd.concat(frames, ignore_index=True).dropna(subset=["ts_ms"])
    idx = pd.DatetimeIndex(pd.to_datetime(raw["ts_ms"], unit="ms", utc=True)).tz_convert(config.TZ)
    # pass a plain array so pandas assigns positionally instead of aligning on RangeIndex
    out = (
        pd.DataFrame({"value": raw["value"].astype(float).to_numpy()}, index=idx)
        .rename_axis("timestamp")
        .sort_index()
    )
    out = out[~out.index.duplicated(keep="first")]
    return out.loc[out.index >= pd.Timestamp(start, tz=config.TZ)]


def fetch_smard_price(start: str = config.DATA_START) -> pd.DataFrame:
    """SMARD day-ahead wholesale price (EUR/MWh) for DE/LU — the headline target.

    Thin wrapper over `fetch_smard_series`, reusing all its machinery (weekly chunks,
    UTC->Berlin conversion). The value column is renamed to the configured target name
    so the rest of the pipeline is source-agnostic.
    """
    px = fetch_smard_series(
        config.SMARD_PRICE_FILTER, config.SMARD_PRICE_REGION, start=start
    )
    return px.rename(columns={"value": config.PRICE_TARGET})


def fetch_smard_forecasts(start: str = config.DATA_START) -> pd.DataFrame:
    """SMARD day-ahead forecast fundamentals, aligned on one hourly index.

    Each series (forecast load, wind onshore/offshore, PV, and SMARD's directly
    delivered residual-load forecast) is published before the 12:00 gate closure, so
    every column is a leakage-free input for predicting the next day's prices.
    """
    cols = []
    for name, filter_id in config.SMARD_FORECAST_FILTERS.items():
        s = fetch_smard_series(filter_id, config.SMARD_FORECAST_REGION, start=start)
        cols.append(s["value"].rename(name))
    return pd.concat(cols, axis=1).rename_axis("timestamp")


def _check_residual_load(df: pd.DataFrame) -> None:
    """Plausibility cross-check on the residual-load forecast (prints, never raises).

    SMARD ships residual load two ways: directly (`resload_fc_MW`, filter 4362) and via
    its components (load - wind - PV). They should agree closely; a large gap would mean
    a wrong filter id or a unit mismatch — a built-in sanity check on the fundamentals.
    """
    need = {"load_fc_MW", "wind_on_fc_MW", "wind_off_fc_MW", "pv_fc_MW", "resload_fc_MW"}
    if not need.issubset(df.columns):
        return
    derived = (
        df["load_fc_MW"] - df["wind_on_fc_MW"] - df["wind_off_fc_MW"] - df["pv_fc_MW"]
    )
    both = pd.concat([derived.rename("derived"), df["resload_fc_MW"]], axis=1).dropna()
    if both.empty:
        print("residual-load cross-check: no overlapping rows to compare")
        return
    corr = both["derived"].corr(both["resload_fc_MW"])
    mad = (both["derived"] - both["resload_fc_MW"]).abs().mean()
    print(
        f"residual-load cross-check: corr={corr:.4f}, mean abs diff={mad:,.0f} MW, "
        f"n={len(both):,}  (derived load-wind-PV vs. SMARD 4362)"
    )


def build_dataset(save: bool = True) -> pd.DataFrame:
    """Fetch the day-ahead price + SMARD forecast fundamentals, align hourly, cache."""
    PROCESSED.mkdir(parents=True, exist_ok=True)

    price = fetch_smard_price()          # SMARD day-ahead price -> target
    forecasts = fetch_smard_forecasts()  # day-ahead fundamentals, known before gate closure

    df = price.join(forecasts, how="left")
    df = df.loc[df[config.PRICE_TARGET].notna().idxmax():]  # trim leading all-NaN price

    _check_residual_load(df)  # sanity-check the forecast fundamentals

    if save:
        df.to_parquet(config.DATASET_PARQUET)
        print(
            f"Saved {len(df):,} hourly rows "
            f"({df.index.min()} -> {df.index.max()}) to {config.DATASET_PARQUET}"
        )
    return df


if __name__ == "__main__":
    build_dataset()
