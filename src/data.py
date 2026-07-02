"""Data loading for the energy-forecast project.

Primary target: German grid load (Netzlast) from SMARD.de, the Bundesnetzagentur's
open market-data platform — hourly, no API token. Alongside the target we pull a few
exogenous drivers a real forecaster would plausibly have:

  * weather (Open-Meteo ERA5 reanalysis) — temperature dominates short-term demand;
  * day-ahead spot price and generation mix (Energy-Charts, Fraunhofer ISE).

Everything is aligned onto one hourly index in Europe/Berlin and is reproducible
from a clean clone: `python -m src.data` fetches and caches the lot.

Leakage note (see README): the weather series is reanalysis *actuals* standing in for
a perfect forecast, and the generation mix is consumed downstream only as a *lagged*
feature — its contemporaneous value is not known at forecast time.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from . import config

RAW = config.RAW
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


def _year_chunks(start: str, end: str):
    """Yield (start, end) date strings split at calendar-year boundaries."""
    cur, last = pd.Timestamp(start), pd.Timestamp(end)
    while cur <= last:
        chunk_end = min(pd.Timestamp(f"{cur.year}-12-31"), last)
        yield cur.strftime("%Y-%m-%d"), chunk_end.strftime("%Y-%m-%d")
        cur = pd.Timestamp(f"{cur.year + 1}-01-01")


# --- SMARD load ------------------------------------------------------------

def fetch_smard_series(
    filter_id: int = config.SMARD_LOAD_FILTER,
    region: str = config.SMARD_LOAD_REGION,
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


def load_raw_smard(path: str | Path) -> pd.DataFrame:
    """Parse a manually downloaded SMARD load CSV (a fallback to the API).

    SMARD's export uses ';' separators, ',' as decimal and '.' as thousands
    separator, with the interval start held in the first date/time columns. Returns a
    [value] frame indexed by an Europe/Berlin timestamp.
    """
    df = pd.read_csv(path, sep=";", decimal=",", thousands=".", na_values=["-", ""])
    cols = list(df.columns)
    ts = pd.to_datetime(
        df[cols[0]].astype(str) + " " + df[cols[1]].astype(str),
        dayfirst=True,
        format="mixed",
    ).dt.tz_localize(config.TZ, ambiguous="infer", nonexistent="shift_forward")

    value_col = next(
        (c for c in cols if "netzlast" in c.lower() or "load" in c.lower()),
        df.select_dtypes("number").columns[-1],
    )
    out = pd.DataFrame(
        {"value": pd.to_numeric(df[value_col], errors="coerce").to_numpy()},
        index=pd.DatetimeIndex(ts),
    ).rename_axis("timestamp")
    out = out[~out.index.duplicated(keep="first")].sort_index()
    return out


def to_hourly_load(df: pd.DataFrame) -> pd.DataFrame:
    """Tidy a raw [value] load frame into an hourly `load_MW` series.

    Guarantees a regular hourly index in Europe/Berlin, resamples any sub-hourly data
    to hourly means, and flags gaps (`load_imputed`) rather than silently filling large
    holes — only short gaps (<= 3h) are interpolated.
    """
    s = df["value"].sort_index()
    s = s[~s.index.duplicated(keep="first")]
    s = s.resample("1h").mean()

    out = s.to_frame(config.TARGET)
    out["load_imputed"] = out[config.TARGET].isna()
    out[config.TARGET] = out[config.TARGET].interpolate(limit=3, limit_area="inside")
    return out


# --- weather (Open-Meteo) --------------------------------------------------

def fetch_weather_openmeteo(
    start: str = config.DATA_START, end: str | None = None
) -> pd.DataFrame:
    """Population-weighted national 2m temperature from Open-Meteo's ERA5 archive.

    One request per city over the full range, combined into a weighted mean. Returns
    an hourly `temp_DE` (deg C) indexed in Europe/Berlin. Times are requested in UTC
    and converted, which sidesteps any DST ambiguity.
    """
    end = end or pd.Timestamp.now(tz=config.TZ).strftime("%Y-%m-%d")
    session = _session()

    series, weights = {}, {}
    for city in config.WEATHER_CITIES:
        params = {
            "latitude": city["lat"],
            "longitude": city["lon"],
            "start_date": start,
            "end_date": end,
            "hourly": "temperature_2m",
            "timezone": "UTC",
        }
        hourly = session.get(config.OPEN_METEO_ARCHIVE, params=params, timeout=60).json()["hourly"]
        idx = pd.to_datetime(hourly["time"], utc=True).tz_convert(config.TZ)
        series[city["name"]] = pd.Series(hourly["temperature_2m"], index=idx)
        weights[city["name"]] = city["weight"]

    wide = pd.DataFrame(series)
    w = pd.Series(weights)
    temp = (wide * w).sum(axis=1) / w.sum()
    return temp.rename("temp_DE").rename_axis("timestamp").to_frame()


# --- price & generation mix (Energy-Charts) --------------------------------

def fetch_price_energy_charts(
    start: str = config.DATA_START, end: str | None = None
) -> pd.DataFrame:
    """Hourly day-ahead spot price (EUR/MWh) for the DE-LU zone from Energy-Charts."""
    end = end or pd.Timestamp.now(tz=config.TZ).strftime("%Y-%m-%d")
    session = _session()

    frames = []
    for chunk_start, chunk_end in _year_chunks(start, end):
        params = {"bzn": config.PRICE_BIDDING_ZONE, "start": chunk_start, "end": chunk_end}
        j = session.get(f"{config.ENERGY_CHARTS_BASE}/price", params=params, timeout=60).json()
        idx = pd.to_datetime(j["unix_seconds"], unit="s", utc=True).tz_convert(config.TZ)
        frames.append(pd.Series(j["price"], index=idx, name="price_EUR_MWh"))

    s = pd.concat(frames)
    s = s[~s.index.duplicated(keep="first")].sort_index()
    s = s.resample("1h").mean()  # normalise hourly / quarter-hourly to hourly
    return s.rename_axis("timestamp").to_frame()


def fetch_generation_mix(
    start: str = config.DATA_START, end: str | None = None
) -> pd.DataFrame:
    """Compact hourly generation-mix features from Energy-Charts `public_power`.

    Returns `solar_MW`, `wind_MW` and `renewable_share` (0-1). Used downstream only as
    lagged features, since the contemporaneous mix isn't known at forecast time.
    """
    end = end or pd.Timestamp.now(tz=config.TZ).strftime("%Y-%m-%d")
    session = _session()

    frames = []
    for chunk_start, chunk_end in _year_chunks(start, end):
        params = {"country": config.GENERATION_COUNTRY, "start": chunk_start, "end": chunk_end}
        j = session.get(f"{config.ENERGY_CHARTS_BASE}/public_power", params=params, timeout=120).json()
        idx = pd.to_datetime(j["unix_seconds"], unit="s", utc=True).tz_convert(config.TZ)
        prod = {p["name"]: pd.Series(p["data"], index=idx) for p in j["production_types"]}
        frames.append(pd.DataFrame(prod))

    wide = pd.concat(frames)
    wide = wide[~wide.index.duplicated(keep="first")].sort_index().resample("1h").mean()

    def pick(name: str) -> pd.Series:
        for c in wide.columns:
            if c.strip().lower() == name.lower():
                return wide[c]
        return pd.Series(np.nan, index=wide.index)

    solar = pick("Solar")
    wind = pick("Wind onshore").fillna(0) + pick("Wind offshore").fillna(0)
    share = pick("Renewable share of load")
    if share.notna().any():
        share = (share / 100.0).clip(0, 1)
    return pd.DataFrame(
        {"solar_MW": solar, "wind_MW": wind, "renewable_share": share}
    ).rename_axis("timestamp")


# --- assembly --------------------------------------------------------------

def build_dataset(save: bool = True) -> pd.DataFrame:
    """Fetch load + exogenous drivers, align on one hourly index, cache to parquet."""
    PROCESSED.mkdir(parents=True, exist_ok=True)

    load = to_hourly_load(fetch_smard_series())
    weather = fetch_weather_openmeteo()
    price = fetch_price_energy_charts()
    mix = fetch_generation_mix()

    df = load.join([weather, price, mix], how="left")
    df = df.loc[df[config.TARGET].notna().idxmax():]  # trim leading all-NaN load

    if save:
        load.to_parquet(config.LOAD_PARQUET)
        df.to_parquet(config.DATASET_PARQUET)
        print(
            f"Saved {len(df):,} hourly rows "
            f"({df.index.min()} -> {df.index.max()}) to {config.DATASET_PARQUET}"
        )
    return df


if __name__ == "__main__":
    build_dataset()
