"""Data loading for the energy-forecast project.

Source: SMARD.de (Bundesnetzagentur) — "Realisierter Stromverbrauch" (actual load).
Goal: turn raw downloads into a tidy hourly series indexed by timestamp (Europe/Berlin).

TODO: implement. Keep it reproducible — a fresh clone should be able to fetch + build
the processed frame from scratch.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"
PROCESSED = Path(__file__).resolve().parents[1] / "data" / "processed"


def load_raw_smard(path: str | Path) -> pd.DataFrame:
    """Read a raw SMARD load CSV. TODO: parse the German CSV format + timestamps."""
    raise NotImplementedError


def to_hourly_load(df: pd.DataFrame) -> pd.DataFrame:
    """Return a tidy frame [timestamp (Europe/Berlin, hourly), load_MW].

    TODO: resample 15-min → hourly if needed, handle DST, fill/flag gaps.
    """
    raise NotImplementedError


def build_dataset() -> pd.DataFrame:
    """End-to-end: raw files in data/raw/ → tidy hourly load, cached to data/processed/."""
    raise NotImplementedError
