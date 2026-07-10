"""Generate the yearly-MAE report plot from the cached backtest regime table.

Reproducible companion to `python -m src.evaluate`: reads the cached regime table
(data/processed/price_backtest_regime.csv) and writes reports/price_mae_by_year.png —
LightGBM vs. the canonical naive baseline (daily persistence, 'yesterday') per calendar
year, the honest regime story a rolling-origin backtest tells.

Run: python scripts/make_mae_plot.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import config  # noqa: E402

# daily persistence ('yesterday') is the strongest naive baseline in this market, so it
# is the reference the headline reduction is reported against (see README / evaluate.py).
BASELINE_COL = "MAE_naive"
BASELINE_LABEL = "naiv (gestern)"
GREY = "#aeb4bf"
BLUE = "#2c6fb3"


def main() -> None:
    regime = pd.read_csv(config.PROCESSED / "price_backtest_regime.csv")
    years = regime["year"].astype(int).astype(str).to_numpy()
    x = np.arange(len(years))
    width = 0.4

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width / 2, regime[BASELINE_COL], width, label=BASELINE_LABEL, color=GREY)
    bars = ax.bar(x + width / 2, regime["MAE_lightgbm"], width, label="LightGBM", color=BLUE)

    # value labels on the LightGBM bars, German decimal comma
    for rect, val in zip(bars, regime["MAE_lightgbm"]):
        ax.annotate(
            f"{val:.1f}".replace(".", ","),
            (rect.get_x() + rect.get_width() / 2, rect.get_height()),
            textcoords="offset points",
            xytext=(0, 4),
            ha="center",
            va="bottom",
            color=BLUE,
            fontsize=9,
        )

    ax.set_title("Day-Ahead-Preis — MAE nach Jahr (rollierender Backtest)")
    ax.set_ylabel("MAE (€/MWh)")
    ax.set_xticks(x, years)
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()

    out = ROOT / "reports" / "price_mae_by_year.png"
    fig.savefig(out, dpi=110)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
