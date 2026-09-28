"""Convert the original data files into the CSV layout read by ``portopt.data``.

Usage:
    python scripts/prepare_data.py /path/to/original/data [--out data]

The source folder must contain the BCST-Library ``.mat`` files, the Updated
Library ``wk_price_*`` / ``b1_*.txt`` files and the ``rf_*.txt`` risk-free series.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.io import loadmat

BCST_FILES = {
    "DJIA": ("DowJones.mat", "rf_90_16.txt"),
    "NASDAQ100": ("NASDAQ100.mat", "rf_04_16a.txt"),
    "FTSE100": ("FTSE100.mat", "rf_02_16.txt"),
    "SP500": ("SP500.mat", "rf_04_16b.txt"),
}
UPDATED_FILES = {
    "HDAX": ("wk_price_HDAX", "b1_HDAX.txt"),
    "FTSE": ("wk_price_FTSE", "b1_FTSE.txt"),
    "HSCI": ("wk_price_HSCI", "b1_HSCI.txt"),
    "SP500U": ("wk_price_SP500", "b1_SP500.txt"),
}
UPDATED_RF = "rf_00_17.txt"


def save(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, np.atleast_1d(array), delimiter=",", fmt="%.12g")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[1] / "data")
    args = parser.parse_args()
    src, out = args.source, args.out

    for name, (mat, rf) in BCST_FILES.items():
        m = loadmat(src / mat)
        folder = out / "bcst" / name
        save(folder / "asset_returns.csv", m["Assets_Returns"])
        save(folder / "index_returns.csv", m["Index_Returns"].ravel())
        save(folder / "risk_free.csv", np.loadtxt(src / rf, delimiter=","))
        print(f"bcst/{name}: {m['Assets_Returns'].shape[1]} assets, {m['Assets_Returns'].shape[0]} weeks")

    rf = np.loadtxt(src / UPDATED_RF, delimiter=",")
    for name, (prices, index) in UPDATED_FILES.items():
        p = np.loadtxt(src / prices)
        folder = out / "updated" / name
        save(folder / "asset_prices.csv", p)
        save(folder / "index_prices.csv", np.loadtxt(src / index, delimiter=","))
        save(folder / "risk_free.csv", rf)
        print(f"updated/{name}: {p.shape[1]} assets, {p.shape[0]} weekly prices")


if __name__ == "__main__":
    main()
