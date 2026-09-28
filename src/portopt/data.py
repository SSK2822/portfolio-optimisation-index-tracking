"""Dataset definitions and loaders.

Two libraries are used:

* BCST-Library (Bruni et al., 2016): weekly asset and index returns.
* Updated Library (Leung et al., 2022a): weekly asset and index prices, converted
  to simple returns here.

Every dataset is stored under ``data/<library>/<name>/`` as plain CSV files.
Arrays are returned with assets on rows and weeks on columns, which matches the
layout of the original MATLAB code.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

DEFAULT_DATA_DIR = Path(__file__).resolve().parents[2] / "data"


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    library: str  # "bcst" or "updated"
    years: float  # length of the full sample in years, used for annualisation
    num_weeks: int | None = None  # BCST only: number of return weeks kept (even split)


DATASETS: dict[str, DatasetSpec] = {
    # BCST-Library. The raw files have an odd number of weeks; the last week is
    # dropped so the sample splits into two equal halves.
    "DJIA": DatasetSpec("DJIA", "bcst", 26.25, 1362),
    "NASDAQ100": DatasetSpec("NASDAQ100", "bcst", 11.5, 596),
    "FTSE100": DatasetSpec("FTSE100", "bcst", 13.83, 716),
    "SP500": DatasetSpec("SP500", "bcst", 11.5, 594),
    # Updated Library, 03/01/2000 - 29/11/2017.
    "HDAX": DatasetSpec("HDAX", "updated", 18.0),
    "FTSE": DatasetSpec("FTSE", "updated", 18.0),
    "HSCI": DatasetSpec("HSCI", "updated", 18.0),
    "SP500U": DatasetSpec("SP500U", "updated", 18.0),
}

BCST = ["DJIA", "NASDAQ100", "FTSE100", "SP500"]
UPDATED = ["HDAX", "FTSE", "HSCI", "SP500U"]

# Names used in tables, matching the report.
LABELS = {
    "DJIA": "DJIA",
    "NASDAQ100": "NASDAQ100",
    "FTSE100": "FTSE100",
    "SP500": "S&P500",
    "HDAX": "HDAX",
    "FTSE": "FTSE100",
    "HSCI": "HSCI",
    "SP500U": "S&P500",
}


@dataclass
class Market:
    spec: DatasetSpec
    asset_returns: np.ndarray  # (num_assets, num_weeks)
    index_returns: np.ndarray  # (num_index_weeks,), may be shorter than num_weeks
    risk_free: np.ndarray  # (>= num_weeks,), weekly risk-free rate

    @property
    def num_assets(self) -> int:
        return self.asset_returns.shape[0]

    @property
    def num_weeks(self) -> int:
        return self.asset_returns.shape[1]

    @property
    def annualisation(self) -> float:
        """sqrt(weeks per year), the factor the FYP applied to every ratio."""
        return float(np.sqrt(self.num_weeks / self.spec.years))


def simple_returns(prices: np.ndarray, axis: int = 0) -> np.ndarray:
    """Periodic simple returns p[t] / p[t-1] - 1 along ``axis``.

    Equivalent to MATLAB ``price2ret(prices, [], 'Periodic')``.
    """
    prices = np.asarray(prices, dtype=float)
    now = np.take(prices, np.arange(1, prices.shape[axis]), axis=axis)
    before = np.take(prices, np.arange(0, prices.shape[axis] - 1), axis=axis)
    return now / before - 1.0


def _read(path: Path) -> np.ndarray:
    return np.loadtxt(path, delimiter=",", ndmin=1)


def data_dir() -> Path:
    return Path(os.environ.get("PORTOPT_DATA_DIR", DEFAULT_DATA_DIR))


def load(name: str, root: Path | None = None) -> Market:
    spec = DATASETS[name]
    folder = (root or data_dir()) / spec.library / name
    risk_free = _read(folder / "risk_free.csv")

    if spec.library == "bcst":
        assets = np.loadtxt(folder / "asset_returns.csv", delimiter=",", ndmin=2)
        index = _read(folder / "index_returns.csv")
        assets = assets[: spec.num_weeks].T
        index = index[: spec.num_weeks]
    else:
        prices = np.loadtxt(folder / "asset_prices.csv", delimiter=",", ndmin=2)
        index_prices = _read(folder / "index_prices.csv")
        assets = simple_returns(prices, axis=0).T
        index = simple_returns(index_prices)

    if assets.shape[1] % 2:
        raise ValueError(f"{name}: expected an even number of weeks, got {assets.shape[1]}")
    if risk_free.shape[0] < assets.shape[1]:
        raise ValueError(f"{name}: risk-free series shorter than the return series")
    return Market(spec, assets, index, risk_free)
