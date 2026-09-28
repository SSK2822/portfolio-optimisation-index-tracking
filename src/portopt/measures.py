"""Drawdown risk measures.

Drawdowns are measured on the cumulative sum of weekly returns (not the compounded
wealth path), as in the original code:

    P_j = r_1 + ... + r_j,    d_j = max_{i <= j} P_i - P_j.

Given the drawdown series d_1..d_k:

* Calmar uses the maximum drawdown,     max_j d_j
* Pain uses the average drawdown,       (1/k) sum_j d_j
* Martin uses the Ulcer index,          sqrt((1/k) sum_j d_j^2)
"""

from __future__ import annotations

import numpy as np

RATIOS = ("calmar", "martin", "pain")


def drawdowns(returns: np.ndarray) -> np.ndarray:
    """Drawdown series of a sequence of weekly returns."""
    path = np.cumsum(np.asarray(returns, dtype=float))
    return np.maximum.accumulate(path) - path


def max_drawdown(returns: np.ndarray) -> float:
    return float(drawdowns(returns).max())


def average_drawdown(returns: np.ndarray) -> float:
    return float(drawdowns(returns).mean())


def mean_squared_drawdown(returns: np.ndarray) -> float:
    return float(np.mean(drawdowns(returns) ** 2))


def ulcer_index(returns: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_drawdown(returns)))


MEASURES = {
    "calmar": max_drawdown,
    "pain": average_drawdown,
    "martin": ulcer_index,
}


def risk(ratio: str, returns: np.ndarray) -> float:
    return MEASURES[ratio](returns)
