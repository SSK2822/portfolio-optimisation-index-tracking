"""Ex-post performance ratios.

    ratio = mean weekly excess return / drawdown measure of the weekly returns

multiplied by sqrt(weeks per year) to annualise, as in the FYP.
"""

from __future__ import annotations

import numpy as np

from .measures import risk


def ex_post_ratio(ratio: str, returns: np.ndarray, risk_free: np.ndarray) -> float:
    """Un-annualised ex-post ratio of a weekly return series."""
    returns = np.asarray(returns, dtype=float)
    excess = returns - np.asarray(risk_free, dtype=float)[: returns.size]
    return float(excess.mean() / risk(ratio, returns))


def calmar_ratio(returns, risk_free):
    return ex_post_ratio("calmar", returns, risk_free)


def martin_ratio(returns, risk_free):
    return ex_post_ratio("martin", returns, risk_free)


def pain_ratio(returns, risk_free):
    return ex_post_ratio("pain", returns, risk_free)
