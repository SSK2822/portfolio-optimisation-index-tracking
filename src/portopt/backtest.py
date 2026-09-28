"""Rolling out-of-sample backtest.

The sample of T weeks is split in half. For every out-of-sample week t
(t = T/2 + 1, ..., T in 1-based numbering) the portfolio is fitted on all weeks
before t, held for week t, and refitted the following week. Weights for week t
therefore only use returns up to week t - 1.

No-trade rule: if every asset's in-sample mean return is below the current
risk-free rate, the portfolio stays in cash (all-zero weights) for that week.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .data import Market
from .ratios import ex_post_ratio
from .solvers import solve as default_solve


@dataclass
class BacktestResult:
    ratio: str
    weeks: np.ndarray  # 1-based out-of-sample week numbers
    weights: np.ndarray  # (num_assets, num_test_weeks)
    returns: np.ndarray  # weekly portfolio returns
    risk_free: np.ndarray  # weekly risk-free rate over the test period
    no_trade: np.ndarray  # bool per week
    solver_ok: np.ndarray  # bool per week
    objectives: np.ndarray = field(default=None)


def fit_week(market: Market, ratio: str, week: int, solve=default_solve):
    """Weights for 1-based out-of-sample week ``week`` using weeks 1..week-1."""
    train = market.asset_returns[:, : week - 1]
    rf = float(market.risk_free[week - 1])
    mu = train.mean(axis=1)
    if np.all(mu < rf):
        return None
    return solve(ratio, train, mu, rf)


def run(market: Market, ratio: str, solve=default_solve, progress=None) -> BacktestResult:
    T, n = market.num_weeks, market.num_assets
    weeks = np.arange(T // 2 + 1, T + 1)
    W = np.zeros((n, weeks.size))
    no_trade = np.zeros(weeks.size, bool)
    ok = np.ones(weeks.size, bool)
    obj = np.full(weeks.size, np.nan)
    for k, week in enumerate(weeks):
        sol = fit_week(market, ratio, week, solve)
        if sol is None:
            no_trade[k] = True
        else:
            W[:, k], ok[k], obj[k] = sol.weights, sol.success, sol.objective
        if progress:
            progress(k + 1, weeks.size)
    test = market.asset_returns[:, weeks - 1]
    returns = np.einsum("ij,ij->j", W, test)
    return BacktestResult(ratio, weeks, W, returns, market.risk_free[weeks - 1], no_trade, ok, obj)


def benchmark_returns(market: Market):
    """Weekly returns of the equally weighted portfolio and the market index over
    the test period, with the matching risk-free series for each.

    The index series can be shorter than the asset series (Updated S&P500 has one
    fewer index price); its test period is then truncated to the available weeks.
    """
    T = market.num_weeks
    weeks = np.arange(T // 2 + 1, T + 1)
    ew = market.asset_returns[:, weeks - 1].mean(axis=0)
    idx_weeks = weeks[weeks <= market.index_returns.size]
    mi = market.index_returns[idx_weeks - 1]
    return (ew, market.risk_free[weeks - 1]), (mi, market.risk_free[idx_weeks - 1])


def summary(market: Market, result: BacktestResult) -> dict:
    """Annualised ex-post ratios for NO (the optimised portfolio), EW and MI."""
    a = market.annualisation
    (ew, ew_rf), (mi, mi_rf) = benchmark_returns(market)
    return {
        "NO": a * ex_post_ratio(result.ratio, result.returns, result.risk_free),
        "EW": a * ex_post_ratio(result.ratio, ew, ew_rf),
        "MI": a * ex_post_ratio(result.ratio, mi, mi_rf),
    }
