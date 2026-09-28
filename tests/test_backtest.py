import numpy as np
import pytest

from portopt import backtest
from portopt.data import DatasetSpec, Market


def make_market(returns, rf=0.0005):
    n, T = returns.shape
    spec = DatasetSpec("TEST", "bcst", years=T / 52)
    return Market(spec, returns, returns.mean(axis=0), np.full(T, rf))


@pytest.fixture(scope="module")
def market():
    rng = np.random.default_rng(1)
    return make_market(rng.normal(0.002, 0.03, size=(6, 80)))


@pytest.mark.parametrize("ratio", ["calmar", "pain", "martin"])
def test_weights_valid(market, ratio):
    res = backtest.run(market, ratio)
    assert res.weights.shape == (6, 40)
    assert res.weeks[0] == 41 and res.weeks[-1] == 80
    sums = res.weights.sum(axis=0)
    traded = ~res.no_trade
    np.testing.assert_allclose(sums[traded], 1.0, atol=1e-6)
    assert np.all(res.weights[:, res.no_trade] == 0)
    assert np.all(res.weights >= -1e-9)


def test_no_look_ahead(market):
    """Changing returns from week t onwards must not change the weights for week t."""
    week = 60
    before = backtest.fit_week(market, "calmar", week).weights
    shocked = market.asset_returns.copy()
    shocked[:, week - 1 :] *= -5
    after = backtest.fit_week(make_market(shocked), "calmar", week).weights
    np.testing.assert_allclose(before, after)


def test_no_trade_rule():
    # Every asset loses money in-sample: stay in cash.
    returns = np.full((3, 20), -0.01)
    res = backtest.run(make_market(returns), "calmar")
    assert res.no_trade.all()
    assert np.all(res.weights == 0)
