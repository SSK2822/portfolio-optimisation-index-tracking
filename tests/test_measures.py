import numpy as np
import pytest

from portopt.data import simple_returns
from portopt.measures import average_drawdown, drawdowns, max_drawdown, mean_squared_drawdown, ulcer_index
from portopt.ratios import ex_post_ratio

# Path of cumulative returns: 0.10, 0.05, 0.12, 0.02, 0.07
RETURNS = np.array([0.10, -0.05, 0.07, -0.10, 0.05])
EXPECTED_DD = np.array([0.0, 0.05, 0.0, 0.10, 0.05])


def test_drawdown_series_by_hand():
    np.testing.assert_allclose(drawdowns(RETURNS), EXPECTED_DD, atol=1e-12)


def test_measures_by_hand():
    assert max_drawdown(RETURNS) == pytest.approx(0.10)
    assert average_drawdown(RETURNS) == pytest.approx(0.04)
    assert mean_squared_drawdown(RETURNS) == pytest.approx((0.05**2 + 0.10**2 + 0.05**2) / 5)
    assert ulcer_index(RETURNS) == pytest.approx(np.sqrt(0.003))


def test_no_drawdown_for_rising_path():
    assert max_drawdown([0.01, 0.02, 0.0, 0.03]) == 0.0


def test_ratio_by_hand():
    rf = np.full(5, 0.01)
    mean_excess = RETURNS.mean() - 0.01  # 0.004
    assert ex_post_ratio("calmar", RETURNS, rf) == pytest.approx(mean_excess / 0.10)
    assert ex_post_ratio("pain", RETURNS, rf) == pytest.approx(mean_excess / 0.04)
    assert ex_post_ratio("martin", RETURNS, rf) == pytest.approx(mean_excess / np.sqrt(0.003))


def test_simple_returns_matches_price2ret():
    prices = np.array([[100.0, 10.0], [110.0, 9.0], [99.0, 9.9]])
    np.testing.assert_allclose(simple_returns(prices, axis=0), [[0.1, -0.1], [-0.1, 0.1]])
