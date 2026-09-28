import numpy as np
import pytest

from portopt.measures import drawdowns
from portopt.solvers import cvxpy_solve, lp_solve, sqp_solve


@pytest.fixture(scope="module")
def problem():
    rng = np.random.default_rng(0)
    n, m = 8, 120
    train = rng.normal(0.002, 0.03, size=(n, m)) + rng.normal(0, 0.01, size=(1, m))
    mu = train.mean(axis=1)
    return train, mu, 0.0005


def in_sample_drawdowns(x, train):
    # The solver's peak is floored at zero (xi >= 0), so prepend the starting point.
    return drawdowns(np.concatenate([[0.0], x @ train]))[1:]


@pytest.mark.parametrize("ratio", ["calmar", "pain"])
def test_lp_matches_cvxpy(problem, ratio):
    a, b = lp_solve(ratio, *problem), cvxpy_solve(ratio, *problem)
    assert a.success and b.success
    assert a.objective == pytest.approx(b.objective, rel=1e-4)


def test_sqp_matches_cvxpy(problem):
    a, b = sqp_solve(*problem), cvxpy_solve("martin", *problem)
    assert a.success and b.success
    assert a.objective == pytest.approx(b.objective, rel=1e-4)


@pytest.mark.parametrize("ratio", ["calmar", "pain", "martin"])
def test_weights_feasible(problem, ratio):
    sol = sqp_solve(*problem) if ratio == "martin" else lp_solve(ratio, *problem)
    assert np.all(sol.weights >= -1e-9)
    assert sol.weights.sum() == pytest.approx(1.0, abs=1e-6)


@pytest.mark.parametrize("ratio", ["calmar", "pain", "martin"])
def test_optimum_equals_best_in_sample_ratio(problem, ratio):
    """At the optimum, mu'y - rf*eta equals the in-sample ratio of x = y/eta
    (for Martin the denominator is the Ulcer index, because the constraint is
    on the mean squared drawdown and scales quadratically)."""
    train, mu, rf = problem
    sol = sqp_solve(*problem) if ratio == "martin" else lp_solve(ratio, *problem)
    d = in_sample_drawdowns(sol.weights, train)
    risk = {"calmar": d.max(), "pain": d.mean(), "martin": np.sqrt(np.mean(d**2))}[ratio]
    assert (mu - rf) @ sol.weights / risk == pytest.approx(sol.objective, rel=1e-4)

    # Equal weights cannot beat the optimum.
    ew = np.full(len(mu), 1 / len(mu))
    d = in_sample_drawdowns(ew, train)
    risk_ew = {"calmar": d.max(), "pain": d.mean(), "martin": np.sqrt(np.mean(d**2))}[ratio]
    assert (mu - rf) @ ew / risk_ew <= sol.objective + 1e-9
