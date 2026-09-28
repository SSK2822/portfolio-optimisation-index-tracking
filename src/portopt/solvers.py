"""Ratio-maximisation solvers.

Each ratio  (mu - rf)'x / Risk(x)  over long-only, fully invested weights x is a
fractional program. With the Charnes-Cooper / Schaible substitution y = eta * x,
eta > 0, it becomes the convex program

    max_{y, eta, xi}  mu'y - rf * eta
    s.t.  e'y = eta,   y >= 0,   eta >= 0,
          xi_j >= R_j'y,  xi_j >= xi_{j-1},  xi_j >= 0,     j = 1..m
          Risk(xi - R y) <= 1

where R_j = r_1 + ... + r_j is the cumulative sum of the first j training weeks,
so xi_j is the running peak of the portfolio path and xi_j - R_j'y its drawdown.
The drawdown condition differs per ratio:

    Calmar:  xi_j - R_j'y <= 1  for all j                    (linear)
    Pain:    (1/m) sum_j (xi_j - R_j'y) <= 1                 (linear)
    Martin:  (1/m) sum_j (xi_j - R_j'y)^2 <= 1               (convex quadratic)

Weights are recovered as x = y / eta.

* ``lp_solve``    Calmar and Pain with scipy's HiGHS LP solver.
* ``sqp_solve``   Martin with SLSQP (the FYP used MATLAB fmincon SQP).
* ``cvxpy_solve`` any of the three with cvxpy, used as an independent check.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
from scipy.optimize import linprog, minimize


@dataclass
class Solution:
    weights: np.ndarray
    objective: float  # optimal value of mu'y - rf * eta
    success: bool
    message: str = ""


def _cumulative(train: np.ndarray) -> np.ndarray:
    """(m, n) matrix whose row j is R_j = r_1 + ... + r_j."""
    return np.cumsum(train, axis=1).T


def _weights(y: np.ndarray, eta: float) -> np.ndarray:
    y = np.maximum(y, 0.0)
    return y / eta if eta > 0 else y / y.sum()


def lp_solve(ratio: str, train: np.ndarray, mu: np.ndarray, rf: float) -> Solution:
    """Solve the Calmar or Pain program as a linear program.

    Variables are ordered [y (n), eta (1), xi (m)].
    """
    if ratio not in ("calmar", "pain"):
        raise ValueError("lp_solve handles 'calmar' and 'pain' only")
    n, m = train.shape
    R = sp.csr_matrix(_cumulative(train))
    I = sp.identity(m, format="csr")
    zero_col = sp.csr_matrix((m, 1))

    # xi_j >= R_j'y   ->   R_j'y - xi_j <= 0
    peak = sp.hstack([R, zero_col, -I])
    # xi_j >= xi_{j-1} ->  xi_{j-1} - xi_j <= 0
    order = sp.hstack([sp.csr_matrix((m - 1, n + 1)), sp.eye(m - 1, m) - sp.eye(m - 1, m, k=1)])
    if ratio == "calmar":
        risk = sp.hstack([-R, zero_col, I])  # xi_j - R_j'y <= 1
        risk_b = np.ones(m)
    else:
        row = np.concatenate([-np.asarray(R.sum(axis=0)).ravel(), [0.0], np.ones(m)]) / m
        risk = sp.csr_matrix(row)  # (1/m) sum_j (xi_j - R_j'y) <= 1
        risk_b = np.ones(1)

    A_ub = sp.vstack([peak, order, risk], format="csr")
    b_ub = np.concatenate([np.zeros(m), np.zeros(m - 1), risk_b])
    A_eq = sp.csr_matrix(np.concatenate([np.ones(n), [-1.0], np.zeros(m)]))
    c = np.concatenate([-mu, [rf], np.zeros(m)])

    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=[0.0], bounds=(0, None), method="highs")
    if res.status != 0:
        return Solution(np.zeros(n), np.nan, False, res.message)
    y, eta = res.x[:n], res.x[n]
    return Solution(_weights(y, eta), -res.fun, True, res.message)


def _peak_drawdown(R: np.ndarray, y: np.ndarray):
    """Drawdowns xi_j - R_j'y with xi at its smallest feasible value, and the peak index."""
    path = R @ y
    peak_idx = np.zeros(path.size, dtype=int)
    best, best_i = 0.0, -1  # xi >= 0, so the peak starts at zero
    for j, v in enumerate(path):
        if v > best:
            best, best_i = v, j
        peak_idx[j] = best_i
    peak = np.where(peak_idx >= 0, path[np.maximum(peak_idx, 0)], 0.0)
    return peak - path, peak_idx


def sqp_solve(
    train: np.ndarray,
    mu: np.ndarray,
    rf: float,
    max_iter: int = 3000,
    tol: float = 1e-10,
) -> Solution:
    """Solve the Martin program with SLSQP.

    For fixed y the smallest feasible xi is the running peak of R y floored at
    zero, and it minimises every drawdown term at once, so xi can be eliminated.
    SLSQP then works on [y; eta] only, with the constraint

        g(y) = 1 - (1/m) sum_j D_j(y)^2 >= 0,   D_j(y) = max(0, max_{i<=j} R_i'y) - R_j'y.

    Note the sign: SLSQP expects inequality constraints as fun(u) >= 0, whereas
    MATLAB's fmincon expects c(u) <= 0.

    The starting point is the one used in the FYP: equal weights scaled by the
    inverse of their in-sample volatility.
    """
    n, m = train.shape
    R = _cumulative(train)

    w0 = np.full(n, 1.0 / n)
    vol0 = float(np.sqrt(w0 @ np.cov(train) @ w0))
    u0 = np.concatenate([w0 / vol0, [1.0 / vol0]])

    def objective(u):
        return -(mu @ u[:n] - rf * u[n])

    def objective_grad(u):
        return np.concatenate([-mu, [rf]])

    def risk(u):
        d, _ = _peak_drawdown(R, u[:n])
        return 1.0 - np.mean(d**2)

    def risk_grad(u):
        d, peak_idx = _peak_drawdown(R, u[:n])
        # dD_j/dy = R_peak(j) - R_j   (R_peak = 0 while the peak is the zero floor)
        dD = -R.copy()
        has_peak = peak_idx >= 0
        dD[has_peak] += R[peak_idx[has_peak]]
        grad_y = -(2.0 / m) * (d @ dD)
        return np.concatenate([grad_y, [0.0]])

    res = minimize(
        objective,
        u0,
        jac=objective_grad,
        method="SLSQP",
        bounds=[(0.0, None)] * (n + 1),
        constraints=[
            {"type": "eq", "fun": lambda u: u[:n].sum() - u[n], "jac": lambda u: np.concatenate([np.ones(n), [-1.0]])},
            {"type": "ineq", "fun": risk, "jac": risk_grad},
        ],
        options={"maxiter": max_iter, "ftol": tol},
    )
    y, eta = res.x[:n], res.x[n]
    return Solution(_weights(y, eta), -res.fun, bool(res.success), res.message)


def cvxpy_solve(ratio: str, train: np.ndarray, mu: np.ndarray, rf: float) -> Solution:
    """Solve any of the three programs with cvxpy (explicit xi variables)."""
    import cvxpy as cp

    n, m = train.shape
    R = _cumulative(train)
    y = cp.Variable(n, nonneg=True)
    eta = cp.Variable(nonneg=True)
    xi = cp.Variable(m, nonneg=True)
    dd = xi - R @ y
    cons = [cp.sum(y) == eta, dd >= 0, xi[1:] >= xi[:-1]]
    if ratio == "calmar":
        cons.append(dd <= 1)
    elif ratio == "pain":
        cons.append(cp.sum(dd) / m <= 1)
    elif ratio == "martin":
        cons.append(cp.sum_squares(dd) / m <= 1)
    else:
        raise ValueError(ratio)
    prob = cp.Problem(cp.Maximize(mu @ y - rf * eta), cons)
    prob.solve(solver=cp.CLARABEL)
    ok = prob.status == cp.OPTIMAL
    if not ok or y.value is None:
        return Solution(np.zeros(n), np.nan, False, prob.status)
    return Solution(_weights(y.value, float(eta.value)), float(prob.value), True, prob.status)


def solve(ratio: str, train: np.ndarray, mu: np.ndarray, rf: float) -> Solution:
    """Default solver per ratio: LP for Calmar and Pain, SLSQP for Martin."""
    if ratio == "martin":
        return sqp_solve(train, mu, rf)
    return lp_solve(ratio, train, mu, rf)
