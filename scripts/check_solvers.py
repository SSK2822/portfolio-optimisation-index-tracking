"""Check the LP and SLSQP solutions against cvxpy on sampled weeks.

Usage:
    python scripts/check_solvers.py [--samples 5]

For each dataset and ratio, a few out-of-sample weeks are re-solved with cvxpy
(Clarabel). The problems are convex, so cvxpy's optimum is the global optimum.
Writes results/solver_check.csv.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from portopt import data, solvers  # noqa: E402
from portopt.measures import RATIOS  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=5)
    args = parser.parse_args()

    rows = []
    for ds in data.UPDATED + data.BCST:
        market = data.load(ds)
        T = market.num_weeks
        weeks = np.linspace(T // 2 + 1, T, args.samples).round().astype(int)
        for ratio in RATIOS:
            gaps, wdiff = [], []
            for week in weeks:
                train = market.asset_returns[:, : week - 1]
                mu, rf = train.mean(axis=1), float(market.risk_free[week - 1])
                a = solvers.solve(ratio, train, mu, rf)
                b = solvers.cvxpy_solve(ratio, train, mu, rf)
                gaps.append(abs(a.objective - b.objective) / abs(b.objective))
                wdiff.append(np.abs(a.weights - b.weights).max())
            rows.append(
                {
                    "dataset": ds,
                    "ratio": ratio,
                    "solver": "SLSQP" if ratio == "martin" else "HiGHS LP",
                    "weeks": len(weeks),
                    "max_rel_objective_gap": f"{max(gaps):.1e}",
                    "max_abs_weight_diff": f"{max(wdiff):.1e}",
                }
            )
            print(rows[-1], flush=True)

    out = ROOT / "results" / "solver_check.csv"
    out.parent.mkdir(exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
