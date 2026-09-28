"""Run every (dataset, ratio) backtest and write the results tables.

Usage:
    python scripts/run_experiments.py [--datasets DJIA HDAX ...] [--ratios calmar ...] [--jobs 4]

Weekly weights are cached in results/weights/<dataset>_<ratio>.npz; delete a file
to recompute it. Tables are written to results/ex_post_ratios.csv and
results/ex_post_ratios.md.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from portopt import backtest, data  # noqa: E402
from portopt.measures import RATIOS  # noqa: E402

RESULTS = ROOT / "results"
WEIGHTS = RESULTS / "weights"
RATIO_NAMES = {"calmar": "Calmar", "martin": "Martin", "pain": "Pain"}


def weights_path(dataset: str, ratio: str) -> Path:
    return WEIGHTS / f"{dataset}_{ratio}.npz"


def run_one(dataset: str, ratio: str) -> str:
    path = weights_path(dataset, ratio)
    if path.exists():
        return f"{dataset} {ratio}: cached"
    start = time.time()
    market = data.load(dataset)
    res = backtest.run(market, ratio)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        weeks=res.weeks,
        weights=res.weights,
        no_trade=res.no_trade,
        solver_ok=res.solver_ok,
        objectives=res.objectives,
    )
    return (
        f"{dataset} {ratio}: {time.time() - start:.0f}s, no-trade weeks {res.no_trade.sum()}, "
        f"solver failures {(~res.solver_ok).sum()}"
    )


def load_result(dataset: str, ratio: str) -> backtest.BacktestResult:
    market = data.load(dataset)
    z = np.load(weights_path(dataset, ratio))
    weeks, W = z["weeks"], z["weights"]
    returns = np.einsum("ij,ij->j", W, market.asset_returns[:, weeks - 1])
    return backtest.BacktestResult(
        ratio, weeks, W, returns, market.risk_free[weeks - 1], z["no_trade"], z["solver_ok"], z["objectives"]
    )


def write_tables(datasets: list[str], ratios: list[str]) -> None:
    rows = []
    for ratio in ratios:
        for ds in datasets:
            if not weights_path(ds, ratio).exists():
                continue
            s = backtest.summary(data.load(ds), load_result(ds, ratio))
            lib = "BCST-Library" if ds in data.BCST else "Updated Library"
            rows.append({"library": lib, "ratio": RATIO_NAMES[ratio], "dataset": data.LABELS[ds], **s})

    RESULTS.mkdir(exist_ok=True)
    with open(RESULTS / "ex_post_ratios.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["library", "ratio", "dataset", "NO", "EW", "MI"])
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in r.items()})

    lines = ["# Ex-post annualised ratios", ""]
    for lib in ("Updated Library", "BCST-Library"):
        sub = [r for r in rows if r["library"] == lib]
        if not sub:
            continue
        lines += [f"## {lib}", "", "| Ratio | Dataset | NO | EW | MI |", "|---|---|---:|---:|---:|"]
        for r in sub:
            best = max(("NO", "EW", "MI"), key=lambda k: r[k])
            cells = [f"**{r[k]:.4f}**" if k == best else f"{r[k]:.4f}" for k in ("NO", "EW", "MI")]
            lines.append(f"| {r['ratio']} | {r['dataset']} | " + " | ".join(cells) + " |")
        lines.append("")
    (RESULTS / "ex_post_ratios.md").write_text("\n".join(lines))
    print("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=data.UPDATED + data.BCST)
    parser.add_argument("--ratios", nargs="+", default=list(RATIOS))
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--tables-only", action="store_true")
    args = parser.parse_args()

    if not args.tables_only:
        # Slowest jobs first so the pool stays busy.
        tasks = sorted(
            ((d, r) for d in args.datasets for r in args.ratios),
            key=lambda t: (t[1] != "martin", -data.load(t[0]).num_assets),
        )
        with ProcessPoolExecutor(max_workers=args.jobs) as pool:
            for msg in pool.map(run_one, *zip(*tasks)):
                print(msg, flush=True)
    write_tables(args.datasets, args.ratios)


if __name__ == "__main__":
    main()
