"""Cumulative-return and weight plots from cached backtest weights.

Usage:
    python scripts/plot.py [--datasets HDAX DJIA ...] [--ratios calmar ...]

Run scripts/run_experiments.py first. Figures go to results/figures/.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from portopt import backtest, data  # noqa: E402
from run_experiments import RATIO_NAMES, load_result, weights_path  # noqa: E402

FIGURES = ROOT / "results" / "figures"


def cumulative_plot(dataset: str, ratios: list[str]) -> None:
    market = data.load(dataset)
    (ew, _), (mi, _) = backtest.benchmark_returns(market)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for ratio in ratios:
        if weights_path(dataset, ratio).exists():
            res = load_result(dataset, ratio)
            ax.plot(np.cumsum(res.returns), label=f"NO ({RATIO_NAMES[ratio]})")
    ax.plot(np.cumsum(ew), "--", color="0.3", label="EW")
    ax.plot(np.cumsum(mi), ":", color="0.3", label="MI")
    ax.set_title(f"{data.LABELS[dataset]} ({'BCST' if dataset in data.BCST else 'Updated'}): out-of-sample")
    ax.set_xlabel("Out-of-sample week")
    ax.set_ylabel("Cumulative sum of weekly returns")
    ax.legend(frameon=False)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES / f"{dataset}_cumulative.png", dpi=130)
    plt.close(fig)


def weights_plot(dataset: str, ratio: str, top: int = 8) -> None:
    res = load_result(dataset, ratio)
    W = res.weights
    order = np.argsort(-W.mean(axis=1))
    main, rest = W[order[:top]], W[order[top:]].sum(axis=0)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.stackplot(np.arange(W.shape[1]), *main, rest, labels=[f"asset {i + 1}" for i in order[:top]] + ["other"])
    ax.set_xlim(0, W.shape[1] - 1)
    ax.set_ylim(0, 1)
    ax.set_title(f"{data.LABELS[dataset]}: {RATIO_NAMES[ratio]} portfolio weights")
    ax.set_xlabel("Out-of-sample week")
    ax.set_ylabel("Weight")
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / f"{dataset}_{ratio}_weights.png", dpi=130)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=data.UPDATED + data.BCST)
    parser.add_argument("--ratios", nargs="+", default=list(RATIO_NAMES))
    args = parser.parse_args()
    FIGURES.mkdir(parents=True, exist_ok=True)
    for ds in args.datasets:
        cumulative_plot(ds, args.ratios)
        for ratio in args.ratios:
            if weights_path(ds, ratio).exists():
                weights_plot(ds, ratio)
    print(f"figures written to {FIGURES}")


if __name__ == "__main__":
    main()
