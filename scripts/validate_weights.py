"""Compare Python weights with the weekly weight files of the original MATLAB run.

Usage:
    python scripts/validate_weights.py /path/to/original/code [--out results/weight_validation.csv]

The original code folder holds one directory per ratio (``Calmar_ratio`` etc.),
each with result folders such as ``UPDATE_HDAX_Calmar/xt_470.txt``, one file per
out-of-sample week. Files are matched to weeks by the number in the file name.
Empty files (weeks the MATLAB run did not finish) are skipped.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from run_experiments import RATIO_NAMES, weights_path  # noqa: E402

FOLDERS = {
    "DJIA": "NEW_DJIA",
    "NASDAQ100": "NEW_NASDAQ100",
    "FTSE100": "NEW_FTSE100",
    "SP500": "NEW_SP500",
    "HDAX": "UPDATE_HDAX",
    "FTSE": "UPDATE_FTSE",
    "HSCI": "UPDATE_HSCI",
    "SP500U": "UPDATE_SP500",
}


def read_original(folder: Path) -> dict[int, np.ndarray]:
    out = {}
    for f in folder.glob("xt_*.txt"):
        if f.stat().st_size == 0:
            continue
        out[int(re.search(r"xt_(\d+)", f.name).group(1))] = np.loadtxt(f)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "weight_validation.csv")
    args = parser.parse_args()

    rows = []
    for ratio, label in RATIO_NAMES.items():
        for ds, prefix in FOLDERS.items():
            folder = args.source / f"{label}_ratio" / f"{prefix}_{label}"
            if not folder.is_dir() or not weights_path(ds, ratio).exists():
                continue
            orig = read_original(folder)
            z = np.load(weights_path(ds, ratio))
            ours = {int(w): z["weights"][:, k] for k, w in enumerate(z["weeks"])}
            common = sorted(set(orig) & set(ours))
            if not common:
                continue
            diffs = np.array([np.abs(orig[w] - ours[w]).max() for w in common])
            rows.append(
                {
                    "ratio": label,
                    "dataset": ds,
                    "weeks_compared": len(common),
                    "weeks_expected": len(ours),
                    "max_abs_diff": f"{diffs.max():.2e}",
                    "median_max_abs_diff": f"{np.median(diffs):.2e}",
                    "share_weeks_within_1e-3": f"{np.mean(diffs < 1e-3):.3f}",
                }
            )
            print(rows[-1])

    args.out.parent.mkdir(exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
