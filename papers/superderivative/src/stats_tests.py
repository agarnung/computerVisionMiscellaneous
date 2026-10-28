"""Paired t-tests and bootstrap intervals quoted in the paper.

Run from the repository root: python src/stats_tests.py [--split val|test]
[--norm minmax|percentile] [--n-boot 10000]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy import stats

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE / "official_bsds"))
sys.path.insert(0, str(HERE))

from eval_bsds500_official_code import (  # noqa: E402
    CANNY_SIGMAS,
    _best_f,
    _cache_path,
    _image_ids,
    _metrics,
)

RNG_SEED = 20261007

# Pairs compared in the paper, as (label, method A, method B).
PAIRS = [
    ("Canny vs p=0.25", "canny", "parametric_p0.25"),
    ("p=0.25 vs p=0.5", "parametric_p0.25", "parametric_p0.5"),
    ("p=0.25 vs Sobel", "parametric_p0.25", "sobel"),
    ("p=0.25 vs Prewitt", "parametric_p0.25", "prewitt"),
    ("p=0.25 vs Scharr", "parametric_p0.25", "scharr"),
    ("p=1 vs Sobel", "parametric_p1.0", "sobel"),
    ("p=1.5 vs Sobel", "parametric_p1.5", "sobel"),
]


def _counts_for(name: str, stem: str, split: str, norm: str) -> np.ndarray:
    if name == "canny":
        best, best_f = None, -1.0
        for sigma in CANNY_SIGMAS:
            c = np.load(_cache_path(split, norm, f"canny_s{sigma}", stem))["counts"]
            f = _best_f(c)
            if f > best_f:
                best_f, best = f, c
        return best
    return np.load(_cache_path(split, norm, name, stem))["counts"]


def _per_image_ois(name: str, ids: list[str], split: str,
                   norm: str) -> np.ndarray:
    values = []
    for stem in ids:
        try:
            counts = _counts_for(name, stem, split, norm)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"cache missing for {name}/{stem}; run "
                "src/official_bsds/eval_bsds500_official_code.py first."
            ) from None
        values.append(_metrics(counts)[0])
    return np.array(values)


def _bootstrap_ci(diff: np.ndarray, n_boot: int,
                  rng: np.random.Generator) -> tuple[float, float]:
    n = diff.size
    idx = rng.integers(0, n, size=(n_boot, n))
    means = diff[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def _report(label: str, a: np.ndarray, b: np.ndarray, n_boot: int,
            rng: np.random.Generator) -> None:
    diff = a - b
    _, p = stats.ttest_rel(a, b)
    lo, hi = _bootstrap_ci(diff, n_boot, rng)
    print(f"  {label:32s} diff={diff.mean():+.4f}  "
          f"95% CI [{lo:+.4f}, {hi:+.4f}]  p={float(p):.4g}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--norm", choices=("minmax", "percentile"),
                        default="minmax")
    parser.add_argument("--n-boot", type=int, default=10000)
    args = parser.parse_args()

    ids = _image_ids(args.split)
    if not ids:
        raise RuntimeError(f"No BSDS500 images with ground truth found for {args.split}.")

    needed = sorted({m for _, a, b in PAIRS for m in (a, b)})
    ois = {}
    for name in needed:
        try:
            ois[name] = _per_image_ois(name, ids, args.split, args.norm)
        except FileNotFoundError:
            print(f"  note: no cached counts for {name}; skipping its pairs")

    rng = np.random.default_rng(RNG_SEED)
    print(f"Official BSDS500 benchmark ({args.split} split, {len(ids)} images, "
          f"per-image OIS, norm={args.norm}):")
    for label, a, b in PAIRS:
        try:
            ois_a, ois_b = ois[a], ois[b]
        except KeyError:
            print(f"  {label:32s} skipped (no cached counts)")
            continue
        _report(label, ois_a, ois_b, args.n_boot, rng)


if __name__ == "__main__":
    main()
