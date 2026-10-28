"""Synthetic edge-detection benchmark: superderivative vs classical filters.

Run from the repository root: python src/demos/demo_edge_detection.py
"""
from __future__ import annotations
import sys

import os

from pathlib import Path

import numpy as np

# Anchored on this file, not on the working directory: run from src/
# with a relative "figures/" and the output lands in src/figures/,
# where the paper never looks.
FIG_DIR = Path(__file__).resolve().parents[2] / "figures"
from skimage import morphology

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from superderivative_edge import (
    parametric_magnitude,
    classical_magnitude,
    superderivative_score,
    canny_edges,
    canny_magnitude,
    bsds_best_f1,
)


def make_circle_image(size: int = 256) -> tuple[np.ndarray, np.ndarray]:
    """Concentric annuli; ground truth is the annulus boundaries."""
    y, x = np.mgrid[:size, :size] - (size - 1) / 2.0
    r = np.sqrt(x ** 2 + y ** 2)

    image = np.zeros((size, size), dtype=np.float64)
    image[r < size * 0.2] = 0.3
    image[(r >= size * 0.35) & (r < size * 0.55)] = 0.6
    image[(r >= size * 0.7) & (r < size * 0.9)] = 0.9

    gt = np.zeros_like(image)
    for radius in [size * 0.2, size * 0.35, size * 0.55, size * 0.7, size * 0.9]:
        mask = np.abs(r - radius) <= 1.5
        gt[mask] = 1.0
    gt = morphology.thin(gt.astype(bool)).astype(np.float64)
    return image, gt


def make_square_image(size: int = 256) -> tuple[np.ndarray, np.ndarray]:
    """Square on a background; ground truth is the square boundary."""
    image = np.ones((size, size), dtype=np.float64) * 0.2
    margin = size // 4
    image[margin: size - margin, margin: size - margin] = 0.8

    gt = np.zeros_like(image)
    gt[margin, margin: size - margin] = 1.0
    gt[size - margin, margin: size - margin] = 1.0
    gt[margin: size - margin, margin] = 1.0
    gt[margin: size - margin, size - margin] = 1.0
    return image, gt


def make_checker_image(size: int = 256) -> tuple[np.ndarray, np.ndarray]:
    """Checkerboard; ground truth is the grid lines."""
    image = np.zeros((size, size), dtype=np.float64)
    step = size // 8
    for i in range(8):
        for j in range(8):
            if (i + j) % 2 == 0:
                image[i * step:(i + 1) * step, j * step:(j + 1) * step] = 0.9
            else:
                image[i * step:(i + 1) * step, j * step:(j + 1) * step] = 0.2

    gt = np.zeros_like(image)
    for k in range(1, 8):
        gt[k * step, :] = 1.0
        gt[:, k * step] = 1.0
    return image, gt


def make_triangle_image(size: int = 256) -> tuple[np.ndarray, np.ndarray]:
    """Triangle; ground truth is its boundary."""
    image = np.ones((size, size), dtype=np.float64) * 0.2
    y, x = np.mgrid[:size, :size]
    # Equilateral-ish triangle.
    v1 = (y - size * 0.2) - (x - size * 0.5) * np.tan(np.pi / 3)
    v2 = (y - size * 0.2) + (x - size * 0.5) * np.tan(np.pi / 3)
    v3 = y - size * 0.8
    mask = (v1 > 0) & (v2 > 0) & (v3 < 0)
    image[mask] = 0.8

    # Ground truth is the boundary of the triangle mask itself, computed
    # analytically as the set difference between the mask and its erosion. It does
    # not depend on any derivative kernel, so the comparison against the
    # detectors is not circular.
    from scipy import ndimage
    gt = mask & ~ndimage.binary_erosion(mask)
    gt = morphology.thin(gt.astype(bool)).astype(np.float64)
    return image, gt


def make_gradient_image(size: int = 256) -> tuple[np.ndarray, np.ndarray]:
    """Linear intensity ramp with a step."""
    image = np.linspace(0.2, 0.8, size)[None, :] * np.ones((size, 1))
    image[:, size // 2:] += 0.15
    image = np.clip(image, 0.0, 1.0)

    gt = np.zeros_like(image)
    gt[:, size // 2] = 1.0
    return image, gt


# Threshold sweep and localisation tolerance for the synthetic benchmark.
# The tolerance is one pixel, the minimum the matcher allows; matching the BSDS
# rule keeps thin (Roberts, Canny) and thick (Sobel, Scharr) responses on the
# same footing instead of charging a one-pixel offset as both a false positive and
# a false negative.  A strict pixel-exact rule is *not* used, because it would
# charge that same offset as both a false positive and a false negative; the paper
# therefore reports only the tolerance-based rule.
SYNTH_THRESHOLDS = np.linspace(0.02, 0.90, 23)
SYNTH_TOL_FRAC = 1.0 / float(np.hypot(256, 256))


def evaluate_on_image(image: np.ndarray, gt: np.ndarray) -> dict:
    """Run all detectors and compute the best F1 over a threshold sweep."""
    eps = 1e-3
    image_pos = np.clip(image, eps, 1.0)

    h, w = image.shape
    y, x = np.mgrid[:h, :w]
    y = y - (h - 1) / 2.0
    x = x - (w - 1) / 2.0
    r = np.sqrt(x ** 2 + y ** 2)

    L = superderivative_score(image_pos, derivative="sobel")
    radial_sd = np.abs(L) / (r + 1e-9)

    detectors = {
        "$E_{0.25}$": parametric_magnitude(image_pos, p=0.25, derivative="sobel"),
        "Radial SD": radial_sd,
        "Sobel": classical_magnitude(image, method="sobel"),
        "Scharr": classical_magnitude(image, method="scharr"),
        "Prewitt": classical_magnitude(image, method="prewitt"),
        "Roberts": classical_magnitude(image, method="roberts"),
        "Canny": canny_edges(image, sigma=1.5),
    }

    results = {}
    for name, emap in detectors.items():
        if emap.max() > 0:
            emap_norm = emap / emap.max()
        else:
            emap_norm = emap
        results[name] = bsds_best_f1(emap_norm, gt, SYNTH_THRESHOLDS,
                                    tol_frac=SYNTH_TOL_FRAC)
        results[name]["map"] = emap_norm
    return results


def main() -> None:
    datasets = {
        "Circles": make_circle_image(),
        "Square": make_square_image(),
        "Checkerboard": make_checker_image(),
        "Triangle": make_triangle_image(),
        "Ramp": make_gradient_image(),
    }

    all_results: dict[str, dict] = {}
    for dname, (img, gt) in datasets.items():
        all_results[dname] = evaluate_on_image(img, gt)

    print("F1 scores on synthetic ground-truth edges (higher is better)")
    print()
    methods = ["$E_{0.25}$", "Radial SD", "Sobel", "Scharr", "Prewitt", "Roberts", "Canny"]
    header = f"{'Image':<15} " + " ".join(f"{m:>12}" for m in methods)
    print(header)
    print("-" * len(header))
    for dname in datasets:
        row = f"{dname:<15} "
        for m in methods:
            row += f"{all_results[dname][m]['f1']:>12.3f} "
        print(row)

    try:
        import matplotlib.pyplot as plt

        n_datasets = len(datasets)
        n_methods = len(methods) + 2
        fig, axes = plt.subplots(n_datasets, n_methods,
                                 figsize=(2.0 * n_methods, 2.0 * n_datasets))
        for row, (dname, (img, gt)) in enumerate(datasets.items()):
            res = all_results[dname]
            items = [("Image", img), ("Ground truth", gt)]
            for m in methods:
                if m == "Canny":
                    # Use a smoothed Canny map for visual consistency.
                    arr = canny_magnitude(img, sigma=1.5)
                    arr = np.clip(arr / (max(float(np.percentile(arr, 99.0)), 1e-9)), 0, 1)
                else:
                    arr = res[m]["map"]
                items.append((m, arr))
            for col, (name, arr) in enumerate(items):
                ax = axes[row, col]
                ax.imshow(arr, cmap="magma" if col > 1 else "gray", vmin=0, vmax=1)
                ax.axis("off")
                if row == 0:
                    ax.set_title(name, fontsize=9)
                if col == 0:
                    ax.text(-0.12, 0.5, dname, transform=ax.transAxes,
                            rotation=90, va="center", ha="center", fontsize=10,
                            fontweight="bold")
        fig.tight_layout()
        FIG_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(FIG_DIR / "edge_comparison.png", dpi=200)
        print("\nSaved figures/edge_comparison.png")
    except Exception as exc:  # pragma: no cover
        print(f"\nSkipping figure generation: {exc}")


if __name__ == "__main__":
    main()
