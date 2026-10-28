"""Edge detection under low-light / low-contrast conditions.

Run from the repository root: python src/demos/demo_low_light.py
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
from skimage import data, color

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from superderivative_edge import (parametric_magnitude, classical_magnitude,
                                  regularized_magnitude)


def edge_map(image: np.ndarray, method: str) -> np.ndarray:
    """Edge-strength map of ``image`` for one of the compared methods."""
    if method == "Superderivative (p=0.5)":
        return parametric_magnitude(np.clip(image, 1e-3, 1.0), p=0.5, derivative="sobel")
    if method == "Superderivative (p=1)":
        return parametric_magnitude(np.clip(image, 1e-3, 1.0), p=1.0, derivative="sobel")
    if method.startswith("Reg."):
        return regularized_magnitude(np.clip(image, 1e-3, 1.0), p=1.0, alpha=0.05)
    return classical_magnitude(image, method=method.lower())


def _p99(arr: np.ndarray) -> np.ndarray:
    """Scale an edge map by its 99th percentile, as the figure does."""
    return arr / (max(float(np.percentile(arr, 99.0)), 1e-9))


def degradation(m0: np.ndarray, m1: np.ndarray) -> float:
    """Mean absolute normal/low-light difference, relative to the normal mean."""
    m0, m1 = _p99(m0), _p99(m1)
    return float(np.mean(np.abs(m1 - m0)) / (np.mean(np.abs(m0)) + 1e-12))


def main() -> None:
    im = data.immunohistochemistry()
    if im.ndim == 3:
        im = color.rgb2gray(im)
    img = im.astype(np.float64)[:256, :256]
    img = img / img.max()
    img = img / img.max()

    # The regularised variant is included because it is this paper's own
    # remedy for dark-region noise (Section V-B). Showing only the raw
    # p=0.5 column displays the problem without the fix.
    methods = ["Superderivative (p=0.5)", "Superderivative (p=1)",
               "Reg. $\\alpha=0.05$", "Sobel", "Scharr", "Prewitt"]

    # Quantitative degradation: ten independent noise realisations with a fixed
    # seed, so the numbers quoted in the paper are reproducible.
    rng = np.random.default_rng(42)
    n_real = 10
    dark = img ** 2.5
    normal_maps = {m: edge_map(img, m) for m in methods}
    deg = {m: [] for m in methods}
    for _ in range(n_real):
        noisy_dark = np.clip(dark + rng.normal(0, 0.02, dark.shape), 0, 1)
        for m in methods:
            deg[m].append(degradation(normal_maps[m], edge_map(noisy_dark, m)))
    print("Relative degradation (mean +/- std over 10 realisations, lower is better):")
    for m in methods:
        vals = np.array(deg[m])
        print(f"  {m:26s} {vals.mean():.3f} +/- {vals.std():.3f}")

    noisy_dark = np.clip(dark + np.random.default_rng(42).normal(0, 0.02, dark.shape), 0, 1)

    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, len(methods) + 1,
                                 figsize=(2.0 * (len(methods) + 1), 4.0))

        for col, name in enumerate(["Image"] + methods):
            if name == "Image":
                axes[0, col].imshow(img, cmap="gray", vmin=0, vmax=1)
                axes[1, col].imshow(noisy_dark, cmap="gray", vmin=0, vmax=1)
            else:
                m0 = normal_maps[name]
                m1 = edge_map(noisy_dark, name)
                # Scaled on the 99th percentile, not the maximum. A relative operator
                # divides by I, so one pixel in a dark region sets the maximum and
                # blackens the rest of the panel; the classical operators never divide
                # by I, so only the proposed one is penalised. That is a display
                # artefact, not a property of the filter.
                axes[0, col].imshow(np.clip(m0 / (max(float(np.percentile(m0, 99.0)), 1e-9)), 0, 1), cmap="magma", vmin=0, vmax=1)
                axes[1, col].imshow(np.clip(m1 / (max(float(np.percentile(m1, 99.0)), 1e-9)), 0, 1), cmap="magma", vmin=0, vmax=1)
            axes[0, col].axis("off")
            axes[1, col].axis("off")
            if col == 0:
                axes[0, col].set_title("Original", fontsize=9)
            else:
                axes[0, col].set_title(name, fontsize=9)

        axes[0, 0].text(-0.12, 0.5, "Normal", transform=axes[0, 0].transAxes,
                        rotation=90, va="center", ha="center", fontsize=10,
                        fontweight="bold")
        axes[1, 0].text(-0.12, 0.5, "Low light", transform=axes[1, 0].transAxes,
                        rotation=90, va="center", ha="center", fontsize=10,
                        fontweight="bold")

        fig.tight_layout()
        FIG_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(FIG_DIR / "low_light.png", dpi=200)
        print(f"Saved {FIG_DIR / "low_light.png"!s}")
    except Exception as exc:  # pragma: no cover
        print(f"Skipping figure generation: {exc}")


if __name__ == "__main__":
    main()
