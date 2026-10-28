"""Demonstrate intensity-scale invariance of the superderivative operator.

Run from the repository root: python src/demos/demo_intensity_invariance.py
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
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from superderivative_edge import superderivative_magnitude


def make_image(size: int = 256) -> np.ndarray:
    """Piecewise-constant image with strictly positive grey levels."""
    image = np.ones((size, size), dtype=np.float64) * 0.15
    image[size // 4:3 * size // 4, size // 4:3 * size // 4] = 0.55
    image[size // 3:2 * size // 3, size // 3:2 * size // 3] = 0.95
    return image


def sobel_magnitude(image: np.ndarray) -> np.ndarray:
    ix = ndimage.sobel(image, axis=1)
    iy = ndimage.sobel(image, axis=0)
    return np.sqrt(ix ** 2 + iy ** 2)


def main() -> None:
    image = make_image(256)
    scales = [0.3, 1.0, 3.0]

    rows = []
    for c in scales:
        scaled = c * image  # strictly positive because image > 0
        rows.append({
            "scale": c,
            "image": scaled,
            "sobel": sobel_magnitude(scaled),
            "sd": superderivative_magnitude(scaled, derivative="sobel"),
        })

    # Print absolute map ranges to verify invariance numerically.
    print("Absolute Sobel maps (should scale with c):")
    for r in rows:
        m = r["sobel"]
        print(f"  c={r['scale']:.1f}  min={m.min():.4f}  max={m.max():.4f}")
    print("\nAbsolute Superderivative maps (should be identical):")
    for r in rows:
        m = r["sd"]
        print(f"  c={r['scale']:.1f}  min={m.min():.4f}  max={m.max():.4f}")

    # Common absolute scales so that changes are visible.
    max_sobel = max(r["sobel"].max() for r in rows)
    max_sd = max(r["sd"].max() for r in rows)

    try:
        import matplotlib.pyplot as plt

        n = len(scales)
        fig, axes = plt.subplots(n, 3, figsize=(8, 2.6 * n))
        for i, r in enumerate(rows):
            axes[i, 0].imshow(r["image"], cmap="gray", vmin=0, vmax=1)
            axes[i, 0].set_title(f"Image ($c={r['scale']:.1f}$)")
            axes[i, 0].axis("off")

            axes[i, 1].imshow(r["sobel"], cmap="magma", vmin=0, vmax=max_sobel)
            axes[i, 1].set_title("Sobel (common scale)")
            axes[i, 1].axis("off")

            axes[i, 2].imshow(r["sd"], cmap="magma", vmin=0, vmax=max_sd)
            axes[i, 2].set_title("Superderivative (common scale)")
            axes[i, 2].axis("off")

        fig.tight_layout()
        FIG_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(FIG_DIR / "intensity_invariance.png", dpi=200)
        print("\nSaved figures/intensity_invariance.png")
    except Exception as exc:  # pragma: no cover
        print(f"\nSkipping figure generation: {exc}")


if __name__ == "__main__":
    main()
