"""Visual comparison of edge detectors on natural images.

Run from the repository root: python src/demos/demo_edge_natural.py
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
from superderivative_edge import (
    parametric_magnitude,
    classical_magnitude,
    canny_magnitude,
)


def to_gray(im: np.ndarray) -> np.ndarray:
    if im.ndim == 3:
        return color.rgb2gray(im)
    return im.astype(np.float64) / 255.0


def load_images(size: int = 256) -> dict[str, np.ndarray]:
    """Load natural images and resize/crop to a common square size."""
    images = {}
    candidates = {
        "brick": data.brick(),
        "camera": data.camera(),
        "chelsea": data.chelsea(),
        "coffee": data.coffee(),
        "coins": data.coins(),
        "moon": data.moon(),
    }
    for name, im in candidates.items():
        im = to_gray(im)
        h, w = im.shape
        s = min(h, w)
        im = im[(h - s) // 2:(h + s) // 2, (w - s) // 2:(w + s) // 2]
        if s != size:
            from skimage.transform import resize
            im = resize(im, (size, size), anti_aliasing=True, mode="reflect")
        images[name] = np.clip(im, 0.0, 1.0)
    return images


def main() -> None:
    images = load_images(256)
    methods = ["$E_{0.25}$", "Sobel", "Scharr", "Prewitt", "Roberts", "Canny"]

    try:
        import matplotlib.pyplot as plt

        n_images = len(images)
        n_methods = len(methods) + 1  # original + detectors
        fig, axes = plt.subplots(n_images, n_methods,
                                 figsize=(2.0 * n_methods, 2.0 * n_images))

        for row, (img_name, img) in enumerate(images.items()):
            eps = 1e-3
            img_pos = np.clip(img, eps, 1.0)
            maps = {
                "Original": img,
                "$E_{0.25}$": parametric_magnitude(img_pos, p=0.25, derivative="sobel"),
                "Sobel": classical_magnitude(img, method="sobel"),
                "Scharr": classical_magnitude(img, method="scharr"),
                "Prewitt": classical_magnitude(img, method="prewitt"),
                "Roberts": classical_magnitude(img, method="roberts"),
                "Canny": canny_magnitude(img, sigma=1.5),
            }
            for col, m_name in enumerate(["Original"] + methods):
                ax = axes[row, col]
                arr = maps[m_name]
                if m_name != "Original":
                    # Scaled on the 99th percentile, not the maximum. A relative operator
                    # divides by I, so one pixel in a dark region sets the maximum and
                    # blackens the rest of the panel; the classical operators never divide
                    # by I, so only the proposed one is penalised. That is a display
                    # artefact, not a property of the filter.
                    arr = np.clip(arr / (max(float(np.percentile(arr, 99.0)), 1e-9)), 0, 1)
                ax.imshow(arr, cmap="magma" if m_name != "Original" else "gray", vmin=0, vmax=1)
                ax.axis("off")
                if row == 0:
                    ax.set_title(m_name, fontsize=9)
                if col == 0:
                    ax.text(-0.12, 0.5, img_name, transform=ax.transAxes,
                            rotation=90, va="center", ha="center", fontsize=10,
                            fontweight="bold")

        fig.tight_layout()
        FIG_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(FIG_DIR / "edge_natural.png", dpi=200)
        print(f"Saved {FIG_DIR / "edge_natural.png"!s}")
    except Exception as exc:  # pragma: no cover
        print(f"Skipping figure generation: {exc}")


if __name__ == "__main__":
    main()
