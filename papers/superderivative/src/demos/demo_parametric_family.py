"""Visualise the parametric family E_p = ||grad I|| / I^p.

Run from the repository root: python src/demos/demo_parametric_family.py
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
from superderivative_edge import parametric_magnitude


def to_gray(im: np.ndarray) -> np.ndarray:
    if im.ndim == 3:
        return color.rgb2gray(im)
    return im.astype(np.float64) / 255.0


def load_images(size: int = 256) -> dict[str, np.ndarray]:
    """Load natural images and resize/crop to a common square size."""
    images = {}
    for name, im in [("camera", data.camera()),
                     ("coins", data.coins()),
                     ("moon", data.moon())]:
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
    p_values = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5]

    try:
        import matplotlib.pyplot as plt

        n_images = len(images)
        n_cols = len(p_values) + 1  # original + p values
        fig, axes = plt.subplots(n_images, n_cols,
                                 figsize=(2.0 * n_cols, 2.0 * n_images))

        for row, (img_name, img) in enumerate(images.items()):
            img_pos = np.clip(img, 1e-3, 1.0)
            axes[row, 0].imshow(img, cmap="gray", vmin=0, vmax=1)
            axes[row, 0].axis("off")
            if row == 0:
                axes[row, 0].set_title("Original", fontsize=9)

            for col, p in enumerate(p_values, start=1):
                emap = parametric_magnitude(img_pos, p=p, derivative="sobel")
                # Scaled on the 99th percentile, not the maximum. A relative operator
                # divides by I, so one pixel in a dark region sets the maximum and
                # blackens the rest of the panel; the classical operators never divide
                # by I, so only the proposed one is penalised. That is a display
                # artefact, not a property of the filter.
                emap = np.clip(emap / (max(float(np.percentile(emap, 99.0)), 1e-9)), 0, 1)
                axes[row, col].imshow(emap, cmap="magma", vmin=0, vmax=1)
                axes[row, col].axis("off")
                if row == 0:
                    axes[row, col].set_title(f"$p={p}$", fontsize=9)

            axes[row, 0].text(-0.12, 0.5, img_name, transform=axes[row, 0].transAxes,
                              rotation=90, va="center", ha="center", fontsize=10,
                              fontweight="bold")

        fig.tight_layout()
        FIG_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(FIG_DIR / "parametric_family.png", dpi=200)
        print(f"Saved {FIG_DIR / "parametric_family.png"!s}")
    except Exception as exc:  # pragma: no cover
        print(f"Skipping figure generation: {exc}")


if __name__ == "__main__":
    main()
