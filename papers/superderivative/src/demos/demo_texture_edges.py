"""Edge detection on textured natural images.

Run from the repository root: python src/demos/demo_texture_edges.py
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
from superderivative_edge import parametric_magnitude, classical_magnitude


def main() -> None:
    # Brodatz-like textures are not in skimage; use natural textured images.
    candidates = {
        "gravel": data.gravel(),
        "page": data.page(),
        "cell": data.cell(),
    }
    methods = ["$E_{0.25}$", "Sobel", "Scharr", "Prewitt"]

    try:
        import matplotlib.pyplot as plt

        n_imgs = len(candidates)
        n_methods = len(methods) + 1
        fig, axes = plt.subplots(n_imgs, n_methods,
                                 figsize=(2.0 * n_methods, 2.0 * n_imgs))

        for row, (img_name, im) in enumerate(candidates.items()):
            if im.ndim == 3:
                im = color.rgb2gray(im)
            im = im.astype(np.float64) / 255.0
            im = np.clip(im, 0.0, 1.0)

            axes[row, 0].imshow(im, cmap="gray", vmin=0, vmax=1)
            axes[row, 0].axis("off")
            if row == 0:
                axes[row, 0].set_title("Original", fontsize=9)
            axes[row, 0].text(-0.12, 0.5, img_name, transform=axes[row, 0].transAxes,
                              rotation=90, va="center", ha="center", fontsize=10,
                              fontweight="bold")

            for col, method in enumerate(methods, start=1):
                if method == "$E_{0.25}$":
                    emap = parametric_magnitude(np.clip(im, 1e-3, 1.0), p=0.25, derivative="sobel")
                else:
                    emap = classical_magnitude(im, method=method.lower())
                # Scaled on the 99th percentile, not the maximum. A relative operator
                # divides by I, so one pixel in a dark region sets the maximum and
                # blackens the rest of the panel; the classical operators never divide
                # by I, so only the proposed one is penalised. That is a display
                # artefact, not a property of the filter.
                emap = np.clip(emap / (max(float(np.percentile(emap, 99.0)), 1e-9)), 0, 1)
                axes[row, col].imshow(emap, cmap="magma", vmin=0, vmax=1)
                axes[row, col].axis("off")
                if row == 0:
                    axes[row, col].set_title(method, fontsize=9)

        fig.tight_layout()
        FIG_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(FIG_DIR / "texture_edges.png", dpi=200)
        print(f"Saved {FIG_DIR / "texture_edges.png"!s}")
    except Exception as exc:  # pragma: no cover
        print(f"Skipping figure generation: {exc}")


if __name__ == "__main__":
    main()
