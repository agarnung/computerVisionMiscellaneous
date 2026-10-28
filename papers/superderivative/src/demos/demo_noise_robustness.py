"""Robustness of edge detectors to additive Gaussian noise.

Run from the repository root: python src/demos/demo_noise_robustness.py
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
from skimage.transform import resize

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from superderivative_edge import parametric_magnitude, classical_magnitude


def load_image(size: int = 256, zoom: float = 0.62) -> np.ndarray:
    """Centre crop of the retinal fundus, skipping the black aperture."""
    im = color.rgb2gray(data.retina())
    h, w = im.shape
    s = int(min(h, w) * zoom)
    im = im[(h - s) // 2:(h + s) // 2, (w - s) // 2:(w + s) // 2]
    im = resize(im, (size, size), anti_aliasing=True, mode="reflect")
    return np.clip(im, 0.0, 1.0)


def add_noise(image: np.ndarray, sigma: float) -> np.ndarray:
    noisy = image + np.random.normal(0, sigma, image.shape)
    return np.clip(noisy, 0.0, 1.0)


def main() -> None:
    np.random.seed(42)
    img = load_image()
    img = img / img.max()

    noise_levels = [0.0, 0.05, 0.10, 0.15, 0.20]
    methods = ["Superderivative (p=0.5)", "Sobel", "Scharr", "Prewitt"]

    try:
        import matplotlib.pyplot as plt

        n_levels = len(noise_levels)
        n_methods = len(methods)
        fig, axes = plt.subplots(n_levels, n_methods + 1,
                                 figsize=(2.0 * (n_methods + 1), 2.0 * n_levels))

        for row, sigma in enumerate(noise_levels):
            noisy = add_noise(img, sigma)
            axes[row, 0].imshow(noisy, cmap="gray", vmin=0, vmax=1)
            axes[row, 0].axis("off")
            if row == 0:
                axes[row, 0].set_title("Noisy image", fontsize=9)
            if row == 0:
                label = "clean"
            else:
                label = f"$\\sigma={sigma:.2f}$"
            axes[row, 0].text(-0.12, 0.5, label, transform=axes[row, 0].transAxes,
                              rotation=90, va="center", ha="center", fontsize=10,
                              fontweight="bold")

            for col, method in enumerate(methods, start=1):
                if method == "Superderivative (p=0.5)":
                    emap = parametric_magnitude(np.clip(noisy, 1e-3, 1.0), p=0.5, derivative="sobel")
                else:
                    emap = classical_magnitude(noisy, method=method.lower())
                # Normalise on the 99th percentile, not the maximum. The
                # maximum is the most noise-sensitive statistic available,
                # and for a relative operator a single pixel in the dark
                # macula where I ~ 1e-3 sets it, crushing the whole map to
                # black. The classical operators never divide by I, so their
                # maximum is well behaved and only the proposed one is
                # penalised -- a display artefact, not a property of the
                # filter. The rest of the paper's figures already use the
                # 99th percentile.
                vmax = max(float(np.percentile(emap, 99.0)), 1e-9)
                axes[row, col].imshow(np.clip(emap / vmax, 0, 1),
                                      cmap="magma", vmin=0, vmax=1)
                axes[row, col].axis("off")
                if row == 0:
                    axes[row, col].set_title(method, fontsize=9)

        fig.tight_layout()
        FIG_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(FIG_DIR / "noise_robustness.png", dpi=200)
        print(f"Saved {FIG_DIR / "noise_robustness.png"!s}")
    except Exception as exc:  # pragma: no cover
        print(f"Skipping figure generation: {exc}")


if __name__ == "__main__":
    main()
