"""Compare the radial score |L|/r with the isotropic map ||grad I|| / I.

Run from the repository root: python src/demos/demo_radial_comparison.py
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
from superderivative_edge import superderivative_score, superderivative_magnitude


def make_circle_image(size: int = 256) -> np.ndarray:
    """Concentric annuli; strong radial symmetry."""
    y, x = np.mgrid[:size, :size] - (size - 1) / 2.0
    r = np.sqrt(x ** 2 + y ** 2)
    image = np.zeros((size, size), dtype=np.float64)
    image[r < size * 0.2] = 0.3
    image[(r >= size * 0.35) & (r < size * 0.55)] = 0.6
    image[(r >= size * 0.7) & (r < size * 0.9)] = 0.9
    return image


def radial_score_L(image: np.ndarray, derivative: str = "sobel") -> np.ndarray:
    """Canonical radial score L = (x I_x + y I_y) / I."""
    return superderivative_score(image, eps=1e-6, derivative=derivative)


def radial_score_normalized(image: np.ndarray, derivative: str = "sobel") -> np.ndarray:
    """Normalised radial score |L|/r."""
    h, w = image.shape
    ys, xs = np.mgrid[:h, :w]
    ys = ys - (h - 1) / 2.0
    xs = xs - (w - 1) / 2.0
    r = np.sqrt(xs ** 2 + ys ** 2)
    L = superderivative_score(image, eps=1e-6, derivative=derivative)
    return np.abs(L) / (r + 1e-9)


def isotropic_sd(image: np.ndarray, derivative: str = "sobel") -> np.ndarray:
    """Isotropic superderivative magnitude ||grad I|| / I."""
    return superderivative_magnitude(image, eps=1e-6, derivative=derivative)


def to_gray(im: np.ndarray) -> np.ndarray:
    if im.ndim == 3:
        return color.rgb2gray(im)
    return im.astype(np.float64) / 255.0


def load_images(size: int = 256) -> dict[str, np.ndarray]:
    """Load one radially structured image and one general natural image."""
    images = {}
    candidates = {
        "retina": data.retina(),
        "astronaut": data.astronaut(),
    }
    # fraction of the short side kept, per image
    zooms = {"retina": 0.62}
    for name, im in candidates.items():
        im = to_gray(im)
        h, w = im.shape
        s = int(min(h, w) * zooms.get(name, 1.0))
        im = im[(h - s) // 2:(h + s) // 2, (w - s) // 2:(w + s) // 2]
        if s != size:
            from skimage.transform import resize
            im = resize(im, (size, size), anti_aliasing=True, mode="reflect")
        images[name] = np.clip(im, 0.0, 1.0)
    return images


def main() -> None:
    images = load_images(256)
    methods = [
        ("Radial score $L$", radial_score_L),
        ("Radial $|L|/r$", radial_score_normalized),
        ("Isotropic $E_{\\mathrm{SD}}$", isotropic_sd),
    ]

    try:
        import matplotlib.pyplot as plt

        n_images = len(images)
        n_cols = len(methods) + 1
        fig, axes = plt.subplots(n_images, n_cols,
                                 figsize=(2.2 * n_cols, 2.2 * n_images))

        for row, (img_name, img) in enumerate(images.items()):
            img_pos = np.clip(img, 1e-3, 1.0)
            axes[row, 0].imshow(img, cmap="gray", vmin=0, vmax=1)
            axes[row, 0].axis("off")
            if row == 0:
                axes[row, 0].set_title("Original", fontsize=9)
            axes[row, 0].text(-0.12, 0.5, img_name, transform=axes[row, 0].transAxes,
                              rotation=90, va="center", ha="center", fontsize=10,
                              fontweight="bold")

            for col, (name, fn) in enumerate(methods, start=1):
                emap = fn(img_pos, derivative="sobel")
                emap = np.nan_to_num(emap, nan=0.0, posinf=0.0, neginf=0.0)
                vmax = np.percentile(emap, 99.0)
                vmax = max(vmax, 1e-6)
                axes[row, col].imshow(emap, cmap="magma", vmin=0, vmax=vmax)
                axes[row, col].axis("off")
                if row == 0:
                    axes[row, col].set_title(name, fontsize=9)

        fig.tight_layout()
        FIG_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(FIG_DIR / "radial_comparison.png", dpi=200)
        print(f"Saved {FIG_DIR / "radial_comparison.png"!s}")
    except Exception as exc:  # pragma: no cover
        print(f"Skipping figure generation: {exc}")


if __name__ == "__main__":
    main()
