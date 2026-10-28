"""Compare superderivative formulations: baseline, regularised and local-mean.

Run from the repository root: python src/demos/demo_formulation_comparison.py
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
    superderivative_magnitude,
    parametric_magnitude,
    regularized_magnitude,
    local_mean_magnitude,
    classical_magnitude,
)


def to_gray(im: np.ndarray) -> np.ndarray:
    if im.ndim == 3:
        return color.rgb2gray(im)
    return im.astype(np.float64) / 255.0


def load_images(size: int = 256) -> dict[str, np.ndarray]:
    images = {}
    # First image: a real Moon against a dark sky, so the unregularised
    # log-gradient has to contend with a near-zero denominator.
    moon_path = Path(__file__).resolve().parents[2] / "data" / "moon_sky.jpg"
    from skimage import io
    candidates = {
        "moon": io.imread(str(moon_path)),
        "astronaut": data.astronaut(),
        "camera": data.camera(),
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
    images = {k: images[k] for k in ["moon", "astronaut"]}

    # One row per image, one column per formulation, every column named.
    # The previous layout split the five formulations over two stacked 2x3
    # blocks with hspace=0.05, which clipped the second row's titles: three
    # of the five variants went to print unlabelled.
    variants = [
        ("Original", None),
        ("Sobel", lambda img: classical_magnitude(img, method="sobel")),
        ("SD $p=1$", lambda img: superderivative_magnitude(img, eps=1e-3)),
        ("SD $p=0.5$", lambda img: parametric_magnitude(img, p=0.5, eps=1e-3)),
        ("Reg. $\\alpha=0.05$",
         lambda img: regularized_magnitude(img, p=1.0, alpha=0.05)),
        ("Local mean", lambda img: local_mean_magnitude(img, p=1.0, radius=5)),
    ]

    try:
        import matplotlib.pyplot as plt
        from matplotlib import gridspec

        n_rows, n_cols = len(images), len(variants)
        fig = plt.figure(figsize=(2.0 * n_cols, 2.0 * n_rows + 0.35))
        gs = gridspec.GridSpec(n_rows, n_cols, figure=fig,
                               wspace=0.03, hspace=0.06)

        for row, (img_name, img) in enumerate(images.items()):
            img_pos = np.clip(img, 1e-3, 1.0)
            for col, (name, fn) in enumerate(variants):
                ax = fig.add_subplot(gs[row, col])
                if fn is None:
                    ax.imshow(img, cmap="gray", vmin=0.0, vmax=1.0)
                else:
                    arr = np.nan_to_num(fn(img_pos), nan=0.0,
                                        posinf=0.0, neginf=0.0)
                    # 99th percentile, not the maximum: one pixel in the dark
                    # sky sets the maximum for the unregularised variants and
                    # renders the rest of the panel black.
                    vmax = max(float(np.percentile(arr, 99.0)), 1e-9)
                    ax.imshow(np.clip(arr / vmax, 0, 1), cmap="magma",
                              vmin=0.0, vmax=1.0)
                ax.set_xticks([]); ax.set_yticks([])
                for sp in ax.spines.values():
                    sp.set_visible(False)
                if row == 0:
                    ax.set_title(name, fontsize=9, pad=3)
                if col == 0:
                    ax.set_ylabel(img_name, fontsize=9, fontweight="bold",
                                  labelpad=3)

        fig.tight_layout()
        FIG_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(FIG_DIR / "formulation_comparison.png", dpi=200)
        print(f"Saved {FIG_DIR / 'formulation_comparison.png'!s}")
    except Exception as exc:  # pragma: no cover
        print(f"Skipping figure generation: {exc}")


if __name__ == "__main__":
    main()
