"""Synthetic checks of relative versus absolute change.

Run from the repository root: python src/demos/demo_relative_advantage.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
FIG_DIR = ROOT / "figures"
sys.path.insert(0, str(HERE.parent))

from superderivative_edge import (  # noqa: E402
    classical_magnitude,
    log_gradient_magnitude,
    parametric_magnitude,
    bsds_best_f1,
)
from demo_edge_detection import (  # noqa: E402
    SYNTH_THRESHOLDS,
    SYNTH_TOL_FRAC,
)

from scipy import ndimage  # noqa: E402


def _best_f1(emap: np.ndarray, gt: np.ndarray) -> float:
    emap = emap / max(float(emap.max()), 1e-12)
    return bsds_best_f1(emap, gt, SYNTH_THRESHOLDS,
                        tol_frac=SYNTH_TOL_FRAC)["f1"]


def _two_steps(background_a: float, step_a: float,
               background_b: float, step_b: float,
               size: int = 256) -> np.ndarray:
    """Build two vertical steps on different backgrounds."""
    img = np.full((size, size), background_a, dtype=np.float64)
    img[:, : size // 2] = background_a
    img[:, size // 4: size // 2] = background_a + step_a
    img[:, size // 2:] = background_b
    img[:, 3 * size // 4:] = background_b + step_b
    return np.clip(img, 1e-3, 1.0)


def contrast_vs_absolute() -> None:
    """Peak-response ratio at the two steps (relative vs absolute operator)."""
    print("Contrast ratio vs absolute step")
    print("  fixed absolute step 0.05, backgrounds 0.10 and 0.50")
    img = _two_steps(0.10, 0.05, 0.50, 0.05)
    e = parametric_magnitude(img, p=1.0)
    g = classical_magnitude(img, method="sobel")
    dark, bright = 256 // 4, 3 * 256 // 4
    print(f"    E_SD  bright/dark ratio = {e[:, bright].max() / e[:, dark].max():.3f}")
    print(f"    Sobel bright/dark ratio = {g[:, bright].max() / g[:, dark].max():.3f}")
    print("  fixed contrast ratio 0.20, backgrounds 0.10 and 0.50")
    img = _two_steps(0.10, 0.02, 0.50, 0.10)
    e = parametric_magnitude(img, p=1.0)
    g = classical_magnitude(img, method="sobel")
    print(f"    E_SD  bright/dark ratio = {e[:, bright].max() / e[:, dark].max():.3f}")
    print(f"    Sobel bright/dark ratio = {g[:, bright].max() / g[:, dark].max():.3f}")


def varying_illumination() -> None:
    """Smooth illumination ``I = R * L``; ground truth is the edge set of R."""
    print("\nSpatially varying illumination I = R * L")
    size = 256
    # Faint reflectance edges (contrast ratio 0.25) so the illumination
    # gradient competes with them.
    r = np.full((size, size), 0.30, dtype=np.float64)
    m = size // 4
    r[m: size - m, m: size - m] = 0.40
    gt = np.zeros_like(r)
    gt[m, m: size - m] = 1.0
    gt[size - m, m: size - m] = 1.0
    gt[m: size - m, m] = 1.0
    gt[m: size - m, size - m] = 1.0

    yy, xx = np.mgrid[:size, :size] / float(size)
    # Strong low-frequency illumination: a broad Gaussian bump plus a tilt.
    l = 0.5 + 1.5 * np.exp(-((yy - 0.5) ** 2 + (xx - 0.5) ** 2) / (2 * 0.30 ** 2)) \
        + 0.5 * xx
    l = l / l.mean()
    image = np.clip(r * l, 1e-3, 1.0)

    def pr(emap: np.ndarray) -> str:
        res = bsds_best_f1(emap / max(float(emap.max()), 1e-12), gt,
                           SYNTH_THRESHOLDS, tol_frac=SYNTH_TOL_FRAC)
        return f"F1={res['f1']:.3f} P={res['precision']:.3f} R={res['recall']:.3f}"

    print("  on I = R * L")
    for name, emap in [
        ("Sobel", classical_magnitude(image, method="sobel")),
        ("E_p p=0.25", parametric_magnitude(image, p=0.25)),
        ("E_p p=0.5", parametric_magnitude(image, p=0.5)),
        ("E_SD p=1", parametric_magnitude(image, p=1.0)),
    ]:
        print(f"    {name:12s} {pr(emap)}")
    # Reference: perfect illumination removal (divide by L).
    r_est = np.clip(image / l, 1e-3, 1.0)
    print("  on I/L (reference: illumination removed)")
    for name, emap in [
        ("Sobel", classical_magnitude(r_est, method="sobel")),
        ("E_SD p=1", parametric_magnitude(r_est, p=1.0)),
    ]:
        print(f"    {name:12s} {pr(emap)}")


def noise_metric() -> None:
    """Best F1 against ground truth as a function of additive noise."""
    print("\nNoise: best F1 vs ground truth")
    size = 256
    rng = np.random.default_rng(0)
    base = np.full((size, size), 0.25, dtype=np.float64)
    m = size // 4
    base[m: size - m, m: size - m] = 0.75
    gt = np.zeros_like(base)
    gt[m, m: size - m] = 1.0
    gt[size - m, m: size - m] = 1.0
    gt[m: size - m, m] = 1.0
    gt[m: size - m, size - m] = 1.0
    for sigma in (0.02, 0.05, 0.10):
        noisy = np.clip(base + rng.normal(0, sigma, base.shape), 1e-3, 1.0)
        row = [f"  sigma={sigma:.2f}"]
        for name, emap in [
            ("Sobel", classical_magnitude(noisy, method="sobel")),
            ("E_p p=0.25", parametric_magnitude(noisy, p=0.25)),
            ("E_p p=0.5", parametric_magnitude(noisy, p=0.5)),
            ("E_p p=1", parametric_magnitude(noisy, p=1.0)),
        ]:
            row.append(f"{name}={_best_f1(emap, gt):.3f}")
        print("  " + "  ".join(row))


def discretisation() -> None:
    """Two-step ``||kernel*I||/I`` vs log-image ``||kernel*log I||``."""
    print("\nDiscretisation on synthetic piecewise-constant images")
    size = 256
    img = np.full((size, size), 0.25, dtype=np.float64)
    m = size // 4
    img[m: size - m, m: size - m] = 0.75
    gt = np.zeros_like(img)
    gt[m, m: size - m] = 1.0
    gt[size - m, m: size - m] = 1.0
    gt[m: size - m, m] = 1.0
    gt[m: size - m, size - m] = 1.0
    two_step = parametric_magnitude(img, p=1.0)
    log_img = log_gradient_magnitude(img)
    print(f"  two-step  ||kernel*I||/I   best F1 = {_best_f1(two_step, gt):.3f}")
    print(f"  log-image ||kernel*log I|| best F1 = {_best_f1(log_img, gt):.3f}")


def main() -> None:
    contrast_vs_absolute()
    varying_illumination()
    noise_metric()
    discretisation()

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        size = 256
        r = np.full((size, size), 0.25, dtype=np.float64)
        m = size // 4
        r[m: size - m, m: size - m] = 0.75
        yy, xx = np.mgrid[:size, :size] / float(size)
        l = 0.6 + 0.8 * np.exp(-((yy - 0.5) ** 2 + (xx - 0.5) ** 2) / (2 * 0.35 ** 2)) + 0.3 * xx
        l = l / l.mean()
        image = np.clip(r * l, 1e-3, 1.0)

        fig, axes = plt.subplots(2, 3, figsize=(9, 6))
        panels = [
            ("Reflectance R", r, "gray"),
            ("Illumination L", l, "inferno"),
            ("I = R * L", image, "gray"),
            ("Sobel on I", classical_magnitude(image, method="sobel"), "magma"),
            ("E_SD on I", parametric_magnitude(image, p=1.0), "magma"),
            ("E_SD p=0.25 on I", parametric_magnitude(image, p=0.25), "magma"),
        ]
        for ax, (title, arr, cmap) in zip(axes.ravel(), panels):
            if title in ("Sobel on I", "E_SD on I", "E_SD p=0.25 on I"):
                arr = arr / max(float(np.percentile(arr, 99.0)), 1e-12)
                arr = np.clip(arr, 0, 1)
            ax.imshow(arr, cmap=cmap)
            ax.set_title(title, fontsize=9)
            ax.axis("off")
        fig.tight_layout()
        FIG_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(FIG_DIR / "relative_advantage.png", dpi=200)
        print("\nSaved figures/relative_advantage.png")
    except Exception as exc:  # pragma: no cover
        print(f"\nSkipping figure generation: {exc}")


if __name__ == "__main__":
    main()
