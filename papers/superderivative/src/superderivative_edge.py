"""Superderivative and relative-gradient edge operators.

Implements the radial score L=(x I_x+y I_y)/I, the log-gradient magnitude
||grad I||/I, and the parametric family ||grad I||/I^p.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage


def _validate_image(image: np.ndarray) -> np.ndarray:
    """Return a float64, strictly-positive image array."""
    img = np.asarray(image, dtype=np.float64)
    if img.ndim != 2:
        raise ValueError("image must be 2-D greyscale")
    if np.any(img <= 0):
        raise ValueError("image must be strictly positive; add an offset if needed")
    return img


def _derivatives(image: np.ndarray, method: str = "sobel") -> tuple[np.ndarray, np.ndarray]:
    """Estimate I_x and I_y with a classical derivative kernel.

    Kernels are applied by convolution; the 2x2 Roberts mask is therefore flipped.
    """
    if method == "sobel":
        ix = ndimage.sobel(image, axis=1)
        iy = ndimage.sobel(image, axis=0)
    elif method == "scharr":
        kx = np.array([[-3, 0, 3], [-10, 0, 10], [-3, 0, 3]], dtype=np.float64)
        ky = kx.T
        ix = ndimage.convolve(image, kx)
        iy = ndimage.convolve(image, ky)
    elif method == "prewitt":
        ix = ndimage.prewitt(image, axis=1)
        iy = ndimage.prewitt(image, axis=0)
    elif method == "roberts":
        ix = ndimage.convolve(image, np.array([[1, 0], [0, -1]], dtype=np.float64))
        iy = ndimage.convolve(image, np.array([[0, 1], [-1, 0]], dtype=np.float64))
    else:
        raise ValueError(f"unknown derivative method: {method}")
    return ix, iy


def superderivative_score(image: np.ndarray, eps: float = 1e-6,
                          derivative: str = "sobel") -> np.ndarray:
    """Return the original radial score L(x,y) = (x I_x + y I_y) / I."""
    img = _validate_image(image)
    h, w = img.shape
    ys, xs = np.mgrid[:h, :w]
    ys = ys - (h - 1) / 2.0
    xs = xs - (w - 1) / 2.0
    ix, iy = _derivatives(img, method=derivative)
    return (xs * ix + ys * iy) / (img + eps)


def superderivative_magnitude(image: np.ndarray, eps: float = 1e-6,
                              derivative: str = "sobel") -> np.ndarray:
    """Log-gradient magnitude ||grad I|| / I (two-step discretisation)."""
    img = _validate_image(image)
    ix, iy = _derivatives(img, method=derivative)
    return np.sqrt(ix ** 2 + iy ** 2) / (img + eps)


def log_gradient_magnitude(image: np.ndarray, eps: float = 1e-6,
                           derivative: str = "sobel") -> np.ndarray:
    """Log-gradient magnitude ||grad log I|| (kernel applied to log I)."""
    img = _validate_image(image)
    ix, iy = _derivatives(np.log(img), method=derivative)
    return np.sqrt(ix ** 2 + iy ** 2)


def radial_score_magnitude(image: np.ndarray, eps: float = 1e-6,
                           derivative: str = "sobel") -> np.ndarray:
    """Radial log-derivative |L| / r = |partial_r log I|."""
    img = _validate_image(image)
    h, w = img.shape
    ys, xs = np.mgrid[:h, :w]
    ys = ys - (h - 1) / 2.0
    xs = xs - (w - 1) / 2.0
    r = np.sqrt(xs ** 2 + ys ** 2)
    L = superderivative_score(img, eps=eps, derivative=derivative)
    return np.abs(L) / (r + eps)


def parametric_magnitude(image: np.ndarray, p: float = 1.0,
                         eps: float = 1e-6,
                         derivative: str = "sobel") -> np.ndarray:
    """Parametric family E_p = ||grad I|| / I^p."""
    img = _validate_image(image)
    ix, iy = _derivatives(img, method=derivative)
    if p == 0:
        return np.sqrt(ix ** 2 + iy ** 2)
    return np.sqrt(ix ** 2 + iy ** 2) / (img ** p + eps)


def parametric_log_magnitude(image: np.ndarray, p: float = 1.0,
                            eps: float = 1e-6,
                            derivative: str = "sobel") -> np.ndarray:
    """Parametric family with the gradient estimated on log I."""
    img = _validate_image(image)
    ix, iy = _derivatives(np.log(img), method=derivative)
    if p == 1:
        return np.sqrt(ix ** 2 + iy ** 2)
    return np.sqrt(ix ** 2 + iy ** 2) * (img ** (1 - p))


def regularized_magnitude(image: np.ndarray, p: float = 1.0,
                          alpha: float = 0.05,
                          derivative: str = "sobel") -> np.ndarray:
    """Regularised relative gradient ||grad I|| / (I^p + alpha)."""
    img = _validate_image(image)
    ix, iy = _derivatives(img, method=derivative)
    return np.sqrt(ix ** 2 + iy ** 2) / (img ** p + alpha)


def local_mean_magnitude(image: np.ndarray, p: float = 1.0,
                         radius: int = 5,
                         derivative: str = "sobel") -> np.ndarray:
    """Relative gradient with a local-mean denominator."""
    img = _validate_image(image)
    size = 2 * radius + 1
    local_mean = ndimage.uniform_filter(img, size=size)
    ix, iy = _derivatives(img, method=derivative)
    return np.sqrt(ix ** 2 + iy ** 2) / (local_mean ** p + 1e-6)


def multiplicative_gradient(image: np.ndarray, eps: float = 1e-6,
                            derivative: str = "sobel") -> np.ndarray:
    """Multiplicative gradient exp(||grad I|| / I) - 1."""
    img = _validate_image(image)
    ix, iy = _derivatives(img, method=derivative)
    log_grad = np.sqrt(ix ** 2 + iy ** 2) / (img + eps)
    return np.expm1(log_grad)


def classical_magnitude(image: np.ndarray, method: str = "sobel") -> np.ndarray:
    """Classical absolute gradient magnitude for comparison."""
    img = np.asarray(image, dtype=np.float64)
    ix, iy = _derivatives(img, method=method)
    return np.sqrt(ix ** 2 + iy ** 2)


def canny_edges(image: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """Canny edge map (binary) using scikit-image."""
    from skimage import feature
    return feature.canny(image, sigma=sigma).astype(np.float64)


def canny_magnitude(image: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """Gaussian-smoothed Canny map for visualisation."""
    edges = canny_edges(image, sigma=sigma)
    return ndimage.gaussian_filter(edges.astype(np.float64), sigma=sigma)


def bsds_metrics(edge_map: np.ndarray, ground_truth: np.ndarray,
                 threshold: float = 0.1, tol_frac: float = 0.0075) -> dict:
    """Precision/recall/F1 under the BSDS distance-transform matching rule.

    Predictions are thinned and matched within ``tol_frac`` of the image diagonal.
    """
    from scipy.ndimage import distance_transform_edt
    from skimage.morphology import skeletonize

    gt = np.asarray(ground_truth) > 0
    pred = np.asarray(edge_map) >= threshold
    if pred.any():
        pred = skeletonize(pred)

    h, w = gt.shape
    tol = max(1.0, tol_frac * float(np.hypot(h, w)))

    if not pred.any() or not gt.any():
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    # distance from every pixel to the nearest ground-truth boundary pixel
    d_to_gt = distance_transform_edt(~gt)
    d_to_pred = distance_transform_edt(~pred)

    tp_pred = int((d_to_gt[pred] <= tol).sum())      # predictions that matched
    n_pred = int(pred.sum())
    tp_gt = int((d_to_pred[gt] <= tol).sum())        # gt pixels that were found
    n_gt = int(gt.sum())

    precision = tp_pred / n_pred if n_pred else 0.0
    recall = tp_gt / n_gt if n_gt else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)
    return {"precision": precision, "recall": recall, "f1": f1}


def bsds_best_f1(edge_map: np.ndarray, ground_truth: np.ndarray,
                 thresholds, tol_frac: float = 0.0075) -> dict:
    """Best :func:`bsds_metrics` F1 over a threshold sweep."""
    from scipy.ndimage import distance_transform_edt
    from skimage.morphology import skeletonize

    gt = np.asarray(ground_truth) > 0
    h, w = gt.shape
    tol = max(1.0, tol_frac * float(np.hypot(h, w)))
    if not gt.any():
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "threshold": 0.0}

    d_to_gt = distance_transform_edt(~gt)
    n_gt = int(gt.sum())
    best = {"precision": 0.0, "recall": 0.0, "f1": 0.0, "threshold": 0.0}

    for t in thresholds:
        pred = np.asarray(edge_map) >= t
        if not pred.any():
            continue
        pred = skeletonize(pred)
        n_pred = int(pred.sum())
        if not n_pred:
            continue
        tp_pred = int((d_to_gt[pred] <= tol).sum())
        d_to_pred = distance_transform_edt(~pred)
        tp_gt = int((d_to_pred[gt] <= tol).sum())
        prec = tp_pred / n_pred
        rec = tp_gt / n_gt
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        if f1 > best["f1"]:
            best = {"precision": prec, "recall": rec, "f1": f1,
                    "threshold": float(t)}
    return best


def edge_metrics(edge_map: np.ndarray, ground_truth: np.ndarray,
                 threshold: float = 0.1) -> dict:
    """Pixel-exact F1 (no thinning or localisation tolerance)."""
    binary = (edge_map >= threshold).astype(np.uint8)
    gt = (ground_truth > 0).astype(np.uint8)

    tp = float(np.sum((binary == 1) & (gt == 1)))
    fp = float(np.sum((binary == 1) & (gt == 0)))
    fn = float(np.sum((binary == 0) & (gt == 1)))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {"precision": precision, "recall": recall, "f1": f1}
