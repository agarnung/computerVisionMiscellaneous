"""Official BSDS500 boundary benchmark (ODS/OIS/AP).

Uses the official Berkeley matching code and caches per-image counts under
``cache/<split>/<norm>/<method>/<id>.npz``.  See ``--help`` for options.
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import io, ndimage
from skimage.morphology import thin

HERE = Path(__file__).parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "src"))

from superderivative_edge import (  # noqa: E402
    canny_edges,
    classical_magnitude,
    log_gradient_magnitude,
    parametric_log_magnitude,
    parametric_magnitude,
    regularized_magnitude,
)

DATA_ROOT = ROOT / "data" / "BSR" / "BSDS500" / "data"
IMAGE_DIR = DATA_ROOT / "images"
GT_DIR = DATA_ROOT / "groundTruth"
CACHE_DIR = HERE / "cache"
RESULTS_PATH = ROOT / "src" / "bsds_results_official_code.json"
CSV_PATH = ROOT / "src" / "bsds_results_official_code.csv"

NTHRESH = 99
MAX_DIST_FRAC = 0.0075
OUTLIER_COST = 100.0
SEED = 42
THRESHOLDS = np.linspace(1 / (NTHRESH + 1), 1 - 1 / (NTHRESH + 1), NTHRESH)

CANNY_SIGMAS = [0.5, 1.0, 1.5, 2.0, 2.5]
SIGMA_GRID = [0.5, 1.0, 2.0]

# Fixed methods (no per-image parameter).  ``kind`` selects the edge map and
# ``sigma`` the optional Gaussian pre-smoothing.
METHOD_SPECS: list[tuple[str, str, dict, float]] = [
    ("sobel", "classical", {"method": "sobel"}, 0.0),
    ("scharr", "classical", {"method": "scharr"}, 0.0),
    ("prewitt", "classical", {"method": "prewitt"}, 0.0),
    ("roberts", "classical", {"method": "roberts"}, 0.0),
    ("parametric_p0.25", "parametric", {"p": 0.25}, 0.0),
    ("parametric_p0.5", "parametric", {"p": 0.5}, 0.0),
    ("parametric_p0.75", "parametric", {"p": 0.75}, 0.0),
    ("parametric_p1.0", "parametric", {"p": 1.0}, 0.0),
    ("parametric_p1.5", "parametric", {"p": 1.5}, 0.0),
    ("log_sobel", "log", {"method": "sobel"}, 0.0),
    ("parametric_log_p0.25", "parametric_log", {"p": 0.25}, 0.0),
    ("parametric_p0.5_alpha", "regularised", {"p": 0.5, "alpha": 0.05}, 0.0),
]

# Gaussian-smoothed classical kernel and family over the Canny sigma grid.
SWEEP_SPECS: list[tuple[str, str, dict, float]] = [
    (f"sobel_s{s}", "classical", {"method": "sobel"}, s)
    for s in SIGMA_GRID
] + [
    (f"parametric_p0.25_s{s}", "parametric", {"p": 0.25}, s)
    for s in SIGMA_GRID
]


def _image_ids(split: str) -> list[str]:
    ids = []
    for img_path in sorted((IMAGE_DIR / split).glob("*.jpg")):
        if (GT_DIR / split / (img_path.stem + ".mat")).exists():
            ids.append(img_path.stem)
    return ids


def _load_image(stem: str, split: str) -> np.ndarray:
    img = Image.open(IMAGE_DIR / split / (stem + ".jpg")).convert("L")
    return np.asarray(img, dtype=np.float64) / 255.0


def _load_ground_truth(stem: str, split: str) -> list[np.ndarray]:
    mat = io.loadmat(GT_DIR / split / (stem + ".mat"))
    gts = mat["groundTruth"]
    return [np.asarray(gts[0, i]["Boundaries"][0, 0]) > 0
            for i in range(gts.shape[1])]


def _normalize(map_: np.ndarray, norm: str = "minmax") -> np.ndarray:
    if norm == "percentile":
        hi = float(np.percentile(map_, 99.0))
        if hi > 0:
            return np.clip(map_ / hi, 0.0, 1.0)
        return np.zeros_like(map_)
    mn, mx = map_.min(), map_.max()
    if mx > mn:
        return (map_ - mn) / (mx - mn)
    return np.zeros_like(map_)


def _edge_map(img_pos: np.ndarray, kind: str, kwargs: dict,
               sigma: float) -> np.ndarray:
    """Compute one edge map; ``sigma`` > 0 applies Gaussian pre-smoothing."""
    if sigma > 0:
        img_pos = ndimage.gaussian_filter(img_pos, sigma=sigma)
    if kind == "classical":
        return classical_magnitude(img_pos, **kwargs)
    if kind == "log":
        return log_gradient_magnitude(img_pos, derivative=kwargs["method"])
    if kind == "parametric_log":
        return parametric_log_magnitude(
            img_pos, p=kwargs["p"], derivative=kwargs.get("method", "sobel"))
    if kind == "regularised":
        return regularized_magnitude(
            img_pos, p=kwargs["p"], alpha=kwargs["alpha"],
            derivative=kwargs.get("method", "sobel"))
    if kind == "parametric":
        return parametric_magnitude(
            img_pos, p=kwargs["p"], derivative=kwargs.get("method", "sobel"))
    raise ValueError(f"unknown kind: {kind}")


def _threshold_counts(emap: np.ndarray, gts: list[np.ndarray]) -> np.ndarray:
    """Per-threshold (cntR, sumR, cntP, sumP) as evaluation_bdry_image.m."""
    from official_matching import correspond_threshold

    sum_r = float(sum(int(g.sum()) for g in gts))
    gt_list = [g.astype(np.float64) for g in gts]

    # A binary prediction (e.g. Canny) is unchanged by thresholding, so the
    # whole threshold sweep is degenerate; compute it once and replicate.
    values = np.unique(emap)
    if values.size <= 2 and values.min() >= 0.0 and values.max() <= 1.0:
        bmap = thin(emap >= 0.5).astype(np.float64)
        cnt_r, cnt_p = correspond_threshold(
            bmap, gt_list, MAX_DIST_FRAC, OUTLIER_COST
        )
        return np.tile(
            [cnt_r, sum_r, cnt_p, float(bmap.sum())], (NTHRESH, 1)
        )

    counts = np.zeros((NTHRESH, 4), dtype=np.float64)
    for t, thr in enumerate(THRESHOLDS):
        bmap = thin(emap >= thr).astype(np.float64)
        cnt_r, cnt_p = correspond_threshold(
            bmap, gt_list, MAX_DIST_FRAC, OUTLIER_COST
        )
        counts[t] = (cnt_r, sum_r, cnt_p, float(bmap.sum()))
    return counts


def _best_f(counts: np.ndarray) -> float:
    """Best per-image F over thresholds (the OIS candidate)."""
    cnt_r, sum_r, cnt_p, sum_p = counts.T
    r = cnt_r / np.where(sum_r > 0, sum_r, 1.0)
    p = cnt_p / np.where(sum_p > 0, sum_p, 1.0)
    denom = p + r
    f = np.where(denom > 0, 2 * p * r / np.where(denom > 0, denom, 1.0), 0.0)
    return float(f.max())


def _cache_path(split: str, norm: str, name: str, stem: str) -> Path:
    return CACHE_DIR / split / norm / name / (stem + ".npz")


def _compute_method_image(stem: str, split: str, norm: str, name: str,
                          kind: str, kwargs: dict, sigma: float) -> str:
    """Compute and cache the per-threshold counts for one (method, image)."""
    out_path = _cache_path(split, norm, name, stem)
    if out_path.exists():
        return name
    out_path.parent.mkdir(parents=True, exist_ok=True)

    from official_matching import correspond_seed

    correspond_seed(SEED)

    img = _load_image(stem, split)
    img_pos = np.clip(img, 1e-3, 1.0)
    gts = _load_ground_truth(stem, split)

    emap = _edge_map(img_pos, kind, kwargs, sigma)
    counts = _threshold_counts(_normalize(emap, norm), gts)

    np.savez(out_path, counts=counts)
    return name


def _compute_canny_image(stem: str, split: str, norm: str,
                         sigma: float) -> None:
    out_path = _cache_path(split, norm, f"canny_s{sigma}", stem)
    if out_path.exists():
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)

    from official_matching import correspond_seed

    correspond_seed(SEED)

    img = _load_image(stem, split)
    gts = _load_ground_truth(stem, split)
    emap = canny_edges(img, sigma=sigma).astype(np.float64)
    counts = _threshold_counts(_normalize(emap, norm), gts)
    np.savez(out_path, counts=counts)


def _metrics(counts: np.ndarray) -> tuple[float, float]:
    """OIS and AP for one image (counts already pooled over human maps)."""
    cnt_r, sum_r, cnt_p, sum_p = counts.T
    r = cnt_r / np.where(sum_r > 0, sum_r, 1.0)
    p = cnt_p / np.where(sum_p > 0, sum_p, 1.0)
    denom = p + r
    f = np.where(denom > 0, 2 * p * r / np.where(denom > 0, denom, 1.0), 0.0)
    ois = float(f.max())

    ru, ind = np.unique(r, return_index=True)
    pu = p[ind]
    ri = np.arange(0.0, 1.0 + 0.005, 0.01)
    p_int = np.interp(ri, ru, pu)
    ap = float(np.sum(p_int) * 0.01)
    return ois, ap


def _summary_from_counts(per_image: list[np.ndarray]) -> dict:
    total = np.sum(np.stack(per_image), axis=0)
    cnt_r, sum_r, cnt_p, sum_p = total.T
    r = cnt_r / np.where(sum_r > 0, sum_r, 1.0)
    p = cnt_p / np.where(sum_p > 0, sum_p, 1.0)
    denom = p + r
    f = np.where(denom > 0, 2 * p * r / np.where(denom > 0, denom, 1.0), 0.0)
    best = int(np.argmax(f))
    ods = float(f[best])
    ois = float(np.mean([_metrics(c)[0] for c in per_image]))
    ap = float(np.mean([_metrics(c)[1] for c in per_image]))
    return {"ods": ods, "ois": ois, "ap": ap,
            "threshold": float(THRESHOLDS[best])}


def _worker(task: tuple[str, str, str, str, dict, float]) -> tuple[str, str]:
    stem, split, norm, name, kind, kwargs, sigma = task
    _compute_method_image(stem, split, norm, name, kind, kwargs, sigma)
    return name, stem


def _worker_canny(task: tuple[str, str, str, float]) -> tuple[str, str]:
    stem, split, norm, sigma = task
    _compute_canny_image(stem, split, norm, sigma)
    return "canny", stem


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--norm", choices=("minmax", "percentile"),
                        default="minmax")
    parser.add_argument("--sweep", action="store_true",
                        help="add Gaussian-smoothed kernels over the Canny sigma grid")
    parser.add_argument("--methods", default=None,
                        help="comma-separated subset of method names (default: all)")
    parser.add_argument("--workers", type=int, default=20)
    args = parser.parse_args()

    split, norm = args.split, args.norm
    ids = _image_ids(split)
    if not ids:
        raise RuntimeError(f"No BSDS500 images with ground truth found for {split}.")

    specs = list(METHOD_SPECS) + list(SWEEP_SPECS)
    if args.methods:
        wanted = {m.strip() for m in args.methods.split(",") if m.strip()}
        specs = [s for s in specs if s[0] in wanted]
    print(f"Found {len(ids)} {split} images with ground truth; "
          f"norm={norm}; {len(specs)} fixed methods"
          f"{' + sigma sweep' if args.sweep else ''}.")

    tasks = [(stem, split, norm, name, kind, kwargs, sigma)
             for name, kind, kwargs, sigma in specs
             for stem in ids]
    todo = [t for t in tasks if not _cache_path(split, norm, t[3], t[0]).exists()]
    print(f"Tasks: {len(tasks)}; already cached: {len(tasks) - len(todo)}")
    if todo:
        done = 0
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futures = [ex.submit(_worker, t) for t in todo]
            for fut in as_completed(futures):
                fut.result()
                done += 1
                if done % 20 == 0 or done == len(todo):
                    print(f"  {done}/{len(todo)} computed")

    # Canny, per sigma, then two aggregations: per-image best sigma (as the
    # paper) and a single global sigma chosen by ODS (no oracle).
    canny_tasks = [(stem, split, norm, sigma)
                   for sigma in CANNY_SIGMAS for stem in ids]
    canny_todo = [t for t in canny_tasks
                  if not _cache_path(split, norm, f"canny_s{t[3]}", t[0]).exists()]
    if canny_todo:
        done = 0
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futures = [ex.submit(_worker_canny, t) for t in canny_todo]
            for fut in as_completed(futures):
                fut.result()
                done += 1
                if done % 20 == 0 or done == len(canny_todo):
                    print(f"  canny {done}/{len(canny_todo)} computed")

    def per_image_counts(name: str) -> list[np.ndarray]:
        return [np.load(_cache_path(split, norm, name, stem))["counts"]
                for stem in ids]

    summary: dict[str, dict] = {}
    for name, _, _, _ in specs:
        summary[name] = _summary_from_counts(per_image_counts(name))

    # Canny (per-image best sigma, the paper's protocol).
    per_image_canny = []
    for stem in ids:
        best, best_f = None, -1.0
        for sigma in CANNY_SIGMAS:
            c = np.load(_cache_path(split, norm, f"canny_s{sigma}", stem))["counts"]
            f = _best_f(c)
            if f > best_f:
                best_f, best = f, c
        per_image_canny.append(best)
    summary["canny"] = _summary_from_counts(per_image_canny)

    # Canny (single global sigma chosen by ODS).
    ods_by_sigma = {}
    for sigma in CANNY_SIGMAS:
        pooled = np.sum(np.stack(
            [np.load(_cache_path(split, norm, f"canny_s{sigma}", stem))["counts"]
             for stem in ids]), axis=0)
        cnt_r, sum_r, cnt_p, sum_p = pooled.T
        r = cnt_r / np.where(sum_r > 0, sum_r, 1.0)
        p = cnt_p / np.where(sum_p > 0, sum_p, 1.0)
        denom = p + r
        f = np.where(denom > 0, 2 * p * r / np.where(denom > 0, denom, 1.0), 0.0)
        ods_by_sigma[sigma] = float(f.max())
    best_sigma = max(ods_by_sigma, key=ods_by_sigma.get)
    summary["canny_global_sigma"] = {
        **_summary_from_counts(per_image_counts(f"canny_s{best_sigma}")),
        "sigma": best_sigma,
    }

    output = {
        "dataset": f"BSDS500 {split} split ({len(ids)} images)",
        "protocol": "official BSDS500 boundary benchmark (ODS/OIS/AP)",
        "matching": "official Berkeley C++ code (libcorrespond.so)",
        "norm": norm,
        "n_gt": "human boundary maps used separately",
        "nthresh": NTHRESH,
        "max_dist_frac": MAX_DIST_FRAC,
        "outlier_cost": OUTLIER_COST,
        "thinpb": True,
        "seed": SEED,
        "thresholds": THRESHOLDS.tolist(),
        "canny_sigmas": CANNY_SIGMAS,
        "canny_sigma_ods": ods_by_sigma,
        "summary": summary,
    }
    suffix = f"_{split}" if split != "val" else ""
    suffix += f"_{norm}" if norm != "minmax" else ""
    results_path = RESULTS_PATH.with_name(
        RESULTS_PATH.stem + suffix + RESULTS_PATH.suffix)
    csv_path = CSV_PATH.with_name(CSV_PATH.stem + suffix + CSV_PATH.suffix)
    results_path.write_text(json.dumps(output, indent=2))
    with open(csv_path, "w") as f:
        f.write("method,ods,ois,ap,threshold\n")
        for name, s in sorted(summary.items(), key=lambda x: -x[1]["ods"]):
            f.write(f"{name},{s['ods']:.6f},{s['ois']:.6f},"
                    f"{s['ap']:.6f},{s['threshold']:.2f}\n")

    print(f"\n=== Official BSDS500 boundary benchmark ({split}, {norm}) ===")
    print(f"{'method':24s} {'ODS':>7s} {'OIS':>7s} {'AP':>7s} {'thr':>5s}")
    for name, s in sorted(summary.items(), key=lambda x: -x[1]["ods"]):
        print(f"{name:24s} {s['ods']:7.4f} {s['ois']:7.4f} {s['ap']:7.4f} "
              f"{s['threshold']:5.2f}")


if __name__ == "__main__":
    main()
