"""ctypes binding for the official Berkeley BSDS500 matching code."""
from __future__ import annotations

import ctypes
from pathlib import Path

import numpy as np

_LIB_PATH = Path(__file__).parent / "libcorrespond.so"

if not _LIB_PATH.exists():
    raise ImportError(
        f"{_LIB_PATH} not found; run src/official_bsds/build.sh first."
    )

_lib = ctypes.CDLL(str(_LIB_PATH))
_lib.correspond_seed.restype = None
_lib.correspond_seed.argtypes = [ctypes.c_ulong]
_lib.eval_bdry_threshold.restype = None
_lib.eval_bdry_threshold.argtypes = [
    ctypes.POINTER(ctypes.c_double),   # bmap
    ctypes.POINTER(ctypes.c_double),   # gts
    ctypes.c_int,                     # n_gt
    ctypes.c_int,                     # rows
    ctypes.c_int,                     # cols
    ctypes.c_double,                  # maxDistFrac
    ctypes.c_double,                  # outlierCost
    ctypes.POINTER(ctypes.c_double),   # out
]


def correspond_seed(seed: int = 42) -> None:
    """Seed the official matching code's random outlier selection."""
    _lib.correspond_seed(ctypes.c_ulong(seed))


def correspond_threshold(
    bmap: np.ndarray,
    gts: list[np.ndarray],
    max_dist_frac: float = 0.0075,
    outlier_cost: float = 100.0,
) -> tuple[float, float]:
    """Match a thresholded prediction to each human map."""
    bmap = np.ascontiguousarray(bmap, dtype=np.float64)
    rows, cols = bmap.shape
    gts_arr = np.ascontiguousarray(
        np.stack([np.asarray(g, dtype=np.float64) for g in gts]), dtype=np.float64
    )
    n_gt = gts_arr.shape[0]
    out = np.zeros(2, dtype=np.float64)
    _lib.eval_bdry_threshold(
        bmap.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        gts_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        n_gt, rows, cols, max_dist_frac, outlier_cost,
        out.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
    )
    return float(out[0]), float(out[1])
