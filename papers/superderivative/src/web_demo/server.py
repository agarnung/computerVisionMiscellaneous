#!/usr/bin/env python3
"""Mini web demo for superderivative edge detection.

Run from the repository root: python3 src/web_demo/server.py
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import sys
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
SRC = HERE.parent
sys.path.insert(0, str(SRC))

from superderivative_edge import (
    parametric_magnitude,
    regularized_magnitude,
    local_mean_magnitude,
    superderivative_score,
    classical_magnitude,
    canny_edges,
)


# Valid user-tunable settings.
_KERNELS = {"sobel", "scharr", "prewitt"}
_MODES = {"parametric", "regularized", "local_mean", "radial", "radial_l"}


def _pil_to_b64(img: Image.Image) -> str:
    """Encode a PIL image as a base64 PNG data URI."""
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def _array_to_pil(arr: np.ndarray) -> Image.Image:
    """Convert a float64 [0,1] array to a PIL greyscale image."""
    img = np.clip(arr * 255.0, 0.0, 255.0).astype(np.uint8)
    return Image.fromarray(img, mode="L")


def _load_image(raw: bytes) -> np.ndarray:
    """Load raw bytes into a float64 [0,1] greyscale numpy array."""
    pil = Image.open(io.BytesIO(raw)).convert("L")
    return np.asarray(pil, dtype=np.float64) / 255.0


def _parse_params(params: dict | None) -> dict:
    """Validate and normalize frontend parameters."""
    params = params or {}

    kernel = str(params.get("kernel", "sobel")).lower()
    if kernel not in _KERNELS:
        kernel = "sobel"

    def _float(key: str, default: float, lo: float, hi: float) -> float:
        try:
            value = float(params.get(key, default))
        except (TypeError, ValueError):
            value = default
        return max(lo, min(hi, value))

    mode = str(params.get("mode", "parametric")).lower()
    if mode not in _MODES:
        mode = "parametric"

    return {
        "kernel": kernel,
        "mode": mode,
        "threshold": _float("threshold", 0.15, 0.0, 1.0),
        "canny_sigma": _float("canny_sigma", 1.5, 0.5, 3.0),
        "p": _float("p", 0.5, 0.0, 1.5),
        "alpha": _float("alpha", 0.05, 0.0, 0.5),
        "radius": int(_float("radius", 5, 1, 15)),
    }


def _normalize(emap: np.ndarray) -> np.ndarray:
    """Scale an edge-strength map to [0, 1]."""
    emap = np.nan_to_num(emap, nan=0.0, posinf=0.0, neginf=0.0)
    m = float(emap.max())
    if m > 0:
        return emap / m
    return emap


def _process(raw: bytes, params: dict | None = None) -> dict:
    """Process an uploaded image and return edge maps."""
    cfg = _parse_params(params)

    original = _load_image(raw)
    eps = 1e-3
    image_pos = np.clip(original, eps, 1.0)

    mode = cfg["mode"]
    if mode == "parametric":
        super_raw = parametric_magnitude(
            image_pos, p=cfg["p"], derivative=cfg["kernel"]
        )
        mode_label = f"Parametric (p={cfg['p']:.2f})"
    elif mode == "regularized":
        super_raw = regularized_magnitude(
            image_pos, p=cfg["p"], alpha=cfg["alpha"], derivative=cfg["kernel"]
        )
        mode_label = f"Regularized (p={cfg['p']:.2f}, alpha={cfg['alpha']:.2f})"
    elif mode == "local_mean":
        super_raw = local_mean_magnitude(
            image_pos, p=cfg["p"], radius=cfg["radius"], derivative=cfg["kernel"]
        )
        mode_label = f"Local mean (p={cfg['p']:.2f}, r={cfg['radius']})"
    elif mode == "radial":
        h, w = image_pos.shape
        ys, xs = np.mgrid[:h, :w]
        ys = ys - (h - 1) / 2.0
        xs = xs - (w - 1) / 2.0
        r = np.sqrt(xs ** 2 + ys ** 2)
        L = superderivative_score(image_pos, eps=eps, derivative=cfg["kernel"])
        super_raw = np.abs(L) / (r + eps)
        mode_label = "Radial SD |L|/r"
    else:  # radial_l
        L = superderivative_score(image_pos, eps=eps, derivative=cfg["kernel"])
        super_raw = np.abs(L)
        mode_label = "Radial score |L|"

    super_norm = _normalize(super_raw)
    super_binary = (super_norm >= cfg["threshold"]).astype(np.float64)

    detectors = {
        mode_label: super_norm,
        f"{mode_label} (binary)": super_binary,
        cfg["kernel"].capitalize(): classical_magnitude(original, method=cfg["kernel"]),
        "Roberts": classical_magnitude(original, method="roberts"),
        "Canny": canny_edges(original, sigma=cfg["canny_sigma"]),
    }

    payload = {"Original": {"image": _pil_to_b64(_array_to_pil(original))}}
    for name, emap in detectors.items():
        payload[name] = {"image": _pil_to_b64(_array_to_pil(_normalize(emap)))}
    return {"ok": True, "results": payload}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):  # silence per-request log
        pass

    def _send(self, code: int, body: bytes | str, ctype: str = "application/json") -> None:
        if isinstance(body, str):
            body = body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            body = (HERE / "index.html").read_bytes()
            self._send(200, body, "text/html; charset=utf-8")
        else:
            self._send(404, json.dumps({"ok": False, "error": "not found"}))

    def do_POST(self):
        try:
            if self.path == "/process":
                n = int(self.headers.get("Content-Length", 0))
                raw_body = self.rfile.read(n)
                ctype = self.headers.get("Content-Type", "")

                if ctype.startswith("application/json"):
                    body = json.loads(raw_body)
                    image_b64 = body.get("image", "")
                    if "," in image_b64:
                        image_b64 = image_b64.split(",", 1)[1]
                    raw = base64.b64decode(image_b64)
                    params = {k: body.get(k) for k in (
                        "kernel", "mode", "threshold", "canny_sigma",
                        "p", "alpha", "radius",
                    )}
                else:
                    # Legacy raw-file fallback.
                    raw = raw_body
                    params = {}

                self._send(200, json.dumps(_process(raw, params)))
                return
            self._send(404, json.dumps({"ok": False, "error": "not found"}))
        except Exception as exc:  # noqa: BLE001
            self._send(200, json.dumps({
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
                "trace": traceback.format_exc()[-800:],
            }))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args()
    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Demo at http://{args.host}:{args.port}")
    print("Drag an image, tune the controls, and click Process. Ctrl-C to stop.")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")


if __name__ == "__main__":
    main()
