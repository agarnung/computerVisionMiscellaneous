"""Regenerate and verify the paper's frozen result files.

Run from the repository root: python src/freeze_results.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "demos"))
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402

from demo_edge_detection import (  # noqa: E402
    SYNTH_THRESHOLDS,
    SYNTH_TOL_FRAC,
    evaluate_on_image,
    make_checker_image,
    make_circle_image,
    make_gradient_image,
    make_square_image,
    make_triangle_image,
)

DATASETS = {
    "Circles": make_circle_image,
    "Square": make_square_image,
    "Checkerboard": make_checker_image,
    "Triangle": make_triangle_image,
    "Ramp": make_gradient_image,
}
METHODS = [
    "$E_{0.25}$",
    "Radial SD",
    "Sobel",
    "Scharr",
    "Prewitt",
    "Roberts",
    "Canny",
]

# Values as printed in Table III of main.tex, for verification.
TABLE_III = {
    "Circles": [0.986, 0.985, 0.986, 0.988, 0.984, 0.979, 0.980],
    "Square": [1.000, 1.000, 0.998, 0.998, 0.998, 0.998, 0.996],
    "Checkerboard": [1.000, 0.942, 0.998, 0.998, 0.998, 1.000, 0.998],
    "Triangle": [1.000, 1.000, 1.000, 1.000, 1.000, 0.871, 0.998],
    "Ramp": [0.996, 0.188, 0.996, 0.996, 0.996, 1.000, 0.996],
}

# Values as printed in Table IV of main.tex, for the stored BSDS summary.
TABLE_IV = {
    "canny": (0.679, 0.674, 0.613),
    "parametric_p0.25": (0.591, 0.643, 0.651),
    "prewitt": (0.587, 0.638, 0.647),
    "sobel": (0.586, 0.637, 0.647),
    "scharr": (0.582, 0.637, 0.645),
    "roberts": (0.570, 0.629, 0.631),
    "parametric_p0.5": (0.538, 0.643, 0.647),
    "parametric_p0.75": (0.483, 0.638, 0.641),
    "parametric_p1.0": (0.456, 0.592, 0.644),
    "parametric_p1.5": (0.433, 0.478, 0.644),
}

# Values as printed in Table VI of main.tex (validation vs held-out test split).
TABLE_TEST = {
    "canny": (0.679, 0.674, 0.613, 0.681, 0.674, 0.617),
    "parametric_p0.25": (0.591, 0.643, 0.651, 0.588, 0.637, 0.646),
    "prewitt": (0.587, 0.638, 0.647, 0.591, 0.633, 0.642),
    "sobel": (0.586, 0.637, 0.647, 0.589, 0.633, 0.642),
    "scharr": (0.582, 0.637, 0.645, 0.586, 0.631, 0.641),
    "parametric_p1.0": (0.456, 0.592, 0.644, 0.458, 0.599, 0.645),
}


def main() -> None:
    per_image: dict[str, dict[str, float]] = {}
    for name, make in DATASETS.items():
        image, gt = make()
        res = evaluate_on_image(image, gt)
        per_image[name] = {m: round(float(res[m]["f1"]), 3) for m in METHODS}

    out = {
        "dataset": "synthetic ground-truth edge images (256x256)",
        "protocol": "best F1 over a threshold sweep with one-pixel localisation tolerance",
        "n_thresholds": int(len(SYNTH_THRESHOLDS)),
        "threshold_min": float(SYNTH_THRESHOLDS[0]),
        "threshold_max": float(SYNTH_THRESHOLDS[-1]),
        "tol_frac": float(SYNTH_TOL_FRAC),
        "seed": None,
        "methods": METHODS,
        "per_image_f1": per_image,
    }
    (HERE / "synthetic_results.json").write_text(json.dumps(out, indent=2) + "\n")
    print("wrote src/synthetic_results.json\n")

    # Verify against Table III.
    ok = True
    print(f"{'Image':<15}" + "".join(f"{m[:10]:>12}" for m in METHODS))
    for name in DATASETS:
        got = [per_image[name][m] for m in METHODS]
        exp = TABLE_III[name]
        row = f"{name:<15}" + "".join(f"{v:>12.3f}" for v in got)
        if [round(v, 3) for v in exp] != [round(v, 3) for v in got]:
            row += "  <-- MISMATCH vs paper"
            ok = False
        print(row)
    print("Table III", "OK" if ok else "MISMATCH")

    # Verify the stored BSDS summary against Table IV.
    bsds = json.loads((HERE / "bsds_results_official_code.json").read_text())
    ok4 = True
    for key, (ods, ois, ap) in TABLE_IV.items():
        s = bsds["summary"][key]
        if (round(s["ods"], 3), round(s["ois"], 3), round(s["ap"], 3)) != (ods, ois, ap):
            print(f"  BSDS mismatch {key}: {s['ods']:.3f}/{s['ois']:.3f}/{s['ap']:.3f} "
                  f"vs paper {ods}/{ois}/{ap}")
            ok4 = False
    print("Table IV", "OK" if ok4 else "MISMATCH")

    # Verify the held-out test split against Table VI.
    test = json.loads(
        (HERE / "bsds_results_official_code_test.json").read_text())
    ok6 = True
    for key, (v_ods, v_ois, v_ap, t_ods, t_ois, t_ap) in TABLE_TEST.items():
        sv = bsds["summary"][key]
        st = test["summary"][key]
        got = (round(sv["ods"], 3), round(sv["ois"], 3), round(sv["ap"], 3),
               round(st["ods"], 3), round(st["ois"], 3), round(st["ap"], 3))
        if got != (v_ods, v_ois, v_ap, t_ods, t_ois, t_ap):
            print(f"  test mismatch {key}: {got} vs paper "
                  f"{(v_ods, v_ois, v_ap, t_ods, t_ois, t_ap)}")
            ok6 = False
    print("Table VI", "OK" if ok6 else "MISMATCH")

    if not (ok and ok4 and ok6):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
