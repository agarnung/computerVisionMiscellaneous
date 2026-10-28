"""Plot the official BSDS500 boundary-benchmark summary.

Reads ``src/bsds_results_official_code.json`` and writes
``figures/bsds_summary.png``.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
ROOT = HERE.parent.parent
RESULTS_PATH = ROOT / "src" / "bsds_results_official_code.json"
FIGURE_PATH = ROOT / "figures" / "bsds_summary.png"


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = json.loads(RESULTS_PATH.read_text())
    summary = data["summary"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    methods = sorted(summary.items(), key=lambda x: -x[1]["ods"])
    names = [m[0].replace("_", " ") for m in methods]
    ods = [m[1]["ods"] for m in methods]
    colors = plt.cm.viridis(np.linspace(0.25, 0.85, len(names)))

    axes[0].barh(names, ods, color=colors)
    axes[0].set_xlim(0, max(ods) * 1.25)
    axes[0].set_xlabel("ODS")
    axes[0].set_title("Official BSDS500 ODS per method")
    axes[0].invert_yaxis()
    for i, f in enumerate(ods):
        axes[0].text(f + 0.003, i, f"{f:.3f}", va="center", fontsize=8)

    parametric = sorted(
        ((k, v) for k, v in summary.items()
         if k.startswith("parametric_p") and k.split("_p")[1].replace(".", "").isdigit()),
        key=lambda x: float(x[0].split("_p")[1]),
    )
    ps = [float(k.split("_p")[1]) for k, _ in parametric]
    p_ods = [v["ods"] for _, v in parametric]
    p_ois = [v["ois"] for _, v in parametric]
    axes[1].plot(ps, p_ods, marker="o", linewidth=2, markersize=8,
                 label="ODS", color="C0")
    axes[1].plot(ps, p_ois, marker="s", linewidth=2, markersize=8,
                 label="OIS", color="C1")
    axes[1].set_xlabel("p")
    axes[1].set_ylabel("score")
    axes[1].set_title("Parametric family $E_p = \\|\\nabla I\\| / I^p$")
    axes[1].set_xticks(ps)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    plt.tight_layout()
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_PATH, dpi=150)
    plt.close(fig)
    print(f"Saved summary figure to {FIGURE_PATH}")


if __name__ == "__main__":
    main()
