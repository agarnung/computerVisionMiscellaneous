# Paper — The Superderivative as a First-Derivative Edge Operator

This folder contains the LaTeX paper and a reference implementation for a new
edge-detection operator derived from the superderivative. The operator
measures *relative* change rather than absolute grey-level change, making it
invariant under global intensity rescaling.

## Files

| File | Description |
|------|-------------|
| `main.tex` | LaTeX article (IEEEtran). |
| `main.pdf` | Compiled article. |
| `src/superderivative_edge.py` | Core implementation: superderivative score, log-gradient edge-strength map, radial score, parametric/regularized/local-mean variants, classical gradient magnitudes, Canny, F1 metrics. |
| `src/demos/demo_edge_detection.py` | Reproducible demo on synthetic images with ground-truth edges. |
| `src/demos/demo_edge_natural.py` | Visual comparison on natural images. |
| `src/demos/demo_intensity_invariance.py` | Demonstrates intensity-scale invariance vs. Sobel. |
| `src/demos/demo_parametric_family.py` | Visual sweep of the parametric family $E_p$. |
| `src/demos/demo_noise_robustness.py` | Robustness to additive Gaussian noise. |
| `src/demos/demo_low_light.py` | Low-light edge-detection comparison. |
| `src/demos/demo_radial_comparison.py` | Compares radial score $L$, radial $|L|/r$ and isotropic $E_{\mathrm{SD}}$. |
| `src/demos/demo_texture_edges.py` | Edge detection on textured images. |
| `src/demos/demo_relative_advantage.py` | Controlled checks: contrast ratio vs absolute step, smooth illumination, noise F1 and the two discretisations. |
| `src/demos/demo_formulation_comparison.py` | Compares baseline, regularized and local-mean formulations (single-column figure). |
| `src/official_bsds/correspond_wrap.cc` | C wrapper exposing the official Berkeley matching code (`matchEdgeMaps`) through a C ABI. |
| `src/official_bsds/build.sh` | Compiles the unmodified BSDS500 C++ matching code into `libcorrespond.so`. |
| `src/official_bsds/official_matching.py` | ctypes binding for `libcorrespond.so`. |
| `src/official_bsds/eval_bsds500_official_code.py` | Official BSDS500 boundary benchmark (ODS/OIS/AP) on the 100 validation images, with a resumable per-image cache. |
| `src/official_bsds/plot_summary.py` | Plots `figures/bsds_summary.png` from the official results. |
| `src/stats_tests.py` | Paired $t$-tests over the per-image OIS reported in the paper. |
| `src/synthetic_results.json` | Frozen numbers of Table III (per-image best F1) produced by `freeze_results.py`. |
| `src/freeze_results.py` | Regenerates `synthetic_results.json` and checks both reported tables against the paper. |
| `src/web_demo/server.py` | Local web server to upload an image and compare parametric, regularized, local-mean and radial superderivative variants interactively. |
| `src/web_demo/index.html` | Frontend for the interactive demo. |
| `src/README.md` | This file. |

## Quick start

The paper lives at the repository root; the commands below are run from there.

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python src/demos/demo_edge_detection.py        # synthetic benchmark
.venv/bin/python src/demos/demo_edge_natural.py          # natural images
.venv/bin/python src/demos/demo_intensity_invariance.py  # invariance demo
.venv/bin/python src/demos/demo_relative_advantage.py    # controlled checks
.venv/bin/python src/web_demo/server.py                  # interactive demo
```

To reproduce the BSDS500 benchmark, first build the official matching library
and then run the evaluation (it caches per-image counts, so it resumes if
interrupted):

```bash
bash src/official_bsds/build.sh
.venv/bin/python src/official_bsds/eval_bsds500_official_code.py --workers 20
.venv/bin/python src/official_bsds/plot_summary.py
```

## The operator

The 1-D superderivative is

```
SD(f(x)) = exp( x f'(x) / f(x) )
```

In 2-D this gives the radial score

```
L(x,y) = (x I_x + y I_y) / I.
```

For an orientation-independent edge detector we use the log-gradient magnitude

```
E(x,y) = ||grad log I|| = sqrt(I_x^2 + I_y^2) / I,
```

which is invariant under `I -> c I`.  It can be implemented in two steps:

1. Compute `I_x`, `I_y` with any classical derivative kernel (Sobel, Scharr,
   Prewitt, ...).
2. Form `E = sqrt(I_x^2 + I_y^2) / I` pointwise.

## Parametric family

```
E_p(x,y) = ||grad I|| / I^p
```

- `p = 0` → classical absolute gradient (Sobel, Prewitt, ...)
- `p = 1` → superderivative / log-gradient
- `p = 0.25` → best empirical exponent on the official BSDS500 benchmark

## Robust variants

The raw log-gradient can be faint or unstable in dark regions.  The library
also provides:

- `regularized_magnitude(..., alpha=0.05)` — $E = \|\nabla I\|/(I^p+\alpha)$.
- `local_mean_magnitude(..., radius=5)` — denominator replaced by a local mean.
- `multiplicative_gradient(...)` — $\exp(\|\nabla I\|/I)-1$ (experimental;
  not used in paper figures because it is numerically unstable).

Run `demo_formulation_comparison.py` to compare them visually.

## BSDS500 benchmark results

Official BSDS500 boundary benchmark on the 100 validation images (ODS/OIS/AP;
five human maps used separately; 99 thresholds; thresholded prediction thinned to
one pixel; optimal-assignment matching within 0.0075 of the image diagonal;
outlier cost 100).  The matching code is the unmodified Berkeley C++ implementation
(`correspondPixels`), compiled by `src/official_bsds/build.sh`.  Our own detector
implementations, so the values are not directly comparable to published results for
other detectors. See the paper for numerical details.

`src/stats_tests.py` reproduces the paired $t$-tests and bootstrap 95\% intervals
quoted in the paper. Run with `--norm percentile` to check the ranking under a robust 
normalisation, in which the family still peaks at a small interior exponent but p = 1 
becomes indistinguishable from the kernels, and with `--split test` to recompute the 
per-image OIS tests on the 200 held-out images.

Two robustness checks are included in the evaluator: a percentile normalisation of the
edge maps, and Gaussian pre-smoothing of the kernels over the same σ grid as Canny
(`--sweep`).

## Synthetic benchmark results

See the paper for numerical details.

## Intensity-scale invariance

When an image is multiplied by a global constant `c`, the Sobel response
scales linearly with `c`, while the superderivative response stays unchanged
(up to numerical precision).  Run `demo_intensity_invariance.py` to see the
figure.

## Web demo

Run `python src/web_demo/server.py` and open `http://localhost:8000`.  Drag an
image onto the drop zone and click **Detect edges**.  The server returns the
original image and the edge-strength maps from Superderivative, Sobel, Scharr,
Prewitt, Roberts and Canny.

## Design choices

- **Sobel/Scharr/Prewitt kernels** for the derivative estimates, so the
  computational structure is identical to classical operators.
- **Pointwise division by `I`** to obtain the relative gradient.
- **Strictly positive input**; zero-valued images need a small offset.
- **Dependency-light.** NumPy, SciPy, scikit-image, matplotlib and Pillow.

## Reproducibility

Both tables of the paper are frozen to JSON so that a reader can check them
without re-running the whole pipeline:

- Table III (synthetic F1): `src/synthetic_results.json`, regenerated by
  `.venv/bin/python src/freeze_results.py`, which also checks it against the
  values printed in the paper.
- Table IV (official BSDS500 ODS/OIS/AP): `src/bsds_results_official_code.json`,
  produced by `src/official_bsds/eval_bsds500_official_code.py` and checked by
  `src/freeze_results.py`.

## Future work

- Non-maximum suppression and hysteresis thresholding (Canny-like
  post-processing).
- Multi-scale extensions.
- Applications to image enhancement, segmentation and feature detection.
