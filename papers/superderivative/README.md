# The Superderivative as a First-Derivative Edge Operator

Paper (IEEEtran) plus a reference implementation of a first-derivative edge
operator derived from the superderivative. The operator measures *relative* change
rather than absolute grey-level change and is invariant under a global intensity
rescaling.

## Layout

```
.
├── main.tex                     # IEEEtran paper
├── main.pdf                     # compiled paper
├── ARTICLE_SPEC.md              # working specification
├── figures/                     # compiled figures
├── refs/                        # bibliography manifest
│   └── MANIFEST.json
├── data/                        # BSDS500 data and sample images
├── requirements.txt
└── src/                         # implementation and demos
```

See `src/README.md` for the full file-by-file description, the operator
definition and the benchmark tables.

## The operator

The 1-D superderivative is

```
SD(f(x)) = exp( x f'(x) / f(x) ).
```

In 2-D this gives the radial score `L = (x I_x + y I_y) / I`; the isotropic,
origin-free edge strength is the log-gradient magnitude

```
E(x,y) = ||grad log I|| = sqrt(I_x^2 + I_y^2) / I,
```

invariant under `I -> c I`. It is implemented in two steps: a classical derivative
kernel, then a pointwise division by `I`.

The parametric family `E_p = ||grad I|| / I^p` interpolates the absolute gradient
(`p = 0`) and the log-gradient (`p = 1`); only `p = 1` is intensity-invariant.

## Reproduce

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python src/demos/demo_edge_detection.py       # synthetic benchmark
.venv/bin/python src/demos/demo_edge_natural.py           # natural images
.venv/bin/python src/demos/demo_intensity_invariance.py    # invariance demo
```

The official BSDS500 boundary benchmark uses the unmodified Berkeley C++ matching
code; build it first and then run the evaluation (it caches per-image counts and
resumes if interrupted):

```bash
bash src/official_bsds/build.sh
.venv/bin/python src/official_bsds/eval_bsds500_official_code.py --workers 20
.venv/bin/python src/official_bsds/plot_summary.py
```

## Honest scope

The route from the superderivative to the relative-gradient operator is the
contribution, not the operator itself (`||grad log I||` already exists). On the
official BSDS500 benchmark the family has an interior optimum (`p ≈ 0.25`–`0.5`)
that beats Sobel/Prewitt/Scharr by a small margin under both a min-max and a robust
percentile normalisation; on the held-out test split the per-image OIS and AP
advantages replicate but the single-threshold ODS margin over Sobel/Prewitt does not.
The fully relative `p = 1` is never the best and its ranking depends on the
normalisation. Tuned Canny stays ahead. No learned detectors and no Canny-style
post-processing are used.

## Author

Alejandro Garnung Menendez
PhD Candidate, University of Oviedo
`uo269564@uniovi.es` / `garnungalejandro@gmail.com`
