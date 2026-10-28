#!/usr/bin/env bash
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
echo "START $(date)"
.venv/bin/python src/official_bsds/eval_bsds500_official_code.py --split test --methods sobel,scharr,prewitt,parametric_p0.25,parametric_p0.5,parametric_p1.0 --norm minmax --workers 22
echo "DONE $(date)"
