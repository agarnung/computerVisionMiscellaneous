#!/usr/bin/env bash
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
echo "START $(date)"
.venv/bin/python src/official_bsds/eval_bsds500_official_code.py --split val --norm minmax --workers 22
echo "DONE $(date)"
