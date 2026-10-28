#!/usr/bin/env bash
# Compile the official Berkeley BSDS500 matching code into a shared library.
#
# The C++ sources are taken unmodified from the BSDS500 benchmark release
# (data/BSR/bench/source).  The only added file is correspond_wrap.cc, which
# exposes matchEdgeMaps() through a plain C ABI callable from Python with ctypes.
# -fpermissive is needed because the 2003 sources use default arguments in
# friend declarations, which modern g++ rejects by default.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
SRC="$HERE/../../data/BSR/bench/source"

g++ -O3 -DNOBLAS -fpermissive -fPIC -shared \
    -I"$SRC" \
    -o "$HERE/libcorrespond.so" \
    "$HERE/correspond_wrap.cc" \
    "$SRC/csa.cc" "$SRC/kofn.cc" "$SRC/match.cc" \
    "$SRC/Exception.cc" "$SRC/Matrix.cc" "$SRC/Random.cc" \
    "$SRC/String.cc" "$SRC/Timer.cc"

echo "Built $HERE/libcorrespond.so"
