// Thin C wrapper around the official Berkeley BSDS500 boundary-matching code.
//
// This mirrors the mex wrapper in data/BSR/bench/source/correspondPixels.cc but
// exposes a plain C ABI usable from Python via ctypes.  The heavy lifting (the
// CSA optimal assignment and the edge-map matching) is exactly the official
// implementation in csa.cc / match.cc, unmodified.
//
// It reproduces the per-image loop of
// data/BSR/bench/benchmarks/evaluation_bdry_image.m for a single threshold:
//
//   for each ground-truth map i:
//       [match1, match2] = correspondPixels(bmap, gt_i, maxDist)
//       accP = accP | match1                 % machine matches (precision)
//       cntR = cntR + sum(match2 > 0)         % recall
//   cntP = sum(accP)
//
// The caller supplies sumR and sumP (the number of ground-truth and prediction
// pixels), which do not depend on the matching.

#include <cmath>
#include <vector>

#include "Matrix.hh"
#include "Random.hh"
#include "match.hh"

extern "C" {

// Seed the random generator used for the outlier connections of the official
// matching code.  The stock code seeds it from the clock, which makes a run
// non-reproducible; a fixed seed makes it reproducible while leaving the
// algorithm itself unchanged.
void correspond_seed(unsigned long seed)
{
    Random::rand.reseed((u_int64_t)seed);
}

void eval_bdry_threshold(
    const double* bmap,      // rows*cols, thresholded (and thinned) prediction
    const double* gts,       // n_gt*rows*cols, ground-truth boundary maps
    int n_gt, int rows, int cols,
    double maxDistFrac, double outlierCost,
    double* out)             // out[0] = cntR, out[1] = cntP
{
    const double idiag =
        std::sqrt(double(rows) * double(rows) + double(cols) * double(cols));
    const double maxDist = maxDistFrac * idiag;
    const double oc = outlierCost * maxDist;
    const long n = (long)rows * (long)cols;

    Matrix bm(rows, cols, const_cast<double*>(bmap));
    std::vector<char> accP(n, 0);
    double cntR = 0.0;

    for (int k = 0; k < n_gt; ++k) {
        Matrix gt(rows, cols, const_cast<double*>(gts) + (long)k * n);
        Matrix m1, m2;
        matchEdgeMaps(bm, gt, maxDist, oc, m1, m2);
        const double* d1 = m1.data();
        const double* d2 = m2.data();
        for (long i = 0; i < n; ++i) {
            if (d1[i] > 0) accP[i] = 1;
            if (d2[i] > 0) cntR += 1.0;
        }
    }

    double cntP = 0.0;
    for (long i = 0; i < n; ++i) {
        if (accP[i]) cntP += 1.0;
    }
    out[0] = cntR;
    out[1] = cntP;
}

}  // extern "C"
