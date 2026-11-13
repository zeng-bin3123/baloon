// native_gr_norm.hpp - reference interface for the native weighted RMS norm.
// native_gr_rms_norm_weighted: one block per row, weighted by a per-element gamma.
#pragma once
#include <cstdint>

namespace baloon {
namespace ref_ngr_norm {

// Apply weighted RMS norm: out[r*n+c] = rsqrt(mean(x^2) + eps) * x[r*n+c] * gamma[r*n+c].
// n_rows rows, n_cols columns; each row is normalised independently.
// eps must be >= 0 and finite; all pointers must be non-null and 4-byte aligned.
void native_gr_rms_norm_weighted(const float* input, const float* gamma, float* output,
                                  int n_rows, int n_cols, float eps);

}  // namespace ref_ngr_norm
}  // namespace baloon

