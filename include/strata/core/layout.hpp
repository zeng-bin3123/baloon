#pragma once
// strata/core/layout.hpp — Weight tensor shape and engine-form validation
// Target: Iluvatar BI-V100 (ivcore10)
//
// Validates that every loaded weight tensor matches the shapes the kernels
// expect. A shape mismatch at load time produces a clear message; a shape
// mismatch at kernel time produces a plausible wrong answer.
//
// Two layer families:
//   QSA (quantized sparse attention) at indices l where l % 4 == 3
//     → 12 layers at {3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47}
//   GDN (gated delta network / SSM) at all other indices
//     → 36 layers
//
// The 1-D kind check is not cosmetic: Bf16InF32 stores 2 bytes/element
// while F32 stores 4. A consumer casting to const float* would read 2×
// the length, silently producing garbage.

#include "strata/core/types_fwd.hpp"
#include "strata/core/weights.hpp"
#include <cstdint>
#include <string>

namespace strata {

// ─── Layer type predicate ────────────────────────────────────────────
// Returns true if the layer at index `l` is a QSA layer.
// QSA layers are placed every 4th layer starting at 3.
inline bool is_qsa_layer(const Geometry& g, int64_t l) {
    (void)g;
    return l >= 0 && l % 4 == 3;
}

// ─── check_layer ─────────────────────────────────────────────────────
// Validates that all required weight tensors for layer `l` are present
// in `table` with the correct shapes and kinds.
// Returns true on success, false with `err` set on mismatch.
bool check_layer(const WeightTable& table, const Geometry& g,
                 int64_t layer, std::string& err);

// ─── check_all ───────────────────────────────────────────────────────
// Validates all n_layers layers and checks the QSA/GDN split.
// Returns true on success.
bool check_all(const WeightTable& table, const Geometry& g,
               std::string& err);

}  // namespace strata

