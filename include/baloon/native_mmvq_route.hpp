// native_mmvq_route.hpp - dispatch table of the native MMVQ front end as pure constexpr host logic: block geometry of the
// sixteen ggml type ids the dispatcher knows, the support predicate, the IQ routing split, the weight byte count and the
// shape rule. A type is "native" when the Q8_1 kernels of native_mmvq.cu run it, "iq" when the call is forwarded to the
// i-quant kernels, "unsupported" otherwise.
#pragma once
#include <cstddef>
#include <cstdint>

namespace baloon {
namespace mmvq_route {

constexpr uint32_t kMaxCols = 8;

constexpr uint32_t block_elems(uint32_t t) {
    switch (t) {
        case 2: case 6: case 8: case 20: return 32;
        case 11: case 12: case 13: case 14: case 23: return 256;
        case 42: return 64;
        case 16: case 17: case 18: case 21: case 22: case 29: return 256;
        default: return 0;
    }
}
constexpr uint32_t block_bytes(uint32_t t) {
    switch (t) {
        case 2: return 18;   case 6: return 22;   case 8: return 34;   case 20: return 18;
        case 11: return 110; case 12: return 144; case 13: return 176; case 14: return 210; case 23: return 136;
        case 42: return 18;
        case 16: return 66;  case 17: return 74;  case 18: return 98;  case 21: return 110; case 22: return 82;
        case 29: return 56;
        default: return 0;
    }
}

// Every id with a block geometry is supported by the dispatcher (natively or through the IQ kernels).
constexpr bool supported(uint32_t t) { return block_elems(t) != 0 && block_bytes(t) != 0; }
// Types forwarded to the i-quant kernels.
constexpr bool iq_routed(uint32_t t) { return t == 16 || t == 17 || t == 18 || t == 21 || t == 22 || t == 29; }

enum class Route { Native, Iq, Unsupported };
constexpr Route route(uint32_t t) {
    return iq_routed(t) ? Route::Iq : (supported(t) ? Route::Native : Route::Unsupported);
}

// Bytes of an n_out x n_in matrix; 0 for an unknown type or a row that is not a whole number of blocks.
constexpr uint64_t weight_bytes(uint32_t t, uint64_t n_in, uint64_t n_out) {
    return (!supported(t) || n_in == 0 || n_in % block_elems(t) != 0) ? 0 : (n_in / block_elems(t)) * block_bytes(t) * n_out;
}
// Shape rule: positive n_in that is a whole number of weight blocks and of Q8_1 blocks, 1..8 columns.
constexpr bool valid_shape(uint64_t n_in, uint32_t ncols, uint32_t elems) {
    return n_in > 0 && elems > 0 && n_in % elems == 0 && n_in % 32 == 0 && ncols >= 1 && ncols <= kMaxCols;
}
// Small-K rule: fewer weight blocks per row than one iteration shares a block between 4 rows.
constexpr uint32_t rows_per_block(uint64_t blocks_per_row, uint64_t blocks_per_iteration) {
    return blocks_per_row < blocks_per_iteration ? 4u : 1u;
}
// Multi-column layout: the exact layout always uses 4 warps; the upstream layout uses 4 warps up to 4 columns, else 2.
constexpr uint32_t multi_warps(bool exact, uint32_t ncols) { return exact ? 4u : (ncols <= 4 ? 4u : 2u); }
// Shared partial buffer of the exact multi-column kernel: 3 warps x ncols x 4 rows x 32 lanes floats.
constexpr uint64_t multi_shared_bytes(uint32_t ncols) { return 3ull * ncols * 4 * 32 * 4; }

static_assert(supported(2) && supported(42) && supported(29) && !supported(0) && !supported(3) && !supported(99), "table");
static_assert(route(16) == Route::Iq && route(8) == Route::Native && route(7) == Route::Unsupported, "routes");
static_assert(multi_shared_bytes(8) <= 49152, "shared partials fit 48 KiB");

}  // namespace mmvq_route
}  // namespace baloon
