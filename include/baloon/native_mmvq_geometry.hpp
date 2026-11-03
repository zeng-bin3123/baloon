// native_mmvq_geometry.hpp - launch geometry and byte accounting of the native MMVQ kernels: block byte sizes,
// iteration traits (blocks per iteration, small-K rule), thread->block maps, 4-rows-per-block grouping,
// weight and Q8_1 buffer sizes.
#pragma once
#include <cstddef>
#include <cstdint>

namespace baloon {
namespace mmvq_geom {

constexpr uint32_t WARP = 32;
constexpr uint32_t WARPS = 4;
constexpr uint32_t THREADS = WARPS * WARP;  // 128
constexpr uint32_t ROWS = 4;                // rows per block (one per warp)

enum class Fmt { Q3_K, Q4_K, Q5_K, Q6_K, IQ4_XS, Q4_0, IQ4_NL, Q5_0, Q8_0, Q2_0, Q8_1 };

struct Traits {
    uint32_t qk;     // block_elems
    uint32_t bytes;  // block size in bytes
    uint32_t qi;     // ints_per_block
    uint32_t vdr;    // values_per_thread
};

constexpr Traits traits(Fmt f) {
    switch (f) {
        case Fmt::Q3_K:   return {256, 110, 16, 1};
        case Fmt::Q4_K:   return {256, 144, 32, 2};
        case Fmt::Q5_K:   return {256, 176, 32, 2};
        case Fmt::Q6_K:   return {256, 210, 32, 1};
        case Fmt::IQ4_XS: return {256, 136, 32, 4};
        case Fmt::Q4_0:   return {32, 18, 4, 2};
        case Fmt::IQ4_NL: return {32, 18, 4, 2};
        case Fmt::Q5_0:   return {32, 22, 4, 2};
        case Fmt::Q8_0:   return {32, 34, 8, 2};
        case Fmt::Q2_0:   return {64, 18, 2, 1};
        case Fmt::Q8_1:   return {32, 36, 8, 1};
    }
    return {0, 0, 1, 0};
}

// Blocks consumed per loop iteration by one block of threads.
constexpr uint32_t blocks_per_iteration(Fmt f) { return traits(f).vdr * WARPS * WARP / traits(f).qi; }
// Small-K rule: fewer blocks per row than one iteration.
constexpr bool small_k(Fmt f, uint32_t n_in) { return n_in / traits(f).qk < blocks_per_iteration(f); }
constexpr uint32_t small_k_threshold_elems(Fmt f) { return blocks_per_iteration(f) * traits(f).qk; }

// Thread -> block-in-iteration maps (tid < 128).
constexpr uint32_t q3k_thread_block(uint32_t tid) { return tid / 16; }
constexpr uint32_t q6k_thread_block(uint32_t tid) { return tid / 32; }
constexpr uint32_t iq4xs_thread_block(uint32_t tid) { return tid / 8; }
constexpr uint32_t iq4xs_kqs(uint32_t tid) { return 4 * (tid % 8); }

// Row grouping: block blockIdx.x handles rows ROWS*blockIdx.x + warp.
constexpr uint32_t grid_blocks(uint32_t n_out) { return (n_out + ROWS - 1) / ROWS; }
constexpr uint32_t row_of(uint32_t block, uint32_t warp) { return block * ROWS + warp; }
constexpr uint32_t block_of_row(uint32_t row) { return row / ROWS; }
constexpr uint32_t warp_of_row(uint32_t row) { return row % ROWS; }

// Byte accounting.
constexpr uint64_t weight_bytes(uint32_t elems, uint32_t bytes, uint64_t n_in, uint64_t n_out) {
    return (n_in / elems) * bytes * n_out;
}
constexpr uint64_t weight_bytes(Fmt f, uint64_t n_in, uint64_t n_out) {
    return weight_bytes(traits(f).qk, traits(f).bytes, n_in, n_out);
}
constexpr uint64_t row_bytes(Fmt f, uint64_t n_in) { return (n_in / traits(f).qk) * traits(f).bytes; }
constexpr uint64_t row_byte_offset(Fmt f, uint64_t n_in, uint64_t row) { return row * row_bytes(f, n_in); }
// Q8_1 activation buffer for ncols vectors of n_in values.
constexpr uint64_t q8_1_bytes(uint64_t ncols, uint64_t n_in) { return ncols * (n_in / 32) * 36; }
constexpr uint64_t q8_1_blocks(uint64_t ncols, uint64_t n_in) { return ncols * (n_in / 32); }

// ---------------------------------------------------------------------------------------------------------
// Iteration traits (DIV, T, KBY, BPI): weight elements per block, threads per block, Q8_1 blocks per weight
// block, weight blocks per K iteration. Thread tid (0..127) handles block tid / T of each iteration and
// sub-index tid % T; T * BPI == 128 so one iteration covers all 128 threads exactly.
// ---------------------------------------------------------------------------------------------------------
struct IterTraits {
    uint32_t div;  // weight elements per block
    uint32_t t;    // threads per block
    uint32_t kby;  // Q8_1 blocks per weight block
    uint32_t bpi;  // weight blocks per K iteration
};

constexpr IterTraits kIterQ45K{256, 16, 8, 8};   // Q5_K / Q4_K
constexpr IterTraits kIterQ20{64, 2, 2, 64};     // Q2_0
constexpr IterTraits kIterQ3K{256, 16, 8, 8};    // Q3_K
constexpr IterTraits kIterQ6K{256, 32, 8, 4};    // Q6_K
constexpr IterTraits kIterIq4Xs{256, 8, 8, 16};  // IQ4_XS
constexpr IterTraits kIterSmall4{32, 2, 1, 64};  // Q4_0 / Q5_0 / IQ4_NL (QI = 4)
constexpr IterTraits kIterSmall8{32, 4, 1, 32};  // Q8_0 (QI = 8)

constexpr IterTraits iter_traits(Fmt f) {
    switch (f) {
        case Fmt::Q4_K: case Fmt::Q5_K: return kIterQ45K;
        case Fmt::Q3_K: return kIterQ3K;
        case Fmt::Q6_K: return kIterQ6K;
        case Fmt::IQ4_XS: return kIterIq4Xs;
        case Fmt::Q2_0: return kIterQ20;
        case Fmt::Q4_0: case Fmt::Q5_0: case Fmt::IQ4_NL: return kIterSmall4;
        case Fmt::Q8_0: return kIterSmall8;
        case Fmt::Q8_1: break;  // activation format: not a mat-vec weight format
    }
    return {0, 0, 0, 0};
}
constexpr bool has_iter_traits(Fmt f) { return f != Fmt::Q8_1; }

static_assert(kIterQ45K.t * kIterQ45K.bpi == THREADS && kIterQ20.t * kIterQ20.bpi == THREADS &&
              kIterQ3K.t * kIterQ3K.bpi == THREADS && kIterQ6K.t * kIterQ6K.bpi == THREADS &&
              kIterIq4Xs.t * kIterIq4Xs.bpi == THREADS && kIterSmall4.t * kIterSmall4.bpi == THREADS &&
              kIterSmall8.t * kIterSmall8.bpi == THREADS, "128 threads tile one iteration");
static_assert(kIterQ45K.kby == kIterQ45K.div / 32 && kIterQ20.kby == kIterQ20.div / 32 &&
              kIterQ3K.kby == kIterQ3K.div / 32 && kIterQ6K.kby == kIterQ6K.div / 32 &&
              kIterIq4Xs.kby == kIterIq4Xs.div / 32 && kIterSmall4.kby == kIterSmall4.div / 32 &&
              kIterSmall8.kby == kIterSmall8.div / 32, "KBY = DIV / 32");

// tid = T * slot + sub, unique per thread.
constexpr uint32_t tid_slot(uint32_t tid, uint32_t t) { return tid / t; }
constexpr uint32_t tid_sub(uint32_t tid, uint32_t t) { return tid % t; }
constexpr uint32_t tid_join(uint32_t slot, uint32_t sub, uint32_t t) { return t * slot + sub; }
// Sub-index (kqs) of the K-format / small formats; kqs .. kqs+VDR-1 stays inside the format's QI ints.
constexpr uint32_t kqs_q4k_q5k(uint32_t tid) { return 2 * (tid % 16); }
constexpr uint32_t kqs_iq4xs(uint32_t tid) { return 4 * (tid % 8); }
constexpr uint32_t kqs_q6k(uint32_t tid) { return tid % 32; }
constexpr uint32_t kqs_q3k(uint32_t tid) { return tid % 16; }
constexpr uint32_t kqs_q20(uint32_t tid) { return tid % 2; }
constexpr uint32_t kqs_small4(uint32_t tid) { return 2 * (tid % 2); }
constexpr uint32_t kqs_small8(uint32_t tid) { return 2 * (tid % 4); }

// Small-K: n_in / DIV < BPI picks 4 rows per block, otherwise 1 row per block.
constexpr uint32_t rows_per_block(Fmt f, uint32_t n_in) { return small_k(f, n_in) ? 4u : 1u; }
constexpr uint32_t grid_blocks_for(Fmt f, uint32_t n_in, uint32_t n_out) {
    return rows_per_block(f, n_in) == 4u ? grid_blocks(n_out) : n_out;
}
// Number of K iterations needed for a row (ceil over blocks).
constexpr uint32_t k_iterations(Fmt f, uint32_t n_in) {
    return (n_in / traits(f).qk + blocks_per_iteration(f) - 1) / blocks_per_iteration(f);
}

// ---------------------------------------------------------------------------------------------------------
// Multi-column addressing and shared-memory budgets.
// ---------------------------------------------------------------------------------------------------------
constexpr uint32_t kMaxNcols = 8;
constexpr uint64_t kSharedBudgetBytes = 49152;  // 48 KiB
// Column j of the activation starts at Q8_1 block j * (n_in / 32); output of column j, row r at j*n_out + r.
constexpr uint64_t column_block_start(uint64_t j, uint64_t n_in) { return j * (n_in / 32); }
constexpr uint64_t y_index(uint64_t j, uint64_t r, uint64_t n_out) { return j * n_out + r; }
// Partials array [NW-1][NCOLS][ROWS][32] floats.
constexpr uint64_t partial_bytes(uint64_t nw, uint64_t ncols, uint64_t rows) { return (nw - 1) * ncols * rows * 32 * 4; }
constexpr bool partial_fits_exact(uint32_t ncols) { return partial_bytes(4, ncols, 4) <= kSharedBudgetBytes; }
constexpr bool partial_fits_upstream(uint32_t ncols) { return partial_bytes(2, ncols, 2) <= 2048; }
// Upstream layout: ncols <= 4 uses 4 warps, otherwise 2; two rows per block.
constexpr uint32_t upstream_warps(uint32_t ncols) { return ncols <= 4 ? 4u : 2u; }
constexpr uint32_t upstream_rows = 2;
constexpr uint32_t upstream_grid(uint32_t n_out) { return (n_out + upstream_rows - 1) / upstream_rows; }

// ---------------------------------------------------------------------------------------------------------
// Affine corrections of the 32-element formats: d * (sumi * ds.x - c * ds.y); Q4_0 total 8 * sum(x), Q5_0 16.
// Each of the T = 2 threads subtracts c per call, so c * T equals the format's offset.
// ---------------------------------------------------------------------------------------------------------
constexpr uint32_t affine_per_thread(Fmt f) { return f == Fmt::Q4_0 ? 4u : f == Fmt::Q5_0 ? 8u : 0u; }
constexpr uint32_t affine_offset(Fmt f) { return affine_per_thread(f) * kIterSmall4.t; }
static_assert(affine_offset(Fmt::Q4_0) == 8 && affine_offset(Fmt::Q5_0) == 16, "affine offsets");

// ---------------------------------------------------------------------------------------------------------
// The Q8_1 quantiser: 256-thread blocks = 8 warps; element i's 32-block is blockIdx*8 + warp.
// ---------------------------------------------------------------------------------------------------------
constexpr uint32_t kQuantBlockThreads = 256;
constexpr uint32_t kQuantWarpsPerBlock = kQuantBlockThreads / WARP;
constexpr uint64_t quant_block_of_thread(uint64_t b, uint32_t t) { return (b * kQuantBlockThreads + t) / 32; }
constexpr uint64_t quant_element(uint64_t b, uint32_t t) { return b * kQuantBlockThreads + t; }
constexpr uint64_t quant_grid(uint64_t n) { return (n + kQuantBlockThreads - 1) / kQuantBlockThreads; }
// With n % 32 == 0 the early return `i >= n` removes only whole warps.
constexpr bool quant_active(uint64_t i, uint64_t n) { return i < n; }

// ---------------------------------------------------------------------------------------------------------
// Alignment helpers.
// ---------------------------------------------------------------------------------------------------------
// Q3_K has a 110-byte stride: every second block is only 2-byte aligned, so loads go through 16-bit halves.
constexpr uint64_t block_byte_offset(Fmt f, uint64_t b) { return b * traits(f).bytes; }
constexpr uint32_t load_int_b2(uint16_t lo, uint16_t hi) { return (uint32_t)lo | ((uint32_t)hi << 16); }
constexpr bool is_4byte_aligned(uint64_t off) { return off % 4 == 0; }

// ---------------------------------------------------------------------------------------------------------
// ggml type table: (elements per block, bytes per block) for the 16 supported type codes.
// ---------------------------------------------------------------------------------------------------------
struct BlockInfo {
    uint32_t elems;
    uint32_t bytes;
    bool ok;
};

constexpr BlockInfo block_info(uint32_t ty) {
    switch (ty) {
        case 2:  return {32, 18, true};
        case 6:  return {32, 22, true};
        case 8:  return {32, 34, true};
        case 20: return {32, 18, true};
        case 11: return {256, 110, true};
        case 12: return {256, 144, true};
        case 13: return {256, 176, true};
        case 14: return {256, 210, true};
        case 23: return {256, 136, true};
        case 42: return {64, 18, true};
        case 16: return {256, 66, true};
        case 17: return {256, 74, true};
        case 18: return {256, 98, true};
        case 21: return {256, 110, true};
        case 22: return {256, 82, true};
        case 29: return {256, 56, true};
        default: return {0, 0, false};
    }
}

constexpr uint32_t kSupportedTypes[16] = {2, 6, 8, 11, 12, 13, 14, 20, 23, 42, 16, 17, 18, 21, 22, 29};
constexpr size_t kNumSupportedTypes = sizeof(kSupportedTypes) / sizeof(kSupportedTypes[0]);

constexpr bool type_supported(uint32_t ty) { return block_info(ty).ok; }

// Row bytes for n elements of a table entry.
constexpr uint64_t table_row_bytes(uint32_t elems, uint32_t bytes, uint64_t n) { return n / elems * bytes; }
constexpr uint64_t table_row_bytes(uint32_t ty, uint64_t n) {
    return block_info(ty).ok ? table_row_bytes(block_info(ty).elems, block_info(ty).bytes, n) : 0;
}

// Host overflow guard: row <= MAX / n_out implies row * n_out <= MAX. Returns false (and leaves *out alone) when
// the product would exceed `max_bytes`.
constexpr bool weight_bytes_checked(uint64_t row, uint64_t n_out, uint64_t max_bytes, uint64_t* out) {
    if (n_out == 0) { *out = 0; return true; }
    if (row > max_bytes / n_out) return false;
    *out = row * n_out;
    return true;
}

}  // namespace mmvq_geom
}  // namespace baloon
