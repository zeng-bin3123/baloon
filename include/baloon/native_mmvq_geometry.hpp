// native_mmvq_geometry.hpp - launch geometry and byte accounting of the native MMVQ kernels: block byte sizes,
// iteration traits (blocks per iteration, small-K rule), thread->block maps, 4-rows-per-block grouping,
// weight and Q8_1 buffer sizes.
#pragma once
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

}  // namespace mmvq_geom
}  // namespace baloon
