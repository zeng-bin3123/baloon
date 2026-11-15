// ref/prefill/gemm.hpp — reference specification: Gemm descriptor for one dense prefill projection
// C[M][N] = A[M][K] * W[N][K]^T, row-major, f32 accumulate. Tile geometry: 64×64×32.
// Overflow-safe accessors: valid(), scratch_bytes(), workspace_bytes(), out_bytes(), flops(), fits().
#pragma once
#include <cstdint>
#include <cstddef>

namespace baloon {
namespace pfgemm {

enum class Elem : uint8_t { F32 = 0, F16 = 1, BF16 = 2 };
inline uint64_t elem_bytes(Elem e) { return e == Elem::F32 ? 4u : 2u; }

constexpr uint64_t kTileM = 64, kTileN = 64, kTileK = 32;
inline uint64_t round_up(uint64_t v, uint64_t a) { const uint64_t r = v % a; return r == 0 ? v : v + (a - r); }
inline bool mul_overflows(uint64_t a, uint64_t b) { return a != 0 && b > UINT64_MAX / a; }
inline bool mul_checked(uint64_t a, uint64_t b, uint64_t* out) {
    if (mul_overflows(a, b)) return false; *out = a * b; return true;
}

// Gemm: validated description of one projection pass. All byte/flop quantities fit in uint64_t when valid() is true.
struct Gemm {
    uint64_t m = 0, n = 0, k = 0;             // tokens (rows of A), output width (rows of W), reduction width
    Elem a_type = Elem::F32, w_type = Elem::F32;

    bool valid() const;
    uint64_t flops()          const { return 2ull * m * n * k; }
    uint64_t out_bytes()      const { return m * n * 4ull; }
    uint64_t scratch_bytes()  const;
    uint64_t workspace_bytes()const;
    bool fits(uint64_t scratch_cap, uint64_t ws_cap) const;
};

}  // namespace pfgemm
}  // namespace baloon
