// prefill_gemm.hpp - Gemm: a validated description of one dense prefill projection  C[M][N] = A[M][K] * W[N][K]^T
// (row-major activations A, row-major weight rows W, f32 accumulate), with its scratch/workspace accounting and a host
// reference evaluation used as the oracle for device kernels.
#pragma once
#include <cstdint>
#include <cstddef>

namespace baloon {
namespace pfgemm {

enum class Elem : uint8_t { F32 = 0, F16 = 1, BF16 = 2 };

inline uint64_t elem_bytes(Elem e) { return e == Elem::F32 ? 4u : 2u; }

// Tile geometry: M, N rounded up to multiples of kTileM / kTileN, K to kTileK (zero padded).
constexpr uint64_t kTileM = 64, kTileN = 64, kTileK = 32;

inline uint64_t round_up(uint64_t v, uint64_t a) {
    const uint64_t r = v % a;
    return r == 0 ? v : v + (a - r);
}
inline bool mul_overflows(uint64_t a, uint64_t b) { return a != 0 && b > UINT64_MAX / a; }

struct Gemm {
    uint64_t m = 0, n = 0, k = 0;          // tokens, output rows, reduction width
    Elem a_type = Elem::F32, w_type = Elem::F32;

    bool valid() const {
        if (m == 0 || n == 0 || k == 0) return false;
        // every padded extent and byte count must fit in 64 bits
        if (round_up(m, kTileM) < m || round_up(n, kTileN) < n || round_up(k, kTileK) < k) return false;
        if (mul_overflows(round_up(m, kTileM), round_up(k, kTileK) * elem_bytes(a_type))) return false;
        if (mul_overflows(round_up(n, kTileN), round_up(k, kTileK) * elem_bytes(w_type))) return false;
        if (mul_overflows(m, n)) return false;
        return true;
    }
    uint64_t flops() const { return 2ull * m * n * k; }                       // multiply + add per term
    uint64_t out_bytes() const { return m * n * 4ull; }                        // f32 output
    // Scratch holding A converted to the weight type when they differ, padded to tiles; 0 when no conversion is needed.
    uint64_t scratch_bytes() const {
        if (a_type == w_type) return 0;
        return round_up(m, kTileM) * round_up(k, kTileK) * elem_bytes(w_type);
    }
    // Workspace: one f32 partial tile per output tile column block (split-k staging).
    uint64_t workspace_bytes() const { return (round_up(m, kTileM) / kTileM) * (round_up(n, kTileN) / kTileN) * kTileM * kTileN * 4ull; }
    bool fits(uint64_t scratch_cap, uint64_t ws_cap) const {
        return valid() && scratch_bytes() <= scratch_cap && workspace_bytes() <= ws_cap;
    }

    // Host reference: out[t*n + r] = sum_c a[t*k + c] * w[r*k + c], summed in ascending c order in f32.
    // Returns false (nothing written) when the description is invalid or a pointer is null.
    bool reference(const float* a, const float* w, float* out) const {
        if (!valid() || !a || !w || !out) return false;
        for (uint64_t t = 0; t < m; ++t)
            for (uint64_t r = 0; r < n; ++r) {
                float acc = 0.f;
                for (uint64_t c = 0; c < k; ++c) acc += a[t * k + c] * w[r * k + c];
                out[t * n + r] = acc;
            }
        return true;
    }
};

// Splits m tokens into chunks of at most `chunk`; the chunk spans tile every token exactly once.
inline uint64_t chunk_count(uint64_t m, uint64_t chunk) { return chunk == 0 ? 0 : m / chunk + (m % chunk != 0); }

}  // namespace pfgemm
}  // namespace baloon
