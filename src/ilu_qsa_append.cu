// ilu_qsa_append.cu - append one raw indexer key to the QSA indexer state (single block, 256 threads, 128 active dims).
// slot = pos % 4: slots 0..2 are stored in the 3-row raw tail; slot 3 completes block b = pos/4: mean of the four raw
// keys (tail rows then the incoming key), RMS-normalisation, NeoX rotation of dims 0..63 (pairs (d, d+32)) at
// rope position pos_base + 4*b, store to pooled row b and copy the spare row to row b+1.
// pos == 0 initialises pooled row 0 and the spare row from the incoming key (gather: four copies of cell zero,
// rotation position 0).
#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>
#include "baloon/ilu_qsa_kernels.hpp"
#include "ilu_qsa_common.cuh"

namespace baloon {
namespace ilu_qsa {
namespace {

constexpr int D = 128, NT = 256;

__global__ void __launch_bounds__(NT) pool_append_kernel(AppendArgs a) {
    __shared__ float sums[32];
    __shared__ float y[D];
    const int d = threadIdx.x;
    const int pos = a.pos;
    if (pos < 0 || pos >= a.max_cells) return;  // guard: only accepted positions are valid cell indices
    const int slot = pos & 3;
    const bool act = d < D;
    const float incoming = act ? a.key[d] : 0.f;

    if (slot < 3 && act) a.tail[slot * D + d] = incoming;
    if (slot != 3 && pos != 0) return;  // nothing to pool

    // gather: position 0 -> four copies of the incoming key, otherwise tail rows 0..2 then the incoming key
    float mean;
    if (pos == 0) mean = incoming;
    else mean = act ? (a.tail[d] + a.tail[D + d] + a.tail[2 * D + d] + incoming) * 0.25f : 0.f;
    const float ss = block_sum256(mean * mean, sums);
    const float v = mean * rsqrtf(ss / (float)D + a.eps);
    if (act) y[d] = v;
    __syncthreads();

    float outv = v;
    if (act && d < 64) {  // NeoX pairing (d, d+32)
        const long long rpos = pos == 0 ? 0 : a.pos_base + 4ll * (pos >> 2);
        const int pair = d & 31;
        const float ang = (float)rpos * powf(a.freq_step, (float)pair);
        float sn, cs;
        sincosf(ang, &sn, &cs);
        const float lo = y[pair], hi = y[pair + 32];
        outv = d < 32 ? lo * cs - hi * sn : hi * cs + lo * sn;
    }
    if (!act) return;
    if (pos == 0) {
        a.pooled[d] = outv;  // row 0
        a.dead[d] = outv;    // spare row
    } else {
        const int b = pos >> 2;
        a.pooled[(size_t)b * D + d] = outv;
        a.pooled[(size_t)(b + 1) * D + d] = a.dead[d];  // spare moves to row b+1
    }
}

}  // namespace

int pool_append(const AppendArgs& a, void* stream) {
    if (!a.key || !a.tail || !a.pooled || !a.dead || a.pos < 0 || a.max_cells < 1 || a.pos >= a.max_cells || a.pos_base < 0) return kInvalid;
    if (!native_qsa_indexer::params_ok((uint64_t)a.pos_base, (uint64_t)a.max_cells, (uint64_t)a.rows)) return kInvalid;
    if (!(a.eps >= 0.f) || !(a.freq_step > 0.f)) return kInvalid;
    const Region r[] = {region(a.key, 512, false), region(a.tail, native_qsa_indexer::kTailBytes, true),
                        region(a.pooled, (uint64_t)a.rows * 512, true), region(a.dead, 512, true)};
    if (!regions_ok(r, sizeof(r) / sizeof(r[0]))) return kInvalid;
    pool_append_kernel<<<1, NT, 0, static_cast<cudaStream_t>(stream)>>>(a);
    const cudaError_t e = cudaGetLastError();
    return e == cudaSuccess ? kOk : (int)e;
}

}  // namespace ilu_qsa
}  // namespace baloon
