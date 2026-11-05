// ilu_qsa_geometry.cu - device-side evaluation of the QSA step record, paged KV row, cell->block map and f32 order key.
// The formulas are those of the decode kernels (ilu_qsa_step.cu); this file exposes them as standalone kernels so the
// unit test can compare the device result with literal expectations and with the host model.
#include <cuda_runtime.h>
#include "baloon/qsa_geometry_dev.hpp"

namespace baloon {
namespace qsa_geom_dev {
namespace {

inline unsigned grid_for(int n) { return (unsigned)((n + 255) / 256); }
int launch_status() { const cudaError_t e = cudaGetLastError(); return e == cudaSuccess ? 0 : (int)e; }

__global__ void step_kernel(const int32_t* pos, int n, int top_k, int r, int32_t* out) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int p = pos[i], nkv = p + 1, cap = top_k + r - 1;
    out[4 * i + 0] = p;
    out[4 * i + 1] = nkv;
    out[4 * i + 2] = nkv / r;
    out[4 * i + 3] = nkv < cap ? nkv : cap;
}

__global__ void kv_row_kernel(const int32_t* cells, const int32_t* heads, int n, const int32_t* table, int table_len,
                              int n_head_kv, int page_size, int64_t* rows) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int c = cells[i], h = heads[i];
    const int page_idx = c / page_size;
    if (c < 0 || h < 0 || h >= n_head_kv || page_idx >= table_len) { rows[i] = -1; return; }
    rows[i] = ((int64_t)table[page_idx] * n_head_kv + h) * page_size + (c % page_size);
}

__global__ void cell_block_kernel(const int32_t* cells, int n, int n_kv, int r, int32_t* blocks, int32_t* tail) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int j = cells[i], s = n_kv / r, b = j / r;
    blocks[i] = b < s ? b : s;
    tail[i] = (s * r <= j) ? 1 : 0;
}

__global__ void order_key_kernel(const uint32_t* bits, int n, uint32_t* keys) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    uint32_t b = bits[i];
    if (b == 0x80000000u) b = 0u;  // s + 0.0f folds -0.0 onto +0.0
    keys[i] = (b & 0x80000000u) ? ~b : (b | 0x80000000u);
}

}  // namespace

int step_eval(const int32_t* pos, int n, int top_k, int r, int32_t* out, void* stream) {
    if (!pos || !out || n < 1 || top_k < 1 || r < 1) return -1;
    step_kernel<<<grid_for(n), 256, 0, static_cast<cudaStream_t>(stream)>>>(pos, n, top_k, r, out);
    return launch_status();
}

int kv_row_eval(const int32_t* cells, const int32_t* heads, int n, const int32_t* page_table, int table_len,
                int n_head_kv, int page_size, int64_t* rows, void* stream) {
    if (!cells || !heads || !page_table || !rows || n < 1 || table_len < 1 || n_head_kv < 1 || page_size < 1) return -1;
    kv_row_kernel<<<grid_for(n), 256, 0, static_cast<cudaStream_t>(stream)>>>(cells, heads, n, page_table, table_len,
                                                                            n_head_kv, page_size, rows);
    return launch_status();
}

int cell_block_eval(const int32_t* cells, int n, int n_kv, int r, int32_t* blocks, int32_t* tail, void* stream) {
    if (!cells || !blocks || !tail || n < 1 || n_kv < 1 || r < 1) return -1;
    cell_block_kernel<<<grid_for(n), 256, 0, static_cast<cudaStream_t>(stream)>>>(cells, n, n_kv, r, blocks, tail);
    return launch_status();
}

int order_key_eval(const uint32_t* bits, int n, uint32_t* keys, void* stream) {
    if (!bits || !keys || n < 1) return -1;
    order_key_kernel<<<grid_for(n), 256, 0, static_cast<cudaStream_t>(stream)>>>(bits, n, keys);
    return launch_status();
}

}  // namespace qsa_geom_dev
}  // namespace baloon
