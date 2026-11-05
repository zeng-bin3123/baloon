// qsa_geometry_dev.hpp - device evaluation of the QSA step / paging / order-key formulas (src/ilu_qsa_geometry.cu).
// Used to cross-check the host arithmetic of qsa_cache_step.hpp against what the GPU computes.
// Return codes: 0 ok, -1 rejected by host validation (nothing launched), >0 a cudaError_t value.
#pragma once
#include <cstdint>

namespace baloon {
namespace qsa_geom_dev {

// out[4*i..4*i+3] = {pos, n_kv, n_bid, width} derived on the device from pos[i].
int step_eval(const int32_t* pos, int n, int top_k, int r, int32_t* out, void* stream);
// rows[i] = (page_table[cell/page_size] * n_head_kv + h) * page_size + cell % page_size, or -1 when the cell is outside
// the table. cells/heads are device arrays of length n.
int kv_row_eval(const int32_t* cells, const int32_t* heads, int n, const int32_t* page_table, int table_len,
                int n_head_kv, int page_size, int64_t* rows, void* stream);
// blocks[i] = min(cells[i]/r, n_kv/r); tail[i] = 1 when the cell lies in the incomplete tail (gets the +1e9 bias).
int cell_block_eval(const int32_t* cells, int n, int n_kv, int r, int32_t* blocks, int32_t* tail, void* stream);
// keys[i] = order key of the f32 bit pattern bits[i] after folding -0.0 onto +0.0.
int order_key_eval(const uint32_t* bits, int n, uint32_t* keys, void* stream);

}  // namespace qsa_geom_dev
}  // namespace baloon
