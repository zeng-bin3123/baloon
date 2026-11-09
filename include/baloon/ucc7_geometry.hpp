// ucc7_geometry.hpp - device evaluation of the closed-form facts of the QSA / short flash modules (step state, tail weight,
// float order key, tie quota). Implementation: src/kernels/cuda/ucc7_geometry.cu. Return codes: 0 ok, -1 rejected, >0 cudaError_t.
#pragma once
#include <cstdint>

namespace baloon {
namespace ucc7 {

constexpr int kStepFields = 6;   // n_kv, n_bid, width, tail_weight, has_tail, kv_head_of(pos % 24)
// out[i*6 + k] for position pos[i]: n_kv = pos+1, n_bid = n_kv / r, width = min(n_kv, top_k + r - 1),
// tail_weight = n_kv - r*n_bid, has_tail = tail_weight != 0, kv_head = (pos % 24) / 12.
int geometry_eval(const int32_t* pos, int n, int top_k, int r, int32_t* out, void* stream);
// keys[i] = order key of float bit pattern bits[i]: b < 2^31 ? b + 2^31 : 2^32 - 1 - b.
int order_key_eval(const uint32_t* bits, int n, uint32_t* keys, void* stream);
// quota[i] = clamp(budget[i] - before[i], 0, eq[i]).
int quota_eval(const int32_t* budget, const int32_t* before, const int32_t* eq, int n, int32_t* quota, void* stream);

}  // namespace ucc7
}  // namespace baloon
