// verify_commit_dev.hpp - device-side commit vector and accepted-prefix kernels of the speculative verify window.
// Pure C++ (no CUDA headers); implemented in src/kernels/cuda/verify_commit_dev.cu.
// Return codes: 0 ok, -1 rejected by host validation (nothing launched), >0 a cudaError_t value.
#pragma once
#include <cstdint>

namespace baloon {
namespace vcommit {

constexpr int kMaxRows = 1024;   // commit vector rows handled by one launch

// out[0] = n_keep, out[1] = n_keep - 1, out[2 + t] = t < n_keep ? pos0 + t : -1 for t < last_t. n_keep is a DEVICE value,
// clamped to [0, last_t]; out holds 2 + last_t words. last_t in [0, kMaxRows].
int commit_rows(int32_t* out, const int32_t* n_keep_dev, int pos0, int last_t, void* stream);

// Accepted drafts a = length of the longest prefix with draft[i] == target[i] (i < n_draft); *n_keep = a + 1 (the verified
// token after the accepted prefix is always kept), *accepted = a. n_draft in [0, kMaxRows].
int accept_prefix(const int32_t* draft, const int32_t* target, int n_draft, int32_t* n_keep, int32_t* accepted, void* stream);

}  // namespace vcommit
}  // namespace baloon
