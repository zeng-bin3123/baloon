// prefill_gemm_dev.hpp - device-side dense prefill projection  C[M][N] = A[M][K] * W[N][K]^T  (row-major, f32 accumulate).
//
// Gemm owns (init) or borrows (init_external) one workspace and one stream. The workspace holds the activation matrix
// converted to the weight element type when the two differ (pfgemm::Gemm::scratch_bytes()). Every product is enqueued on the
// stream and never synchronises; sync() waits. Summation per output element runs in ascending k, accumulated in f32.
// ck() is the error gate shared by every call: it records the failing call's text and returns false on any CUDA error.
#pragma once
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

#include "baloon/prefill_gemm.hpp"

namespace baloon {
namespace pfdev {

using pfgemm::Elem;

// true iff e == cudaSuccess; otherwise stores "<what>: <cuda error string>" (per thread) and returns false.
bool ck(cudaError_t e, const char* what);
const char* ck_last();   // last failure text of this thread ("" when none)
void ck_clear();

class Gemm {
public:
    Gemm() = default;
    ~Gemm();
    Gemm(const Gemm&) = delete;
    Gemm& operator=(const Gemm&) = delete;

    // Owns a device workspace of `workspace_bytes` (may be 0) and a private stream on `device`.
    bool init(int device, uint64_t workspace_bytes);
    // Borrows `workspace` (16-byte aligned, may be null iff bytes == 0) and `stream` (0 = default); frees nothing on destruction.
    bool init_external(void* workspace, uint64_t bytes, cudaStream_t stream);

    // f32 x f32 -> f32.
    bool native(const float* a, const float* w, float* out, uint64_t m, uint64_t n, uint64_t k);
    // W holds bf16 bits. A is bf16 bits (a_type BF16) or f32 (a_type F32, converted into the workspace first).
    bool bf16(const void* a, Elem a_type, const uint16_t* w, float* out, uint64_t m, uint64_t n, uint64_t k);
    // W holds f16 bits. A is f16 bits (a_type F16) or f32 (a_type F32, converted into the workspace first).
    bool f16(const void* a, Elem a_type, const uint16_t* w, float* out, uint64_t m, uint64_t n, uint64_t k);

    bool sync();
    bool ready() const { return ready_; }
    bool owns() const { return owned_; }
    uint64_t workspace_bytes() const { return ws_bytes_; }
    cudaStream_t stream() const { return stream_; }

private:
    bool run(const void* a, Elem a_type, Elem w_type, const void* w, float* out, uint64_t m, uint64_t n, uint64_t k);
    void release();

    bool ready_ = false, owned_ = false;
    int device_ = 0;
    void* ws_ = nullptr;
    uint64_t ws_bytes_ = 0;
    cudaStream_t stream_ = 0;
};

}  // namespace pfdev
}  // namespace baloon
