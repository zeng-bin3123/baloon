// strata/core/native_dense.cpp — Native weight substitution and BF16 dense GEMV
// Target: Iluvatar BI-V100 (ivcore10)
//
// replace_native: iterates the WeightTable, reads each eligible tensor
// from the GGUF in its original quantization, uploads to device memory,
// and sets native_data + native_type on the WeightRef. The canonical
// data stays for verification; gemv_quantized prefers native when available.
//
// bf16_gemv: FP32-accumulated BF16 GEMV for the router and SSM gates.
// The router's BF16 weight measured 8.100e-03 error improvement — enough
// to change which expert is selected.

#include "strata/core/native_dense.hpp"
#include "strata/artifact/gguf_reader.hpp"
#include "strata/kernels/f16_bits.hpp"

#include <cstdio>
#include <cstring>
#include <vector>

#if defined(__CUDACC__) || defined(__ILUVATAR__)
#include <cuda_runtime.h>
#else
using cudaError_t = int;
static constexpr cudaError_t cudaSuccess = 0;
static cudaError_t cudaMalloc(void** p, size_t n) {
    *p = std::malloc(n); return *p ? cudaSuccess : 1;
}
static void cudaFree(void* p) { std::free(p); }
static cudaError_t cudaMemcpy(void* dst, const void* src, size_t n, int) {
    std::memcpy(dst, src, n); return cudaSuccess;
}
static constexpr int cudaMemcpyHostToDevice = 1;
#endif

namespace strata {

namespace {

// Eligibility filter for native replacement.
// Excludes router weights (ffn_gate_inp) and SSM weights (ssm_),
// and optionally PLE key tensors.
bool eligible(const std::string& name, bool include_ple_key) {
    if (name.find("ffn_gate_inp") != std::string::npos) return false;
    if (name.find("ssm_") != std::string::npos) return false;
    if (!include_ple_key && name.find("ple_key") != std::string::npos) return false;
    return true;
}

// Q8_1 scratch bytes for native MMVQ activation quantization.
// One Q8_1 block = 36 bytes (2×fp16 scale/sum + 32×int8 quants) per 32 elements.
static uint64_t q8_1_scratch_bytes(int64_t n_in) {
    int64_t n_blocks = (n_in + 31) / 32;
    return (uint64_t)n_blocks * 36;
}

}  // namespace

int replace_native(WeightTable& table, const std::string& gguf_path,
                   bool include_ple_key, std::string& err) {
    artifact::GgufReader reader(gguf_path);
    if (!reader.ok()) {
        err = "GgufReader: " + reader.error();
        return -1;
    }

    int count = 0;
    // Iterate over all tensors in the weight table
    // We need to work with the mutable map
    auto& all = const_cast<std::map<std::string, WeightRef>&>(table.all());

    for (auto& [name, ref] : all) {
        if (!eligible(name, include_ple_key)) continue;

        // Check if the GGUF has this tensor
        // Tensor names in GGUF may differ from pack names.
        // Common pattern: pack uses short names, GGUF uses full names.
        // Try the name as-is first.
        const artifact::GgufTensor* gt = reader.find(name);
        if (!gt) {
            // Try common prefix patterns
            std::string alt = "blk." + name;  // not found? skip
            gt = reader.find(alt);
            if (!gt) continue;
        }

        // Read tensor data to host
        std::vector<uint8_t> host_buf(gt->nbytes);
        if (!reader.read_tensor(gt->name, host_buf.data(), gt->nbytes)) {
            err = "replace_native: cannot read tensor " + gt->name;
            return -1;
        }

        // Allocate device memory for native data
        void* dev_native = nullptr;
        if (cudaMalloc(&dev_native, gt->nbytes) != cudaSuccess) {
            err = "replace_native: cudaMalloc failed for " + name;
            return -1;
        }

        // Upload to device
        if (cudaMemcpy(dev_native, host_buf.data(), gt->nbytes,
                       cudaMemcpyHostToDevice) != cudaSuccess) {
            cudaFree(dev_native);
            err = "replace_native: cudaMemcpy failed for " + name;
            return -1;
        }

        // Allocate Q8_1 scratch for activation quantization
        uint64_t q8_bytes = q8_1_scratch_bytes(ref.ne0);
        void* dev_q8 = nullptr;
        if (q8_bytes > 0) {
            if (cudaMalloc(&dev_q8, q8_bytes) != cudaSuccess) {
                cudaFree(dev_native);
                err = "replace_native: q8 scratch alloc failed for " + name;
                return -1;
            }
        }

        // Update the WeightRef
        ref.native_data = dev_native;
        ref.native_type = gt->type;
        ref.native_q8_data = dev_q8;
        ++count;
    }

    return count;
}

// ─── bf16_gemv ───────────────────────────────────────────────────────
// Host-side BF16 GEMV with FP32 accumulation.
// W: [n_out × n_in] row-major BF16 (stored as uint16_t, upper 16 bits of f32)
// x: [n_in] BF16
// out: [n_out] f32
//
// For GPU dispatch this would launch a kernel; this is the CPU fallback
// and reference implementation.
void bf16_gemv(const uint16_t* W, const uint16_t* x,
               float* out, int64_t n_out, int64_t n_in,
               void* stream) {
    (void)stream;  // GPU dispatch in a future unit

    for (int64_t i = 0; i < n_out; ++i) {
        float acc = 0.0f;
        const uint16_t* row = W + i * n_in;
        for (int64_t j = 0; j < n_in; ++j) {
            // BF16 → f32: shift left by 16 bits
            uint32_t w_bits = (uint32_t)row[j] << 16;
            uint32_t x_bits = (uint32_t)x[j] << 16;
            float w_f, x_f;
            std::memcpy(&w_f, &w_bits, 4);
            std::memcpy(&x_f, &x_bits, 4);
            acc += w_f * x_f;
        }
        out[i] = acc;
    }
}

}  // namespace strata

