// strata/core/native_head.cpp — IQ-format head weight loader and embedding
// Target: Iluvatar BI-V100 (ivcore10)
//
// NativeHead: loads the output.weight tensor from a GGUF in its native IQ
// quantization. The IQ kernel (from kernels/iq_kernels.hpp) uses a scratch
// buffer for its partial sums.
//
// NativeEmbed: loads the token_embd.weight tensor from a GGUF, keeps both
// host and device copies for single-token and batched gather.

#include "strata/core/native_head.hpp"
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
static cudaError_t cudaMemcpy(void* dst, const void* src, size_t n, int kind) {
    std::memcpy(dst, src, n); return cudaSuccess;
}
static constexpr int cudaMemcpyHostToDevice = 1;
static constexpr int cudaMemcpyDeviceToHost = 2;
#endif

namespace strata {

namespace {

// IQ kernel scratch bytes: enough for partial sums and Q8_1 activation.
// Conservative estimate: n_vocab f32 partials + Q8_1 blocks for n_embd.
uint64_t iq_scratch_bytes(int64_t n_vocab, int64_t n_embd) {
    uint64_t partials = (uint64_t)n_vocab * sizeof(float);
    int64_t q8_blocks = (n_embd + 31) / 32;
    uint64_t q8_data = (uint64_t)q8_blocks * 36;  // Q8_1 block = 36 bytes
    return partials + q8_data;
}

}  // namespace

// ─── NativeHead ──────────────────────────────────────────────────────

NativeHead::~NativeHead() {
    if (scratch_) cudaFree(scratch_);
    if (weights_) cudaFree(weights_);
}

bool NativeHead::load(const std::string& gguf_path, const std::string& tensor_name,
                      std::string& err) {
    artifact::GgufReader reader(gguf_path);
    if (!reader.ok()) {
        err = "NativeHead: " + reader.error();
        return false;
    }

    const artifact::GgufTensor* t = reader.find(tensor_name);
    if (!t) {
        err = "NativeHead: tensor '" + tensor_name + "' not found in " + gguf_path;
        return false;
    }

    // Record shape
    n_embd_ = t->ne[0];
    n_vocab_ = t->n_dims >= 2 ? t->ne[1] : t->ne[0];
    type_ = t->type;
    weight_bytes_ = t->nbytes;

    // Compute scratch size
    scratch_bytes_ = iq_scratch_bytes(n_vocab_, n_embd_);

    // Allocate device memory
    if (cudaMalloc(&weights_, weight_bytes_) != cudaSuccess) {
        err = "NativeHead: weight alloc failed";
        return false;
    }
    if (cudaMalloc(&scratch_, scratch_bytes_) != cudaSuccess) {
        err = "NativeHead: scratch alloc failed";
        cudaFree(weights_);
        weights_ = nullptr;
        return false;
    }

    // Read tensor data to host, then upload
    std::vector<uint8_t> host_buf(weight_bytes_);
    if (!reader.read_tensor(tensor_name, host_buf.data(), weight_bytes_)) {
        err = "NativeHead: cannot read tensor data";
        cudaFree(weights_); weights_ = nullptr;
        cudaFree(scratch_); scratch_ = nullptr;
        return false;
    }

    if (cudaMemcpy(weights_, host_buf.data(), weight_bytes_,
                   cudaMemcpyHostToDevice) != cudaSuccess) {
        err = "NativeHead: H2D copy failed";
        cudaFree(weights_); weights_ = nullptr;
        cudaFree(scratch_); scratch_ = nullptr;
        return false;
    }

    return true;
}

bool NativeHead::run(const float* mixed, float* logits, void* stream,
                     std::string& err) const {
    if (!loaded()) {
        err = "NativeHead: not loaded";
        return false;
    }
    if (!mixed || !logits) {
        err = "NativeHead: null input or output";
        return false;
    }
    // The actual IQ MMVQ kernel dispatch will be connected in a later unit.
    // For now, validate inputs and return success (logits untouched).
    // This is the dispatch point for iq_kernels::native_mmvq.
    (void)stream;
    return true;
}

// ─── NativeEmbed ─────────────────────────────────────────────────────

NativeEmbed::~NativeEmbed() {
    if (host_) std::free(host_);
    if (dev_) cudaFree(const_cast<void*>(dev_));
}

bool NativeEmbed::load(const std::string& gguf_path, int64_t n_embd, int64_t n_vocab,
                       std::string& err) {
    artifact::GgufReader reader(gguf_path);
    if (!reader.ok()) {
        err = "NativeEmbed: " + reader.error();
        return false;
    }

    // Try common embedding tensor names
    const char* names[] = {"token_embd.weight", "tok_embeddings.weight", nullptr};
    const artifact::GgufTensor* t = nullptr;
    for (int i = 0; names[i]; ++i) {
        t = reader.find(names[i]);
        if (t) break;
    }
    if (!t) {
        err = "NativeEmbed: embedding tensor not found in " + gguf_path;
        return false;
    }

    n_embd_ = n_embd;
    n_vocab_ = n_vocab;
    type_ = t->type;
    bytes_ = t->nbytes;

    // Compute row size
    if (n_vocab > 0) {
        row_bytes_ = (size_t)(bytes_ / (uint64_t)n_vocab);
    }

    // Allocate host + device
    host_ = std::malloc((size_t)bytes_);
    if (!host_) {
        err = "NativeEmbed: host alloc failed";
        return false;
    }
    void* dev_ptr = nullptr;
    if (cudaMalloc(&dev_ptr, bytes_) != cudaSuccess) {
        err = "NativeEmbed: device alloc failed";
        std::free(host_); host_ = nullptr;
        return false;
    }
    dev_ = dev_ptr;

    // Read and upload
    if (!reader.read_tensor(t->name, host_, bytes_)) {
        err = "NativeEmbed: cannot read tensor data";
        std::free(host_); host_ = nullptr;
        cudaFree(dev_ptr); dev_ = nullptr;
        return false;
    }
    if (cudaMemcpy(dev_ptr, host_, (size_t)bytes_,
                   cudaMemcpyHostToDevice) != cudaSuccess) {
        err = "NativeEmbed: H2D copy failed";
        std::free(host_); host_ = nullptr;
        cudaFree(dev_ptr); dev_ = nullptr;
        return false;
    }

    return true;
}

void NativeEmbed::gather_dev(const int32_t* tokens, int64_t n_tok, float* out,
                             void* stream) const {
    (void)stream;
    // GPU gather kernel will be connected in a later unit.
    // For safety, zero-initialize output.
    if (out && n_tok > 0 && n_embd_ > 0) {
        std::memset(out, 0, (size_t)(n_tok * n_embd_ * sizeof(float)));
    }
}

void NativeEmbed::gather_one(int64_t token, float* out, void* stream) const {
    (void)stream;
    if (!host_ || !out || token < 0 || token >= n_vocab_) return;

    // Dequantize a single row from host copy.
    // For f32 type (type 0), just memcpy the row.
    // For f16 type (type 1), widen each element.
    const uint8_t* row = static_cast<const uint8_t*>(host_) + (size_t)token * row_bytes_;

    if (type_ == 0) {
        // F32
        std::memcpy(out, row, (size_t)(n_embd_ * sizeof(float)));
    } else if (type_ == 1) {
        // F16
        const uint16_t* h = reinterpret_cast<const uint16_t*>(row);
        for (int64_t i = 0; i < n_embd_; ++i) {
            out[i] = kernels::f32_from_f16(h[i]);
        }
    } else if (type_ == 30) {
        // BF16
        const uint16_t* h = reinterpret_cast<const uint16_t*>(row);
        for (int64_t i = 0; i < n_embd_; ++i) {
            uint32_t bits = (uint32_t)h[i] << 16;
            std::memcpy(&out[i], &bits, 4);
        }
    } else {
        // Quantized format — dequant kernel needed (later unit)
        std::memset(out, 0, (size_t)(n_embd_ * sizeof(float)));
    }
}

}  // namespace strata

