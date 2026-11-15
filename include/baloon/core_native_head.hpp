// core_native_head.hpp - device-resident embedding table and output head (device-paired path).
// NativeEmbed owns a float32 [vocab, n] table on the GPU and provides gather_dev() (batch device→device
// gather) and gather_one() (single host-side lookup from the host-pinned copy).  NativeHead owns a BF16
// [vocab, n] weight matrix and runs the output projection W x with host in/out buffers.
// g_embed is the process-wide embedding pointer; native_embed() reads it, set_native_embed() writes it.
#pragma once

#if defined(__has_include)
#if __has_include(<cuda_runtime.h>)
#define BALOON_NH_HAVE_CUDA 1
#endif
#endif

#ifdef BALOON_NH_HAVE_CUDA
#include <cuda_runtime.h>
#else
typedef void* cudaStream_t;
#endif

#include <cstdint>
#include <string>

namespace baloon {
namespace head {

// Token embedding table [vocab × n] resident on device (float32).
struct NativeEmbed {
    NativeEmbed() = default;
    ~NativeEmbed();
    NativeEmbed(const NativeEmbed&)            = delete;
    NativeEmbed& operator=(const NativeEmbed&) = delete;

    // Uploads float32 [vocab, n] from host to device `ordinal`; keeps a host pointer for gather_one.
    bool load(const float* host_table, int64_t vocab, int64_t n, int ordinal,
              std::string* err = nullptr);
    // Device gather: d_tokens[count] int32 → d_out[count * n] float32 (device pointers).
    bool gather_dev(const int32_t* d_tokens, int64_t count, float* d_out,
                    cudaStream_t s = nullptr) const;
    // Host lookup: copies one row out[n] from the host mirror; false for out-of-range token.
    bool gather_one(int64_t token, float* out) const;

    bool    valid() const { return dev_ != nullptr; }
    int64_t vocab() const { return vocab_; }
    int64_t width() const { return n_; }

private:
    void*         dev_   = nullptr;   // device float32 [vocab, n]
    const float*  host_  = nullptr;   // non-owned host pointer for gather_one
    int64_t       vocab_ = 0;
    int64_t       n_     = 0;
#ifdef BALOON_NH_HAVE_CUDA
    int           ord_   = 0;         // device ordinal — only needed in CUDA path
#endif
};

// Output projection W[vocab × n] BF16 resident on device; run() computes logits = W x.
struct NativeHead {
    NativeHead() = default;
    ~NativeHead();
    NativeHead(const NativeHead&)            = delete;
    NativeHead& operator=(const NativeHead&) = delete;

    // Uploads BF16 [vocab, n] weight matrix from host to device `ordinal`.
    bool load(const uint16_t* host_w, int64_t vocab, int64_t n, int ordinal,
              std::string* err = nullptr);
    // Host x[n] → host logits[vocab]: copies in, launches kernel, copies out, synchronises.
    bool run(const float* x, float* logits) const;

    bool    valid() const { return dev_ != nullptr; }
    int64_t vocab() const { return vocab_; }
    int64_t width() const { return n_; }

private:
    void*   dev_   = nullptr;   // device uint16_t [vocab, n]
    int64_t vocab_ = 0;
    int64_t n_     = 0;
#ifdef BALOON_NH_HAVE_CUDA
    int     ord_   = 0;         // device ordinal — only needed in CUDA path
#endif
};

// Process-wide embedding pointer (not owned); nullptr when not set.
extern const NativeEmbed* g_embed;
const NativeEmbed* native_embed();
void set_native_embed(const NativeEmbed* e);

}  // namespace head
}  // namespace baloon

