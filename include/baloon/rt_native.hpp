// rt_native.hpp - device-resident dense bf16 weights, embedding table and output head.
#pragma once
#include <cuda_runtime.h>

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "baloon/core_weights.hpp"

namespace baloon {
namespace rt {

struct DeviceFree { void operator()(void* p) const noexcept { if (p) cudaFree(p); } };
using DevicePtr = std::unique_ptr<void, DeviceFree>;

// name -> (rows, cols) of a dense matrix; bytes of a bf16 matrix are rows*cols*2.
using ShapeMap = std::map<std::string, std::pair<uint64_t, uint64_t>>;

// A tensor queued for upload.
struct Pending { std::string name; const uint16_t* host = nullptr; uint64_t rows = 0, cols = 0; };

// A weight row can be served densely when it was placed, is raw BF16 (engine form 1) and holds a whole, non-empty
// number of 2-byte elements.
bool eligible(const core::WIndexRow& r);

struct DenseView { const uint16_t* dev = nullptr; uint64_t rows = 0, cols = 0; };

class NativeDense {
public:
    NativeDense() = default;
    ~NativeDense();
    NativeDense(const NativeDense&) = delete;
    NativeDense& operator=(const NativeDense&) = delete;
    // Uploads every eligible row of `t` that has a shape in `shapes` with rows*cols*2 == dst_bytes (host_arena + arena_off).
    // A row that is eligible and has a shape of the wrong size makes the whole load fail (nothing stays resident).
    bool load(const core::WeightTable& t, const uint8_t* host_arena, const ShapeMap& shapes, int ordinal, std::string* err = nullptr);
    std::vector<std::string> served_names() const;           // sorted
    bool find(const std::string& name, DenseView* out) const;
    // out = W x for a served tensor; d_x has `cols` floats, d_out `rows` floats (device pointers).
    bool apply(const std::string& name, const float* d_x, float* d_out, cudaStream_t s = nullptr) const;
    uint64_t resident_bytes() const { return bytes_; }
    int ordinal() const { return ord_; }
private:
    void clear();
    struct Entry { DevicePtr buf; uint64_t rows = 0, cols = 0; };
    std::map<std::string, Entry> e_;
    uint64_t bytes_ = 0;
    int ord_ = 0;
};

// Token embedding table, float32 [vocab, n]; a host copy pointer is kept (not owned) for single-row lookups.
class NativeEmbed {
public:
    NativeEmbed() = default;
    ~NativeEmbed();
    NativeEmbed(const NativeEmbed&) = delete;
    NativeEmbed& operator=(const NativeEmbed&) = delete;
    bool load(const float* host_table, int64_t vocab, int64_t n, int ordinal, std::string* err = nullptr);
    bool valid() const { return dev_ != nullptr; }
    int64_t vocab() const { return vocab_; }
    int64_t width() const { return n_; }
    // Device gather of `count` token ids (device pointers); invalid ids give zero rows.
    bool gather_dev(const int32_t* d_tokens, int64_t count, float* d_out, cudaStream_t s = nullptr) const;
    // Host lookup of one row; false for an out-of-range token (out untouched).
    bool gather_one(int64_t token, float* out) const;
private:
    DevicePtr dev_;
    const float* host_ = nullptr;
    int64_t vocab_ = 0, n_ = 0;
    int ord_ = 0;
};

// Output projection logits = W x with W bf16 [vocab, n], resident on device.
// Optional: load_norm() uploads the BF16 model.language_model.norm weight as f32;
// run() will then apply rms_norm_one() before the bf16_gemv projection.
class NativeHead {
public:
    NativeHead() = default;
    ~NativeHead();
    NativeHead(const NativeHead&) = delete;
    NativeHead& operator=(const NativeHead&) = delete;
    bool load(const uint16_t* host_w, int64_t vocab, int64_t n, int ordinal, std::string* err = nullptr);
    // Upload the BF16 norm weight (shape [n]) converted to f32 on the device.
    // Must be called after load() so ordinal and n_ are already set; n must equal width().
    bool load_norm(const uint16_t* host_norm_w, int64_t n, std::string* err = nullptr);
    bool valid() const { return dev_ != nullptr; }
    // Host x[n] -> host logits[vocab] (copies in, runs, copies out, synchronizes).
    // If a norm weight is loaded, applies rms_norm_one(x, norm_w, eps=1e-6) before projection.
    bool run(const float* x, float* logits) const;
    // Device x[n] -> device logits[vocab], fully async on s — suitable for session.head CUDA graph capture.
    // d_x is the f32 hidden state from the last layer (already on device); d_logits receives the
    // 248 320-token vocabulary logit vector. Applies norm if loaded, then projects through lm_head.
    bool run_dev(const float* d_x, float* d_logits, cudaStream_t s = nullptr) const;
    int64_t vocab() const { return vocab_; }
    int64_t width() const { return n_; }
private:
    DevicePtr dev_;
    DevicePtr norm_dev_;    // f32 [n_] RMS-norm weight, or null when not loaded
    DevicePtr norm_buf_;    // f32 [n_] scratch for the normed hidden state (pre-alloc'd in load_norm()
                            // so run_dev() contains no cudaMalloc and is CUDA-graph-capturable)
    int64_t vocab_ = 0, n_ = 0;
    int ord_ = 0;
};

// Per-layer weighted RMS-norm, resident on device.  One instance per layer norm weight
// (input_layernorm.weight or post_attention_layernorm.weight for a given layer index).
// apply_inplace overwrites the hidden-state buffer in place:
//   d_x = d_x * rsqrt(mean(d_x^2) + eps) * w
// using the rms_norm_one kernel (d_x serves as both input and output).
class NativeLayerNorm {
public:
    NativeLayerNorm() = default;
    ~NativeLayerNorm();
    NativeLayerNorm(const NativeLayerNorm&) = delete;
    NativeLayerNorm& operator=(const NativeLayerNorm&) = delete;
    // Upload the BF16 weight (shape [n]) converted to f32 on device ordinal.
    // Sets valid() to true on success; resets to empty on failure.
    bool load(const uint16_t* host_w, int64_t n, int ordinal, std::string* err = nullptr);
    bool valid() const { return dev_ != nullptr; }
    int64_t width() const { return n_; }
    // Apply weighted RMS-norm in-place (input_layernorm or post_attention_layernorm).
    // d_x[n] is overwritten with rms_norm(d_x) * w.  Runs async on s. False on bad args.
    bool apply_inplace(float* d_x, cudaStream_t s = nullptr) const;
private:
    DevicePtr dev_;
    int64_t n_ = 0;
    int ord_ = 0;
};

// Per-layer weighted RMS-norm weight uploaded once and applied in-place on the device.
// Covers input_layernorm.weight and post_attention_layernorm.weight (one instance per layer).
// (input_layernorm.weight or post_attention_layernorm.weight for a given layer index).
// apply_inplace overwrites the hidden-state buffer in place:
//   d_x = d_x * rsqrt(mean(d_x^2) + eps) * w
// using the rms_norm_one kernel (d_x serves as both input and output).
class NativeLayerNorm {
public:
    NativeLayerNorm() = default;
    ~NativeLayerNorm();
    NativeLayerNorm(const NativeLayerNorm&) = delete;
    NativeLayerNorm& operator=(const NativeLayerNorm&) = delete;
    // Upload the BF16 weight (shape [n]) converted to f32 on device ordinal.
    // Sets valid() to true on success; resets to empty on failure.
    bool load(const uint16_t* host_w, int64_t n, int ordinal, std::string* err = nullptr);
    bool valid() const { return dev_ != nullptr; }
    int64_t width() const { return n_; }
    // Apply weighted RMS-norm in-place (input_layernorm or post_attention_layernorm).
    // d_x[n] is overwritten with rms_norm(d_x) * w.  Runs async on s. False on bad args.
    bool apply_inplace(float* d_x, cudaStream_t s = nullptr) const;
private:
    DevicePtr dev_;
    int64_t n_ = 0;
    int ord_ = 0;
};

// Process-wide embedding binding (not owned); nullptr clears it.
const NativeEmbed* native_embed();
void set_native_embed(const NativeEmbed* e);

}  // namespace rt
}  // namespace baloon
