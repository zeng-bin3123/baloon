// u11_native_dense.hpp - device-resident dense BF16 weights of the reference layout (src/core/native_dense.cpp).
#pragma once
#include <stddef.h>
#if defined(__has_include)
#if __has_include(<cuda_runtime.h>)
#include <cuda_runtime.h>
#define U11_HAVE_CUDA_HEADER 1
#endif
#endif
#ifndef U11_HAVE_CUDA_HEADER
// Host-only translation units (no CUDA include path): the few runtime entry points this unit uses, with the runtime's C ABI.
extern "C" {
int cudaFree(void*);
int cudaMalloc(void**, size_t);
int cudaMemcpy(void*, const void*, size_t, int);
int cudaGetDevice(int*);
int cudaSetDevice(int);
int cudaGetLastError(void);
}
#define cudaSuccess 0
#define cudaMemcpyHostToDevice 1
#endif

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "baloon/core_weights.hpp"

namespace baloon {
namespace u11 {

struct DeviceFree { void operator()(void* p) const noexcept { if (p) cudaFree(p); } };
using DevicePtr = std::unique_ptr<void, DeviceFree>;
using ShapeMap = std::map<std::string, std::pair<uint64_t, uint64_t>>;   // name -> (rows, cols)

struct Pending { std::string name; const uint16_t* host = nullptr; uint64_t rows = 0, cols = 0; };

// ---- NativeDenseOverride: GGUF block-quantized copies of the GDN/QSA dense matrices ---------------------------------
// The ten attention / SSM / shared-expert matrix suffixes served natively.
constexpr const char* kNativeSuffixes[10] = {
    ".attn_qkv.weight", ".attn_gate.weight", ".ssm_out.weight", ".attn_q.weight", ".attn_k.weight",
    ".attn_v.weight", ".attn_output.weight", ".ffn_gate_shexp.weight", ".ffn_up_shexp.weight", ".ffn_down_shexp.weight"};
// Eligible: a `blk.` tensor whose name ends with one of the ten suffixes, or `blk.1.ple_key.weight` when requested.
bool eligible(const std::string& name, bool include_ple_key);

// All-or-nothing: check (and upload) every item; the first failure aborts the whole batch.
template <class T, class R, class F>
bool check_all(const std::vector<T>& items, std::vector<R>& out, F check, std::string* err) {
    std::vector<R> acc;
    for (const auto& a : items) { R r; if (!check(a, r, err)) return false; acc.push_back(std::move(r)); }
    out = std::move(acc);
    return true;
}
// Publish only a fully successful batch; a failed batch leaves the table exactly as it was.
template <class Tbl, class R, class Apply>
void publish(Tbl& tbl, bool ok, std::vector<R>& batch, Apply apply) { if (ok) apply(tbl, batch); }

// A native override published into the canonical table: device bytes of the GGUF blocks plus their geometry.
struct OverrideRef { const void* dev = nullptr; uint64_t rows = 0, cols = 0, block_elems = 0, block_bytes = 0; };
using OverrideTable = std::map<std::string, OverrideRef>;

// Servable densely: placed, raw BF16 (engine form 1), a whole non-empty number of 2-byte elements.
bool eligible(const core::WIndexRow& r);

class NativeDense {
public:
    NativeDense() = default;
    ~NativeDense();
    NativeDense(const NativeDense&) = delete;
    NativeDense& operator=(const NativeDense&) = delete;
    // Uploads every eligible row with a shape (rows*cols*2 == dst_bytes). A wrong-sized shape fails the whole load and nothing stays resident.
    bool load(const core::WeightTable& t, const uint8_t* host_arena, const ShapeMap& shapes, int ordinal, std::string* err = nullptr);
    std::vector<std::string> served_names() const;   // sorted
    bool find(const std::string& name, const uint16_t** dev, uint64_t* rows, uint64_t* cols) const;
    uint64_t resident_bytes() const { return bytes_; }
    // Attach the eligible GGUF tensors of `data` (data section of `payload` bytes, directory `dir`) as native overrides:
    // every entry is span-checked, every eligible one uploaded, and only then are the references published into `tbl`.
    // Any failure leaves `tbl` untouched and nothing of this batch resident.
    bool attach(const uint8_t* data, uint64_t payload, const std::vector<GgufEntry>& dir, bool include_ple_key, int ordinal,
                OverrideTable& tbl, std::string* err = nullptr);
    uint64_t scratch_in() const { return scratch_in_; }   // max_in over the attached overrides
private:
    void clear();
    struct Entry { DevicePtr buf; uint64_t rows = 0, cols = 0; };
    std::map<std::string, Entry> e_;
    std::map<std::string, Entry> o_;   // attached overrides (owned)
    uint64_t scratch_in_ = 0;
    uint64_t bytes_ = 0;
    int ord_ = 0;
};

}  // namespace u11
}  // namespace baloon
