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

struct DevFree { void operator()(void* p) const noexcept { if (p) cudaFree(p); } };
using DevBuf = std::unique_ptr<void, DevFree>;

// name -> (rows, cols) of a dense matrix; bytes of a bf16 matrix are rows*cols*2.
using ShapeMap = std::map<std::string, std::pair<uint64_t, uint64_t>>;

// A tensor queued for upload.
struct PendingTensor { std::string name; const uint16_t* host = nullptr; uint64_t rows = 0, cols = 0; };

// A weight row can be served densely when it was placed, is raw BF16 (engine form 1) and holds a whole, non-empty
// number of 2-byte elements.
bool eligible(const core::WIndexRow& r);

struct DenseView { const uint16_t* dev = nullptr; uint64_t rows = 0, cols = 0; };

class DenseBank {
public:
    DenseBank() = default;
    ~DenseBank();
    DenseBank(const DenseBank&) = delete;
    DenseBank& operator=(const DenseBank&) = delete;
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
    struct Entry { DevBuf buf; uint64_t rows = 0, cols = 0; };
    std::map<std::string, Entry> e_;
    uint64_t bytes_ = 0;
    int ord_ = 0;
};

// Token embedding table, float32 [vocab, n]; a host copy pointer is kept (not owned) for single-row lookups.
class EmbedTable {
public:
    EmbedTable() = default;
    ~EmbedTable();
    EmbedTable(const EmbedTable&) = delete;
    EmbedTable& operator=(const EmbedTable&) = delete;
    bool load(const float* host_table, int64_t vocab, int64_t n, int ordinal, std::string* err = nullptr);
    bool valid() const { return dev_ != nullptr; }
    int64_t vocab() const { return vocab_; }
    int64_t width() const { return n_; }
    // Device gather of `count` token ids (device pointers); invalid ids give zero rows.
    bool gather_dev(const int32_t* d_tokens, int64_t count, float* d_out, cudaStream_t s = nullptr) const;
    // Host lookup of one row; false for an out-of-range token (out untouched).
    bool gather_one(int64_t token, float* out) const;
private:
    DevBuf dev_;
    const float* host_ = nullptr;
    int64_t vocab_ = 0, n_ = 0;
    int ord_ = 0;
};

// Output projection logits = W x with W bf16 [vocab, n], resident on device.
class HeadProj {
public:
    HeadProj() = default;
    ~HeadProj();
    HeadProj(const HeadProj&) = delete;
    HeadProj& operator=(const HeadProj&) = delete;
    bool load(const uint16_t* host_w, int64_t vocab, int64_t n, int ordinal, std::string* err = nullptr);
    bool valid() const { return dev_ != nullptr; }
    // Host x[n] -> host logits[vocab] (copies in, runs, copies out, synchronizes).
    bool run(const float* x, float* logits) const;
    int64_t vocab() const { return vocab_; }
    int64_t width() const { return n_; }
private:
    DevBuf dev_;
    int64_t vocab_ = 0, n_ = 0;
    int ord_ = 0;
};

// Process-wide embedding binding (not owned); nullptr clears it.
const EmbedTable* native_embed();
void set_native_embed(const EmbedTable* e);

}  // namespace rt
}  // namespace baloon
