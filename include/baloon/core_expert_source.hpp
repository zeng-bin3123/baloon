// core_expert_source.hpp - adapter between the host loop and the CPU expert pool: file- or arena-backed blob lookup,
// hit/miss job building, the PCIe share of missed experts, and chunked loading of GGUF expert tensors.
// Blobs are addressed by layer*n_expert+expert and `expert >= n_expert` is rejected separately (the flat index alone
// would alias the next layer's first expert). For every router entry exactly one engine computes it: jobs + gpu = k.
// PCIe share m = (nmiss * pcie_num) >> 8: the LAST m missed experts in routing order go to the GPU.
#pragma once
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace baloon {
namespace core {

constexpr int64_t kGgufExpertsPerChunk = 16;

class ExpertSource {
public:
    virtual ~ExpertSource() = default;
    // nullptr when (layer, expert) is out of range; every accepted lookup bumps reads().
    virtual const uint8_t* blob(int64_t layer, int64_t expert) = 0;
    uint64_t reads() const { return reads_; }
    int64_t n_layers() const { return n_layers_; }
    int64_t n_expert() const { return n_expert_; }
    int64_t blob_bytes() const { return blob_; }
    bool in_range(int64_t layer, int64_t expert) const {
        return layer >= 0 && layer < n_layers_ && expert >= 0 && expert < n_expert_;
    }
    uint64_t file_bytes() const { return (uint64_t)n_layers_ * (uint64_t)n_expert_ * (uint64_t)blob_; }

protected:
    int64_t n_layers_ = 0, n_expert_ = 0, blob_ = 0;
    uint64_t reads_ = 0;
};

// experts.bin read through a read-only mapping; the size must be exactly n_layers * n_expert * blob.
class FileExpertSource : public ExpertSource {
public:
    FileExpertSource() = default;
    ~FileExpertSource() override;
    bool open(const std::string& path, int64_t n_layers, int64_t n_expert, int64_t blob);
    void close();
    const uint8_t* blob(int64_t layer, int64_t expert) override;

private:
    const uint8_t* base_ = nullptr;
    size_t map_bytes_ = 0;
    int fd_ = -1;
    std::vector<uint8_t> fallback_;
};

// The whole file resident in one arena that is one max-blob LONGER than the file, so a slot-sized copy may start at any
// expert. pinned() says whether the arena could be page-locked; device_alias() is the device-visible base address.
class ArenaExpertSource : public ExpertSource {
public:
    ArenaExpertSource() = default;
    ~ArenaExpertSource() override;
    bool open(const std::string& path, int64_t n_layers, int64_t n_expert, int64_t blob, int64_t max_slot_blob = 0);
    bool open_bytes(const uint8_t* data, size_t bytes, int64_t n_layers, int64_t n_expert, int64_t blob, int64_t max_slot_blob = 0);
    void close();
    const uint8_t* blob(int64_t layer, int64_t expert) override;
    bool pinned() const { return pinned_; }
    const uint8_t* device_alias() const { return arena_.empty() ? nullptr : arena_.data(); }
    uint64_t arena_bytes() const { return arena_.size(); }

private:
    std::vector<uint8_t> arena_;
    bool pinned_ = false;
};

struct ExpertJob {
    int32_t out_index = 0;          // router entry i (the output row)
    int32_t expert = 0;
    const uint8_t* blob = nullptr;  // resolved by expert_pool_dispatch
};

struct ExpertDispatch {
    int32_t k = 0;
    std::vector<ExpertJob> jobs;        // CPU jobs: the array index njobs is NOT the output index (out_index is)
    std::vector<int32_t> gpu_hits;      // output rows served by resident GPU slots
    std::vector<ExpertJob> gpu_pcie;    // missed experts shipped to the GPU (last m misses in routing order)
    bool failed = false;
    size_t njobs() const { return jobs.size(); }
    size_t n_gpu() const { return gpu_hits.size() + gpu_pcie.size(); }
};

struct GpuPlanSink {
    virtual ~GpuPlanSink() = default;
    virtual void plan(int64_t layer, const ExpertDispatch& d) = 0;
};

constexpr uint64_t pcie_share_of(uint64_t nmiss, uint64_t pcie_num) { return (nmiss * pcie_num) >> 8; }

// Classify router entries (no blobs): hits -> gpu_hits; misses -> jobs, the last m of them -> gpu_pcie. pcie_num in [0,256].
bool expert_hit_run(const std::vector<bool>& is_hit, int pcie_num, ExpertDispatch& out, const int32_t* experts = nullptr);
bool expert_pool_dispatch(ExpertSource& src, GpuPlanSink* sink, int64_t layer, const int32_t* experts,
                          const std::vector<bool>& is_hit, int pcie_num, ExpertDispatch& out);
bool expert_pool_dispatch_multi(ExpertSource& src, GpuPlanSink* sink, int64_t layer,
                                const std::vector<std::vector<int32_t>>& experts,
                                const std::vector<std::vector<bool>>& is_hit, int pcie_num,
                                std::vector<ExpertDispatch>& out);

struct BlobRegions { int64_t per[3]; int64_t at[3]; };
// Regions [gate | up | down] of one blob: per = {up_off, up_off, blob - down_off}, at = {0, up_off, down_off}.
constexpr BlobRegions blob_regions(int64_t up_off, int64_t down_off, int64_t blob) {
    return BlobRegions{{up_off, up_off, blob - down_off}, {0, up_off, down_off}};
}
// Build one layer's blobs from the three source tensors (n_expert * per[i] bytes each), 16 experts per chunk.
// Requires down_off == 2 * up_off <= blob (the regions then tile the blob).
bool load_experts_gguf(const uint8_t* gate, const uint8_t* up, const uint8_t* down, int64_t n_expert, int64_t up_off,
                       int64_t down_off, int64_t blob, uint8_t* dst, int64_t* n_chunks = nullptr);

}  // namespace core
}  // namespace baloon
