#pragma once
// strata/core/expert_source.hpp — Expert blob retrieval: abstract interface + file/arena sources
// Target: Iluvatar BI-V100 (ivcore10)
//
// ExpertSource: abstract interface for retrieving expert weight blobs.
// FileExpertSource: memory-mapped (mmap), no copy, ~44 GB/s warm path.
// ArenaExpertSource: reads into PinnedArena, GPU-visible, ~42.8 GB/s.
//
// GpuPlanSink: per-layer GPU dispatch plan (VRAM hits + PCIe share).
// ExpertDispatch: per-session adapter state for expert routing.

#include <cstdint>
#include <string>
#include <vector>

namespace strata {

// Forward declarations
class ExpertCache;

namespace kernels { namespace cpu {

// Constants matching the MoE architecture
inline constexpr int64_t NE = 512;

// Expert evaluation job descriptor (single token).
struct ExpertJob {
    int64_t layer = 0;
    int64_t expert = 0;
    const uint8_t* blob = nullptr;
    const float* x_f = nullptr;
    float* out = nullptr;
};

// Multi-token expert job descriptor.
struct ExpertJobMulti {
    int64_t layer = 0;
    int64_t expert = 0;
    const uint8_t* blob = nullptr;
    int64_t n_tok = 0;
    const float** x_f_ptrs = nullptr;
    float** out_ptrs = nullptr;
};

// Activation quantization state.
struct ActQ {
    const uint8_t* data = nullptr;
    const float* scale = nullptr;
    int64_t n_elements = 0;
};

// Expert pool (CPU-side compute).  Defined in a later unit.
class ExpertPool;

}}  // namespace kernels::cpu

// ─── ExpertSource (abstract) ─────────────────────────────────────────
// Interface for expert blob retrieval.  Each blob is a contiguous
// byte buffer of fixed size (1,382,400 bytes for this architecture).
//
// blob() returns a pointer valid until the NEXT blob() call
// (non-caching sources reuse an internal buffer).
class ExpertSource {
public:
    virtual ~ExpertSource() = default;

    // Retrieve expert blob for (layer, expert).
    // Returns nullptr if unavailable.
    virtual const uint8_t* blob(int64_t layer, int64_t expert) = 0;

    // Number of blob() calls made (monotonically non-decreasing).
    virtual int64_t reads() const { return 0; }

    // Whether this blob resides in page-locked (pinned) memory.
    virtual bool pinned(int64_t layer, int64_t expert) const {
        (void)layer; (void)expert; return false;
    }

    // Advisory prefetch hint: layer about to process experts ids[0..k).
    virtual void begin_layer(int64_t layer, const int32_t* ids, int64_t k) {
        (void)layer; (void)ids; (void)k;
    }

    // GPU-visible alias of the blob (if available via cudaHostRegister).
    virtual const uint8_t* device_alias(int64_t layer, int64_t expert) const {
        (void)layer; (void)expert; return nullptr;
    }
};

// ─── GpuPlanSink ─────────────────────────────────────────────────────
// GPU's per-layer dispatch plan: VRAM hits + PCIe-fetched experts.
//
// counts[0] = VRAM groups, counts[1] = all entries, counts[2] = PCIe groups
// pcie_mode: 0 = DMA staging, 1 = mapped read, 2 = copy kernel
struct GpuPlanSink {
    int32_t* counts = nullptr;        // [3]
    int32_t* start = nullptr;         // [cap+1] group start indices
    int32_t* dst = nullptr;           // [cap] destination token indices
    int32_t* tok = nullptr;           // [cap] token indices
    unsigned long long* ptr = nullptr; // [cap] device addresses of VRAM blobs
    unsigned long long* ptr2 = nullptr;// [cap] device addresses for PCIe blobs
    int32_t* start2 = nullptr;        // [cap+1] PCIe group starts
    unsigned long long staging = 0;    // staging buffer device address
    int64_t staging_cap = 0;          // staging buffer capacity (bytes)
    int64_t cap = 0;                  // max entries
    void (*publish)(void* ctx) = nullptr;     // signal GPU: plan is ready
    void (*fetch)(void* ctx, const uint8_t* const* src,
                  int count, size_t bytes) = nullptr; // DMA host blobs to staging
    void* ctx = nullptr;              // callback context
    int pcie_mode = 0;                // 0=DMA, 1=mapped, 2=copy kernel
};

// ─── HitPhase ────────────────────────────────────────────────────────
// Phases of GPU expert hit processing.
enum class HitPhase : int {
    Prepare = 0,   // quantize input, set up slot/dst arrays
    Compute = 1,   // run GPU expert evaluation on cache hits
    Combine = 2    // accumulate hit results into output
};

// ─── ExpertDispatch ──────────────────────────────────────────────────
// Per-session adapter state for expert routing: manages the interplay
// between CPU expert pool, ExpertSource, and GPU ExpertCache.
struct ExpertDispatch {
    kernels::cpu::ExpertPool* pool = nullptr;
    ExpertSource* src = nullptr;
    int64_t n_expert = kernels::cpu::NE;

    // Counters
    int64_t layers = 0;
    int64_t experts = 0;
    int64_t missing = 0;

    // Cache state
    ExpertCache* cache = nullptr;
    int64_t cache_hits = 0;
    int64_t cache_admitted = 0;
    int64_t cache_refused = 0;
    void* cache_stream = nullptr;
    const char* cache_fail = nullptr;

    // GPU-side cache pointers
    const uint8_t* cache_base = nullptr;
    int64_t cache_blob = 0;
    const uint64_t* cache_slot_off = nullptr;
    void* hit_scratch = nullptr;
    float* parts_out = nullptr;
    float* hit_out = nullptr;
    int64_t parts_elems = 0;
    const float* mixed = nullptr;
    uint8_t* x_q8_0_hit = nullptr;
    float* x_q8_0_hit_scale = nullptr;
    int32_t* d_slot = nullptr;
    int32_t* d_dst = nullptr;
    std::vector<int32_t> h_slot, h_dst;
    std::vector<uint8_t> is_hit;
    bool decided = false;
    const int32_t* host_res = nullptr;
    bool split_rows = true;

    // Hit timing
    void* hit_done = nullptr;
    int64_t hit_ready = 0;
    int64_t hit_late = 0;
    bool hit_poke = false;
    bool hit_cpu_order = false;
    int64_t n_hits = 0;
    bool hit_pending = false;
    const char* hit_fail = nullptr;

    // Predicate: all GPU-side resources are allocated and ready.
    bool hits_ready() const {
        return cache != nullptr && cache_base != nullptr &&
               parts_out != nullptr && hit_out != nullptr &&
               mixed != nullptr && hit_scratch != nullptr &&
               x_q8_0_hit != nullptr && x_q8_0_hit_scale != nullptr &&
               d_slot != nullptr && d_dst != nullptr;
    }

    // CPU job descriptors
    std::vector<kernels::cpu::ExpertJob> jobs;
    kernels::cpu::ActQ act;
    std::vector<kernels::cpu::ActQ> act_multi;
    std::vector<uint8_t> nact_multi;
    std::vector<kernels::cpu::ExpertJobMulti> jobs_multi;
    std::vector<int16_t> job_of;

    // GPU plan
    GpuPlanSink* plan = nullptr;
    int pcie_num = 0;
    int64_t pcie_experts = 0;

    // Timing (ms)
    double ms_plan = 0, ms_actq = 0, ms_jobs = 0, ms_run = 0;

    // Usage stats
    std::vector<float> usage;
    int64_t multi_misses = 0;
    int64_t multi_entries = 0;

    // Failure latch: once failed = true, it is PERMANENT.
    bool failed = false;
    const char* fail = nullptr;
    int64_t fail_layer = -1;
    int64_t fail_expert = -1;
};

// ─── Dispatch functions ──────────────────────────────────────────────

// Single-token expert dispatch: evaluates k experts for one token.
// out[j] = expert_eval(blob(layer, ids[j]), x_f)  or zeros if blob is null.
// Weight NOT applied here; moe_combine does sum(w_j * out[j]).
void expert_pool_dispatch(void* user, const float* x_f,
                          const int32_t* ids, const float* weights,
                          int64_t n_embd, int64_t k, float* out);

// Multi-token expert dispatch: evaluates experts for n_tok tokens.
// Deduplicates: each distinct expert computed once, results scattered.
void expert_pool_dispatch_multi(ExpertDispatch& d, const float* x_f,
                                const int32_t* ids, int64_t n_tok,
                                int64_t k, float* out);

// GPU expert hit run: processes cache-hit experts on GPU.
// Phases: Prepare → Compute → Combine.
// No-op when n_hits = 0.
void expert_hit_run(void* user, void* stream, HitPhase phase,
                    const int32_t* ids, int64_t k);

// ─── FileExpertSource ────────────────────────────────────────────────
// Memory-mapped expert source.  Maps the entire experts.bin file and
// serves blob pointers directly from the mapping.  ~44 GB/s warm path
// from page cache when entire file fits in RAM.
//
// File size must equal n_layers * n_expert * BLOB (1,382,400 bytes).
class FileExpertSource : public ExpertSource {
public:
    FileExpertSource() = default;
    ~FileExpertSource() override;

    FileExpertSource(const FileExpertSource&) = delete;
    FileExpertSource& operator=(const FileExpertSource&) = delete;

    bool open(const std::string& pack_dir, int64_t n_layers,
              int64_t n_expert, std::string& err);
    void close();

    bool mapped() const { return base_ != nullptr; }
    int64_t blobs() const { return blobs_; }

    const uint8_t* blob(int64_t layer, int64_t expert) override;
    int64_t reads() const override { return read_count_; }

private:
    const uint8_t* base_ = nullptr;   // mmap base
    int64_t blobs_ = 0;               // total blob count (n_layers * n_expert)
    int64_t n_expert_ = 0;
    int64_t read_count_ = 0;
#if defined(_WIN32)
    void* file_ = nullptr;
    void* mapping_ = nullptr;
#else
    int fd_ = -1;
#endif
};

// ─── ArenaExpertSource ───────────────────────────────────────────────
// Reads experts.bin into a PinnedArena (anonymous memory).  GPU-visible
// via cudaHostRegister.  ~42.8 GB/s vs 19 GB/s for mmap when OS page
// cache is under pressure (34 GB experts + 26.8 GB n-gram > 63 GB RAM).
//
// Provides device_alias() for zero-copy GPU reads of pinned blobs.
class ArenaExpertSource : public ExpertSource {
public:
    ArenaExpertSource() = default;
    ~ArenaExpertSource() override;

    ArenaExpertSource(const ArenaExpertSource&) = delete;
    ArenaExpertSource& operator=(const ArenaExpertSource&) = delete;

    bool open(const std::string& pack_dir, int64_t n_layers,
              int64_t n_expert, int threads, std::string& err);
    void set_gguf(const std::string& shard1) { gguf_path_ = shard1; }
    void close();

    bool mapped() const { return base_ != nullptr; }
    int64_t blobs() const { return blobs_; }

    const uint8_t* blob(int64_t layer, int64_t expert) override;
    int64_t reads() const override { return read_count_; }
    bool pinned(int64_t layer, int64_t expert) const override;
    const uint8_t* device_alias(int64_t layer, int64_t expert) const override;

    const std::string& note() const { return note_; }
    double load_gib_per_second() const { return load_speed_; }

private:
    void* arena_ = nullptr;                  // PinnedArena handle
    std::vector<const uint8_t*> dev_slice_;  // per-slice device aliases
    uint64_t slice_bytes_ = 0;
    const uint8_t* base_ = nullptr;          // arena data pointer
    int64_t blobs_ = 0;
    int64_t n_expert_ = 0;
    int64_t read_count_ = 0;
    std::string note_;
    double load_speed_ = 0.0;
    uint64_t pinned_bytes_ = 0;
    std::string gguf_path_;
};

}  // namespace strata

