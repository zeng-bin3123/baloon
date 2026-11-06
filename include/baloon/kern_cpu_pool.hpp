// kern_cpu_pool.hpp - CPU expert thread pool: row-range splitting of experts across tasks, the flat-batch counter
// protocol (claim / completion / park counters + publish epoch), worker-count selection and core reservation.
// Implementation: src/cpu/expert_pool.cpp.
#pragma once
#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <thread>
#include <vector>

#include "baloon/kern_cpu_expert.hpp"
#include "baloon/kern_cpu_iq.hpp"
#include "baloon/kern_geometry.hpp"

namespace baloon {
namespace kern {

// One expert evaluated for one token (Q2_0 blob layout).
struct ExpertJob {
    const uint8_t* blob = nullptr;
    const float* x = nullptr;       // H inputs
    float* out = nullptr;           // H outputs
    ExpertScratch* scratch = nullptr;
};

// One expert evaluated for nt tokens. Q2_0 blob when gu_type == 42 (the default); otherwise the native i-quant layout
// gate rows | up rows | down rows (gu_type for gate/up, d_type for down; n_ff and n_embd multiples of 256).
struct ExpertJobMulti {
    const uint8_t* blob = nullptr;
    const float* x = nullptr;       // nt * H (or nt * n_embd for native)
    uint32_t nt = 0;
    float* out = nullptr;           // nt * H (or nt * n_embd)
    ExpertScratchMulti* scratch = nullptr;   // Q2_0 path
    uint32_t gu_type = 42, d_type = 42;
    const uint8_t* gu_w = nullptr;  // native: gate rows then up rows
    const uint8_t* d_w = nullptr;   // native: down rows
    size_t gu_row_bytes = 0, d_row_bytes = 0;
    uint32_t n_ff = 0, n_embd = 0;
};

// Logical CPU ids, one per physical core, in id order (falls back to hardware_concurrency()).
std::vector<int> physical_cores();
// Pin the calling thread to one logical CPU (remembers the original mask on first use); false when refused.
bool pin_current_thread(int cpu);
void restore_thread_affinity();

class ExpertPool {
public:
    static constexpr uint32_t kMaxSplit = poolsched::kSplitCap;             // split capacity (single-token experts per batch)
    static constexpr uint32_t kMaxSplitMulti = poolsched::kSplitMultiCap;  // split capacity (multi-token experts per batch)

    // requested: worker count (0 = one per physical core, never fewer than one). skip_first reserves the first core for
    // the host thread. host_works: the calling thread also drains tasks during a batch.
    explicit ExpertPool(uint32_t requested = 0, bool skip_first = false, bool host_works = false);
    ~ExpertPool();
    ExpertPool(const ExpertPool&) = delete;
    ExpertPool& operator=(const ExpertPool&) = delete;

    uint32_t workers() const { return nw_; }
    uint32_t threads() const { return poolsched::drain_threads(nw_, host_works_); }

    // One task per expert.
    void run(const ExpertJob* jobs, uint32_t n);
    // Each expert is cut into row ranges; returns false (nothing done) when n exceeds kMaxSplit, the pool has a single
    // worker, or the oracle contract is on.
    bool run_split(const ExpertJob* jobs, uint32_t n);
    // Multi-token phases over global row ranges (may span experts). Same refusal rule.
    bool run_split_multi(const ExpertJobMulti* jobs, uint32_t n);
    // Same for experts in the native i-quant layout (every job must have gu_type != 42). A layer with more than
    // kMaxSplitMulti experts is processed in consecutive batches of at most kMaxSplitMulti (all jobs validated first).
    bool run_split_multi_native(const ExpertJobMulti* jobs, uint32_t n);

    // Test hook: byte offsets of the contended words inside the pool object.
    static size_t off_head();
    static size_t off_done();
    static size_t off_parked();
    static size_t off_epoch();
    static size_t off_stop();

private:
    struct NativeScratch {
        std::vector<Q8KBlock> x8, a8;
        std::vector<float> act;
    };
    void worker(uint32_t idx);
    void drain();
    void exec(uint32_t mode, uint32_t i);
    void run_phase(uint32_t mode, uint32_t ntasks);
    bool multi_impl(const ExpertJobMulti* jobs, uint32_t n, bool native_only);

    uint32_t nw_ = 1;
    bool host_works_ = false;
    std::vector<int> cpus_;
    std::vector<std::thread> th_;
    std::mutex m_;
    std::condition_variable cv_work_, cv_host_;
    // Each contended word sits alone on its own 64-byte line (no false sharing between claim / completion / park / publish).
    alignas(64) std::atomic<uint64_t> head_{0};
    alignas(64) std::atomic<uint64_t> done_{0};
    alignas(64) std::atomic<uint32_t> parked_{0};
    alignas(64) uint64_t epoch_ = 0;
    alignas(64) bool stop_ = false;
    [[maybe_unused]] alignas(64) char line_end_[1] = {0};   // keeps stop_ alone on its line
    // current batch descriptor (written by the host only while every worker is parked)
    uint32_t mode_ = 0, ntasks_ = 0, njobs_ = 0, parts_ = 1;
    const ExpertJob* jobs_ = nullptr;
    const ExpertJobMulti* mjobs_ = nullptr;
    std::vector<NativeScratch> ns_;
};

}  // namespace kern
}  // namespace baloon
