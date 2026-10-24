// strata/kernels/cpu/expert_thread_pool.hpp — Work-stealing pool with core pinning
// Target: Iluvatar BI-V100 host CPU
//
// Fixed-size thread pool for parallel expert MLP evaluation.
// Workers spin-park on an epoch counter, drain jobs via atomic head fetch-add.
// The host optionally participates ("host works" mode).
//
// Modes:
//   run()       — single-token whole-expert dispatch
//   run_split() — split (gate/up → quantize → down) for intra-expert parallelism
//   run_multi() — multi-token batched dispatch
#pragma once

#include "strata/kernels/cpu/expert_kernel.hpp"
#include <atomic>
#include <cstdint>
#include <thread>
#include <vector>

namespace strata { namespace kernels { namespace cpu {

// ─── Expert job descriptors ─────────────────────────────────────

// Single-token job.
struct PoolJob {
    const uint8_t* blob = nullptr;
    const ActQ* act = nullptr;
    float* out = nullptr;
    float weight = 1.0f;
    int slot = 0;
};

// Multi-token job.
struct PoolJobMulti {
    const uint8_t* blob = nullptr;
    int n_tok = 0;
    const ActQ* const* act_ptrs = nullptr;
    float* const* out_ptrs = nullptr;
};

// ─── Physical core enumeration ──────────────────────────────────
// Returns list of CPU IDs for physical cores.
// If skip_first, omits core 0 (reserved for main thread / GPU driver).
std::vector<int> physical_cores(bool skip_first);

// Pin current thread to a specific core.
void pin_thread(int core);

// ─── Native format linkage for split dispatch ───────────────────
struct NativeFmt;  // forward decl from expert_layout.hpp

// ─── ExpertPool ─────────────────────────────────────────────────

static constexpr int kMaxExperts = 20;       // max experts per layer (top-k)
static constexpr int kMaxSplit   = 10240;    // max split tasks (gate/up rows)
static constexpr int kMaxSplitMulti = 20480; // max multi-token split tasks

class ExpertPool {
public:
    // Create pool with nw workers.
    // pin: pin workers to physical cores.
    // host_works: host thread participates in drain.
    ExpertPool(int nw, bool pin, bool host_works);
    ~ExpertPool();

    ExpertPool(const ExpertPool&) = delete;
    ExpertPool& operator=(const ExpertPool&) = delete;

    // Number of worker threads.
    int count() const { return count_; }

    // ── Dispatch modes ─────────────────────────────────────────
    // Single-token: run all jobs through expert_kernel.
    void run(PoolJob* jobs, int njobs);

    // Split: phase 1 (gate/up), host quantizes, phase 2 (down).
    void run_split(PoolJob* jobs, int njobs);

    // Multi-token: batched dispatch for n_tok > 1.
    void run_multi(PoolJobMulti* jobs, int njobs, int n_tok);

    // Set native format for native-path dispatch.
    void set_native_fmt(const NativeFmt* fmt) { nfmt_ = fmt; }

private:
    void worker(int id);
    void drain(int id, ExpertScratch& ws);
    void run_phase(int mode, int ntasks);

    int count_ = 0;
    bool host_works_ = false;
    const NativeFmt* nfmt_ = nullptr;

    // Thread management
    std::vector<std::thread> threads_;
    std::atomic<bool> stop_{false};

    // Epoch-based spin-park
    std::atomic<uint32_t> epoch_{0};
    std::atomic<uint32_t> parked_{0};

    // Job queue (shared state, written by host before epoch bump)
    PoolJob* jobs_ = nullptr;
    PoolJobMulti* jobs_multi_ = nullptr;
    int mode_ = 0;          // 0=whole, 1=split-gu, 2=split-down, 3=multi-gu, 4=multi-down
    int njobs_ = 0;
    int n_tok_ = 0;
    std::atomic<int> head_{0};
    std::atomic<int> done_{0};

    // Per-worker scratch
    std::vector<ExpertScratch> scratch_;
    ExpertScratch host_scratch_;

    // Split dispatch state
    int parts_a_ = 0;   // gate/up chunks per expert
    int parts_b_ = 0;   // down chunks per expert
};

}}}  // namespace strata::kernels::cpu
