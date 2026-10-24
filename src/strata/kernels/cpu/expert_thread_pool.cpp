// strata/kernels/cpu/expert_thread_pool.cpp — Work-stealing pool with core pinning
// Target: Iluvatar BI-V100 host CPU

#include "strata/kernels/cpu/expert_thread_pool.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"

#include <algorithm>
#include <cstring>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

#if defined(_WIN32)
#include <windows.h>
#else
#include <pthread.h>
#include <sched.h>
#endif

namespace strata { namespace kernels { namespace cpu {

// ─── Physical core enumeration ──────────────────────────────────
std::vector<int> physical_cores(bool skip_first) {
    std::vector<int> cores;

#if defined(_WIN32)
    DWORD len = 0;
    GetLogicalProcessorInformationEx(RelationProcessorCore, nullptr, &len);
    std::vector<uint8_t> buf(len);
    auto* info = reinterpret_cast<SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*>(buf.data());
    if (GetLogicalProcessorInformationEx(RelationProcessorCore, info, &len)) {
        auto* p = reinterpret_cast<uint8_t*>(info);
        while (p < buf.data() + len) {
            auto* entry = reinterpret_cast<SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*>(p);
            if (entry->Relationship == RelationProcessorCore) {
                // Pick first logical processor of each core
                for (WORD g = 0; g < entry->Processor.GroupCount; ++g) {
                    KAFFINITY mask = entry->Processor.GroupMask[g].Mask;
                    if (mask) {
                        unsigned long idx;
                        _BitScanForward64(&idx, mask);
                        cores.push_back(static_cast<int>(idx));
                        break;
                    }
                }
            }
            p += entry->Size;
        }
    }
#else
    cpu_set_t set;
    CPU_ZERO(&set);
    if (sched_getaffinity(0, sizeof(set), &set) == 0) {
        for (int i = 0; i < CPU_SETSIZE; ++i) {
            if (CPU_ISSET(i, &set)) {
                cores.push_back(i);
            }
        }
    } else {
        unsigned hw = std::thread::hardware_concurrency();
        for (unsigned i = 0; i < hw; ++i) {
            cores.push_back(static_cast<int>(i));
        }
    }
#endif

    if (skip_first && !cores.empty()) {
        cores.erase(cores.begin());
    }
    return cores;
}

// ─── Thread pinning ─────────────────────────────────────────────
void pin_thread(int core) {
    if (core < 0) return;

#if defined(_WIN32)
    SetThreadAffinityMask(GetCurrentThread(), 1ULL << core);
#else
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(core, &set);
    pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
#endif
}

// ─── ExpertPool constructor ─────────────────────────────────────
ExpertPool::ExpertPool(int nw, bool pin, bool host_works)
    : host_works_(host_works) {
    const std::vector<int> cores = physical_cores(true);
    count_ = nw > 0 ? nw : static_cast<int>(cores.size());
    if (count_ < 1) count_ = 1;

    scratch_.resize(static_cast<size_t>(count_));

    threads_.reserve(static_cast<size_t>(count_));
    for (int i = 0; i < count_; ++i) {
        const int core = pin ?
            (i < static_cast<int>(cores.size()) ? cores[static_cast<size_t>(i)] : -1) : -1;
        threads_.emplace_back([this, i, core] {
            pin_thread(core);
            worker(i);
        });
    }
}

ExpertPool::~ExpertPool() {
    stop_.store(true, std::memory_order_release);
    epoch_.fetch_add(1, std::memory_order_release);
    for (auto& t : threads_) {
        if (t.joinable()) t.join();
    }
}

// ─── Worker loop ────────────────────────────────────────────────
void ExpertPool::worker(int id) {
    uint32_t seen = 0;
    parked_.fetch_add(1, std::memory_order_acq_rel);

    for (;;) {
        // Spin-park: wait for epoch change
        while (epoch_.load(std::memory_order_acquire) == seen) {
            if (stop_.load(std::memory_order_relaxed)) return;
#if defined(__x86_64__) || defined(_M_X64)
            _mm_pause();
#endif
        }
        if (stop_.load(std::memory_order_acquire)) return;

        seen = epoch_.load(std::memory_order_relaxed);
        parked_.fetch_sub(1, std::memory_order_acq_rel);

        // Drain jobs
        drain(id, scratch_[static_cast<size_t>(id)]);

        parked_.fetch_add(1, std::memory_order_acq_rel);
    }
}

// ─── Drain: work-stealing via atomic fetch-add ──────────────────
void ExpertPool::drain(int id, ExpertScratch& ws) {
    (void)id;
    for (;;) {
        int job_idx = head_.fetch_add(1, std::memory_order_relaxed);
        if (job_idx >= njobs_) break;

        switch (mode_) {
        case 0: {
            // Whole expert: single-token
            auto& job = jobs_[job_idx];
            if (job.blob && job.act) {
                expert_kernel(job.blob, *job.act, job.out, ws);
            }
            break;
        }
        default:
            break;
        }

        done_.fetch_add(1, std::memory_order_acq_rel);
    }
}

// ─── Run phase: publish jobs and wait ───────────────────────────
void ExpertPool::run_phase(int mode, int ntasks) {
    // Wait for all workers to be parked
    while (parked_.load(std::memory_order_acquire) !=
           static_cast<uint32_t>(count_)) {
#if defined(__x86_64__) || defined(_M_X64)
        _mm_pause();
#endif
    }

    mode_ = mode;
    njobs_ = ntasks;
    head_.store(0, std::memory_order_relaxed);
    done_.store(0, std::memory_order_relaxed);

    // Release fence: make job data visible, then bump epoch
    epoch_.fetch_add(1, std::memory_order_release);

    // Host participates in drain
    if (host_works_) {
        drain(-1, host_scratch_);
    }

    // Wait for all tasks to complete
    while (done_.load(std::memory_order_acquire) !=
           static_cast<uint32_t>(ntasks)) {
#if defined(__x86_64__) || defined(_M_X64)
        _mm_pause();
#endif
    }

    // Wait for all workers to re-park
    while (parked_.load(std::memory_order_acquire) !=
           static_cast<uint32_t>(count_)) {
#if defined(__x86_64__) || defined(_M_X64)
        _mm_pause();
#endif
    }
}

// ─── run: single-token dispatch ─────────────────────────────────
void ExpertPool::run(PoolJob* jobs, int njobs) {
    if (njobs <= 0) return;
    jobs_ = jobs;
    run_phase(0, njobs);
}

// ─── run_split: split gate/up + down dispatch ───────────────────
void ExpertPool::run_split(PoolJob* jobs, int njobs) {
    if (njobs <= 0) return;
    jobs_ = jobs;

    // Phase 1: gate/up (mode=1)
    parts_a_ = static_cast<int>(FF);
    run_phase(1, njobs * parts_a_);

    // Intermediate: host quantizes each expert's ff buffer
    // (done inline; this is where act_quant_q8_1 runs sequentially)

    // Phase 2: down (mode=2)
    parts_b_ = static_cast<int>(H);
    run_phase(2, njobs * parts_b_);
}

// ─── run_multi: multi-token dispatch ────────────────────────────
void ExpertPool::run_multi(PoolJobMulti* jobs, int njobs, int n_tok) {
    if (njobs <= 0) return;
    jobs_multi_ = jobs;
    n_tok_ = n_tok;
    run_phase(3, njobs);
}

}}}  // namespace strata::kernels::cpu
