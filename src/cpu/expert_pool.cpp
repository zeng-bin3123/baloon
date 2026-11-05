// expert_pool.cpp - CPU expert pool. Protocol: workers park; the host publishes a batch only when parked == n_workers
// (resetting head and done, bumping the epoch), waits for done == n_tasks and then for parked == n_workers again, so
// nobody can still claim from a finished batch when the next one is published.
#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif
#include "baloon/kern_cpu_pool.hpp"

#include <algorithm>
#include <cstdio>
#include <fstream>
#include <set>
#include <string>
#include <utility>

#ifdef __linux__
#include <pthread.h>
#include <sched.h>
#endif

namespace baloon {
namespace kern {

namespace {
#ifdef __linux__
thread_local bool g_saved = false;
thread_local cpu_set_t g_orig;

int read_int(const std::string& path, int dflt) {
    std::ifstream f(path);
    int v = dflt;
    if (f >> v) return v;
    return dflt;
}
#endif
}  // namespace

std::vector<int> physical_cores() {
    std::vector<int> out;
#ifdef __linux__
    cpu_set_t set;
    CPU_ZERO(&set);
    if (sched_getaffinity(0, sizeof(set), &set) == 0) {
        std::set<std::pair<int, int>> seen;
        for (int c = 0; c < CPU_SETSIZE; ++c) {
            if (!CPU_ISSET(c, &set)) continue;
            const std::string base = "/sys/devices/system/cpu/cpu" + std::to_string(c) + "/topology/";
            const int core = read_int(base + "core_id", c), pkg = read_int(base + "physical_package_id", 0);
            if (seen.insert({pkg, core}).second) out.push_back(c);
        }
    }
#endif
    if (out.empty()) {
        const unsigned n = std::max(1u, std::thread::hardware_concurrency());
        for (unsigned i = 0; i < n; ++i) out.push_back((int)i);
    }
    return out;
}

static bool pin_this_thread(int cpu) {
#ifdef __linux__
    if (cpu < 0 || cpu >= CPU_SETSIZE) return false;
    if (!g_saved) {
        if (pthread_getaffinity_np(pthread_self(), sizeof(g_orig), &g_orig) != 0) return false;
        g_saved = true;
    }
    cpu_set_t s;
    CPU_ZERO(&s);
    CPU_SET(cpu, &s);
    return pthread_setaffinity_np(pthread_self(), sizeof(s), &s) == 0;
#else
    (void)cpu;
    return false;
#endif
}

bool pin_current_thread(int cpu) { return pin_this_thread(cpu); }

void restore_thread_affinity() {
#ifdef __linux__
    if (g_saved) pthread_setaffinity_np(pthread_self(), sizeof(g_orig), &g_orig);
#endif
}

ExpertPool::ExpertPool(uint32_t requested, bool skip_first, bool host_works) : host_works_(host_works) {
    std::vector<int> cores = physical_cores();
    if (skip_first && !cores.empty()) cores.erase(cores.begin());   // reserveHost: exactly one core dropped
    nw_ = poolsched::pool_workers(requested, (uint32_t)cores.size());
    cpus_ = cores;
    for (uint32_t i = 0; i < nw_; ++i) th_.emplace_back([this, i] { worker(i); });
    std::unique_lock<std::mutex> lk(m_);
    cv_host_.wait(lk, [&] { return parked_.load() == nw_; });
}

ExpertPool::~ExpertPool() {
    {
        std::lock_guard<std::mutex> lk(m_);
        stop_ = true;
    }
    cv_work_.notify_all();
    for (auto& t : th_) t.join();
}

void ExpertPool::worker(uint32_t idx) {
    if (!cpus_.empty()) pin_this_thread(cpus_[idx % cpus_.size()]);
    uint64_t seen = 0;
    for (;;) {
        {
            std::unique_lock<std::mutex> lk(m_);
            parked_.fetch_add(1);
            cv_host_.notify_all();
            cv_work_.wait(lk, [&] { return stop_ || epoch_ != seen; });
            parked_.fetch_sub(1);
            if (stop_) return;
            seen = epoch_;
        }
        drain();
    }
}

void ExpertPool::drain() {
    const uint32_t n = ntasks_;
    for (;;) {
        const uint64_t i = head_.fetch_add(1);
        if (i >= n) break;                       // a claim is a task iff it is below the batch size
        exec(mode_, (uint32_t)i);
        if (done_.fetch_add(1) + 1 == n) {
            std::lock_guard<std::mutex> lk(m_);
            cv_host_.notify_all();
        }
    }
}

void ExpertPool::run_phase(uint32_t mode, uint32_t ntasks) {
    if (ntasks == 0) return;
    {
        std::unique_lock<std::mutex> lk(m_);
        cv_host_.wait(lk, [&] { return parked_.load() == nw_; });   // publish only when everyone is parked
        mode_ = mode;
        ntasks_ = ntasks;
        head_.store(0);
        done_.store(0);
        ++epoch_;
    }
    cv_work_.notify_all();
    if (host_works_) drain();
    std::unique_lock<std::mutex> lk(m_);
    cv_host_.wait(lk, [&] { return done_.load() == ntasks && parked_.load() == nw_; });
}

namespace {
inline uint32_t parts_for(uint32_t thr, uint32_t n) { return std::max<uint32_t>(1, (3 * thr + n - 1) / n); }
}

void ExpertPool::exec(uint32_t mode, uint32_t i) {
    switch (mode) {
        case 1: {   // whole expert
            const ExpertJob& j = jobs_[i];
            s2_expert_vnni(j.blob, j.x, j.out, *j.scratch);
            break;
        }
        case 2: case 3: {   // split: task i = expert e, part p
            const uint32_t e = i / parts_, p = i % parts_;
            const ExpertJob& j = jobs_[e];
            const uint32_t total = mode == 2 ? FF : H;
            const uint32_t r0 = (uint32_t)poolsched::row_start(total, parts_, p), r1 = (uint32_t)poolsched::row_start(total, parts_, p + 1);
            if (mode == 2) s2_expert_gu_rows(j.blob, j.scratch->x, r0, r1, j.scratch->act);
            else s2_expert_down_rows(j.blob, j.scratch->a, r0, r1, j.out);
            break;
        }
        case 5: case 6: {   // multi token: global row range across experts; route by type id (42 = Q2_0)
            const bool gu = mode == 5;
            uint64_t per = 0;
            // all jobs of a batch share the row count of the phase; take it from the first job
            const ExpertJobMulti& j0 = mjobs_[0];
            const bool q2 = (gu ? j0.gu_type : j0.d_type) == 42;
            per = q2 ? (gu ? FF : H) : (gu ? j0.n_ff : j0.n_embd);
            const uint64_t total = per * njobs_;
            const uint64_t g0 = poolsched::row_start(total, ntasks_, i), g1 = poolsched::row_start(total, ntasks_, i + 1);
            if (g0 >= g1) break;
            for (const auto& pc : poolsched::pieces(per, g0, g1)) {
                const ExpertJobMulti& j = mjobs_[pc.expert];
                const uint32_t r0 = (uint32_t)pc.r0, r1 = (uint32_t)pc.r1;
                if (q2) {
                    if (gu) s2_expert_gu_rows_multi(j.blob, j.scratch->x, j.nt, r0, r1, j.scratch->act);
                    else s2_expert_down_rows_multi(j.blob, j.scratch->a, j.nt, r0, r1, j.out, H);
                } else {
                    NativeScratch& ns = ns_[pc.expert];
                    if (gu) iq512_gu_rows(j.gu_type, j.gu_w, j.gu_row_bytes, j.n_ff, r0, r1, j.n_embd / 256, ns.x8.data(), j.nt, ns.act.data(), j.n_ff);
                    else iq512_rows(j.d_type, j.d_w, j.d_row_bytes, r0, r1, j.n_ff / 256, ns.a8.data(), j.nt, j.out, j.n_embd);
                }
            }
            break;
        }
        default: break;
    }
}

void ExpertPool::run(const ExpertJob* jobs, uint32_t n) {
    if (n == 0) return;
    jobs_ = jobs;
    njobs_ = n;
    run_phase(1, n);
}

bool ExpertPool::run_split(const ExpertJob* jobs, uint32_t n) {
    if (n == 0) return true;
    if (!poolsched::use_split(n, kMaxSplit, expert_oracle_q8_0_enabled())) return false;
    jobs_ = jobs;
    njobs_ = n;
    parts_ = parts_for(threads(), n);
    for (uint32_t e = 0; e < n; ++e) act_quant_q8_1(jobs[e].x, H, jobs[e].scratch->x.q, jobs[e].scratch->x.d, jobs[e].scratch->x.hx);
    run_phase(2, n * parts_);
    for (uint32_t e = 0; e < n; ++e) act_quant_q8_1(jobs[e].scratch->act, FF, jobs[e].scratch->a.q, jobs[e].scratch->a.d, jobs[e].scratch->a.hx);
    run_phase(3, n * parts_);
    return true;
}

bool ExpertPool::multi_impl(const ExpertJobMulti* jobs, uint32_t n, bool native_only) {
    if (n == 0) return true;
    if (!poolsched::use_split(n, kMaxSplitMulti, expert_oracle_q8_0_enabled())) return false;
    const bool q2 = jobs[0].gu_type == 42;
    for (uint32_t e = 0; e < n; ++e) {   // a batch is homogeneous: all Q2_0 or all native, same geometry
        if ((jobs[e].gu_type == 42) != q2 || jobs[e].nt == 0 || jobs[e].nt > MAXT) return false;
        if (q2 != (jobs[e].d_type == 42)) return false;
        if (!q2 && (jobs[e].n_ff != jobs[0].n_ff || jobs[e].n_embd != jobs[0].n_embd || jobs[e].n_ff % 256 || jobs[e].n_embd % 256)) return false;
    }
    if (native_only && q2) return false;
    mjobs_ = jobs;
    njobs_ = n;
    const uint32_t tasks = 3 * threads();
    if (q2) {
        for (uint32_t e = 0; e < n; ++e)
            for (uint32_t t = 0; t < jobs[e].nt; ++t)
                act_quant_q8_1(jobs[e].x + (size_t)t * H, H, jobs[e].scratch->x[t].q, jobs[e].scratch->x[t].d, jobs[e].scratch->x[t].hx);
        run_phase(5, tasks);
        for (uint32_t e = 0; e < n; ++e)
            for (uint32_t t = 0; t < jobs[e].nt; ++t)
                act_quant_q8_1(jobs[e].scratch->act[t], FF, jobs[e].scratch->a[t].q, jobs[e].scratch->a[t].d, jobs[e].scratch->a[t].hx);
        run_phase(6, tasks);
        return true;
    }
    ns_.assign(n, NativeScratch{});
    for (uint32_t e = 0; e < n; ++e) {
        const ExpertJobMulti& j = jobs[e];
        const uint32_t nbe = j.n_embd / 256, nbf = j.n_ff / 256;
        ns_[e].x8.resize((size_t)j.nt * nbe);
        ns_[e].a8.resize((size_t)j.nt * nbf);
        ns_[e].act.assign((size_t)j.nt * j.n_ff, 0.f);
        for (uint32_t t = 0; t < j.nt; ++t) iquant512::q8k_quantize(j.x + (size_t)t * j.n_embd, j.n_embd, ns_[e].x8.data() + (size_t)t * nbe);
    }
    run_phase(5, tasks);
    for (uint32_t e = 0; e < n; ++e) {
        const ExpertJobMulti& j = jobs[e];
        const uint32_t nbf = j.n_ff / 256;
        for (uint32_t t = 0; t < j.nt; ++t) iquant512::q8k_quantize(ns_[e].act.data() + (size_t)t * j.n_ff, j.n_ff, ns_[e].a8.data() + (size_t)t * nbf);
    }
    run_phase(6, tasks);
    return true;
}

bool ExpertPool::run_split_multi(const ExpertJobMulti* jobs, uint32_t n) { return multi_impl(jobs, n, false); }
bool ExpertPool::run_split_multi_native(const ExpertJobMulti* jobs, uint32_t n) { return multi_impl(jobs, n, true); }

}  // namespace kern
}  // namespace baloon
