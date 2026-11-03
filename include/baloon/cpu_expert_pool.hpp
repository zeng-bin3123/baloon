// cpu_expert_pool.hpp - CPU expert thread pool arithmetic: task splitting, row-range segmentation across experts,
// Q2_0 / native routing by phase mode, expert batching, park accounting and core selection.
#pragma once
#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <fstream>
#include <functional>
#include <mutex>
#include <string>
#include <thread>
#include <vector>
#ifdef __linux__
#include <pthread.h>
#include <sched.h>
#endif

namespace baloon {
namespace cpu_pool {

constexpr uint64_t split_pt(uint64_t rows, uint64_t tasks, uint64_t i) { return rows * i / tasks; }

constexpr uint64_t threads(uint64_t n_workers, bool host_works) { return n_workers + (host_works ? 1 : 0); }
constexpr uint64_t tasks_per_phase(uint64_t n_workers, bool host_works) { return 3 * threads(n_workers, host_works); }
constexpr uint64_t parts_per_expert(uint64_t thr, uint64_t n) { return std::max<uint64_t>(1, (3 * thr + n - 1) / n); }

struct Seg { uint64_t expert, r0, r1; };

// Cut global rows [g0, g1) across experts of `per` rows each. Every step advances and never overshoots.
inline std::vector<Seg> segment(uint64_t per, uint64_t g0, uint64_t g1) {
    std::vector<Seg> out;
    if (per == 0) return out;
    uint64_t r = g0;
    while (r < g1) {
        uint64_t e = r / per, r0 = r % per;
        uint64_t r1 = std::min(per, r0 + (g1 - r));
        out.push_back({e, r0, r1});
        r += r1 - r0;
    }
    return out;
}

enum class Path { q2_gu, native_gu, q2_down, native_down, other };

// Phase mode 5 = gate/up, 6 = down; type id 42 = Q2_0.
constexpr Path route(uint64_t mode, uint64_t gu, uint64_t d) {
    return mode == 5 ? (gu == 42 ? Path::q2_gu : Path::native_gu)
         : mode == 6 ? (d == 42 ? Path::q2_down : Path::native_down)
                     : Path::other;
}

constexpr uint64_t batch_len(uint64_t cap, uint64_t n, uint64_t b0) { return std::min(cap, n - b0); }

// ---- claiming and completion handover -----------------------------------------------------------------------
// The k-th fetch_add on a counter that starts at 0 returns k; a claim is a job iff k < n.
constexpr uint64_t claim_index(uint64_t k) { return k; }
constexpr bool valid_claim(uint64_t i, uint64_t njobs) { return i < njobs; }
// Workers currently outside the park. With every worker parked nobody can race a reset of the claim counter.
constexpr uint64_t in_drain(uint64_t n_workers, uint64_t parked) { return n_workers - parked; }

// ---- worker count, batching -------------------------------------------------------------------------------------
// The request if positive, else every remaining core; never fewer than one worker.
constexpr uint64_t n_workers(uint64_t req, uint64_t ncores) {
    return std::max<uint64_t>(1, req > 0 ? req : ncores);
}
constexpr uint64_t kMaxSplitMulti = 96;  // native batches cover at most this many experts

// Batch sizes of `for (b0 = 0; b0 < n; b0 += k) nb = min(k, n - b0)`.
inline std::vector<uint64_t> chunks(uint64_t k, uint64_t n) {
    std::vector<uint64_t> out;
    if (k == 0) return out;
    for (uint64_t b0 = 0; b0 < n; b0 += k) out.push_back(batch_len(k, n, b0));
    return out;
}

// run_split task i of n*parts names (expert i / parts, part i % parts).
constexpr uint64_t task_expert(uint64_t i, uint64_t parts) { return i / parts; }
constexpr uint64_t task_part(uint64_t i, uint64_t parts) { return i % parts; }

// First core is reserved for the host.
inline std::vector<int> skip_first(const std::vector<int>& cores) {
    return cores.empty() ? cores : std::vector<int>(cores.begin() + 1, cores.end());
}
inline int core_for(const std::vector<int>& cores, uint64_t i) { return i < cores.size() ? cores[i] : -1; }
constexpr bool should_pin(int core) { return core >= 0; }

// Parse a Linux cpu list such as "0-3,8,10-11". Malformed input yields an empty list.
inline std::vector<int> parse_cpu_list(const std::string& s) {
    std::vector<int> out;
    size_t i = 0;
    auto num = [&](int* v) {
        if (i >= s.size() || s[i] < '0' || s[i] > '9') return false;
        long x = 0;
        while (i < s.size() && s[i] >= '0' && s[i] <= '9') { x = x * 10 + (s[i] - '0'); if (x > (1 << 20)) return false; ++i; }
        *v = (int)x; return true;
    };
    while (i < s.size() && s[i] != '\n') {
        int a = 0, b = 0;
        if (!num(&a)) return {};
        b = a;
        if (i < s.size() && s[i] == '-') { ++i; if (!num(&b) || b < a) return {}; }
        for (int c = a; c <= b; ++c) out.push_back(c);
        if (i < s.size() && s[i] == ',') ++i;
        else if (i < s.size() && s[i] != '\n') return {};
    }
    return out;
}

// One logical cpu per physical core: keep a cpu iff it is the lowest id of its hyper-thread siblings.
inline bool is_core_leader(int cpu, const std::vector<int>& siblings) {
    return siblings.empty() || cpu == *std::min_element(siblings.begin(), siblings.end());
}

// Physical cores usable by this process (affinity mask, one thread per core); the first is dropped on request.
inline std::vector<int> physical_cores(bool skip_first_core) {
    std::vector<int> cpus;
#ifdef __linux__
    cpu_set_t set; CPU_ZERO(&set);
    if (sched_getaffinity(0, sizeof set, &set) == 0)
        for (int c = 0; c < CPU_SETSIZE; ++c) if (CPU_ISSET(c, &set)) cpus.push_back(c);
#endif
    if (cpus.empty()) cpus.push_back(0);
    std::vector<int> cores;
    for (int c : cpus) {
        std::ifstream f("/sys/devices/system/cpu/cpu" + std::to_string(c) + "/topology/thread_siblings_list");
        std::string line;
        std::vector<int> sib;
        if (f && std::getline(f, line)) sib = parse_cpu_list(line);
        if (is_core_leader(c, sib)) cores.push_back(c);
    }
    if (cores.empty()) cores = cpus;
    return skip_first_core ? skip_first(cores) : cores;
}

// ---- the pool ---------------------------------------------------------------------------------------------------
// Jobs 0..n-1 are claimed with head_.fetch_add(1) (valid iff index < n, so each index is handed out once). Workers
// arrive at the park once before their first wait. run() resets the counters only while every worker is parked, and
// returns only when done == n AND parked == n_: done alone would leave a worker still inside the drain.
class Pool {
public:
    // n_workers background threads; core i (if any) pins worker i. host_works: run() also drains.
    explicit Pool(unsigned nw, std::vector<int> cores = {}, bool host_works = true)
        : n_(nw), host_works_(host_works), cores_(std::move(cores)) {
        for (unsigned i = 0; i < n_; ++i) threads_.emplace_back([this, i] { worker(i); });
        while (parked_.load(std::memory_order_acquire) != n_) std::this_thread::yield();  // parked == n_ from construction
    }
    ~Pool() {
        {
            std::lock_guard<std::mutex> lk(m_);
            stop_ = true;
        }
        cv_.notify_all();
        for (auto& t : threads_) t.join();
    }
    Pool(const Pool&) = delete;
    Pool& operator=(const Pool&) = delete;

    unsigned workers() const { return n_; }
    uint64_t thread_count() const { return threads(n_, host_works_); }
    bool host_works() const { return host_works_; }

    // Run fn(i) exactly once for every i in [0, n). Not reentrant; call from one thread at a time.
    void run(uint64_t n, const std::function<void(uint64_t)>& fn) {
        if (n == 0) return;
        if (n_ == 0 && !host_works_) { for (uint64_t i = 0; i < n; ++i) fn(i); return; }
        {
            std::lock_guard<std::mutex> lk(m_);
            fn_ = &fn; njobs_ = n;
            head_.store(0, std::memory_order_relaxed);   // safe: every worker is parked
            done_.store(0, std::memory_order_relaxed);
            parked_.store(0, std::memory_order_relaxed); // counts workers that finished this batch's drain
            ++epoch_;
        }
        cv_.notify_all();
        if (host_works_) drain();
        while (done_.load(std::memory_order_acquire) != n || parked_.load(std::memory_order_acquire) != n_)
            std::this_thread::yield();
    }

    // Each expert cut into `parts` row ranges of [0, ff); task i -> (expert i / parts, part i % parts).
    void run_split(uint64_t n_experts, uint64_t ff, const std::function<void(uint64_t, uint64_t, uint64_t)>& fn) {
        if (n_experts == 0) return;
        const uint64_t parts = parts_per_expert(thread_count(), n_experts);
        run(n_experts * parts, [&](uint64_t i) {
            const uint64_t e = task_expert(i, parts), p = task_part(i, parts);
            const uint64_t r0 = split_pt(ff, parts, p), r1 = split_pt(ff, parts, p + 1);
            if (r0 < r1) fn(e, r0, r1);
        });
    }

    // `per` rows for each of n experts cut as one flat range into 3*threads tasks; each task walks expert by expert.
    void run_multi(uint64_t n_experts, uint64_t per, const std::function<void(const Seg&)>& fn) {
        const uint64_t mrows = n_experts * per, mtasks = tasks_per_phase(n_, host_works_);
        if (mrows == 0 || mtasks == 0) return;
        run(mtasks, [&](uint64_t i) {
            for (const Seg& sg : segment(per, split_pt(mrows, mtasks, i), split_pt(mrows, mtasks, i + 1))) fn(sg);
        });
    }

private:
    void drain() {
        for (;;) {
            const uint64_t k = head_.fetch_add(1, std::memory_order_acq_rel);
            if (!valid_claim(claim_index(k), njobs_)) break;
            (*fn_)(k);
            done_.fetch_add(1, std::memory_order_acq_rel);
        }
    }
    void worker(unsigned idx) {
#ifdef __linux__
        if (idx < cores_.size() && should_pin(cores_[idx])) {
            cpu_set_t set; CPU_ZERO(&set); CPU_SET(cores_[idx], &set);
            (void)pthread_setaffinity_np(pthread_self(), sizeof set, &set);  // best effort
        }
#else
        (void)idx;
#endif
        uint64_t seen = 0;
        parked_.fetch_add(1, std::memory_order_acq_rel);  // arrive at the park before the first wait
        for (;;) {
            {
                std::unique_lock<std::mutex> lk(m_);
                cv_.wait(lk, [&] { return stop_ || epoch_ != seen; });
                if (stop_) return;
                seen = epoch_;
            }
            drain();
            parked_.fetch_add(1, std::memory_order_acq_rel);  // done-then-parked handover
        }
    }

    unsigned n_;
    bool host_works_;
    std::vector<int> cores_;
    std::vector<std::thread> threads_;
    std::mutex m_;
    std::condition_variable cv_;
    bool stop_ = false;
    uint64_t epoch_ = 0;
    const std::function<void(uint64_t)>* fn_ = nullptr;
    uint64_t njobs_ = 0;
    std::atomic<uint64_t> head_{0}, done_{0}, parked_{0};
};

}  // namespace cpu_pool
}  // namespace baloon
