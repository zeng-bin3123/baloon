// bl_expert_pool_protocol.h - CPU expert thread pool: workers park on an epoch counter, drain a shared claim
// counter, and the host optionally drains too. Plus the pure partition / walk / batching rules.
#pragma once
#include <atomic>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>
#include "bl_nat.h"

namespace baloon::expert_pool {

// Core list after the optional host-core skip.
inline std::vector<nat> skipFirst(bool skip, const std::vector<nat>& cores) {
  if (skip && !cores.empty()) return std::vector<nat>(cores.begin() + 1, cores.end());
  return cores;
}

// Row-range partition [lo(i), lo(i+1)) of `rows` into `parts`.
constexpr nat lo(nat rows, nat parts, nat i) { return ndiv(rows * i, parts); }

// One step of the per-task expert walk: from global row r (< g1) in expert row r0 (< per), the run length
// inside the expert never leaves the expert and never overshoots the task end g1.
constexpr nat walkRun(nat per, nat r0, nat g1, nat r) { return nmin(per, r0 + (g1 - r)) - r0; }

constexpr nat partsA(nat threads, nat n) { return nmax(1, ndiv(3 * threads + n - 1, n)); }

constexpr nat perOf(nat ff, nat h, nat mode) { return (mode == 3 || mode == 5) ? ff : h; }
constexpr bool useRun(nat n, nat kMax, nat nWorkers, bool oracle) { return n > kMax || nWorkers == 1 || oracle; }
constexpr nat batchLen(nat kMax, nat n, nat b0) { return nmin(kMax, sat_sub(n, b0)); }

// Park accounting state.
struct PState { nat parked, draining; };
constexpr PState leave(PState s) { return {s.parked - 1, s.draining + 1}; }
constexpr PState arrive(PState s) { return {s.parked + 1, s.draining - 1}; }

class ExpertPool {
 public:
  // nworkers background workers (the host may additionally drain inside run()).
  explicit ExpertPool(nat nworkers) : n_(nworkers) {
    for (nat i = 0; i < n_; ++i) th_.emplace_back([this] { worker(); });
    std::unique_lock<std::mutex> lk(m_);
    cv_.wait(lk, [&] { return parked_ == n_; });  // constructed => all parked
  }
  ~ExpertPool() {
    { std::lock_guard<std::mutex> lk(m_); stop_ = true; ++epoch_; }
    cv_.notify_all();
    for (auto& t : th_) t.join();
  }
  nat workers() const { return n_; }
  nat parkedNow() { std::lock_guard<std::mutex> lk(m_); return parked_; }
  // Peak of (parked + draining) observed; must always equal workers().
  bool accountingHeld() const { return accounting_ok_.load(); }

  // Run njobs jobs: workers + host drain the claim counter; returns when done_counter == njobs.
  void run(nat njobs, const std::function<void(nat)>& job, bool hostDrains = true) {
    {
      std::lock_guard<std::mutex> lk(m_);
      job_ = &job; njobs_ = njobs; head_.store(0); done_.store(0); acked_ = 0;
      ++epoch_;
    }
    cv_.notify_all();
    if (hostDrains) drain();
    std::unique_lock<std::mutex> lk(m_);
    done_cv_.wait(lk, [&] { return done_.load() >= njobs_; });
    // Quiescence: every worker has seen this epoch (acked) and is parked again, so none can wake late.
    cv_.wait(lk, [&] { return acked_ == n_ && parked_ == n_; });
  }

 private:
  void drain() {
    for (;;) {
      nat i = head_.fetch_add(1);
      if (i >= njobs_) break;  // claim_stop
      (*job_)(i);
      if (done_.fetch_add(1) + 1 >= njobs_) { std::lock_guard<std::mutex> lk(m_); done_cv_.notify_all(); }
    }
  }
  void worker() {
    nat seen = 0;
    std::unique_lock<std::mutex> lk(m_);
    ++parked_; cv_.notify_all();
    for (;;) {
      cv_.wait(lk, [&] { return epoch_ != seen; });
      seen = epoch_;
      if (stop_) return;
      ++acked_;
      --parked_; ++draining_;
      if (parked_ + draining_ != n_) accounting_ok_.store(false);
      lk.unlock();
      drain();
      lk.lock();
      --draining_; ++parked_;
      if (parked_ + draining_ != n_) accounting_ok_.store(false);
      cv_.notify_all();
    }
  }

  nat n_;
  std::vector<std::thread> th_;
  std::mutex m_;
  std::condition_variable cv_, done_cv_;
  nat parked_ = 0, draining_ = 0, epoch_ = 0, acked_ = 0;
  bool stop_ = false;
  std::atomic<nat> head_{0}, done_{0};
  nat njobs_ = 0;
  const std::function<void(nat)>* job_ = nullptr;
  std::atomic<bool> accounting_ok_{true};
};

}  // namespace baloon::expert_pool
