// graph.cpp - implementation of core_graph_dev.hpp: capture validity, event-bracketed replay, bounded event polling and
// the capture-once registry.
#include "baloon/core_graph_dev.hpp"

#include <chrono>
#include <cstdio>

namespace baloon {
namespace core {
namespace dev {

#ifndef BALOON_DEV_HAVE_CUDA
namespace {
cudaError_t cudaEventCreate(cudaEvent_t*) { return cudaErrorInvalidValue; }
cudaError_t cudaEventDestroy(cudaEvent_t) { return cudaErrorInvalidValue; }
cudaError_t cudaEventRecord(cudaEvent_t, cudaStream_t) { return cudaErrorInvalidValue; }
cudaError_t cudaEventQuery(cudaEvent_t) { return cudaErrorInvalidValue; }
cudaError_t cudaGetLastError() { return cudaSuccess; }
}  // namespace
#endif

namespace {
bool fail(const char* msg) { std::fprintf(stderr, "graph: %s\n", msg); return false; }
double now_ms() {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}
}  // namespace

void CapturedGraph::begin() { ops_.clear(); recording_ = true; ready_ = false; ran_ = false; failed_ = false; }

bool CapturedGraph::push(StreamOp op) {
    if (!recording_ || !op) return false;
    ops_.push_back(std::move(op));
    return true;
}

bool CapturedGraph::end() {
    if (!recording_) return fail("end() without begin()");
    recording_ = false;
    if (ops_.empty()) { ready_ = false; return fail("empty capture"); }   // endOk 0 = false
    ready_ = true;
    return true;
}

bool CapturedGraph::launch(cudaStream_t s) {
    if (!ready_) return fail("launch before end()");
    if (!e0_) {
        if (cudaEventCreate(&e0_) != cudaSuccess || cudaEventCreate(&e1_) != cudaSuccess) {
            cudaGetLastError(); reset(); return fail("event create");
        }
    }
    ran_ = false; failed_ = false;
    if (cudaEventRecord(e0_, s) != cudaSuccess) { cudaGetLastError(); failed_ = true; return fail("event record"); }
    for (auto& op : ops_) {
        if (op(s) != cudaSuccess) { cudaGetLastError(); failed_ = true; return fail("node failed"); }
    }
    if (cudaEventRecord(e1_, s) != cudaSuccess) { cudaGetLastError(); failed_ = true; return fail("event record"); }
    ran_ = true; ++launches_;
    return true;
}

bool CapturedGraph::bounded_poll(const std::function<cudaError_t()>& query, double timeout_ms) {
    if (!(timeout_ms > 0)) return false;                     // an expired deadline never reports success
    const double t0 = now_ms();
    while (now_ms() - t0 < timeout_ms) {
        const cudaError_t e = query();
        if (e == cudaSuccess) return true;
        if (e != cudaErrorNotReady) { cudaGetLastError(); return false; }
        cudaGetLastError();
    }
    return false;
}

bool CapturedGraph::wait_ms(double timeout_ms) {
    if (!ran_ || failed_ || !e1_) return false;
    return bounded_poll([this] { return cudaEventQuery(e1_); }, timeout_ms);
}

void CapturedGraph::reset() {
    ops_.clear(); recording_ = ready_ = ran_ = failed_ = false;
    if (e0_) { cudaEventDestroy(e0_); e0_ = nullptr; }
    if (e1_) { cudaEventDestroy(e1_); e1_ = nullptr; }
}

bool GraphRegistry::record(int kind, int n_tokens, const std::function<void(CapturedGraph&)>& body) {
    const Key k{kind, n_tokens};
    if (g_.count(k)) return true;                            // record_found: no body, no capture
    auto g = std::make_unique<CapturedGraph>();
    g->begin();
    body(*g);
    if (!g->end()) return false;
    g_[k] = std::move(g);
    ++captures_;
    return true;
}

CapturedGraph* GraphRegistry::find(int kind, int n_tokens) {
    auto it = g_.find(Key{kind, n_tokens});
    return it == g_.end() ? nullptr : it->second.get();
}

bool GraphRegistry::launch(int kind, int n_tokens, cudaStream_t s, double timeout_ms) {
    CapturedGraph* g = find(kind, n_tokens);
    return g && g->launch(s) && g->wait_ms(timeout_ms);
}

}  // namespace dev
}  // namespace core
}  // namespace baloon
