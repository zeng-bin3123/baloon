// graph.cpp - implementation of rt_graph.hpp.
#include "baloon/rt_graph.hpp"

#include <cstdio>

namespace baloon {
namespace rt {

bool graph_fail(const char* msg) { std::fprintf(stderr, "graph: %s\n", msg); return false; }

void ReplayGraph::begin() {
    ops_.clear(); recording_ = true; ready_ = false; ran_ = false; failed_ = false;
}

bool ReplayGraph::push(StreamOp op) {
    if (!recording_ || !op) return false;
    ops_.push_back(std::move(op));
    return true;
}

bool ReplayGraph::end() {
    if (!recording_) return false;
    recording_ = false; ready_ = true;
    return true;
}

bool ReplayGraph::launch(cudaStream_t s) {
    if (!ready_) return graph_fail("launch before end()");
    if (!e0_) {
        if (cudaEventCreate(&e0_) != cudaSuccess || cudaEventCreate(&e1_) != cudaSuccess) { cudaGetLastError(); reset(); return graph_fail("event create"); }
    }
    ran_ = false; failed_ = false;
    if (cudaEventRecord(e0_, s) != cudaSuccess) { cudaGetLastError(); failed_ = true; return graph_fail("event record"); }
    for (auto& op : ops_) {
        if (op(s) != cudaSuccess) { cudaGetLastError(); failed_ = true; return graph_fail("op failed"); }
    }
    if (cudaEventRecord(e1_, s) != cudaSuccess) { cudaGetLastError(); failed_ = true; return graph_fail("event record"); }
    ran_ = true; ++launches_;
    return true;
}

double ReplayGraph::wait_ms() {
    if (!ran_ || failed_ || !e1_) return -1.0;
    if (cudaEventSynchronize(e1_) != cudaSuccess) { cudaGetLastError(); return -1.0; }
    float ms = 0;
    if (cudaEventElapsedTime(&ms, e0_, e1_) != cudaSuccess) { cudaGetLastError(); return -1.0; }
    return ms;
}

void ReplayGraph::reset() {
    ops_.clear(); recording_ = ready_ = ran_ = failed_ = false;
    if (e0_) { cudaEventDestroy(e0_); e0_ = nullptr; }
    if (e1_) { cudaEventDestroy(e1_); e1_ = nullptr; }
}

bool GraphBook::record(const std::string& name, const std::function<void(ReplayGraph&)>& body) {
    auto g = std::make_unique<ReplayGraph>();
    g->begin();
    body(*g);
    if (!g->end()) return false;
    g_[name] = std::move(g);
    return true;
}

ReplayGraph* GraphBook::find(const std::string& name) {
    auto it = g_.find(name);
    return it == g_.end() ? nullptr : it->second.get();
}

bool GraphBook::launch(const std::string& name, cudaStream_t s) {
    ReplayGraph* g = find(name);
    return g && g->launch(s);
}

}  // namespace rt
}  // namespace baloon
