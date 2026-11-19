// rt_graph.hpp - recorded launch sequences replayed on a stream with event timing. A CapturedGraph stores an ordered list
// of stream operations (kernel launches / async copies as callables) and replays them in recording order; a GraphRegistry
// owns named graphs.
#pragma once
#include <cuda_runtime.h>

#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace baloon {
namespace rt {

using StreamOp = std::function<cudaError_t(cudaStream_t)>;

class CapturedGraph {
public:
    CapturedGraph() = default;
    ~CapturedGraph() { reset(); }
    CapturedGraph(const CapturedGraph&) = delete;
    CapturedGraph& operator=(const CapturedGraph&) = delete;
    void begin();                      // starts a fresh recording (drops old ops, marks not-ready)
    bool push(StreamOp op);            // only while recording
    bool end();                        // finishes; false when not recording
    bool ready() const { return ready_; }
    size_t size() const { return ops_.size(); }
    uint64_t launches() const { return launches_; }
    // Replays all ops in order on `s` bracketed by two events. False when not ready or an op failed (the replay then
    // stops at the failing op; later ops are not issued).
    bool launch(cudaStream_t s = nullptr);
    // Blocks until the last launch finished; milliseconds between its events, or -1 when it never ran / failed.
    double wait_ms();
    void reset();                      // drops ops and events
private:
    std::vector<StreamOp> ops_;
    bool recording_ = false, ready_ = false, ran_ = false, failed_ = false;
    uint64_t launches_ = 0;
    cudaEvent_t e0_ = nullptr, e1_ = nullptr;
};

class GraphRegistry {
public:
    // Records `name` by running `body` between begin() and end(); an existing graph of that name is replaced.
    bool record(const std::string& name, const std::function<void(CapturedGraph&)>& body);
    CapturedGraph* find(const std::string& name);
    bool launch(const std::string& name, cudaStream_t s = nullptr);
    size_t size() const { return g_.size(); }
private:
    std::map<std::string, std::unique_ptr<CapturedGraph>> g_;
};

// Prints "graph: <msg>" to stderr and returns false (so callers can `return fail(...)`).
bool fail(const char* msg);

}  // namespace rt
}  // namespace baloon
