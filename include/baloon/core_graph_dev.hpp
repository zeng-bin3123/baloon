// core_graph_dev.hpp - device-backed capture-once / replay-many graph registry (GPU model of core_graph.hpp).
// A CapturedGraph records an ordered list of stream operations (kernel launches / async copies) between begin() and
// end(); end() rejects an empty capture. launch() replays the list on a stream bracketed by two events; wait_ms()
// polls the closing event under a deadline and never reports a completion that did not occur. A GraphRegistry owns
// graphs keyed by (layer kind, token count): record() on an existing key is a no-op that does not run the body, so
// captures() == size() always holds.
#pragma once
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <utility>
#include <vector>

// The project-wide host build compiles src/core/*.cpp without the CUDA include path; there the runtime types degrade to
// opaque stand-ins and every device operation reports failure. Device builds (-x cuda) get the real runtime.
#if defined(__has_include)
#if __has_include(<cuda_runtime.h>)
#define BALOON_DEV_HAVE_CUDA 1
#endif
#endif
#ifdef BALOON_DEV_HAVE_CUDA
#include <cuda_runtime.h>
#else
typedef int cudaError_t;
typedef void* cudaStream_t;
typedef void* cudaEvent_t;
enum { cudaSuccess = 0, cudaErrorNotReady = 600, cudaErrorInvalidValue = 1 };
#endif

namespace baloon {
namespace core {
namespace dev {

using StreamOp = std::function<cudaError_t(cudaStream_t)>;

class CapturedGraph {
public:
    CapturedGraph() = default;
    ~CapturedGraph() { reset(); }
    CapturedGraph(const CapturedGraph&) = delete;
    CapturedGraph& operator=(const CapturedGraph&) = delete;

    void begin();                 // fresh recording: drops old nodes, marks not-ready
    bool push(StreamOp op);       // only while recording
    bool end();                   // false (and not ready) when not recording or when zero nodes were recorded
    bool ready() const { return ready_; }
    size_t nodes() const { return ops_.size(); }
    uint64_t launches() const { return launches_; }
    // Replays all nodes in order on `s` between two events; stops at the first failing node (later nodes not issued).
    bool launch(cudaStream_t s = nullptr);
    // Polls the closing event until it completes (true), a real error occurs (false) or `timeout_ms` expires (false).
    bool wait_ms(double timeout_ms);
    // Deadline polling over an arbitrary query (cudaSuccess = done, cudaErrorNotReady = keep polling, else false).
    static bool bounded_poll(const std::function<cudaError_t()>& query, double timeout_ms);
    void reset();                 // drops nodes and events

private:
    std::vector<StreamOp> ops_;
    bool recording_ = false, ready_ = false, ran_ = false, failed_ = false;
    uint64_t launches_ = 0;
    cudaEvent_t e0_ = nullptr, e1_ = nullptr;
};

class GraphRegistry {
public:
    using Key = std::pair<int, int>;   // (layer kind: 0 = GDN, 1 = QSA ; n_tokens)
    // Captures `body` under `key` unless the key exists (then true, body not run). False if the capture is invalid.
    bool record(int kind, int n_tokens, const std::function<void(CapturedGraph&)>& body);
    CapturedGraph* find(int kind, int n_tokens);
    // Replays the graph under the key and waits for it (bounded). False when missing or failed.
    bool launch(int kind, int n_tokens, cudaStream_t s = nullptr, double timeout_ms = 10000.0);
    size_t size() const { return g_.size(); }
    uint64_t captures() const { return captures_; }
private:
    std::map<Key, std::unique_ptr<CapturedGraph>> g_;
    uint64_t captures_ = 0;
};

}  // namespace dev
}  // namespace core
}  // namespace baloon
