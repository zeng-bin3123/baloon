#pragma once
// strata/core/graph.hpp — CUDA graph capture, RAII ownership, registry
// Target: Iluvatar BI-V100 (ivcore10, CUDA 10.2 compat)

#include <cuda_runtime.h>
#include <cstddef>
#include <functional>
#include <map>
#include <string>
#include <utility>

namespace strata {

// ─── LayerType ───────────────────────────────────────────────────────
enum class LayerType : int { GDN = 0, QSA = 1 };

inline const char* to_string(LayerType t) {
    return t == LayerType::GDN ? "GDN" : "QSA";
}

// ─── CapturedGraph ───────────────────────────────────────────────────
// Move-only RAII wrapper around a CUDA graph execution handle.
// Captures a stream region into a cudaGraph, instantiates an exec,
// and replays it on demand.  No double-destroy of cudaGraphExec_t.
class CapturedGraph {
public:
    CapturedGraph() = default;
    ~CapturedGraph() { reset(); }

    CapturedGraph(const CapturedGraph&) = delete;
    CapturedGraph& operator=(const CapturedGraph&) = delete;

    CapturedGraph(CapturedGraph&& o) noexcept
        : graph_(o.graph_), exec_(o.exec_), done_(o.done_), node_count_(o.node_count_) {
        o.graph_ = nullptr;
        o.exec_ = nullptr;
        o.done_ = nullptr;
        o.node_count_ = 0;
    }

    CapturedGraph& operator=(CapturedGraph&& o) noexcept {
        if (this != &o) {
            reset();
            graph_ = o.graph_;
            exec_ = o.exec_;
            done_ = o.done_;
            node_count_ = o.node_count_;
            o.graph_ = nullptr;
            o.exec_ = nullptr;
            o.done_ = nullptr;
            o.node_count_ = 0;
        }
        return *this;
    }

    // Start stream capture.  Returns false on CUDA error.
    bool begin(cudaStream_t stream, std::string& err);

    // End capture and instantiate.  Returns false if node_count == 0 or CUDA error.
    bool end(cudaStream_t stream, std::string& err);

    // Launch the captured graph on the given stream.
    // Pre: exec_ != nullptr (i.e. valid() == true).
    bool launch(cudaStream_t stream, std::string& err) const;

    // Poll the done_ event with the given timeout in milliseconds.
    // Returns false on timeout or CUDA error.
    bool wait_ms(int timeout_ms) const;

    size_t nodes() const { return node_count_; }
    bool valid() const { return exec_ != nullptr; }

private:
    void reset();

    cudaGraph_t graph_ = nullptr;
    cudaGraphExec_t exec_ = nullptr;
    cudaEvent_t done_ = nullptr;
    size_t node_count_ = 0;
};

// ─── GraphRegistry ───────────────────────────────────────────────────
// Keyed cache of CapturedGraphs: (LayerType, n_tokens) → CapturedGraph.
// `record` is idempotent — second call with same key is a no-op.
class GraphRegistry {
public:
    explicit GraphRegistry(cudaStream_t stream) : stream_(stream) {}

    // Capture `body` into a graph for the given key, or no-op if already present.
    bool record(LayerType layer, int n_tokens,
                const std::function<void()>& body, std::string& err);

    // Look up a previously captured graph.  Returns nullptr if not found.
    const CapturedGraph* find(LayerType layer, int n_tokens) const;

    // Launch the graph for the given key and wait up to timeout_ms.
    bool launch(LayerType layer, int n_tokens, int timeout_ms, std::string& err) const;

    size_t size() const { return graphs_.size(); }
    size_t captures() const { return capture_count_; }

private:
    using Key = std::pair<int, int>;

    cudaStream_t stream_ = nullptr;
    std::map<Key, CapturedGraph> graphs_;
    size_t capture_count_ = 0;
};

}  // namespace strata

