// strata/core/graph.cpp — CUDA graph capture and registry implementation
#include "strata/core/graph.hpp"
#include <chrono>
#include <cstdio>
#include <ctime>

namespace strata {

// ─── CapturedGraph ───────────────────────────────────────────────────

void CapturedGraph::reset() {
    if (exec_) { cudaGraphExecDestroy(exec_); exec_ = nullptr; }
    if (graph_) { cudaGraphDestroy(graph_); graph_ = nullptr; }
    if (done_) { cudaEventDestroy(done_); done_ = nullptr; }
    node_count_ = 0;
}

bool CapturedGraph::begin(cudaStream_t stream, std::string& err) {
    reset();
    cudaError_t rc = cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
    if (rc != cudaSuccess) {
        err = std::string("cudaStreamBeginCapture: ") + cudaGetErrorString(rc);
        return false;
    }
    return true;
}

bool CapturedGraph::end(cudaStream_t stream, std::string& err) {
    cudaError_t rc = cudaStreamEndCapture(stream, &graph_);
    if (rc != cudaSuccess) {
        err = std::string("cudaStreamEndCapture: ") + cudaGetErrorString(rc);
        return false;
    }

    // Count nodes
    size_t count = 0;
    rc = cudaGraphGetNodes(graph_, nullptr, &count);
    if (rc != cudaSuccess) {
        err = std::string("cudaGraphGetNodes: ") + cudaGetErrorString(rc);
        cudaGraphDestroy(graph_); graph_ = nullptr;
        return false;
    }
    node_count_ = count;

    if (node_count_ == 0) {
        err = "captured graph has 0 nodes";
        cudaGraphDestroy(graph_); graph_ = nullptr;
        return false;
    }

    // Instantiate
    rc = cudaGraphInstantiate(&exec_, graph_, nullptr, nullptr, 0);
    if (rc != cudaSuccess) {
        err = std::string("cudaGraphInstantiate: ") + cudaGetErrorString(rc);
        cudaGraphDestroy(graph_); graph_ = nullptr;
        return false;
    }

    // Create completion event
    rc = cudaEventCreateWithFlags(&done_, cudaEventDisableTiming);
    if (rc != cudaSuccess) {
        err = std::string("cudaEventCreate: ") + cudaGetErrorString(rc);
        cudaGraphExecDestroy(exec_); exec_ = nullptr;
        cudaGraphDestroy(graph_); graph_ = nullptr;
        return false;
    }

    return true;
}

bool CapturedGraph::launch(cudaStream_t stream, std::string& err) const {
    if (!exec_) {
        err = "cannot launch invalid graph (exec_ == null)";
        return false;
    }
    cudaError_t rc = cudaGraphLaunch(exec_, stream);
    if (rc != cudaSuccess) {
        err = std::string("cudaGraphLaunch: ") + cudaGetErrorString(rc);
        return false;
    }
    // Record completion
    rc = cudaEventRecord(done_, stream);
    if (rc != cudaSuccess) {
        err = std::string("cudaEventRecord: ") + cudaGetErrorString(rc);
        return false;
    }
    return true;
}

bool CapturedGraph::wait_ms(int timeout_ms) const {
    if (!done_) return false;
    // Bounded poll using REAL TIME (steady_clock), not pause counts.
    // _mm_pause is a few cycles, so counting pauses as µs would make
    // the timeout 10-100× longer than requested.  cudaEventQuery gives
    // the driver the poll call WDDM needs to flush command batches.
    auto deadline = std::chrono::steady_clock::now()
                  + std::chrono::milliseconds(timeout_ms);
    for (;;) {
        cudaError_t rc = cudaEventQuery(done_);
        if (rc == cudaSuccess) return true;
        if (rc != cudaErrorNotReady) return false;
        if (std::chrono::steady_clock::now() >= deadline) return false;
        // Brief yield between polls — avoids burning 100% of a core
        struct timespec ts = {0, 100000};  // 100 µs
        nanosleep(&ts, nullptr);
    }
}

// ─── GraphRegistry ───────────────────────────────────────────────────

bool GraphRegistry::record(LayerType layer, int n_tokens,
                           const std::function<void()>& body, std::string& err) {
    Key key{static_cast<int>(layer), n_tokens};
    if (graphs_.count(key)) return true;  // idempotent

    CapturedGraph cg;
    if (!cg.begin(stream_, err)) return false;

    body();  // execute the body under stream capture

    // Body-error detection: a silent error from any kernel in the body
    // would produce a broken graph.  Check BEFORE EndCapture so the
    // error is reported here, not as a mysterious replay failure.
    {
        cudaError_t body_err = cudaGetLastError();
        if (body_err != cudaSuccess) {
            // Must still end capture to leave the stream in a valid state
            cudaGraph_t discard = nullptr;
            cudaStreamEndCapture(stream_, &discard);
            if (discard) cudaGraphDestroy(discard);
            err = std::string("body error during capture: ") + cudaGetErrorString(body_err);
            return false;
        }
    }

    if (!cg.end(stream_, err)) return false;

    graphs_.emplace(key, std::move(cg));
    ++capture_count_;
    return true;
}

const CapturedGraph* GraphRegistry::find(LayerType layer, int n_tokens) const {
    Key key{static_cast<int>(layer), n_tokens};
    auto it = graphs_.find(key);
    return it != graphs_.end() ? &it->second : nullptr;
}

bool GraphRegistry::launch(LayerType layer, int n_tokens, int timeout_ms,
                           std::string& err) const {
    const CapturedGraph* cg = find(layer, n_tokens);
    if (!cg) {
        err = std::string("no captured graph for ") + to_string(layer)
              + " n_tokens=" + std::to_string(n_tokens);
        return false;
    }
    if (!cg->launch(stream_, err)) return false;
    if (!cg->wait_ms(timeout_ms)) {
        err = "graph launch timed out after " + std::to_string(timeout_ms) + " ms";
        return false;
    }
    return true;
}

}  // namespace strata

