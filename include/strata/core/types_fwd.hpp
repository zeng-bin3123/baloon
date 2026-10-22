#pragma once
// strata/core/types_fwd.hpp — Forward declarations for types defined in later units

#include <cstdint>

namespace strata {

// Model geometry (dimensions, layer counts, etc.) — defined in a later unit.
struct Geometry {
    int64_t n_embd = 0;
    int64_t n_head = 0;
    int64_t n_kv_head = 0;
    int64_t n_layers = 0;
    int64_t n_vocab = 0;
    int64_t n_experts = 0;
    int64_t n_experts_active = 0;
    int64_t expert_hidden = 0;
    int64_t hc = 0;            // head count (for residual sizing)
    float rms_eps = 1e-5f;
};

// Session state (KV cache, position tracking) — defined in a later unit.
class SessionState {
public:
    SessionState() = default;
    virtual ~SessionState() = default;
};

// Head projection (final lm_head) — defined in a later unit.
struct HeadProjection {
    const void* weight = nullptr;
    int64_t n_vocab = 0;
    int64_t n_embd = 0;
};

}  // namespace strata

