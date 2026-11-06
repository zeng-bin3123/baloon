// expert_dispatch_contract.hpp - adapter contract: all-or-nothing hits_ready, latching failure flag,
// geometry refusal, GPU-plan list layout.
#pragma once
#include <cstdint>

namespace baloon {
namespace expert_dispatch {

struct HitCfg {
    bool cache = false, cache_base = false, parts_out = false, hit_out = false, mixed = false;
    bool scratch = false, xq = false, xq_scale = false, d_slot = false, d_dst = false;
};

constexpr bool hits_ready(const HitCfg& c) {
    return c.cache && c.cache_base && c.parts_out && c.hit_out && c.mixed && c.scratch && c.xq && c.xq_scale &&
           c.d_slot && c.d_dst;
}

constexpr bool latch(bool failed, bool fresh) { return failed || fresh; }
constexpr uint64_t step_layers(bool failed, uint64_t layers) { return failed ? layers : layers + 1; }
constexpr bool geometry_ok(uint64_t n_embd, uint64_t h_width) { return n_embd == h_width; }

struct PlanState {
    uint64_t groups = 0, entries = 0, fetches = 0;
};

// Add a group of m entries; refused (state unchanged) if it would exceed cap.
constexpr bool plan_add_group(PlanState& st, uint64_t cap, uint64_t m) {
    if (st.entries + m > cap) return false;
    st.groups += 1;
    st.entries += m;
    return true;
}

// A fetch is only issued while below both the staging capacity and the 64-slot arrays.
constexpr bool may_fetch(uint64_t fetches, uint64_t staging_cap) { return fetches < staging_cap && fetches < 64; }

}  // namespace expert_dispatch
}  // namespace baloon
