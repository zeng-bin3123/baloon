// tests/u7f1/test_state_bytes.cpp — VramMemoryPlan theorems: stateBytes_default, state_is_85_slots.
// Theory:
//   recurrence_per_layer : 128 * 48 * 128 * 4 = 3145728
//   conv_per_layer       : 3 * 10240 * 4 = 122880
//   stateBytes_default   : stateBytes {} = 117669888
//   state_is_85_slots    : stateBytes {} / 1382400 = 85
#include "baloon/plan.hpp"
#include "u7f1/check.hpp"
#include <cstdio>
int main() {
    std::puts("=== test_state_bytes ===");
    using namespace baloon::plan;
    // recurrence_per_layer: ssm_state * ssm_v_heads * ssm_state * 4 bytes
    constexpr uint64_t rec = 128ull * 48 * 128 * 4;   // 3 145 728
    CHK(rec == 3145728u);

    // conv_per_layer: (ssm_conv - 1) * ssm_channels * 4 bytes
    constexpr uint64_t conv = 3ull * 10240 * 4;        // 122 880
    CHK(conv == 122880u);

    // stateBytes_default: 36 GDN layers * (rec + conv)
    constexpr uint64_t want = 36ull * (rec + conv);    // 117 669 888
    CHK(want == 117669888u);
    CHK(kStateBytesDefault == 117669888u);

    Geometry g;
    const uint64_t sb = state_bytes(g);
    CHK(sb == 117669888u);

    // state_is_85_slots: 117669888 / 1382400 = 85 (whole expert slots)
    constexpr uint64_t blob = 1382400u;   // default expert blob size
    CHK(sb / blob == 85u);
    CHK(kStateBytesDefault / baloon::planner::kExpertBlobBytes == 85u);

    return u7f1::finish("TEST_STATE_BYTES");
}

