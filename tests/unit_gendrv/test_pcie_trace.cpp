// GenerateDriver theory § PCIe share and routing trace
// Contracts: pcieNum, pcieNum_le, traceLayer.
// pcieNum x = min 256 x  (Nat clamp)
// pcieNum_le: pcieNum x <= 256
// traceLayer afterDispatch = afterDispatch - 1  (Int)
#include "check.hpp"
#include "baloon/generate_driver_accounting.hpp"
using namespace baloon::generate;

void test_pcie_trace() {
    // pcieNum_le: clamp_pcie(x) <= 256 for all int64_t x
    for (int64_t x = -10; x <= 300; ++x) {
        const int64_t c = clamp_pcie(x);
        CHECK_MSG(c >= 0 && c <= 256, "pcieNum_le: x=%lld c=%lld", (long long)x, (long long)c);
        // pcieNum definition: min(256, x) for x in [0, 256]
        if (x >= 0 && x <= 256) CHECK_MSG(c == x, "pcieNum def: x=%lld c=%lld", (long long)x, (long long)c);
        if (x < 0)   CHECK_MSG(c == 0,   "pcieNum lo: x=%lld", (long long)x);
        if (x > 256) CHECK_MSG(c == 256, "pcieNum hi: x=%lld", (long long)x);
    }
    // traceLayer: afterDispatch - 1
    // For layers 0..47 the counter l+1 after dispatch maps back to l
    for (uint64_t l = 0; l < 48; ++l) {
        const int64_t t = trace_layer(l + 1);
        CHECK_MSG(t == (int64_t)l, "traceLayer: l=%llu t=%lld", (unsigned long long)l, (long long)t);
        CHECK_MSG(t >= 0 && t < 48, "traceLayer range: l=%llu t=%lld", (unsigned long long)l, (long long)t);
    }
    // traceLayer(0) = -1 (the counter was 0, so afterDispatch-1 = -1)
    CHECK(trace_layer(0) == -1);
}
