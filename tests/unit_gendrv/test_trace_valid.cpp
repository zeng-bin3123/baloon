// GenerateDriver theory § routing trace (traceLayer_valid)
// traceLayer_valid: l < 48 -> traceLayer(l+1) = l  &&  0 <= traceLayer(l+1)  &&  traceLayer(l+1) < 48
// The adapter advances the counter before we read it, so afterDispatch = l+1 -> index = l.
#include "check.hpp"
#include "baloon/generate_driver_accounting.hpp"
using namespace baloon::generate;

void test_trace_valid() {
    // traceLayer_valid exhaustive for layers 0..47
    for (uint64_t l = 0; l < 48; ++l) {
        const int64_t t = trace_layer(l + 1);
        CHECK_MSG(t == (int64_t)l,  "traceLayer_valid eq:  l=%llu t=%lld", (unsigned long long)l, (long long)t);
        CHECK_MSG(t >= 0,           "traceLayer_valid ge0: l=%llu t=%lld", (unsigned long long)l, (long long)t);
        CHECK_MSG(t < 48,           "traceLayer_valid lt48:l=%llu t=%lld", (unsigned long long)l, (long long)t);
    }
    // the counter value 0 (before any layer) produces -1, which is outside 0..47
    CHECK(trace_layer(0) == -1);
}
