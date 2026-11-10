// doorbell_support.cpp - test-binary support only. src/core/session.cpp (session_loop) reads the doorbell through
// core::doorbell_value, which lives in src/core/layer.cpp together with the whole layer runtime. test_ua96 does not
// exercise session_loop, so it links this one accessor (same acquire-load semantics) instead of the layer runtime.
#include <atomic>
#include "baloon/core_layer.hpp"
namespace baloon { namespace core {
uint64_t doorbell_value(const Doorbell* d) { return (d && d->word) ? d->word->load(std::memory_order_acquire) : 0; }
} }
