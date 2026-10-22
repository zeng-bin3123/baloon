// strata/core/weights.cpp — Weight table loading and lookup
#include "strata/core/weights.hpp"
#include <cstring>

namespace strata {

bool WeightTable::pool_bytes(const std::string& dir, uint64_t& out_bytes,
                             std::string& err,
                             const std::set<std::string>* skip) {
    // TODO: read index file from dir, sum tensor sizes (minus skip set).
    // For now, return 0 — later units will implement the index parser.
    out_bytes = 0;
    err = "pool_bytes: index parser not yet implemented";
    return false;
}

bool WeightTable::load(const std::string& dir, void* arena_base,
                       uint64_t arena_bytes, std::string& err,
                       const std::set<std::string>* skip) {
    // Pre: arena_bytes >= pool_bytes (checked, not trusted)
    uint64_t required = 0;
    // In full implementation: parse index, validate, memcpy/convert tensors.
    // Placeholder for unit 1 — later units supply the index format.
    err = "load: index parser not yet implemented";
    return false;
}

const WeightRef* WeightTable::find(const std::string& name) const {
    auto it = table_.find(name);
    return it != table_.end() ? &it->second : nullptr;
}

}  // namespace strata

