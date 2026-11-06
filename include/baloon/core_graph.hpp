// core_graph.hpp - layer types and captured graphs (host model). Layers l % 4 == 3 are QSA, the rest GDN. A CapturedGraph
// is an ordered node list replayed in capture order; GraphRegistry owns named graphs.
#pragma once
#include <cstdint>
#include <functional>
#include <map>
#include <string>
#include <vector>

#include "baloon/core_layer.hpp"

namespace baloon {
namespace core {

enum class LayerType { Gdn, Qsa };
inline LayerType layer_type_of(int l) { return is_qsa_layer(l) ? LayerType::Qsa : LayerType::Gdn; }
inline const char* to_string(LayerType t) { return t == LayerType::Qsa ? "QSA" : "GDN"; }

struct CapturedGraph {
    std::vector<std::function<void()>> nodes;
    bool captured = false;
    uint64_t replays = 0;
    void begin() { nodes.clear(); captured = false; }
    void add(std::function<void()> f) { nodes.push_back(std::move(f)); }
    void end() { captured = true; }
    bool replay() { if (!captured) return false; for (auto& n : nodes) n(); ++replays; return true; }
    void free() { nodes.clear(); captured = false; }
};

class GraphRegistry {
public:
    CapturedGraph& get(const std::string& name) { return g_[name]; }
    const CapturedGraph* find(const std::string& name) const { auto it = g_.find(name); return it == g_.end() ? nullptr : &it->second; }
    bool replay(const std::string& name) { auto it = g_.find(name); return it != g_.end() && it->second.replay(); }
    bool erase(const std::string& name) { return g_.erase(name) > 0; }
    void clear() { g_.clear(); }
    size_t size() const { return g_.size(); }
private:
    std::map<std::string, CapturedGraph> g_;
};

}  // namespace core
}  // namespace baloon
