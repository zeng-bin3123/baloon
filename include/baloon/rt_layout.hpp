// rt_layout.hpp - per-layer tensor layout verification over a name -> shape index (pure host code).
// Layers with l % 4 == 3 are QSA attention layers, all others GDN. Every layer carries two norm vectors; a GDN layer
// carries ssm_in / ssm_out matrices, a QSA layer carries attn_q / attn_k / attn_v / attn_output; the other kind's
// tensors must be absent. Tensor names are "blk.<layer>.<suffix>".
#pragma once
#include <cstdint>
#include <map>
#include <string>

namespace baloon {
namespace rt {

struct TensorShape { uint64_t rows = 0, cols = 0; };         // cols == 0 marks a vector of `rows` elements
using LayoutIndex = std::map<std::string, TensorShape>;

struct LayoutGeometry {
    int n_layers = 48;
    uint64_t n_embd = 2048, n_head = 16, head_dim = 256, n_head_kv = 2;
    int qsa_period = 4;                                       // layer l is QSA iff l % qsa_period == qsa_period - 1
    bool is_qsa(int l) const { return qsa_period > 0 && l >= 0 && l % qsa_period == qsa_period - 1; }
};

struct LayerView {
    int layer = 0;
    bool qsa = false;
    std::string name(const char* suffix) const;               // "blk.<layer>.<suffix>"
};

struct Want1 { const char* suffix; uint64_t len; };           // vector of exactly len elements
struct Want2 { const char* suffix; uint64_t rows, cols; };    // matrix; a 0 dimension accepts any positive value

bool layout_fail(std::string* why, const std::string& msg);   // stores msg, returns false
bool check_one(const LayoutIndex& idx, const LayerView& v, const Want1& w, std::string* why = nullptr);
bool check_one(const LayoutIndex& idx, const LayerView& v, const Want2& w, std::string* why = nullptr);
bool check_layer(const LayoutIndex& idx, const LayoutGeometry& g, int layer, std::string* why = nullptr);
bool check_all(const LayoutIndex& idx, const LayoutGeometry& g, std::string* why = nullptr);

}  // namespace rt
}  // namespace baloon
