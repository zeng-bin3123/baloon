// layout_check.cpp - implementation of rt_layout.hpp.
#include "baloon/rt_layout.hpp"

namespace baloon {
namespace rt {

std::string LayerView::name(const char* suffix) const { return "blk." + std::to_string(layer) + "." + suffix; }

bool layout_fail(std::string* why, const std::string& msg) { if (why) *why = msg; return false; }

bool check_one(const LayoutIndex& idx, const LayerView& v, const Want1& w, std::string* why) {
    const std::string n = v.name(w.suffix);
    auto it = idx.find(n);
    if (it == idx.end()) return layout_fail(why, "missing " + n);
    if (it->second.cols != 0) return layout_fail(why, n + " must be a vector");
    if (it->second.rows != w.len) return layout_fail(why, n + " has wrong length");
    return true;
}

bool check_one(const LayoutIndex& idx, const LayerView& v, const Want2& w, std::string* why) {
    const std::string n = v.name(w.suffix);
    auto it = idx.find(n);
    if (it == idx.end()) return layout_fail(why, "missing " + n);
    const TensorShape& s = it->second;
    if (s.rows == 0 || s.cols == 0) return layout_fail(why, n + " must be a matrix");
    if (w.rows && s.rows != w.rows) return layout_fail(why, n + " has wrong row count");
    if (w.cols && s.cols != w.cols) return layout_fail(why, n + " has wrong column count");
    return true;
}

bool check_layer(const LayoutIndex& idx, const LayoutGeometry& g, int layer, std::string* why) {
    if (layer < 0 || layer >= g.n_layers) return layout_fail(why, "layer out of range");
    if (g.n_embd == 0 || g.head_dim == 0 || g.n_head == 0 || g.n_head_kv == 0) return layout_fail(why, "bad geometry");
    LayerView v; v.layer = layer; v.qsa = g.is_qsa(layer);
    if (!check_one(idx, v, Want1{"attn_norm", g.n_embd}, why)) return false;
    if (!check_one(idx, v, Want1{"ffn_norm", g.n_embd}, why)) return false;
    const char* foreign_gdn[] = {"ssm_in", "ssm_out"};
    const char* foreign_qsa[] = {"attn_q", "attn_k", "attn_v", "attn_output"};
    if (v.qsa) {
        if (!check_one(idx, v, Want2{"attn_q", g.n_head * g.head_dim, g.n_embd}, why)) return false;
        if (!check_one(idx, v, Want2{"attn_k", g.n_head_kv * g.head_dim, g.n_embd}, why)) return false;
        if (!check_one(idx, v, Want2{"attn_v", g.n_head_kv * g.head_dim, g.n_embd}, why)) return false;
        if (!check_one(idx, v, Want2{"attn_output", g.n_embd, g.n_head * g.head_dim}, why)) return false;
        for (const char* s : foreign_gdn) if (idx.count(v.name(s))) return layout_fail(why, "QSA layer carries " + v.name(s));
    } else {
        if (!check_one(idx, v, Want2{"ssm_in", 0, g.n_embd}, why)) return false;
        if (!check_one(idx, v, Want2{"ssm_out", g.n_embd, 0}, why)) return false;
        for (const char* s : foreign_qsa) if (idx.count(v.name(s))) return layout_fail(why, "GDN layer carries " + v.name(s));
    }
    return true;
}

bool check_all(const LayoutIndex& idx, const LayoutGeometry& g, std::string* why) {
    if (g.n_layers <= 0) return layout_fail(why, "no layers");
    for (int l = 0; l < g.n_layers; ++l) if (!check_layer(idx, g, l, why)) return false;
    return true;
}

}  // namespace rt
}  // namespace baloon
