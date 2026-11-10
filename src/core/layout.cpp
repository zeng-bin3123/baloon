// [unit u0063babba8] reference-layout compilation unit; adapted from src/rt/layout_check.cpp (same algorithms, same invariants).
// layout_check.cpp - implementation of rt_layout.hpp.
#include "baloon/layer_shape_check.hpp"
#include "baloon/rt_layout.hpp"

namespace baloon {
namespace rt {

std::string LayerView::name(const char* suffix) const { return "blk." + std::to_string(layer) + "." + suffix; }

bool fail(std::string* why, const std::string& msg) { if (why) *why = msg; return false; }

bool check_one(const LayoutIndex& idx, const LayerView& v, const Want1& w, std::string* why) {
    const std::string n = v.name(w.suffix);
    auto it = idx.find(n);
    if (it == idx.end()) return fail(why, "missing " + n);
    if (it->second.cols != 0) return fail(why, n + " must be a vector");
    if (it->second.rows != w.len) return fail(why, n + " has wrong length");
    return true;
}

bool check_one(const LayoutIndex& idx, const LayerView& v, const Want2& w, std::string* why) {
    const std::string n = v.name(w.suffix);
    auto it = idx.find(n);
    if (it == idx.end()) return fail(why, "missing " + n);
    const TensorShape& s = it->second;
    if (s.rows == 0 || s.cols == 0) return fail(why, n + " must be a matrix");
    if (w.rows && s.rows != w.rows) return fail(why, n + " has wrong row count");
    if (w.cols && s.cols != w.cols) return fail(why, n + " has wrong column count");
    return true;
}

bool check_layer(const LayoutIndex& idx, const LayoutGeometry& g, int layer, std::string* why) {
    if (layer < 0 || layer >= g.n_layers) return fail(why, "layer out of range");
    if (g.n_embd == 0 || g.head_dim == 0 || g.n_head == 0 || g.n_head_kv == 0) return fail(why, "bad geometry");
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
        for (const char* s : foreign_gdn) if (idx.count(v.name(s))) return fail(why, "QSA layer carries " + v.name(s));
    } else {
        if (!check_one(idx, v, Want2{"ssm_in", 0, g.n_embd}, why)) return false;
        if (!check_one(idx, v, Want2{"ssm_out", g.n_embd, 0}, why)) return false;
        for (const char* s : foreign_qsa) if (idx.count(v.name(s))) return fail(why, "GDN layer carries " + v.name(s));
    }
    return true;
}

bool check_all(const LayoutIndex& idx, const LayoutGeometry& g, std::string* why) {
    if (g.n_layers <= 0) return fail(why, "no layers");
    for (int l = 0; l < g.n_layers; ++l) if (!check_layer(idx, g, l, why)) return false;
    return true;
}

}  // namespace rt
// ---- LayerShapeCheck: the dense weight table pre-flight (16 two-D, 6/7 one-D with engine form, 12/36 split) ---------
namespace shape {

bool shape_fail(std::string* why, const std::string& m) { if (why) *why = m; return false; }

bool check_one(const TensorIndex& idx, const Geometry& g, int layer, const Want2& w, std::string* why) {
    if (!applies(w.qsa_only, w.gdn_only, g.is_qsa(layer))) return true;
    const std::string n = tensor_name(layer, w.suffix);
    auto it = idx.find(n);
    if (it == idx.end()) return shape_fail(why, "missing " + n);
    if (it->second.ne0 != g.v(w.ne0) || it->second.ne1 != g.v(w.ne1)) return shape_fail(why, n + " has wrong shape");
    return true;
}

bool check_one(const TensorIndex& idx, const Geometry& g, int layer, const Want1& w, std::string* why) {
    if (!applies(w.qsa_only, w.gdn_only, g.is_qsa(layer))) return true;
    const std::string n = tensor_name(layer, w.suffix);
    auto it = idx.find(n);
    if (it == idx.end()) return shape_fail(why, "missing " + n);
    const TensorDesc& d = it->second;
    if (d.ne1 != 0) return shape_fail(why, n + " must be one-D");
    if (d.ne0 != g.v(w.ne0)) return shape_fail(why, n + " has wrong length");
    if (!form_ok(d.form, w.form)) return shape_fail(why, n + " has wrong engine form (byte count differs)");
    return true;
}

bool check_layer(const TensorIndex& idx, const Geometry& g, int layer, std::string* why) {
    if (!layer_in_range(layer, g.n_layers)) return shape_fail(why, "layer out of range");
    for (const auto& w : kWant2) if (!check_one(idx, g, layer, w, why)) return false;
    for (const auto& w : kWant1) if (!check_one(idx, g, layer, w, why)) return false;
    return true;
}

bool check_all(const TensorIndex& idx, const Geometry& g, std::string* why) {
    if (g.n_layers <= 0) return shape_fail(why, "no layers");
    int nq = 0, ng = 0;
    for (int l = 0; l < g.n_layers; ++l) {
        if (!check_layer(idx, g, l, why)) return false;
        (g.is_qsa(l) ? nq : ng)++;
    }
    if (nq + ng != g.n_layers) return shape_fail(why, "layer kinds do not cover the layers");
    if (!split_ok(nq, ng, g.expect_qsa, g.expect_gdn)) return shape_fail(why, "layer kind counts differ from geometry");
    return true;
}

}  // namespace shape

}  // namespace baloon
