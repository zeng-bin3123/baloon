// layer_shape_check.hpp - pre-flight shape check of the dense weight table (LayerShapeCheck).
//
// For every layer the required tensor set depends on the layer family (QSA at l % period == period - 1, GDN otherwise):
//   * 22 two-D requirements are declared, 16 of them apply to either family (10 common + 6 family specific);
//   * 10 one-D requirements are declared, 6 apply to a GDN layer (3 + 3) and 7 to a QSA layer (3 + 4);
//   * a two-D tensor must have exactly (ne0, ne1), a one-D tensor exactly ne0 elements AND the engine form (f32 or
//     bf16-held-in-f32). The form fixes the byte count: reading a bf16 tensor as f32 overreads by 2x its length.
// check_all additionally counts the layer kinds: QSA + GDN == n_layers and both counts equal the geometry's.
#pragma once
#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace baloon {
namespace shape {

// ---- engine forms ------------------------------------------------------------------------------------------------
enum class Form : uint8_t { F32 = 0, Bf16InF32 = 1 };
constexpr uint64_t bytes_per(Form f) { return f == Form::F32 ? 4 : 2; }
constexpr uint64_t form_bytes(Form f, uint64_t elements) { return elements * bytes_per(f); }
constexpr bool form_ok(Form have, Form want) { return have == want; }

// ---- which requirement applies to which family -------------------------------------------------------------------
constexpr bool applies(bool qsa_only, bool gdn_only, bool is_qsa) {
    return !(qsa_only && !is_qsa) && !(gdn_only && is_qsa);
}

// Symbolic dimensions; their values come from the geometry.
enum class Dim : uint8_t {
    Embd, HcDim, HcMix, NHead, QDim, KvDim, HeadDim, IdxDim, Expert, ExpertFf, ShexpFf,
    SsmIn, SsmInner, SsmConv, SsmHeads, SsmHeadDim, One, Count
};

struct Geometry {
    int n_layers = 48;
    int qsa_period = 4;            // layer l is QSA iff l % qsa_period == qsa_period - 1
    int expect_qsa = 12, expect_gdn = 36;
    uint64_t dim[(int)Dim::Count] = {};
    Geometry() {
        const uint64_t embd = 2560, hc = 4, nh = 16, hd = 256, nkv = 2;
        dim[(int)Dim::Embd] = embd;
        dim[(int)Dim::HcDim] = hc * embd;                   // 4 streams of 2560 = 10240
        dim[(int)Dim::HcMix] = 2 * hc + hc * hc;
        dim[(int)Dim::NHead] = nh;
        dim[(int)Dim::QDim] = nh * hd;
        dim[(int)Dim::KvDim] = nkv * hd;
        dim[(int)Dim::HeadDim] = hd;
        dim[(int)Dim::IdxDim] = 4 * 64;
        dim[(int)Dim::Expert] = 512;
        dim[(int)Dim::ExpertFf] = 512 * 512;
        dim[(int)Dim::ShexpFf] = 512;
        dim[(int)Dim::SsmIn] = 8192;
        dim[(int)Dim::SsmInner] = 4096;
        dim[(int)Dim::SsmConv] = 4;
        dim[(int)Dim::SsmHeads] = 32;
        dim[(int)Dim::SsmHeadDim] = 128;
        dim[(int)Dim::One] = 1;
    }
    bool is_qsa(int l) const { return qsa_period > 0 && l >= 0 && l % qsa_period == qsa_period - 1; }
    uint64_t v(Dim d) const { return dim[(int)d]; }
};

struct Want2 { const char* suffix; bool qsa_only, gdn_only; Dim ne0, ne1; };
struct Want1 { const char* suffix; bool qsa_only, gdn_only; Dim ne0; Form form; };

// 22 two-D requirements: 10 common, 6 GDN-only, 6 QSA-only.
constexpr Want2 kWant2[22] = {
    {"hc_attn_fn", false, false, Dim::HcDim, Dim::HcMix},
    {"hc_ffn_fn", false, false, Dim::HcDim, Dim::HcMix},
    {"ffn_gate_inp", false, false, Dim::Embd, Dim::Expert},
    {"ffn_gate_exps", false, false, Dim::Embd, Dim::ExpertFf},
    {"ffn_up_exps", false, false, Dim::Embd, Dim::ExpertFf},
    {"ffn_down_exps", false, false, Dim::ShexpFf, Dim::ExpertFf},
    {"ffn_gate_shexp", false, false, Dim::Embd, Dim::ShexpFf},
    {"ffn_up_shexp", false, false, Dim::Embd, Dim::ShexpFf},
    {"ffn_down_shexp", false, false, Dim::ShexpFf, Dim::Embd},
    {"hc_post_fn", false, false, Dim::HcDim, Dim::HcMix},
    {"ssm_in", false, true, Dim::Embd, Dim::SsmIn},
    {"ssm_out", false, true, Dim::SsmInner, Dim::Embd},
    {"ssm_conv1d", false, true, Dim::SsmConv, Dim::SsmIn},
    {"ssm_alpha", false, true, Dim::Embd, Dim::SsmHeads},
    {"ssm_beta", false, true, Dim::Embd, Dim::SsmHeads},
    {"ssm_gate", false, true, Dim::Embd, Dim::SsmInner},
    {"attn_q", true, false, Dim::Embd, Dim::QDim},
    {"attn_k", true, false, Dim::Embd, Dim::KvDim},
    {"attn_v", true, false, Dim::Embd, Dim::KvDim},
    {"attn_output", true, false, Dim::QDim, Dim::Embd},
    {"idx_q", true, false, Dim::Embd, Dim::IdxDim},
    {"idx_k", true, false, Dim::Embd, Dim::HeadDim},
};

// 10 one-D requirements: 3 common, 3 GDN-only, 4 QSA-only. The shared-expert gate is bf16 held in f32 (2560 elements).
constexpr Want1 kWant1[10] = {
    {"attn_norm", false, false, Dim::Embd, Form::F32},
    {"ffn_norm", false, false, Dim::Embd, Form::F32},
    {"ffn_gate_inp_shexp", false, false, Dim::Embd, Form::Bf16InF32},
    {"ssm_a", false, true, Dim::SsmHeads, Form::F32},
    {"ssm_dt_bias", false, true, Dim::SsmHeads, Form::F32},
    {"ssm_norm", false, true, Dim::SsmHeadDim, Form::F32},
    {"attn_q_norm", true, false, Dim::HeadDim, Form::F32},
    {"attn_k_norm", true, false, Dim::HeadDim, Form::F32},
    {"attn_sinks", true, false, Dim::NHead, Form::F32},
    {"idx_norm", true, false, Dim::HeadDim, Form::F32},
};

constexpr int count_two_d(bool is_qsa) {
    int n = 0; for (const auto& w : kWant2) n += applies(w.qsa_only, w.gdn_only, is_qsa) ? 1 : 0; return n;
}
constexpr int count_one_d(bool is_qsa) {
    int n = 0; for (const auto& w : kWant1) n += applies(w.qsa_only, w.gdn_only, is_qsa) ? 1 : 0; return n;
}
static_assert(count_two_d(true) == 16 && count_two_d(false) == 16, "16 two-D tensors per layer, either family");
static_assert(count_one_d(true) == 7 && count_one_d(false) == 6, "7 one-D tensors on QSA, 6 on GDN");
static_assert(form_bytes(Form::Bf16InF32, 2560) == 5120 && form_bytes(Form::F32, 2560) == 10240, "shexp gate bytes");

// ---- the weight table index --------------------------------------------------------------------------------------
struct TensorDesc { uint64_t ne0 = 0, ne1 = 0; Form form = Form::F32; };   // ne1 == 0: one-D
using TensorIndex = std::map<std::string, TensorDesc>;

inline std::string tensor_name(int layer, const char* suffix) { return "blk." + std::to_string(layer) + "." + suffix; }

struct Requirement { std::string name; uint64_t ne0, ne1; Form form; bool one_d; };

// The exact list a layer is checked against (16 two-D then 6/7 one-D).
inline std::vector<Requirement> required_for_layer(const Geometry& g, int layer) {
    std::vector<Requirement> r;
    const bool q = g.is_qsa(layer);
    for (const auto& w : kWant2)
        if (applies(w.qsa_only, w.gdn_only, q)) r.push_back({tensor_name(layer, w.suffix), g.v(w.ne0), g.v(w.ne1), Form::F32, false});
    for (const auto& w : kWant1)
        if (applies(w.qsa_only, w.gdn_only, q)) r.push_back({tensor_name(layer, w.suffix), g.v(w.ne0), 0, w.form, true});
    return r;
}

// ---- the checks themselves are production code: src/rt/layout_check.cpp defines them ------------------------------
// Records `m` in *why (when given) and returns false.
bool shape_fail(std::string* why, const std::string& m);

// The layer-kind count check of check_all: counted kinds equal the geometry's (and thereby cover every layer).
constexpr bool split_ok(int n_qsa, int n_gdn, int g_qsa, int g_gdn) { return n_qsa == g_qsa && n_gdn == g_gdn; }
constexpr bool layer_in_range(int layer, int n_layers) { return layer >= 0 && layer < n_layers; }

// One requirement of one layer: presence, exact (ne0, ne1) for two-D, exact length + engine form for one-D.
bool check_one(const TensorIndex& idx, const Geometry& g, int layer, const Want2& w, std::string* why = nullptr);
bool check_one(const TensorIndex& idx, const Geometry& g, int layer, const Want1& w, std::string* why = nullptr);
// Every requirement that applies to the layer's family (16 two-D + 6/7 one-D); a layer outside 0..n_layers-1 is refused.
bool check_layer(const TensorIndex& idx, const Geometry& g, int layer, std::string* why = nullptr);
// Every layer, then the QSA/GDN split: kinds cover all layers and equal the geometry's (12 QSA, 36 GDN).
bool check_all(const TensorIndex& idx, const Geometry& g, std::string* why = nullptr);

}  // namespace shape
}  // namespace baloon

