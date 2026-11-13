// tests/u7f1/test_main.cpp — u7f1ccea730 test suite runner.
// Covers VramMemoryPlan, DenseWeightLoader (constants), NativeDenseOverride (string matching),
// and LayerShapeCheck theorems.
// NativeDense eligible() is reproduced locally to avoid the CUDA-dependent u11_native_dense.hpp.
#include "baloon/plan.hpp"
#include "baloon/layer_shape_check.hpp"
#include "baloon/core_weights.hpp"
#include "u7f1/check.hpp"
#include <cstdio>
#include <cstring>
#include <variant>
#include <vector>
#include <algorithm>
#include <string>

// Inline replica of eligible() from NativeDenseOverride (no CUDA headers needed).
static const char* kSuffixes[10] = {
    ".attn_qkv.weight", ".attn_gate.weight", ".ssm_out.weight",
    ".attn_q.weight", ".attn_k.weight", ".attn_v.weight",
    ".attn_output.weight", ".ffn_gate_shexp.weight",
    ".ffn_up_shexp.weight", ".ffn_down_shexp.weight"
};
static bool eligible_local(const std::string& name, bool ple_key) {
    if (name.compare(0, 4, "blk.") != 0) return false;
    if (ple_key && name == "blk.1.ple_key.weight") return true;
    for (const char* s : kSuffixes) {
        const size_t n = std::strlen(s);
        if (name.size() >= n && name.compare(name.size() - n, n, s) == 0) return true;
    }
    return false;
}

// ---- VramMemoryPlan theorems -----------------------------------------------

static int run_vram_memory_plan() {
    std::puts("\n=== VramMemoryPlan ===");
    using namespace baloon::plan;
    int fails_before = u7f1::_fails;

    Geometry g;
    // default_gdn_layers
    CHK(g.n_layers == 48 && g.n_qsa == 12 && g.n_gdn() == 36);
    // recurrence_per_layer, conv_per_layer, stateBytes_default
    CHK(baloon::plan::kStateBytesDefault == 117669888u);
    CHK(state_bytes(g) == 117669888u);
    // state_is_85_slots
    CHK(state_bytes(g) / baloon::planner::kExpertBlobBytes == 85u);
    // indexer_default
    CHK(g.idx_key_heads * g.idx_key_dim / g.qsa_block == 32u);
    // kvBytesPerToken_default
    CHK(kv_bytes_per_token(g) == 13056u);
    // vram_pool_bytes truncation
    CHK(vram_pool_bytes(100, 200) == 0u);
    CHK(vram_pool_bytes(32ull<<30, 512ull<<20) == 33822867456ull);

    // plan_closes / plan_has_slot / plan_slots_fit
    Costs c; c.state = state_bytes(g);
    uint64_t pool = vram_pool_bytes(32ull<<30, 512ull<<20);
    Result r = make_plan(32768, g, c, pool);
    CHK(std::holds_alternative<Plan>(r));
    if (std::holds_alternative<Plan>(r)) {
        const Plan& p = std::get<Plan>(r);
        CHK(p.kv_bytes + c.state + p.cache_bytes == pool);       // plan_closes
        CHK(p.cache_slots * c.expert_blob <= p.cache_bytes);     // slots_fit
        CHK(p.cache_slots >= 1u);                                 // has_slot
        CHK(p.vram_used <= p.vram_budget);
        CHK(p.kv_bytes == 13056u * 32768u);
    }
    // plan_refuses
    CHK(std::holds_alternative<DoesNotClose>(make_plan(999999999, g, c, pool)));

    // slots_antitone
    Result r64 = make_plan(65536, g, c, pool);
    if (std::holds_alternative<Plan>(r) && std::holds_alternative<Plan>(r64)) {
        CHK(std::get<Plan>(r64).cache_slots <= std::get<Plan>(r).cache_slots);
    }

    return u7f1::_fails - fails_before;
}

// ---- LayerShapeCheck theorems -----------------------------------------------

static int run_layer_shape_check() {
    std::puts("\n=== LayerShapeCheck ===");
    using namespace baloon::shape;
    Geometry g;
    int fails_before = u7f1::_fails;

    // split_48
    int nq = 0, ng = 0;
    for (int l = 0; l < g.n_layers; ++l) { if (g.is_qsa(l)) ++nq; else ++ng; }
    CHK(nq == 12 && ng == 36);

    // isQsa_last / isQsa_other
    for (int m = 0; m < 12; ++m) {
        CHK(g.is_qsa(4*m+3));
        CHK(!g.is_qsa(4*m)); CHK(!g.is_qsa(4*m+1)); CHK(!g.is_qsa(4*m+2));
    }

    // checked2_counts: 16 two-D per family
    int n2q = 0, n2g = 0;
    for (const auto& w : kWant2) {
        if (applies(w.qsa_only, w.gdn_only, true))  ++n2q;
        if (applies(w.qsa_only, w.gdn_only, false)) ++n2g;
    }
    CHK(n2q == 16 && n2g == 16);

    // want1_single_bf16
    int nbf = 0;
    for (const auto& w : kWant1) if (w.form == Form::Bf16InF32) ++nbf;
    CHK(nbf == 1);

    // bf16_as_f32_overrun
    CHK(form_bytes(Form::F32, 2560) == form_bytes(Form::Bf16InF32, 2560) + 2*2560u);

    return u7f1::_fails - fails_before;
}

// ---- NativeDenseOverride eligible -------------------------------------------

static int run_native_dense_override() {
    std::puts("\n=== NativeDenseOverride ===");
    int fails_before = u7f1::_fails;

    CHK( baloon::u11::eligible("blk.0.attn_qkv.weight", false));
    CHK(!baloon::u11::eligible("token_embd.weight", false));
    CHK(!baloon::u11::eligible("blk.1.ple_key.weight", false));
    CHK( baloon::u11::eligible("blk.1.ple_key.weight", true));
    CHK( baloon::u11::eligible("blk.5.ssm_out.weight",  false));

    return u7f1::_fails - fails_before;
}

int main() {
    std::puts("=== u7f1ccea730 TEST SUITE ===");
    run_vram_memory_plan();
    run_layer_shape_check();
    run_native_dense_override();
    return u7f1::finish("U7F1CCEA730 SUITE");
}

