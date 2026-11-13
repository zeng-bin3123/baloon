// tests/u7f1/test_want_flags.cpp — LayerShapeCheck theorems: gdnOnly_applies, qsaOnly_applies,
// checked2_counts, want2_no_dead, want1_single_bf16, bf16_as_f32_overrun, formBytes_injective.
// Theory:
//   gdnOnly_applies : applies false true qsa = !qsa
//   qsaOnly_applies : applies true false qsa = qsa
//   checked2_counts : checked2 false = 16 ∧ checked2 true = 16
//   want2_no_dead   : no (true, true) entry
//   want1_single_bf16: exactly one BF16 requirement (ffn_gate_inp_shexp)
//   bf16_as_f32_overrun: formBytes F32 e = formBytes Bf16InF32 e + 2*e
//   formBytes_injective: equal bytes → equal form
#include "baloon/layer_shape_check.hpp"
#include "u7f1/check.hpp"
#include <cstdio>
int main() {
    std::puts("=== test_want_flags ===");
    using namespace baloon::shape;

    // ---- gdnOnly_applies / qsaOnly_applies ----
    CHK( applies(false, true,  false));  // gdn_only on GDN layer: applies
    CHK(!applies(false, true,  true));   // gdn_only on QSA layer: skipped
    CHK( applies(true,  false, true));   // qsa_only on QSA layer: applies
    CHK(!applies(true,  false, false));  // qsa_only on GDN layer: skipped
    CHK( applies(false, false, true));   // neither: always applies (QSA)
    CHK( applies(false, false, false));  // neither: always applies (GDN)

    // ---- checked2_counts: 16 two-D requirements apply on each family ----
    int n_qsa = 0, n_gdn = 0;
    for (const auto& w : kWant2) {
        if (applies(w.qsa_only, w.gdn_only, true))  ++n_qsa;
        if (applies(w.qsa_only, w.gdn_only, false)) ++n_gdn;
    }
    CHK(n_qsa == 16);
    CHK(n_gdn == 16);

    // ---- want2_no_dead: no entry is both qsa_only and gdn_only ----
    for (const auto& w : kWant2) {
        CHK(!(w.qsa_only && w.gdn_only));
    }

    // ---- want1_single_bf16: exactly one BF16 form in the 1-D requirements ----
    int nbf16 = 0;
    for (const auto& w : kWant1) {
        if (w.form == Form::Bf16InF32) ++nbf16;
    }
    CHK(nbf16 == 1);  // ffn_gate_inp_shexp is the sole BF16 vector

    // ---- formBytes: F32 is 4 B/elem, Bf16InF32 is 2 B/elem ----
    for (uint64_t e : {1u, 128u, 2560u}) {
        CHK(form_bytes(Form::F32,       e) == 4u * e);
        CHK(form_bytes(Form::Bf16InF32, e) == 2u * e);
    }

    // ---- bf16_as_f32_overrun: reading BF16 as F32 walks 2*e bytes too far ----
    for (uint64_t e : {1u, 2560u}) {
        CHK(form_bytes(Form::F32, e) == form_bytes(Form::Bf16InF32, e) + 2u * e);
        CHK(form_bytes(Form::Bf16InF32, e) < form_bytes(Form::F32, e));
    }

    // ---- formBytes_injective: different forms always give different byte counts ----
    for (uint64_t e = 1; e <= 2560; e *= 2) {
        CHK(form_bytes(Form::F32, e) != form_bytes(Form::Bf16InF32, e));
    }

    return u7f1::finish("TEST_WANT_FLAGS");
}

