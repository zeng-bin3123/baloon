// tests/u7f1/test_layer_pred.cpp — LayerShapeCheck theorems: isQsa_last, isQsa_other,
// countQsa_mul, countGdn_mul, count_split, split_48.
// Theory:
//   isQsa_last  : isQsa (4*m + 3) = true  for all m
//   isQsa_other : isQsa (4*m + r) = false for r < 3
//   countQsa_mul: countQsa (4*m) = m
//   countGdn_mul: countGdn (4*m) = 3*m
//   count_split : countQsa n + countGdn n = n
//   split_48    : countQsa 48 = 12 ∧ countGdn 48 = 36
#include "baloon/layer_shape_check.hpp"
#include "u7f1/check.hpp"
#include <cstdio>
int main() {
    std::puts("=== test_layer_pred ===");
    using baloon::shape::Geometry;
    Geometry g;  // default: n_layers=48, qsa_period=4

    // ---- isQsa_last: layer 4m+3 is QSA ----
    for (int m = 0; m < 12; ++m) {
        CHK(g.is_qsa(4 * m + 3));
    }

    // ---- isQsa_other: layers 4m, 4m+1, 4m+2 are GDN ----
    for (int m = 0; m < 12; ++m) {
        CHK(!g.is_qsa(4 * m + 0));
        CHK(!g.is_qsa(4 * m + 1));
        CHK(!g.is_qsa(4 * m + 2));
    }

    // ---- count_split: QSA + GDN = n_layers ----
    int nq = 0, ng = 0;
    for (int l = 0; l < g.n_layers; ++l) {
        if (g.is_qsa(l)) ++nq; else ++ng;
    }
    CHK(nq + ng == g.n_layers);

    // ---- countQsa_mul: for 4m layers, m are QSA ----
    for (int m = 0; m <= 12; ++m) {
        int q = 0;
        for (int l = 0; l < 4 * m; ++l) if (g.is_qsa(l)) ++q;
        CHK(q == m);
    }

    // ---- countGdn_mul: for 4m layers, 3m are GDN ----
    for (int m = 0; m <= 12; ++m) {
        int gd = 0;
        for (int l = 0; l < 4 * m; ++l) if (!g.is_qsa(l)) ++gd;
        CHK(gd == 3 * m);
    }

    // ---- split_48: 48 layers → 12 QSA, 36 GDN ----
    CHK(nq == 12);
    CHK(ng == 36);
    CHK(g.expect_qsa == 12);
    CHK(g.expect_gdn == 36);

    return u7f1::finish("TEST_LAYER_PRED");
}

