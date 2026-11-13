// tests/u7f1/test_geometry.cpp — VramMemoryPlan theorem: Geometry defaults and n_gdn.
// Theory: default_gdn_layers: ({} : Geometry).nGdnLayers = 36
#include "baloon/plan.hpp"
#include "u7f1/check.hpp"
#include <cstdio>
int main() {
    std::puts("=== test_geometry ===");
    using G = baloon::plan::Geometry;
    G g;
    // default_gdn_layers: 48 total − 12 QSA = 36 GDN
    CHK(g.n_layers == 48);
    CHK(g.n_qsa    == 12);
    CHK(g.n_gdn()  == 36);
    // geometry fields match the artifact defaults
    CHK(g.n_kv_heads   == 2);
    CHK(g.head_dim     == 256);
    CHK(g.kv_group     == 64);
    CHK(g.qsa_block    == 4);
    CHK(g.idx_key_heads == 1);
    CHK(g.idx_key_dim  == 128);
    CHK(g.ssm_state    == 128);
    CHK(g.ssm_v_heads  == 48);
    CHK(g.ssm_conv     == 4);
    CHK(g.ssm_channels == 10240);
    return u7f1::finish("TEST_GEOMETRY");
}

