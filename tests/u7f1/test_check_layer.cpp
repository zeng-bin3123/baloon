// tests/u7f1/test_check_layer.cpp — LayerShapeCheck: check_layer and check_all C++ API.
// Tests the actual check_layer / check_all from src/core/layout.cpp.
// Theory equivalents: applies predicate, Want1/Want2 presence checks, split_ok.
#include "baloon/layer_shape_check.hpp"
#include "u7f1/check.hpp"
#include <cstdio>
#include <map>
#include <string>
int main() {
    std::puts("=== test_check_layer ===");
    using namespace baloon::shape;

    Geometry g;  // default: 48 layers, period=4, expect_qsa=12, expect_gdn=36

    // Build a minimal but complete TensorIndex satisfying all requirements for layers 0 and 3.
    // Layer 0 is GDN (not QSA), layer 3 is QSA.
    auto make_idx = [&](int layer) -> TensorIndex {
        TensorIndex idx;
        bool qsa = g.is_qsa(layer);
        // Add all required tensors for this layer.
        const auto& reqs = required_for_layer(g, layer);
        for (const auto& r : reqs) {
            TensorDesc d;
            if (r.one_d) {
                d.ne0 = r.ne0; d.ne1 = 0; d.form = r.form;
            } else {
                d.ne0 = r.ne0; d.ne1 = r.ne1; d.form = Form::F32;
            }
            idx[r.name] = d;
        }
        return idx;
    };

    // ---- check_layer: GDN layer 0 with correct tensors ----
    {
        TensorIndex idx = make_idx(0);
        std::string why;
        CHK(check_layer(idx, g, 0, &why));
    }

    // ---- check_layer: QSA layer 3 with correct tensors ----
    {
        TensorIndex idx = make_idx(3);
        std::string why;
        CHK(check_layer(idx, g, 3, &why));
    }

    // ---- check_layer: out-of-range layer rejected ----
    {
        TensorIndex idx;
        std::string why;
        CHK(!check_layer(idx, g, -1, &why));
        CHK(!check_layer(idx, g, 48, &why));
        CHK(!why.empty());
    }

    // ---- check_layer: missing tensor detected ----
    {
        TensorIndex idx = make_idx(0);
        // Remove one required tensor.
        idx.erase("blk.0.attn_norm");
        std::string why;
        CHK(!check_layer(idx, g, 0, &why));
        CHK(why.find("missing") != std::string::npos || !why.empty());
    }

    // ---- check_layer: wrong form detected (bf16 as f32) ----
    {
        TensorIndex idx = make_idx(0);
        // The ffn_gate_inp_shexp tensor must be BF16; force it to F32.
        auto it = idx.find("blk.0.ffn_gate_inp_shexp");
        if (it != idx.end()) {
            it->second.form = Form::F32;  // wrong: should be Bf16InF32
            std::string why;
            CHK(!check_layer(idx, g, 0, &why));
        }
    }

    // ---- check_all: build full 48-layer index and verify split ----
    {
        TensorIndex idx;
        for (int l = 0; l < 48; ++l) {
            TensorIndex li = make_idx(l);
            for (auto& kv : li) idx[kv.first] = kv.second;
        }
        std::string why;
        CHK(check_all(idx, g, &why));
    }

    return u7f1::finish("TEST_CHECK_LAYER");
}

