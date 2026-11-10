// test_u11_core.cpp - layout checker (src/core/layout.cpp) and NativeDense (src/core/native_dense.cpp) invariants.
// Opening the device (cudaMalloc / cudaSetDevice) is part of the test; no device -> failure.
#include <cuda_runtime.h>

#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "baloon/rt_layout.hpp"
#include "baloon/u11_native_dense.hpp"

using namespace baloon;
static int g_fail = 0, g_checks = 0;
#define CHECK(c) do { ++g_checks; if (!(c)) { ++g_fail; if (g_fail < 60) std::printf("FAIL %s:%d  %s\n", __FILE__, __LINE__, #c); } } while (0)

static void add_layer(rt::LayoutIndex& idx, const rt::LayoutGeometry& g, int l) {
    rt::LayerView v; v.layer = l; v.qsa = g.is_qsa(l);
    idx[v.name("attn_norm")] = {g.n_embd, 0}; idx[v.name("ffn_norm")] = {g.n_embd, 0};
    if (v.qsa) {
        idx[v.name("attn_q")] = {g.n_head * g.head_dim, g.n_embd}; idx[v.name("attn_k")] = {g.n_head_kv * g.head_dim, g.n_embd};
        idx[v.name("attn_v")] = {g.n_head_kv * g.head_dim, g.n_embd}; idx[v.name("attn_output")] = {g.n_embd, g.n_head * g.head_dim};
    } else {
        idx[v.name("ssm_in")] = {5, g.n_embd}; idx[v.name("ssm_out")] = {g.n_embd, 7};
    }
}

static void test_layout() {
    rt::LayoutGeometry g; g.n_layers = 8; g.n_embd = 64; g.n_head = 4; g.head_dim = 16; g.n_head_kv = 2;
    CHECK(!g.is_qsa(0) && !g.is_qsa(2) && g.is_qsa(3) && g.is_qsa(7));
    rt::LayerView lv; lv.layer = 12;
    CHECK(lv.name("attn_q") == "blk.12.attn_q");
    rt::LayoutIndex idx; for (int l = 0; l < g.n_layers; ++l) add_layer(idx, g, l);
    std::string why;
    for (int l = 0; l < g.n_layers; ++l) CHECK(rt::check_layer(idx, g, l, &why));
    CHECK(rt::check_all(idx, g, &why));
    CHECK(!rt::check_layer(idx, g, -1, &why) && !rt::check_layer(idx, g, g.n_layers, &why));
    { auto m = idx; m.erase("blk.3.attn_k"); CHECK(!rt::check_layer(m, g, 3, &why) && why.find("blk.3.attn_k") != std::string::npos); CHECK(!rt::check_all(m, g, &why)); CHECK(rt::check_layer(m, g, 2, &why)); }
    { auto m = idx; m["blk.3.attn_q"].rows += 1; CHECK(!rt::check_layer(m, g, 3, &why)); CHECK(!rt::check_all(m, g, nullptr)); }
    { auto m = idx; m["blk.3.attn_q"].cols += 1; CHECK(!rt::check_layer(m, g, 3, &why)); }
    { auto m = idx; m["blk.0.attn_norm"].cols = 4; CHECK(!rt::check_layer(m, g, 0, &why) && why.find("vector") != std::string::npos); }
    { auto m = idx; m["blk.0.attn_norm"].rows = 63; CHECK(!rt::check_layer(m, g, 0, &why)); }
    { auto m = idx; m["blk.0.ssm_in"].rows = 0; CHECK(!rt::check_layer(m, g, 0, &why)); }                       // a matrix needs positive dimensions
    { auto m = idx; m["blk.3.ssm_in"] = {5, g.n_embd}; CHECK(!rt::check_layer(m, g, 3, &why)); }                // a QSA layer must not carry GDN tensors
    { auto m = idx; m["blk.0.attn_q"] = {64, 64}; CHECK(!rt::check_layer(m, g, 0, &why)); }                     // nor a GDN layer QSA tensors
    { rt::LayoutGeometry bad = g; bad.n_embd = 0; CHECK(!rt::check_layer(idx, bad, 0, &why)); }
    { rt::LayoutGeometry none = g; none.n_layers = 0; CHECK(!rt::check_all(idx, none, &why)); }
    CHECK(rt::check_one(idx, rt::LayerView{3, true}, rt::Want2{"attn_q", 0, 0}, &why));
    CHECK(!rt::check_one(idx, rt::LayerView{3, true}, rt::Want2{"attn_q", 0, 5}, &why));
    CHECK(rt::check_one(idx, rt::LayerView{3, true}, rt::Want1{"attn_norm", 64}, &why) && !rt::check_one(idx, rt::LayerView{3, true}, rt::Want1{"attn_norm", 65}, &why));
    CHECK(!rt::check_one(idx, rt::LayerView{3, true}, rt::Want1{"nope", 1}, &why) && why.find("missing") != std::string::npos);
}

static std::vector<uint16_t> bf16s(std::mt19937& r, size_t n) { std::vector<uint16_t> v(n); for (auto& x : v) x = (uint16_t)(0x3800 + (r() & 0x3ff)); return v; }

static void test_dense() {
    core::WIndexRow r; r.placed = true; r.engine_kind = 1; r.dst_bytes = 12;
    CHECK(u11::eligible(r));
    { auto m = r; m.placed = false; CHECK(!u11::eligible(m)); }
    { auto m = r; m.engine_kind = 3; CHECK(!u11::eligible(m)); }
    { auto m = r; m.engine_kind = 0; CHECK(!u11::eligible(m)); }
    { auto m = r; m.dst_bytes = 0; CHECK(!u11::eligible(m)); }
    { auto m = r; m.dst_bytes = 7; CHECK(!u11::eligible(m)); }
    int ndev = 0; CHECK(cudaGetDeviceCount(&ndev) == cudaSuccess && ndev >= 1);

    std::mt19937 rng(7);
    const int64_t R1 = 6, C1 = 40, R2 = 3, C2 = 17;
    auto w1 = bf16s(rng, R1 * C1), w2 = bf16s(rng, R2 * C2);
    std::vector<uint8_t> src; src.insert(src.end(), (uint8_t*)w1.data(), (uint8_t*)(w1.data() + w1.size()));
    const uint64_t off2 = src.size(); src.insert(src.end(), (uint8_t*)w2.data(), (uint8_t*)(w2.data() + w2.size()));
    std::vector<float> junk(8, 1.f); const uint64_t off3 = src.size();
    src.insert(src.end(), (uint8_t*)junk.data(), (uint8_t*)(junk.data() + junk.size()));
    std::string index = "zeta.w 4 0 " + std::to_string(w1.size() * 2) + " " + std::to_string(w1.size() * 2) + " 0 0 0 0\n" +
                        "alpha.w 4 " + std::to_string(off2) + " " + std::to_string(w2.size() * 2) + " " + std::to_string(w2.size() * 2) + " 0 0 0 0\n" +
                        "other.f32 0 " + std::to_string(off3) + " 32 32 0 0 0 0\n";
    core::WReader rd = [&](uint64_t off, uint64_t n, uint8_t* buf) { if (off + n > src.size()) return false; std::memcpy(buf, src.data() + off, n); return true; };
    core::WeightTable tab, placer; std::string err;
    CHECK(placer.place(index, {}, &err));
    std::vector<uint8_t> arena(placer.pool_bytes() ? placer.pool_bytes() : 1);
    CHECK(tab.load_from(index, rd, arena.data(), arena.size(), {}, &err));
    u11::ShapeMap shapes; shapes["zeta.w"] = {R1, C1}; shapes["alpha.w"] = {R2, C2}; shapes["other.f32"] = {4, 2}; shapes["ghost"] = {1, 1};
    u11::NativeDense nd;
    CHECK(nd.served_names().empty() && nd.resident_bytes() == 0);
    CHECK(nd.load(tab, arena.data(), shapes, 0, &err));
    CHECK((nd.served_names() == std::vector<std::string>{"alpha.w", "zeta.w"}));                      // sorted; f32 and unknown rows skipped
    CHECK(nd.resident_bytes() == (w1.size() + w2.size()) * 2);
    const uint16_t* dev = nullptr; uint64_t rows = 0, cols = 0;
    CHECK(nd.find("zeta.w", &dev, &rows, &cols) && rows == (uint64_t)R1 && cols == (uint64_t)C1 && dev);
    std::vector<uint16_t> back(w1.size());
    CHECK(cudaMemcpy(back.data(), dev, back.size() * 2, cudaMemcpyDeviceToHost) == cudaSuccess && back == w1);   // bytes arrive unchanged
    CHECK(nd.find("alpha.w", &dev, &rows, &cols) && rows == (uint64_t)R2 && cols == (uint64_t)C2);
    std::vector<uint16_t> back2(w2.size());
    CHECK(cudaMemcpy(back2.data(), dev, back2.size() * 2, cudaMemcpyDeviceToHost) == cudaSuccess && back2 == w2);
    CHECK(!nd.find("other.f32", &dev, &rows, &cols) && !nd.find("ghost", &dev, &rows, &cols));
    u11::ShapeMap wrong = shapes; wrong["zeta.w"] = {R1, C1 + 1};
    CHECK(!nd.load(tab, arena.data(), wrong, 0, &err) && err.find("zeta.w") != std::string::npos);
    CHECK(nd.served_names().empty() && nd.resident_bytes() == 0);                                        // a failed load leaves nothing resident
    u11::ShapeMap zero = shapes; zero["zeta.w"] = {0, C1};
    CHECK(!nd.load(tab, arena.data(), zero, 0, &err));
    CHECK(!nd.load(tab, nullptr, shapes, 0, &err));
    CHECK(!nd.load(tab, arena.data(), shapes, ndev + 2, &err) && nd.served_names().empty());
    cudaGetLastError();
    CHECK(nd.load(tab, arena.data(), shapes, 0, &err) && nd.served_names().size() == 2);                 // reloading after failures works
}

int main() {
    test_layout();
    test_dense();
    std::printf("u11 core: %d checks, %d failed\n", g_checks, g_fail);
    return g_fail ? 1 : 0;
}
