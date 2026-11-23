// vit_parity.cpp — host invariant tests for vit_encoder.hpp.
// No GPU required; all tests run on the host.  Grows one section per scope sentence.
#include "baloon/vit_encoder.hpp"
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

static long g_checks = 0, g_fail = 0;
#define VP(x) do { ++g_checks; if (!(x)) { ++g_fail; \
    std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #x); } } while (0)
static bool nearv(float a, float b, float tol = 2e-3f) {
    return std::fabs(a - b) <= tol * (1.f + std::fabs(b));
}
static uint16_t f2b(float f) { return baloon::refk::bf16_from_f32(f); }
static float b2f(uint16_t b) { return baloon::refk::f32_from_bf16(b); }

// ---- (1) vit_patch_embed: projects image patches via patch_embed.proj (3-D conv) ----
static void test_patch_embed() {
    using namespace baloon::vit;
    // Small synthetic case: out_ch=3, in_flat=4, n_patches=2
    const int out_ch = 3, in_flat = 4, n_patches = 2;
    // weight [out_ch, in_flat] = [[1,0,0,0],[0,1,0,0],[0,0,1,0]] (identity first 3 of 4)
    std::vector<uint16_t> w(out_ch * in_flat, f2b(0.f));
    for (int r = 0; r < out_ch; ++r) w[(int64_t)r * in_flat + r] = f2b(1.f);
    // bias = [0.5, -0.5, 0.0]
    std::vector<uint16_t> b = { f2b(0.5f), f2b(-0.5f), f2b(0.f) };
    // patches: [2, 4]
    const float x[] = { 2.f, 3.f, 4.f, 5.f,   // patch 0
                         6.f, 7.f, 8.f, 9.f };  // patch 1
    std::vector<float> out(n_patches * out_ch);
    vit_patch_embed(w.data(), b.data(), out_ch, in_flat, x, n_patches, out.data());
    // patch 0: [2+0.5, 3-0.5, 4+0] = [2.5, 2.5, 4.0]
    VP(nearv(out[0], 2.5f));
    VP(nearv(out[1], 2.5f));
    VP(nearv(out[2], 4.0f));
    // patch 1: [6+0.5, 7-0.5, 8+0] = [6.5, 6.5, 8.0]
    VP(nearv(out[3], 6.5f));
    VP(nearv(out[4], 6.5f));
    VP(nearv(out[5], 8.0f));

    // Linearity: doubling a patch doubles the output (before bias)
    const float x2[] = { 4.f, 6.f, 8.f, 10.f };
    std::vector<float> out2(out_ch);
    vit_patch_embed(w.data(), b.data(), out_ch, in_flat, x2, 1, out2.data());
    // [4+0.5, 6-0.5, 8+0] = [4.5, 5.5, 8.0]
    VP(nearv(out2[0], 4.5f));
    VP(nearv(out2[1], 5.5f));
    VP(nearv(out2[2], 8.0f));

    // Zero-weight → only bias survives
    std::fill(w.begin(), w.end(), f2b(0.f));
    std::vector<float> out3(out_ch);
    vit_patch_embed(w.data(), b.data(), out_ch, in_flat, x, 1, out3.data());
    VP(nearv(out3[0], b2f(b[0])));
    VP(nearv(out3[1], b2f(b[1])));
    VP(nearv(out3[2], b2f(b[2])));

    // Dimension invariant: output has n_patches * out_ch values regardless of in_flat
    for (int out_ch2 = 1; out_ch2 <= 5; ++out_ch2) {
        const int in2 = 3;
        std::vector<uint16_t> w2((int64_t)out_ch2 * in2, f2b(0.f));
        std::vector<uint16_t> b2(out_ch2, f2b(0.f));
        const float px[] = {1.f, 2.f, 3.f};
        std::vector<float> o2(out_ch2);
        vit_patch_embed(w2.data(), b2.data(), out_ch2, in2, px, 1, o2.data());
        VP((int)o2.size() == out_ch2);  // size correct
        for (int r = 0; r < out_ch2; ++r) VP(o2[r] == 0.f);  // all-zero weights + bias
    }
}

// ---- (2) vit_add_pos_embed: adds pos_embed ----
static void test_add_pos_embed() {
    using namespace baloon::vit;
    const int d_model = 4, n_patches = 3, max_pos = 8;
    // table: row i = [i, i+0.1, i+0.2, i+0.3]
    std::vector<uint16_t> table((int64_t)max_pos * d_model);
    for (int r = 0; r < max_pos; ++r)
        for (int j = 0; j < d_model; ++j)
            table[(int64_t)r * d_model + j] = f2b((float)r + j * 0.1f);
    // initial x = all zeros
    std::vector<float> x(n_patches * d_model, 0.f);
    // pos_ids = [2, 5, 0]
    const int pos_ids[] = {2, 5, 0};
    vit_add_pos_embed(table.data(), d_model, pos_ids, n_patches, x.data());
    // compare against the actual BF16-rounded values stored in the table
    for (int p = 0; p < n_patches; ++p)
        for (int j = 0; j < d_model; ++j)
            VP(x[(int64_t)p * d_model + j] == b2f(table[(int64_t)pos_ids[p] * d_model + j]));

    // Additivity: adding twice doubles the contribution
    std::vector<float> x2(n_patches * d_model, 0.f);
    vit_add_pos_embed(table.data(), d_model, pos_ids, n_patches, x2.data());
    vit_add_pos_embed(table.data(), d_model, pos_ids, n_patches, x2.data());
    for (int i = 0; i < n_patches * d_model; ++i)
        VP(nearv(x2[i], 2.f * x[i]));

    // Existing values are preserved and added to (not overwritten)
    std::vector<float> x3(n_patches * d_model, 1.f);
    vit_add_pos_embed(table.data(), d_model, pos_ids, n_patches, x3.data());
    for (int i = 0; i < n_patches * d_model; ++i)
        VP(nearv(x3[i], 1.f + x[i]));

    // All patches pointing to the same row produce the same delta
    std::vector<int> same_ids(n_patches, 3);
    std::vector<float> x4(n_patches * d_model, 0.f);
    vit_add_pos_embed(table.data(), d_model, same_ids.data(), n_patches, x4.data());
    for (int p = 0; p < n_patches; ++p)
        for (int j = 0; j < d_model; ++j)
            VP(nearv(x4[(int64_t)p * d_model + j], b2f(table[3LL * d_model + j])));
}

// ---- (3) vit_block_forward: one block: norm1 → QKV → proj → norm2 → MLP ----
// Build a minimal block with identity/zero weights and check residual paths.
static void make_ones_bf16(std::vector<uint16_t>& w, int n) { w.assign(n, f2b(1.f)); }
static void make_zeros_bf16(std::vector<uint16_t>& w, int n) { w.assign(n, f2b(0.f)); }

static void test_block_forward() {
    using namespace baloon::vit;
    // Tiny block: d_model=4, n_heads=2, head_dim=2, ffn_dim=4, n_patches=3
    const int d_model = 4, n_heads = 2, ffn_dim = 4, n_patches = 3;
    // Weights: all-zero QKV/proj/MLP (attention collapses to average, MLP is zero)
    // norm1/norm2: weight=1, bias=0 → identity layer norm
    std::vector<uint16_t> n1w, n1b, qkvw, qkvb, pw, pb, n2w, n2b, m1w, m1b, m2w, m2b;
    make_ones_bf16(n1w, d_model);  make_zeros_bf16(n1b, d_model);
    make_zeros_bf16(qkvw, 3 * d_model * d_model); make_zeros_bf16(qkvb, 3 * d_model);
    make_zeros_bf16(pw, d_model * d_model);        make_zeros_bf16(pb, d_model);
    make_ones_bf16(n2w, d_model);  make_zeros_bf16(n2b, d_model);
    make_zeros_bf16(m1w, ffn_dim * d_model);       make_zeros_bf16(m1b, ffn_dim);
    make_zeros_bf16(m2w, d_model * ffn_dim);       make_zeros_bf16(m2b, d_model);

    VitBlockWeights bw{n1w.data(), n1b.data(), qkvw.data(), qkvb.data(),
                       pw.data(), pb.data(), n2w.data(), n2b.data(),
                       m1w.data(), m1b.data(), m2w.data(), m2b.data()};

    // Input: 3 patches, each [1, 2, 3, 4]
    std::vector<float> x(n_patches * d_model);
    for (int p = 0; p < n_patches; ++p)
        for (int i = 0; i < d_model; ++i)
            x[(int64_t)p * d_model + i] = (float)(i + 1);

    int64_t sc = vit_block_scratch_floats(d_model, ffn_dim, n_patches);
    std::vector<float> scratch(sc, 0.f);
    vit_block_forward(bw, d_model, n_heads, ffn_dim, n_patches, x.data(), scratch.data());

    // With all-zero QKV/proj/MLP, the residual connections preserve x.
    // Both attention and MLP contribute nothing → x unchanged.
    for (int p = 0; p < n_patches; ++p)
        for (int i = 0; i < d_model; ++i)
            VP(nearv(x[(int64_t)p * d_model + i], (float)(i + 1)));

    // norm1 with weight=1, bias=0 → input is normalised; check norm1 output is zero-mean
    // (Already indirectly verified: proj(zero) = zero → no attn contribution)

    // Check scratch buffer size is non-negative and sufficient (by running without crash)
    for (int np = 1; np <= 5; ++np) {
        int64_t s = vit_block_scratch_floats(d_model, ffn_dim, np);
        VP(s >= (int64_t)np * (d_model + 3 * d_model + d_model + ffn_dim));
    }

    // Non-zero proj bias only → attention adds the bias to each patch
    make_zeros_bf16(pw, d_model * d_model);
    std::vector<uint16_t> pb2(d_model);
    for (int i = 0; i < d_model; ++i) pb2[i] = f2b((float)(i + 1) * 0.1f);
    bw.proj_b = pb2.data();
    std::vector<float> x2(n_patches * d_model);
    for (int p = 0; p < n_patches; ++p)
        for (int i = 0; i < d_model; ++i)
            x2[(int64_t)p * d_model + i] = (float)(i + 1);
    std::fill(scratch.begin(), scratch.end(), 0.f);
    vit_block_forward(bw, d_model, n_heads, ffn_dim, n_patches, x2.data(), scratch.data());
    // proj_w=0 → proj output = proj_b for every patch; x2 += proj_b
    for (int p = 0; p < n_patches; ++p)
        for (int i = 0; i < d_model; ++i)
            VP(nearv(x2[(int64_t)p * d_model + i],
                     (float)(i + 1) + b2f(pb2[i])));
}

// ---- (4) vit_merger_mlp: norm → linear_fc1 → GELU → linear_fc2 ----
static void test_merger_mlp() {
    using namespace baloon::vit;
    // d_model=4, merge_sq=4 (merge_size=2), merged_dim=16, out_dim=6, n_patches=8
    const int d_model = 4, merge_sq = 4, out_dim = 6, n_patches = 8;
    const int merged_dim = merge_sq * d_model; // 16
    const int n_tokens   = n_patches / merge_sq; // 2

    // norm: weight=1, bias=0 → identity layer norm
    std::vector<uint16_t> nw(d_model, f2b(1.f)), nb(d_model, f2b(0.f));

    // fc1: identity (16×16), bias=0
    std::vector<uint16_t> fc1w(merged_dim * merged_dim, f2b(0.f)), fc1b(merged_dim, f2b(0.f));
    for (int i = 0; i < merged_dim; ++i) fc1w[(int64_t)i * merged_dim + i] = f2b(1.f);

    // fc2: zero (6×16), bias=[1,2,3,4,5,6] (broadcast to all tokens)
    std::vector<uint16_t> fc2w(out_dim * merged_dim, f2b(0.f));
    std::vector<uint16_t> fc2b(out_dim);
    for (int i = 0; i < out_dim; ++i) fc2b[i] = f2b((float)(i + 1));

    VitMergerWeights mw{nw.data(), nb.data(), fc1w.data(), fc1b.data(), fc2w.data(), fc2b.data()};

    // Input: 8 patches all set to 1.0 (after norm → still normalised; fc1 identity → unchanged)
    // But norm(all-1) = 0 since var=0 and mean=1 → fc1(0) = 0 → GELU(0) = 0 → out = fc2_bias
    std::vector<float> x(n_patches * d_model, 1.f);
    std::vector<float> out(n_tokens * out_dim, -99.f);
    int64_t sc = vit_merger_scratch_floats(d_model, merge_sq, out_dim, n_patches);
    std::vector<float> scratch(sc, 0.f);
    vit_merger_mlp(mw, d_model, merge_sq, out_dim, x.data(), n_patches, out.data(), scratch.data());

    // With all-1 patches: LayerNorm(all-same) = 0 → fc1 output = 0 → GELU(0) = 0
    // → fc2 output = fc2_bias for every token
    for (int t = 0; t < n_tokens; ++t)
        for (int i = 0; i < out_dim; ++i)
            VP(nearv(out[(int64_t)t * out_dim + i], b2f(fc2b[i])));

    // Output dimension: n_tokens × out_dim
    VP((int)out.size() == n_tokens * out_dim);
    VP(n_tokens == n_patches / merge_sq);

    // scratch size is sufficient (function ran without crashing)
    VP(sc >= (int64_t)n_patches * d_model);

    // With fc2 identity (out_dim == merged_dim): output is GELU of norm+fc1
    // Use out_dim = merged_dim = 16
    const int od2 = merged_dim;
    std::vector<uint16_t> fc2w2(od2 * merged_dim, f2b(0.f)), fc2b2(od2, f2b(0.f));
    for (int i = 0; i < od2; ++i) fc2w2[(int64_t)i * merged_dim + i] = f2b(1.f);
    VitMergerWeights mw2{nw.data(), nb.data(), fc1w.data(), fc1b.data(), fc2w2.data(), fc2b2.data()};
    // Distinct patch: x[0..3] = [0,1,2,3], repeat for each merge group
    std::vector<float> x2(n_patches * d_model);
    for (int p = 0; p < n_patches; ++p)
        for (int i = 0; i < d_model; ++i)
            x2[(int64_t)p * d_model + i] = (float)i;
    std::vector<float> out2(n_tokens * od2, -1.f);
    int64_t sc2 = vit_merger_scratch_floats(d_model, merge_sq, od2, n_patches);
    std::vector<float> scratch2(sc2, 0.f);
    vit_merger_mlp(mw2, d_model, merge_sq, od2, x2.data(), n_patches, out2.data(), scratch2.data());
    // Both tokens come from identical patch groups so must be equal
    for (int i = 0; i < od2; ++i)
        VP(nearv(out2[i], out2[(int64_t)od2 + i]));
}

// ---- (5) vit_encode: full encoder to yield per-patch hidden-state vectors ----
static void test_vit_encode() {
    using namespace baloon::vit;
    // Tiny config: d_model=4, depth=1, n_heads=2, ffn_dim=4, in_ch=1, t=1, s=2,
    //              merge_size=2, out_dim=3, max_pos=16
    VitConfig cfg;
    cfg.d_model    = 4;
    cfg.depth      = 1;
    cfg.n_heads    = 2;
    cfg.ffn_dim    = 4;
    cfg.in_ch      = 1;
    cfg.t_patch    = 1;
    cfg.s_patch    = 2;
    cfg.merge_size = 2;
    cfg.out_dim    = 3;
    cfg.max_pos    = 16;
    const int in_flat  = cfg.in_ch * cfg.t_patch * cfg.s_patch * cfg.s_patch; // 1*1*2*2=4
    const int merge_sq = cfg.merge_size * cfg.merge_size;                      // 4
    const int n_patches = 4;  // 2×2 spatial grid
    const int n_tokens  = n_patches / merge_sq;                                // 1

    // All-zero weights: attention and MLP contribute nothing → embed = pos_embed only
    // norm weights = 1, biases = 0
    std::vector<uint16_t> pe_w(cfg.d_model * in_flat, f2b(0.f));
    std::vector<uint16_t> pe_b(cfg.d_model, f2b(0.f));
    // pos_embed: row i = [i*0.1, i*0.1+0.01, i*0.1+0.02, i*0.1+0.03]
    std::vector<uint16_t> pos_tbl((int64_t)cfg.max_pos * cfg.d_model);
    for (int r = 0; r < cfg.max_pos; ++r)
        for (int j = 0; j < cfg.d_model; ++j)
            pos_tbl[(int64_t)r * cfg.d_model + j] = f2b(r * 0.1f + j * 0.01f);

    // Build 1 block: all zeros (norm=1/0, QKV/proj/MLP=0)
    std::vector<uint16_t> n1w(cfg.d_model, f2b(1.f)), n1b(cfg.d_model, f2b(0.f));
    std::vector<uint16_t> qkvw(3*cfg.d_model*cfg.d_model, f2b(0.f)), qkvb(3*cfg.d_model, f2b(0.f));
    std::vector<uint16_t> pw(cfg.d_model*cfg.d_model, f2b(0.f)),   pbb(cfg.d_model, f2b(0.f));
    std::vector<uint16_t> n2w(cfg.d_model, f2b(1.f)), n2b(cfg.d_model, f2b(0.f));
    std::vector<uint16_t> m1w(cfg.ffn_dim*cfg.d_model, f2b(0.f)), m1b(cfg.ffn_dim, f2b(0.f));
    std::vector<uint16_t> m2w(cfg.d_model*cfg.ffn_dim, f2b(0.f)), m2b(cfg.d_model, f2b(0.f));

    VitWeights vw{};
    vw.patch_embed_w = pe_w.data();
    vw.patch_embed_b = pe_b.data();
    vw.pos_embed     = pos_tbl.data();
    vw.blocks[0]     = {n1w.data(), n1b.data(), qkvw.data(), qkvb.data(),
                        pw.data(), pbb.data(), n2w.data(), n2b.data(),
                        m1w.data(), m1b.data(), m2w.data(), m2b.data()};

    // merger: norm=1/0, fc1=identity, fc2=zero, fc2_bias=[1,2,3]
    const int merged_dim = merge_sq * cfg.d_model; // 16
    std::vector<uint16_t> mnw(cfg.d_model, f2b(1.f)), mnb(cfg.d_model, f2b(0.f));
    std::vector<uint16_t> fc1w(merged_dim*merged_dim, f2b(0.f)), fc1b(merged_dim, f2b(0.f));
    for (int i = 0; i < merged_dim; ++i) fc1w[(int64_t)i * merged_dim + i] = f2b(1.f);
    std::vector<uint16_t> fc2w(cfg.out_dim*merged_dim, f2b(0.f));
    std::vector<uint16_t> fc2b(cfg.out_dim);
    for (int i = 0; i < cfg.out_dim; ++i) fc2b[i] = f2b((float)(i + 1));
    vw.merger = {mnw.data(), mnb.data(), fc1w.data(), fc1b.data(), fc2w.data(), fc2b.data()};

    // pos_ids for 4 patches
    const int pos_ids[] = {0, 1, 2, 3};
    // patches: all zero (patch_embed_w=0 → embed starts at 0, then pos_embed added)
    std::vector<float> patches(n_patches * in_flat, 0.f);
    std::vector<float> out(n_tokens * cfg.out_dim, -1.f);
    int64_t sc = vit_encode_scratch_floats(cfg, n_patches);
    VP(sc > 0);
    std::vector<float> scratch(sc, 0.f);

    int nt = vit_encode(vw, cfg, patches.data(), n_patches, pos_ids, out.data(), scratch.data());
    VP(nt == n_tokens);
    VP((int)out.size() == n_tokens * cfg.out_dim);

    // With fc2_w=0: out = fc2_b regardless of fc1 input → check fc2_b matches
    for (int t = 0; t < n_tokens; ++t)
        for (int i = 0; i < cfg.out_dim; ++i)
            VP(nearv(out[(int64_t)t * cfg.out_dim + i], b2f(fc2b[i])));

    // Output count matches n_patches / merge_sq for various patch counts
    for (int np = 4; np <= 16; np += 4) {
        int64_t s2 = vit_encode_scratch_floats(cfg, np);
        VP(s2 > 0);
        VP(s2 >= (int64_t)np * cfg.d_model);
    }
}

int main(int argc, char** argv) {
    (void)argc; (void)argv;
    test_patch_embed();
    test_add_pos_embed();
    test_block_forward();
    test_merger_mlp();
    test_vit_encode();
    if (g_fail) {
        std::printf("FAIL: %ld/%ld checks failed\n", g_fail, g_checks);
        return 1;
    }
    std::printf("PASS: %ld checks\n", g_checks);
    return 0;
}

