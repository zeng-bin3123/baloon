// test_ilu_qsa.cu - GPU invariant tests of the QSA kernels (decode attention, block score, radix select, tile score,
// pooled append). Every check compares against a double-precision host reference or a LAYER-1 statement (chunk
// coverage, masking, merge equivalence to one-shot softmax, exact top-k with lowest-index tie budget, tail weight and
// bias, ring-buffer slots, rotation pairing, rejection of aliased/invalid launches). Exit code is non-zero on any failure.
#include <cuda_runtime.h>
#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <random>
#include <vector>
#include "baloon/f16_conversion.hpp"
#include "baloon/ilu_qsa_kernels.hpp"

using namespace baloon;
namespace q = baloon::ilu_qsa;

static long g_checks = 0, g_fail = 0;
#define CHECK(c) do { ++g_checks; if (!(c)) { ++g_fail; std::printf("FAIL %s:%d  %s\n", __FILE__, __LINE__, #c); } } while (0)
#define CUOK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { std::printf("CUDA error '%s' at %d\n", cudaGetErrorString(e_), __LINE__); std::exit(2); } } while (0)

// Typed device buffer with canary words on both sides (catches out-of-bounds writes).
template <typename T>
struct Dev {
    static constexpr size_t kPad = 1024 / sizeof(T);
    T* base = nullptr; size_t n = 0;
    static void fill_canary(T* p, size_t cnt) { std::vector<unsigned char> c(cnt * sizeof(T), 0xA5); CUOK(cudaMemcpy(p, c.data(), c.size(), cudaMemcpyHostToDevice)); }
    explicit Dev(size_t count) : n(count) {
        CUOK(cudaMalloc((void**)&base, (count + 2 * kPad) * sizeof(T)));
        fill_canary(base, count + 2 * kPad);
    }
    explicit Dev(const std::vector<T>& h) : Dev(h.size()) { CUOK(cudaMemcpy(ptr(), h.data(), n * sizeof(T), cudaMemcpyHostToDevice)); }
    ~Dev() { cudaFree(base); }
    T* ptr() const { return base + kPad; }
    std::vector<T> get() const { std::vector<T> h(n); CUOK(cudaMemcpy(h.data(), ptr(), n * sizeof(T), cudaMemcpyDeviceToHost)); return h; }
    bool canary_ok() const {
        std::vector<unsigned char> a(kPad * sizeof(T)), b(kPad * sizeof(T));
        CUOK(cudaMemcpy(a.data(), base, a.size(), cudaMemcpyDeviceToHost));
        CUOK(cudaMemcpy(b.data(), base + kPad + n, b.size(), cudaMemcpyDeviceToHost));
        for (size_t i = 0; i < a.size(); ++i) if (a[i] != 0xA5 || b[i] != 0xA5) return false;
        return true;
    }
};
static void sync() { CUOK(cudaDeviceSynchronize()); }

static std::mt19937 rng(20261103);
static std::vector<float> randv(size_t n, float s = 1.f) { std::normal_distribution<float> nd(0.f, s); std::vector<float> v(n); for (auto& x : v) x = nd(rng); return v; }

static double max_abs_err(const std::vector<float>& g, const std::vector<double>& r, double& scale) {
    double mx = 0; scale = 1e-30;
    for (size_t i = 0; i < r.size(); ++i) { mx = std::fmax(mx, std::fabs((double)g[i] - r[i])); scale = std::fmax(scale, std::fabs(r[i])); }
    return mx;
}

// ------------------------------------------------------------------------------------------------------------
// decode attention
struct DecodeCase { int nq, n_kv, cap, page, n_cells; bool q8; std::vector<int> n_ids; bool bad_ids; };

static void run_decode(const DecodeCase& cs, const char* name) {
    const int HD = 256, G = 12, n_head = cs.n_kv * G;
    const int pages = (cs.n_cells + cs.page - 1) / cs.page;
    const size_t rows = (size_t)pages * cs.n_kv * cs.page;
    const float scale = 1.f / 16.f;
    auto qv = randv((size_t)cs.nq * n_head * HD, 1.f);
    std::vector<float> kf = randv(rows * HD, 1.f), vf = randv(rows * HD, 1.f);
    std::vector<uint16_t> kh, vh; std::vector<int8_t> kc, vc; std::vector<float> ks, vs;
    std::vector<double> kd(rows * HD), vd(rows * HD);  // the exact values the kernel must see
    if (!cs.q8) {
        kh.resize(kf.size()); vh.resize(vf.size());
        for (size_t i = 0; i < kf.size(); ++i) { kh[i] = f16::float_to_half(kf[i]); vh[i] = f16::float_to_half(vf[i]); kd[i] = f16::half_to_float(kh[i]); vd[i] = f16::half_to_float(vh[i]); }
    } else {
        kc.resize(kf.size()); vc.resize(vf.size()); ks.resize(rows * 4); vs.resize(rows * 4);
        for (size_t r = 0; r < rows; ++r)
            for (int g = 0; g < 4; ++g) {
                float mk = 0, mv = 0;
                for (int j = 0; j < 64; ++j) { mk = std::fmax(mk, std::fabs(kf[r * HD + g * 64 + j])); mv = std::fmax(mv, std::fabs(vf[r * HD + g * 64 + j])); }
                ks[r * 4 + g] = mk / 127.f + 1e-8f; vs[r * 4 + g] = mv / 127.f + 1e-8f;
                for (int j = 0; j < 64; ++j) {
                    const size_t i = r * HD + g * 64 + j;
                    kc[i] = (int8_t)std::lrintf(kf[i] / ks[r * 4 + g]); vc[i] = (int8_t)std::lrintf(vf[i] / vs[r * 4 + g]);
                    kd[i] = (double)kc[i] * ks[r * 4 + g]; vd[i] = (double)vc[i] * vs[r * 4 + g];
                }
            }
    }
    // ids: distinct random cells per query, in random (not ascending) order; optional out-of-range / negative entries
    std::vector<int32_t> ids((size_t)cs.nq * cs.cap, 0), nids(cs.nq);
    for (int i = 0; i < cs.nq; ++i) {
        nids[i] = cs.n_ids[i];
        std::vector<int> perm(cs.n_cells); std::iota(perm.begin(), perm.end(), 0); std::shuffle(perm.begin(), perm.end(), rng);
        for (int c = 0; c < nids[i]; ++c) ids[(size_t)i * cs.cap + c] = perm[c % cs.n_cells];
        if (cs.bad_ids) for (int c = 3; c < nids[i]; c += 7) ids[(size_t)i * cs.cap + c] = (c % 2) ? cs.n_cells + c : -1 - c;
    }
    const int n_chunks = (cs.cap + 63) / 64;
    const size_t scratch_n = (size_t)cs.nq * n_chunks * n_head * (HD + 2);
    Dev<float> dq(qv), dks(cs.q8 ? ks : std::vector<float>(4, 0.f)), dvs(cs.q8 ? vs : std::vector<float>(4, 0.f)), dscr(scratch_n), dout((size_t)cs.nq * n_head * HD);
    Dev<int32_t> dids(ids), dn(nids);
    std::vector<unsigned char> kraw, vraw;
    auto raw = [&](auto& v) { std::vector<unsigned char> b(v.size() * sizeof(v[0])); std::memcpy(b.data(), v.data(), b.size()); return b; };
    kraw = cs.q8 ? raw(kc) : raw(kh); vraw = cs.q8 ? raw(vc) : raw(vh);
    Dev<unsigned char> dk(kraw), dv(vraw);
    q::DecodeArgs a{dq.ptr(), dk.ptr(), dv.ptr(), dks.ptr(), dvs.ptr(), dids.ptr(), dn.ptr(), dscr.ptr(), dout.ptr(),
                    cs.nq, cs.n_kv, cs.cap, cs.page, cs.n_cells, cs.q8, scale};
    const int rc = q::decode_attention(a, nullptr);
    CHECK(rc == q::kOk); sync();
    auto got = dout.get();

    // reference: one-shot softmax in double over the valid selected cells
    std::vector<double> ref((size_t)cs.nq * n_head * HD, 0.0);
    long attended_heads = 0;
    for (int i = 0; i < cs.nq; ++i)
        for (int h = 0; h < n_head; ++h) {
            const int kvh = h / G;
            std::vector<size_t> rws;
            for (int c = 0; c < nids[i]; ++c) {
                const int id = ids[(size_t)i * cs.cap + c];
                if (id < 0 || id >= cs.n_cells) continue;
                rws.push_back(((size_t)(id / cs.page) * cs.n_kv + kvh) * cs.page + id % cs.page);
            }
            if (rws.empty()) continue;
            ++attended_heads;
            std::vector<double> s(rws.size()); double m = -1e30;
            for (size_t j = 0; j < rws.size(); ++j) {
                double d = 0; for (int t = 0; t < HD; ++t) d += (double)qv[((size_t)i * n_head + h) * HD + t] * scale * kd[rws[j] * HD + t];
                s[j] = d; m = std::fmax(m, d);
            }
            double L = 0; for (auto& x : s) { x = std::exp(x - m); L += x; }
            for (size_t j = 0; j < rws.size(); ++j) for (int t = 0; t < HD; ++t) ref[((size_t)i * n_head + h) * HD + t] += s[j] / L * vd[rws[j] * HD + t];
        }
    double sc; const double err = max_abs_err(got, ref, sc);
    std::printf("  decode[%s] max_abs_err=%.3e (scale %.3f, attended heads %ld)\n", name, err, sc, attended_heads);
    CHECK(err < 2e-4 * std::fmax(1.0, sc));
    CHECK(dout.canary_ok()); CHECK(dscr.canary_ok());
    // scratch invariant: l >= 0 for every slot, acc/m/l regions contiguous as in qsa_decode:: offsets
    auto scr = dscr.get();
    bool lok = true;
    for (int i = 0; i < cs.nq; ++i) {
        const size_t xs = (size_t)n_chunks * n_head, base = (size_t)i * xs * (HD + 2);
        for (size_t j = 0; j < xs; ++j) if (!(scr[base + xs * HD + xs + j] >= 0.f)) lok = false;
    }
    CHECK(lok);
}

static void test_decode() {
    std::printf("[decode attention]\n");
    // single chunk, partial chunk (n_here < 64), multi-chunk with ragged tail, both pool formats, GQA kv heads = 2
    run_decode({1, 1, 64, 16, 200, false, {40}, false}, "fp16 1 kv, 40 cells");
    run_decode({2, 2, 256, 32, 700, false, {256, 130}, false}, "fp16 2 kv, 256+130 cells");
    run_decode({2, 2, 256, 32, 700, true, {256, 65}, false}, "q8 2 kv, 256+65 cells");
    run_decode({1, 1, 192, 16, 300, true, {129}, false}, "q8 chunk edge 64*2+1");
    run_decode({3, 1, 128, 8, 90, false, {0, 1, 128}, false}, "fp16 empty / single / full");
    run_decode({1, 2, 128, 16, 400, false, {128}, true}, "fp16 invalid ids masked");
    run_decode({1, 1, 64, 16, 400, true, {64}, true}, "q8 invalid ids masked");
    // all ids invalid -> zeros, no NaN
    {
        const int HD = 256, n_head = 12, cap = 64, n_cells = 32;
        auto qv = randv(n_head * HD);
        std::vector<uint16_t> kv((size_t)n_cells * HD, f16::float_to_half(1.f));
        std::vector<int32_t> ids(cap, -5), nids(1, 64);
        Dev<float> dq(qv), dscr((size_t)1 * 1 * n_head * (HD + 2)), dout(n_head * HD), dsd(4);
        Dev<int32_t> di(ids), dn(nids);
        Dev<uint16_t> dk(kv);
        q::DecodeArgs a{dq.ptr(), dk.ptr(), dk.ptr(), nullptr, nullptr, di.ptr(), dn.ptr(), dscr.ptr(), dout.ptr(), 1, 1, cap, 16, n_cells, false, 0.0625f};
        CHECK(q::decode_attention(a, nullptr) == q::kOk); sync();
        auto o = dout.get(); bool z = true; for (float x : o) if (x != 0.f) z = false;
        CHECK(z);
    }
    // rejections: aliased output/scratch, null, batch zero
    {
        const int HD = 256, n_head = 12;
        Dev<float> dq((size_t)n_head * HD), dscr((size_t)n_head * (HD + 2)), dout((size_t)n_head * HD);
        Dev<int32_t> di(64), dn(1);
        Dev<uint16_t> dk((size_t)64 * HD);
        q::DecodeArgs a{dq.ptr(), dk.ptr(), dk.ptr(), nullptr, nullptr, di.ptr(), dn.ptr(), dscr.ptr(), dout.ptr(), 1, 1, 64, 16, 64, false, 1.f};
        q::DecodeArgs b = a; b.out = b.scratch; CHECK(q::decode_attention(b, nullptr) == q::kInvalid);
        b = a; b.q = nullptr; CHECK(q::decode_attention(b, nullptr) == q::kInvalid);
        b = a; b.nq = 0; CHECK(q::decode_attention(b, nullptr) == q::kInvalid);
        b = a; b.q8 = true; CHECK(q::decode_attention(b, nullptr) == q::kInvalid);  // q8 without scales
    }
}

// ------------------------------------------------------------------------------------------------------------
// block score + select
static std::vector<double> ref_block_scores(const std::vector<float>& pooled, const std::vector<float>& iq, const std::vector<float>& hw,
                                            const std::vector<float>* bias, int n_kv) {
    const int n_bid = n_kv / 4, tail = n_kv % 4, nb = n_bid + (tail ? 1 : 0);
    std::vector<double> s(nb);
    for (int b = 0; b < nb; ++b) {
        double acc = 0;
        for (int h = 0; h < 4; ++h) { double d = 0; for (int t = 0; t < 128; ++t) d += (double)pooled[(size_t)b * 128 + t] * iq[h * 128 + t]; acc += hw[h] * std::fmax(d, 0.0); }
        if (bias) acc += (*bias)[b];
        if (b == n_bid && tail) acc += 1e9;
        s[b] = acc;
    }
    return s;
}

static void test_block_score() {
    std::printf("[block score / tile score]\n");
    for (int n_kv : {1, 3, 4, 5, 31, 32, 33, 130, 1023, 4099}) {
        const int rows = n_kv / 4 + 1, max_cells = n_kv + 7;
        const int rows_cap = max_cells / 4 + 1;
        const int R = std::max(rows, rows_cap);
        auto pooled = randv((size_t)R * 128), iq = randv(4 * 128), hw = randv(4, 0.5f);
        for (auto& x : hw) x = std::fabs(x) + 0.1f;
        std::vector<float> bias; const int nb = n_kv / 4 + (n_kv % 4 ? 1 : 0); bias = randv(nb, 0.3f);
        Dev<float> dp(pooled), dq(iq), dh(hw), db(bias), ds(nb), dcs(n_kv), dbs2(nb);
        for (int use_bias = 0; use_bias < 2; ++use_bias) {
            const std::vector<float>* bp = use_bias ? &bias : nullptr;
            auto ref = ref_block_scores(pooled, iq, hw, bp, n_kv);
            CHECK(q::block_score(dp.ptr(), dq.ptr(), dh.ptr(), use_bias ? db.ptr() : nullptr, ds.ptr(), n_kv, R, nullptr) == q::kOk); sync();
            auto g = ds.get(); bool ok = true;
            for (int b = 0; b < nb; ++b) { const double tol = 1e-4 * std::fmax(1.0, std::fabs(ref[b])) + (ref[b] > 5e8 ? 200.0 : 0.0); if (std::fabs(g[b] - ref[b]) > tol) ok = false; }
            CHECK(ok); CHECK(ds.canary_ok());
            // tail bias lands exactly on the partial block, only when n_kv % 4 != 0
            if (n_kv % 4) CHECK(g[nb - 1] > 5e8); else if (nb >= 1) CHECK(g[nb - 1] < 5e8);
            for (int b = 0; b + 1 < nb; ++b) if (g[b] > 5e8) ok = false;
            CHECK(ok);

            // tile kernel: same scores, plus broadcast to cells
            const int pos = n_kv - 1, n_bid = n_kv / 4, width = std::min(n_kv, 2051);
            CHECK(q::mma_block_score(dp.ptr(), dq.ptr(), dh.ptr(), use_bias ? db.ptr() : nullptr, dbs2.ptr(), dcs.ptr(), pos, n_kv, n_bid, width, max_cells, R, nullptr) ==
                  (rows_cap == R ? q::kOk : q::kInvalid));
            if (rows_cap != R) continue;
            sync();
            auto gb = dbs2.get(), gc = dcs.get(); bool ok2 = true, ok3 = true;
            for (int b = 0; b < nb; ++b) { const double tol = 1e-3 * std::fmax(1.0, std::fabs(ref[b])) + (ref[b] > 5e8 ? 200.0 : 0.0); if (std::fabs(gb[b] - ref[b]) > tol) ok2 = false; }
            for (int c = 0; c < n_kv; ++c) if (gc[c] != gb[c / 4]) ok3 = false;  // exact broadcast
            CHECK(ok2); CHECK(ok3); CHECK(dbs2.canary_ok()); CHECK(dcs.canary_ok());
        }
    }
    // step guard: inconsistent pos / n_bid / width are rejected, nothing launched
    {
        const int n_kv = 100, max_cells = 128, rows = max_cells / 4 + 1;
        Dev<float> dp((size_t)rows * 128), dq(512), dh(4), dcs(n_kv);
        auto call = [&](int pos, int n, int nbid, int w, int rws) { return q::mma_block_score(dp.ptr(), dq.ptr(), dh.ptr(), nullptr, nullptr, dcs.ptr(), pos, n, nbid, w, max_cells, rws, nullptr); };
        CHECK(call(99, 100, 25, 100, rows) == q::kOk); sync();
        CHECK(call(98, 100, 25, 100, rows) == q::kInvalid);
        CHECK(call(99, 100, 24, 100, rows) == q::kInvalid);
        CHECK(call(99, 100, 25, 99, rows) == q::kInvalid);
        CHECK(call(99, 100, 25, 100, rows - 1) == q::kInvalid);
        // width formula: width = min(n, 2051)
        Dev<float> dp2((size_t)(5000 / 4 + 1) * 128), dcs2(3000);
        CHECK(q::mma_block_score(dp2.ptr(), dq.ptr(), dh.ptr(), nullptr, nullptr, dcs2.ptr(), 2999, 3000, 750, 2051, 5000, 5000 / 4 + 1, nullptr) == q::kOk); sync();
        CHECK(q::mma_block_score(dp2.ptr(), dq.ptr(), dh.ptr(), nullptr, nullptr, dcs2.ptr(), 2999, 3000, 750, 3000, 5000, 5000 / 4 + 1, nullptr) == q::kInvalid);
        // aliasing: output over input is rejected
        CHECK(q::mma_block_score(dp.ptr(), dq.ptr(), dh.ptr(), nullptr, dp.ptr(), dcs.ptr(), 99, 100, 25, 100, max_cells, rows, nullptr) == q::kInvalid);
    }
}

// host reference of the selection: the `width` highest-scoring cells by block, ties broken by lowest cell index
static std::vector<int32_t> ref_select(const std::vector<float>& sc, int n_kv, int width) {
    std::vector<int32_t> out;
    if (n_kv <= width) { for (int i = 0; i < n_kv; ++i) out.push_back(i); return out; }
    std::vector<int> cells(n_kv); std::iota(cells.begin(), cells.end(), 0);
    std::stable_sort(cells.begin(), cells.end(), [&](int a, int b) { return q::order_key(sc[a / 4]) > q::order_key(sc[b / 4]); });
    cells.resize(width); std::sort(cells.begin(), cells.end());
    return std::vector<int32_t>(cells.begin(), cells.end());
}

static void run_select(const std::vector<float>& sc, int n_kv, int width, const char* name) {
    const int nb = n_kv / 4 + (n_kv % 4 ? 1 : 0);
    Dev<float> ds(sc); Dev<int32_t> dids(width), dn(1);
    CHECK(q::block_select(ds.ptr(), n_kv, width, dids.ptr(), dn.ptr(), nullptr) == q::kOk); sync();
    auto ids = dids.get(); const int n_out = dn.get()[0];
    auto ref = ref_select(sc, n_kv, width);
    const bool match = n_out == (int)ref.size() && std::equal(ref.begin(), ref.end(), ids.begin());
    if (!match) std::printf("  select[%s] n_kv=%d width=%d: n_out=%d ref=%zu\n", name, n_kv, width, n_out, ref.size());
    CHECK(match);
    // invariants: ascending, distinct, in range, exactly min(n_kv, width) cells
    bool asc = true; for (int i = 1; i < n_out; ++i) if (ids[i] <= ids[i - 1]) asc = false;
    CHECK(asc); CHECK(n_out == std::min(n_kv, width));
    if (n_out > 0) CHECK(ids[0] >= 0 && ids[n_out - 1] < n_kv);
    // padding after the selection (identity branch only) is -1
    for (int i = n_out; i < width; ++i) CHECK(ids[i] == -1);
    CHECK(dids.canary_ok()); CHECK(dn.canary_ok());
    (void)nb;
}

static void test_select() {
    std::printf("[block select]\n");
    // identity branch
    { std::vector<float> s(10, 1.f); run_select(s, 37, 2051, "identity"); run_select(std::vector<float>(1, 3.f), 1, 5, "single cell"); }
    // random scores, many sizes (nb < 256, nb == 256 +/- 1, nb >> 256), tail weights 0..3
    for (int n_kv : {2052, 2053, 2054, 2055, 2056, 4097, 8191, 20000, 65537}) {
        const int nb = n_kv / 4 + (n_kv % 4 ? 1 : 0);
        auto sc = randv(nb, 3.f);
        if (n_kv % 4) sc[nb - 1] += 1e9f;
        run_select(sc, n_kv, 2051, "random");
        run_select(sc, n_kv, 1000, "random w=1000");
        run_select(sc, n_kv, 1, "w=1");
    }
    // heavy ties: only a few distinct values, so the threshold bin holds many blocks split across threads
    for (int n_kv : {3000, 5001, 12345, 40000}) {
        const int nb = n_kv / 4 + (n_kv % 4 ? 1 : 0);
        std::vector<float> sc(nb); std::uniform_int_distribution<int> u(0, 3);
        for (auto& x : sc) x = (float)u(rng);
        run_select(sc, n_kv, 2051, "ties");
        run_select(sc, n_kv, 777, "ties w=777");
    }
    // all equal: lowest cells win, exactly `width` of them (the budget clamp)
    for (int n_kv : {2052, 2053, 9999}) { std::vector<float> sc(n_kv / 4 + (n_kv % 4 ? 1 : 0), 1.f); run_select(sc, n_kv, 2051, "all equal"); }
    // negative scores and mixed signs: order key must be monotone across the sign bit
    { std::vector<float> sc(2000); for (size_t i = 0; i < sc.size(); ++i) sc[i] = (float)((int)(i % 17) - 8) * 0.5f; run_select(sc, 8000, 2051, "mixed sign"); }
    // tail block is always selected through the bias, and its partial weight is honoured
    for (int tailw = 1; tailw <= 3; ++tailw) {
        const int n_kv = 4 * 1000 + tailw, nb = 1001;
        auto sc = randv(nb, 1.f); sc[nb - 1] += 1e9f;
        const int width = 2051;
        Dev<float> ds(sc); Dev<int32_t> dids(width), dn(1);
        CHECK(q::block_select(ds.ptr(), n_kv, width, dids.ptr(), dn.ptr(), nullptr) == q::kOk); sync();
        auto ids = dids.get(); bool has_tail = true;
        for (int c = 0; c < tailw; ++c) if (!std::binary_search(ids.begin(), ids.begin() + dn.get()[0], 4000 + c)) has_tail = false;
        CHECK(has_tail);
        CHECK(!std::binary_search(ids.begin(), ids.begin() + dn.get()[0], n_kv));  // never a cell beyond n_kv
    }
    // rejections
    { Dev<float> ds(100); Dev<int32_t> di(10), dn(1);
      CHECK(q::block_select(ds.ptr(), 0, 10, di.ptr(), dn.ptr(), nullptr) == q::kInvalid);
      CHECK(q::block_select(ds.ptr(), 100, 0, di.ptr(), dn.ptr(), nullptr) == q::kInvalid);
      CHECK(q::block_select(ds.ptr(), 100, 10, (int32_t*)ds.ptr(), dn.ptr(), nullptr) == q::kInvalid); }
}

// ------------------------------------------------------------------------------------------------------------
// pooled append
static void ref_pool(const std::vector<std::vector<double>>& keys4, long long rpos, float freq, float eps, std::vector<double>& out) {
    out.assign(128, 0.0);
    for (int d = 0; d < 128; ++d) { double s = 0; for (int i = 0; i < 4; ++i) s += keys4[i][d]; out[d] = s / 4.0; }
    double ss = 0; for (double x : out) ss += x * x;
    const double inv = 1.0 / std::sqrt(ss / 128.0 + (double)eps);
    for (auto& x : out) x *= inv;
    std::vector<double> y = out;
    for (int d = 0; d < 32; ++d) {
        const double ang = (double)rpos * std::pow((double)freq, (double)d);
        out[d] = y[d] * std::cos(ang) - y[d + 32] * std::sin(ang);
        out[d + 32] = y[d + 32] * std::cos(ang) + y[d] * std::sin(ang);
    }
}

static void test_append() {
    std::printf("[pooled append]\n");
    const int max_cells = 64, rows = max_cells / 4 + 1;
    const long long pos_base = 64; const float freq = 0.8f, eps = 1e-6f;
    std::vector<std::vector<float>> keys;
    for (int i = 0; i < max_cells; ++i) keys.push_back(randv(128, 1.f + 0.1f * (i % 5)));
    Dev<float> dkey(128), dtail(3 * 128), dpool((size_t)rows * 128), ddead(128);
    std::vector<float> zero_pool((size_t)rows * 128, 0.f);
    CUOK(cudaMemcpy(dpool.ptr(), zero_pool.data(), zero_pool.size() * 4, cudaMemcpyHostToDevice));
    std::vector<std::vector<double>> pool_ref(rows, std::vector<double>(128, 0.0));
    std::vector<double> dead_ref;
    std::vector<std::vector<double>> ref_tail;
    double worst = 0;
    for (int pos = 0; pos < max_cells; ++pos) {
        CUOK(cudaMemcpy(dkey.ptr(), keys[pos].data(), 512, cudaMemcpyHostToDevice));
        q::AppendArgs a{dkey.ptr(), dtail.ptr(), dpool.ptr(), ddead.ptr(), pos, max_cells, rows, pos_base, freq, eps};
        CHECK(q::pool_append(a, nullptr) == q::kOk); sync();
        std::vector<std::vector<double>> k4;
        if (pos == 0) {
            std::vector<double> k0(keys[0].begin(), keys[0].end());
            k4 = {k0, k0, k0, k0};
            ref_pool(k4, 0, freq, eps, pool_ref[0]); dead_ref = pool_ref[0];
        } else if (pos % 4 == 3) {
            for (int i = 0; i < 4; ++i) k4.push_back(std::vector<double>(keys[pos - 3 + i].begin(), keys[pos - 3 + i].end()));
            ref_pool(k4, pos_base + 4ll * (pos / 4), freq, eps, pool_ref[pos / 4]);
            pool_ref[pos / 4 + 1] = dead_ref;
        }
        // tail rows hold the raw keys of the current block's slots written so far
        if (pos % 4 < 3) {
            auto t = dtail.get();
            for (int s = 0; s <= pos % 4; ++s) { bool eq = true; for (int d = 0; d < 128; ++d) if (t[s * 128 + d] != keys[pos - (pos % 4) + s][d]) eq = false; CHECK(eq); }
        }
        auto p = dpool.get();
        // rows 0..(completed blocks) match the reference
        const int completed = pos == 0 ? 1 : (pos % 4 == 3 ? pos / 4 + 1 : pos / 4);
        for (int r = 0; r < completed && r < rows; ++r) for (int d = 0; d < 128; ++d) worst = std::fmax(worst, std::fabs(p[(size_t)r * 128 + d] - pool_ref[r][d]));
        if (pos % 4 == 3) for (int d = 0; d < 128; ++d) worst = std::fmax(worst, std::fabs(p[(size_t)(pos / 4 + 1) * 128 + d] - pool_ref[pos / 4 + 1][d]));  // spare row
    }
    std::printf("  append max_abs_err=%.3e\n", worst);
    CHECK(worst < 5e-4);
    CHECK(dpool.canary_ok()); CHECK(dtail.canary_ok()); CHECK(ddead.canary_ok()); CHECK(dkey.canary_ok());
    // pooled rows are RMS-normalised and the rotation preserves the norm: mean square over 128 dims is ~1
    { auto p = dpool.get(); bool ok = true;
      for (int r = 0; r < max_cells / 4; ++r) { double ss = 0; for (int d = 0; d < 128; ++d) ss += (double)p[(size_t)r * 128 + d] * p[(size_t)r * 128 + d]; if (std::fabs(ss / 128.0 - 1.0) > 1e-3) ok = false; }
      CHECK(ok); }
    // dims 64..127 are not rotated: they equal the RMS-normalised mean directly (block 0 at pos 3)
    {
        std::vector<std::vector<double>> k4;
        for (int i = 0; i < 4; ++i) k4.push_back(std::vector<double>(keys[i].begin(), keys[i].end()));
        std::vector<double> rot; ref_pool(k4, pos_base, freq, eps, rot);
        double ss = 0, mean[128]; for (int d = 0; d < 128; ++d) { double s = 0; for (int i = 0; i < 4; ++i) s += k4[i][d]; mean[d] = s / 4; ss += mean[d] * mean[d]; }
        const double inv = 1.0 / std::sqrt(ss / 128.0 + (double)eps);
        auto p = dpool.get(); bool ok = true;
        for (int d = 64; d < 128; ++d) if (std::fabs(p[d] - mean[d] * inv) > 1e-4) ok = false;
        CHECK(ok);
    }
    // position guard: pos >= max_cells and bad parameters are rejected without touching state
    {
        q::AppendArgs a{dkey.ptr(), dtail.ptr(), dpool.ptr(), ddead.ptr(), max_cells, max_cells, rows, pos_base, freq, eps};
        CHECK(q::pool_append(a, nullptr) == q::kInvalid);
        a.pos = 5; a.pos_base = 6; CHECK(q::pool_append(a, nullptr) == q::kInvalid);                              // base not a multiple of 4
        a.pos_base = 2147483644ll; CHECK(q::pool_append(a, nullptr) == q::kInvalid);                          // int32 overflow
        a.pos_base = 64; a.rows = rows - 1; CHECK(q::pool_append(a, nullptr) == q::kInvalid);                    // wrong capacity
        a.rows = rows; a.tail = dpool.ptr(); CHECK(q::pool_append(a, nullptr) == q::kInvalid);                      // aliased buffers
        a.tail = dtail.ptr(); a.key = dpool.ptr() + 4; CHECK(q::pool_append(a, nullptr) == q::kInvalid);            // key overlaps pooled
    }
    // tiny capacity: max_cells < 4 -> a single pooled row, position 0 only touches row 0
    {
        const int mc = 3, rws = mc / 4 + 1;
        Dev<float> k(128), t(3 * 128), p((size_t)rws * 128), d(128);
        auto kk = randv(128); CUOK(cudaMemcpy(k.ptr(), kk.data(), 512, cudaMemcpyHostToDevice));
        q::AppendArgs a{k.ptr(), t.ptr(), p.ptr(), d.ptr(), 0, mc, rws, 0, freq, eps};
        CHECK(q::pool_append(a, nullptr) == q::kOk); sync();
        CHECK(p.canary_ok());
    }
}

int main() {
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess || n < 1) { std::printf("no GPU device\n"); return 2; }
    CUOK(cudaSetDevice(0));
    cudaDeviceProp prop; CUOK(cudaGetDeviceProperties(&prop, 0));
    std::printf("device 0: %s (warpSize %d)\n", prop.name, prop.warpSize);
    test_decode();
    test_block_score();
    test_select();
    test_append();
    std::printf("checks=%ld failures=%ld\n", g_checks, g_fail);
    if (g_fail) { std::printf("UNIT7 GPU TESTS FAILED\n"); return 1; }
    std::printf("UNIT7 GPU TESTS PASSED\n");
    return 0;
}
