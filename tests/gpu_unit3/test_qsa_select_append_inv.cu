// test_qsa_select_append_inv.cu - GPU invariant tests for the QSA block-select and indexer-append kernels, stated
// directly from the theory (QsaBlockSelect / NativeQsaIndexerAppend):
//  select: block weights sum to n_kv, exactly `width` cells chosen when n_kv > width, ids ascending/unique/in range,
//          every cell above the threshold block key is chosen, none below, ties broken lowest cell index first,
//          the tail block (weight n_kv - 4*n_bid) expands only to its real cells, identity branch when n_kv <= width.
//  append: slot = pos%4; slots 0..2 touch only tail[slot]; pos 0 initialises row 0 + spare; slot 3 writes row b,
//          copies the spare to row b+1, rotates at pos_base + 4b with pairs (d, d+32); nothing outside the
//          pooled rows is written; invalid parameters are rejected with nothing launched.
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>
#include "baloon/ilu_qsa_kernels.hpp"

using namespace baloon;
namespace q = baloon::ilu_qsa;

static long g_checks = 0, g_fail = 0;
#define CHECK(c, ...)                                                                   \
    do {                                                                                \
        ++g_checks;                                                                     \
        if (!(c)) {                                                                     \
            ++g_fail;                                                                   \
            if (g_fail <= 40) { std::printf("FAIL %s:%d: %s | ", __FILE__, __LINE__, #c); std::printf(__VA_ARGS__); std::printf("\n"); } \
        }                                                                               \
    } while (0)
#define CUOK(x)                                                                         \
    do {                                                                                \
        cudaError_t e_ = (x);                                                           \
        if (e_ != cudaSuccess) { std::printf("CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); std::exit(2); } \
    } while (0)

static std::mt19937 rng(777);

// ---------------------------------------------------------------- host-side theory sweeps
static void test_host_theory() {
    for (uint32_t n = 1; n <= 5000; ++n) {
        const uint32_t nbid = n / 4;
        uint64_t sum = 0;
        for (uint32_t b = 0; b <= nbid; ++b) sum += qsa_select::weight(n, nbid, b);
        CHECK(sum == n, "sumW n=%u", n);
        CHECK((qsa_select::weight(n, nbid, nbid) > 0) == (n % 4 != 0), "tail n=%u", n);
        for (uint32_t j = 0; j < n; j += 7) CHECK(qsa_select::cell_id(j / 4, j % 4) == j, "roundtrip");
    }
    for (int i = 0; i < 2000; ++i) {
        const uint32_t d3 = rng() & 255, d2 = rng() & 255, d1 = rng() & 255, d0 = rng() & 255;
        const uint32_t k = qsa_select::assemble(d3, d2, d1, d0);
        CHECK(qsa_select::digit(k, 0) == d3 && qsa_select::digit(k, 1) == d2 && qsa_select::digit(k, 2) == d1 &&
                  qsa_select::digit(k, 3) == d0, "digits");
    }
    for (uint32_t pos = 0; pos < 4096; ++pos) {
        CHECK(native_qsa_indexer::completes(pos) == ((pos + 1) % 4 == 0), "slot3 iff next block");
        if (pos != 0 && native_qsa_indexer::completes(pos)) CHECK(4 * native_qsa_indexer::block_of(pos) + 3 == pos, "complete idx");
        CHECK(native_qsa_indexer::rope_pos(1024, pos) <= 1024 + (uint64_t)pos, "rope le");
        CHECK(native_qsa_indexer::rope_pos(1024, pos) % 4 == 0, "rope aligned");
    }
    for (uint32_t d = 0; d < 64; ++d) {
        CHECK(native_qsa_indexer::rot_pair_index(d) < 32, "pair range");
        CHECK(native_qsa_indexer::rot_partner(native_qsa_indexer::rot_partner(d)) == d, "partner involution");
    }
}

// ---------------------------------------------------------------- block select on the device
struct SelDev {
    float* sc = nullptr; int32_t* ids = nullptr; int32_t* nout = nullptr;
    SelDev(int nb, int width) {
        CUOK(cudaMalloc(&sc, (size_t)std::max(nb, 1) * 4));
        CUOK(cudaMalloc(&ids, (size_t)std::max(width, 1) * 4));
        CUOK(cudaMalloc(&nout, 4));
    }
    ~SelDev() { cudaFree(sc); cudaFree(ids); cudaFree(nout); }
};

static void run_select(const std::vector<float>& scores, int n_kv, int width, const char* name) {
    const int nbid = n_kv / 4, tail = n_kv - 4 * nbid, nb = nbid + (tail > 0);
    CHECK((int)scores.size() == nb, "%s: score count", name);
    SelDev d(nb, width);
    CUOK(cudaMemcpy(d.sc, scores.data(), (size_t)nb * 4, cudaMemcpyHostToDevice));
    std::vector<int32_t> sentinel(width, -7);
    CUOK(cudaMemcpy(d.ids, sentinel.data(), (size_t)width * 4, cudaMemcpyHostToDevice));
    CUOK(cudaMemset(d.nout, 0xFF, 4));
    const int rc = q::block_select(d.sc, n_kv, width, d.ids, d.nout, nullptr);
    CUOK(cudaDeviceSynchronize());
    CHECK(rc == q::kOk, "%s: rc=%d", name, rc);
    int32_t n_out = -1;
    std::vector<int32_t> ids(width);
    CUOK(cudaMemcpy(&n_out, d.nout, 4, cudaMemcpyDeviceToHost));
    CUOK(cudaMemcpy(ids.data(), d.ids, (size_t)width * 4, cudaMemcpyDeviceToHost));

    if (n_kv <= width) {  // identity branch
        CHECK(n_out == n_kv, "%s: identity n_out=%d", name, n_out);
        for (int i = 0; i < width; ++i) CHECK(ids[i] == (i < n_kv ? i : -1), "%s: identity id[%d]=%d", name, i, ids[i]);
        return;
    }
    CHECK(n_out == width, "%s: n_out=%d want %d", name, n_out, width);
    if (n_out != width) return;
    // independent threshold: sort cells by block key descending, the width-th cell's key is the threshold
    std::vector<uint32_t> key(nb);
    for (int b = 0; b < nb; ++b) key[b] = q::order_key(scores[b]);
    std::vector<uint32_t> cell_keys;
    cell_keys.reserve(n_kv);
    for (int j = 0; j < n_kv; ++j) cell_keys.push_back(key[j / 4]);
    std::vector<uint32_t> srt = cell_keys;
    std::sort(srt.begin(), srt.end(), std::greater<uint32_t>());
    const uint32_t thr = srt[width - 1];
    std::vector<char> chosen(n_kv, 0);
    int prev = -1;
    bool ok_range = true, ok_asc = true;
    for (int i = 0; i < width; ++i) {
        if (ids[i] < 0 || ids[i] >= n_kv) { ok_range = false; continue; }
        if (ids[i] <= prev) ok_asc = false;
        prev = ids[i];
        chosen[ids[i]] = 1;
    }
    CHECK(ok_range, "%s: ids in [0,n_kv)", name);
    CHECK(ok_asc, "%s: ids strictly ascending", name);
    if (!ok_range) return;
    long tie_total = 0, tie_chosen = 0;
    bool above_all = true, below_none = true, tie_prefix = true, seen_unchosen_tie = false;
    for (int j = 0; j < n_kv; ++j) {
        const uint32_t k = cell_keys[j];
        if (k > thr && !chosen[j]) above_all = false;
        if (k < thr && chosen[j]) below_none = false;
        if (k == thr) {
            ++tie_total;
            if (chosen[j]) { ++tie_chosen; if (seen_unchosen_tie) tie_prefix = false; }
            else seen_unchosen_tie = true;
        }
    }
    CHECK(above_all, "%s: every cell above threshold chosen", name);
    CHECK(below_none, "%s: no cell below threshold chosen", name);
    CHECK(tie_prefix, "%s: ties taken lowest index first", name);
    CHECK(tie_chosen >= 1 && tie_chosen <= tie_total, "%s: tie budget", name);
    // block integrity: a chosen cell never belongs beyond the real cells of the tail block
    CHECK(ids[width - 1] < n_kv, "%s: last id", name);
    // host reference agrees
    const std::vector<uint32_t> ref = qsa_select::select_cells(key, (uint32_t)n_kv, (uint32_t)nbid, (uint32_t)width);
    bool same = ref.size() == (size_t)width;
    for (size_t i = 0; same && i < ref.size(); ++i) same = (int32_t)ref[i] == ids[i];
    CHECK(same, "%s: matches host radix reference", name);
}

static std::vector<float> rand_scores(int nb, int mode) {
    std::vector<float> s(nb);
    std::normal_distribution<float> nd(0.f, 5.f);
    for (int b = 0; b < nb; ++b) {
        if (mode == 0) s[b] = nd(rng);
        else if (mode == 1) s[b] = (float)(int)(rng() % 3) - 1.f;      // heavy ties incl. negatives
        else if (mode == 2) s[b] = 0.f;                               // all equal
        else if (mode == 3) s[b] = (b % 17 == 0) ? 3.f : (b % 5 == 0 ? -0.f : 0.f);  // signed zeros
        else s[b] = std::floor(nd(rng)) * 0.5f;                      // many ties, wide range
    }
    return s;
}

static void test_select() {
    const int W = q::kTopKWidth;
    const int sizes[] = {1, 3, 4, 5, 2050, 2051, 2052, 2053, 2054, 2055, 2056, 4099, 4100, 8191, 10000, 16385};
    char nm[96];
    for (int n_kv : sizes) {
        const int nbid = n_kv / 4, nb = nbid + (n_kv % 4 != 0);
        for (int mode = 0; mode < 5; ++mode) {
            std::snprintf(nm, sizeof nm, "select n_kv=%d mode=%d", n_kv, mode);
            run_select(rand_scores(nb, mode), n_kv, W, nm);
        }
    }
    // tail block with the dominant score: all its real cells are chosen, none beyond n_kv
    for (int tail : {1, 2, 3}) {
        const int n_kv = 6000 + tail, nbid = n_kv / 4, nb = nbid + 1;
        std::vector<float> s = rand_scores(nb, 0);
        s[nbid] = 1.0e9f;
        std::snprintf(nm, sizeof nm, "select tail-dominant tail=%d", tail);
        run_select(s, n_kv, W, nm);
    }
    // narrower widths exercise the same weighted descent
    for (int w : {1, 4, 5, 7, 64, 255, 256, 1000}) {
        const int n_kv = 3001, nb = n_kv / 4 + 1;
        for (int mode : {0, 1, 2, 4}) {
            std::snprintf(nm, sizeof nm, "select width=%d mode=%d", w, mode);
            run_select(rand_scores(nb, mode), n_kv, w, nm);
        }
    }
    // rejects
    SelDev d(8, 8);
    CHECK(q::block_select(d.sc, 0, 8, d.ids, d.nout, nullptr) == q::kInvalid, "n_kv=0 rejected");
    CHECK(q::block_select(d.sc, 8, 0, d.ids, d.nout, nullptr) == q::kInvalid, "width=0 rejected");
    CHECK(q::block_select(nullptr, 8, 8, d.ids, d.nout, nullptr) == q::kInvalid, "null scores rejected");
}

// ---------------------------------------------------------------- pooled append on the device
static const float kGuard = 31337.f;

static void ref_block(const float* t0, const float* t1, const float* t2, const float* key, long long rpos,
                      float freq, float eps, std::vector<double>& out) {
    std::vector<double> m(128), y(128);
    double ss = 0;
    for (int d = 0; d < 128; ++d) { m[d] = ((double)t0[d] + t1[d] + t2[d] + key[d]) * 0.25; ss += m[d] * m[d]; }
    const double inv = 1.0 / std::sqrt(ss / 128.0 + (double)eps);
    for (int d = 0; d < 128; ++d) y[d] = m[d] * inv;
    out = y;
    for (int d = 0; d < 64; ++d) {
        const int pair = d % 32;
        const double ang = (double)rpos * std::pow((double)freq, (double)pair);
        const double lo = y[pair], hi = y[pair + 32];
        out[d] = d < 32 ? lo * std::cos(ang) - hi * std::sin(ang) : hi * std::cos(ang) + lo * std::sin(ang);
    }
}

static void run_append_sequence(int max_cells, long long pos_base, double tol, const char* name) {
    const int rows = max_cells / 4 + 1;
    const float freq = std::pow(10000.f, -1.f / 32.f), eps = 1e-6f;
    const size_t pooled_n = (size_t)(rows + 2) * 128;  // two guard rows after the buffer
    float *dkey, *dtail, *dpooled, *ddead;
    CUOK(cudaMalloc(&dkey, 512));
    CUOK(cudaMalloc(&dtail, 3 * 512));
    CUOK(cudaMalloc(&dpooled, pooled_n * 4));
    CUOK(cudaMalloc(&ddead, 512));
    std::vector<float> h_tail(384, -5.f), h_pooled(pooled_n, kGuard), h_dead(128, -6.f);
    CUOK(cudaMemcpy(dtail, h_tail.data(), 384 * 4, cudaMemcpyHostToDevice));
    CUOK(cudaMemcpy(dpooled, h_pooled.data(), pooled_n * 4, cudaMemcpyHostToDevice));
    CUOK(cudaMemcpy(ddead, h_dead.data(), 512, cudaMemcpyHostToDevice));
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> raw[4];
    for (int pos = 0; pos < max_cells; ++pos) {
        std::vector<float> key(128);
        for (auto& x : key) x = nd(rng) + 0.3f;
        CUOK(cudaMemcpy(dkey, key.data(), 512, cudaMemcpyHostToDevice));
        const std::vector<float> pb = h_pooled, tb = h_tail, db = h_dead;
        q::AppendArgs a{dkey, dtail, dpooled, ddead, pos, max_cells, rows, pos_base, freq, eps};
        const int rc = q::pool_append(a, nullptr);
        CUOK(cudaDeviceSynchronize());
        CHECK(rc == q::kOk, "%s pos=%d rc=%d", name, pos, rc);
        CUOK(cudaMemcpy(h_tail.data(), dtail, 384 * 4, cudaMemcpyDeviceToHost));
        CUOK(cudaMemcpy(h_pooled.data(), dpooled, pooled_n * 4, cudaMemcpyDeviceToHost));
        CUOK(cudaMemcpy(h_dead.data(), ddead, 512, cudaMemcpyDeviceToHost));
        const int slot = pos & 3, b = pos >> 2;
        // guard rows never written
        bool guard = true;
        for (size_t i = (size_t)rows * 128; i < pooled_n; ++i) guard = guard && h_pooled[i] == kGuard;
        CHECK(guard, "%s pos=%d guard rows intact", name, pos);
        // tail: only tail[slot] changes, and only for slot < 3
        for (int s = 0; s < 3; ++s)
            for (int d = 0; d < 128; ++d) {
                const float want = (s == slot && slot < 3) ? key[d] : tb[s * 128 + d];
                if (h_tail[s * 128 + d] != want) { CHECK(false, "%s pos=%d tail[%d][%d]", name, pos, s, d); s = 3; break; }
            }
        if (slot < 3) raw[slot] = key;
        if (pos != 0 && slot != 3) {  // early return: pooled and spare untouched
            CHECK(h_pooled == pb, "%s pos=%d pooled untouched", name, pos);
            CHECK(h_dead == db, "%s pos=%d spare untouched", name, pos);
            continue;
        }
        std::vector<double> ref;
        if (pos == 0) {
            ref_block(key.data(), key.data(), key.data(), key.data(), 0, freq, eps, ref);  // four copies, rope pos 0
            for (int r = 1; r < rows; ++r)
                for (int d = 0; d < 128; ++d) CHECK(h_pooled[r * 128 + d] == pb[r * 128 + d], "%s pos=0 row %d untouched", name, r);
            double e = 0, e2 = 0;
            for (int d = 0; d < 128; ++d) { e = std::max(e, std::fabs((double)h_pooled[d] - ref[d])); e2 = std::max(e2, std::fabs((double)h_dead[d] - h_pooled[d])); }
            CHECK(e < 2e-4, "%s pos=0 row0 err %.3g", name, e);
            CHECK(e2 == 0.0, "%s pos=0 spare == row0 (%.3g)", name, e2);
        } else {
            // before the call the spare row held a prior value; slot 3 copies it to row b+1
            ref_block(raw[0].data(), raw[1].data(), raw[2].data(), key.data(), pos_base + 4ll * b, freq, eps, ref);
            double e = 0, nrm_ref = 0, nrm_got = 0;
            for (int d = 0; d < 128; ++d) {
                e = std::max(e, std::fabs((double)h_pooled[(size_t)b * 128 + d] - ref[d]));
                nrm_ref += ref[d] * ref[d]; nrm_got += (double)h_pooled[(size_t)b * 128 + d] * h_pooled[(size_t)b * 128 + d];
            }
            CHECK(e < tol, "%s pos=%d row %d err %.3g", name, pos, b, e);
            CHECK(std::fabs(nrm_ref - nrm_got) < 1e-3 * nrm_ref, "%s pos=%d rotation preserves norm", name, pos);
            bool spare = true;
            for (int d = 0; d < 128; ++d) spare = spare && h_pooled[(size_t)(b + 1) * 128 + d] == db[d];
            CHECK(spare, "%s pos=%d row %d == spare", name, pos, b + 1);
            CHECK(h_dead == db, "%s pos=%d spare itself unchanged", name, pos);
            for (int r = 0; r < rows; ++r) {
                if (r == b || r == b + 1) continue;
                bool same = true;
                for (int d = 0; d < 128; ++d) same = same && h_pooled[(size_t)r * 128 + d] == pb[(size_t)r * 128 + d];
                CHECK(same, "%s pos=%d row %d untouched", name, pos, r);
            }
        }
    }
    cudaFree(dkey); cudaFree(dtail); cudaFree(dpooled); cudaFree(ddead);
}

static void test_append_rejects() {
    float *dkey, *dtail, *dpooled, *ddead;
    CUOK(cudaMalloc(&dkey, 512)); CUOK(cudaMalloc(&dtail, 3 * 512)); CUOK(cudaMalloc(&dpooled, 40 * 512)); CUOK(cudaMalloc(&ddead, 512));
    CUOK(cudaMemset(dpooled, 0, 40 * 512));
    const float f = 0.75f, eps = 1e-6f;
    const int mc = 64, rows = 17;
    auto rc = [&](q::AppendArgs a) { return q::pool_append(a, nullptr); };
    q::AppendArgs ok{dkey, dtail, dpooled, ddead, 3, mc, rows, 0, f, eps};
    auto mod = [&](auto fn) { q::AppendArgs a = ok; fn(a); return a; };
    CHECK(rc(mod([](q::AppendArgs& a) { a.pos = mc; })) == q::kInvalid, "pos >= max_cells");
    CHECK(rc(mod([](q::AppendArgs& a) { a.pos = -1; })) == q::kInvalid, "pos < 0");
    CHECK(rc(mod([](q::AppendArgs& a) { a.pos_base = 2; })) == q::kInvalid, "unaligned pos_base");
    CHECK(rc(mod([](q::AppendArgs& a) { a.pos_base = 2147483647ll - 60; })) == q::kInvalid, "int32 overflow");
    CHECK(rc(mod([](q::AppendArgs& a) { a.rows = rows + 1; })) == q::kInvalid, "wrong row count");
    CHECK(rc(mod([](q::AppendArgs& a) { a.rows = rows - 1; })) == q::kInvalid, "too few rows");
    CHECK(rc(mod([](q::AppendArgs& a) { a.key = nullptr; })) == q::kInvalid, "null key");
    CHECK(rc(mod([&](q::AppendArgs& a) { a.key = dtail; })) == q::kInvalid, "key aliases tail");
    CHECK(rc(mod([&](q::AppendArgs& a) { a.dead = dpooled; })) == q::kInvalid, "spare aliases pooled");
    CHECK(rc(mod([](q::AppendArgs& a) { a.freq_step = 0.f; })) == q::kInvalid, "freq_step <= 0");
    CHECK(rc(mod([](q::AppendArgs& a) { a.eps = -1.f; })) == q::kInvalid, "eps < 0");
    CHECK(rc(ok) == q::kOk, "valid call accepted");
    CUOK(cudaDeviceSynchronize());
    cudaFree(dkey); cudaFree(dtail); cudaFree(dpooled); cudaFree(ddead);
}

int main() {
    int n = 0;
    CUOK(cudaGetDeviceCount(&n));
    if (n < 1) { std::printf("no CUDA device\n"); return 2; }
    CUOK(cudaSetDevice(0));
    cudaDeviceProp p;
    CUOK(cudaGetDeviceProperties(&p, 0));
    std::printf("device 0: %s warpSize=%d\n", p.name, p.warpSize);
    test_host_theory();
    test_select();
    run_append_sequence(64, 0, 2e-4, "append base=0");
    run_append_sequence(256, 1024, 2e-3, "append base=1024");
    run_append_sequence(3, 0, 2e-4, "append max_cells=3");  // last block never completes
    test_append_rejects();
    std::printf("unit3 gpu invariants: checks=%ld failures=%ld\n", g_checks, g_fail);
    return g_fail == 0 ? 0 : 1;
}
