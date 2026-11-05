// test_qsa_cache_step_gpu.cu - GPU invariants of unit u316bd6b9ec (QsaCacheStep / QsaSelect / NativeQsaScore).
// Device results are compared with literal expectations and with the host model in qsa_cache_step.hpp; the existing
// QSA kernels (step fill, paged fp16 KV, indexer scoring, one-block top-k, tensor-tile scoring) are driven end to end.
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include "baloon/ilu_qsa_kernels.hpp"
#include "baloon/qsa_cache_step.hpp"
#include "baloon/qsa_geometry_dev.hpp"
#include "baloon/qsa_step_kernels.hpp"

using namespace baloon;
static long g_checks = 0, g_fail = 0;
#define CHECK(c) do { ++g_checks; if (!(c)) { ++g_fail; if (g_fail < 60) std::printf("FAIL %s:%d  %s\n", __FILE__, __LINE__, #c); } } while (0)
#define CUOK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { std::printf("CUDA error '%s' at line %d\n", cudaGetErrorString(e_), __LINE__); std::exit(2); } } while (0)

template <class T> static T* dalloc(size_t n) { T* p = nullptr; CUOK(cudaMalloc((void**)&p, n * sizeof(T))); return p; }
template <class T> static T* dup(const std::vector<T>& v) {
    T* p = dalloc<T>(v.empty() ? 1 : v.size());
    if (!v.empty()) CUOK(cudaMemcpy(p, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice));
    return p;
}
template <class T> static std::vector<T> fetch(const T* p, size_t n) {
    std::vector<T> v(n);
    CUOK(cudaMemcpy(v.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost));
    return v;
}
struct Rng {
    uint64_t s; explicit Rng(uint64_t x) : s(x * 0x9E3779B97F4A7C15ull + 99) {}
    uint64_t next() { s ^= s << 13; s ^= s >> 7; s ^= s << 17; return s; }
    uint32_t u32() { return (uint32_t)(next() >> 16); }
    uint32_t below(uint32_t n) { return u32() % n; }
    float gauss() { double u = ((next() >> 11) + 1) / 9007199254740993.0, v = (next() >> 11) / 9007199254740992.0; return (float)(std::sqrt(-2 * std::log(u)) * std::cos(6.283185307179586 * v)); }
};

static void test_step_device() {
    std::vector<int32_t> pos;
    for (int p = 0; p < 5000; ++p) pos.push_back(p);
    for (int p : {8191, 16383, 32766, 32767}) pos.push_back(p);
    int32_t* d_pos = dup(pos);
    int32_t* d_out = dalloc<int32_t>(pos.size() * 4);
    CHECK(qsa_geom_dev::step_eval(d_pos, (int)pos.size(), 2048, 4, d_out, nullptr) == 0);
    CUOK(cudaDeviceSynchronize());
    const auto out = fetch(d_out, pos.size() * 4);
    for (size_t i = 0; i < pos.size(); ++i) {
        const auto h = qsa_cs::step_fill_host(pos[i]);
        CHECK(out[4 * i] == h.pos && out[4 * i + 1] == h.n_kv && out[4 * i + 2] == h.n_bid && out[4 * i + 3] == h.width);
        CHECK(out[4 * i + 1] == pos[i] + 1 && out[4 * i + 2] == (pos[i] + 1) / 4);
        CHECK(out[4 * i + 3] == std::min(pos[i] + 1, 2051));
    }
    // literals
    auto at = [&](int p, int f) { return out[4 * (size_t)(std::find(pos.begin(), pos.end(), p) - pos.begin()) + f]; };
    CHECK(at(0, 1) == 1 && at(0, 2) == 0 && at(0, 3) == 1);
    CHECK(at(3, 2) == 1 && at(2, 2) == 0 && at(2047, 3) == 2048 && at(2049, 3) == 2050 && at(2050, 3) == 2051 && at(2051, 3) == 2051);
    CHECK(at(32767, 1) == 32768 && at(32767, 2) == 8192 && at(32767, 3) == 2051);
    CHECK(qsa_geom_dev::step_eval(nullptr, 1, 2048, 4, d_out, nullptr) == -1);
    CHECK(qsa_geom_dev::step_eval(d_pos, 0, 2048, 4, d_out, nullptr) == -1);
    // the production step kernel (derive entries 1..3 from entry 0)
    int32_t* d_step = dalloc<int32_t>(4);
    for (int p : {0, 1, 2, 3, 4, 2046, 2047, 2048, 2049, 2050, 4000, 32767}) {
        CHECK(qsa_step::step_upload(d_step, p, 2048, 4, nullptr) == 0);
        const auto s = fetch(d_step, 4);
        const auto h = qsa_cs::step_fill_host(p);
        CHECK(s[0] == h.pos && s[1] == h.n_kv && s[2] == h.n_bid && s[3] == h.width);
        CHECK(sizeof(qsa_step::Step) == qsa_cs::qsa_step_bytes());
    }
    cudaFree(d_pos); cudaFree(d_out); cudaFree(d_step);
}

static void test_kv_rows_device() {
    const int H = 3, PS = 16, P = 7;
    std::vector<int32_t> table = {5, 2, 6, 0, 3, 1, 4};
    std::vector<int32_t> cells, heads;
    for (int c = 0; c < P * PS; ++c) for (int h = 0; h < H; ++h) { cells.push_back(c); heads.push_back(h); }
    cells.push_back(P * PS); heads.push_back(0);      // outside the table
    cells.push_back(3); heads.push_back(H);           // head out of range
    int32_t* d_c = dup(cells); int32_t* d_h = dup(heads); int32_t* d_t = dup(table);
    int64_t* d_r = dalloc<int64_t>(cells.size());
    CHECK(qsa_geom_dev::kv_row_eval(d_c, d_h, (int)cells.size(), d_t, P, H, PS, d_r, nullptr) == 0);
    CUOK(cudaDeviceSynchronize());
    const auto rows = fetch(d_r, cells.size());
    std::vector<int> hit((size_t)P * H * PS, 0);
    const size_t n_real = (size_t)P * PS * H;
    for (size_t i = 0; i < n_real; ++i) {
        CHECK(rows[i] >= 0 && rows[i] < (int64_t)hit.size());
        CHECK((uint64_t)rows[i] == qsa_cs::kv_row_of_cell(table.data(), cells[i], H, heads[i], PS));
        if (rows[i] >= 0 && rows[i] < (int64_t)hit.size()) hit[rows[i]]++;
    }
    for (int c : hit) CHECK(c == 1);                  // injective and onto the pool
    CHECK(rows[n_real] == -1 && rows[n_real + 1] == -1);
    // literal: cell 17 (page index 1 -> physical page 2), head 1, slot 1 -> ((2*3)+1)*16+1 = 113
    CHECK(rows[(size_t)17 * H + 1] == 113);
    CHECK(qsa_geom_dev::kv_row_eval(d_c, d_h, 4, d_t, 0, H, PS, d_r, nullptr) == -1);
    cudaFree(d_c); cudaFree(d_h); cudaFree(d_t); cudaFree(d_r);
}

static void test_cell_block_device() {
    for (int n : {1, 2, 3, 4, 5, 8, 9, 2051, 2052, 4097}) {
        std::vector<int32_t> cells(n);
        for (int j = 0; j < n; ++j) cells[j] = j;
        int32_t* d_c = dup(cells); int32_t* d_b = dalloc<int32_t>(n); int32_t* d_t = dalloc<int32_t>(n);
        CHECK(qsa_geom_dev::cell_block_eval(d_c, n, n, 4, d_b, d_t, nullptr) == 0);
        CUOK(cudaDeviceSynchronize());
        const auto b = fetch(d_b, n), t = fetch(d_t, n);
        int tails = 0;
        for (int j = 0; j < n; ++j) {
            CHECK(b[j] == j / 4 && b[j] == qsa_cs::cell_block(j, n));
            CHECK((t[j] == 1) == qsa_cs::cell_in_tail(j, n) && (t[j] == 1) == (j / 4 == n / 4));
            tails += t[j];
        }
        CHECK(tails == n % 4);
        cudaFree(d_c); cudaFree(d_b); cudaFree(d_t);
    }
}

static void test_order_key_device() {
    std::vector<uint32_t> bits = {0u, 0x80000000u, 1u, 0x80000001u, 0x3F800000u, 0xBF800000u, 0x7F800000u, 0xFF800000u, 0x7F7FFFFFu, 0xFF7FFFFFu};
    Rng r(5);
    for (int i = 0; i < 20000; ++i) { uint32_t b = (r.u32() << 16) ^ r.u32(); if (((b >> 23) & 255u) != 255u) bits.push_back(b); }
    uint32_t* d_b = dup(bits); uint32_t* d_k = dalloc<uint32_t>(bits.size());
    CHECK(qsa_geom_dev::order_key_eval(d_b, (int)bits.size(), d_k, nullptr) == 0);
    CUOK(cudaDeviceSynchronize());
    const auto keys = fetch(d_k, bits.size());
    for (size_t i = 0; i < bits.size(); ++i) CHECK(keys[i] == qsa_cs::order_key_bits(qsa_cs::neg_zero_fold(bits[i])));
    CHECK(keys[0] == 0x80000000u && keys[1] == 0x80000000u);          // -0.0 ties +0.0
    CHECK(keys[2] == 0x80000001u && keys[3] == 0x7FFFFFFEu);
    for (size_t i = 0; i < 1500; ++i) for (size_t j = 0; j < 1500; ++j) {
        float a, b; std::memcpy(&a, &bits[i], 4); std::memcpy(&b, &bits[j], 4);
        if (a < b) CHECK(keys[i] < keys[j]);
        if (a == b) CHECK(keys[i] == keys[j]);
    }
    cudaFree(d_b); cudaFree(d_k);
}

static void test_kv_roundtrip() {
    const int H = 3, D = 16, PS = 4, P = 6;
    const int64_t pool_rows = (int64_t)P * H * PS;
    std::vector<int32_t> table = {4, 1, 5, 0, 3, 2};
    uint16_t* d_pool = dalloc<uint16_t>((size_t)pool_rows * D);
    CUOK(cudaMemset(d_pool, 0, (size_t)pool_rows * D * 2));
    int32_t* d_t = dup(table);
    float* d_src = dalloc<float>((size_t)H * D);
    const int ncell = P * PS;
    auto val = [&](int cell, int h, int d) { return (float)(((cell * 5 + h * 3 + d) % 97) + 1) * 0.5f; };   // exact in fp16
    for (int c = 0; c < ncell; ++c) {
        std::vector<float> src((size_t)H * D);
        for (int h = 0; h < H; ++h) for (int d = 0; d < D; ++d) src[(size_t)h * D + d] = val(c, h, d);
        CUOK(cudaMemcpy(d_src, src.data(), src.size() * 4, cudaMemcpyHostToDevice));
        CHECK(qsa_step::kv_append_step(d_src, d_pool, d_t, P, c, H, D, PS, pool_rows, nullptr) == 0);
    }
    CUOK(cudaDeviceSynchronize());
    CHECK(qsa_step::kv_append_step(d_src, d_pool, d_t, P, ncell, H, D, PS, pool_rows, nullptr) == -1);   // past the table
    CHECK(qsa_step::kv_append_step(nullptr, d_pool, d_t, P, 0, H, D, PS, pool_rows, nullptr) == -1);
    std::vector<int32_t> ids;
    for (int c = 0; c < ncell; ++c) ids.push_back(c);
    ids.push_back(-1); ids.push_back(ncell);           // masked / out of table -> zeros
    int32_t* d_ids = dup(ids);
    float* d_out = dalloc<float>(ids.size() * H * D);
    CHECK(qsa_step::kv_gather_step(d_pool, d_t, P, d_ids, (int)ids.size(), H, D, PS, pool_rows, d_out, nullptr) == 0);
    CUOK(cudaDeviceSynchronize());
    const auto out = fetch(d_out, ids.size() * H * D);
    for (int c = 0; c < ncell; ++c) for (int h = 0; h < H; ++h) for (int d = 0; d < D; ++d)
        CHECK(out[((size_t)c * H + h) * D + d] == val(c, h, d));       // no row was overwritten by another (injective)
    for (size_t k = 0; k < (size_t)2 * H * D; ++k) CHECK(out[(size_t)ncell * H * D + k] == 0.f);
    // the physical rows are exactly the model rows
    const auto pool = fetch(d_pool, (size_t)pool_rows * D);
    for (int c = 0; c < ncell; ++c) for (int h = 0; h < H; ++h) {
        const uint64_t row = qsa_cs::kv_row_of_cell(table.data(), c, H, h, PS);
        CHECK(row < (uint64_t)pool_rows);
        CHECK(pool[row * D + 0] != 0 || val(c, h, 0) == 0.f);
    }
    cudaFree(d_pool); cudaFree(d_t); cudaFree(d_src); cudaFree(d_ids); cudaFree(d_out);
}

static void test_topk() {
    Rng r(21);
    for (int n : {1, 5, 100, 2051, 2052, 2053, 3000, 8000}) {
        for (int alpha : {3, 40, 100000}) {
            std::vector<float> sc(n);
            for (auto& s : sc) s = (float)((int)r.below(alpha)) - (float)alpha * 0.25f;   // negatives and ties
            // scores reach the selector already folded (s + 0.0f), so -0.0 never appears here
            const int width = qsa_cs::qsa_selection_width(n);
            std::vector<int> idx(n); for (int i = 0; i < n; ++i) idx[i] = i;
            std::stable_sort(idx.begin(), idx.end(), [&](int a, int b) { return qsa_cs::order_key_f32(sc[a]) > qsa_cs::order_key_f32(sc[b]); });
            std::vector<int> ref(idx.begin(), idx.begin() + width); std::sort(ref.begin(), ref.end());
            float* d_s = dup(sc); int32_t* d_ids = dalloc<int32_t>(width); int32_t* d_n = dalloc<int32_t>(1);
            CHECK(qsa_step::topk_512_step(d_s, n, width, d_ids, d_n, nullptr) == 0);
            CUOK(cudaDeviceSynchronize());
            const auto got = fetch(d_ids, width); const int cnt = fetch(d_n, 1)[0];
            CHECK(cnt == width);
            bool same = cnt == width;
            for (int i = 0; i < width && same; ++i) same = (got[i] == ref[i]);
            CHECK(same);
            if (n <= width) for (int i = 0; i < n; ++i) CHECK(got[i] == i);   // identity below the bound
            cudaFree(d_s); cudaFree(d_ids); cudaFree(d_n);
        }
    }
    float* d_s = dalloc<float>(4); int32_t* d_ids = dalloc<int32_t>(4); int32_t* d_n = dalloc<int32_t>(1);
    CHECK(qsa_step::topk_512_step(d_s, 4, 2052, d_ids, d_n, nullptr) == -1);   // width above 2051 rejected
    CHECK(qsa_step::topk_512_step(nullptr, 4, 2, d_ids, d_n, nullptr) == -1);
    cudaFree(d_s); cudaFree(d_ids); cudaFree(d_n);
}

static void test_index_and_score() {
    Rng r(33);
    const int max_cells = 4096, rows = max_cells / 4 + 1;
    std::vector<float> pooled((size_t)rows * 128), q(4 * 128), hw = {0.7f, 0.3f, 1.1f, 0.5f};
    for (auto& x : pooled) x = r.gauss();
    for (auto& x : q) x = r.gauss();
    float* d_p = dup(pooled); float* d_q = dup(q); float* d_hw = dup(hw);
    float* d_cs = dalloc<float>(max_cells); float* d_ms = dalloc<float>(max_cells);
    int32_t* d_unused = nullptr; (void)d_unused;
    for (int n : {1, 2, 3, 4, 5, 6, 7, 8, 33, 127, 128, 129, 130, 131, 132, 1000, 2051, 2053, 4095, 4096}) {
        const int nb = n / 4 + (n % 4 != 0);
        std::vector<double> ref(n);
        for (int b = 0; b < nb; ++b) {
            double s = 0;
            for (int h = 0; h < 4; ++h) { double dot = 0; for (int d = 0; d < 128; ++d) dot += (double)q[h * 128 + d] * pooled[(size_t)b * 128 + d]; s += hw[h] * (dot > 0.0 ? dot : 0.0); }
            for (int c = b * 4; c < std::min(n, b * 4 + 4); ++c) ref[c] = s;
        }
        CUOK(cudaMemset(d_cs, 0, max_cells * 4)); CUOK(cudaMemset(d_ms, 0, max_cells * 4));
        CHECK(qsa_step::qsa_index_step(d_p, d_q, d_hw, n, rows, d_cs, nullptr) == 0);
        CHECK(ilu_qsa::mma_block_score(d_p, d_q, d_hw, nullptr, nullptr, d_ms, n - 1, n, n / 4, qsa_cs::qsa_selection_width(n), max_cells, rows, nullptr) == 0);
        CUOK(cudaDeviceSynchronize());
        const auto cs = fetch(d_cs, n), ms = fetch(d_ms, n);
        for (int c = 0; c < n; ++c) {
            const bool tail = qsa_cs::cell_in_tail(c, n) && qsa_cs::has_tail(n);
            if (tail) { CHECK(cs[c] >= 0.99e9f && ms[c] >= 0.99e9f); }     // +1e9 bias exactly on the incomplete tail
            else {
                CHECK(std::fabs(cs[c] - ref[c]) <= 1e-3 * std::fabs(ref[c]) + 1e-3);
                CHECK(std::fabs(ms[c] - ref[c]) <= 0.02 * std::fabs(ref[c]) + 0.25);   // tf32 tensor tile
                CHECK(cs[c] < 1e6f && ms[c] < 1e6f);
            }
        }
    }
    // the tile kernel only accepts consistent steps
    const int n = 100;
    const int w = qsa_cs::qsa_selection_width(n);
    CHECK(ilu_qsa::mma_block_score(d_p, d_q, d_hw, nullptr, nullptr, d_ms, n - 1, n, n / 4, w, max_cells, rows, nullptr) == 0);
    CHECK(ilu_qsa::mma_block_score(d_p, d_q, d_hw, nullptr, nullptr, d_ms, n, n, n / 4, w, max_cells, rows, nullptr) == ilu_qsa::kInvalid);       // pos != n-1
    CHECK(ilu_qsa::mma_block_score(d_p, d_q, d_hw, nullptr, nullptr, d_ms, n - 1, n, n / 4 + 1, w, max_cells, rows, nullptr) == ilu_qsa::kInvalid);  // n_bid
    CHECK(ilu_qsa::mma_block_score(d_p, d_q, d_hw, nullptr, nullptr, d_ms, n - 1, n, n / 4, w + 1, max_cells, rows, nullptr) == ilu_qsa::kInvalid);  // width
    CHECK(ilu_qsa::mma_block_score(d_p, d_q, d_hw, nullptr, nullptr, d_ms, n - 1, n, n / 4, w, max_cells, rows + 1, nullptr) == ilu_qsa::kInvalid); // max_blocks
    CHECK(ilu_qsa::mma_block_score(d_p, d_q, d_hw, nullptr, nullptr, d_ms, n - 1, n, n / 4, w, n - 1, rows, nullptr) == ilu_qsa::kInvalid);         // n > max_cells
    CHECK(ilu_qsa::mma_block_score(d_p, d_q, d_hw, nullptr, nullptr, d_p + 64, n - 1, n, n / 4, w, max_cells, rows, nullptr) == ilu_qsa::kInvalid);  // output overlaps input
    cudaFree(d_p); cudaFree(d_q); cudaFree(d_hw); cudaFree(d_cs); cudaFree(d_ms);
}

static void test_indexer_append() {
    const int max_cells = 64, dim = 128;
    const auto layout = qsa_cs::indexer_buffers(max_cells);
    std::vector<float> keys((size_t)max_cells * dim);
    Rng r(44);
    for (auto& x : keys) x = r.gauss();
    char* d_arena = dalloc<char>(layout.total);
    CUOK(cudaMemset(d_arena, 0, layout.total));
    float* d_tail = (float*)(d_arena + layout.tail_off); float* d_pooled = (float*)(d_arena + layout.pooled_off);
    float* d_key = dalloc<float>(dim);
    for (int pos = 0; pos < max_cells; ++pos) {
        CUOK(cudaMemcpy(d_key, keys.data() + (size_t)pos * dim, dim * 4, cudaMemcpyHostToDevice));
        CHECK(qsa_step::indexer_key_append(d_key, d_tail, d_pooled, pos, max_cells, qsa_cs::qsa_rms_eps(), nullptr) == 0);
    }
    CUOK(cudaDeviceSynchronize());
    const auto pooled = fetch(d_pooled, (size_t)layout.rows * dim);
    for (int b = 0; b < max_cells / 4; ++b) {              // every completed block = RMS-normalised mean of its 4 keys
        std::vector<double> m(dim); double ss = 0;
        for (int d = 0; d < dim; ++d) { m[d] = 0.25 * ((double)keys[(size_t)(4 * b) * dim + d] + keys[(size_t)(4 * b + 1) * dim + d] + keys[(size_t)(4 * b + 2) * dim + d] + keys[(size_t)(4 * b + 3) * dim + d]); ss += m[d] * m[d]; }
        const double inv = 1.0 / std::sqrt(ss / dim + qsa_cs::qsa_rms_eps());
        for (int d = 0; d < dim; ++d) CHECK(std::fabs(pooled[(size_t)b * dim + d] - m[d] * inv) <= 1e-4 * std::fabs(m[d] * inv) + 1e-5);
    }
    CHECK(qsa_step::indexer_key_append(d_key, d_tail, d_pooled, max_cells, max_cells, 1e-6f, nullptr) == -1);   // pos out of range
    CHECK(qsa_step::indexer_key_append(d_key, d_tail, d_pooled, 0, max_cells, 0.f, nullptr) == -1);
    cudaFree(d_arena); cudaFree(d_key);
}

int main() {
    int ndev = 0;
    CUOK(cudaGetDeviceCount(&ndev));
    if (ndev < 1) { std::printf("no GPU device\n"); return 2; }
    CUOK(cudaSetDevice(0));
    test_step_device();
    test_kv_rows_device();
    test_cell_block_device();
    test_order_key_device();
    test_kv_roundtrip();
    test_topk();
    test_index_and_score();
    test_indexer_append();
    CUOK(cudaDeviceSynchronize());
    std::printf("%ld checks, %ld failures\n", g_checks, g_fail);
    if (g_fail) { std::printf("QSA CACHE STEP GPU TESTS FAILED\n"); return 1; }
    std::printf("QSA CACHE STEP GPU TESTS PASSED\n");
    return 0;
}
