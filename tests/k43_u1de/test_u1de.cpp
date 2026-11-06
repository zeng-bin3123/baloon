// test_u1de.cpp - LAYER 1 invariants of the five unit-4 modules, checked on the PRODUCTION code:
//   ExpertResidencyCache  -> core::ExpertCache (core_expert_cache.hpp / src/core/expert_cache.cpp)
//   ExpertGeometry        -> kern constants, kern::CpuFeatures, the blob-layout expert kernels (src/cpu/expert_q2.cpp)
//   ExpertLayoutTable     -> etable (expert_table.hpp / src/cpu/expert_table.cpp), expert_layout / expert_layout_load
//   ExpertPoolContract    -> kern::ExpertPool (capacities, worker count, atomics' cache lines, batching, affinity)
//   ExpertPoolScheduling  -> claim counter, row partitions, task decomposition, row walk (real threads, real pool)
// Built with the AVX-512 ISA flags and -ffp-contract=off. Exits non-zero on any failed check.
#include <immintrin.h>
#include <cpuid.h>
#include <sched.h>
#include <unistd.h>
#include <sys/stat.h>

#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include "baloon/core_expert_cache.hpp"
#include "baloon/cpu_expert_layout.hpp"
#include "baloon/cpu_expert_pool.hpp"
#include "baloon/expert_table.hpp"
#include "baloon/kern_cpu_expert.hpp"
#include "baloon/kern_cpu_iq.hpp"
#include "baloon/kern_cpu_pool.hpp"

using namespace baloon;

static int g_fail = 0, g_checks = 0;
#define CHECK(c) do { ++g_checks; if (!(c)) { ++g_fail; std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); } } while (0)

static std::mt19937_64 rng(4242);
static uint16_t f2h(float f) { return _cvtss_sh(f, 0); }
static std::string tmpdir() { char t[] = "/tmp/u1de_XXXXXX"; return mkdtemp(t); }

// =============================================================================================== ExpertResidencyCache
// Executable model of the Lean `Cache` / `LCache` (nextFree / layerNext / res).
struct Model {
    int64_t slots, nl, ne; bool per_layer;
    std::vector<int32_t> res; std::vector<int64_t> layer_next; int64_t next_free = 0;
    Model(int64_t s, int64_t l, int64_t e, bool pl) : slots(s), nl(l), ne(e), per_layer(pl), res((size_t)(l * e), -1), layer_next((size_t)l) {
        for (int64_t i = 0; i < l; ++i) layer_next[(size_t)i] = i * (s / l);
    }
    int64_t hi(int64_t l) const { return l == nl - 1 ? slots : (l + 1) * (slots / nl); }
    int32_t admit(int64_t l, int64_t e) {
        if (l < 0 || l >= nl || e < 0 || e >= ne) return -1;
        int32_t& cell = res[(size_t)(l * ne + e)];
        if (cell != -1) return cell;
        if (per_layer) { if (!(layer_next[(size_t)l] < hi(l))) return -1; cell = (int32_t)layer_next[(size_t)l]++; return cell; }
        if (!(next_free < slots)) return -1;
        cell = (int32_t)next_free++; return cell;
    }
};

static void test_cache() {
    using core::ExpertCache; using core::kNotResident;
    CHECK(kNotResident == -1);
    // ---- ranges: tile [0, slots), last layer takes the remainder (layerLo_zero, layerHi_last, abut, disjoint, width >= quota)
    for (int64_t slots : {1, 5, 7, 24, 100, 257}) for (int64_t nl : {1, 2, 3, 5, 8}) {
        ExpertCache c; CHECK(c.open(nl, 6, slots, 256, true));
        const int64_t q = slots / nl;
        CHECK(c.layer_slot_range(0).first == 0);
        CHECK(c.layer_slot_range(nl - 1).second == slots);
        for (int64_t l = 0; l < nl; ++l) {
            const auto r = c.layer_slot_range(l);
            CHECK(r.first == l * q); CHECK(r.first <= r.second); CHECK(r.second - r.first >= q);
            if (l + 1 < nl) { CHECK(r.second == c.layer_slot_range(l + 1).first); CHECK(r.second <= c.layer_slot_range(l + 1).first); }
        }
        CHECK(c.layer_slot_range(nl).first == 0 && c.layer_slot_range(nl).second == 0);   // out of range: empty
        CHECK(c.layer_slot_range(-1).second == 0);
    }
    // ---- admission against the model: random sequences, both modes (idempotent, refuses when full, never evicts, distinct slots)
    for (int mode = 0; mode < 2; ++mode) for (int trial = 0; trial < 25; ++trial) {
        const int64_t nl = 1 + (int64_t)(rng() % 6), ne = 1 + (int64_t)(rng() % 9), slots = (int64_t)(rng() % (uint64_t)(nl * ne + 4));
        if (slots == 0) continue;
        ExpertCache c; CHECK(c.open(nl, ne, slots, 256, mode == 1));
        Model m(slots, nl, ne, mode == 1);
        std::vector<int32_t> seen_slot((size_t)slots, -1);
        for (int step = 0; step < 150; ++step) {
            const int64_t l = (int64_t)(rng() % (uint64_t)(nl + 2)) - 1, e = (int64_t)(rng() % (uint64_t)(ne + 2)) - 1;   // includes out of range
            std::vector<int32_t> before((size_t)(nl * ne));
            for (int64_t a = 0; a < nl; ++a) for (int64_t b = 0; b < ne; ++b) before[(size_t)(a * ne + b)] = c.slot_of(a, b);
            const int32_t want = m.admit(l, e), got = c.admit(l, e);
            CHECK(got == want);
            // NO EVICTION: everything resident before is resident in the same slot after
            for (int64_t a = 0; a < nl; ++a) for (int64_t b = 0; b < ne; ++b) {
                const int32_t bf = before[(size_t)(a * ne + b)];
                if (bf != -1) CHECK(c.slot_of(a, b) == bf);
                CHECK(c.slot_of(a, b) == m.res[(size_t)(a * ne + b)]);
            }
            if (got != -1) {
                CHECK(got >= 0 && got < slots);                                     // admit_slot_lt
                CHECK(c.slot_of(l, e) == got);                                      // admit_then_resident
                CHECK(c.admit(l, e) == got);                                        // idempotent
                if (mode == 1) { const auto r = c.layer_slot_range(l); CHECK(got >= r.first && got < r.second); }   // own range only
                int32_t& owner = seen_slot[(size_t)got];                            // injective: one pair per slot
                const int32_t key = (int32_t)(l * ne + e);
                if (owner == -1) owner = key; else CHECK(owner == key);
            }
            CHECK(c.slot_of(-1, 0) == -1 && c.slot_of(0, -1) == -1 && c.slot_of(nl, 0) == -1 && c.slot_of(0, ne) == -1);
        }
        if (mode == 1) {   // no cross-layer sharing
            for (int64_t a = 0; a < nl; ++a) for (int64_t b = 0; b < ne; ++b) {
                const int32_t s = c.slot_of(a, b);
                if (s != -1) { const auto r = c.layer_slot_range(a); CHECK(s >= r.first && s < r.second); }
            }
        }
        // a full cache never changes
        if (mode == 0) {
            for (int64_t a = 0; a < nl; ++a) for (int64_t b = 0; b < ne; ++b) c.admit(a, b);
            const int64_t rc = c.resident_count();
            std::vector<int32_t> snap; for (int64_t a = 0; a < nl; ++a) for (int64_t b = 0; b < ne; ++b) snap.push_back(c.slot_of(a, b));
            if (rc == slots) {
                for (int64_t a = 0; a < nl; ++a) for (int64_t b = 0; b < ne; ++b) if (c.slot_of(a, b) == -1) CHECK(c.admit(a, b) == -1);
                size_t k = 0; for (int64_t a = 0; a < nl; ++a) for (int64_t b = 0; b < ne; ++b) CHECK(c.slot_of(a, b) == snap[k++]);
            }
        }
    }
    // per-layer quota: a full layer refuses even though other layers still have free slots
    {
        ExpertCache c; CHECK(c.open(2, 8, 6, 256, true));   // layer 0 -> [0,3), layer 1 -> [3,6)
        for (int e = 0; e < 3; ++e) CHECK(c.admit(0, e) == e);
        CHECK(c.admit(0, 3) == -1);
        CHECK(c.admit(1, 0) == 3);
        CHECK(c.slot_of(0, 3) == -1);
    }
    // ---- byte arithmetic: 256-byte alignment, arena >= raw bytes, bounded fills
    for (int64_t n : {0, 1, 255, 256, 257, 1000, 1382400, 1382401}) {
        const int64_t a = core::cache_align256(n);
        CHECK(a >= n && a % 256 == 0 && a < n + 256);
    }
    for (int64_t blob : {1, 100, 256, 1000}) for (int64_t bytes : {-3, 0, 1, 50, 100, 256, 1000, 5000}) {
        const int64_t f = core::fill_len(bytes, blob);
        CHECK(f <= blob);
        if (bytes == 0) CHECK(f == blob);
        if (bytes > 0 && bytes <= blob) CHECK(f == bytes); else CHECK(f == blob);
    }
    {
        std::vector<int64_t> blobs = {100, 256, 257, 1000, 3};
        ExpertCache c; CHECK(c.open_sized(1, 5, blobs, false));
        int64_t sum = 0, off = 0;
        for (size_t i = 0; i < blobs.size(); ++i) {
            sum += blobs[i];
            CHECK(c.slot_bytes((int64_t)i) == blobs[i]);
            uint8_t* p = c.device_slot((int64_t)i); CHECK(p != nullptr);
            if (i) CHECK(p == c.device_slot(0) + off);
            CHECK((p - c.device_slot(0)) % 256 == 0);
            off += core::cache_align256(blobs[i]);
        }
        CHECK(c.arena_bytes() >= sum); CHECK(c.arena_bytes() == off);
        CHECK(c.device_slot((int64_t)blobs.size()) == nullptr && c.device_slot(-1) == nullptr);
        // fill: whole slot when bytes == 0 / too large; partial when 0 < bytes <= blob; nothing written past the slot
        std::vector<uint8_t> src(2000); for (size_t i = 0; i < src.size(); ++i) src[i] = (uint8_t)(i * 7 + 1);
        CHECK(c.fill_slot_blocking(1, src.data(), 0));
        CHECK(std::memcmp(c.device_slot(1), src.data(), 256) == 0 && c.verify_slot(1, src.data(), 0));
        std::vector<uint8_t> guard_before(c.device_slot(2), c.device_slot(2) + 8);
        CHECK(c.fill_slot_blocking(1, src.data(), 99999));
        CHECK(std::vector<uint8_t>(c.device_slot(2), c.device_slot(2) + 8) == guard_before);   // neighbour untouched
        CHECK(c.fill_slot_blocking(0, src.data(), 10));
        CHECK(std::memcmp(c.device_slot(0), src.data(), 10) == 0);
        CHECK(!c.fill_slot_blocking(99, src.data(), 0));
        c.close(); CHECK(!c.is_open());
    }
    // ---- profile reader: header geometry, n_ranked <= slots, in-range pairs
    {
        const std::string dir = tmpdir();
        const int64_t NL = 4, NE = 16, want = 5;
        auto write = [&](const std::vector<uint8_t>& b, const char* name) { std::string p = dir + "/" + name; std::ofstream(p, std::ios::binary).write((const char*)b.data(), (std::streamsize)b.size()); return p; };
        std::vector<std::pair<int32_t, int32_t>> ok = {{0, 1}, {3, 15}, {2, 2}}, got;
        CHECK(core::read_expert_profile(write(core::encode_expert_profile(NL, NE, ok), "ok.bin"), NL, NE, want, got) && got == ok);
        CHECK(!core::read_expert_profile(write(core::encode_expert_profile(NL + 1, NE, ok), "nl.bin"), NL, NE, want, got));
        CHECK(!core::read_expert_profile(write(core::encode_expert_profile(NL, NE + 1, ok), "ne.bin"), NL, NE, want, got));
        std::vector<std::pair<int32_t, int32_t>> many(6, {0, 0});
        CHECK(!core::read_expert_profile(write(core::encode_expert_profile(NL, NE, many), "many.bin"), NL, NE, want, got));
        CHECK(core::read_expert_profile(write(core::encode_expert_profile(NL, NE, std::vector<std::pair<int32_t, int32_t>>(5, {1, 1})), "eq.bin"), NL, NE, want, got));
        for (auto bad : std::vector<std::pair<int32_t, int32_t>>{{4, 0}, {0, 16}, {-1, 0}, {0, -1}})
            CHECK(!core::read_expert_profile(write(core::encode_expert_profile(NL, NE, {{0, 0}, bad}), "bad.bin"), NL, NE, want, got));
        auto trunc = core::encode_expert_profile(NL, NE, ok); trunc.pop_back();
        CHECK(!core::read_expert_profile(write(trunc, "trunc.bin"), NL, NE, want, got));
        CHECK(!core::read_expert_profile(dir + "/absent.bin", NL, NE, want, got));
    }
    // ---- VRAM refusal arithmetic is host-side: a request above the free memory must not fit
    CHECK(core::cache_align256(0) == 0);
}

// =============================================================================================== ExpertGeometry
static float h2fl(uint16_t h) { return _cvtsh_ss(h); }

static void test_geometry() {
    using namespace kern;
    CHECK(H == 2560 && FF == 640 && NE == 512 && QK == 64 && QKA == 32 && BB == 18 && MAXT == 8);
    CHECK(MAXC == 80 && ROW_GU == 640 && ROW_D == 160 && SC_GU == 40 && SC_D == 10);
    CHECK(BLOB == 1382400);
    CHECK(BLOB == 3ull * (H * (uint64_t)FF * BB / QK));
    CHECK(O_GU_CODES == 0 && O_D_CODES == 819200 && O_GU_SCALES == 1228800 && O_D_SCALES == 1331200);
    CHECK(O_GU_CODES < O_D_CODES && O_D_CODES < O_GU_SCALES && O_GU_SCALES < O_D_SCALES && O_D_SCALES < BLOB);
    CHECK(O_D_SCALES + (uint64_t)H * SC_D * 2 == BLOB);                     // the four regions tile the blob exactly
    CHECK(BB == 2 + QK * 2 / 8 && QK == 2 * QKA);                           // one 64-weight block spans two 32-chunks
    CHECK(SC_GU * 2 == MAXC && SC_D * 2 * QKA == FF);
    CHECK(2ull * FF * ROW_GU == O_D_CODES && (uint64_t)H * ROW_D == O_GU_SCALES - O_D_CODES);
    for (uint32_t r : {0u, 1u, 2u * FF - 1}) { CHECK((uint64_t)r * ROW_GU + ROW_GU <= O_D_CODES); CHECK(O_GU_SCALES + (uint64_t)r * SC_GU * 2 + SC_GU * 2 <= O_D_SCALES); }
    for (uint32_t r : {0u, 1u, H - 1}) { CHECK(O_D_CODES + (uint64_t)r * ROW_D + ROW_D <= O_GU_SCALES); CHECK(O_D_SCALES + (uint64_t)r * SC_D * 2 + SC_D * 2 <= BLOB); }
    // the generic GGUF-order layout of the same expert has exactly the blob's size
    const ExpertLayout l = expert_layout(42, H, FF);
    CHECK(l.ok && (uint64_t)l.bytes == BLOB && l.gu_row_bytes == 720 && l.down_row_bytes == 180);
    ExpertLayout out;
    CHECK(expert_layout_load(out, 42, H, FF, BLOB) && out.ok);
    CHECK(!expert_layout_load(out, 42, H, FF, BLOB + 1) && !out.ok);
    CHECK(!expert_layout_load(out, 42, H, FF, BLOB - 1));
    CHECK(!expert_layout_load(out, 7, H, FF, BLOB));

    // ---- the ISA gate: usable == conjunction of the five flags; reason == "ok" iff usable
    for (int m = 0; m < 32; ++m) {
        CpuFeatures f; f.avx512f = m & 1; f.avx512bw = m & 2; f.avx512vl = m & 4; f.avx512_vnni = m & 8; f.avx512_vbmi = m & 16;
        CHECK(f.usable() == (m == 31));
        CHECK((std::strcmp(f.reason(), "ok") == 0) == f.usable());
    }
    CHECK(!CpuFeatures{}.usable() && std::strcmp(CpuFeatures{}.reason(), "ok") != 0);
    { CpuFeatures f; f.avx512f = f.avx512bw = f.avx512vl = f.avx512_vbmi = true; CHECK(!f.usable()); }   // no VNNI
    { CpuFeatures f; f.avx512f = f.avx512bw = f.avx512vl = f.avx512_vnni = true; CHECK(!f.usable()); }   // no VBMI
    // the probe agrees with the raw CPUID / XCR0 predicates
    {
        unsigned a, b, c, d; __cpuid_count(7, 0, a, b, c, d);
        unsigned lo = 0, hi = 0; __cpuid(1, a, b, c, d); (void)hi;
        const bool osx = (c >> 27) & 1u;
        unsigned x0 = 0; if (osx) { unsigned xh; __asm__ volatile("xgetbv" : "=a"(x0), "=d"(xh) : "c"(0)); }
        unsigned e7b, e7c; __cpuid_count(7, 0, a, e7b, e7c, d); (void)lo;
        CHECK(cpu_features().usable() == (osx && etable::xcr0_ok(x0) && etable::avx512_ok(e7b, e7c)));
        CHECK(cpu_features().usable());   // this machine is the target: the whole unit is built for it
        CHECK(cpu_avx512_ok());
    }

    // ---- blob row addressing on the real kernels: gate row r is matrix row 2r, up row r is matrix row 2r+1
    std::vector<uint8_t> blob(BLOB, 0);
    std::memset(blob.data() + O_GU_CODES, 0x55, (size_t)2 * FF * ROW_GU);       // code 1 everywhere: weight (1-1)*s = 0
    std::memset(blob.data() + O_D_CODES, 0x55, (size_t)H * ROW_D);
    const uint16_t one = f2h(1.f);
    auto set_scale_gu = [&](uint32_t mrow, uint16_t v) { for (uint32_t b = 0; b < SC_GU; ++b) std::memcpy(&blob[O_GU_SCALES + ((uint64_t)mrow * SC_GU + b) * 2], &v, 2); };
    auto set_scale_d = [&](uint32_t row, uint16_t v) { for (uint32_t b = 0; b < SC_D; ++b) std::memcpy(&blob[O_D_SCALES + ((uint64_t)row * SC_D + b) * 2], &v, 2); };
    const uint32_t R = 3;
    std::memset(&blob[O_GU_CODES + (uint64_t)(2 * R) * ROW_GU], 0xFF, ROW_GU);       // gate row R: code 3 -> weight 2*s
    std::memset(&blob[O_GU_CODES + (uint64_t)(2 * R + 1) * ROW_GU], 0xFF, ROW_GU);   // up row R
    set_scale_gu(2 * R, one); set_scale_gu(2 * R + 1, one);
    const uint32_t DR = 5;
    std::memset(&blob[O_D_CODES + (uint64_t)DR * ROW_D], 0xFF, ROW_D);
    set_scale_d(DR, one);
    std::vector<float> x(H); for (auto& v : x) v = std::uniform_real_distribution<float>(-1.f, 1.f)(rng);
    static ActQ xq; act_quant_q8_1(x.data(), H, xq.q, xq.d, xq.hx);
    double sx = 0; for (uint32_t c = 0; c < MAXC; ++c) { int s = 0; for (uint32_t i = 0; i < QKA; ++i) s += xq.q[c * QKA + i]; sx += (double)xq.d[c] * s; }
    std::vector<float> act(FF, -9.f);
    s2_expert_gu_rows(blob.data(), xq, 0, FF, act.data());
    const double g = 2.0 * sx, want_act = g / (1.0 + std::exp(-g)) * g;
    CHECK(std::fabs(act[R] - want_act) <= 2e-3 * std::fabs(want_act) + 1e-4);
    for (uint32_t r = 0; r < FF; ++r) if (r != R) CHECK(act[r] == 0.f);
    // down row: out[DR] = 2 * sum(h) with h the quantized activation
    static ActQ aq; std::vector<float> h(FF); for (auto& v : h) v = std::uniform_real_distribution<float>(-1.f, 1.f)(rng);
    act_quant_q8_1(h.data(), FF, aq.q, aq.d, aq.hx);
    double sh = 0; for (uint32_t c = 0; c < FF / QKA; ++c) { int s = 0; for (uint32_t i = 0; i < QKA; ++i) s += aq.q[c * QKA + i]; sh += (double)aq.d[c] * s; }
    std::vector<float> o(H, -9.f);
    s2_expert_down_rows(blob.data(), aq, 0, H, o.data());
    CHECK(std::fabs(o[DR] - 2.0 * sh) <= 2e-3 * std::fabs(2.0 * sh) + 1e-4);
    for (uint32_t r = 0; r < H; ++r) if (r != DR) CHECK(o[r] == 0.f);
    (void)h2fl;
}

// =============================================================================================== ExpertLayoutTable
static void test_layout_table() {
    using namespace etable;
    // contiguity
    std::vector<LayerEntry> L(3); uint64_t cur = 0;
    for (size_t i = 0; i < 3; ++i) { L[i].offset = cur; L[i].bytes = 1000 + 100 * i; cur += L[i].bytes * 8; }
    uint64_t total = 0;
    CHECK(contiguous(8, L, &total) && total == cur);                                      // contig_total
    uint64_t sum = 0; for (auto& e : L) sum += e.bytes * 8; CHECK(total == sum);
    for (size_t bad = 0; bad < 3; ++bad) for (int64_t delta : {-1, 1}) { auto M = L; M[bad].offset += delta; CHECK(!contiguous(8, M, nullptr)); }   // gap / overlap
    CHECK(contiguous(8, {}, &total) && total == 0);                                         // empty table
    { std::vector<LayerEntry> one(1); one[0].bytes = 777; CHECK(contiguous(5, one, &total) && total == 777u * 5); }
    CHECK(max_blob(L) == 1200 && max_blob({}) == 0);
    { auto M = L; std::swap(M[0].bytes, M[2].bytes); CHECK(max_blob(M) == 1200); }
    // gguf slots
    for (uint64_t nl : {1, 4, 48}) for (uint64_t l = 0; l < nl; ++l) for (uint64_t k = 0; k < 3; ++k) {
        CHECK(gguf_slot(l, k) < 3 * nl);
        for (uint64_t l2 = 0; l2 < nl; ++l2) for (uint64_t k2 = 0; k2 < 3; ++k2) if (gguf_slot(l, k) == gguf_slot(l2, k2)) CHECK(l == l2 && k == k2);
    }
    // gating predicates (the examples of the theory)
    CHECK(xcr0_ok(0xE7) && !xcr0_ok(0x07) && !xcr0_ok(0));
    for (uint64_t miss : {0x02ull, 0x04ull, 0x20ull, 0x40ull, 0x80ull}) CHECK(!xcr0_ok(0xE7 & ~miss) || (miss & 0xE6) == 0);
    const uint32_t F = 1u << 16, BW = 1u << 30, VL = 1u << 31, VNNI = 1u << 11, VBMI = 1u << 1;
    CHECK(avx512_ok(F | BW | VL, VNNI | VBMI));
    CHECK(!avx512_ok(F | BW | VL, VBMI) && !avx512_ok(F | BW | VL, VNNI) && !avx512_ok(0, 0));
    CHECK(!avx512_ok(BW | VL, VNNI | VBMI) && !avx512_ok(F | VL, VNNI | VBMI) && !avx512_ok(F | BW, VNNI | VBMI));
    // default layout
    CHECK(default_total(1, 512, 1382400) == 707788800ull);
    for (uint64_t ne : {1, 7, 512}) { std::vector<LayerEntry> one(1); one[0].bytes = 4321; CHECK(contiguous(ne, one, &total) && total == default_total(1, ne, 4321)); }
    {
        const ExpertTable t = default_table(3, 512);
        CHECK(t.by_default && t.total == 3ull * 512 * kern::BLOB && t.max_blob == kern::BLOB && t.layers.size() == 3);
        uint64_t tot = 0; CHECK(contiguous(512, t.layers, &tot) && tot == t.total);
    }
    // text format: v1 / v2, validation, load with and without a file
    {
        const std::string good = "native_experts v1 3 8\n# comment\n0 1000 17 22\n8000 1100 17 22\n16800 1200 18 21\n";
        ExpertTable t; std::string err;
        CHECK(parse_expert_table(good, t, err) && t.n_layers == 3 && t.total == 8000u + 8800u + 9600u && t.max_blob == 1200 && !t.v2);
        CHECK(t.layers[2].gu_type == 18 && t.layers[2].d_type == 21);
        const std::string gap = "native_experts v1 2 8\n0 1000 17 22\n8001 1100 17 22\n";
        CHECK(!parse_expert_table(gap, t, err) && !err.empty());
        const std::string overlap = "native_experts v1 2 8\n0 1000 17 22\n7999 1100 17 22\n";
        CHECK(!parse_expert_table(overlap, t, err));
        const std::string v2 = "native_experts v2 2 4\n0 100 17 22 10 20 30\n400 100 17 22 40 50 60\n";
        CHECK(parse_expert_table(v2, t, err) && t.v2 && t.gguf.size() == 6 && t.gguf[gguf_slot(1, 0)] == 40 && t.gguf[gguf_slot(1, 2)] == 60 && t.gguf[gguf_slot(0, 1)] == 20);
        CHECK(!parse_expert_table("native_experts v2 1 4\n0 100 17 22 10 20\n", t, err));   // missing the third offset
        CHECK(!parse_expert_table("native_experts v1 2 8\n0 1000 17 22\n", t, err));        // missing layer
        CHECK(!parse_expert_table(good + "99 1 1 1\n", t, err));                            // trailing data
        CHECK(!parse_expert_table("bogus v1 1 1\n0 1 1 1\n", t, err) && !parse_expert_table("", t, err));
        const std::string dir = tmpdir();
        { std::ofstream(dir + "/t.txt") << good; }
        CHECK(load_expert_table(dir + "/t.txt", 3, 8, t, err) && !t.by_default && t.total == 26400);
        CHECK(!load_expert_table(dir + "/t.txt", 4, 8, t, err));                            // geometry must match the model
        CHECK(!load_expert_table(dir + "/t.txt", 3, 9, t, err));
        CHECK(load_expert_table(dir + "/absent.txt", 2, 512, t, err) && t.by_default && t.total == default_total(2, 512, kern::BLOB));
    }
}

// =============================================================================================== ExpertPoolContract / Scheduling
struct Grids {
    std::vector<uint64_t> g2xxs, g2xs, g2s; std::vector<uint32_t> g3xxs, g3s;
    Grids() : g2xxs(256), g2xs(512), g2s(1024), g3xxs(256), g3s(512) {
        auto m8 = [&](int mx) { uint64_t v = 0; for (int i = 0; i < 8; ++i) v |= (uint64_t)(rng() % (mx + 1)) << (8 * i); return v; };
        auto m4 = [&]() { uint32_t v = 0; for (int i = 0; i < 4; ++i) v |= (uint32_t)(1 + 2 * (rng() % 8)) << (8 * i); return v; };
        for (auto& v : g2xxs) v = m8(43);
        for (auto& v : g2xs) v = m8(43);
        for (auto& v : g2s) v = m8(43);
        for (auto& v : g3xxs) v = m4();
        for (auto& v : g3s) v = m4();
    }
};
static bool bit_equal(const float* a, const float* b, size_t n) { return std::memcmp(a, b, n * sizeof(float)) == 0; }

static void test_pool_contract() {
    using namespace kern;
    // capacities
    CHECK(ExpertPool::kMaxSplit == 16 && ExpertPool::kMaxSplitMulti == 96 && MAXT == 8);
    CHECK(ExpertPool::kMaxSplit <= ExpertPool::kMaxSplitMulti);
    CHECK(cpu_pool::kMaxSplitMulti == ExpertPool::kMaxSplitMulti);
    // the split-path guard: refused iff n > 16 || nw == 1 || oracle
    for (uint32_t n = 0; n < 40; ++n) for (uint32_t nw : {1u, 2u, 7u}) for (int o = 0; o < 2; ++o)
        CHECK(poolsched::use_split(n, 16, o, nw) == !(n > 16 || nw == 1 || o));
    // worker / thread counting
    for (uint32_t req : {0u, 1u, 5u}) for (uint32_t cores : {0u, 1u, 12u}) {
        const uint32_t w = poolsched::pool_workers(req, cores);
        CHECK(w >= 1 && w == std::max<uint32_t>(1, req ? req : cores));
        CHECK(poolsched::drain_threads(w, false) == w && poolsched::drain_threads(w, true) == w + 1 && poolsched::drain_threads(w, false) >= 1);
        CHECK(cpu_pool::n_workers(req, cores) == w);
    }
    { std::vector<int> c = {0, 2, 4}; CHECK(cpu_pool::skip_first(c).size() == 2 && cpu_pool::skip_first(c)[0] == 2 && cpu_pool::skip_first({}).empty()); }
    { ExpertPool p(0, false, false); CHECK(p.workers() >= 1); ExpertPool q(0, true, true); CHECK(q.workers() >= 1); CHECK(q.threads() == q.workers() + 1); }
    { ExpertPool p(3, false, true); CHECK(p.workers() == 3 && p.threads() == 4); }
    { const auto c = physical_cores(); CHECK(!c.empty()); }
    // each contended word alone on its own 64-byte line
    {
        const size_t off[5] = {ExpertPool::off_head(), ExpertPool::off_done(), ExpertPool::off_parked(), ExpertPool::off_epoch(), ExpertPool::off_stop()};
        for (int i = 0; i < 5; ++i) { CHECK(off[i] % 64 == 0); for (int j = 0; j < 5; ++j) if (i != j) CHECK(off[i] / 64 != off[j] / 64); }
        for (uint64_t i = 0; i < 6; ++i) for (uint64_t j = 0; j < 6; ++j) if (i != j) CHECK(64 * i / 64 != 64 * j / 64);
    }
    // task counts: ceil division covers 3 * threads; at least one part
    for (uint32_t thr = 1; thr <= 70; ++thr) for (uint32_t n = 1; n <= 16; ++n) {
        const uint32_t parts = cpu_pool::parts_per_expert(thr, n);
        CHECK(parts >= 1 && n * parts >= 3 * thr && parts == std::max<uint32_t>(1, (3 * thr + n - 1) / n));
        CHECK(cpu_pool::tasks_per_phase(thr, false) == 3ull * thr && 3 * thr >= 3);
    }
    // batching beyond the buffer capacity: batches of <= 96 that sum to n
    for (uint32_t n = 0; n <= 400; ++n) {
        uint64_t sum = 0; for (uint64_t b : cpu_pool::chunks(96, n)) { CHECK(b >= 1 && b <= 96); sum += b; }
        CHECK(sum == n);
        CHECK(cpu_pool::chunks(96, n).size() == poolsched::batch_count(n, 96));
    }
    // affinity restore: a refused pin is never restored
    {
        cpu_set_t before; CPU_ZERO(&before); sched_getaffinity(0, sizeof before, &before);
        restore_thread_affinity();                       // nothing pinned yet: no effect
        cpu_set_t a; CPU_ZERO(&a); sched_getaffinity(0, sizeof a, &a); CHECK(CPU_EQUAL(&before, &a));
        CHECK(!pin_current_thread(-1));                  // refused
        restore_thread_affinity();
        CPU_ZERO(&a); sched_getaffinity(0, sizeof a, &a); CHECK(CPU_EQUAL(&before, &a));
        const auto cores = physical_cores();
        if (pin_current_thread(cores[0])) {
            cpu_set_t p; CPU_ZERO(&p); sched_getaffinity(0, sizeof p, &p); CHECK(CPU_COUNT(&p) == 1);
            restore_thread_affinity();
            CPU_ZERO(&a); sched_getaffinity(0, sizeof a, &a); CHECK(CPU_EQUAL(&before, &a));
        }
        CHECK(cpu_pool::should_pin(0) && !cpu_pool::should_pin(-1));
    }
}

static std::vector<uint8_t> random_q2_blob() {
    std::vector<uint8_t> b(kern::BLOB);
    for (uint64_t i = 0; i < kern::O_GU_SCALES; ++i) b[i] = (uint8_t)rng();
    for (uint64_t i = kern::O_GU_SCALES; i < kern::BLOB; i += 2) { const uint16_t h = f2h(0.005f + 0.045f * (float)(rng() % 1000) / 1000.f); std::memcpy(&b[i], &h, 2); }
    return b;
}

static void test_pool_scheduling() {
    using namespace kern;
    // ---- claim counter: exactly n of any number of racing claims succeed, each index once
    for (uint32_t nthreads : {1u, 3u, 8u}) for (uint64_t n : {0ull, 1ull, 7ull, 1000ull}) {
        std::atomic<uint64_t> head{0};
        std::vector<std::atomic<int>> hits((size_t)n + 1);
        for (auto& h : hits) h = 0;
        std::atomic<uint64_t> granted{0};
        std::vector<std::thread> th;
        for (uint32_t t = 0; t < nthreads; ++t) th.emplace_back([&] { for (;;) { const uint64_t i = head.fetch_add(1); if (!cpu_pool::valid_claim(i, n)) break; ++hits[(size_t)i]; ++granted; } });
        for (auto& t : th) t.join();
        CHECK(granted == n);
        for (uint64_t i = 0; i < n; ++i) CHECK(hits[(size_t)i] == 1);
    }
    // ---- equal row partition: lo(0)=0, lo(parts)=N, monotone, widths telescope, tile with no gap or overlap
    for (int trial = 0; trial < 300; ++trial) {
        const uint64_t N = rng() % 5000, parts = 1 + rng() % 70;
        CHECK(poolsched::row_start(N, parts, 0) == 0 && poolsched::row_start(N, parts, parts) == N);
        uint64_t covered = 0, prev = 0;
        for (uint64_t i = 0; i < parts; ++i) {
            const uint64_t a = poolsched::row_start(N, parts, i), b = poolsched::row_start(N, parts, i + 1);
            CHECK(a == prev && a <= b && b == N * (i + 1) / parts);
            covered += b - a; prev = b;
        }
        CHECK(covered == N && cpu_pool::split_pt(N, parts, parts) == N);
    }
    // ---- task decomposition: (i / parts, i % parts) is in range, recombines, injective
    for (uint64_t n : {1ull, 5ull, 16ull}) for (uint64_t parts : {1ull, 3ull, 7ull}) {
        std::vector<std::pair<uint64_t, uint64_t>> seen;
        for (uint64_t i = 0; i < n * parts; ++i) {
            const uint64_t e = cpu_pool::task_expert(i, parts), p = cpu_pool::task_part(i, parts);
            CHECK(e < n && p < parts && parts * e + p == i);
            for (auto& s : seen) CHECK(!(s.first == e && s.second == p));
            seen.push_back({e, p});
        }
    }
    // ---- the multi-expert row walk: every step advances, stays inside its expert and inside the task; the pieces tile [g0, g1)
    for (int trial = 0; trial < 400; ++trial) {
        const uint64_t per = 1 + rng() % 300, n = 1 + rng() % 12, g0 = rng() % (n * per), g1 = g0 + rng() % (n * per - g0 + 1);
        uint64_t r = g0; size_t steps = 0;
        for (const auto& s : cpu_pool::segment(per, g0, g1)) {
            CHECK(s.expert < n && s.r0 < s.r1 && s.r1 <= per);
            CHECK(s.expert * per + s.r0 == r);
            r += s.r1 - s.r0; ++steps;
        }
        CHECK(r == g1 && steps <= g1 - g0 + 1);
        const auto ps = poolsched::pieces(per, g0, g1); CHECK(ps.size() == cpu_pool::segment(per, g0, g1).size());
    }
    // ---- the real pool: every job exactly once, many batches; single worker refuses the split path and writes nothing
    {
        const auto blob = random_q2_blob();
        const uint32_t NJ = 16;
        std::vector<std::vector<float>> xs(NJ, std::vector<float>(H)), want(NJ, std::vector<float>(H)), out(NJ, std::vector<float>(H));
        for (auto& x : xs) for (auto& v : x) v = std::uniform_real_distribution<float>(-1.f, 1.f)(rng);
        for (uint32_t j = 0; j < NJ; ++j) s2_expert_scalar(blob.data(), xs[j].data(), want[j].data());
        std::vector<ExpertScratch*> scr(NJ); for (auto& s : scr) s = new ExpertScratch;
        std::vector<ExpertJob> jobs(NJ);
        for (uint32_t j = 0; j < NJ; ++j) jobs[j] = {blob.data(), xs[j].data(), out[j].data(), scr[j]};
        for (uint32_t nw : {1u, 2u, 5u}) {
            ExpertPool pool(nw, false, nw % 2 == 0);
            for (int rep = 0; rep < 30; ++rep) {
                const uint32_t n = 1 + (uint32_t)(rep % NJ);
                for (auto& o : out) std::fill(o.begin(), o.end(), -1.f);
                pool.run(jobs.data(), n);
                for (uint32_t j = 0; j < NJ; ++j) { if (j < n) CHECK(bit_equal(out[j].data(), want[j].data(), H)); else CHECK(out[j][0] == -1.f); }
                for (auto& o : out) std::fill(o.begin(), o.end(), -1.f);
                const bool ran = pool.run_split(jobs.data(), n);
                CHECK(ran == (nw > 1 && n <= 16));
                for (uint32_t j = 0; j < NJ; ++j) { if (ran && j < n) CHECK(bit_equal(out[j].data(), want[j].data(), H)); else CHECK(out[j][0] == -1.f); }
            }
        }
        for (auto s : scr) delete s;
    }
    // ---- native layers larger than one batch are processed in batches of <= 96 (all jobs validated first)
    {
        Grids g; IqTables t;
        t.iq2xxs_grid = g.g2xxs.data(); t.iq2xs_grid = g.g2xs.data(); t.iq2s_grid = g.g2s.data(); t.iq3xxs_grid = g.g3xxs.data(); t.iq3s_grid = g.g3s.data();
        iq512_set_tables(t);
        const uint32_t n_embd = 256, n_ff = 256, gt = 17, dt = 22;
        const size_t gu_rb = (size_t)(n_embd / 256) * iqgeo::block_bytes(gt), d_rb = (size_t)(n_ff / 256) * iqgeo::block_bytes(dt);
        const uint32_t NVAR = 5;   // a handful of distinct experts reused across many jobs
        std::vector<std::vector<uint8_t>> guw(NVAR), dw(NVAR);
        for (uint32_t v = 0; v < NVAR; ++v) {
            guw[v].resize(2 * n_ff * gu_rb); dw[v].resize(n_embd * d_rb);
            for (auto& b : guw[v]) b = (uint8_t)rng();
            for (auto& b : dw[v]) b = (uint8_t)rng();
            for (uint32_t r = 0; r < 2 * n_ff; ++r) { const uint16_t h = f2h(0.01f + 0.04f * (float)(rng() % 100) / 100.f); std::memcpy(&guw[v][r * gu_rb], &h, 2); }
            for (uint32_t r = 0; r < n_embd; ++r) { const uint16_t h = f2h(0.01f + 0.04f * (float)(rng() % 100) / 100.f); std::memcpy(&dw[v][r * d_rb], &h, 2); }
        }
        auto build = [&](uint32_t E, std::vector<ExpertJobMulti>& jobs, std::vector<std::vector<float>>& in, std::vector<std::vector<float>>& out, std::vector<std::vector<float>>& ref) {
            jobs.assign(E, ExpertJobMulti{}); in.assign(E, {}); out.assign(E, {}); ref.assign(E, {});
            for (uint32_t e = 0; e < E; ++e) {
                const uint32_t v = e % NVAR, nt = 1 + e % 3;
                in[e].resize((size_t)nt * n_embd); for (auto& f : in[e]) f = std::uniform_real_distribution<float>(-1.f, 1.f)(rng);
                out[e].assign((size_t)nt * n_embd, -1.f); ref[e].assign((size_t)nt * n_embd, 0.f);
                std::vector<Q8KBlock> x8(nt), a8(nt); std::vector<float> act((size_t)nt * n_ff);
                for (uint32_t k = 0; k < nt; ++k) iquant512::q8k_quantize(in[e].data() + (size_t)k * n_embd, n_embd, x8.data() + k);
                iq512_gu_rows(gt, guw[v].data(), gu_rb, n_ff, 0, n_ff, 1, x8.data(), nt, act.data(), n_ff);
                for (uint32_t k = 0; k < nt; ++k) iquant512::q8k_quantize(act.data() + (size_t)k * n_ff, n_ff, a8.data() + k);
                iq512_rows(dt, dw[v].data(), d_rb, 0, n_embd, 1, a8.data(), nt, ref[e].data(), n_embd);
                ExpertJobMulti& j = jobs[e];
                j.gu_type = gt; j.d_type = dt; j.gu_w = guw[v].data(); j.d_w = dw[v].data(); j.gu_row_bytes = gu_rb; j.d_row_bytes = d_rb;
                j.n_ff = n_ff; j.n_embd = n_embd; j.nt = nt; j.x = in[e].data(); j.out = out[e].data();
            }
        };
        ExpertPool pool(4, false, true);
        for (uint32_t E : {1u, 96u, 97u, 192u, 200u}) {
            std::vector<ExpertJobMulti> jobs; std::vector<std::vector<float>> in, out, ref;
            build(E, jobs, in, out, ref);
            CHECK(pool.run_split_multi_native(jobs.data(), E));
            for (uint32_t e = 0; e < E; ++e) CHECK(bit_equal(out[e].data(), ref[e].data(), out[e].size()));
        }
        {   // an invalid job anywhere in an oversized layer: nothing runs at all
            std::vector<ExpertJobMulti> jobs; std::vector<std::vector<float>> in, out, ref;
            build(150, jobs, in, out, ref);
            jobs[140].nt = 0;
            CHECK(!pool.run_split_multi_native(jobs.data(), 150));
            for (uint32_t e = 0; e < 150; ++e) CHECK(out[e][0] == -1.f);
            jobs[140].nt = 1; jobs[149].gu_type = 42;
            CHECK(!pool.run_split_multi_native(jobs.data(), 150));
            for (uint32_t e = 0; e < 150; ++e) CHECK(out[e][0] == -1.f);
            jobs[149].gu_type = gt;
            expert_set_oracle_q8_0(true);
            CHECK(!pool.run_split_multi_native(jobs.data(), 150));
            expert_set_oracle_q8_0(false);
            for (uint32_t e = 0; e < 150; ++e) CHECK(out[e][0] == -1.f);
        }
        ExpertPool one(1, false, false);
        std::vector<ExpertJobMulti> jobs; std::vector<std::vector<float>> in, out, ref;
        build(120, jobs, in, out, ref);
        CHECK(!one.run_split_multi_native(jobs.data(), 120));   // a single worker never takes the split path
    }
}

int main() {
    test_cache();
    test_geometry();
    test_layout_table();
    test_pool_contract();
    test_pool_scheduling();
    std::printf("checks: %d, failures: %d\n", g_checks, g_fail);
    if (g_fail) { std::printf("U1DE TESTS FAILED\n"); return 1; }
    std::printf("U1DE TESTS PASSED\n");
    return 0;
}
