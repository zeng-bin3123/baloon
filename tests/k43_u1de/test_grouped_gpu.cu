// test_grouped_gpu.cu - the VRAM-resident expert tier end to end on a real device: the ExpertCache residency table is
// uploaded as the kernels' slot_of_expert table and its slot arena as the expert pool; hit selection, grouping and the
// grouped S2 kernels must agree with the cache and with the CPU Q2_0 expert kernel.
//  * hit selection == the residency table (dense router-order compaction, out-of-range ids are misses);
//  * a slot admitted by the per-layer mode lies inside its layer's range, and layers never share a slot;
//  * the expert evaluated from slot s is the expert filled into slot s (CPU parity), non-resident ids contribute nothing;
//  * where an expert sits in the arena does not change its output (re-admit in another order: bit-identical result).
#include "../k14gpu_common.h"

#include "baloon/core_expert_cache.hpp"
#include "baloon/kern_cpu_expert.hpp"
#include "baloon/kern_cuda_misc.hpp"

using namespace baloon;
using namespace baloon::kern;

static void put_h(uint8_t* p, float f) { const uint16_t h = h_of(f); p[0] = (uint8_t)(h & 255); p[1] = (uint8_t)(h >> 8); }

static std::vector<uint8_t> expert_blob(uint64_t seed) {
    Rng r(seed);
    std::vector<uint8_t> b(q2geo::BLOB);
    for (uint64_t i = 0; i < q2geo::O_GU_SCALES; ++i) b[i] = (uint8_t)r.next();
    for (uint64_t i = q2geo::O_GU_SCALES; i < q2geo::BLOB; i += 2) put_h(&b[i], 0.005f + (float)r.uni() * 0.045f);
    return b;
}


// ---- device-side views of the residency tier --------------------------------------------------------------------------
// slot_checksum_kernel: one block per listed slot sums the slot's bytes (weighted by position mod 251) on the device, so the
// pool really holds the blob that was filled into that slot of the cache arena.
__global__ void slot_checksum_kernel(const uint8_t* pool, uint64_t slot_bytes, const int32_t* slots, uint32_t* partial) {
    const uint8_t* p = pool + (uint64_t)slots[blockIdx.x] * slot_bytes;
    uint32_t s = 0;
    for (uint64_t i = threadIdx.x; i < slot_bytes; i += blockDim.x) s += (uint32_t)p[i] * (uint32_t)((i % 251) + 1);
    partial[blockIdx.x * blockDim.x + threadIdx.x] = s;
}
// table_lookup_kernel: the device's slot_of(layer, expert) for a list of ids; out-of-range ids are not resident (-1).
__global__ void table_lookup_kernel(const int32_t* table, const int32_t* ids, int n, int ne, int32_t* out) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { const int e = ids[i]; out[i] = (e >= 0 && e < ne) ? table[e] : -1; }
}
static uint32_t host_checksum(const std::vector<uint8_t>& b) {
    uint32_t s = 0;
    for (size_t i = 0; i < b.size(); ++i) s += (uint32_t)b[i] * (uint32_t)((i % 251) + 1);
    return s;
}

struct Fixture {
    static constexpr int NL = 2, NE = 512, SLOTS = 24, K = 10, NT = 3;
    std::vector<int> experts;                 // experts admitted in layer 1
    std::vector<std::vector<uint8_t>> blob;   // blob[e] for e in `experts` (index by position)
};

// Admit `order` into layer 1 of a fresh per-layer cache and fill each slot with that expert's blob. Returns the cache.
static void build_cache(core::ExpertCache& c, const std::vector<int>& order, const std::vector<std::vector<uint8_t>>& blobs, const std::vector<int>& experts) {
    CHECK(c.open(Fixture::NL, Fixture::NE, Fixture::SLOTS, (int64_t)q2geo::BLOB, true));
    for (int e : order) {
        const int32_t s = c.admit(1, e);
        CHECK(s >= 12 && s < 24);                                          // layer 1 owns [12, 24)
        CHECK(c.slot_of(1, e) == s && c.admit(1, e) == s);                 // resident, idempotent
        CHECK(c.slot_of(0, e) == -1);                                      // layer 0 never saw it
        size_t pos = 0; while (experts[pos] != e) ++pos;
        CHECK(c.fill_slot_blocking(s, blobs[pos].data(), 0));
        CHECK(c.verify_slot(s, blobs[pos].data(), 0));
    }
}

int main() {
    int ndev = 0;
    CUOK(cudaGetDeviceCount(&ndev));
    CHECK(ndev > 0);
    CUOK(cudaSetDevice(0));
    cudaDeviceProp prop; CUOK(cudaGetDeviceProperties(&prop, 0));
    std::printf("device 0: %s\n", prop.name);
    cpu_features();

    Rng rg(2024);
    Fixture fx;
    for (int i = 0; i < 12; ++i) {                       // 12 distinct resident experts of layer 1 (exactly its quota)
        int e; bool dup_ = true;
        while (dup_) { e = (int)rg.below(Fixture::NE); dup_ = false; for (int x : fx.experts) dup_ |= (x == e); }
        fx.experts.push_back(e); fx.blob.push_back(expert_blob(100 + (uint64_t)e));
    }
    std::vector<int> order1 = fx.experts, order2(fx.experts.rbegin(), fx.experts.rend());

    core::ExpertCache c1, c2;
    build_cache(c1, order1, fx.blob, fx.experts);
    build_cache(c2, order2, fx.blob, fx.experts);
    {   // the quota of layer 1 is full: a 13th expert is refused although layer 0 still has free slots, nothing evicted
        int extra = 0; while (true) { bool used = false; for (int x : fx.experts) used |= (x == extra); if (!used) break; ++extra; }
        const int32_t before = c1.slot_of(1, fx.experts[0]);
        CHECK(c1.admit(1, extra) == -1 && c1.slot_of(1, extra) == -1);
        CHECK(c1.slot_of(1, fx.experts[0]) == before);
        CHECK(c1.admit(0, extra) >= 0 && c1.admit(0, extra) < 12);   // layer 0 still admits, inside [0, 12)
    }
    // the two caches place the experts differently
    bool differs = false; for (int e : fx.experts) differs |= (c1.slot_of(1, e) != c2.slot_of(1, e));
    CHECK(differs);

    // slot_of_expert tables of layer 1 (host), and of layer 0 before its extra admission was made (all misses) ---------------------
    std::vector<int32_t> tab1(Fixture::NE), tab2(Fixture::NE), tab0(Fixture::NE, -1);
    for (int e = 0; e < Fixture::NE; ++e) { tab1[e] = c1.slot_of(1, e); tab2[e] = c2.slot_of(1, e); }

    // router rows: resident, non-resident and out-of-range ids mixed ------------------------------------------------------------
    const int n_tok = Fixture::NT, K = Fixture::K;
    std::vector<int32_t> ids((size_t)n_tok * K);
    for (int t = 0; t < n_tok; ++t) for (int k = 0; k < K; ++k) {
        const uint32_t m = rg.below(8);
        ids[(size_t)t * K + k] = m < 4 ? fx.experts[rg.below(12)] : m < 6 ? (int)rg.below(Fixture::NE) : m == 6 ? -1 : 512 + (int)rg.below(100);
    }
    ids[0] = fx.experts[0]; ids[1] = -5; ids[2] = fx.experts[5]; ids[(size_t)K] = 700;   // pin a few positions
    std::vector<float> x((size_t)n_tok * q2geo::H), w((size_t)n_tok * K), out0((size_t)n_tok * q2geo::H, 0.25f);
    for (auto& v : x) v = rg.sym(1.f);
    for (auto& v : w) v = 0.1f + (float)rg.uni() * 0.5f;

    // host model: resident hits per token, CPU kernel per resident pair, router-slot order
    std::vector<int> nh_want(n_tok, 0); std::vector<int32_t> hs_want((size_t)n_tok * K, -1), hd_want((size_t)n_tok * K, -1);
    std::vector<float> want = out0;
    for (int t = 0; t < n_tok; ++t) {
        std::vector<float> acc(q2geo::H, 0.f), o(q2geo::H);
        for (int k = 0; k < K; ++k) {
            const int e = ids[(size_t)t * K + k];
            const int s = (e >= 0 && e < Fixture::NE) ? tab1[e] : -1;
            if (s < 0) continue;
            hs_want[(size_t)t * K + nh_want[t]] = s; hd_want[(size_t)t * K + nh_want[t]] = k; ++nh_want[t];
            size_t pos = 0; while (fx.experts[pos] != e) ++pos;
            s2_expert_scalar(fx.blob[pos].data(), &x[(size_t)t * q2geo::H], o.data());
            for (int i = 0; i < (int)q2geo::H; ++i) acc[i] += w[(size_t)t * K + k] * o[i];
        }
        for (int i = 0; i < (int)q2geo::H; ++i) want[(size_t)t * q2geo::H + i] += acc[i];
    }
    double maxabs = 0; for (float v : want) maxabs = std::max(maxabs, (double)std::fabs(v));

    // device side: the cache arenas are the expert pools -------------------------------------------------------------------------
    const uint64_t slot_bytes = (uint64_t)core::cache_align256((int64_t)q2geo::BLOB);
    CHECK(slot_bytes == q2geo::BLOB);                                             // blob is a multiple of 256: slots are back to back
    CHECK(c1.device_slot(1) == c1.device_slot(0) + slot_bytes);
    std::vector<uint8_t> pool1(c1.device_slot(0), c1.device_slot(0) + c1.arena_bytes()), pool2(c2.device_slot(0), c2.device_slot(0) + c2.arena_bytes());
    CHECK(pool1.size() == (size_t)Fixture::SLOTS * slot_bytes);
    uint8_t* dpool1 = dup(pool1); uint8_t* dpool2 = dup(pool2);
    int32_t* dtab1 = dup(tab1); int32_t* dtab2 = dup(tab2); int32_t* dtab0 = dup(tab0);
    int32_t* dids = dup(ids); float* dx = dup(x); float* dw = dup(w);
    const size_t sb = moe_hit_grouped_scratch_bytes(n_tok, K);
    void* scratch = nullptr; CUOK(cudaMalloc(&scratch, sb));
    int32_t* hit_slot = dalloc<int32_t>((size_t)n_tok * K); int32_t* hit_dst = dalloc<int32_t>((size_t)n_tok * K); int32_t* n_hit = dalloc<int32_t>(n_tok);

    // 1) hit selection == residency table
    CHECK(moe_hit_select_multi(dids, dtab1, n_tok, K, hit_slot, hit_dst, n_hit, 0) == cudaSuccess);
    sync();
    {
        const auto hs = fetch(hit_slot, (size_t)n_tok * K), hd = fetch(hit_dst, (size_t)n_tok * K), nh = fetch(n_hit, n_tok);
        for (int t = 0; t < n_tok; ++t) CHECK(nh[t] == nh_want[t]);
        CHECK(hs == hs_want); CHECK(hd == hd_want);
        for (int t = 0; t < n_tok; ++t) for (int p = 0; p < nh[t]; ++p) CHECK(hs[(size_t)t * K + p] >= 12 && hs[(size_t)t * K + p] < 24);   // inside layer 1's range
        const auto nonresident = [&] { int n = 0; for (int t = 0; t < n_tok; ++t) n += K - nh_want[t]; return n; }();
        CHECK(nonresident > 0);                                                   // the fixture really contains misses
    }
    // layer 0's table (nothing admitted): every id misses, the output is untouched
    {
        int32_t* h0s = dalloc<int32_t>((size_t)n_tok * K); int32_t* h0d = dalloc<int32_t>((size_t)n_tok * K); int32_t* n0 = dalloc<int32_t>(n_tok);
        CHECK(moe_hit_select_multi(dids, dtab0, n_tok, K, h0s, h0d, n0, 0) == cudaSuccess);
        float* dout = dup(out0);
        CHECK(moe_hit_grouped_s2_cpu_order(dpool1, slot_bytes, dx, h0s, h0d, n_tok, K, dw, dout, scratch, 0) == cudaSuccess);
        sync();
        const auto nz = fetch(n0, n_tok); for (int t = 0; t < n_tok; ++t) CHECK(nz[t] == 0);
        CHECK(fetch(dout, out0.size()) == out0);
        cudaFree(h0s); cudaFree(h0d); cudaFree(n0); cudaFree(dout);
    }
    // 2) CPU parity of the resident experts, CPU-order and fast arithmetic
    float* dout_a = dup(out0);
    CHECK(moe_hit_grouped_s2_cpu_order(dpool1, slot_bytes, dx, hit_slot, hit_dst, n_tok, K, dw, dout_a, scratch, 0) == cudaSuccess);
    sync();
    const auto got_a = fetch(dout_a, out0.size());
    { bool ok = true; for (size_t i = 0; i < got_a.size(); ++i) ok &= std::fabs(got_a[i] - want[i]) <= 2e-4 * maxabs; CHECK(ok); }
    {
        float* dout = dup(out0);
        CHECK(moe_hit_grouped_s2_multi(dpool1, slot_bytes, dx, hit_slot, hit_dst, n_tok, K, dw, dout, scratch, 0) == cudaSuccess);
        sync();
        const auto got = fetch(dout, out0.size());
        bool ok = true; for (size_t i = 0; i < got.size(); ++i) ok &= std::fabs(got[i] - want[i]) <= 5e-3 * maxabs; CHECK(ok);
        cudaFree(dout);
    }
    // 3) the arena position is irrelevant: the second cache holds the same experts in other slots; same table semantics, same bits
    {
        int32_t* h2s = dalloc<int32_t>((size_t)n_tok * K); int32_t* h2d = dalloc<int32_t>((size_t)n_tok * K); int32_t* n2 = dalloc<int32_t>(n_tok);
        CHECK(moe_hit_select_multi(dids, dtab2, n_tok, K, h2s, h2d, n2, 0) == cudaSuccess);
        float* dout_b = dup(out0);
        CHECK(moe_hit_grouped_s2_cpu_order(dpool2, slot_bytes, dx, h2s, h2d, n_tok, K, dw, dout_b, scratch, 0) == cudaSuccess);
        sync();
        const auto got_b = fetch(dout_b, out0.size());
        CHECK(std::memcmp(got_a.data(), got_b.data(), got_a.size() * sizeof(float)) == 0);
        const auto hs1 = fetch(hit_slot, (size_t)n_tok * K), hs2 = fetch(h2s, (size_t)n_tok * K);
        bool moved = false; for (size_t i = 0; i < hs1.size(); ++i) moved |= (hs1[i] != hs2[i]);
        CHECK(moved);
        cudaFree(h2s); cudaFree(h2d); cudaFree(n2); cudaFree(dout_b);
    }
    // 4) grouped by resident expert: same result as the per-token hit lists
    {
        int32_t* ps = dalloc<int32_t>((size_t)n_tok * K); int32_t* pt = dalloc<int32_t>((size_t)n_tok * K); int32_t* pd = dalloc<int32_t>((size_t)n_tok * K);
        CHECK(moe_group_resident(dids, dtab1, n_tok, K, ps, pt, pd, 0) == cudaSuccess);
        sync();
        const auto s = fetch(ps, (size_t)n_tok * K), tk = fetch(pt, (size_t)n_tok * K), d = fetch(pd, (size_t)n_tok * K);
        int npairs = 0; for (int t = 0; t < n_tok; ++t) npairs += nh_want[t];
        for (int p = 0; p < npairs; ++p) {   // every pair names the slot the cache gave that expert, ascending expert order
            CHECK(s[p] >= 12 && s[p] < 24 && tk[p] >= 0 && tk[p] < n_tok && d[p] >= 0 && d[p] < K);
            CHECK(tab1[ids[(size_t)tk[p] * K + d[p]]] == s[p]);
        }
        for (int p = npairs; p < n_tok * K; ++p) CHECK(s[p] == -1);
        float* dout = dup(out0);
        CHECK(moe_grouped_s2(dpool1, slot_bytes, dx, ps, pt, pd, n_tok, K, dw, dout, scratch, 0) == cudaSuccess);
        sync();
        const auto got = fetch(dout, out0.size());
        bool ok = true; for (size_t i = 0; i < got.size(); ++i) ok &= std::fabs(got[i] - want[i]) <= 5e-3 * maxabs; CHECK(ok);
        cudaFree(ps); cudaFree(pt); cudaFree(pd); cudaFree(dout);
    }
    // 4b) device-side kernels: every resident slot of the pool holds exactly the blob filled into it; the device lookup of
    //     the residency table agrees with the cache for resident, absent and out-of-range ids
    {
        std::vector<int32_t> slots; for (int e : fx.experts) slots.push_back(tab1[e]);
        int32_t* dslots = dup(slots); uint32_t* dpart = dalloc<uint32_t>(slots.size() * 256);
        slot_checksum_kernel<<<(unsigned)slots.size(), 256>>>(dpool1, slot_bytes, dslots, dpart);
        sync();
        const auto part = fetch(dpart, slots.size() * 256);
        for (size_t k = 0; k < slots.size(); ++k) {
            uint32_t sum = 0; for (int t = 0; t < 256; ++t) sum += part[k * 256 + t];
            CHECK(sum == host_checksum(fx.blob[k]));
        }
        int32_t* dout_l = dalloc<int32_t>(ids.size());
        table_lookup_kernel<<<((int)ids.size() + 63) / 64, 64>>>(dtab1, dids, (int)ids.size(), Fixture::NE, dout_l);
        sync();
        const auto look = fetch(dout_l, ids.size());
        for (size_t i = 0; i < ids.size(); ++i) CHECK(look[i] == ((ids[i] >= 0 && ids[i] < Fixture::NE) ? c1.slot_of(1, ids[i]) : -1));
        cudaFree(dslots); cudaFree(dpart); cudaFree(dout_l);
    }
    // 5) a slot stride below one blob is refused (the arena slots are never smaller than the blob)
    CHECK(moe_hit_grouped_s2_multi(dpool1, slot_bytes - 1, dx, hit_slot, hit_dst, n_tok, K, dw, dx, scratch, 0) != cudaSuccess);

    std::printf("checks: %ld, failures: %ld\n", g_checks, g_fail);
    if (g_fail) { std::printf("U1DE GPU TESTS FAILED\n"); return 1; }
    std::printf("U1DE GPU TESTS PASSED\n");
    return 0;
}
