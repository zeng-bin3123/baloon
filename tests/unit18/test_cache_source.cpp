// ExpertCache + ExpertSource invariants (theory modules ExpertCache.lean / ExpertSource.lean).
#include <cstdio>
#include <cstring>
#include <string>
#include <unistd.h>
#include <vector>
#include "baloon/core_expert_cache.hpp"
#include "baloon/core_expert_source.hpp"
#include "check.hpp"

using namespace baloon::core;

void test_cache_ranges() {
    // layer ranges tile [0, slots): adjacent, ordered, in slots, last layer takes the remainder and >= quota.
    for (int64_t nl = 1; nl <= 9; ++nl)
        for (int64_t slots = nl; slots <= 60; ++slots) {
            ExpertCache c;
            CHECK(c.open(nl, 5, slots, 16));
            int64_t prev_hi = 0;
            for (int64_t l = 0; l < nl; ++l) {
                auto r = c.layer_slot_range(l);
                CHECK(r.first == prev_hi);                                   // no gap, no overlap
                CHECK(r.first <= r.second && r.second <= slots);
                if (l == 0) CHECK(r.first == 0);
                if (l == nl - 1) { CHECK(r.second == slots); CHECK(r.second - r.first >= slots / nl); }
                prev_hi = r.second;
            }
            CHECK((c.layer_slot_range(-1) == std::make_pair<int64_t, int64_t>(0, 0)));
            CHECK((c.layer_slot_range(nl) == std::make_pair<int64_t, int64_t>(0, 0)));
        }
    // flat index recovers (layer, expert); out-of-range queries are "not resident"
    ExpertCache c;
    CHECK(c.open(4, 7, 8, 32));
    CHECK(c.slot_of(0, 7) == kNotResident && c.slot_of(4, 0) == kNotResident && c.slot_of(-1, 0) == kNotResident);
    CHECK(!c.open(0, 7, 8, 32) && !c.open(4, 7, 0, 32) && !c.open(4, 7, 8, 0));
    // open_sized: offsets 256-aligned, back to back, one more entry than slots
    ExpertCache s;
    CHECK(s.open_sized(2, 4, {100, 256, 257, 1}));
    CHECK(s.arena_bytes() == 256 + 256 + 512 + 256);
    CHECK(s.device_slot(0) + 256 == s.device_slot(1));
    CHECK(s.device_slot(1) + 256 == s.device_slot(2));
    CHECK(s.device_slot(2) + 512 == s.device_slot(3));
    for (int i = 0; i < 4; ++i) CHECK((s.device_slot(i) - s.device_slot(0)) % 256 == 0);
    CHECK(s.device_slot(4) == nullptr && s.device_slot(-1) == nullptr);
    CHECK(!s.open_sized(2, 4, {}) && !s.open_sized(2, 4, {8, 0}));
}

void test_cache_admit_fill() {
    ExpertCache c;
    CHECK(c.open(3, 10, 10, 8));                       // quota 3, last layer gets 4
    // per-layer admission: never leaves [lo,hi), never evicts, idempotent for resident pairs
    for (int64_t l = 0; l < 3; ++l) {
        auto r = c.layer_slot_range(l);
        const int64_t cap = r.second - r.first;
        for (int64_t e = 0; e < 10; ++e) {
            int32_t s = c.admit(l, e);
            if (e < cap) { CHECK(s >= r.first && s < r.second); CHECK(c.slot_of(l, e) == s); CHECK(c.admit(l, e) == s); }
            else CHECK(s == kNotResident);               // full layer admits nothing
        }
    }
    CHECK(c.resident_count() == 10);
    CHECK(c.admit(0, 99) == kNotResident && c.admit(9, 0) == kNotResident);
    // earlier admissions still resident after the layer filled (no eviction)
    CHECK(c.slot_of(0, 0) == 0 && c.slot_of(0, 1) == 1 && c.slot_of(0, 2) == 2 && c.slot_of(0, 3) == kNotResident);
    // global admission hands out each slot below `slots` once
    ExpertCache g;
    CHECK(g.open(2, 6, 5, 8, false));
    int ok = 0;
    for (int64_t l = 0; l < 2; ++l) for (int64_t e = 0; e < 6; ++e) { int32_t s = g.admit(l, e); if (s != kNotResident) { CHECK(s < 5); ++ok; } }
    CHECK(ok == 5);
    // fill: never exceeds the blob, never empty, whole blob when bytes is 0 / too large
    CHECK(fill_len(0, 8) == 8 && fill_len(-3, 8) == 8 && fill_len(9, 8) == 8 && fill_len(5, 8) == 5 && fill_len(8, 8) == 8);
    uint8_t src[32];
    for (int i = 0; i < 32; ++i) src[i] = (uint8_t)(i + 1);
    CHECK(c.fill_slot(2, src, 5));
    CHECK(c.pending_fills() == 1);
    CHECK(c.verify_slot(2, src, 5));
    uint8_t* d = c.device_slot(2);
    CHECK(std::memcmp(d, src, 5) == 0 && d[5] == 0);
    uint8_t bad[8] = {1, 2, 3, 4, 5, 9, 9, 9};
    CHECK(c.fill_slot_blocking(3, src, 99));          // 99 > blob: whole slot, never beyond it
    CHECK(c.pending_fills() == 0);
    CHECK(std::memcmp(c.device_slot(3), src, 8) == 0);
    CHECK(c.device_slot(4)[0] == 0);                  // neighbour untouched
    CHECK(!c.verify_slot(3, bad, 8));
    CHECK(!c.fill_slot(10, src, 1) && !c.fill_slot(-1, src, 1) && !c.fill_slot(0, nullptr, 1));
    c.close();
    CHECK(!c.is_open() && c.slot_of(0, 0) == kNotResident && c.admit(0, 0) == kNotResident);
}

void test_profile() {
    std::vector<std::pair<int32_t, int32_t>> in = {{0, 1}, {2, 3}, {1, 0}}, out;
    auto b = encode_expert_profile(3, 4, in);
    CHECK(parse_expert_profile(b.data(), b.size(), 3, 4, 3, out) && out == in);
    CHECK(parse_expert_profile(b.data(), b.size(), 3, 4, 10, out) && out.size() == 3);
    CHECK(!parse_expert_profile(b.data(), b.size(), 3, 4, 2, out));        // ranks more pairs than slots
    CHECK(!parse_expert_profile(b.data(), b.size(), 4, 4, 3, out));        // layer count mismatch
    CHECK(!parse_expert_profile(b.data(), b.size(), 3, 5, 3, out));        // expert count mismatch
    CHECK(!parse_expert_profile(b.data(), b.size() - 1, 3, 4, 3, out));    // truncated
    auto bad = encode_expert_profile(3, 4, {{0, 1}, {3, 0}});              // layer 3 out of range
    CHECK(!parse_expert_profile(bad.data(), bad.size(), 3, 4, 8, out));
    auto bad2 = encode_expert_profile(3, 4, {{0, 4}});                     // expert 4 out of range
    CHECK(!parse_expert_profile(bad2.data(), bad2.size(), 3, 4, 8, out));
    auto neg = encode_expert_profile(3, 4, {{-1, 0}});
    CHECK(!parse_expert_profile(neg.data(), neg.size(), 3, 4, 8, out));
    CHECK(!parse_expert_profile(b.data(), 5, 3, 4, 3, out));
    // file round trip
    std::string path = "/tmp/baloon_u18_profile_" + std::to_string(getpid()) + ".bin";
    FILE* f = std::fopen(path.c_str(), "wb");
    CHECK(f != nullptr);
    if (f) { std::fwrite(b.data(), 1, b.size(), f); std::fclose(f); }
    CHECK(read_expert_profile(path, 3, 4, 3, out) && out == in);
    CHECK(!read_expert_profile(path + ".missing", 3, 4, 3, out));
    std::remove(path.c_str());
    // a ranked pair list that fits the slots can all be admitted (global mode)
    ExpertCache c;
    CHECK(c.open(3, 4, 3, 8, false));
    for (auto& p : in) CHECK(c.admit(p.first, p.second) != kNotResident);
}

static std::vector<uint8_t> make_pack(int64_t nl, int64_t ne, int64_t blob) {
    std::vector<uint8_t> v((size_t)(nl * ne * blob));
    for (int64_t l = 0; l < nl; ++l) for (int64_t e = 0; e < ne; ++e)
        for (int64_t i = 0; i < blob; ++i) v[(size_t)((l * ne + e) * blob + i)] = (uint8_t)(l * 31 + e * 7 + i);
    return v;
}

void test_source_blobs() {
    const int64_t NL = 3, NE = 5, B = 12;
    auto pack = make_pack(NL, NE, B);
    std::string path = "/tmp/baloon_u18_experts_" + std::to_string(getpid()) + ".bin";
    FILE* f = std::fopen(path.c_str(), "wb");
    CHECK(f != nullptr);
    if (f) { std::fwrite(pack.data(), 1, pack.size(), f); std::fclose(f); }
    // artifact size theorems
    CHECK(48ull * 512 * 1382400 == 33973862400ull && 48 * 512 == 24576);
    FileExpertSource fs;
    CHECK(!fs.open(path, NL, NE, B + 1));                       // wrong size: exact size or nothing
    CHECK(!fs.open(path + ".nope", NL, NE, B));
    CHECK(fs.open(path, NL, NE, B));
    CHECK(fs.file_bytes() == pack.size());
    for (int64_t l = 0; l < NL; ++l) for (int64_t e = 0; e < NE; ++e) {
        const uint8_t* p = fs.blob(l, e);
        CHECK(p != nullptr && std::memcmp(p, pack.data() + (l * NE + e) * B, (size_t)B) == 0);
    }
    CHECK(fs.reads() == (uint64_t)(NL * NE));                   // one read per accepted lookup
    // the expert axis is validated separately: (0, NE) must NOT alias (1, 0)
    CHECK(fs.blob(0, NE) == nullptr && fs.blob(1, -1) == nullptr && fs.blob(NL, 0) == nullptr && fs.blob(-1, 0) == nullptr);
    CHECK(fs.reads() == (uint64_t)(NL * NE));                   // rejected lookups do not count
    fs.close();
    CHECK(fs.blob(0, 0) == nullptr);
    // distinct pairs -> distinct blobs
    ArenaExpertSource as;
    CHECK(as.open(path, NL, NE, B, 40));
    CHECK(as.arena_bytes() == pack.size() + 40);                // one max-blob longer than the file
    std::vector<const uint8_t*> seen;
    for (int64_t l = 0; l < NL; ++l) for (int64_t e = 0; e < NE; ++e) {
        const uint8_t* p = as.blob(l, e);
        CHECK(p && std::memcmp(p, pack.data() + (l * NE + e) * B, (size_t)B) == 0);
        for (auto q : seen) CHECK(q != p);
        seen.push_back(p);
        // a slot-sized copy (40 bytes) from ANY expert stays inside the arena
        CHECK(p + 40 <= as.device_alias() + as.arena_bytes());
    }
    CHECK(as.blob(0, NE) == nullptr);
    CHECK(as.device_alias() != nullptr);
    as.close();
    CHECK(as.device_alias() == nullptr && as.blob(0, 0) == nullptr);
    CHECK(!as.open_bytes(pack.data(), pack.size() - 1, NL, NE, B));
    CHECK(!as.open_bytes(nullptr, 0, NL, NE, B));
    std::remove(path.c_str());
}

namespace {
struct Rec : GpuPlanSink {
    std::vector<size_t> jobs, gpu;
    std::vector<int64_t> layers;
    void plan(int64_t layer, const ExpertDispatch& d) override { layers.push_back(layer); jobs.push_back(d.njobs()); gpu.push_back(d.n_gpu()); }
};
}

void test_dispatch() {
    testh::Rng rng(7);
    const int64_t NL = 2, NE = 16, B = 8;
    auto pack = make_pack(NL, NE, B);
    ArenaExpertSource src;
    CHECK(src.open_bytes(pack.data(), pack.size(), NL, NE, B));
    for (int iter = 0; iter < 4000; ++iter) {
        const int k = 1 + (int)rng.below(10);
        std::vector<bool> hit((size_t)k);
        std::vector<int32_t> ex((size_t)k);
        for (int i = 0; i < k; ++i) { hit[(size_t)i] = rng.below(2); ex[(size_t)i] = (int32_t)rng.below(NE); }
        const int num = (int)rng.below(257);
        ExpertDispatch d;
        CHECK(expert_hit_run(hit, num, d, ex.data()));
        size_t nh = 0, nm = 0;
        for (int i = 0; i < k; ++i) (hit[(size_t)i] ? nh : nm)++;
        const size_t m = (nm * (size_t)num) >> 8;
        CHECK(m <= nm);
        CHECK(d.gpu_hits.size() == nh && d.gpu_pcie.size() == m && d.jobs.size() == nm - m);
        CHECK(d.njobs() + d.n_gpu() == (size_t)k);              // exactly one engine computes each entry
        std::vector<int> cover((size_t)k, 0);
        for (int32_t i : d.gpu_hits) { ++cover[(size_t)i]; CHECK(hit[(size_t)i]); }
        for (auto& j : d.jobs) { ++cover[(size_t)j.out_index]; CHECK(!hit[(size_t)j.out_index]); CHECK(j.expert == ex[(size_t)j.out_index]); }
        for (auto& j : d.gpu_pcie) { ++cover[(size_t)j.out_index]; CHECK(!hit[(size_t)j.out_index]); }
        for (int c : cover) CHECK(c == 1);
        // PCIe entries are the LAST m misses in routing order
        if (m) {
            int32_t first_pcie = d.gpu_pcie.front().out_index;
            for (auto& j : d.jobs) CHECK(j.out_index < first_pcie);
            for (size_t a = 1; a < d.gpu_pcie.size(); ++a) CHECK(d.gpu_pcie[a - 1].out_index < d.gpu_pcie[a].out_index);
        }
        if (nh == 0) { CHECK(d.jobs.size() + d.gpu_pcie.size() == (size_t)k); for (size_t q = 0; q < d.jobs.size(); ++q) CHECK(d.jobs[q].out_index == (int32_t)q); }
        if (num == 0) CHECK(d.gpu_pcie.empty());
        if (num == 256) CHECK(d.jobs.empty());
    }
    ExpertDispatch d;
    CHECK(!expert_hit_run({true}, 257, d) && !expert_hit_run({true}, -1, d));
    CHECK(pcie_share_of(5, 128) == 2 && pcie_share_of(7, 0) == 0 && pcie_share_of(7, 256) == 7);
    // full dispatch: blobs resolved, plan published, reads counted once per accepted lookup
    Rec sink;
    std::vector<int32_t> ex = {3, 4, 5, 6};
    std::vector<bool> hit = {true, false, false, false};
    const uint64_t r0 = src.reads();
    CHECK(expert_pool_dispatch(src, &sink, 1, ex.data(), hit, 128, d));       // m = 3*128>>8 = 1
    CHECK(d.jobs.size() == 2 && d.gpu_pcie.size() == 1 && d.gpu_hits.size() == 1);
    for (auto& j : d.jobs) CHECK(j.blob == src.blob(1, j.expert));
    CHECK(d.gpu_pcie[0].expert == 6 && d.gpu_pcie[0].blob != nullptr);
    CHECK(sink.layers.size() == 1 && sink.layers[0] == 1 && sink.jobs[0] == 2 && sink.gpu[0] == 2);
    CHECK(src.reads() >= r0 + 3);
    // expert >= n_expert is rejected, the dispatch latches failure and nothing is published
    std::vector<int32_t> bad = {NE, 0};
    Rec sink2;
    CHECK(!expert_pool_dispatch(src, &sink2, 0, bad.data(), {false, false}, 0, d) && d.failed && sink2.layers.empty());
    CHECK(!expert_pool_dispatch(src, nullptr, 0, nullptr, {false}, 0, d));
    // multi-token dispatch
    std::vector<ExpertDispatch> outs;
    CHECK(expert_pool_dispatch_multi(src, &sink, 0, {{1, 2}, {3}}, {{false, true}, {false}}, 0, outs) && outs.size() == 2);
    CHECK(!expert_pool_dispatch_multi(src, &sink, 0, {{1}, {NE}, {2}}, {{false}, {false}, {false}}, 0, outs) && outs.size() == 2);
    CHECK(!expert_pool_dispatch_multi(src, &sink, 0, {{1}}, {}, 0, outs));
}

void test_gguf_load() {
    // the three regions tile the blob when down_off == 2*up_off
    for (int64_t up = 1; up <= 12; ++up)
        for (int64_t blob = 2 * up; blob <= 2 * up + 15; ++blob) {
            BlobRegions r = blob_regions(up, 2 * up, blob);
            CHECK(r.at[0] == 0 && r.at[0] + r.per[0] == r.at[1]);
            CHECK(r.at[1] + r.per[1] == r.at[2] && r.at[2] == 2 * up);
            CHECK(r.at[2] + r.per[2] == blob);
        }
    const int64_t NE = 37, UP = 5, BLOB = 17;           // 37 experts: chunks of 16, 16, 5
    std::vector<uint8_t> gate((size_t)(NE * UP)), up((size_t)(NE * UP)), down((size_t)(NE * (BLOB - 2 * UP)));
    for (size_t i = 0; i < gate.size(); ++i) { gate[i] = (uint8_t)(i * 3 + 1); up[i] = (uint8_t)(i * 5 + 2); }
    for (size_t i = 0; i < down.size(); ++i) down[i] = (uint8_t)(i * 7 + 3);
    std::vector<uint8_t> dst((size_t)(NE * BLOB), 0xEE);
    int64_t chunks = 0;
    CHECK(load_experts_gguf(gate.data(), up.data(), down.data(), NE, UP, 2 * UP, BLOB, dst.data(), &chunks));
    CHECK(chunks == 3);
    for (int64_t e = 0; e < NE; ++e) {
        CHECK(std::memcmp(&dst[(size_t)(e * BLOB)], &gate[(size_t)(e * UP)], UP) == 0);
        CHECK(std::memcmp(&dst[(size_t)(e * BLOB + UP)], &up[(size_t)(e * UP)], UP) == 0);
        CHECK(std::memcmp(&dst[(size_t)(e * BLOB + 2 * UP)], &down[(size_t)(e * (BLOB - 2 * UP))], (size_t)(BLOB - 2 * UP)) == 0);
    }
    CHECK(!load_experts_gguf(gate.data(), up.data(), down.data(), NE, UP, 2 * UP + 1, BLOB, dst.data()));
    CHECK(!load_experts_gguf(gate.data(), up.data(), down.data(), NE, UP, 2 * UP, 2 * UP - 1, dst.data()));
    CHECK(!load_experts_gguf(nullptr, up.data(), down.data(), NE, UP, 2 * UP, BLOB, dst.data()));
    for (int64_t n = 1; n <= 70; ++n) {
        int64_t c = 0;
        std::vector<uint8_t> g((size_t)(n * UP), 1), u((size_t)(n * UP), 2), d((size_t)(n * (BLOB - 2 * UP)), 3), o((size_t)(n * BLOB));
        CHECK(load_experts_gguf(g.data(), u.data(), d.data(), n, UP, 2 * UP, BLOB, o.data(), &c));
        CHECK(c == (n + 15) / 16);
    }
}
