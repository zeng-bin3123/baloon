// Direct checks of the LAYER-1 statements for gguf_lazy_loader and strata_pack_builder that the other tests
// only cover indirectly: exact byte totals, Python-form equality, arena addressing, absolute alignment.
#include <cstdint>
#include <random>
#include <vector>
#include "baloon/gguf_lazy_loader.hpp"
#include "baloon/strata_pack_builder.hpp"
#include "check.hpp"
using namespace baloon;

void test_theory_lazy_loader_facts() {
    CHECK(gguf::expert_bytes(2560ull * 640, 64, 18) == 460800ull);
    CHECK((2560ull * 640) % 64 == 0);
    CHECK(512ull * 460800ull == 235929600ull);
    CHECK(gguf::tensor_bytes(512ull * 2560 * 640, 64, 18) == 235929600ull);
    CHECK(gguf::ple_row_bytes() == 90 && 160 / 32 * 18 == 90);
    CHECK(gguf::ple_table_bytes() == 28800138240ull);
    CHECK(26ull * 1073741824ull < gguf::ple_table_bytes() && gguf::ple_table_bytes() < 27ull * 1073741824ull);
    CHECK(gguf::ple_fp32_bytes() == 204800983040ull);
    CHECK(gguf::ple_token_floats() == 2560);
    std::mt19937_64 rng(77);
    for (int i = 0; i < 200000; ++i) {
        const uint64_t el = 1 + rng() % 300, by = 1 + rng() % 300, per = rng() % 100000, e = rng() % 1000, n = e + 1 + rng() % 50;
        CHECK(gguf::expert_start_py(e, per, el, by) == gguf::expert_start(e, per, el, by));
        const uint64_t len = gguf::expert_bytes(per, el, by);
        CHECK(gguf::expert_start(e, per, el, by) + len == gguf::expert_start(e + 1, per, el, by));   // adjacent
        CHECK(gguf::expert_start(e, per, el, by) + len <= n * len);                                   // in bounds
        const uint64_t e2 = e + 1 + rng() % 50;
        CHECK(gguf::expert_start(e, per, el, by) + len <= gguf::expert_start(e2, per, el, by));       // disjoint
        const uint64_t blocks = rng() % 1000;
        CHECK(gguf::tensor_bytes(n * blocks * el, el, by) == n * gguf::expert_bytes(blocks * el, el, by));  // el | per
        const uint16_t h = (uint16_t)rng();
        CHECK(gguf::bf16_widen_bits(h) % 65536 == 0 && gguf::bf16_widen_bits(h) / 65536 == h);
        const int64_t idx = (int64_t)(rng() % 200) - 100, nn = (int64_t)(rng() % 100);
        CHECK((0 <= idx && idx < nn) == (idx >= 0 && gguf::row_in_range((uint64_t)idx, (uint64_t)nn)));
    }
}

void test_theory_image_size() {
    const uint8_t junk[4] = {0, 0, 0, 0};
    gguf::LazyImage li(junk, sizeof junk);
    CHECK(!li.ok() && li.size() == sizeof junk);
}

void test_theory_pack_facts() {
    CHECK(8 / 2 == 4 && 8 / 4 == 2 && 8 / 8 == 1 && pack::per_byte(2) == 4 && pack::per_byte(4) == 2 && pack::per_byte(8) == 1);
    CHECK(pack::kRoleBlocks == 25600 && pack::kRoleBlocks * 18 == 460800 && 3 * 460800 == 1382400);
    for (uint64_t k = 0; k < 5000; ++k) CHECK(k * pack::kBlobBytes % 64 == 0);
    CHECK(pack::expert_flat_index(7, 511) < pack::expert_flat_index(8, 0) && pack::expert_flat_index(8, 0) == 8 * 512);
    CHECK(pack::expert_blob_offset(3, 5) == (3 * 512 + 5) * 1382400ull && pack::expert_blob_offset(3, 5) % 64 == 0);
    CHECK(pack::experts_checked(512, 32) == 16 && pack::role_checks(512, 32) == 48 && pack::experts_checked(512, 1) == 512);
    CHECK(pack::experts_checked(5, 0) == 0);
    for (uint64_t blk = 0; blk < 25600; ++blk) {
        CHECK(blk / 10 < 2560 && blk % 10 < 10 && blk / 10 * 10 + blk % 10 == blk);
        CHECK(blk / 40 < 640 && blk % 40 < 40 && blk / 40 * 40 + blk % 40 == blk);
    }
    // absolute alignment with a 64-aligned base; two codes per byte stay below 256
    std::mt19937_64 rng(5);
    for (int i = 0; i < 100000; ++i) {
        const uint64_t base = 64 * (rng() % 100000), len = rng() % 100000;
        CHECK((base + len + pack::pad64(len)) % 64 == 0 && (len + pack::pad64(len)) % 64 == 0);
    }
    for (unsigned a = 0; a < 16; ++a) for (unsigned b = 0; b < 16; ++b) {
        const uint8_t c[2] = {(uint8_t)a, (uint8_t)b};
        auto p = pack::pack_codes(c, 2, 4);
        CHECK(p && p->size() == 1 && (*p)[0] == a + 16 * b);
    }
}
