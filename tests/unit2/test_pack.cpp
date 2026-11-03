#include <cstring>
#include <string>
#include <vector>
#include "baloon/strata_pack_builder.hpp"
#include "baloon/ggml_dequant_layouts.hpp"
#include "baloon/mtp_expert_quantizer.hpp"
#include "check.hpp"
using namespace baloon;
using namespace baloon::pack;
using testh::Rng;

void test_pack_codes() {
    CHECK(per_byte(2) == 4 && per_byte(4) == 2 && per_byte(8) == 1);
    Rng r(40);
    for (unsigned bits : {2u, 4u, 8u}) {
        const unsigned pb = per_byte(bits), mask = (1u << bits) - 1u;
        for (size_t n : {size_t(0), size_t(pb), size_t(4 * pb), size_t(64), size_t(1000 * pb)}) {
            std::vector<uint8_t> codes(n);
            for (auto& c : codes) c = (uint8_t)r.below(mask + 1);
            auto p = pack_codes(codes.data(), n, bits);
            CHECK(p.has_value() && p->size() == n / pb);
            if (!p) continue;
            for (size_t i = 0; i < n; ++i) CHECK(((*p)[i / pb] >> (bits * (i % pb)) & mask) == codes[i]);  // first code in low bits
            auto u = unpack_codes(p->data(), n, bits);
            CHECK(u.has_value() && *u == codes);
        }
    }
    // explicit byte values
    const uint8_t c2[4] = {1, 2, 3, 0}; auto p2 = pack_codes(c2, 4, 2);
    CHECK(p2 && (*p2)[0] == (1 + 4 * 2 + 16 * 3 + 64 * 0));
    const uint8_t c4[2] = {0xA, 0x5}; auto p4 = pack_codes(c4, 2, 4);
    CHECK(p4 && (*p4)[0] == 0x5A);
    for (int a = 0; a < 4; ++a) for (int b = 0; b < 4; ++b) for (int c = 0; c < 4; ++c) for (int d = 0; d < 4; ++d) {
        const uint8_t cc[4] = {(uint8_t)a, (uint8_t)b, (uint8_t)c, (uint8_t)d};
        auto p = pack_codes(cc, 4, 2);
        CHECK(p && (*p)[0] == a + 4 * b + 16 * c + 64 * d);
    }
    // rejections
    const uint8_t big[4] = {4, 0, 0, 0};
    CHECK(!pack_codes(big, 4, 2));       // code does not fit
    CHECK(!pack_codes(c2, 3, 2));        // n not a multiple of codes-per-byte
    CHECK(!pack_codes(c2, 4, 3));        // unsupported width
    CHECK(!unpack_codes(c2, 3, 4) && !unpack_codes(c2, 4, 5));
}

void test_pack_alignment() {
    for (uint64_t n = 0; n < 5000; ++n) { CHECK((n + pad64(n)) % 64 == 0); CHECK(pad64(n) < 64); }
    Rng r(41);
    PlaneWriter w;
    std::vector<std::vector<uint8_t>> planes; std::vector<size_t> offs;
    for (int i = 0; i < 40; ++i) {
        std::vector<uint8_t> p(r.below(300)); for (auto& b : p) b = (uint8_t)r.u32();
        offs.push_back(w.append_plane(p));
        planes.push_back(p);
    }
    for (size_t i = 0; i < planes.size(); ++i) {
        CHECK(offs[i] % 64 == 0);
        CHECK(std::memcmp(w.bytes().data() + offs[i], planes[i].data(), planes[i].size()) == 0);
        if (i + 1 < planes.size()) CHECK(offs[i + 1] >= offs[i] + planes[i].size() && offs[i + 1] - (offs[i] + planes[i].size()) < 64);
    }
}

void test_pack_blob_arithmetic() {
    CHECK(kGuCodes + kGuScales + kDCodes + kDScales == 1382400);
    CHECK(kBlobBytes == 1382400 && 3 * (kH * kFF / kQK * 18) == 1382400);
    CHECK(kGuCodes == 819200 && kGuScales == 102400 && kDCodes == 409600 && kDScales == 51200);
    CHECK(kScGu == 40 && kScD == 10 && kFF * kScGu == kH * kScD && kFF * kScGu == kH * kFF / kQK);
    for (uint64_t blk = 0; blk < 2000; ++blk) {
        CHECK((blk / 40) * 40 + blk % 40 == blk && blk % 40 < 40);
        CHECK((blk / 10) * 10 + blk % 10 == blk && blk % 10 < 10);
    }
    for (uint64_t rr = 0; rr < kFF; ++rr) CHECK(2 * rr != 2 * rr + 1 && 2 * rr + 1 < 2 * kFF);
    // per-expert quantised bytes of the MTP quantiser equal one blob
    CHECK(mtpq::total_bytes(64, 18) / mtpq::E == kBlobBytes);
    CHECK(mtpq::E * mtpq::GU_OUT * mtpq::row_bytes(mtpq::HID, 64, 18) / mtpq::E + mtpq::HID * mtpq::row_bytes(mtpq::INTER, 64, 18) == kBlobBytes);
}

static std::vector<uint8_t> rand_role(Rng& r, uint8_t fill = 0, bool use_fill = false) {
    std::vector<uint8_t> v((size_t)kRoleBytes);
    for (auto& b : v) b = use_fill ? fill : (uint8_t)r.u32();
    return v;
}

void test_pack_expert_blob_roles() {
    Rng r(42);
    auto gate = rand_role(r), up = rand_role(r), down = rand_role(r);
    std::vector<uint8_t> blob;
    CHECK(build_expert_blob(gate, up, down, &blob) && blob.size() == kBlobBytes);
    std::vector<uint8_t> g, u, d;
    CHECK(rebuild_role(blob.data(), blob.size(), Role::Gate, &g) && g == gate);
    CHECK(rebuild_role(blob.data(), blob.size(), Role::Up, &u) && u == up);
    CHECK(rebuild_role(blob.data(), blob.size(), Role::Down, &d) && d == down);
    CHECK(verify_expert_blob(blob, gate, up, down));
    // explicit slot placement: gate row r block b -> slot 2r, up -> slot 2r+1 (codes at 16/block, scales at 2/block)
    for (int it = 0; it < 3000; ++it) {
        const uint64_t row = r.below((uint32_t)kFF), b = r.below((uint32_t)kScGu);
        const uint64_t blk = row * kScGu + b;
        const uint64_t sg = 2 * row * kScGu + b, su = (2 * row + 1) * kScGu + b;
        CHECK(std::memcmp(blob.data() + kGuCodesOff + sg * 16, gate.data() + blk * 18 + 2, 16) == 0);
        CHECK(std::memcmp(blob.data() + kGuScalesOff + sg * 2, gate.data() + blk * 18, 2) == 0);
        CHECK(std::memcmp(blob.data() + kGuCodesOff + su * 16, up.data() + blk * 18 + 2, 16) == 0);
        CHECK(std::memcmp(blob.data() + kGuScalesOff + su * 2, up.data() + blk * 18, 2) == 0);
        const uint64_t drow = r.below((uint32_t)kH), db = r.below((uint32_t)kScD), dblk = drow * kScD + db;
        CHECK(std::memcmp(blob.data() + kDCodesOff + dblk * 16, down.data() + dblk * 18 + 2, 16) == 0);
        CHECK(std::memcmp(blob.data() + kDScalesOff + dblk * 2, down.data() + dblk * 18, 2) == 0);
    }
    // every blob byte is used exactly once: fill roles with 1/2/3 and count
    std::vector<uint8_t> b2;
    CHECK(build_expert_blob(rand_role(r, 1, true), rand_role(r, 2, true), rand_role(r, 3, true), &b2));
    uint64_t cnt[4] = {0, 0, 0, 0};
    for (uint8_t b : b2) { CHECK(b >= 1 && b <= 3); ++cnt[b]; }
    CHECK(cnt[1] == kRoleBytes && cnt[2] == kRoleBytes && cnt[3] == kRoleBytes);
    for (uint64_t s = 0; s < 2 * kFF; ++s) CHECK(b2[kGuCodesOff + s * kScGu * 16] == (s % 2 == 0 ? 1 : 2));  // interleave
    // corruption anywhere is detected by verification
    for (int it = 0; it < 200; ++it) {
        std::vector<uint8_t> bad = blob;
        bad[r.below((uint32_t)kBlobBytes)] ^= (uint8_t)(1u << r.below(8));
        CHECK(!verify_expert_blob(bad, gate, up, down));
    }
    // size guards
    CHECK(!rebuild_role(blob.data(), blob.size() - 1, Role::Gate, &g));
    std::vector<uint8_t> shortrole(kRoleBytes - 1);
    CHECK(!build_expert_blob(shortrole, up, down, &blob));
}

void test_pack_quantised_role_end_to_end() {
    // quantise a role, pack the blob, rebuild, dequantise through the unit-1 Q2_0 dequantiser: identical values
    Rng r(43);
    std::vector<float> w((size_t)(kFF * kH));
    for (auto& x : w) x = (float)r.gauss();
    std::vector<uint8_t> gate, up, down;
    CHECK(mtpq::quantize_tensor(mtpq::Format::Q2_0, w.data(), kFF, kH, &gate) && gate.size() == kRoleBytes);
    for (auto& x : w) x = (float)(r.gauss() * 0.1);
    CHECK(mtpq::quantize_tensor(mtpq::Format::Q2_0, w.data(), kFF, kH, &up));
    CHECK(mtpq::quantize_tensor(mtpq::Format::Q2_0, w.data(), kH, kFF, &down) && down.size() == kRoleBytes);
    std::vector<uint8_t> blob, g2;
    CHECK(build_expert_blob(gate, up, down, &blob));
    CHECK(rebuild_role(blob.data(), blob.size(), Role::Gate, &g2));
    for (uint64_t blk = 0; blk < kRoleBlocks; blk += 97) {
        float a[64], b[64];
        dequant::dequantize_q2_0(gate.data() + blk * 18, a);
        dequant::dequantize_q2_0(g2.data() + blk * 18, b);
        CHECK(std::memcmp(a, b, sizeof a) == 0);
    }
}

void test_pack_scale_dtype_and_partition() {
    CHECK(scale_bytes(true) == 2 && scale_bytes(false) == 4);
    for (uint64_t n : {0ull, 1ull, 7ull, 1000ull}) CHECK(n * scale_bytes(true) * 2 == n * scale_bytes(false));
    Rng r(44);
    std::vector<float> exact(500);
    for (auto& x : exact) { uint16_t h = (uint16_t)r.u32(); h = (uint16_t)((h & 0x7BFF) | (h & 0x8000)); x = f16::half_to_float(h); }
    bool fp16 = false;
    auto e = encode_scales(exact.data(), exact.size(), &fp16);
    CHECK(fp16 && e.size() == exact.size() * 2);
    for (size_t i = 0; i < exact.size(); ++i) CHECK(f16::half_to_float((uint16_t)(e[2 * i] | (e[2 * i + 1] << 8))) == exact[i]);
    std::vector<float> inexact = exact; inexact[123] = 1.0000001f;
    auto f = encode_scales(inexact.data(), inexact.size(), &fp16);
    CHECK(!fp16 && f.size() == inexact.size() * 4);
    for (size_t i = 0; i < inexact.size(); ++i) {
        uint32_t b = 0; for (int k = 0; k < 4; ++k) b |= (uint32_t)f[4 * i + k] << (8 * k);
        CHECK(b == f16::f32_bits(inexact[i]));
    }
    CHECK(choose_scale_dtype_fp16(exact.data(), 0));  // vacuous: no scales
    const float big = 1e10f; CHECK(!choose_scale_dtype_fp16(&big, 1));  // overflows fp16

    const std::vector<std::string> names = {
        "model.embed_tokens.weight", "model.layers.0.self_attn.q_proj.weight", "model.layers.0.mlp.experts.gate_up_proj",
        "model.layers.0.mlp.experts.down_proj", "token_embd.weight", "blk.3.ffn_gate_exps.weight", "output.weight", "norm.weight"};
    const Partition p = partition_tensors(names);
    CHECK(p.expert.size() == 3 && p.embd.size() == 2 && p.dense.size() == 3);
    CHECK(p.dense.size() + p.embd.size() + p.expert.size() == names.size());
    std::vector<int> seen(names.size(), 0);
    for (size_t i : p.dense) ++seen[i];
    for (size_t i : p.embd) ++seen[i];
    for (size_t i : p.expert) ++seen[i];
    for (int s : seen) CHECK(s == 1);  // exactly one destination each
    for (size_t i : p.expert) CHECK(is_expert(names[i]));
    for (size_t i : p.embd) CHECK(!is_expert(names[i]) && is_embd(names[i]));
    CHECK(shapes_equal(5, 5) && !shapes_equal(5, 6));
}
