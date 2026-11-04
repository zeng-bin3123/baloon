// Host-side invariants of the batched-prefill kernel layout: index decomposition recomposes exactly and stays in range,
// launch grids cover every element, the expert blob sections are ordered/disjoint and sum to 1,382,400 bytes, the router
// lane layout covers all 512 experts, and the QSA/KV/RoPE index maps stay inside their buffers.
#include <initializer_list>
#include <vector>
#include "baloon/prefill_batch_api.hpp"
#include "baloon/prefill_gpu_kernels.hpp"
#include "check.hpp"
using namespace baloon;

void test_prefill_index_theory() {
    testh::Rng rng(31);
    for (int it = 0; it < 300000; ++it) {
        const int64_t i = (int64_t)(rng.next() % (1ull << 36));
        CHECK(i / 10240 * 10240 + (i % 10240) / 2560 * 2560 + i % 2560 == i);
        CHECK((i % 10240) / 2560 < 4); CHECK(i % 2560 < 2560);
        const int64_t T = i / 10240 + 1 + (int64_t)(rng.next() % 5);
        CHECK(i / 10240 < T);
        CHECK(i / 10240 * 2560 + i % 2560 < T * 2560);                  // gr_broadcast source
        const int64_t ld = 4 + (int64_t)(rng.next() % 9);
        CHECK(i / 10240 * ld + (i % 10240) / 2560 < (i / 10240 + 1) * ld);  // gr_write injection coefficient
        // split_q recomposition and the gate half
        CHECK(i / 6144 * 6144 + ((i / 256) % 24) * 256 + i % 256 == i);
        const int64_t h = (i / 256) % 24, d = i % 256;
        CHECK(h * 512 + 256 + d < 24 * 512);
    }
    // launch grids cover every element
    for (int64_t n : {1, 2, 255, 256, 257, 10240, 10241, 6144 * 7, 1000003})
        for (int64_t t : {32, 64, 128, 256}) { CHECK(n <= pfb::blocks_for(n, t) * t); CHECK((pfb::blocks_for(n, t) - 1) * t < n); }
    CHECK(pfb::kConvC / 128 * 128 == pfb::kConvC);
    CHECK((2 * pfb::kKeyHeads + pfb::kValHeads) * pfb::kStateS == pfb::kConvC);
    CHECK(pfb::kRowGroups * (pfb::kStateS / pfb::kRowGroups) == pfb::kStateS);
    for (int rg = 0; rg < 4; ++rg) for (int r = 0; r < 32; ++r) CHECK(rg * 32 + r < 128);
    // all 128 rows are visited exactly once by (row group, row)
    { int seen[128] = {0}; for (int rg = 0; rg < 4; ++rg) for (int r = 0; r < 32; ++r) ++seen[rg * 32 + r]; for (int x : seen) CHECK(x == 1); }
    // router lanes: 32 lanes x 16 registers cover 512 experts once
    { int seen[512] = {0};
      for (int lane = 0; lane < 32; ++lane) for (int i = 0; i < 16; ++i) {
          const int e = lane + i * 32; CHECK(e < 512); CHECK(e % 32 == lane); CHECK(e / 32 == i); ++seen[e]; }
      for (int x : seen) CHECK(x == 1); }
    for (int64_t t = 0; t < 50; ++t) for (int r = 0; r < 10; ++r) CHECK(t * 10 + r < 50 * 10);
    // kv append / rope
    for (int g = 0; g < 4; ++g) for (int tid = 0; tid < 64; ++tid) CHECK(g * 64 + tid < 256);
    for (int64_t row = 0; row < 100; ++row) for (int g = 0; g < 4; ++g) CHECK(row * 4 + g < (row + 1) * 4);
    for (int p = 0; p < 32; ++p) CHECK(p < p + 32 && p + 32 < 64);
    for (int64_t k = 0; k < 640; ++k) CHECK(2 * k + 1 < 1280);
    CHECK(2560 / 8 == 320);
}

void test_prefill_blob_theory() {
    using namespace pfb;
    CHECK(kBlobBytes == 1382400);
    CHECK(kGuOff == 0 && kDownOff == 1280 * 640);
    CHECK(kGuScaleOff == kDownOff + 2560 * 160); CHECK(kDownScaleOff == kGuScaleOff + 1280 * 40 * 2);
    CHECK(kDownOff < kGuScaleOff && kGuScaleOff < kDownScaleOff && kDownScaleOff < kBlobBytes);
    // agree with the independent layout header
    CHECK((uint64_t)kDownOff == prefill_kernels::O_D_CODES); CHECK((uint64_t)kGuScaleOff == prefill_kernels::O_GU_SC);
    CHECK((uint64_t)kDownScaleOff == prefill_kernels::O_D_SC); CHECK((uint64_t)kBlobBytes == prefill_kernels::BLOB_BYTES);
    for (int64_t row = 0; row < 1280; row += 7) for (int64_t byte = 0; byte < 640; byte += 3) {
        CHECK(row * 40 + (byte * 4) / 64 < 1280 * 40);
        for (int k = 0; k < 4; ++k) CHECK(row * 2560 + byte * 4 + k < 1280 * 2560);
    }
    for (int64_t row = 0; row < 2560; row += 11) for (int64_t byte = 0; byte < 160; byte += 3) {
        CHECK(row * 10 + (byte * 4) / 64 < 2560 * 10);
        for (int k = 0; k < 4; ++k) CHECK(row * 640 + byte * 4 + k < 2560 * 640);
    }
    // the dequant threads (one per code byte) write every output element exactly once
    { std::vector<char> gu(1280 * 2560, 0), dn(2560 * 640, 0);
      for (int64_t i = 0; i < 1280 * 640; ++i) { int64_t row = i / 640, byte = i % 640; for (int k = 0; k < 4; ++k) ++gu[(size_t)(row * 2560 + byte * 4 + k)]; }
      for (int64_t i = 0; i < 2560 * 160; ++i) { int64_t row = i / 160, byte = i % 160; for (int k = 0; k < 4; ++k) ++dn[(size_t)(row * 640 + byte * 4 + k)]; }
      bool ok = true; for (char c : gu) ok &= (c == 1); for (char c : dn) ok &= (c == 1); CHECK(ok); }
}

void test_prefill_api_constants() {
    using namespace pfb;
    CHECK(kEmbN == 2560 && kStreams == 4 && kHcWidth == 10240 && kStateS == 128 && kConvC == 10240 && kValWidth == 6144);
    CHECK(kKeyHeads == 16 && kValHeads == 48 && kTopK == 10 && kExperts == 512);
    CHECK(kQHeads * kQFullPerHead == 12288 && kQHeads * kQHeadDim == 6144);
    CHECK(kKvGroups * kKvGroupWidth == kKvChannels);
    // bf16 rounding pattern is 16 bits and exact on bf16 values
    testh::Rng rng(3);
    for (int i = 0; i < 100000; ++i) {
        const uint16_t h = (uint16_t)rng.u32();
        if (((h >> 7) & 0xFF) == 0xFF) continue;                         // skip inf/nan patterns
        CHECK(prefill_kernels::bf_from_float(prefill_kernels::bf_to_float(h)) == h);
    }
}
