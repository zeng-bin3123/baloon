// Unit 31 invariants: per-format traits agree with the layout and activation-sizing tables, and the backend selection is
// the cached, idempotent function of the CPU feature set.
#include <cstdio>
#include <thread>
#include <vector>

#include "baloon/cpu_native_traits.hpp"

using namespace baloon;
static int fails = 0;
#define CHECK(c) do { if (!(c)) { std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); ++fails; } } while (0)

int main() {
    const NativeFmt all[] = {NativeFmt::Q2_0, NativeFmt::IQ2_XXS, NativeFmt::IQ2_XS, NativeFmt::IQ3_XXS, NativeFmt::IQ3_S, NativeFmt::IQ2_S};
    for (NativeFmt f : all) {
        const NativeTraitsRow r = native_traits_of(f);
        CHECK(r.fmt == f);
        CHECK(r.type_id == native_type_id(f) && r.type_id != 0);
        CHECK(native_fmt(r.type_id) == f);                               // id <-> format round trip
        CHECK(r.block_values == expert_layout_ns::block_values(r.type_id));
        CHECK(r.block_bytes == expert_layout_ns::block_bytes(r.type_id));
        // Activation sizing agrees with the traits: n values of a whole number of act blocks.
        const uint32_t n = r.act_block * 4;
        const uint64_t one = native_act_bytes(f, n, 1);
        CHECK(one != 0 && native_act_bytes(f, n, 3) == 3 * one);        // nt tokens are nt back-to-back vectors (n % 4 == 0)
        CHECK(native_act_bytes(f, n + 1, 1) == 0);                       // not a whole number of blocks
        if (r.act_kind == NativeActKind::Q8K) CHECK(one == 4ull * kern::Q8KBlock{}.qs[0] + 4 * native_detail::kQ8KBytes);
        if (r.act_kind == NativeActKind::Q8_1x32) CHECK(one == (uint64_t)n + 8ull * (n / 32));
        // A row of one act block's worth of values is a whole number of weight blocks iff act_block % block_values == 0.
        CHECK(expert_layout_ns::row_bytes(r.type_id, r.block_values * 5) == 5 * r.block_bytes);
        CHECK(expert_layout_ns::row_bytes(r.type_id, r.block_values * 5 + 1) == 0);
    }
    CHECK(native_traits_of(NativeFmt::None).type_id == 0 && native_traits_of(NativeFmt::None).act_kind == NativeActKind::None);
    CHECK(native_fmt(0) == NativeFmt::None && native_fmt(41) == NativeFmt::None && native_fmt(43) == NativeFmt::None);
    // Q2_0 is id 42 (64 values, 18 bytes, 32-value activation chunks); i-quants are 256-value blocks with Q8_K activations.
    CHECK(native_traits_of(NativeFmt::Q2_0).type_id == 42 && native_traits_of(NativeFmt::Q2_0).block_bytes == 18 &&
          native_traits_of(NativeFmt::Q2_0).act_block == 32);
    CHECK(native_traits_of(NativeFmt::IQ3_S).block_bytes == 110 && native_traits_of(NativeFmt::IQ2_S).block_bytes == 82);

    // Backend selection is exactly the feature predicate, stable across calls and threads.
    const kern::CpuFeatures cf = kern::cpu_features();
    const NativeBackend want = cf.usable() ? NativeBackend::Avx512 : cf.avx2 ? NativeBackend::Avx2 : NativeBackend::Unavailable;
    CHECK(native_backend() == want);
    native_expert_init(); native_expert_init();
    CHECK(native_backend() == want);
    std::vector<std::thread> th; std::vector<int> got(8, -1);
    for (int i = 0; i < 8; ++i) th.emplace_back([&, i] { got[i] = (int)native_backend(); });
    for (auto& t : th) t.join();
    for (int g : got) CHECK(g == (int)want);
    CHECK((native_backend() == NativeBackend::Avx512) == cpu_avx512_ok());
    // Q2_0 experts of the production geometry are available exactly when some backend exists.
    CHECK(native_experts_available(42, 2560, 640) == (native_backend() != NativeBackend::Unavailable));
    CHECK(!native_experts_available(42, 2561, 640) && !native_experts_available(7, 2560, 640));

    if (fails) { std::printf("UNIT 31 TESTS FAILED (%d)\n", fails); return 1; }
    std::printf("backend=%d\nUNIT 31 TESTS PASSED\n", (int)want);
    return 0;
}
