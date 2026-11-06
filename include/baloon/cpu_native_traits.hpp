// cpu_native_traits.hpp - per-format traits of the native CPU expert path and the one-time backend selection.
//
// traits<F> is the compile-time description of a native format (ggml type id, values / bytes per weight block, activation
// block and kind). Every field is cross-checked at compile time against the layout helpers (expert_layout_ns) and the
// activation sizing of cpu_native_expert.hpp, so the three tables cannot drift apart.
// native_expert_init() / native_backend() are implemented in src/cpu/native_expert.cpp: the CPU feature probe runs once
// and its result is cached.
#pragma once
#include <cstdint>

#include "baloon/cpu_native_expert.hpp"

namespace baloon {

enum class NativeActKind : int { None = 0, Q8_1x32, Q8K };
enum class NativeBackend : int { Unavailable = 0, Avx2, Avx512 };

namespace native_traits {

template <NativeFmt F> struct traits;   // only the specializations below exist

#define BALOON_NATIVE_TRAITS(FMT, ID, VALS, BYTES, ACTBLK, KIND)                         \
    template <> struct traits<NativeFmt::FMT> {                                           \
        static constexpr NativeFmt fmt = NativeFmt::FMT;                                  \
        static constexpr int type_id = ID;                                                \
        static constexpr int64_t block_values = VALS;                                     \
        static constexpr int64_t block_bytes = BYTES;                                     \
        static constexpr uint32_t act_block = ACTBLK;                                     \
        static constexpr NativeActKind act_kind = NativeActKind::KIND;                    \
    }
BALOON_NATIVE_TRAITS(Q2_0, 42, 64, 18, 32, Q8_1x32);
BALOON_NATIVE_TRAITS(IQ2_XXS, 16, 256, 66, 256, Q8K);
BALOON_NATIVE_TRAITS(IQ2_XS, 17, 256, 74, 256, Q8K);
BALOON_NATIVE_TRAITS(IQ3_XXS, 18, 256, 98, 256, Q8K);
BALOON_NATIVE_TRAITS(IQ3_S, 21, 256, 110, 256, Q8K);
BALOON_NATIVE_TRAITS(IQ2_S, 22, 256, 82, 256, Q8K);
#undef BALOON_NATIVE_TRAITS

// Cross-checks against the independent tables.
#define BALOON_NATIVE_TRAITS_CHECK(FMT)                                                                       \
    static_assert(traits<NativeFmt::FMT>::block_values == expert_layout_ns::block_values(traits<NativeFmt::FMT>::type_id), #FMT " values"); \
    static_assert(traits<NativeFmt::FMT>::block_bytes == expert_layout_ns::block_bytes(traits<NativeFmt::FMT>::type_id), #FMT " bytes")
BALOON_NATIVE_TRAITS_CHECK(Q2_0);
BALOON_NATIVE_TRAITS_CHECK(IQ2_XXS);
BALOON_NATIVE_TRAITS_CHECK(IQ2_XS);
BALOON_NATIVE_TRAITS_CHECK(IQ3_XXS);
BALOON_NATIVE_TRAITS_CHECK(IQ3_S);
BALOON_NATIVE_TRAITS_CHECK(IQ2_S);
#undef BALOON_NATIVE_TRAITS_CHECK

}  // namespace native_traits

// Runtime row of a format's traits (type_id == 0 for NativeFmt::None).
struct NativeTraitsRow {
    NativeFmt fmt = NativeFmt::None;
    int type_id = 0;
    int64_t block_values = 0, block_bytes = 0;
    uint32_t act_block = 0;
    NativeActKind act_kind = NativeActKind::None;
};

namespace native_traits {
template <NativeFmt F> constexpr NativeTraitsRow row() {
    return NativeTraitsRow{F, traits<F>::type_id, traits<F>::block_values, traits<F>::block_bytes, traits<F>::act_block, traits<F>::act_kind};
}
}  // namespace native_traits

inline NativeTraitsRow native_traits_of(NativeFmt f) {
    using namespace native_traits;
    switch (f) {
        case NativeFmt::Q2_0: return row<NativeFmt::Q2_0>();
        case NativeFmt::IQ2_XXS: return row<NativeFmt::IQ2_XXS>();
        case NativeFmt::IQ2_XS: return row<NativeFmt::IQ2_XS>();
        case NativeFmt::IQ3_XXS: return row<NativeFmt::IQ3_XXS>();
        case NativeFmt::IQ3_S: return row<NativeFmt::IQ3_S>();
        case NativeFmt::IQ2_S: return row<NativeFmt::IQ2_S>();
        default: return NativeTraitsRow{};
    }
}

// Runs the one-time CPU probe (idempotent, thread-safe).
void native_expert_init();
// The best CPU path: Avx512 when the full expert feature set is present, else Avx2, else Unavailable. Cached.
NativeBackend native_backend();

}  // namespace baloon
