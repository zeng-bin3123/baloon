// mmvf_select.hpp - the host-side block-size selection of bf16_gemv_fp32_mmvf (theory Bf16MmvfBlockSelect) as constexpr
// C++, so every statement is checked by static_assert in any compiler that includes it (corex clang++, g++).
#pragma once
#include <cstdint>

namespace baloon {
namespace mmvf {

// Theory iters: per-thread pair iterations of a block of c threads over n elements: (n + 2c - 1) / (2c).
constexpr long long pair_iters(long long n, int c) { return (n + 2LL * c - 1) / (2LL * c); }

// Theory stepBest / init / search: start at (32, (n + 63) / 64); for c = 64, 96, ..., 256 adopt c only on a STRICT
// improvement. Best carries the (block, iterations) bookkeeping pair of the fold.
struct Best { int block; long long iters; };
constexpr Best step_best(long long n, Best a, int c) { const long long it = pair_iters(n, c); return it < a.iters ? Best{c, it} : a; }
constexpr Best search(long long n) {
    Best b{32, (n + 63) / 64};
    for (int c = 64; c <= 256; c += 32) b = step_best(n, b, c);
    return b;
}
constexpr int block_size(long long n) { return search(n).block; }

// Theory stepBest_le / fold_le_init / search_consistent: no step increases the best count, every step's best is no worse
// than the candidate just tried, and the recorded count is the iteration count of the recorded block.
constexpr bool fold_invariants_ok(long long n) {
    Best b{32, (n + 63) / 64};
    if (b.iters != pair_iters(n, 32)) return false;
    for (int c = 64; c <= 256; c += 32) {
        const Best nb = step_best(n, b, c);
        if (nb.iters > b.iters || nb.iters > pair_iters(n, c)) return false;
        b = nb;
    }
    const Best s = search(n);
    return s.block == b.block && s.iters == pair_iters(n, s.block);
}
constexpr bool fold_invariants_range_ok() {
    for (long long n = 1; n <= 600; ++n) if (!fold_invariants_ok(n)) return false;
    return fold_invariants_ok(2560) && fold_invariants_ok(4096) && fold_invariants_ok(65536) && fold_invariants_ok(2147483646LL);
}
static_assert(fold_invariants_range_ok(), "stepBest_le / fold_le_init / search_consistent");

// Theory switchCases / blockSize_mem_switch / blockSize_optimal: the selected block is always one of the eight launch
// cases of bf16_gemv_fp32_mmvf, and needs no more pair iterations than any case, including 32.
constexpr int kSwitchCases[8] = {32, 64, 96, 128, 160, 192, 224, 256};
constexpr bool is_switch_case(int b) { for (int c : kSwitchCases) if (c == b) return true; return false; }
constexpr bool select_ok(long long n) {
    const int b = block_size(n);
    if (!is_switch_case(b)) return false;
    for (int c : kSwitchCases) if (pair_iters(n, b) > pair_iters(n, c)) return false;
    return true;
}
constexpr bool select_range_ok() {
    for (long long n = 1; n <= 600; ++n) if (!select_ok(n)) return false;
    return select_ok(2560) && select_ok(4096) && select_ok(65536) && select_ok(2147483646LL);
}
static_assert(select_range_ok(), "blockSize_mem_switch / blockSize_optimal");
// Theory argsOk / odd_refused / pairs_cover: n_in > 0, even, <= INT_MAX and 0 < n_out <= INT_MAX. An odd n_in is
// refused (the pair loop would drop its last element); with an even n_in, element i is half (i % 2) of pair i / 2 < n_in / 2.
constexpr long long kIntMax = 2147483647LL;
constexpr bool args_ok(long long n_in, long long n_out) { return n_in > 0 && n_in % 2 == 0 && n_in <= kIntMax && n_out > 0 && n_out <= kIntMax; }
constexpr bool odd_refused_ok() {
    for (long long n = -9; n <= 4097; ++n) if (n % 2 != 0 && args_ok(n, 1)) return false;
    return !args_ok(kIntMax, 1) && args_ok(kIntMax - 1, 1) && args_ok(2560, 2560) && !args_ok(0, 1) && !args_ok(2, 0) && !args_ok(2, kIntMax + 1);
}
static_assert(odd_refused_ok(), "odd_refused");
constexpr bool pairs_cover_ok() {
    for (long long n = 2; n <= 64; n += 2)
        for (long long i = 0; i < n; ++i) if (!(i / 2 < n / 2 && 2 * (i / 2) + i % 2 == i)) return false;
    return true;
}
static_assert(pairs_cover_ok(), "pairs_cover");
}  // namespace mmvf
}  // namespace baloon
