// bl_nat.h - natural-number helpers with total (non-trapping) semantics.
// Subtraction saturates at 0 and division/modulo by 0 yield 0, so every function built on
// these helpers is total, matching the math the unit theory states over naturals.
#pragma once
#include <cstddef>
#include <cstdint>

namespace baloon {

using nat = std::uint64_t;

constexpr nat sat_sub(nat a, nat b) { return a > b ? a - b : 0; }
constexpr nat ndiv(nat a, nat b) { return b ? a / b : 0; }
constexpr nat nmod(nat a, nat b) { return b ? a % b : 0; }
constexpr nat nmin(nat a, nat b) { return a < b ? a : b; }
constexpr nat nmax(nat a, nat b) { return a > b ? a : b; }

// Round n up to a multiple of a (a > 0). Total: a == 0 gives 0.
constexpr nat round_up(nat n, nat a) { return ndiv(n + a - 1, a) * a; }

}  // namespace baloon
