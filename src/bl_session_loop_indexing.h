// bl_session_loop_indexing.h - per-layer state indexing, arena sizing, doorbell numbers of the session loop.
#pragma once
#include "bl_nat.h"

namespace baloon::session_loop_indexing {

// Layer l is a QSA layer iff l % 4 == 3. GDN and QSA layers are numbered separately by counting
// the layers of each kind strictly before l (loop form; the closed forms are l - l/4 and l/4).
constexpr bool isQsa(nat l) { return l % 4 == 3; }
constexpr nat gdnIdx(nat l) { nat c = 0; for (nat i = 0; i < l; ++i) c += isQsa(i) ? 0 : 1; return c; }
constexpr nat qsaIdx(nat l) { nat c = 0; for (nat i = 0; i < l; ++i) c += isQsa(i) ? 1 : 0; return c; }
constexpr nat gdnIdxClosed(nat l) { return l - l / 4; }
constexpr nat qsaIdxClosed(nat l) { return l / 4; }

// Arena rounding: smallest multiple of a that is >= n (a > 0).
constexpr nat alignUp(nat n, nat a) { return (n + a - 1) / a * a; }

// Layer l (0-based) waits for ring number l + 1; `want` is the next value of a monotone counter.
constexpr nat want(nat expected) { return expected + 1; }

// Per-stage times from cumulative prefix times; they telescope back to the last prefix time.
constexpr std::int64_t stageTime(const std::int64_t* cum, int i) { return i == 0 ? cum[0] : cum[i] - cum[i - 1]; }

}  // namespace baloon::session_loop_indexing
