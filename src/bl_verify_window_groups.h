// bl_verify_window_groups.h - token-group split, ring numbering, staging and plan-buffer layout of the verify window.
#pragma once
#include "bl_nat.h"

namespace baloon::verify_window_groups {

// Window of T tokens cut into group A = [0,(T+1)/2) and B = [(T+1)/2, T) when G = 2; one group holds all when G = 1.
constexpr nat tbF(nat T, nat grp) { return grp == 0 ? 0 : (T + 1) / 2; }
constexpr nat teF(nat G, nat T, nat grp) { return (G == 2 && grp == 0) ? (T + 1) / 2 : T; }

// Processing order k = l*G + grp; ring awaited is k+1 (stored as grp + G*l + 1).
constexpr nat ringK(nat G, nat l, nat grp) { return grp + G * l + 1; }
constexpr nat ringLayer(nat G, nat k) { return ndiv(k - 1, G); }   // k is a ring number (>= 1)
constexpr nat ringGroup(nat G, nat k) { return nmod(k - 1, G); }

// Staging blobs split evenly per group.
constexpr nat perGroup(nat G, nat K) { return G == 2 ? K / 2 : K; }

// Plan buffer: i32 header/arrays, then 8-byte aligned u64 pointer arrays (offset counted in i32 units).
constexpr nat planI32Count(nat cap) { return 4 + (cap + 1) + cap + cap; }
constexpr nat planPtrOff(nat cap) { return (planI32Count(cap) + 1) / 2 * 2; }

// Flag B only rises.
constexpr nat raiseFlag(nat cur, nat v) { return cur < v ? v : cur; }

// Flag-set ring index.
constexpr nat flagsetIndex(nat n, nat sz) { return nmod(n, sz); }

}  // namespace baloon::verify_window_groups
