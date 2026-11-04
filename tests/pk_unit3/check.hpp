// Minimal harness: CHECK records a failure; the run exits non-zero if any occurred.
#pragma once
#include <cstdint>
#include <cstdio>
#include <cstdlib>
namespace th {
inline long& fails() { static long f = 0; return f; }
inline long& checks() { static long c = 0; return c; }
struct Rng {
    uint64_t s;
    explicit Rng(uint64_t seed) : s(seed * 0x9E3779B97F4A7C15ull + 0x1234567ull) {}
    uint64_t next() { s ^= s << 13; s ^= s >> 7; s ^= s << 17; return s; }
    uint32_t below(uint32_t n) { return n ? (uint32_t)((next() >> 16) % n) : 0; }
};
}  // namespace th
#define CHECK(cond) do { ++th::checks(); if (!(cond)) { if (++th::fails() <= 50) std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); } } while (0)
