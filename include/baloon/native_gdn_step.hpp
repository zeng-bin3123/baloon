// native_gdn_step.hpp - addressing and validation arithmetic of the native gated-delta-net step kernel.
// Block (head, 1, z) with 32x4 threads; col = z*4 + ty; rows i = r*32 + lane for r in 0..3 cover 0..127.
// Requires S=128, h_v % h_k == 0, h_v <= 65535, aligned and pairwise-disjoint spans.
#pragma once
#include <cstddef>
#include <cstdint>

namespace baloon {
namespace native_gdn {

constexpr int kS = 128;
constexpr int kRegs = 4;
constexpr int kLanes = 32;
constexpr int kBlockY = 4;
constexpr int kMaxGridY = 65535;
constexpr uint64_t kUMax = UINT64_MAX;

inline int row_of(int r, int lane) { return r * kLanes + lane; }
inline int col_of(int z, int y) { return z * kBlockY + y; }
inline int grid_z(int S) { return S / kBlockY; }
inline int q_head(int head, int hk) { return head % hk; }
inline int head_group(int hk, int hv) { return hv / hk; }

struct Span { uint64_t addr; uint64_t bytes; };
inline bool overlap(uint64_t a, uint64_t an, uint64_t b, uint64_t bn) { return a < b + bn && b < a + an; }
inline bool overlap(const Span& x, const Span& y) { return overlap(x.addr, x.bytes, y.addr, y.bytes); }
inline bool span_valid(const Span& s) { return s.bytes <= kUMax - s.addr; }       // cannot wrap
inline bool span_aligned4(const Span& s) { return s.addr % 4 == 0 && s.bytes % 4 == 0; }

enum class Status { Ok, BadStateWidth, BadHeadCount, BadHeadRatio, TooManyHeads, SpanWrap, Misaligned, Overlap };

inline Status validate(int S, int hk, int hv, const Span* spans, size_t n) {
    if (S != kS) return Status::BadStateWidth;
    if (hk <= 0 || hv <= 0) return Status::BadHeadCount;
    if (hv % hk != 0) return Status::BadHeadRatio;
    if (hv > kMaxGridY) return Status::TooManyHeads;
    for (size_t i = 0; i < n; ++i) {
        if (!span_valid(spans[i])) return Status::SpanWrap;
        if (!span_aligned4(spans[i])) return Status::Misaligned;
    }
    for (size_t i = 0; i < n; ++i)
        for (size_t j = i + 1; j < n; ++j)
            if (overlap(spans[i], spans[j])) return Status::Overlap;
    return Status::Ok;
}

}  // namespace native_gdn
}  // namespace baloon
