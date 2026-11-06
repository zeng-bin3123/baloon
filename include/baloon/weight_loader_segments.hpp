// weight_loader_segments.hpp - dense-weight loader arithmetic: chunked fp16->f32 widening and BF16 high-half
// promotion with exact progress/staging bounds, quantised-tensor segment tiling, and compacted arena offsets.
#pragma once
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>
#include "baloon/f16_conversion.hpp"

namespace baloon {
namespace weights {

// 8 MiB of source per staging round; widening doubles a chunk so the output stage is 2*kChunk.
constexpr uint64_t kChunk = 8388608ull;
static_assert(kChunk % 4 == 0, "full chunks never split a 4-byte element");

// ---- widening (fp16 -> f32) ------------------------------------------------------------------
inline uint64_t widen_out(uint64_t n) { return n / 2 * 4; }                 // n even source bytes -> (n/2)*4
inline uint64_t even_trim(uint64_t n) { return (n % 2 == 1) ? n - 1 : n; }  // trim an odd tail to whole elements

// ---- BF16 high-half promotion (f32 -> bf16) -----------------------------------------------------
inline uint64_t bf16_out(uint64_t n) { return n / 4 * 2; }                  // keep 2 of every 4 bytes
inline uint64_t quad_trim(uint64_t n) { return n - n % 4; }

// A promoted tensor is 4*elems in and 2*elems out; the saving equals the output size.
inline uint64_t promoted_src_bytes(uint64_t elems) { return elems * 4; }
inline uint64_t promoted_dst_bytes(uint64_t elems) { return elems * 2; }
inline uint64_t promoted_saving(uint64_t elems) { return promoted_src_bytes(elems) - promoted_dst_bytes(elems); }

struct ConvResult {
    uint64_t consumed = 0;  // source bytes consumed (whole elements only)
    uint64_t written = 0;   // destination bytes produced
    uint64_t rounds = 0;    // staging rounds executed
};

// Normalise a user chunk to a positive multiple of 4 (a staging round never splits an element).
inline uint64_t norm_chunk(uint64_t chunk) { return std::max<uint64_t>(4, chunk - chunk % 4); }

// fp16 -> f32, chunk by chunk through a 2*chunk staging buffer. dst must hold widen_out(even_trim(n_src)) bytes.
// Invariant after every round: dst_off == done * 2 (widen_progress).
inline ConvResult widen_f16_chunked(const uint8_t* src, uint64_t n_src, uint8_t* dst, uint64_t chunk = kChunk) {
    chunk = norm_chunk(chunk);
    std::vector<uint8_t> stage_out((size_t)(2 * chunk));
    ConvResult r;
    while (r.consumed < n_src) {
        uint64_t n = std::min<uint64_t>(chunk, n_src - r.consumed);
        n = even_trim(n);
        if (n == 0) break;  // a lone trailing byte is not an element
        const uint8_t* s = src + r.consumed;
        for (uint64_t i = 0; i < n / 2; ++i) {
            const uint16_t h = (uint16_t)(s[2 * i] | (s[2 * i + 1] << 8));
            const uint32_t bits = f16::f32_bits(f16::half_to_float(h));
            for (int b = 0; b < 4; ++b) stage_out[(size_t)(4 * i + b)] = (uint8_t)(bits >> (8 * b));
        }
        std::memcpy(dst + r.written, stage_out.data(), (size_t)widen_out(n));
        r.consumed += n;
        r.written += widen_out(n);
        ++r.rounds;
    }
    return r;
}

// f32 -> bf16 by taking the high 16 bits of each little-endian word. dst must hold bf16_out(quad_trim(n_src)) bytes.
inline ConvResult promote_bf16_high_chunked(const uint8_t* src, uint64_t n_src, uint8_t* dst, uint64_t chunk = kChunk) {
    chunk = norm_chunk(chunk);
    std::vector<uint8_t> stage_out((size_t)(2 * chunk));
    ConvResult r;
    while (r.consumed < n_src) {
        uint64_t n = quad_trim(std::min<uint64_t>(chunk, n_src - r.consumed));
        if (n == 0) break;
        const uint8_t* s = src + r.consumed;
        for (uint64_t i = 0; i < n / 4; ++i) {
            stage_out[(size_t)(2 * i)] = s[4 * i + 2];
            stage_out[(size_t)(2 * i + 1)] = s[4 * i + 3];
        }
        std::memcpy(dst + r.written, stage_out.data(), (size_t)bf16_out(n));
        r.consumed += n;
        r.written += bf16_out(n);
        ++r.rounds;
    }
    return r;
}

// ---- quantised tensor segment tiling -------------------------------------------------------------
struct QuantSegments {
    uint64_t codes = 0, scales = 0, offs = 0;
    uint64_t dst_total() const { return codes + scales + offs; }
    uint64_t dst_codes_start() const { return 0; }
    uint64_t dst_scales_start() const { return codes; }
    uint64_t dst_offs_start() const { return codes + scales; }
    // With fp16 scales (even byte count) the source scale plane is half the destination plane.
    uint64_t src_scale_plane() const { return scales / 2; }
    // The offset segment's source start follows the narrower source scale plane(s).
    uint64_t src_offs_start() const { return codes + scales / 2 + scales / 2; }
};

// ---- compacted arena -------------------------------------------------------------------------------
inline uint64_t align256(uint64_t b) { return (b + 255) / 256 * 256; }

// Offset of kept row i: the sum of aligned sizes of the rows before it (i beyond the end gives the pool size).
inline uint64_t arena_off(const std::vector<uint64_t>& rows, size_t i) {
    uint64_t o = 0;
    for (size_t k = 0; k < i && k < rows.size(); ++k) o += align256(rows[k]);
    return o;
}
inline uint64_t arena_pool(const std::vector<uint64_t>& rows) { return arena_off(rows, rows.size()); }

struct CompactedArena {
    static constexpr int64_t kSkipped = -1;
    std::vector<int64_t> offset;  // per original row; kSkipped for rows in the skip set
    uint64_t pool = 0;            // total arena bytes
};

// Skip set compaction: kept rows are re-placed in index order at 256-byte alignment.
inline CompactedArena compact_arena(const std::vector<uint64_t>& row_bytes, const std::vector<bool>& skip) {
    CompactedArena a;
    a.offset.assign(row_bytes.size(), CompactedArena::kSkipped);
    uint64_t cur = 0;
    for (size_t i = 0; i < row_bytes.size(); ++i) {
        if (i < skip.size() && skip[i]) continue;
        a.offset[i] = (int64_t)cur;
        cur += align256(row_bytes[i]);
    }
    a.pool = cur;
    return a;
}

}  // namespace weights
}  // namespace baloon
