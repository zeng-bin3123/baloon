// weight_loader.hpp - plane-by-plane segment list of a quantized tensor, the 8 MiB chunked transform loop (copy, bf16
// high half, fp16 widen) and arena placement under a skip set. Pure host code.
#pragma once
#include <cstdint>
#include <cstddef>
#include <cstring>
#include <vector>
#include "baloon/f16_conversion.hpp"

namespace baloon {
namespace wldr {

constexpr uint64_t kChunk = 8ull << 20;      // 8 MiB
constexpr uint64_t kStageOut = 2 * kChunk;   // converted output of one chunk fits here for every transform

inline uint64_t chunk_len(uint64_t rem, uint64_t chunk = kChunk) { return rem < chunk ? rem : chunk; }
inline uint64_t src_scale_bytes(uint64_t sb, bool fp16) { return fp16 ? sb / 2 : sb; }
inline uint64_t align_up256(uint64_t n) { return (n + 255) / 256 * 256; }

enum class Xform : uint8_t { Copy = 0, WidenF16 = 1, Bf16High = 2 };
struct Seg { uint64_t src_off, dst_off, src_bytes, dst_bytes; Xform x; };

// Quantized tensor = codes | scales | offsets. If fp16 the source scale plane is half size and widened to f32.
// Returns false (and leaves segs empty) for an odd fp16 scale plane.
inline bool plan_quantized(uint64_t cb, uint64_t sb, uint64_t ob, bool fp16, std::vector<Seg>& segs) {
    segs.clear();
    if (fp16 && (sb % 2) != 0) return false;
    const uint64_t ss = src_scale_bytes(sb, fp16);
    segs.push_back({0, 0, cb, cb, Xform::Copy});
    segs.push_back({cb, cb, ss, sb, fp16 ? Xform::WidenF16 : Xform::Copy});
    segs.push_back({cb + ss, cb + sb, ob, ob, Xform::Copy});
    return true;
}
inline uint64_t seg_src_total(const std::vector<Seg>& s) { uint64_t t = 0; for (auto& g : s) t += g.src_bytes; return t; }
inline uint64_t seg_dst_total(const std::vector<Seg>& s) { uint64_t t = 0; for (auto& g : s) t += g.dst_bytes; return t; }

// A promoted tensor (bf16/f16 held in f32 in the source) is 4*elems in and 2*elems out.
inline bool promoted_ok(uint64_t elems, uint64_t src, uint64_t dst) { return src == elems * 4 && dst == elems * 2 && dst * 2 == src; }

// Chunked transform. Returns bytes written, or UINT64_MAX when the span does not divide the element size.
// `chunk` must be a multiple of 4 (so no chunk splits an element); defaults to the 8 MiB production size.
inline uint64_t transform_chunked(Xform x, const uint8_t* src, uint64_t src_bytes, uint8_t* dst, uint64_t chunk = kChunk) {
    if (chunk == 0 || chunk % 4 != 0) return UINT64_MAX;
    if (x == Xform::Bf16High && src_bytes % 4 != 0) return UINT64_MAX;
    if (x == Xform::WidenF16 && src_bytes % 2 != 0) return UINT64_MAX;
    uint64_t done = 0, out = 0;
    while (done < src_bytes) {
        const uint64_t n = chunk_len(src_bytes - done, chunk);
        switch (x) {
            case Xform::Copy: std::memcpy(dst + out, src + done, n); out += n; break;
            case Xform::Bf16High:
                for (uint64_t i = 0; i < n / 4; ++i) {  // high 16 bits of each little-endian f32
                    dst[out + 2 * i] = src[done + 4 * i + 2];
                    dst[out + 2 * i + 1] = src[done + 4 * i + 3];
                }
                out += n / 4 * 2; break;
            case Xform::WidenF16:
                for (uint64_t i = 0; i < n / 2; ++i) {
                    const uint16_t h = (uint16_t)(src[done + 2 * i] | (src[done + 2 * i + 1] << 8));
                    const float f = f16::half_to_float(h);
                    std::memcpy(dst + out + 4 * i, &f, 4);
                }
                out += n / 2 * 4; break;
        }
        done += n;
    }
    return out;
}

// Arena placement: kept tensors are re-placed in index order at 256-byte alignment.
struct Placement { std::vector<uint64_t> off; uint64_t total = 0; };
inline Placement place(const std::vector<uint64_t>& sizes, const std::vector<bool>& skip) {
    Placement p; p.off.assign(sizes.size(), UINT64_MAX);
    for (size_t i = 0; i < sizes.size(); ++i) {
        if (i < skip.size() && skip[i]) continue;
        p.off[i] = p.total;
        p.total += align_up256(sizes[i]);
    }
    return p;
}

}  // namespace wldr
}  // namespace baloon
