// strata/core/weights.cpp — Dense weight table: index parse, format conversion, staged GPU upload
// Target: Iluvatar BI-V100 (ivcore10)
//
// Loads a weight-pack directory (index.txt + binary shards) into a
// pre-allocated GPU arena. Handles three-plane quantized tensors
// (codes | scales | offsets) with per-plane format conversion.
//
// Conversion types:
//   Copy      — verbatim byte copy (1:1)
//   Bf16High  — extract upper 16 bits of f32 → bf16 (2:1)
//   ToF16     — f32 → fp16 round-to-nearest-even (2:1)
//   WidenF16  — fp16 → f32 exact expansion (1:2)

#include "strata/core/weights.hpp"
#include "strata/kernels/f16_bits.hpp"

#include <chrono>
#include <cstdio>
#include <cstring>
#include <vector>

#if defined(__CUDACC__) || defined(__ILUVATAR__)
#include <cuda_runtime.h>
#else
// Stub for host-only compilation
using cudaStream_t = void*;
using cudaError_t = int;
static constexpr cudaError_t cudaSuccess = 0;
static cudaError_t cudaMemcpy(void* dst, const void* src, size_t n, int) {
    std::memcpy(dst, src, n); return cudaSuccess;
}
static constexpr int cudaMemcpyHostToDevice = 1;
static cudaError_t cudaHostAlloc(void** p, size_t n, unsigned) {
    *p = std::malloc(n); return *p ? cudaSuccess : 1;
}
static void cudaFreeHost(void* p) { std::free(p); }
static constexpr unsigned cudaHostAllocDefault = 0;
#endif

namespace strata {

namespace {

constexpr size_t ALIGN = 256;
constexpr size_t CHUNK = 8 * 1024 * 1024;  // 8 MiB staging

// Segment conversion kind
enum class Conv { Copy, Bf16High, ToF16, WidenF16 };

struct Segment {
    uint64_t src_off = 0;
    uint64_t src_bytes = 0;
    uint64_t dst_off = 0;
    uint64_t dst_bytes = 0;
    Conv conv = Conv::Copy;
};

// An index row parsed from index.txt
struct IndexRow {
    std::string name;
    WeightKind kind = WeightKind::Verbatim;
    bool raw16 = false;  // source is 16-bit, needs widening
    int64_t ne0 = 0;
    int64_t ne1 = 0;
    int64_t n_elements = 0;
    uint64_t src_offset = 0;
    uint64_t src_bytes = 0;
    uint64_t dst_bytes = 0;
    int file_id = 0;
    // Quantization info
    int code_bits = 0;
    int code_bias = 0;
    int group_elems = 0;
    bool codebook_iq4nl = false;
    bool has_offset = false;
    uint64_t codes_bytes = 0;
    uint64_t scales_bytes = 0;
    uint64_t offset_bytes = 0;
    bool scales_fp16 = false;
    int act_kind = 0;
};

static uint64_t align_up(uint64_t v, uint64_t a) {
    return (v + a - 1) / a * a;
}

// Apply conversion to a chunk of bytes in-place (or src→dst)
static void convert_chunk(const void* src, void* dst, uint64_t n_bytes, Conv conv) {
    switch (conv) {
    case Conv::Copy:
        if (dst != src) std::memcpy(dst, src, (size_t)n_bytes);
        break;
    case Conv::Bf16High: {
        // 4 bytes → 2 bytes: extract upper 16 bits of each f32
        const uint32_t* s = reinterpret_cast<const uint32_t*>(src);
        uint16_t* d = reinterpret_cast<uint16_t*>(dst);
        size_t count = (size_t)(n_bytes / 4);
        for (size_t i = 0; i < count; ++i) {
            d[i] = static_cast<uint16_t>(s[i] >> 16);
        }
        break;
    }
    case Conv::ToF16: {
        // 4 bytes → 2 bytes: f32 → fp16
        const float* s = reinterpret_cast<const float*>(src);
        uint16_t* d = reinterpret_cast<uint16_t*>(dst);
        size_t count = (size_t)(n_bytes / 4);
        for (size_t i = 0; i < count; ++i) {
            d[i] = kernels::f16_from_f32(s[i]);
        }
        break;
    }
    case Conv::WidenF16: {
        // 2 bytes → 4 bytes: fp16 → f32
        const uint16_t* s = reinterpret_cast<const uint16_t*>(src);
        float* d = reinterpret_cast<float*>(dst);
        size_t count = (size_t)(n_bytes / 2);
        for (size_t i = 0; i < count; ++i) {
            d[i] = kernels::f32_from_f16(s[i]);
        }
        break;
    }
    }
}

// Build segment list for a row
static std::vector<Segment> build_segments(const IndexRow& r) {
    std::vector<Segment> segs;
    if (r.code_bits == 0) {
        // Non-quantized: single segment
        Conv c = Conv::Copy;
        uint64_t s_bytes = r.src_bytes;
        uint64_t d_bytes = r.dst_bytes;
        if (r.raw16 && r.kind == WeightKind::Bf16InF32) {
            c = Conv::Bf16High;
            // src is f32, dst is bf16: 4→2
        } else if (r.raw16 && r.kind == WeightKind::F16InF32) {
            c = Conv::ToF16;
        } else if (r.scales_fp16) {
            c = Conv::WidenF16;
        }
        segs.push_back({0, s_bytes, 0, d_bytes, c});
    } else {
        // Quantized: up to 3 segments (codes, scales, offsets)
        uint64_t src_pos = 0, dst_pos = 0;

        // Seg 0: codes (always verbatim)
        segs.push_back({src_pos, r.codes_bytes, dst_pos, r.codes_bytes, Conv::Copy});
        src_pos += r.codes_bytes;
        dst_pos += r.codes_bytes;

        // Seg 1: scales
        if (r.scales_bytes > 0) {
            Conv sc = Conv::Copy;
            uint64_t ss = r.scales_bytes;
            if (r.scales_fp16) {
                sc = Conv::WidenF16;
                ss = r.scales_bytes / 2;  // source is half the dst size
            }
            segs.push_back({src_pos, ss, dst_pos, r.scales_bytes, sc});
            src_pos += ss;
            dst_pos += r.scales_bytes;
        }

        // Seg 2: offsets (always verbatim, if present)
        if (r.has_offset && r.offset_bytes > 0) {
            segs.push_back({src_pos, r.offset_bytes, dst_pos, r.offset_bytes, Conv::Copy});
        }
    }
    return segs;
}

// File handle cache: maps file_id → FILE*
struct PackFile {
    std::string path;
    std::FILE* fp = nullptr;
};

static bool read_at(std::FILE* fp, uint64_t offset, void* buf, size_t n) {
    if (std::fseek(fp, (long)offset, SEEK_SET) != 0) return false;
    return std::fread(buf, 1, n, fp) == n;
}

}  // namespace

// ─── pool_bytes ──────────────────────────────────────────────────────
bool WeightTable::pool_bytes(const std::string& dir, uint64_t& out_bytes,
                             std::string& err,
                             const std::set<std::string>* skip) {
    const std::string path = dir + "/index.txt";
    std::FILE* fp = std::fopen(path.c_str(), "rb");
    if (!fp) { err = "cannot open " + path; return false; }

    char line[1024];
    out_bytes = 0;
    uint64_t pool = 0;
    uint64_t compact = 0;
    int align_val = 0;

    while (std::fgets(line, sizeof line, fp)) {
        if (line[0] == '#') {
            unsigned long long p = 0;
            int a = 0, tensors = 0;
            if (std::sscanf(line, "# align %d pool %llu tensors %d", &a, &p, &tensors) == 3) {
                pool = p;
                align_val = a;
            }
            continue;
        }
        if (skip == nullptr) continue;
        char name[256] = {0};
        unsigned long long dst_bytes = 0, dummy = 0;
        int i1 = 0, i2 = 0;
        if (std::sscanf(line, "%255s %d %d %llu %llu %llu %llu",
                        name, &i1, &i2, &dummy, &dummy, &dummy, &dst_bytes) != 7)
            continue;
        if (skip->count(name)) continue;
        const uint64_t a = align_val > 0 ? (uint64_t)align_val : ALIGN;
        compact += align_up(dst_bytes, a);
    }
    std::fclose(fp);

    if (pool == 0) {
        err = "no '# align ... pool ...' header in " + path;
        return false;
    }
    out_bytes = skip ? compact : pool;
    return true;
}

// ─── load ────────────────────────────────────────────────────────────
bool WeightTable::load(const std::string& dir, void* arena_base,
                       uint64_t arena_bytes, std::string& err,
                       const std::set<std::string>* skip) {
    auto t0 = std::chrono::steady_clock::now();

    const std::string idx_path = dir + "/index.txt";
    std::FILE* idx = std::fopen(idx_path.c_str(), "rb");
    if (!idx) { err = "cannot open " + idx_path + " (run tools/pack_index.py)"; return false; }

    // Parse header and rows
    uint64_t pool = 0;
    int align_val = 0;
    std::vector<IndexRow> rows;
    char line[2048];
    // File names by id
    static const char* file_names[] = {"dense.bin", "embd.bin", "experts.bin", "extra.bin"};

    while (std::fgets(line, sizeof line, idx)) {
        if (line[0] == '#') {
            unsigned long long p = 0;
            int a = 0, t = 0;
            if (std::sscanf(line, "# align %d pool %llu tensors %d", &a, &p, &t) == 3) {
                pool = p;
                align_val = a;
            }
            continue;
        }
        IndexRow r;
        char name[256] = {0};
        int kind_raw = 0;
        unsigned long long ne0 = 0, ne1 = 0, n_elements = 0;
        unsigned long long src_offset = 0, src_bytes = 0, dst_bytes = 0;
        int file_id = 0;
        int code_bits = 0, code_bias = 0, group_elems = 0;
        int codebook_iq4nl = 0, has_offset = 0;
        unsigned long long codes_bytes = 0, scales_bytes = 0, offset_bytes = 0;
        int scales_fp16 = 0, act_kind = 0;

        int n = std::sscanf(line,
            "%255s %d %d %llu %llu %llu %llu %llu %llu %d %d %d %d %d %llu %llu %llu %d %d",
            name, &kind_raw, &file_id,
            &ne0, &ne1, &n_elements,
            &src_offset, &src_bytes, &dst_bytes,
            &code_bits, &code_bias, &group_elems,
            &codebook_iq4nl, &has_offset,
            &codes_bytes, &scales_bytes, &offset_bytes,
            &scales_fp16, &act_kind);
        if (n < 7) continue;  // need at least name + kind + file_id + dims + src/dst

        r.name = name;
        // Handle raw16 marker: kind=4 means Bf16InF32 from raw16, kind=5 means F16InF32 from raw16
        if (kind_raw == 4 || kind_raw == 5) {
            r.raw16 = true;
            r.kind = kind_raw == 4 ? WeightKind::Bf16InF32 : WeightKind::F16InF32;
        } else {
            r.kind = static_cast<WeightKind>(kind_raw);
        }
        r.file_id = file_id;
        r.ne0 = (int64_t)ne0;
        r.ne1 = (int64_t)ne1;
        r.n_elements = (int64_t)n_elements;
        r.src_offset = src_offset;
        r.src_bytes = src_bytes;
        r.dst_bytes = dst_bytes;
        if (n >= 19) {
            r.code_bits = code_bits;
            r.code_bias = code_bias;
            r.group_elems = group_elems;
            r.codebook_iq4nl = codebook_iq4nl != 0;
            r.has_offset = has_offset != 0;
            r.codes_bytes = codes_bytes;
            r.scales_bytes = scales_bytes;
            r.offset_bytes = offset_bytes;
            r.scales_fp16 = scales_fp16 != 0;
            r.act_kind = act_kind;
        }
        rows.push_back(std::move(r));
    }
    std::fclose(idx);

    if (pool == 0) {
        err = "no '# align ... pool ...' header in " + idx_path;
        return false;
    }

    // Allocate staging buffers
    void* stage_in = nullptr;
    void* stage_out = nullptr;
    if (cudaHostAlloc(&stage_in, CHUNK, cudaHostAllocDefault) != cudaSuccess) {
        err = "cudaHostAlloc stage_in failed";
        return false;
    }
    if (cudaHostAlloc(&stage_out, CHUNK * 2, cudaHostAllocDefault) != cudaSuccess) {
        cudaFreeHost(stage_in);
        err = "cudaHostAlloc stage_out failed";
        return false;
    }

    auto t_read_start = std::chrono::steady_clock::now();

    // Open pack files on demand
    std::FILE* pack_files[4] = {};
    auto open_pack = [&](int fid) -> std::FILE* {
        if (fid < 0 || fid >= 4) return nullptr;
        if (!pack_files[fid]) {
            std::string p = dir + "/" + file_names[fid];
            pack_files[fid] = std::fopen(p.c_str(), "rb");
        }
        return pack_files[fid];
    };

    const uint64_t a = align_val > 0 ? (uint64_t)align_val : ALIGN;
    uint8_t* dst_cursor = static_cast<uint8_t*>(arena_base);
    size_t tensors_loaded = 0;
    size_t re_rounded = 0;

    for (const auto& r : rows) {
        if (skip && skip->count(r.name)) continue;

        const uint64_t aligned_dst = align_up(r.dst_bytes, a);
        if ((uint64_t)(dst_cursor - static_cast<uint8_t*>(arena_base)) + aligned_dst > arena_bytes) {
            err = "arena overflow loading " + r.name;
            goto cleanup;
        }

        // Build segment list
        auto segs = build_segments(r);

        // Validate segment sums
        {
            uint64_t sum_src = 0, sum_dst = 0;
            for (auto& s : segs) { sum_src += s.src_bytes; sum_dst += s.dst_bytes; }
            if (sum_src > r.src_bytes || sum_dst > r.dst_bytes) {
                err = "segment mismatch for " + r.name;
                goto cleanup;
            }
        }

        // Transfer each segment
        {
            std::FILE* fp = open_pack(r.file_id);
            if (!fp) {
                err = "cannot open pack file " + std::to_string(r.file_id) + " for " + r.name;
                goto cleanup;
            }

            for (auto& seg : segs) {
                uint64_t src_pos = r.src_offset + seg.src_off;
                uint64_t remaining = seg.src_bytes;
                uint64_t dst_written = 0;

                while (remaining > 0) {
                    uint64_t chunk = remaining < CHUNK ? remaining : CHUNK;
                    if (!read_at(fp, src_pos, stage_in, (size_t)chunk)) {
                        err = "read failed for " + r.name;
                        goto cleanup;
                    }

                    // Convert
                    uint64_t out_size = chunk;
                    if (seg.conv == Conv::Bf16High || seg.conv == Conv::ToF16) {
                        out_size = chunk / 2;
                    } else if (seg.conv == Conv::WidenF16) {
                        out_size = chunk * 2;
                    }
                    convert_chunk(stage_in, stage_out, chunk, seg.conv);

                    // Upload to GPU
                    if (cudaMemcpy(dst_cursor + seg.dst_off + dst_written,
                                   stage_out, (size_t)out_size,
                                   cudaMemcpyHostToDevice) != cudaSuccess) {
                        err = "cudaMemcpy failed for " + r.name;
                        goto cleanup;
                    }

                    src_pos += chunk;
                    remaining -= chunk;
                    dst_written += out_size;
                }

                if (seg.conv == Conv::WidenF16) ++re_rounded;
            }
        }

        // Record WeightRef
        {
            WeightRef ref;
            ref.data = dst_cursor;
            ref.total_bytes = r.dst_bytes;
            ref.ne0 = r.ne0;
            ref.ne1 = r.ne1;
            ref.n_elements = r.n_elements;
            ref.kind = r.kind;
            ref.code_bits = r.code_bits;
            ref.code_bias = r.code_bias;
            ref.group_elems = r.group_elems;
            ref.codebook_iq4nl = r.codebook_iq4nl;
            ref.has_offset = r.has_offset;
            ref.codes_bytes = r.codes_bytes;
            ref.scales_bytes = r.scales_bytes;
            ref.offset_bytes = r.offset_bytes;
            ref.src_offset = r.src_offset;
            ref.src_bytes = r.src_bytes;
            ref.file_id = r.file_id;
            ref.scales_fp16 = r.scales_fp16;
            ref.act_kind = r.act_kind;
            table_[r.name] = ref;
        }

        dst_cursor += aligned_dst;
        ++tensors_loaded;
    }

    // Fill report
    {
        auto t1 = std::chrono::steady_clock::now();
        report_.tensors = tensors_loaded;
        report_.arena_bytes = (uint64_t)(dst_cursor - static_cast<uint8_t*>(arena_base));
        report_.re_rounded = re_rounded;
        report_.read_ms = std::chrono::duration<double, std::milli>(t1 - t_read_start).count();
        report_.upload_ms = report_.read_ms;  // combined in this implementation
    }

cleanup:
    for (int i = 0; i < 4; ++i) {
        if (pack_files[i]) std::fclose(pack_files[i]);
    }
    cudaFreeHost(stage_in);
    cudaFreeHost(stage_out);

    return err.empty();
}

// ─── find ────────────────────────────────────────────────────────────
const WeightRef* WeightTable::find(const std::string& name) const {
    auto it = table_.find(name);
    return it != table_.end() ? &it->second : nullptr;
}

}  // namespace strata
