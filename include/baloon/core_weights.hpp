// core_weights.hpp - the dense-weight loader (host model): reads index.txt, places each tensor in the arena (256-byte
// aligned, optionally compacted by a skip set) and streams it plane by plane in 8 MiB chunks, converting widths.
// A quantized tensor is segments codes | scales | offsets; an fp16 scale plane is half as large in the pack as in the
// arena and is widened exactly (fp16 -> f32). Segment totals are checked against the index's src_bytes/dst_bytes before
// any byte moves. Index kinds 4/5 (raw 16-bit BF16/F16) are copied and take engine forms 1 and 3; promoted rows (kind
// 1/3 from f32 containers) are 4 B/elem in and 2 B/elem out.
//
// index.txt line:  name kind src_off src_bytes dst_bytes codes scales offset scales_fp16
// (codes/scales/offset = 0 for unquantized rows; scales_fp16 is 0 or 1).
#pragma once
#include <cstddef>
#include <cstdint>
#include <functional>
#include <set>
#include <string>
#include <vector>

namespace baloon {
namespace core {

constexpr uint64_t kWeightChunk = 8388608ull;     // 8 MiB of source per staging round

enum class WConv { Copy, WidenF16, Bf16High, ToF16 };

struct WSeg { uint64_t src_off = 0, dst_off = 0, src_bytes = 0, dst_bytes = 0; WConv conv = WConv::Copy; };

struct WIndexRow {
    std::string name;
    int kind = 0;                                  // index kind as written
    uint64_t src_off = 0, src_bytes = 0, dst_bytes = 0;
    uint64_t codes = 0, scales = 0, offset = 0;
    bool scales_fp16 = false;
    // filled by load():
    int engine_kind = 0;                           // mapKind(kind)
    bool placed = false;
    uint64_t arena_off = 0;
};

constexpr int map_kind(int k) { return k == 4 ? 1 : (k == 5 ? 3 : k); }
constexpr bool is_raw16(int k) { return k == 4 || k == 5; }
constexpr uint64_t chunk_len_of(uint64_t rem) { return rem < kWeightChunk ? rem : kWeightChunk; }
constexpr uint64_t weight_align256(uint64_t n) { return (n + 255) / 256 * 256; }
// Conversion applied to an unquantized row: raw 16-bit rows copy, promoted BF16 takes the high half, promoted F16 converts.
constexpr WConv conv_for(bool raw16, int kind) {
    return raw16 ? WConv::Copy : (kind == 1 ? WConv::Bf16High : (kind == 3 ? WConv::ToF16 : WConv::Copy));
}

// Segment list of a row; false when the segment totals disagree with src_bytes / dst_bytes.
bool plan_segments(const WIndexRow& r, std::vector<WSeg>& segs);
// Stream one segment (chunked) from `src` (read via the reader) into dst. Returns bytes written, UINT64_MAX on error.
using WReader = std::function<bool(uint64_t off, uint64_t n, uint8_t* buf)>;
uint64_t convert_segment(const WSeg& s, const WReader& read, uint8_t* dst, uint64_t dst_cap, uint64_t chunk = kWeightChunk);

bool parse_index(const std::string& text, std::vector<WIndexRow>& rows, std::string* err = nullptr);

class WeightTable {
public:
    // Reads dir/index.txt and dir/weights.bin; rows named in `skip` are not placed (the arena is compacted).
    bool load(const std::string& dir, uint8_t* arena, uint64_t arena_bytes, const std::set<std::string>& skip = {},
              std::string* err = nullptr);
    bool load_from(const std::string& index_text, const WReader& read, uint8_t* arena, uint64_t arena_bytes,
                   const std::set<std::string>& skip = {}, std::string* err = nullptr);
    // Placement only (no bytes move): fills arena offsets and pool_bytes().
    bool place(const std::string& index_text, const std::set<std::string>& skip = {}, std::string* err = nullptr);
    const WIndexRow* find(const std::string& name) const;
    uint64_t pool_bytes() const;
    double last_load_ms() const { return last_ms_; }
    const std::vector<WIndexRow>& rows() const { return rows_; }

private:
    std::vector<WIndexRow> rows_;
    uint64_t pool_ = 0;
    double last_ms_ = 0;
};

}  // namespace core
}  // namespace baloon
