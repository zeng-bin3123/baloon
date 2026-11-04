// weights.cpp - host model of the dense-weight loader (see core_weights.hpp).
#include "baloon/core_weights.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <sstream>

#include "baloon/f16_conversion.hpp"

namespace baloon {
namespace core {
namespace weights_impl {

static double now_ms() {
    using C = std::chrono::steady_clock;
    return std::chrono::duration<double, std::milli>(C::now().time_since_epoch()).count();
}

static std::string file_name(const std::string& dir, const std::string& f) {
    if (dir.empty()) return f;
    return dir.back() == '/' ? dir + f : dir + "/" + f;
}

static bool read_at(std::FILE* f, uint64_t off, uint64_t n, uint8_t* buf) {
    if (n == 0) return true;
    if (std::fseek(f, (long)off, SEEK_SET) != 0) return false;
    return std::fread(buf, 1, (size_t)n, f) == (size_t)n;
}

static void set_err(std::string* e, const std::string& m) { if (e) *e = m; }

}  // namespace weights_impl
using namespace weights_impl;

bool parse_index(const std::string& text, std::vector<WIndexRow>& rows, std::string* err) {
    rows.clear();
    std::istringstream in(text);
    std::string line;
    int ln = 0;
    while (std::getline(in, line)) {
        ++ln;
        if (line.empty() || line[0] == '#') continue;
        std::istringstream ls(line);
        WIndexRow r;
        unsigned long long so, sb, db, c, s, o;
        int fp16;
        if (!(ls >> r.name >> r.kind >> so >> sb >> db >> c >> s >> o >> fp16) || (fp16 != 0 && fp16 != 1)) {
            set_err(err, "bad index line " + std::to_string(ln));
            return false;
        }
        r.src_off = so; r.src_bytes = sb; r.dst_bytes = db; r.codes = c; r.scales = s; r.offset = o;
        r.scales_fp16 = fp16 == 1;
        r.engine_kind = map_kind(r.kind);
        rows.push_back(std::move(r));
    }
    return true;
}

bool plan_segments(const WIndexRow& r, std::vector<WSeg>& segs) {
    segs.clear();
    const bool quantized = r.codes != 0 || r.scales != 0 || r.offset != 0;
    if (quantized) {
        if (r.scales_fp16 && r.scales % 2 != 0) return false;           // an fp16 plane has whole elements
        const uint64_t src_scales = r.scales_fp16 ? r.scales / 2 : r.scales;   // half as large in the pack
        const uint64_t src_total = r.codes + src_scales + r.offset, dst_total = r.codes + r.scales + r.offset;
        if (src_total != r.src_bytes || dst_total != r.dst_bytes) return false;   // checked before any byte moves
        uint64_t so = r.src_off, d = 0;
        segs.push_back({so, d, r.codes, r.codes, WConv::Copy});
        so += r.codes; d += r.codes;
        segs.push_back({so, d, src_scales, r.scales, r.scales_fp16 ? WConv::WidenF16 : WConv::Copy});
        so += src_scales; d += r.scales;
        segs.push_back({so, d, r.offset, r.offset, WConv::Copy});
        return true;
    }
    const WConv c = conv_for(is_raw16(r.kind), r.kind);
    uint64_t want_dst = r.src_bytes;
    if (c == WConv::Bf16High || c == WConv::ToF16) {
        if (r.src_bytes % 4 != 0) return false;
        want_dst = r.src_bytes / 2;                                       // 4 B/elem in, 2 B/elem out
    }
    if (want_dst != r.dst_bytes) return false;
    segs.push_back({r.src_off, 0, r.src_bytes, r.dst_bytes, c});
    return true;
}

uint64_t convert_segment(const WSeg& s, const WReader& read, uint8_t* dst, uint64_t dst_cap, uint64_t chunk) {
    if (!dst || s.dst_off + s.dst_bytes > dst_cap || chunk == 0) return UINT64_MAX;
    // Chunks never split an element: round the chunk down to the element group of the conversion.
    const uint64_t group = (s.conv == WConv::Copy) ? 1 : (s.conv == WConv::WidenF16 ? 2 : 4);
    uint64_t step = chunk - chunk % group;
    if (step == 0) step = group;
    if (s.conv != WConv::Copy && s.src_bytes % group != 0) return UINT64_MAX;
    std::vector<uint8_t> buf((size_t)std::min<uint64_t>(step, s.src_bytes ? s.src_bytes : 1));
    uint64_t done = 0, out = 0;
    while (done < s.src_bytes) {
        const uint64_t n = std::min<uint64_t>(step, s.src_bytes - done);
        if (!read(s.src_off + done, n, buf.data())) return UINT64_MAX;
        uint8_t* o = dst + s.dst_off + out;
        switch (s.conv) {
            case WConv::Copy: std::memcpy(o, buf.data(), (size_t)n); out += n; break;
            case WConv::WidenF16:
                for (uint64_t i = 0; i < n / 2; ++i) {
                    uint16_t h; std::memcpy(&h, buf.data() + 2 * i, 2);
                    const float f = f16::half_to_float(h);
                    std::memcpy(o + 4 * i, &f, 4);
                }
                out += 4 * (n / 2);
                break;
            case WConv::Bf16High:
                for (uint64_t i = 0; i < n / 4; ++i) {
                    float f; std::memcpy(&f, buf.data() + 4 * i, 4);
                    const uint16_t h = f16::float_high_half(f);
                    std::memcpy(o + 2 * i, &h, 2);
                }
                out += 2 * (n / 4);
                break;
            case WConv::ToF16:
                for (uint64_t i = 0; i < n / 4; ++i) {
                    float f; std::memcpy(&f, buf.data() + 4 * i, 4);
                    const uint16_t h = f16::float_to_half(f);
                    std::memcpy(o + 2 * i, &h, 2);
                }
                out += 2 * (n / 4);
                break;
        }
        done += n;
    }
    return out == s.dst_bytes ? out : UINT64_MAX;
}

bool WeightTable::place(const std::string& index_text, const std::set<std::string>& skip, std::string* err) {
    rows_.clear(); pool_ = 0;
    if (!parse_index(index_text, rows_, err)) return false;
    uint64_t at = 0;
    for (auto& r : rows_) {
        if (skip.count(r.name)) { r.placed = false; continue; }          // compaction: skipped rows take no space
        r.placed = true;
        r.arena_off = at;
        at += weight_align256(r.dst_bytes);
    }
    pool_ = at;
    return true;
}

const WIndexRow* WeightTable::find(const std::string& name) const {
    for (auto& r : rows_) if (r.name == name && r.placed) return &r;
    return nullptr;
}

bool WeightTable::load_from(const std::string& index_text, const WReader& read, uint8_t* arena, uint64_t arena_bytes,
                            const std::set<std::string>& skip, std::string* err) {
    if (!place(index_text, skip, err)) return false;
    if (!(pool_ <= arena_bytes)) { set_err(err, "arena smaller than the weight pool"); return false; }
    const double t0 = weights_impl::now_ms();
    for (auto& r : rows_) {
        if (!r.placed) continue;
        std::vector<WSeg> segs;
        if (!plan_segments(r, segs)) { set_err(err, "segment totals disagree with the index for " + r.name); return false; }
        for (auto s : segs) {
            s.dst_off += r.arena_off;
            if (s.dst_bytes == 0 && s.src_bytes == 0) continue;
            if (convert_segment(s, read, arena, arena_bytes) == UINT64_MAX) { set_err(err, "read/convert failed for " + r.name); return false; }
        }
    }
    last_ms_ = weights_impl::now_ms() - t0;
    return true;
}

bool WeightTable::load(const std::string& dir, uint8_t* arena, uint64_t arena_bytes, const std::set<std::string>& skip,
                       std::string* err) {
    std::FILE* fi = std::fopen(file_name(dir, "index.txt").c_str(), "rb");
    if (!fi) { set_err(err, "cannot open index.txt"); return false; }
    std::string text;
    char tmp[4096];
    size_t n;
    while ((n = std::fread(tmp, 1, sizeof tmp, fi)) > 0) text.append(tmp, n);
    std::fclose(fi);
    std::FILE* fp = std::fopen(file_name(dir, "weights.bin").c_str(), "rb");
    if (!fp) { set_err(err, "cannot open weights.bin"); return false; }
    WReader rd = [fp](uint64_t off, uint64_t len, uint8_t* buf) { return read_at(fp, off, len, buf); };
    const bool ok = load_from(text, rd, arena, arena_bytes, skip, err);
    std::fclose(fp);
    return ok;
}

uint64_t WeightTable::pool_bytes() const { return pool_; }

}  // namespace core
}  // namespace baloon
