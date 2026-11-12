// gguf_reader.hpp - full GGUF (v2/v3) reader: typed metadata key/values, tensor directory, architecture checks.
// Complements gguf_lazy_loader.hpp (which only needs the tensor directory) by decoding the metadata section.
#pragma once
#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#endif
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <map>
#include <optional>
#include <string>
#include <vector>

#if defined(__CUDACC__) || defined(__CUDA__)
#define BALOON_GGUF_HD __host__ __device__
#else
#define BALOON_GGUF_HD
#endif

namespace baloon {
namespace gguf_reader {

// Theory alignUp: ceil(pos / align) * align. For align > 0 the result is a multiple of align, >= pos and
// < pos + align (alignUp_spec). Usable on the device so kernels stride the data section the same way.
BALOON_GGUF_HD inline uint64_t align_up(uint64_t pos, uint64_t align) { return (pos + align - 1) / align * align; }

constexpr uint32_t kMagic = 0x46554747u;  // "GGUF" little-endian
constexpr uint32_t kDefaultAlignment = 32;

// GGUF metadata value types (ids are part of the file format).
enum class MetaType : uint32_t {
    U8 = 0, I8 = 1, U16 = 2, I16 = 3, U32 = 4, I32 = 5, F32 = 6, Bool = 7, String = 8, Array = 9,
    U64 = 10, I64 = 11, F64 = 12
};

inline uint64_t scalar_size(MetaType t) {
    switch (t) {
        case MetaType::U8: case MetaType::I8: case MetaType::Bool: return 1;
        case MetaType::U16: case MetaType::I16: return 2;
        case MetaType::U32: case MetaType::I32: case MetaType::F32: return 4;
        case MetaType::U64: case MetaType::I64: case MetaType::F64: return 8;
        default: return 0;
    }
}

// Bounds-checked little-endian cursor over a byte image. Any overrun sets ok=false and yields zeros.
struct Cursor {
    const uint8_t* p = nullptr; uint64_t n = 0; uint64_t pos = 0; bool ok = true;
    Cursor() = default;
    Cursor(const uint8_t* data, uint64_t size) : p(data), n(size) {}
    // need(k) fails iff pos + k > n (written overflow-free); a failure is sticky and never moves pos.
    bool need(uint64_t k) { if (!ok || pos > n || n - pos < k) { ok = false; return false; } return true; }
    // Theory Cursor.read: need(k) then advance by exactly k. On success pos <= n still holds.
    bool read(uint64_t k) { if (!need(k)) return false; pos += k; return true; }
    // Theory readAll: run the reads in order, stopping at the first failure. On success pos has advanced by
    // exactly the sum of the sizes and is still <= n (readAll_spec); on failure pos stays where the failing read began.
    template <class It> bool read_all(It first, It last) {
        for (; first != last; ++first) if (!read((uint64_t)*first)) return false;
        return true;
    }
    bool read_all(std::initializer_list<uint64_t> ks) { return read_all(ks.begin(), ks.end()); }
    uint64_t uint_le(unsigned bytes) {
        if (!need(bytes)) return 0;
        uint64_t v = 0;
        for (unsigned i = 0; i < bytes; ++i) v |= (uint64_t)p[pos + i] << (8 * i);
        pos += bytes; return v;
    }
    uint8_t u8() { return (uint8_t)uint_le(1); }
    uint16_t u16() { return (uint16_t)uint_le(2); }
    uint32_t u32() { return (uint32_t)uint_le(4); }
    uint64_t u64() { return uint_le(8); }
    std::string str() {
        // string_read_size: a u64 length then that many bytes, i.e. read_all({8, len}) = 8 + len bytes.
        const uint64_t at = pos, l = u64();
        if (!ok || !read(l)) return {};
        return std::string((const char*)p + at + 8, (size_t)l);
    }
};

constexpr uint64_t kMaxArrayElems = 1ull << 24;   // array length cap (16,777,216)
// Theory arrayLenOk: an array header is accepted iff its count is at most 2^24.
inline bool array_len_ok(uint64_t count) { return count <= kMaxArrayElems; }
constexpr uint64_t kArraySample = 64;             // items a summary of an array shows
inline uint64_t array_sample_count(uint64_t count) { return count < kArraySample ? count : kArraySample; }

// A decoded metadata value. Scalars are widened: integers into i (signed) / u (unsigned), floats into f.
struct MetaValue {
    MetaType type = MetaType::U8;
    uint64_t u = 0; int64_t i = 0; double f = 0; bool b = false;
    std::string s;
    MetaType elem = MetaType::U8;          // array element type
    uint64_t count = 0;                    // array length as stored in the file (every element is consumed)
    std::vector<MetaValue> arr;            // theory keptItems: only the first min(count, 64) elements are kept
    uint64_t sample_count() const { return array_sample_count(count); }   // == arr.size(), at most 64
    bool is_integer() const {
        switch (type) { case MetaType::U8: case MetaType::I8: case MetaType::U16: case MetaType::I16:
                        case MetaType::U32: case MetaType::I32: case MetaType::U64: case MetaType::I64: return true;
                        default: return false; }
    }
};

// Array nesting: the top-level value is depth 0; an array read at depth > 2 is refused.
constexpr int kMaxNesting = 2;

// Theory signExtend64: an nb-byte two's-complement field widened to int64. MetaValue stores it in u as
// (uint64_t)i, i.e. modulo 2^64, and (int64_t)u recovers i exactly (signExtend_roundtrip).
inline int64_t sign_extend(uint64_t raw, unsigned nb) {
    if (nb >= 8) return (int64_t)raw;
    const uint64_t sign = 1ull << (8 * nb - 1);
    return (int64_t)(((raw & ((sign << 1) - 1)) ^ sign) - sign);
}

// Theory readBool: booleans are normalised, any non-zero byte reads as true (stored as 1).
inline bool read_bool(uint8_t byte) { return byte != 0; }

// Decode one value of type `t` from the cursor. On malformed data c.ok becomes false.
inline MetaValue read_value(Cursor& c, MetaType t, int depth = 0) {
    MetaValue v; v.type = t;
    switch (t) {
        case MetaType::U8: case MetaType::U16: case MetaType::U32: case MetaType::U64:
            v.u = c.uint_le((unsigned)scalar_size(t)); v.i = (int64_t)v.u; v.f = (double)v.u; break;
        case MetaType::I8: case MetaType::I16: case MetaType::I32: case MetaType::I64: {
            const unsigned nb = (unsigned)scalar_size(t);
            const uint64_t raw = c.uint_le(nb);
            v.i = sign_extend(raw, nb);
            v.u = (uint64_t)v.i; v.f = (double)v.i; break;
        }
        case MetaType::F32: { const uint32_t r = c.u32(); float x; std::memcpy(&x, &r, 4); v.f = x; break; }
        case MetaType::F64: { const uint64_t r = c.u64(); double x; std::memcpy(&x, &r, 8); v.f = x; break; }
        case MetaType::Bool: { const uint8_t r = c.u8(); v.b = read_bool(r); v.u = v.b ? 1 : 0; break; }
        case MetaType::String: v.s = c.str(); break;
        case MetaType::Array: {
            if (depth > kMaxNesting) { c.ok = false; break; }
            const uint32_t et = c.u32(); const uint64_t cnt = c.u64();
            if (!c.ok || et > 12) { c.ok = false; break; }
            v.elem = (MetaType)et;
            const uint64_t sz = scalar_size(v.elem);
            if (!array_len_ok(cnt) || (sz && cnt > c.n / sz)) { c.ok = false; break; }
            v.count = cnt;
            v.arr.reserve((size_t)array_sample_count(cnt));
            for (uint64_t k = 0; k < cnt && c.ok; ++k) {
                MetaValue e = read_value(c, v.elem, depth + 1);   // every element is decoded (and bounds-checked)
                if (k < kArraySample) v.arr.push_back(std::move(e));
            }
            break;
        }
        default: c.ok = false; break;
    }
    return v;
}

// Theory effectiveAlign: general.alignment if present and non-zero, otherwise the default 32. Never zero, so
// align_up is always well defined (effectiveAlign_pos).
inline uint32_t effective_alignment(std::optional<uint64_t> key) {
    return key && *key != 0 ? (uint32_t)*key : kDefaultAlignment;
}

// ---- tensor types -------------------------------------------------------------------------------------
struct BlockGeometry { uint64_t el = 0, by = 0; };  // el == 0 means unknown type

BALOON_GGUF_HD constexpr BlockGeometry block_geometry(uint32_t type) {
    switch (type) {
        case 0: return {1, 4};      case 1: return {1, 2};     case 30: return {1, 2};
        case 2: return {32, 18};    case 3: return {32, 20};   case 6: return {32, 22};
        case 7: return {32, 24};    case 8: return {32, 34};   case 9: return {32, 36};
        case 10: return {256, 84};  case 11: return {256, 110}; case 12: return {256, 144};
        case 13: return {256, 176}; case 14: return {256, 210};
        case 16: return {256, 66};  case 17: return {256, 74}; case 18: return {256, 98};
        case 20: return {32, 18};  case 21: return {256, 110};
        case 22: return {256, 82};  case 23: return {256, 136}; case 29: return {256, 56};
        case 42: return {64, 18};   // Q2_0 (theory geom 42)
        default: return {0, 0};
    }
}

// Theory: every listed quantized type's block bytes = its header fields + its packed payload. The layouts below are
// the per-block fields the dequantizers read (d/m are fp16 = 2 bytes; QK = elements per block); the table above
// must match them, checked at compile time.
constexpr uint64_t block_layout_bytes(uint32_t type) {
    switch (type) {
        case 2:  return 2 + 32 / 2;                        // Q4_0:  d, qs[QK/2]
        case 3:  return 2 + 2 + 32 / 2;                    // Q4_1:  d, m, qs[QK/2]
        case 6:  return 2 + 4 + 32 / 2;                    // Q5_0:  d, qh[4], qs[QK/2]
        case 7:  return 2 + 2 + 4 + 32 / 2;                // Q5_1:  d, m, qh[4], qs[QK/2]
        case 8:  return 2 + 32;                            // Q8_0:  d, qs[QK]
        case 9:  return 2 + 2 + 32;                        // Q8_1:  d, s, qs[QK]
        case 10: return 256 / 16 + 256 / 4 + 2 + 2;        // Q2_K:  scales[16], qs[64], d, dmin
        case 11: return 256 / 8 + 256 / 4 + 12 + 2;        // Q3_K:  hmask[32], qs[64], scales[12], d
        case 12: return 2 + 2 + 12 + 256 / 2;              // Q4_K:  d, dmin, scales[12], qs[128]
        case 13: return 2 + 2 + 12 + 256 / 8 + 256 / 2;    // Q5_K:  d, dmin, scales[12], qh[32], qs[128]
        case 14: return 256 / 2 + 256 / 4 + 256 / 16 + 2;  // Q6_K:  ql[128], qh[64], scales[16], d
        case 16: return 2 + 256 / 4;                       // IQ2_XXS: d, qs[64]
        case 17: return 2 + 256 / 4 + 256 / 32;            // IQ2_XS:  d, qs[64], scales[8]
        case 18: return 2 + 3 * 256 / 8;                   // IQ3_XXS: d, qs[96]
        case 20: return 2 + 32 / 2;                        // IQ4_NL:  d, qs[QK/2]
        case 21: return 2 + 256 / 4 + 256 / 32 + 256 / 8 + 256 / 64;   // IQ3_S: d, qs[64], qh[8], signs[32], scales[4]
        case 22: return 2 + 256 / 4 + 256 / 32 + 256 / 32; // IQ2_S:   d, qs[64], qh[8], scales[8]
        case 23: return 2 + 2 + 256 / 64 + 256 / 2;        // IQ4_XS:  d, scales_h, scales_l[4], qs[128]
        case 29: return 256 / 8 + 256 / 16 + 256 / 32;     // IQ1_M:   qs[32], qh[16], scales[8]
        case 42: return 2 + 64 * 2 / 8;                    // Q2_0:    d, 64 two-bit codes
        default: return 0;                                 // not a quantized type
    }
}
constexpr bool geometry_matches_layouts() {
    for (uint32_t t : {2u, 3u, 6u, 7u, 8u, 9u, 10u, 11u, 12u, 13u, 14u, 16u, 17u, 18u, 20u, 21u, 22u, 23u, 29u, 42u})
        if (block_geometry(t).el == 0 || block_geometry(t).by != block_layout_bytes(t)) return false;
    return true;
}
static_assert(geometry_matches_layouts(), "block_geometry bytes must equal header fields + packed payload");
static_assert(block_geometry(42).el == 64 && block_geometry(42).by == 18, "Q2_0 = (64, 18)");

// Theory rowBytes: bytes of n elements stored as whole blocks of (el elements, by bytes).
BALOON_GGUF_HD inline uint64_t row_bytes(uint64_t el, uint64_t by, uint64_t n) { return n / el * by; }

inline const char* ggml_type_name(uint32_t type) {
    switch (type) {
        case 0: return "F32"; case 1: return "F16"; case 2: return "Q4_0"; case 3: return "Q4_1";
        case 6: return "Q5_0"; case 7: return "Q5_1"; case 8: return "Q8_0"; case 9: return "Q8_1";
        case 10: return "Q2_K"; case 11: return "Q3_K"; case 12: return "Q4_K"; case 13: return "Q5_K";
        case 14: return "Q6_K"; case 15: return "Q8_K"; case 16: return "IQ2_XXS"; case 17: return "IQ2_XS";
        case 18: return "IQ3_XXS"; case 19: return "IQ1_S"; case 20: return "IQ4_NL"; case 21: return "IQ3_S";
        case 22: return "IQ2_S"; case 23: return "IQ4_XS"; case 24: return "I8"; case 25: return "I16";
        case 26: return "I32"; case 27: return "I64"; case 28: return "F64"; case 29: return "IQ1_M";
        case 30: return "BF16"; case 42: return "Q2_0";
        default: return "UNKNOWN";
    }
}

struct TensorInfo {
    std::string name;
    std::vector<uint64_t> shape;  // GGUF order, fastest axis first
    uint32_t type = 0;
    uint64_t offset = 0;          // relative to the data section
    uint64_t elements() const { uint64_t n = 1; for (auto d : shape) n *= d; return n; }
    uint64_t nbytes() const { auto g = block_geometry(type); return g.el ? row_bytes(g.el, g.by, elements()) : 0; }
};

// ---- the file ---------------------------------------------------------------------------------------------
struct GgufFile {
    uint32_t version = 0;
    uint32_t alignment = kDefaultAlignment;
    uint64_t data_start = 0;  // absolute offset of the (aligned) data section
    std::vector<std::string> meta_order;
    std::map<std::string, MetaValue> meta;
    std::vector<TensorInfo> tensors;

    const MetaValue* find_meta(const std::string& k) const { auto it = meta.find(k); return it == meta.end() ? nullptr : &it->second; }
    const TensorInfo* find_tensor(const std::string& n) const { for (auto& t : tensors) if (t.name == n) return &t; return nullptr; }
    std::optional<std::string> get_string(const std::string& k) const {
        auto m = find_meta(k); if (!m || m->type != MetaType::String) return std::nullopt; return m->s;
    }
    std::optional<uint64_t> get_uint(const std::string& k) const {
        auto m = find_meta(k); if (!m || !m->is_integer() || (m->i < 0 && m->type != MetaType::U64)) return std::nullopt; return m->u;
    }

    // Parse an in-memory image. nullopt on malformed input (bad magic/version, truncation, duplicate keys,
    // unknown tensor types, tensors that do not fit inside the image).
    static std::optional<GgufFile> parse(const uint8_t* data, uint64_t size) {
        Cursor c(data, size);
        if (c.u32() != kMagic) return std::nullopt;
        GgufFile f; f.version = c.u32();
        if (f.version != 3) return std::nullopt;
        const uint64_t nt = c.u64(), nkv = c.u64();
        if (!c.ok || nt > (1u << 24) || nkv > (1u << 24)) return std::nullopt;
        for (uint64_t i = 0; i < nkv; ++i) {
            std::string key = c.str();
            const uint32_t t = c.u32();
            if (!c.ok || t > 12) return std::nullopt;
            MetaValue v = read_value(c, (MetaType)t);
            if (!c.ok) return std::nullopt;
            if (f.meta.count(key)) return std::nullopt;
            f.meta_order.push_back(key);
            f.meta.emplace(std::move(key), std::move(v));
        }
        std::optional<uint64_t> align_key;
        if (f.find_meta("general.alignment")) {
            align_key = f.get_uint("general.alignment");               // any non-negative integer type
            if (!align_key || *align_key > 0xFFFFFFFFull) return std::nullopt;   // a non-integer alignment is malformed
        }
        f.alignment = effective_alignment(align_key);
        for (uint64_t i = 0; i < nt; ++i) {
            TensorInfo ti; ti.name = c.str();
            const uint32_t nd = c.u32();
            if (!c.ok || nd == 0 || nd > 4) return std::nullopt;   // 1..4 dimensions
            for (uint32_t d = 0; d < nd; ++d) ti.shape.push_back(c.u64());
            ti.type = c.u32(); ti.offset = c.u64();
            if (!c.ok) return std::nullopt;
            f.tensors.push_back(std::move(ti));
        }
        const uint64_t a = f.alignment;
        f.data_start = align_up(c.pos, a);   // data_start = ceil(header_end / align) * align
        if (f.data_start > size) return std::nullopt;
        for (auto& t : f.tensors) {
            const auto g = block_geometry(t.type);
            if (g.el == 0 || t.elements() % g.el != 0) return std::nullopt;
            if (t.offset % a != 0) return std::nullopt;
            if (t.offset > size - f.data_start || t.nbytes() > size - f.data_start - t.offset) return std::nullopt;
        }
        return f;
    }
};

// ---- architecture checks --------------------------------------------------------------------------------------
// The architecture string lives in "general.architecture"; per-arch keys are prefixed with it.
inline bool check_architecture(const GgufFile& f, const std::string& expected) {
    auto a = f.get_string("general.architecture");
    return a && *a == expected;
}

// Architecture guard constants of the qwen4exp layout (theory: guardDefaults).
constexpr uint64_t kGuardExperts = 512, kGuardTopK = 10, kGuardHeads = 24, kGuardKvHeads = 2, kGuardLayers = 48, kGuardHidden = 2560;
constexpr const char* kGuardArch = "qwen4exp";
static_assert(kGuardTopK <= kGuardExperts && kGuardHeads % kGuardKvHeads == 0 && kGuardHeads / kGuardKvHeads == 12 &&
              kGuardLayers > 0, "guard constants must be mutually consistent");

// Theory guardDefaults: the six required keys, in checking order, with their compiled-in values.
struct GuardReq { const char* key; uint64_t want; };
constexpr GuardReq kGuardDefaults[6] = {
    {"qwen4exp.block_count", kGuardLayers},          {"qwen4exp.embedding_length", kGuardHidden},
    {"qwen4exp.expert_count", kGuardExperts},        {"qwen4exp.expert_used_count", kGuardTopK},
    {"qwen4exp.attention.head_count", kGuardHeads},  {"qwen4exp.attention.head_count_kv", kGuardKvHeads}};

// Theory checkReqs + checkArchitecture: "" iff general.architecture is "qwen4exp" and every kGuardDefaults key is
// present with exactly its value; otherwise the name of the FIRST failure ("general.architecture" or the key).
inline std::string check_architecture(const GgufFile& f) {
    const auto arch = f.get_string("general.architecture");
    if (!arch || *arch != kGuardArch) return "general.architecture";
    for (const GuardReq& r : kGuardDefaults) {
        const auto v = f.get_uint(r.key);
        if (!v || *v != r.want) return r.key;
    }
    return "";
}


// Guard for the qwen4exp layout: general.architecture must be exactly "qwen4exp" (no other "qwen*" string passes)
// and the "qwen4exp.*" keys must carry exactly the six guard constants: expert_count 512, expert_used_count 10,
// attention.head_count 24, attention.head_count_kv 2, block_count 48, embedding_length 2560. All keys are required.
// The fields default to the constants; tests may override them to exercise other fixtures.
struct Qwen4ExpGuard {
    uint64_t expected_experts = kGuardExperts;
    uint64_t expected_used = kGuardTopK;
    uint64_t expected_heads = kGuardHeads;
    uint64_t expected_kv_heads = kGuardKvHeads;
    uint64_t expected_layers = kGuardLayers;
    uint64_t expected_hidden = kGuardHidden;
    bool check(const GgufFile& f, std::string* why = nullptr) const {
        auto fail = [&](const char* m) { if (why) *why = m; return false; };
        auto arch = f.get_string("general.architecture");
        if (!arch) return fail("missing general.architecture");
        if (*arch != kGuardArch) return fail("architecture is not qwen4exp");
        auto ne = f.get_uint(*arch + ".expert_count");
        if (!ne) return fail("missing expert_count");
        if (*ne != expected_experts) return fail("expert_count mismatch");
        auto used = f.get_uint(*arch + ".expert_used_count");
        if (!used) return fail("missing expert_used_count");
        if (*used == 0 || *used > *ne) return fail("expert_used_count out of range");
        if (*used != expected_used) return fail("expert_used_count mismatch");
        auto nh = f.get_uint(*arch + ".attention.head_count");
        if (!nh) return fail("missing head_count");
        if (*nh != expected_heads) return fail("head_count mismatch");
        auto nkv = f.get_uint(*arch + ".attention.head_count_kv");
        if (!nkv) return fail("missing head_count_kv");
        if (*nkv != expected_kv_heads) return fail("head_count_kv mismatch");
        if (*nkv == 0 || *nh % *nkv != 0) return fail("heads do not group over kv heads");
        auto nl = f.get_uint(*arch + ".block_count");
        if (!nl) return fail("missing block_count");
        if (*nl != expected_layers) return fail("block_count mismatch");
        auto hid = f.get_uint(*arch + ".embedding_length");
        if (!hid) return fail("missing embedding_length");
        if (*hid != expected_hidden) return fail("embedding_length mismatch");
        return true;
    }
};

// Theory: Qwen4ExpGuard's defaults are exactly guardDefaults, so the default guard and check_architecture agree.
static_assert(Qwen4ExpGuard{}.expected_layers == kGuardDefaults[0].want && Qwen4ExpGuard{}.expected_hidden == kGuardDefaults[1].want &&
              Qwen4ExpGuard{}.expected_experts == kGuardDefaults[2].want && Qwen4ExpGuard{}.expected_used == kGuardDefaults[3].want &&
              Qwen4ExpGuard{}.expected_heads == kGuardDefaults[4].want && Qwen4ExpGuard{}.expected_kv_heads == kGuardDefaults[5].want,
              "Qwen4ExpGuard defaults must equal kGuardDefaults");

}  // namespace gguf_reader
}  // namespace baloon
