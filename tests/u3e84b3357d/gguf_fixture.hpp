// gguf_fixture.hpp - shared test harness for unit u3e84b3357d (GgufHeaderParsing.lean):
// a little-endian byte writer, a GGUF v3 image builder and a section registry, so that every
// theory statement is one registered section of the host test.
#pragma once
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace u3e {

inline long g_checks = 0, g_fail = 0;
#define CHECK(x) do { ++u3e::g_checks; if (!(x)) { ++u3e::g_fail; std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #x); } } while (0)

struct Section { const char* name; void (*fn)(); };
inline std::vector<Section>& sections() { static std::vector<Section> s; return s; }
struct Reg { Reg(const char* n, void (*f)()) { sections().push_back({n, f}); } };
#define U3E_SECTION(id, title) static void id(); static u3e::Reg id##_reg(title, id); static void id()

struct W {
    std::vector<uint8_t> b;
    void u(uint64_t v, int nb) { for (int i = 0; i < nb; ++i) b.push_back((uint8_t)(v >> (8 * i))); }
    void s(const std::string& x) { u(x.size(), 8); b.insert(b.end(), x.begin(), x.end()); }
};

// One metadata entry: type 4 = U32, 8 = String, 7 = Bool (u holds the byte), 10 = U64,
// 9 = Array (s holds the raw array payload: element type, count, elements).
struct Kv { std::string key; uint32_t type; uint64_t u; std::string s; };

// GGUF v3 image with the given metadata and one tensor "w" of `ttype`/`shape` at offset 0; `*hdr_end`
// receives the end of the tensor directory. The tensor bytes are appended after the aligned data start.
inline std::vector<uint8_t> image(const std::vector<Kv>& kv, uint32_t ttype = 42, std::vector<uint64_t> shape = {64},
                                  uint64_t align = 32, size_t* hdr_end = nullptr, uint64_t tensor_bytes = 18) {
    W w; w.u(0x46554747u, 4); w.u(3, 4); w.u(1, 8); w.u(kv.size(), 8);
    for (auto& e : kv) {
        w.s(e.key); w.u(e.type, 4);
        if (e.type == 8) w.s(e.s);
        else if (e.type == 9) w.b.insert(w.b.end(), e.s.begin(), e.s.end());
        else if (e.type == 7) w.u(e.u, 1);
        else if (e.type == 10) w.u(e.u, 8);
        else w.u(e.u, 4);
    }
    w.s("w"); w.u(shape.size(), 4); for (auto d : shape) w.u(d, 8); w.u(ttype, 4); w.u(0, 8);
    if (hdr_end) *hdr_end = w.b.size();
    while (w.b.size() % align) w.b.push_back(0);
    w.b.resize(w.b.size() + tensor_bytes, 0);
    return w.b;
}

// The six required qwen4exp keys, in the theory's guardDefaults order, with the architecture string first.
inline std::vector<Kv> guard_kvs() {
    return {{"general.architecture", 8, 0, "qwen4exp"},     {"qwen4exp.block_count", 4, 48, ""},
            {"qwen4exp.embedding_length", 4, 2560, ""},     {"qwen4exp.expert_count", 4, 512, ""},
            {"qwen4exp.expert_used_count", 4, 10, ""},      {"qwen4exp.attention.head_count", 4, 24, ""},
            {"qwen4exp.attention.head_count_kv", 4, 2, ""}};
}

}  // namespace u3e
