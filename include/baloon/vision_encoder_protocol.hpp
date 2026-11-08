// vision_encoder_protocol.hpp - wire format and command handling of the resident image encoder:
// `ENC <image> <output>` (image path ends at the LAST space), 20-byte header {SVE1, n_tokens, nx, ny, n_embd} of
// int32 little-endian, then n_tokens*n_embd float32; grid recovered from the last token's position.
#pragma once
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace baloon {
namespace vision_enc {

constexpr uint32_t MAGIC = 0x31455653u;  // bytes 'S','V','E','1' little-endian
constexpr uint64_t HEADER_BYTES = 20;

// sp = index of the last space in the remainder, len = remainder length.
inline bool accepts(uint64_t sp, uint64_t len) { return 0 < sp && sp + 1 < len; }

struct Command { bool ok = false; std::string image, output; };

// rest = remainder after "ENC ". Split at the last space; both parts must be non-empty.
inline Command parse_enc(const std::string& rest) {
    Command c;
    const size_t sp = rest.rfind(' ');
    if (sp == std::string::npos || !accepts(sp, rest.size())) return c;
    c.ok = true; c.image = rest.substr(0, sp); c.output = rest.substr(sp + 1);
    return c;
}

inline uint64_t file_bytes(uint64_t n, uint64_t e) { return HEADER_BYTES + n * (4 * e); }

struct Header { int32_t magic, n_tokens, nx, ny, n_embd; };

inline void put_i32(uint8_t* p, int32_t v) {
    const uint32_t u = (uint32_t)v;
    for (int i = 0; i < 4; ++i) p[i] = (uint8_t)(u >> (8 * i));
}
inline int32_t get_i32(const uint8_t* p) {
    uint32_t u = 0;
    for (int i = 0; i < 4; ++i) u |= (uint32_t)p[i] << (8 * i);
    return (int32_t)u;
}
inline std::vector<uint8_t> encode_header(int32_t n_tokens, int32_t nx, int32_t ny, int32_t n_embd) {
    std::vector<uint8_t> b(HEADER_BYTES);
    put_i32(&b[0], (int32_t)MAGIC); put_i32(&b[4], n_tokens); put_i32(&b[8], nx); put_i32(&b[12], ny); put_i32(&b[16], n_embd);
    return b;
}
inline bool decode_header(const uint8_t* p, size_t n, Header& h) {
    if (n < HEADER_BYTES) return false;
    h = {get_i32(p), get_i32(p + 4), get_i32(p + 8), get_i32(p + 12), get_i32(p + 16)};
    return (uint32_t)h.magic == MAGIC;
}

// Grid from the last token's decoder position (x, y): nx = x+1, ny = y+1.
inline void grid_from_last(int32_t x, int32_t y, int32_t& nx, int32_t& ny) { nx = x + 1; ny = y + 1; }
inline bool grid_ok(uint64_t nx, uint64_t ny, uint64_t n) { return nx * ny == n; }
inline void row_pos(uint64_t i, uint64_t nx, uint64_t& x, uint64_t& y) { x = i % nx; y = i / nx; }


// Reply lines. Success: "OK n nx ny ms"; failure: "ERR msg" (msg never empty).
inline std::string ok_reply(int64_t n, int64_t nx, int64_t ny, int64_t ms) {
    return "OK " + std::to_string(n) + " " + std::to_string(nx) + " " + std::to_string(ny) + " " + std::to_string(ms);
}
inline std::string err_reply(const std::string& msg) { return "ERR " + (msg.empty() ? std::string("error") : msg); }

// Index of the last token on a rectangular grid, and the grid recovered from that token's position.
inline uint64_t last_index(uint64_t nx, uint64_t ny) { return nx * ny - 1; }

// Numeric option overrides the default only when positive.
inline int64_t effective(int64_t req, int64_t dflt) { return req > 0 ? req : dflt; }
inline bool width_ok(int64_t n_embd) { return n_embd > 0; }

// Whole-file builder / validator for the output.
inline std::vector<uint8_t> build_file(int32_t nx, int32_t ny, int32_t n_embd, const std::vector<float>& emb) {
    std::vector<uint8_t> b = encode_header(nx * ny, nx, ny, n_embd);
    b.resize(file_bytes((uint64_t)nx * ny, (uint64_t)n_embd));
    std::memcpy(b.data() + HEADER_BYTES, emb.data(), emb.size() * 4);
    return b;
}
inline bool validate_file(const std::vector<uint8_t>& b) {
    Header h;
    if (!decode_header(b.data(), b.size(), h)) return false;
    if (h.n_tokens < 0 || h.nx <= 0 || h.ny <= 0 || !width_ok(h.n_embd)) return false;
    if (!grid_ok((uint64_t)h.nx, (uint64_t)h.ny, (uint64_t)h.n_tokens)) return false;
    return b.size() == file_bytes((uint64_t)h.n_tokens, (uint64_t)h.n_embd);
}

}  // namespace vision_enc
}  // namespace baloon
