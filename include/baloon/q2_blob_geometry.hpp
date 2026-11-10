// q2_blob_geometry.hpp - fixed geometry of one Q2_0 routed expert blob (H=2560, FF=640) and the pure relayout between the three
// role tensors (gate, up, down; GGUF Q2_0 block order) and the four-plane blob: gate/up codes (rows interleaved g0,u0,g1,u1,...),
// down codes, gate/up fp16 scales, down fp16 scales. Every constant is derived from the primitive widths and static_asserted.
#pragma once
#include <cstddef>
#include <cstdint>
#include <cstring>

namespace baloon {
namespace q2blob {

constexpr uint64_t H = 2560, FF = 640, NE = 512;
constexpr uint64_t QK = 64, QKA = 32, BB = 18, CODE_BYTES_PER_BLOCK = 16, SCALE_BYTES = 2;
constexpr uint64_t MAXC = H / QKA;                       // 80 activation chunks
constexpr uint64_t ROW_GU = H * 2 / 8, ROW_D = FF * 2 / 8;  // code bytes per row: 640 / 160
constexpr uint64_t SC_GU = H / QK, SC_D = FF / QK;       // weight blocks per row: 40 / 10
constexpr uint64_t ROLE_BYTES = H * FF / QK * BB;        // 460,800
constexpr uint64_t BLOB = 3 * ROLE_BYTES;                // 1,382,400
constexpr uint64_t O_GU_CODES = 0;
constexpr uint64_t O_D_CODES = O_GU_CODES + 2 * FF * ROW_GU;
constexpr uint64_t O_GU_SCALES = O_D_CODES + H * ROW_D;
constexpr uint64_t O_D_SCALES = O_GU_SCALES + 2 * FF * SC_GU * SCALE_BYTES;
constexpr uint64_t O_END = O_D_SCALES + H * SC_D * SCALE_BYTES;
constexpr uint64_t arena_bytes(uint64_t layers) { return layers * NE * BLOB; }

static_assert(BB == SCALE_BYTES + QK * 2 / 8, "block = fp16 scale + 16 code bytes");
static_assert(QK == 2 * QKA, "a weight block spans exactly two activation chunks");
static_assert(H == MAXC * QKA && MAXC == 80, "hidden width is a whole number of chunks");
static_assert(H / QKA == 2 * SC_GU && FF / QKA == 2 * SC_D && FF / QKA <= MAXC, "chunks are twice blocks, intermediate fits scratch");
static_assert(ROW_GU == SC_GU * CODE_BYTES_PER_BLOCK && ROW_D == SC_D * CODE_BYTES_PER_BLOCK, "16 code bytes per block");
static_assert(ROW_GU == 640 && ROW_D == 160 && SC_GU == 40 && SC_D == 10, "row widths");
static_assert(ROLE_BYTES == 460800 && BLOB == 1382400, "blob size");
static_assert(O_D_CODES == 819200 && O_GU_SCALES == 1228800 && O_D_SCALES == 1331200, "plane offsets");
static_assert(O_END == BLOB, "the four planes tile the blob exactly");
static_assert(O_GU_CODES <= O_D_CODES && O_D_CODES <= O_GU_SCALES && O_GU_SCALES <= O_D_SCALES, "planes ordered");
static_assert(O_GU_SCALES == 3 * (H * FF * 2 / 8) && BLOB - O_GU_SCALES == 3 * (H * FF / QK * 2), "codes and scales totals");
static_assert(2 * FF == 1280 && 2 * FF * SC_GU * SCALE_BYTES == 102400 && H * SC_D * SCALE_BYTES == 51200, "scale planes");
static_assert(arena_bytes(48) == 33973862400ull, "48-layer arena");
static_assert(arena_bytes(2) == arena_bytes(1) + NE * BLOB, "arena grows by 512 blobs per layer");

// Interleaved slot of a gate (role 0) / up (role 1) row.
constexpr uint64_t slot(uint64_t row, uint64_t role) { return 2 * row + role; }

// Byte offsets inside the blob.
constexpr uint64_t gu_code_off(uint64_t slot_) { return O_GU_CODES + slot_ * ROW_GU; }
constexpr uint64_t d_code_off(uint64_t row) { return O_D_CODES + row * ROW_D; }
constexpr uint64_t gu_scale_off(uint64_t slot_, uint64_t b) { return O_GU_SCALES + (slot_ * SC_GU + b) * SCALE_BYTES; }
constexpr uint64_t d_scale_off(uint64_t row, uint64_t b) { return O_D_SCALES + (row * SC_D + b) * SCALE_BYTES; }
// Arena position of expert e of layer l.
constexpr uint64_t expert_offset(uint64_t layer, uint64_t e) { return (layer * NE + e) * BLOB; }

// Pure relayout: three role tensors (ROLE_BYTES each, GGUF Q2_0 block order, one 18-byte block = scale(2) | codes(16)) -> blob.
inline void pack_expert_blob(const uint8_t* gate, const uint8_t* up, const uint8_t* down, uint8_t* blob) {
    for (uint64_t r = 0; r < FF; ++r) {
        const uint8_t* src[2] = {gate + r * SC_GU * BB, up + r * SC_GU * BB};
        for (uint64_t role = 0; role < 2; ++role) {
            const uint64_t s = slot(r, role);
            for (uint64_t b = 0; b < SC_GU; ++b) {
                std::memcpy(blob + gu_code_off(s) + b * CODE_BYTES_PER_BLOCK, src[role] + b * BB + SCALE_BYTES, CODE_BYTES_PER_BLOCK);
                std::memcpy(blob + gu_scale_off(s, b), src[role] + b * BB, SCALE_BYTES);
            }
        }
    }
    for (uint64_t r = 0; r < H; ++r)
        for (uint64_t b = 0; b < SC_D; ++b) {
            const uint8_t* blk = down + (r * SC_D + b) * BB;
            std::memcpy(blob + d_code_off(r) + b * CODE_BYTES_PER_BLOCK, blk + SCALE_BYTES, CODE_BYTES_PER_BLOCK);
            std::memcpy(blob + d_scale_off(r, b), blk, SCALE_BYTES);
        }
}

// Inverse (rebuild_role): recover the three role tensors from a blob.
inline void unpack_expert_blob(const uint8_t* blob, uint8_t* gate, uint8_t* up, uint8_t* down) {
    for (uint64_t r = 0; r < FF; ++r) {
        uint8_t* dst[2] = {gate + r * SC_GU * BB, up + r * SC_GU * BB};
        for (uint64_t role = 0; role < 2; ++role) {
            const uint64_t s = slot(r, role);
            for (uint64_t b = 0; b < SC_GU; ++b) {
                std::memcpy(dst[role] + b * BB, blob + gu_scale_off(s, b), SCALE_BYTES);
                std::memcpy(dst[role] + b * BB + SCALE_BYTES, blob + gu_code_off(s) + b * CODE_BYTES_PER_BLOCK, CODE_BYTES_PER_BLOCK);
            }
        }
    }
    for (uint64_t r = 0; r < H; ++r)
        for (uint64_t b = 0; b < SC_D; ++b) {
            uint8_t* blk = down + (r * SC_D + b) * BB;
            std::memcpy(blk, blob + d_scale_off(r, b), SCALE_BYTES);
            std::memcpy(blk + SCALE_BYTES, blob + d_code_off(r) + b * CODE_BYTES_PER_BLOCK, CODE_BYTES_PER_BLOCK);
        }
}

}  // namespace q2blob
}  // namespace baloon
