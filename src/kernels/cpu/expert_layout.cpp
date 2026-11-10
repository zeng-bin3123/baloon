// expert_layout.cpp - expert blob layout (gate rows | up rows | down rows) and the format-generic row entry points.
// The layout arithmetic is computed here from the block tables; the compute entry points defer to the engine kernels.
#include "baloon/u11_expert.hpp"

namespace baloon {
namespace u11 {
namespace layout {

namespace {
struct Blk { int64_t values, bytes; };
Blk block_of(int type) {
    switch (type) {
        case 42: return {64, 18};  case 16: return {256, 66};  case 17: return {256, 74};
        case 18: return {256, 98}; case 21: return {256, 110}; case 22: return {256, 82};
        default: return {0, 0};
    }
}
int64_t row_of(int type, int64_t n_in) {
    const Blk b = block_of(type);
    return (b.values == 0 || n_in <= 0 || n_in % b.values) ? 0 : n_in / b.values * b.bytes;
}
}  // namespace

ExpertLayout expert_layout(int type, int64_t n_embd, int64_t n_ff) {
    ExpertLayout l;
    const int64_t gu = row_of(type, n_embd), dn = row_of(type, n_ff);
    if (gu == 0 || dn == 0) return l;
    l.type = type; l.n_embd = n_embd; l.n_ff = n_ff; l.gu_row_bytes = gu; l.down_row_bytes = dn;
    l.up_off = n_ff * gu; l.down_off = 2 * l.up_off; l.bytes = l.down_off + n_embd * dn; l.ok = true;
    return l;
}

bool expert_layout_load(ExpertLayout& out, int type, int64_t n_embd, int64_t n_ff, uint64_t blob_bytes) {
    const ExpertLayout l = expert_layout(type, n_embd, n_ff);
    if (!l.ok || blob_bytes != (uint64_t)l.bytes) { out = ExpertLayout{}; return false; }
    out = l;
    return true;
}

bool cpu_avx512_ok() { return baloon::cpu_avx512_ok(); }
void act_quant_any(const float* x, uint32_t n, int8_t* q, float* d, int32_t* hx) { baloon::act_quant_any(x, n, q, d, hx); }
bool q2_rows_any(const uint8_t* w, uint32_t row0, uint32_t row1, uint32_t nb, uint32_t nt, const int8_t* q, size_t q_stride,
                 const float* xd, const int32_t* hx, size_t ch_stride, float* out, size_t out_stride) {
    return baloon::q2_rows_any(w, row0, row1, nb, nt, q, q_stride, xd, hx, ch_stride, out, out_stride);
}
NativeFmt native_fmt(int t) { return baloon::native_fmt(t); }
bool native_experts_available(int t, int64_t e, int64_t f) { return baloon::native_experts_available(t, e, f); }
bool native_quant_act(NativeFmt f, const float* x, uint32_t n, uint32_t nt, void* out) { return baloon::native_quant_act(f, x, n, nt, out); }
bool native_quant_h(NativeFmt f, const float* h, uint32_t n, uint32_t nt, void* out) { return baloon::native_quant_h(f, h, n, nt, out); }
bool native_gu_rows(NativeFmt f, const uint8_t* blob, uint32_t e, uint32_t ff, uint32_t r0, uint32_t r1, const void* xq, uint32_t nt,
                    float* act, size_t st) { return baloon::native_gu_rows(f, blob, e, ff, r0, r1, xq, nt, act, st); }
bool native_down_rows(NativeFmt f, const uint8_t* blob, uint32_t e, uint32_t ff, uint32_t r0, uint32_t r1, const void* hq, uint32_t nt,
                      float* out, size_t st) { return baloon::native_down_rows(f, blob, e, ff, r0, r1, hq, nt, out, st); }

}  // namespace layout
}  // namespace u11
}  // namespace baloon
