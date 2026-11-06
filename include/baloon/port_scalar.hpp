// port_scalar.hpp - bf16 bit conversion, NeoX/M-RoPE tables, KV-Q8 cell sizing, native-path switches, small status types.
// Header-only, host + device usable where only integer arithmetic / basic float math is used.
#pragma once
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstring>

#if defined(__CUDACC__) || defined(__HIP__) || defined(__CUDA__)
#define REF_BF16_HD __host__ __device__
#define REF_ROPE_HD __host__ __device__
#else
#define REF_BF16_HD
#define REF_ROPE_HD
#endif

namespace baloon {
namespace refk {

// ---- bf16 bits: the top 16 bits of an fp32, rounded to nearest even on the 16 dropped bits -----------------------------
REF_BF16_HD inline uint16_t bf16_from_f32(float f) {
    uint32_t u; std::memcpy(&u, &f, 4);
    if ((u & 0x7FFFFFFFu) > 0x7F800000u) return (uint16_t)((u >> 16) | 0x0040u);   // NaN stays NaN (quiet), sign kept
    const uint32_t lsb = (u >> 16) & 1u;
    u += 0x7FFFu + lsb;                                                             // carry may reach inf: correct
    return (uint16_t)(u >> 16);
}
REF_BF16_HD inline float f32_from_bf16(uint16_t b) {
    const uint32_t u = (uint32_t)b << 16; float f; std::memcpy(&f, &u, 4); return f;
}

// ---- RoPE (NeoX layout: element i pairs with i + n_rot/2) -----------------------------------------------------------------
REF_ROPE_HD inline void rope_neox_pair(float x0, float x1, float c, float s, float* o0, float* o1) {
    *o0 = x0 * c - x1 * s;
    *o1 = x0 * s + x1 * c;
}
// Rotate the first n_rot dims of one head in place; dims >= n_rot pass through. cos/sin hold n_rot/2 entries.
REF_ROPE_HD inline void rope_neox_apply(float* x, int n_rot, const float* cs, const float* sn) {
    const int half = n_rot / 2;
    for (int i = 0; i < half; ++i) {
        float a, b;
        rope_neox_pair(x[i], x[i + half], cs[i], sn[i], &a, &b);
        x[i] = a; x[i + half] = b;
    }
}
// Table for `pos`: angle_i = pos * theta^(-2i / n_rot).
inline void build_rope_table(float* cs, float* sn, int n_rot, int pos, float theta) {
    const int half = n_rot / 2;
    for (int i = 0; i < half; ++i) {
        const double ang = (double)pos * std::pow((double)theta, -2.0 * (double)i / (double)n_rot);
        cs[i] = (float)std::cos(ang); sn[i] = (float)std::sin(ang);
    }
}

// ---- M-RoPE: the rotary pairs are split in three consecutive sections, each driven by one position axis ------------------
inline std::atomic<int>* mrope_sections_storage() { static std::atomic<int> s[3] = {{11}, {11}, {10}}; return s; }
// Sections must be non-negative; returns false (and keeps the old sections) otherwise.
inline bool mrope_table_set(const int sec[3]) {
    if (sec[0] < 0 || sec[1] < 0 || sec[2] < 0) return false;
    std::atomic<int>* s = mrope_sections_storage();
    for (int k = 0; k < 3; ++k) s[k].store(sec[k]);
    return true;
}
// Position used by rotary pair i. Pairs past the last section reuse the last axis.
inline int mrope_pos(const int pos3[3], const int sec[3], int i) {
    if (i < sec[0]) return pos3[0];
    if (i < sec[0] + sec[1]) return pos3[1];
    return pos3[2];
}
// cos/sin for one token with 3-axis positions, using the sections set by mrope_table_set.
inline void mrope_table(float* cs, float* sn, int n_rot, const int pos3[3], float theta) {
    const std::atomic<int>* s = mrope_sections_storage();
    const int sec[3] = {s[0].load(), s[1].load(), s[2].load()};
    const int half = n_rot / 2;
    for (int i = 0; i < half; ++i) {
        const double ang = (double)mrope_pos(pos3, sec, i) * std::pow((double)theta, -2.0 * (double)i / (double)n_rot);
        cs[i] = (float)std::cos(ang); sn[i] = (float)std::sin(ang);
    }
}

// ---- KV-Q8 cell sizing: int8 codes plus one fp16 scale per KV_Q8_GROUP values --------------------------------------------
constexpr int KV_Q8_GROUP = 64;
inline int64_t kv_q8_bytes_per_cell(int kv_heads, int head_dim) {
    if (kv_heads <= 0 || head_dim <= 0 || head_dim % KV_Q8_GROUP != 0) return 0;
    return (int64_t)kv_heads * head_dim + (int64_t)kv_heads * (head_dim / KV_Q8_GROUP) * 2;
}

// ---- process-wide switches of the native paths (default on) ----------------------------------------------------------------
inline std::atomic<bool>& native_gdn_flag() { static std::atomic<bool> f{true}; return f; }
inline bool native_gdn_enabled() { return native_gdn_flag().load(); }
inline void native_gdn_set_enabled(bool on) { native_gdn_flag().store(on); }
inline std::atomic<bool>& native_moe_combine_flag() { static std::atomic<bool> f{true}; return f; }
inline bool native_moe_combine_enabled() { return native_moe_combine_flag().load(); }
inline void native_moe_combine_set_enabled(bool on) { native_moe_combine_flag().store(on); }
inline std::atomic<bool>& native_qsa_flag() { static std::atomic<bool> f{true}; return f; }
inline bool native_qsa_enabled() { return native_qsa_flag().load(); }
inline void native_qsa_set_enabled(bool on) { native_qsa_flag().store(on); }
inline std::atomic<bool>& native_qsa_indexer_flag() { static std::atomic<bool> f{true}; return f; }
inline bool native_qsa_indexer_enabled() { return native_qsa_indexer_flag().load(); }
inline void native_qsa_indexer_set_enabled(bool on) { native_qsa_indexer_flag().store(on); }
inline std::atomic<bool>& native_qsa_score_flag() { static std::atomic<bool> f{true}; return f; }
inline bool native_qsa_score_enabled() { return native_qsa_score_flag().load(); }
inline void native_qsa_score_set_enabled(bool on) { native_qsa_score_flag().store(on); }
inline std::atomic<bool>& native_rope_flag() { static std::atomic<bool> f{true}; return f; }
inline bool native_rope_enabled() { return native_rope_flag().load(); }
inline void native_rope_set_enabled(bool on) { native_rope_flag().store(on); }
inline std::atomic<bool>& native_router_flag() { static std::atomic<bool> f{true}; return f; }
inline bool native_router_enabled() { return native_router_flag().load(); }
inline void native_router_set_enabled(bool on) { native_router_flag().store(on); }

enum class NativeFlashAttnStatus { Ok = 0, BadShape = 1, TooLong = 2 };

}  // namespace refk
}  // namespace baloon
