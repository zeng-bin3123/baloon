// strata/kernels/cpu/expert_layout.cpp — Expert format dispatch and layout
// Target: Iluvatar BI-V100 host CPU
//
// NativeFmt discovery for ggml-cpu delegation.
// AVX-512 capability check for the IQ multi-token acceleration path.

#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/core/cpu_features.hpp"

#include <cstring>
#include <string>

namespace strata { namespace kernels { namespace cpu {

// ─── ggml type traits (minimal subset for expert dispatch) ──────
// These match ggml's type enumeration.  We only need to know:
// - block size in bytes (for row_size calculation)
// - whether vec_dot is available
// - which activation type is needed

namespace {

// ggml type IDs (subset)
enum GgmlType {
    GGML_TYPE_F32  = 0,
    GGML_TYPE_F16  = 1,
    GGML_TYPE_Q4_0 = 2,
    GGML_TYPE_Q4_1 = 3,
    GGML_TYPE_Q5_0 = 6,
    GGML_TYPE_Q5_1 = 7,
    GGML_TYPE_Q8_0 = 8,
    GGML_TYPE_Q8_1 = 9,
    GGML_TYPE_Q2_K = 10,
    GGML_TYPE_Q3_K = 11,
    GGML_TYPE_Q4_K = 12,
    GGML_TYPE_Q5_K = 13,
    GGML_TYPE_Q6_K = 14,
    GGML_TYPE_Q8_K = 15,
    GGML_TYPE_IQ2_XXS = 16,
    GGML_TYPE_IQ2_XS  = 17,
    GGML_TYPE_IQ3_XXS = 18,
    GGML_TYPE_IQ1_S   = 19,
    GGML_TYPE_IQ4_NL  = 20,
    GGML_TYPE_IQ3_S   = 21,
    GGML_TYPE_IQ2_S   = 22,
    GGML_TYPE_IQ4_XS  = 23,
    GGML_TYPE_IQ1_M   = 24,
    GGML_TYPE_Q2_0    = 30,  // custom 2-bit
};

// Block sizes for row_size calculation
struct TypeInfo {
    int block_size;    // values per block
    int block_bytes;   // bytes per block
    int act_type;      // activation quantization type needed
    bool has_dot;      // whether vec_dot is available
};

// Minimal type table (the ones that appear as expert weights)
const TypeInfo* type_info(int type_id) {
    static const TypeInfo table[] = {
        // block_size, block_bytes, act_type, has_dot
        {1, 4, GGML_TYPE_F32, true},       // F32 = 0
        {1, 2, GGML_TYPE_F16, true},        // F16 = 1
        {32, 18, GGML_TYPE_Q8_1, true},     // Q4_0 = 2
        {32, 20, GGML_TYPE_Q8_1, true},     // Q4_1 = 3
        {0, 0, -1, false},                   // 4 unused
        {0, 0, -1, false},                   // 5 unused
        {32, 22, GGML_TYPE_Q8_1, true},     // Q5_0 = 6
        {32, 24, GGML_TYPE_Q8_1, true},     // Q5_1 = 7
        {32, 34, GGML_TYPE_Q8_0, true},     // Q8_0 = 8
        {32, 36, -1, true},                  // Q8_1 = 9
        {256, 84, GGML_TYPE_Q8_K, true},    // Q2_K = 10
        {256, 110, GGML_TYPE_Q8_K, true},   // Q3_K = 11
        {256, 144, GGML_TYPE_Q8_K, true},   // Q4_K = 12
        {256, 176, GGML_TYPE_Q8_K, true},   // Q5_K = 13
        {256, 210, GGML_TYPE_Q8_K, true},   // Q6_K = 14
        {256, 292, -1, false},               // Q8_K = 15
        {256, 66, GGML_TYPE_Q8_K, true},    // IQ2_XXS = 16
        {256, 74, GGML_TYPE_Q8_K, true},    // IQ2_XS = 17
        {256, 98, GGML_TYPE_Q8_K, true},    // IQ3_XXS = 18
        {256, 50, GGML_TYPE_Q8_K, true},    // IQ1_S = 19
        {256, 164, GGML_TYPE_Q8_1, true},   // IQ4_NL = 20
        {256, 110, GGML_TYPE_Q8_K, true},   // IQ3_S = 21
        {256, 82, GGML_TYPE_Q8_K, true},    // IQ2_S = 22
        {256, 166, GGML_TYPE_Q8_K, true},   // IQ4_XS = 23
        {256, 56, GGML_TYPE_Q8_K, true},    // IQ1_M = 24
    };
    static constexpr int table_size = static_cast<int>(sizeof(table) / sizeof(table[0]));

    if (type_id == GGML_TYPE_Q2_0) {
        // Q2_0: custom 2-bit, 64 values per block, 18 bytes/block
        static const TypeInfo q2_0 = {64, 18, GGML_TYPE_Q8_1, true};
        return &q2_0;
    }
    if (type_id < 0 || type_id >= table_size) return nullptr;
    return &table[type_id];
}

size_t row_size(int type_id, int64_t ne) {
    const TypeInfo* t = type_info(type_id);
    if (!t || t->block_size <= 0) return 0;
    return static_cast<size_t>((ne + t->block_size - 1) / t->block_size) *
           static_cast<size_t>(t->block_bytes);
}

}  // anonymous namespace

// ─── native_fmt: discover expert format from ggml type IDs ──────
bool native_fmt(int gu_type, int d_type, int64_t n_embd, int64_t n_ff,
                NativeFmt& fmt, std::string& err) {
    const TypeInfo* tg = type_info(gu_type);
    const TypeInfo* td = type_info(d_type);

    if (!tg || !tg->has_dot) {
        err = "native experts: no dot product for gate/up type " +
              std::to_string(gu_type);
        return false;
    }
    if (!td || !td->has_dot) {
        err = "native experts: no dot product for down type " +
              std::to_string(d_type);
        return false;
    }

    // Check activation quantization types
    const TypeInfo* ag = type_info(tg->act_type);
    const TypeInfo* ad = type_info(td->act_type);
    if (!ag || !ad) {
        err = "native experts: cannot quantize activation for this layer";
        return false;
    }

    fmt.gu_type  = gu_type;
    fmt.d_type   = d_type;
    fmt.gu_act   = tg->act_type;
    fmt.d_act    = td->act_type;
    fmt.n_embd   = n_embd;
    fmt.n_ff     = n_ff;
    fmt.gu_row   = row_size(gu_type, n_embd);
    fmt.d_row    = row_size(d_type, n_ff);
    fmt.up_off   = fmt.gu_row * static_cast<size_t>(n_ff);
    fmt.down_off = 2 * fmt.up_off;
    fmt.bytes    = fmt.down_off + fmt.d_row * static_cast<size_t>(n_embd);
    fmt.act_bytes = row_size(tg->act_type, n_embd);
    fmt.h_bytes  = row_size(td->act_type, n_ff);

    return true;
}

// ─── IQ format support check ────────────────────────────────────
bool iq512_supported(int type_id) {
    return type_id == GGML_TYPE_IQ2_XXS ||
           type_id == GGML_TYPE_IQ2_XS  ||
           type_id == GGML_TYPE_IQ2_S   ||
           type_id == GGML_TYPE_IQ3_XXS ||
           type_id == GGML_TYPE_IQ3_S;
}

// ─── AVX-512 capability check ───────────────────────────────────
bool cpu_avx512_ok() {
    static bool checked = false;
    static bool result = false;
    if (!checked) {
        strata::CpuFeatures f = strata::cpu_features();
        result = f.expert_usable();
        checked = true;
    }
    return result;
}

}}}  // namespace strata::kernels::cpu
