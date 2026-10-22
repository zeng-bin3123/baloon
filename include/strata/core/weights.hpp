#pragma once
// strata/core/weights.hpp — Weight table, tensor references, quantized plane layout
// Target: Iluvatar BI-V100 (ivcore10)
//
// Weight kinds:
//   Pack: 5.305 GiB dense + 0.444 GiB embd
//   Engine: 4.067 GiB dense + 0.444 GiB embd + 10.88 MiB widened scales = 4.522 GiB
//
// BF16→F32 promotion is lossless; engine re-rounds to 16 bits (exact).
// 90 tensors carry fp16 scales → loader WIDENS to f32 (exact, fp16 ⊂ f32).

#include <cstdint>
#include <map>
#include <set>
#include <string>

namespace strata {

// ─── WeightKind ──────────────────────────────────────────────────────
enum class WeightKind : int {
    Verbatim   = 0,  // raw bytes, no conversion
    Bf16InF32  = 1,  // bf16 source stored promoted to f32
    F32        = 2,  // native f32
    F16InF32   = 3,  // fp16 source widened to f32
};

// ─── WeightRef ───────────────────────────────────────────────────────
// Reference to a single tensor in the arena.
//
// Plane layout (quantized tensors):
//   data = [codes | scales | offsets]
//   codes_bytes + scales_bytes + offset_bytes == total_bytes
//
// Activation contract (per tensor, NOT per role):
//   wants_q8k() iff act_kind == 1
//   code_bits == 2 → Q8_0 contract via s2_gemv_q8
//   otherwise quantized → Q8_K contract via s_gemv_q8k
struct WeightRef {
    const void* data = nullptr;
    uint64_t total_bytes = 0;
    int64_t ne0 = 0;          // first dimension
    int64_t ne1 = 0;          // second dimension
    int64_t n_elements = 0;   // total element count
    WeightKind kind = WeightKind::Verbatim;

    // Quantization parameters
    int code_bits = 0;
    int code_bias = 0;
    int group_elems = 0;
    bool codebook_iq4nl = false;
    bool has_offset = false;

    // Plane byte sizes
    uint64_t codes_bytes = 0;
    uint64_t scales_bytes = 0;
    uint64_t offset_bytes = 0;

    // Source location in pack file
    uint64_t src_offset = 0;
    uint64_t src_bytes = 0;
    int file_id = 0;
    bool scales_fp16 = false;

    // Activation dispatch
    int act_kind = 0;
    bool wants_q8k() const { return act_kind == 1; }
    bool quantized() const { return code_bits != 0; }

    // Native-backend pointers (set after upload)
    const void* native_data = nullptr;
    void* native_q8_data = nullptr;
    int native_type = -1;
    bool resident = true;
};

// ─── LoadReport ──────────────────────────────────────────────────────
struct LoadReport {
    size_t tensors = 0;
    uint64_t arena_bytes = 0;
    size_t re_rounded = 0;
    uint64_t bytes_saved = 0;
    double read_ms = 0.0;
    double upload_ms = 0.0;
};

// ─── WeightTable ─────────────────────────────────────────────────────
// Map[String → WeightRef].  Loads from an indexed directory into a
// pre-allocated arena.  Pre: arena_bytes >= pool_bytes (checked, not trusted).
class WeightTable {
public:
    // Compute required arena size from the index in `dir`, before loading.
    // Optionally skip tensors whose names are in `skip`.
    static bool pool_bytes(const std::string& dir, uint64_t& out_bytes,
                           std::string& err,
                           const std::set<std::string>* skip = nullptr);

    // Load tensors from `dir` into `arena_base` (of `arena_bytes`).
    bool load(const std::string& dir, void* arena_base, uint64_t arena_bytes,
              std::string& err,
              const std::set<std::string>* skip = nullptr);

    // Look up a tensor by name.
    const WeightRef* find(const std::string& name) const;

    const std::map<std::string, WeightRef>& all() const { return table_; }
    const LoadReport& report() const { return report_; }

private:
    friend class WeightTableBuilder;
    std::map<std::string, WeightRef> table_;
    LoadReport report_;
};

}  // namespace strata

