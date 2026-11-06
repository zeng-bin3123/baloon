// core_layer.hpp - composition of one GDN/QSA/MoE block (host model): 16-byte aligned arena carving, quantized-plane
// checks, activation-format dispatch, stage-prefix selection of block_layer_pre, the verify-window flag conjunction,
// PLE issue/stage/finish, the doorbell ring, stage timers and the debug-dump layout.
#pragma once
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace baloon {
namespace core {

constexpr uint64_t Q8K_ELEMS_PER_BLOCK = 256;
constexpr uint64_t Q8K_BYTES_PER_BLOCK = 292;
constexpr uint64_t STAGE_SLOTS = 8;
constexpr uint64_t TPR = 64;            // threads per row (one wave64 per output row)
constexpr float RMS_EPS = 1e-6f;
constexpr const char* EMBEDDING_NAME = "token_embd";

constexpr uint64_t align_up16(uint64_t n) { return (n + 15) / 16 * 16; }
constexpr uint64_t q8k_bytes(uint64_t n) { return n / Q8K_ELEMS_PER_BLOCK * Q8K_BYTES_PER_BLOCK; }

// ---- arena cursor: take(count, size) hands out [used, used + count*size) and re-aligns the cursor to 16 ----------------
struct Cursor {
    uint64_t used = 0, cap = UINT64_MAX;
    bool failed = false;
    uint64_t take(uint64_t count, uint64_t size) {
        const uint64_t at = used;
        const uint64_t next = align_up16(used + count * size);
        if (next > cap) { failed = true; return at; }
        used = next;
        return at;
    }
};

// ---- quantized planes: codes | scales | optional offsets --------------------------------------------------------------
struct WeightRef {
    uint64_t codes = 0, scales = 0, offset = 0, bytes = 0;
    int code_bits = 0;                 // 2, 4 or 8
    bool has_offset = false;
    bool native = false;               // native-format tensor
    bool wants_q8k = false;            // K-quant: wants the Q8_K activation image
};
struct Planes {
    bool ok = false;
    uint64_t codes_start = 0, scales_start = 0, offset_start = 0;
    uint64_t codes = 0, scales = 0, offset = 0;
};
// S2/S4/S8, non-empty codes and scales, planes add up to the tensor, has_offset agrees with the offset plane.
Planes plane_ptrs(const WeightRef& w);

enum class Gemv { Native, S2Q80, LegacyQ80, Q8k };
Gemv sform_of(const WeightRef& w);           // native first, then 2-bit -> Q8_0, then Q8_K for K-quants, else Q8_0
struct GemvExec {
    std::function<void(const Planes&)> native, s2q80, legacy_q80, q8k;
};
// Validates the planes, picks the path by sform_of and runs the matching executor; false when the planes are bad.
bool gemv_quantized(const WeightRef& w, const GemvExec& exec, Gemv* chosen = nullptr);

// Attention shapes of a QSA layer: q_dim = n_head*head_dim, kv_dim = n_head_kv*head_dim, q_full = n_head*2*head_dim (query
// plus gate), group = query heads per KV head. Invalid unless n_head is a positive multiple of n_head_kv.
struct QsaShapes { bool ok = false; uint64_t q_dim = 0, kv_dim = 0, q_full = 0, group = 0; };
QsaShapes qsa_shapes(uint64_t n_head, uint64_t head_dim, uint64_t n_head_kv);

// ---- scratch buffers carved from one arena ----------------------------------------------------------------------------
struct BlockDims { uint64_t T = 1, n_embd = 0, hc = 1, n_head = 0, head_dim = 0, n_head_kv = 0; };
struct GdnDims { uint64_t T = 1, qkv = 0, z = 0, heads = 0, out = 0; };
struct MoeDims { uint64_t T = 1, n_embd = 0, n_expert = 0, k = 0; };
struct QsaDims { uint64_t T = 1, n_head = 0, head_dim = 0, n_head_kv = 0, cap = 0; };
struct QsaStateDims { uint64_t max_cells = 0, n_head_kv = 0, head_dim = 0, block = 64; };

constexpr int kBlockRegions = 6, kGdnRegions = 5, kMoeRegions = 6, kQsaRegions = 5, kQsaStateRegions = 4;
template <int N> struct Regions {
    uint8_t* base = nullptr;
    uint64_t bytes = 0;
    uint64_t off[N] = {};
    uint64_t size[N] = {};
    uint8_t* at(int i) const { return base ? base + off[i] : nullptr; }
};
using BlockBuffers = Regions<kBlockRegions>;
using GdnBuffers = Regions<kGdnRegions>;
using MoEBuffers = Regions<kMoeRegions>;
using QsaBuffers = Regions<kQsaRegions>;
using QsaState = Regions<kQsaStateRegions>;

uint64_t block_buffers_bytes(const BlockDims& d);
bool block_buffers_init(BlockBuffers& b, void* base, uint64_t cap, const BlockDims& d);
uint64_t gdn_buffers_bytes(const GdnDims& d);
bool gdn_buffers_init(GdnBuffers& b, void* base, uint64_t cap, const GdnDims& d);
void gdn_buffers_zero_state(float* state, uint64_t floats);
// ---- ArenaCarving: the model-geometry buffer sets (every region rounded up to 16 bytes, `*_bytes` == what `*_init` consumes) ----
struct GdnGeometry { uint64_t n_embd = 0, conv_channels = 0, value_dim = 0, v_heads = 0, state_size = 0, d_conv = 0; };
constexpr GdnGeometry kRealGdnGeometry{2560, 10240, 6144, 48, 128, 4};     // the engine's real geometry
constexpr int kGdnArenaRegions = 16, kMoeArenaRegions = 9;
void gdn_arena_sizes(const GdnGeometry& g, uint64_t (&s)[kGdnArenaRegions]);         // raw (unaligned) region sizes, carve order
uint64_t gdn_arena_bytes(const GdnGeometry& g);
// Walks the same list with a moving cursor; returns the cursor end (== gdn_arena_bytes) or UINT64_MAX when `cap` is too small.
uint64_t gdn_arena_init(Regions<kGdnArenaRegions>& b, void* base, uint64_t cap, const GdnGeometry& g);
void moe_arena_sizes(uint64_t n_embd, uint64_t n_expert, uint64_t k, uint64_t shared_scratch, uint64_t (&s)[kMoeArenaRegions]);
uint64_t moe_arena_bytes(uint64_t n_embd, uint64_t n_expert, uint64_t k, uint64_t shared_scratch);
uint64_t moe_arena_init(Regions<kMoeArenaRegions>& b, void* base, uint64_t cap, uint64_t n_embd, uint64_t n_expert, uint64_t k,
                        uint64_t shared_scratch);
// n QSA layers share one RoPE table: the first state owns `rope` extra bytes, the others borrow it.
constexpr uint64_t qsa_states_total(uint64_t n, uint64_t rest, uint64_t rope) { return n == 0 ? 0 : n * rest + rope; }

uint64_t moe_buffers_bytes(const MoeDims& d);
bool moe_buffers_init(MoEBuffers& b, void* base, uint64_t cap, const MoeDims& d);
uint64_t qsa_buffers_bytes(const QsaDims& d);
bool qsa_buffers_init(QsaBuffers& b, void* base, uint64_t cap, const QsaDims& d);
uint64_t qsa_state_bytes(const QsaStateDims& d);   // depends on qsa_kv_int8()
bool qsa_state_init(QsaState& s, void* base, uint64_t cap, const QsaStateDims& d);
void qsa_state_zero(QsaState& s);

// ---- runtime switches --------------------------------------------------------------------------------------------------
void layer_set_fast_attn(bool v);
void layer_set_fast_select(bool v);
void layer_set_fused_gdn(bool v);
void layer_set_fused_gr(bool v);
void layer_set_native_bf16(bool v);
void layer_set_native_flash_attn_short(bool v);
void layer_set_publish_kernel(bool v);
void layer_set_shared_early(bool v);
void qsa_set_kv_int8(bool v);
bool qsa_kv_int8();
bool layer_shared_early();
bool layer_publish_kernel();
// Fused hyper-connection bookkeeping applies only to the default call shape on supported layers.
bool layer_fused_gr(int stage_prefix, int half, bool supported);
// Verify windows need every fast path on and the short flash adapter off.
bool layer_verify_compatible(bool native_gdn, bool qsa_indexer);

// ---- block composition -------------------------------------------------------------------------------------------------
constexpr bool is_qsa_layer(int l) { return l % 4 == 3; }
bool stage_runs(int stage_prefix, int half, int stage);                // runStage
bool ple_applies(bool ready, int layer, int half);
bool post_writes_r(bool fused_on, int layer, int n_layers);

struct StageFns { std::function<void(int layer)> stage[5]; };
struct BlockCall { int layer = 0, n_layers = 48, stage_prefix = 0, half = 0; bool gr_supported = true, ple_ready = false; };
struct BlockHooks { StageFns stages; std::function<void()> gr_write; std::function<void()> ple; };
struct BlockResult { uint32_t stage_mask = 0; bool fused = false, wrote_r = false, ple_ran = false; };

BlockResult block_layer_pre(const BlockCall& c, const BlockHooks& h);                    // stages 0..4 by prefix / half
BlockResult block_layer_post(const BlockCall& c, const BlockHooks& h, BlockResult pre);  // residual write-back
BlockResult block_layer(const BlockCall& c, const BlockHooks& h);                        // pre then post
bool gdn_layer(const BlockCall& c, const BlockHooks& h, BlockResult* r = nullptr);       // requires a GDN layer
bool qsa_layer(const BlockCall& c, const BlockHooks& h, BlockResult* r = nullptr);       // requires a QSA layer

// ---- MoE pieces ---------------------------------------------------------------------------------------------------------
// Top-k by descending logit (ties: lower index first); weights = softmax over the selected logits.
bool moe_route(const float* logits, int n_expert, int k, int32_t* idx, float* w);
void moe_shared(const float* shared, float gate_logit, int n, float* out);                   // sigmoid(gate) * shared
void moe_combine_parts(const float* parts, const float* w, int k, int n, float* out);        // sum_p w_p * parts_p
void moe_finish(const float* resid, const float* moe, const float* shared, int n, float* out);
bool moe_layer(const float* logits, int n_expert, int k, const std::function<void(int expert, float* out)>& expert_fn,
               const float* shared, float shared_gate, const float* resid, int n, float* out);

// ---- embedding, projections, head ---------------------------------------------------------------------------------------
bool embed_row(const float* table, int64_t vocab, int64_t n, int64_t token, float* out);   // native FP32 table
// Bit-field decode: code of element i is the code_bits-wide field at bit (i % perbyte) * code_bits of byte i / perbyte, where
// perbyte = 8 / code_bits (low field first). code_bits must be 2, 4 or 8.
int embed_code_per_byte(int code_bits);                                                   // 0 for an illegal width
bool embed_code(const uint8_t* row, int code_bits, int64_t i, uint32_t* code);
// Decodes one row of a quantized table (rows are n / perbyte bytes; n must be a multiple of perbyte) into integer codes.
bool embed_row(const uint8_t* table, int code_bits, int64_t vocab, int64_t n, int64_t token, uint32_t* out);
void project_bf16(const uint16_t* w, const float* x, int64_t rows, int64_t cols, float* out);
void lm_head(const float* w, const float* x, int64_t vocab, int64_t n, float* logits);
void lm_head_mix(const float* w, const float* resid, const float* mix, int64_t hc, int64_t vocab, int64_t n, float* logits);

// ---- PLE (per-layer embedding): issue -> stage -> finish --------------------------------------------------------------
struct PleRun {
    enum State { Idle, Issued, Staged, Done } state = Idle;
    int64_t token = -1;
    std::vector<float> row;
};
uint64_t ple_run_scratch_bytes(uint64_t n_embd_ple, uint64_t k_tokens);
bool ple_issue_token(PleRun& r, int64_t token);
bool ple_stage_token(PleRun& r, const float* table, int64_t vocab, int64_t n);
bool ple_finish_token(PleRun& r, float* resid, float scale);

// ---- doorbell ring ------------------------------------------------------------------------------------------------------
struct Doorbell { std::atomic<uint64_t>* word = nullptr; };
void doorbell_init(Doorbell* d);
void doorbell_reset(Doorbell* d);
void doorbell_free(Doorbell* d);
void doorbell_raise(Doorbell* d, uint64_t v);      // monotone: never lowers the ring
uint64_t doorbell_value(const Doorbell* d);

// ---- stage timing -------------------------------------------------------------------------------------------------------
struct StageTimer {
    bool enabled = false;
    double total_ms[STAGE_SLOTS] = {};
    double begin_ms[STAGE_SLOTS] = {};
    uint64_t calls[STAGE_SLOTS] = {};
    bool open[STAGE_SLOTS] = {};
};
void st_begin(StageTimer& t, int slot, double now_ms);
void st_end(StageTimer& t, int slot, double now_ms);
void stage_timing_enable(bool on);
const char* stage_timing_name(int slot);
void stage_mark_begin(int slot);
void stage_mark_end(int slot);
void stage_timing_set_clock(double (*clock_ms)());   // nullptr restores the steady clock
const StageTimer& stage_timing_state();
void stage_timing_reset();
std::string stage_timing_report();

// ---- debug dump layout ---------------------------------------------------------------------------------------------------
constexpr int kDumpSlots = 11;
uint64_t dump_stride_floats(uint64_t n, uint64_t hc, uint64_t a, uint64_t nvk);
struct DumpSlot { uint64_t off, len; };
DumpSlot dump_slot(int i, uint64_t n, uint64_t hc, uint64_t a, uint64_t nvk);
void dump_half(const float* src, uint64_t n, uint16_t* dst);   // fp32 -> fp16 bits (round to nearest even)

}  // namespace core
}  // namespace baloon
