// layer.cpp - host model of the block composition layer (see core_layer.hpp).
#include "baloon/core_layer.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>

#include "baloon/f16_conversion.hpp"
#include "baloon/layer_compose.hpp"

namespace baloon {
namespace core {

// ---- runtime switches (process-wide, set once before the graphs are built) -------------------------------------------
struct LayerFlagStore {
    std::atomic<bool> g_fast_attn{false};
    std::atomic<bool> g_fast_select{false};
    std::atomic<bool> g_fused_gdn{false};
    std::atomic<bool> g_fused_gr{false};
    std::atomic<bool> g_publish_kernel{false};
    std::atomic<bool> g_shared_early{false};
    std::atomic<bool> g_kv_int8{false};
    std::atomic<bool> native_bf16_projections{false};
    std::atomic<bool> native_flash_attn_short{false};
};
static LayerFlagStore g_flag_store;
static std::atomic<bool>* const g_fast_attn = &g_flag_store.g_fast_attn;
static std::atomic<bool>* const g_fast_select = &g_flag_store.g_fast_select;
static std::atomic<bool>* const g_fused_gdn = &g_flag_store.g_fused_gdn;
static std::atomic<bool>* const g_fused_gr = &g_flag_store.g_fused_gr;
static std::atomic<bool>* const g_publish_kernel = &g_flag_store.g_publish_kernel;
static std::atomic<bool>* const g_shared_early = &g_flag_store.g_shared_early;
static std::atomic<bool>* const g_kv_int8 = &g_flag_store.g_kv_int8;
static std::atomic<bool>* const native_bf16_projections = &g_flag_store.native_bf16_projections;
static std::atomic<bool>* const native_flash_attn_short = &g_flag_store.native_flash_attn_short;

void layer_set_fast_attn(bool v) { g_fast_attn->store(v); }
void layer_set_fast_select(bool v) { g_fast_select->store(v); }
void layer_set_fused_gdn(bool v) { g_fused_gdn->store(v); }
void layer_set_fused_gr(bool v) { g_fused_gr->store(v); }
void layer_set_native_bf16(bool v) { native_bf16_projections->store(v); }
void layer_set_native_flash_attn_short(bool v) { native_flash_attn_short->store(v); }
void layer_set_publish_kernel(bool v) { g_publish_kernel->store(v); }
void layer_set_shared_early(bool v) { g_shared_early->store(v); }
void qsa_set_kv_int8(bool v) { g_kv_int8->store(v); }
bool qsa_kv_int8() { return (*g_kv_int8); }
bool layer_shared_early() { return (*g_shared_early); }
bool layer_publish_kernel() { return (*g_publish_kernel); }

bool layer_fused_gr(int stage_prefix, int half, bool supported) {
    return (*g_fused_gr) && stage_prefix == 0 && half == 0 && supported;
}

bool layer_verify_compatible(bool native_gdn, bool qsa_indexer) {
    return (*native_bf16_projections) && (*g_fused_gr) && ((*g_fused_gdn) && native_gdn) &&
           ((*g_fast_attn) && !(*native_flash_attn_short)) && (*g_fast_select) && qsa_indexer;
}

// ---- quantized planes / dispatch ------------------------------------------------------------------------------------------
Planes plane_ptrs(const WeightRef& w) {
    Planes p;
    const bool bits_ok = w.code_bits == 2 || w.code_bits == 4 || w.code_bits == 8;
    if (!bits_ok || w.codes == 0 || w.scales == 0) return p;
    if (w.codes + w.scales + w.offset != w.bytes) return p;
    if (w.has_offset != (w.offset != 0)) return p;
    p.ok = true;
    p.codes = w.codes; p.scales = w.scales; p.offset = w.offset;
    p.codes_start = 0;
    p.scales_start = w.codes;
    p.offset_start = w.codes + w.scales;
    return p;
}

Gemv sform_of(const WeightRef& w) {
    switch (layer_compose::dispatch(w.native, (uint64_t)w.code_bits, w.wants_q8k)) {
        case layer_compose::Dispatch::Native: return Gemv::Native;
        case layer_compose::Dispatch::S2Q80: return Gemv::S2Q80;
        case layer_compose::Dispatch::Q8kSplit: return Gemv::Q8k;
        default: return Gemv::LegacyQ80;
    }
}

bool gemv_quantized(const WeightRef& w, const GemvExec& exec, Gemv* chosen) {
    const Planes p = plane_ptrs(w);
    if (!p.ok) return false;
    const Gemv g = sform_of(w);
    if (chosen) *chosen = g;
    const std::function<void(const Planes&)>* fn = nullptr;
    switch (g) {
        case Gemv::Native: fn = &exec.native; break;
        case Gemv::S2Q80: fn = &exec.s2q80; break;
        case Gemv::Q8k: fn = &exec.q8k; break;
        case Gemv::LegacyQ80: fn = &exec.legacy_q80; break;
    }
    if (!fn || !*fn) return false;
    (*fn)(p);
    return true;
}

QsaShapes qsa_shapes(uint64_t n_head, uint64_t head_dim, uint64_t n_head_kv) {
    QsaShapes s;
    if (n_head == 0 || head_dim == 0 || n_head_kv == 0 || n_head % n_head_kv != 0) return s;
    s.ok = true;
    s.q_dim = n_head * head_dim; s.kv_dim = n_head_kv * head_dim; s.q_full = n_head * 2 * head_dim; s.group = n_head / n_head_kv;
    return s;
}

// ---- buffers ---------------------------------------------------------------------------------------------------------------
template <int N>
static uint64_t total_of(const uint64_t (&sz)[N]) {
    Cursor c;
    for (int i = 0; i < N; ++i) c.take(sz[i], 1);
    return c.used;
}
template <int N>
static bool carve(Regions<N>& r, void* base, uint64_t cap, const uint64_t (&sz)[N]) {
    r = Regions<N>{};
    Cursor c;
    c.cap = cap;
    for (int i = 0; i < N; ++i) { r.off[i] = c.take(sz[i], 1); r.size[i] = sz[i]; }
    if (c.failed || !base || (reinterpret_cast<uintptr_t>(base) & 15u)) return false;
    r.base = static_cast<uint8_t*>(base);
    r.bytes = c.used;
    return true;
}

static void block_sizes(const BlockDims& d, uint64_t (&s)[kBlockRegions]) {
    s[0] = d.T * d.n_embd * 4;                   // normalised input
    s[1] = d.T * d.hc * d.n_embd * 4;            // hyper-connection residual streams
    s[2] = d.T * d.n_head * d.head_dim * 4;      // attention / mixer output
    s[3] = d.T * 2 * d.n_head_kv * d.head_dim * 4;  // K and V rows
    s[4] = d.T * d.n_embd * 4;                   // FFN input
    s[5] = d.T * q8k_bytes(d.n_embd);            // Q8_K activation image
}
uint64_t block_buffers_bytes(const BlockDims& d) { uint64_t s[kBlockRegions]; block_sizes(d, s); return total_of(s); }
bool block_buffers_init(BlockBuffers& b, void* base, uint64_t cap, const BlockDims& d) {
    uint64_t s[kBlockRegions]; block_sizes(d, s); return carve(b, base, cap, s);
}

static void gdn_sizes(const GdnDims& d, uint64_t (&s)[kGdnRegions]) {
    s[0] = d.T * d.qkv * 4;
    s[1] = d.T * d.z * 4;
    s[2] = d.T * 2 * d.heads * 4;                // beta and alpha per head
    s[3] = d.T * d.out * 4;
    s[4] = d.T * q8k_bytes(d.out);
}
uint64_t gdn_buffers_bytes(const GdnDims& d) { uint64_t s[kGdnRegions]; gdn_sizes(d, s); return total_of(s); }
bool gdn_buffers_init(GdnBuffers& b, void* base, uint64_t cap, const GdnDims& d) {
    uint64_t s[kGdnRegions]; gdn_sizes(d, s); return carve(b, base, cap, s);
}
void gdn_buffers_zero_state(float* state, uint64_t floats) {
    if (state && floats) std::memset(state, 0, (size_t)floats * sizeof(float));
}

void gdn_arena_sizes(const GdnGeometry& g, uint64_t (&s)[kGdnArenaRegions]) {
    const uint64_t parts[kGdnArenaRegions] = {
        q8k_bytes(g.n_embd), g.n_embd / 32 * 34, g.n_embd * 2,
        g.conv_channels * 4, g.conv_channels * 4, g.conv_channels * 4,
        g.v_heads * 4, g.v_heads * 4, g.v_heads * 4,
        g.v_heads * g.state_size * 4, g.value_dim * 4, g.value_dim * 4,
        q8k_bytes(g.value_dim), g.value_dim / 32 * 34,
        g.state_size * g.v_heads * g.state_size * 4, g.conv_channels * (g.d_conv > 0 ? g.d_conv - 1 : 0) * 4};
    for (int i = 0; i < kGdnArenaRegions; ++i) s[i] = parts[i];
}
uint64_t gdn_arena_bytes(const GdnGeometry& g) { uint64_t s[kGdnArenaRegions]; gdn_arena_sizes(g, s); return total_of(s); }
uint64_t gdn_arena_init(Regions<kGdnArenaRegions>& b, void* base, uint64_t cap, const GdnGeometry& g) {
    uint64_t s[kGdnArenaRegions]; gdn_arena_sizes(g, s);
    if (!carve(b, base, cap, s)) return UINT64_MAX;
    return b.bytes;
}
void moe_arena_sizes(uint64_t n_embd, uint64_t n_expert, uint64_t k, uint64_t shared_scratch, uint64_t (&s)[kMoeArenaRegions]) {
    const uint64_t parts[kMoeArenaRegions] = {n_embd * 2, n_embd * 2, n_expert * 4, k * 4, k * 4, n_embd * 4, shared_scratch,
                                              n_embd / 32 * 34, q8k_bytes(n_embd)};
    for (int i = 0; i < kMoeArenaRegions; ++i) s[i] = parts[i];
}
uint64_t moe_arena_bytes(uint64_t n_embd, uint64_t n_expert, uint64_t k, uint64_t shared_scratch) {
    uint64_t s[kMoeArenaRegions]; moe_arena_sizes(n_embd, n_expert, k, shared_scratch, s); return total_of(s);
}
uint64_t moe_arena_init(Regions<kMoeArenaRegions>& b, void* base, uint64_t cap, uint64_t n_embd, uint64_t n_expert, uint64_t k,
                        uint64_t shared_scratch) {
    uint64_t s[kMoeArenaRegions]; moe_arena_sizes(n_embd, n_expert, k, shared_scratch, s);
    if (!carve(b, base, cap, s)) return UINT64_MAX;
    return b.bytes;
}

static void moe_sizes(const MoeDims& d, uint64_t (&s)[kMoeRegions]) {
    s[0] = d.T * d.n_expert * 4;                 // router logits
    s[1] = d.T * d.k * 4;                        // top-k ids
    s[2] = d.T * d.k * 4;                        // top-k weights
    s[3] = d.T * d.k * d.n_embd * 4;             // per-expert output parts
    s[4] = d.T * d.n_embd * 4;                   // shared-expert output
    s[5] = d.T * q8k_bytes(d.n_embd);
}
uint64_t moe_buffers_bytes(const MoeDims& d) { uint64_t s[kMoeRegions]; moe_sizes(d, s); return total_of(s); }
bool moe_buffers_init(MoEBuffers& b, void* base, uint64_t cap, const MoeDims& d) {
    uint64_t s[kMoeRegions]; moe_sizes(d, s); return carve(b, base, cap, s);
}

static void qsa_sizes(const QsaDims& d, uint64_t (&s)[kQsaRegions]) {
    s[0] = d.T * d.n_head * d.head_dim * 4;      // queries
    s[1] = d.T * 2 * d.n_head_kv * d.head_dim * 4;  // new K and V
    s[2] = d.T * d.n_head * d.cap * 4;           // scores
    s[3] = d.T * d.cap * 4;                      // selected ids
    s[4] = d.T * d.n_head * d.head_dim * 4;      // attention output
}
uint64_t qsa_buffers_bytes(const QsaDims& d) { uint64_t s[kQsaRegions]; qsa_sizes(d, s); return total_of(s); }
bool qsa_buffers_init(QsaBuffers& b, void* base, uint64_t cap, const QsaDims& d) {
    uint64_t s[kQsaRegions]; qsa_sizes(d, s); return carve(b, base, cap, s);
}

static void qsa_state_sizes(const QsaStateDims& d, uint64_t (&s)[kQsaStateRegions]) {
    const uint64_t cells = d.max_cells * d.n_head_kv * d.head_dim;
    const uint64_t elem = (*g_kv_int8) ? 1 : 4;
    s[0] = cells * elem;                                   // K cache
    s[1] = cells * elem;                                   // V cache
    s[2] = (*g_kv_int8) ? 2 * d.max_cells * d.n_head_kv * 4 : 0;   // per-row scales (int8 mode only)
    s[3] = (d.max_cells / (d.block ? d.block : 1) + 2) * d.head_dim * 4;  // pooled indexer keys
}
uint64_t qsa_state_bytes(const QsaStateDims& d) { uint64_t s[kQsaStateRegions]; qsa_state_sizes(d, s); return total_of(s); }
bool qsa_state_init(QsaState& st, void* base, uint64_t cap, const QsaStateDims& d) {
    uint64_t s[kQsaStateRegions]; qsa_state_sizes(d, s); return carve(st, base, cap, s);
}
void qsa_state_zero(QsaState& st) {
    if (st.base && st.bytes) std::memset(st.base, 0, (size_t)st.bytes);
}

// ---- block composition ------------------------------------------------------------------------------------------------------
bool stage_runs(int stage_prefix, int half, int stage) {
    if (stage < 0 || stage > 4) return false;
    return layer_compose::run_stage((uint64_t)stage, (uint64_t)std::max(stage_prefix, 0), (uint64_t)std::max(half, 0));
}
bool ple_applies(bool ready, int layer, int half) { return ready && layer == 1 && (half == 0 || half == 1); }
bool post_writes_r(bool fused_on, int layer, int n_layers) { return !fused_on || layer + 1 == n_layers; }

BlockResult block_layer_pre(const BlockCall& c, const BlockHooks& h) {
    BlockResult r;
    r.fused = layer_fused_gr(c.stage_prefix, c.half, c.gr_supported);
    for (int i = 0; i < 5; ++i) {
        if (!stage_runs(c.stage_prefix, c.half, i)) continue;
        if (h.stages.stage[i]) h.stages.stage[i](c.layer);
        r.stage_mask |= 1u << i;
        if (i == 2 && ple_applies(c.ple_ready, c.layer, c.half) && h.ple) { h.ple(); r.ple_ran = true; }
    }
    return r;
}

BlockResult block_layer_post(const BlockCall& c, const BlockHooks& h, BlockResult pre) {
    // The post half exists only for the full graph (prefix 0 or 5) and the FFN-side halves.
    const bool full = c.stage_prefix == 0 || c.stage_prefix >= 5;
    if (!full || c.half == 1) return pre;
    if (post_writes_r(pre.fused, c.layer, c.n_layers)) {
        if (h.gr_write) h.gr_write();
        pre.wrote_r = true;
    }
    return pre;
}

BlockResult block_layer(const BlockCall& c, const BlockHooks& h) { return block_layer_post(c, h, block_layer_pre(c, h)); }

bool gdn_layer(const BlockCall& c, const BlockHooks& h, BlockResult* r) {
    if (is_qsa_layer(c.layer)) return false;
    const BlockResult x = block_layer(c, h);
    if (r) *r = x;
    return true;
}
bool qsa_layer(const BlockCall& c, const BlockHooks& h, BlockResult* r) {
    if (!is_qsa_layer(c.layer)) return false;
    const BlockResult x = block_layer(c, h);
    if (r) *r = x;
    return true;
}

// ---- MoE ------------------------------------------------------------------------------------------------------------------------
bool moe_route(const float* logits, int n_expert, int k, int32_t* idx, float* w) {
    if (!logits || !idx || !w || n_expert <= 0 || k <= 0 || k > n_expert || k > 64) return false;   // k in 1..64 (moe_split::k_valid)
    std::vector<int32_t> order((size_t)n_expert);
    for (int i = 0; i < n_expert; ++i) order[(size_t)i] = i;
    std::stable_sort(order.begin(), order.end(), [&](int32_t a, int32_t b) { return logits[a] > logits[b]; });
    float mx = logits[order[0]];
    double z = 0;
    for (int i = 0; i < k; ++i) { idx[i] = order[(size_t)i]; z += std::exp((double)(logits[idx[i]] - mx)); }
    for (int i = 0; i < k; ++i) w[i] = (float)(std::exp((double)(logits[idx[i]] - mx)) / z);
    return true;
}
void moe_shared(const float* shared, float gate_logit, int n, float* out) {
    const float g = 1.0f / (1.0f + std::exp(-gate_logit));
    for (int j = 0; j < n; ++j) out[j] = g * shared[j];
}
void moe_combine_parts(const float* parts, const float* w, int k, int n, float* out) {
    for (int j = 0; j < n; ++j) {
        double a = 0;
        for (int p = 0; p < k; ++p) a += (double)w[p] * (double)parts[(size_t)p * n + j];
        out[j] = (float)a;
    }
}
void moe_finish(const float* resid, const float* moe, const float* shared, int n, float* out) {
    for (int j = 0; j < n; ++j) out[j] = resid[j] + moe[j] + shared[j];
}
bool moe_layer(const float* logits, int n_expert, int k, const std::function<void(int, float*)>& expert_fn,
               const float* shared, float shared_gate, const float* resid, int n, float* out) {
    if (n <= 0 || !expert_fn || !shared || !resid || !out) return false;
    std::vector<int32_t> idx((size_t)k > 0 ? (size_t)k : 0);
    std::vector<float> w(idx.size());
    if (!moe_route(logits, n_expert, k, idx.data(), w.data())) return false;
    std::vector<float> parts((size_t)k * n), moe((size_t)n), sh((size_t)n);
    for (int p = 0; p < k; ++p) expert_fn(idx[(size_t)p], parts.data() + (size_t)p * n);
    moe_combine_parts(parts.data(), w.data(), k, n, moe.data());
    moe_shared(shared, shared_gate, n, sh.data());
    moe_finish(resid, moe.data(), sh.data(), n, out);
    return true;
}

// ---- embedding / projection / head -----------------------------------------------------------------------------------------------
bool embed_row(const float* table, int64_t vocab, int64_t n, int64_t token, float* out) {
    if (!table || !out || token < 0 || token >= vocab || n <= 0) return false;
    std::memcpy(out, table + (size_t)(token * n), (size_t)n * sizeof(float));
    return true;
}
int embed_code_per_byte(int code_bits) {
    switch (code_bits) { case 2: return 4; case 4: return 2; case 8: return 1; default: return 0; }
}
bool embed_code(const uint8_t* row, int code_bits, int64_t i, uint32_t* code) {
    const int per = embed_code_per_byte(code_bits);
    if (!row || !code || per == 0 || i < 0) return false;
    const unsigned shift = (unsigned)(i % per) * (unsigned)code_bits;
    const unsigned mask = (1u << code_bits) - 1u;
    *code = ((unsigned)row[(size_t)(i / per)] >> shift) & mask;
    return true;
}
bool embed_row(const uint8_t* table, int code_bits, int64_t vocab, int64_t n, int64_t token, uint32_t* out) {
    const int per = embed_code_per_byte(code_bits);
    if (!table || !out || per == 0 || token < 0 || token >= vocab || n <= 0 || n % per != 0) return false;
    const uint8_t* row = table + (size_t)token * (size_t)(n / per);
    for (int64_t i = 0; i < n; ++i) if (!embed_code(row, code_bits, i, &out[(size_t)i])) return false;
    return true;
}
void project_bf16(const uint16_t* w, const float* x, int64_t rows, int64_t cols, float* out) {
    for (int64_t r = 0; r < rows; ++r) {
        double a = 0;
        for (int64_t j = 0; j < cols; ++j) a += (double)f16::bf16_to_float(w[(size_t)(r * cols + j)]) * (double)x[j];
        out[r] = (float)a;
    }
}
void lm_head(const float* w, const float* x, int64_t vocab, int64_t n, float* logits) {
    for (int64_t v = 0; v < vocab; ++v) {
        double a = 0;
        for (int64_t j = 0; j < n; ++j) a += (double)w[(size_t)(v * n + j)] * (double)x[j];
        logits[v] = (float)a;
    }
}
void lm_head_mix(const float* w, const float* resid, const float* mix, int64_t hc, int64_t vocab, int64_t n, float* logits) {
    std::vector<float> x((size_t)n, 0.0f);
    for (int64_t j = 0; j < n; ++j) {
        double a = 0;
        for (int64_t h = 0; h < hc; ++h) a += (double)mix[h] * (double)resid[(size_t)(h * n + j)];
        x[(size_t)j] = (float)a;
    }
    lm_head(w, x.data(), vocab, n, logits);
}

// ---- PLE ---------------------------------------------------------------------------------------------------------------------------
uint64_t ple_run_scratch_bytes(uint64_t n_embd_ple, uint64_t k_tokens) {
    Cursor c;
    c.take(k_tokens, n_embd_ple * 4);      // staged rows
    c.take(k_tokens, 8);                   // token ids
    return c.used;
}
bool ple_issue_token(PleRun& r, int64_t token) {
    if (token < 0 || (r.state != PleRun::Idle && r.state != PleRun::Done)) return false;
    r.token = token; r.row.clear(); r.state = PleRun::Issued;
    return true;
}
bool ple_stage_token(PleRun& r, const float* table, int64_t vocab, int64_t n) {
    if (r.state != PleRun::Issued || !table || n <= 0 || r.token >= vocab) return false;
    r.row.assign(table + (size_t)(r.token * n), table + (size_t)(r.token * n) + (size_t)n);
    r.state = PleRun::Staged;
    return true;
}
bool ple_finish_token(PleRun& r, float* resid, float scale) {
    if (r.state != PleRun::Staged || !resid) return false;
    for (size_t j = 0; j < r.row.size(); ++j) resid[j] += scale * r.row[j];
    r.state = PleRun::Done;
    return true;
}

// ---- doorbell ----------------------------------------------------------------------------------------------------------------------
void doorbell_init(Doorbell* d) {
    if (!d) return;
    if (d->word) delete d->word;
    d->word = new std::atomic<uint64_t>(0);
}
void doorbell_reset(Doorbell* d) { if (d && d->word) d->word->store(0, std::memory_order_release); }
void doorbell_free(Doorbell* d) {
    if (!d) return;
    delete d->word;
    d->word = nullptr;
}
void doorbell_raise(Doorbell* d, uint64_t v) {
    if (!d || !d->word) return;
    uint64_t cur = d->word->load(std::memory_order_relaxed);
    while (cur < v && !d->word->compare_exchange_weak(cur, v, std::memory_order_acq_rel)) {}
}
// ---- stage timing ---------------------------------------------------------------------------------------------------------------------
static StageTimer g_timer;
static double (*g_clock_ms)() = nullptr;
static double steady_ms() {
    using C = std::chrono::steady_clock;
    return std::chrono::duration<double, std::milli>(C::now().time_since_epoch()).count();
}
static double now_ms() { return g_clock_ms ? g_clock_ms() : steady_ms(); }

void st_begin(StageTimer& t, int slot, double now) {
    if (!t.enabled || slot < 0 || slot >= (int)STAGE_SLOTS) return;
    t.begin_ms[slot] = now; t.open[slot] = true;
}
void st_end(StageTimer& t, int slot, double now) {
    if (!t.enabled || slot < 0 || slot >= (int)STAGE_SLOTS || !t.open[slot]) return;
    t.total_ms[slot] += now - t.begin_ms[slot];
    ++t.calls[slot];
    t.open[slot] = false;
}
void stage_timing_enable(bool on) { g_timer.enabled = on; }
const char* stage_timing_name(int slot) {
    static const char* names[STAGE_SLOTS] = {"hc_read", "mixer_a", "mixer_b", "ffn_front", "router", "experts", "shared", "write"};
    return (slot >= 0 && slot < (int)STAGE_SLOTS) ? names[slot] : "?";
}
void stage_mark_begin(int slot) { st_begin(g_timer, slot, now_ms()); }
void stage_mark_end(int slot) { st_end(g_timer, slot, now_ms()); }
void stage_timing_set_clock(double (*clock_ms)()) { g_clock_ms = clock_ms; }
const StageTimer& stage_timing_state() { return g_timer; }
void stage_timing_reset() { const bool en = g_timer.enabled; g_timer = StageTimer{}; g_timer.enabled = en; }
std::string stage_timing_report() {
    std::string s;
    char buf[160];
    for (int i = 0; i < (int)STAGE_SLOTS; ++i) {
        if (!g_timer.calls[i]) continue;
        std::snprintf(buf, sizeof buf, "%-10s calls=%llu total_ms=%.3f\n", stage_timing_name(i),
                      (unsigned long long)g_timer.calls[i], g_timer.total_ms[i]);
        s += buf;
    }
    return s;
}

// ---- debug dump ------------------------------------------------------------------------------------------------------------------------
uint64_t dump_stride_floats(uint64_t n, uint64_t hc, uint64_t a, uint64_t nvk) { return 2 * n + 2 * hc + a + 5 * nvk + 8; }
DumpSlot dump_slot(int i, uint64_t n, uint64_t hc, uint64_t a, uint64_t nvk) {
    const uint64_t len[kDumpSlots] = {n, n, hc, hc, a, nvk, nvk, 8, nvk, nvk, nvk};
    if (i < 0 || i >= kDumpSlots) return {0, 0};
    uint64_t off = 0;
    for (int j = 0; j < i; ++j) off += len[j];
    return {off, len[i]};
}
void dump_half(const float* src, uint64_t n, uint16_t* dst) {
    for (uint64_t i = 0; i < n; ++i) dst[i] = f16::float_to_half(src[i]);
}

}  // namespace core
}  // namespace baloon
