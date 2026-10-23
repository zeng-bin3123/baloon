// strata/core/memory_planner.cpp — VRAM budget arithmetic
#include "strata/core/memory_planner.hpp"

#include <algorithm>
#include <cstdio>

namespace strata {

uint64_t kv_cache_bytes(const ModelSpec& spec, int64_t context_length) {
    // KV cache layout: INT8 quantized
    // Per layer: kv_heads * head_dim * 2 (key + value) * context_length
    // Total: n_layers * per_layer
    if (context_length <= 0) return 0;
    uint64_t per_position = static_cast<uint64_t>(spec.kv_heads) *
                            static_cast<uint64_t>(spec.head_dim) * 2;
    uint64_t per_layer = per_position * static_cast<uint64_t>(context_length);
    return static_cast<uint64_t>(spec.n_layers) * per_layer;
}

uint64_t scratch_bytes_auto(const ModelSpec& spec) {
    // Activation buffer: n_embd * hc * sizeof(float) for the hyper-connection
    // residual stack (4 copies × n_embd × 4 bytes = 4 * 2560 * 4 = 40960)
    uint64_t residual_stack = static_cast<uint64_t>(spec.n_embd) *
                              static_cast<uint64_t>(spec.hc) * sizeof(float);

    // Router logits: n_expert * sizeof(float) = 128 * 4 = 512
    uint64_t router_logits = static_cast<uint64_t>(spec.n_expert) * sizeof(float);

    // Expert selection indices: n_expert_active * sizeof(int32_t)
    uint64_t expert_indices = static_cast<uint64_t>(spec.n_expert_active) * sizeof(int32_t);

    // Expert weights (routing coefficients): n_expert_active * sizeof(float)
    uint64_t expert_weights = static_cast<uint64_t>(spec.n_expert_active) * sizeof(float);

    // Temporary GEMV output: n_embd * sizeof(float) * 2 (up + down proj intermediate)
    uint64_t gemv_scratch = static_cast<uint64_t>(spec.n_embd) * sizeof(float) * 2;

    // QSA attention scratch: head_dim * kv_heads * sizeof(float) for scores
    uint64_t qsa_scratch = static_cast<uint64_t>(spec.head_dim) *
                           static_cast<uint64_t>(spec.kv_heads) * sizeof(float) * 2;

    // GDN recurrence scratch: n_embd * sizeof(float) for the state update
    uint64_t gdn_scratch = static_cast<uint64_t>(spec.n_embd) * sizeof(float);

    // Sum and add a 10% overhead margin, aligned to 1 MiB
    uint64_t total = residual_stack + router_logits + expert_indices +
                     expert_weights + gemv_scratch + qsa_scratch + gdn_scratch;
    total = total + total / 10;  // +10% margin

    // Align up to 1 MiB
    constexpr uint64_t MiB = 1024ull * 1024;
    total = (total + MiB - 1) & ~(MiB - 1);

    // Minimum 16 MiB for small models
    if (total < 16 * MiB) total = 16 * MiB;

    return total;
}

VRAMPlan plan_vram(uint64_t vram_total,
                   const ModelSpec& spec,
                   int64_t context_length,
                   uint64_t reserved_bytes,
                   uint64_t scratch_per_token) {
    VRAMPlan plan;
    plan.vram_total = vram_total;
    plan.vram_reserved = reserved_bytes;
    plan.context_length = context_length;

    // Usable = total - reserved
    if (vram_total <= reserved_bytes) {
        plan.vram_usable = 0;
        return plan;
    }
    plan.vram_usable = vram_total - reserved_bytes;

    // 1. Dense weights — always resident
    plan.dense_bytes = static_cast<uint64_t>(spec.dense_bytes);

    // 2. Head weights
    plan.head_bytes = static_cast<uint64_t>(spec.head_bytes);

    // 3. KV cache
    plan.kv_cache_bytes = kv_cache_bytes(spec, context_length);

    // 4. Scratch buffers
    if (scratch_per_token > 0) {
        plan.scratch_bytes = scratch_per_token;
    } else {
        plan.scratch_bytes = scratch_bytes_auto(spec);
    }

    // 5. Expert VRAM cache = remainder
    uint64_t fixed = plan.dense_bytes + plan.head_bytes +
                     plan.kv_cache_bytes + plan.scratch_bytes;
    if (fixed >= plan.vram_usable) {
        plan.expert_vram_bytes = 0;
        plan.expert_slots = 0;
    } else {
        plan.expert_vram_bytes = plan.vram_usable - fixed;
        // Each expert slot holds one expert blob
        if (spec.expert_blob_bytes > 0) {
            plan.expert_slots = static_cast<int64_t>(
                plan.expert_vram_bytes / static_cast<uint64_t>(spec.expert_blob_bytes));
        }
    }

    // Diagnostics
    if (plan.vram_usable > 0) {
        plan.expert_vram_frac = static_cast<double>(plan.expert_vram_bytes) /
                                static_cast<double>(plan.vram_usable);
        plan.kv_frac = static_cast<double>(plan.kv_cache_bytes) /
                       static_cast<double>(plan.vram_usable);
    }

    return plan;
}

std::string VRAMPlan::summary() const {
    char buf[1024];
    double MiB = 1024.0 * 1024.0;
    double GiB = 1024.0 * MiB;

    snprintf(buf, sizeof(buf),
        "VRAM Plan:\n"
        "  total:       %.2f GiB\n"
        "  reserved:    %.2f MiB\n"
        "  usable:      %.2f GiB\n"
        "  dense:       %.2f MiB\n"
        "  head:        %.2f MiB\n"
        "  kv_cache:    %.2f MiB (ctx=%lld)\n"
        "  scratch:     %.2f MiB\n"
        "  expert VRAM: %.2f GiB (%lld slots, %.1f%%)\n"
        "  feasible:    %s",
        (double)vram_total / GiB,
        (double)vram_reserved / MiB,
        (double)vram_usable / GiB,
        (double)dense_bytes / MiB,
        (double)head_bytes / MiB,
        (double)kv_cache_bytes / MiB,
        (long long)context_length,
        (double)scratch_bytes / MiB,
        (double)expert_vram_bytes / GiB,
        (long long)expert_slots,
        expert_vram_frac * 100.0,
        feasible() ? "yes" : "NO");

    return std::string(buf);
}

}  // namespace strata

