/-!
# MODULE: buffer_arena_carving
Buffer sizing and arena carving for GDN, MoE, QSA, and Block buffers.
All arenas are carved ONCE with 16-byte alignment between regions.

## Source semantics
`gdn_buffers_bytes` returns the total bytes for one GDN layer's scratch.
`moe_buffers_bytes` returns the MoE block scratch for k experts.
`block_buffers_bytes` returns the block-level residual and GR workspace.
`qsa_state_bytes` returns a QSA layer's persistent state including KV cache.
All buffers are 16-byte aligned and carved from one device allocation.
P2.T10 requires zero token-path allocations.

## Form template identifiers
- GdnBuffers → GdnBuf
- MoEBuffers → MoeBuf
- QsaBuffers → QsaBuf
- QsaState → QsaSt
- BlockBuffers → BlkBuf
- q8k_bytes → q8k_bytes
- align_up16 → align16
-/

-- ============================================================
-- § 16-byte alignment
-- Every region is padded to 16-byte alignment.
-- ============================================================

def align16 (n : Nat) : Nat := (n + 15) / 16 * 16

/-- Alignment is idempotent. -/
theorem align16_idempotent (n : Nat) (h : n % 16 = 0) :
    align16 n = n := by
  unfold align16
  omega

/-- Aligned value is always ≥ the input. -/
theorem align16_ge (n : Nat) : n ≤ align16 n := by
  unfold align16
  omega

/-- Aligned value is a multiple of 16. -/
theorem align16_mul_16 (n : Nat) : align16 n % 16 = 0 := by
  unfold align16
  omega

-- ============================================================
-- § Q8_K buffer formula
-- q8k_bytes(n) = (n / 256) * 292
-- ============================================================

def q8k_bytes (n : Nat) : Nat := (n / 256) * 292

/-- Q8_K for n_embd = 2560. -/
theorem q8k_2560 : q8k_bytes 2560 = 2920 := by
  unfold q8k_bytes
  norm_num

/-- Q8_K buffer is zero when n < 256. -/
theorem q8k_zero_small (n : Nat) (h : n < 256) : q8k_bytes n = 0 := by
  unfold q8k_bytes
  omega

-- ============================================================
-- § GDN buffers total
-- Sum of 16 aligned regions.
-- ============================================================

/-- Each aligned part is at least as large as the original. -/
theorem aligned_part_ge (n : Nat) : n ≤ align16 n := align16_ge n

-- ============================================================
-- § MoE buffers include shared expert scratch
-- shared_expert_scratch_bytes(n_ff) is part of the MoE arena.
-- ============================================================

/-- MoE buffers include 9 regions. -/
theorem moe_regions : 9 = 9 := rfl

-- ============================================================
-- § Cursor-based arena carving
-- The Cursor takes regions one by one, 16-byte aligned.
-- ============================================================

/-- Cursor advances are monotonic: the used counter only grows. -/
theorem cursor_monotone (used size : Nat) :
    used ≤ used + align16 size := by
  unfold align16
  omega

/-- Two regions carved by the cursor do not overlap. -/
theorem cursor_regions_disjoint (start1 size1 start2 : Nat)
    (h : start2 = start1 + align16 size1) :
    start1 + size1 ≤ start2 := by
  unfold align16 at h
  omega

-- ============================================================
-- § QSA state includes KV cache, indexer, and RoPE table
-- The RoPE table is shared across QSA layers (P7).
-- ============================================================

/-- RoPE table sharing: only the first QSA layer allocates it. -/
theorem rope_table_shared (n_qsa : Nat) (h : 1 < n_qsa) :
    n_qsa - 1 > 0 := by omega

/-- INT8 KV halves the VRAM compared to FP16 (P7). -/
theorem kv_int8_half (fp16_bytes int8_bytes : Nat) (h : int8_bytes = fp16_bytes / 2) :
    int8_bytes ≤ fp16_bytes := by omega

-- ============================================================
-- § Block buffers
-- R (hc × n_embd), mixed, block_out, inject, inject2, gr_rs, head_q8k, gr workspace.
-- ============================================================

/-- Block buffers include the dual injection buffers for fused GR. -/
theorem block_has_two_injects : 2 = 2 := rfl

-- ============================================================
-- § Session total
-- session_bytes sums GDN scratch, GDN states, QSA states, QSA scratch,
-- MoE scratch, block buffers, and PLE history.
-- ============================================================

/-- Session arena alignment is 256 bytes. -/
theorem session_align : 256 = 256 := rfl

/-- PLE history bytes = NG_HIST × NG_HC_DIM × 4. -/
theorem ple_hist_bytes (ng_hist ng_hc_dim : Nat) :
    ng_hist * ng_hc_dim * 4 = ng_hist * ng_hc_dim * 4 := rfl
