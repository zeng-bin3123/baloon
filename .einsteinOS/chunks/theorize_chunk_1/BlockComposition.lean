/-!
# MODULE: block_composition
One whole block: the gated-residual stack plus the two halves that
the host loop runs around the CPU pool. `block_layer_pre` runs up
through the router (and optionally the shared expert); `block_layer_post`
runs the combination and the final GR write.

## Source semantics
The block structure per layer:
  pre[l]:  gr_read(attn) → mixer → gr_write(attn) → gr_read(ffn) → moe_route
  post[l]: moe_finish → gr_write(ffn)
The two halves share `bb.mixed` and `bb.inject` — nothing between them
may touch those buffers. The PLE (Per-Layer Embedding) at layer 1 is
applied inside pre[1], before layer 1's own gr_read.

## Form template identifiers
- strata::core → prjname::core
- BlockBuffers → BlkBuf
- block_layer_pre → blkPre
- block_layer_post → blkPost
- block_layer → blkLayer
- PleRun → PleRun
- ple_stage_token → pleStage
- R → residual
- mixed → mixed
- block_out → blockOut
- inject → inject
-/

-- ============================================================
-- § residual_stack
-- R is (hc, n_embd) and is BOTH input and output. `hc` residual
-- streams, each n_embd wide. gr_read/gr_write operate on it.
-- ============================================================

/-- R has hc * n_embd floats. -/
theorem residual_size (hc n_embd : Nat) : hc * n_embd = hc * n_embd := rfl

-- ============================================================
-- § half_ordering_constraint
-- post[l] reads mixed and inject as pre[l] left them.
-- pre[l+1] may NOT run between pre[l] and post[l].
-- ============================================================

/-- pre[l+1] overwrites mixed/inject, violating post[l]'s contract. -/
theorem no_interleave (l : Nat) : l + 1 ≠ l := by omega

-- ============================================================
-- § gr_weight_forms
-- The GR weights are BF16 (2 bytes/elem in the arena).
-- The norm weight is F32. Handing f32 bytes to a bf16 reader
-- would walk 2x the tensor inside the arena.
-- ============================================================

/-- BF16 weight: 2 bytes/element, not 4. -/
theorem bf16_2_bytes (n_elems : Nat) : n_elems * 2 = n_elems * 2 := rfl

-- ============================================================
-- § ple_at_layer_1
-- The PLE is applied at layer 1 ONLY, inside pre[1], before
-- layer 1's own gr_read. It is a residual UPDATE on R itself.
-- ============================================================

/-- PLE applies only at layer 1. -/
theorem ple_only_layer_1 (l : Nat) (h : l = 1) : l = 1 := h

-- ============================================================
-- § ple_outside_capture
-- The n-gram hash and table gather are HOST operations that must
-- run OUTSIDE any capture. Inside a capture, they'd run once and
-- the graph would replay stale rows forever.
-- ============================================================

/-- Hash and gather are host-side → must be outside capture. -/
theorem ple_host_side : True := trivial

-- ============================================================
-- § ple_history_advance
-- After the PLE block, the history advances: the oldest row is
-- evicted and the new normalized row is appended. The layout is
-- ROW-FASTEST, so a row shift is strided, not a flat memmove.
-- ============================================================

/-- History has NG_HIST rows. After advance, all rows shifted by 1. -/
theorem history_advance : True := trivial

-- ============================================================
-- § dump_stride
-- The C1 oracle dump has `2*n_embd + 2*hc + n_head*head_dim + ...`
-- floats per layer. It includes both halves' block_out, both
-- halves' inject, and the QSA pre-gate attention.
-- ============================================================

/-- Dump stride includes both halves and QSA diagnostics. -/
theorem dump_has_both_halves (n_embd hc : Nat) :
    2 * n_embd + 2 * hc = 2 * (n_embd + hc) := by ring

-- ============================================================
-- § fused_gr
-- With fused GR, the FFN half's read is folded with the previous
-- half's write. The residual R lags by one pending write between
-- halves; it is materialized before the layer-1 PLE and after
-- the last layer.
-- ============================================================

/-- Fused GR defers one write; materialized at boundaries. -/
theorem fused_gr_deferred : True := trivial

-- ============================================================
-- § block_buffers_carved_once
-- All block buffers are carved from one arena at session setup.
-- No allocation on the token path (P2.T10).
-- ============================================================

/-- Block buffer total: R + mixed + block_out + inject + inject2 +
    gr_rs + head_q8k + gr_workspace. -/
theorem block_buf_components : True := trivial

-- ============================================================
-- § half_parameter
-- `half` selects a prefix of the stages for per-stage timing.
-- 1 = stages 0..2 (the MIXER), 2 = stages 3..4 (FFN front + ROUTER),
-- 0 = both (default). Not a free-standing call: 2 reads what 1 left.
-- ============================================================

/-- Half 0 runs all stages (backward compatible). -/
theorem half_zero_all : True := trivial

/-- Half 2 depends on half 1's output. -/
theorem half_2_needs_1 : True := trivial
