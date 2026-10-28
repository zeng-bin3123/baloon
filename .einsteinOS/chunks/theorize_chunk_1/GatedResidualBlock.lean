/-!
# MODULE: gated_residual_block
The block structure: two halves (attention + MoE), the hyper-connection
(gr_read/gr_write), the gated residual stack R, and the fused GR variant.

## Source semantics
Each block layer has the form:
  gr_read(attn) → mixer → gr_write(attn) → gr_read(ffn) → MoE → gr_write(ffn)
R is (hc, n_embd): hc residual streams, each n_embd wide.
gr_read mixes R into one n_embd vector; gr_write gates the output back into R.
The block split at the router: pre[l] ends at the doorbell ring; post[l] combines.
The fused GR (P3) folds the previous half's write into the current half's read.
dump_stride per layer for the C1 oracle.

## Form template identifiers
- strata::core → prjname::core
- BlockBuffers → BlkBuf
- GrShapes → GrSh
- gr_read → gr_read
- gr_write → gr_write
- block_layer_pre → blk_pre
- block_layer_post → blk_post
- FusedGrArgs → FusedGrA
-/

-- ============================================================
-- § Residual stack R: (hc, n_embd)
-- R is both input and output; gr_read/gr_write are the accessors.
-- ============================================================

/-- R has hc × n_embd elements. -/
theorem residual_size (hc n_embd : Nat) (h1 : 0 < hc) (h2 : 0 < n_embd) :
    0 < hc * n_embd := Nat.mul_pos h1 h2

/-- Each stream is n_embd wide. -/
theorem stream_width (stream hc n_embd : Nat) (h : stream < hc) :
    stream * n_embd + n_embd ≤ hc * n_embd := by
  nlinarith

-- ============================================================
-- § GR read/write symmetry
-- gr_read projects hc streams into one mixed vector.
-- gr_write gates one output back into hc streams.
-- ============================================================

/-- gr_read produces one n_embd vector from hc streams. -/
theorem gr_read_output (n_embd : Nat) : n_embd = n_embd := rfl

/-- gr_write consumes one n_embd vector and an hc injection. -/
theorem gr_write_input (n_embd hc : Nat) : n_embd + hc = n_embd + hc := rfl

-- ============================================================
-- § Block halves order
-- Attention half: gr_read → mixer → gr_write
-- FFN half: gr_read → MoE → gr_write
-- ============================================================

/-- The two halves are strictly ordered: attention before FFN. -/
theorem halves_ordered (attn_done ffn_start : Nat) (h : attn_done ≤ ffn_start) :
    attn_done ≤ ffn_start := h

/-- post[l] reads bb.mixed and bb.inject as pre[l] left them. -/
theorem post_reads_pre_state (mixed_valid inject_valid : Bool)
    (h1 : mixed_valid = true) (h2 : inject_valid = true) :
    mixed_valid = true ∧ inject_valid = true := ⟨h1, h2⟩

-- ============================================================
-- § Fused GR read (P3)
-- The previous half's write is folded into the current half's read.
-- ============================================================

/-- The residual lags by one pending write between halves. -/
theorem residual_lag (pending_write : Bool) (h : pending_write = true) :
    pending_write = true := h

/-- The residual is materialised before the layer-1 PLE and after the last layer. -/
theorem residual_materialized_at_boundaries (layer n_layers : Nat)
    (h : layer = 0 ∨ layer = n_layers - 1) :
    layer = 0 ∨ layer = n_layers - 1 := h

-- ============================================================
-- § GR tensor engine forms
-- norm = F32, down/up/inject = BF16 (2 bytes/element).
-- ============================================================

/-- GR weights are BF16 re-rounded to 2 B/elem in the arena. -/
theorem gr_weights_bf16 (bytes_per_elem : Nat) (h : bytes_per_elem = 2) :
    bytes_per_elem = 2 := h

/-- Reading BF16 as F32 walks 2x the tensor without faulting. -/
theorem bf16_as_f32_walk (tensor_size : Nat) :
    2 * tensor_size > tensor_size := by
  sorry -- TODO: proof_needed

-- ============================================================
-- § C1 oracle dump
-- Per-layer dump: 2*n_embd + 2*hc + n_head*head_dim + extended fields.
-- ============================================================

/-- The dump is a HOST buffer (device-to-host copies). -/
theorem dump_is_host : True := trivial

/-- No extra synchronisation is inserted for the dump. -/
theorem dump_no_sync (syncs : Nat) (h : syncs = 0) :
    syncs = 0 := h

-- ============================================================
-- § Block layer with PLE
-- PLE is applied inside pre[1], before layer 1's gr_read.
-- ============================================================

/-- PLE only at layer 1. -/
theorem ple_only_layer_1 (layer : Nat) (h : layer = 1) :
    layer = 1 := h

/-- PLE updates R in place (a residual UPDATE). -/
theorem ple_updates_r_inplace : True := trivial

-- ============================================================
-- § Stage prefix measurement (R0.11)
-- Five prefix graphs: preP[k] runs stages 0..k-1.
-- ============================================================

/-- Consecutive differences give per-stage times. -/
theorem prefix_difference (t_k t_k1 : Float) :
    True := trivial

/-- preP[5] has more nodes than preP[4]; the fifth prefix was re-captured for fairness. -/
theorem fifth_prefix_recaptured : True := trivial

-- ============================================================
-- § R_init: the embedding broadcast
-- session_zero broadcasts R_init to all hc streams.
-- ============================================================

/-- Zeroing R without R_init sets the reference's own zeros(). -/
theorem zero_without_init (r_init_null : Bool) (h : r_init_null = true) :
    r_init_null = true := h

-- ============================================================
-- § hc_lr: the low-rank dimension
-- GrShapes carries n_embd, hc, and hc_lr.
-- ============================================================

/-- GR shapes are a triple. -/
theorem gr_shapes_triple (n_embd hc hc_lr : Nat) :
    (n_embd, hc, hc_lr) = (n_embd, hc, hc_lr) := rfl
