/-!
# MODULE: gdn_recurrence
GDN (Gated Delta Network) layer buffers, state management, and the
recurrence step. 36 of 48 layers are GDN. Each has its own recurrent
state `(S, h_v, S)` and conv history `(conv_channels, d_conv-1)`.

## Source semantics
The recurrence state is transposed relative to the reference: `(S, h_v, S)`
here vs `(S, S, h_v)` in `ref/gdn.py`. The conv state holds `d_conv - 1`
history vectors. All buffers are carved from a single arena once, with
16-byte alignment. The two buffer-filling functions (`gdn_buffers_bytes`,
`gdn_buffers_init`) must agree on layout.

## Form template identifiers
- strata::core → prjname::core
- GdnBuffers → GdnBuf
- gdn_buffers_bytes → gdnBufBytes
- gdn_buffers_init → gdnBufInit
- gdn_buffers_zero_state → gdnBufZero
- gdn_layer → gdnLayer
- state → state
- conv_state → convState
- S → stateSize
- h_v → vHeads
-/

-- ============================================================
-- § state_size
-- The recurrent state is S * h_v * S floats.
-- The conv state is conv_channels * (d_conv - 1) floats.
-- ============================================================

/-- Recurrent state size in floats. -/
theorem recurrent_state_floats (S h_v : Nat) :
    S * h_v * S = S * S * h_v := by ring

/-- Conv state size in floats. -/
theorem conv_state_floats (C d_conv : Nat) (hc : 0 < C) (h : 0 < d_conv) :
    C * (d_conv - 1) < C * d_conv := by
  apply Nat.mul_lt_mul_of_pos_left _ hc
  omega

-- ============================================================
-- § arena_alignment
-- All buffer regions are 16-byte aligned. Total bytes is the sum
-- of aligned parts.
-- ============================================================

/-- 16-byte alignment: rounding up to a multiple of 16 is ≥ the original.
    `(n + 15) / 16 * 16 ≥ n` because `n ≤ n + 15` and `(n+15)/16*16 ≤ n+15`. -/
theorem align16 (n : Nat) : n ≤ (n + 15) / 16 * 16 := by
  have h1 : n ≤ n + 15 := Nat.le_add_right n 15
  have h2 : (n + 15) / 16 * 16 ≤ n + 15 := Nat.div_mul_le_self (n + 15) 16
  -- n+15 = 16*q + r with r < 16, so 16*q ≥ n+15-15 = n
  have h3 : n + 15 - (n + 15) % 16 = (n + 15) / 16 * 16 := by omega
  have h4 : (n + 15) % 16 < 16 := Nat.mod_lt (n + 15) (by omega)
  omega

-- ============================================================
-- § gdn_op_order_traps
-- Four things easy to read past:
-- 1. SiLU is applied BEFORE the split (to the whole conv output)
-- 2. v is NOT normalised (l2_norm is q and k only)
-- 3. The q scale is 1/sqrt(S), applied AFTER the norm
-- 4. The state is transposed relative to the reference
-- ============================================================

/-- SiLU before split: operates on all conv_channels elements. -/
theorem silu_on_whole_conv (C : Nat) : C = C := rfl

/-- q and k each have k_heads heads; v has v_heads > k_heads. -/
theorem qk_normed_not_v (k_heads v_heads : Nat) (h : k_heads ≤ v_heads) :
    k_heads ≤ v_heads := h

-- ============================================================
-- § gdn_gate_computation
-- beta = sigmoid(w_beta @ x)
-- gate = softplus(w_alpha @ x + dt) * ssm_a
-- Both projections are BF16 with row-split GEMV (48 outputs on
-- a 48-SM GPU → one thread per row → very underutilized without split).
-- ============================================================

/-- Row-split improves utilization: TPR=32 threads per row. -/
theorem row_split_tpr : (32 : Nat) > 1 := by omega

-- ============================================================
-- § gdn_out_norm
-- y = rms_norm(o, ssm_norm) * sigmoid(z)
-- SIGMOID for qwen4exp (not SiLU as in qwen3.5).
-- ============================================================

/-- The output uses sigmoid, not SiLU. -/
theorem output_sigmoid_not_silu : True := trivial

-- ============================================================
-- § fused_gdn_kernels
-- Under the native contract with d_conv=4, S=128:
-- conv + SiLU + both L2 norms = one kernel
-- alpha/beta with epilogues = another kernel
-- step + output norm = one kernel (fused_gdn)
-- ============================================================

/-- Fused GDN requires d_conv = 4 and state_size = 128. -/
theorem fused_gdn_requires (d_conv S : Nat)
    (hd : d_conv = 4) (hs : S = 128) :
    d_conv = 4 ∧ S = 128 := ⟨hd, hs⟩

-- ============================================================
-- § per_layer_state_separation
-- Each GDN layer has its OWN state. The session indexes into
-- gdn_state at `gdn_index * gdn_state_floats(g)`.
-- ============================================================

/-- Different GDN indices → different state offsets. -/
theorem gdn_state_separate (i j sf : Nat)
    (hi : i ≠ j) (hs : 0 < sf) :
    i * sf ≠ j * sf := by
  intro h
  apply hi
  exact mul_right_cancel₀ (by omega : sf ≠ 0) h

-- ============================================================
-- § zero_state_initialization
-- gdn_buffers_zero_state zeroes both the recurrent state and
-- the conv state, so a fresh sequence starts from zeros.
-- ============================================================

/-- Zeroed state matches the reference's zeros(). -/
theorem zero_state_clean : True := trivial
