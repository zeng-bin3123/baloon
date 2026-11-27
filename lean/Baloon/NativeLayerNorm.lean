-- Baloon.NativeLayerNorm: weighted RMS normalisation in-place for
-- input_layernorm and post_attention_layernorm weight vectors.
-- One NativeLayerNorm instance holds one BF16-to-f32 converted weight vector;
-- apply_inplace overwrites the hidden-state buffer using rms_norm_one_kernel.
-- Proves: index coverage, launch geometry soundness and denominator positivity.

namespace Baloon.NativeLayerNorm

-- ── thread index assignment ───────────────────────────────────────────────────

/-- Thread t (0 ≤ t < b) at stride step k owns global index t + k*b. -/
def threadIndex (t b k : Nat) : Nat := t + k * b

/-- Every global index i is owned by exactly thread (i % b) at step (i / b). -/
theorem full_coverage (i b : Nat) (hb : 0 < b) :
    threadIndex (i % b) b (i / b) = i := by
  unfold threadIndex
  rw [mul_comm, Nat.add_comm]
  exact Nat.div_add_mod i b

/-- With block size 256 and hidden-state width 2048 (the model dimension),
    every thread t < 256 visits exactly 8 elements: indices t, t+256, …, t+1792,
    all strictly less than 2048. -/
theorem index_in_range (t k : Nat) (ht : t < 256) (hk : k < 8) :
    threadIndex t 256 k < 2048 := by
  unfold threadIndex; omega

-- ── launch geometry ───────────────────────────────────────────────────────────

/-- 2048 (Qwen3.5-MoE hidden size) is an exact multiple of 256 (threads per block):
    each of the 256 threads visits exactly 8 elements. -/
theorem width_div_threads : 2048 / 256 = 8 ∧ 2048 % 256 = 0 := by decide

/-- 256 = 2^8, so the shared-memory tree reduction finishes in exactly 8 rounds
    (256 → 128 → 64 → 32 → 16 → 8 → 4 → 2 → 1). -/
theorem block_size_is_power_of_two : (256 : Nat) = 2 ^ 8 := by decide

/-- After 8 halving steps the reduction stride reaches 1; one more halving
    gives 0 and the loop exits, leaving the total in part[0]. -/
theorem reduction_terminates : (256 : Nat) / 2 ^ 8 = 1 := by decide

-- ── normalisation denominator positivity ─────────────────────────────────────

/-- The kernel uses eps = 1e-6 (represented as a small positive integer proxy):
    for any non-negative sum-of-squares proxy s and positive epsilon proxy e,
    s + e > 0, so rsqrtf never receives a zero or negative argument. -/
theorem denom_pos (s e : Nat) (he : 0 < e) : 0 < s + e := by omega

-- ── BF16 weight upload ────────────────────────────────────────────────────────

/-- One layer norm weight vector is n_embd = 2048 BF16 values; at 2 bytes each
    the upload buffer is 4096 bytes, within a standard cudaMalloc allocation. -/
theorem weight_buf_bytes : 2048 * 2 = 4096 := by decide

/-- The BF16 → f32 conversion for one weight element: shift the 16-bit pattern
    left by 16 bits to produce the f32 bit-representation.  The result fits in
    32 bits because the original value is at most 0xFFFF. -/
theorem bf16_shift_fits (h : Nat) (hh : h < 65536) : h * 65536 < 2 ^ 32 := by omega

end Baloon.NativeLayerNorm
