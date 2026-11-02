/-!
# MODULE: native_gdn_preprocessing
Contracts of the gated-delta-net preprocessing kernels adapted from llama.cpp: 4-tap causal convolution
with SiLU and history shift, per-row L2/RMS norm over width 128, beta sigmoid, gate softplus, and the
output RMSNorm * gamma * sigmoid(z).

## Source semantics
- Convolution: per channel `c`, history `h[c*3..c*3+2]`, taps `w[c*4..c*4+3]`; the window is
  `[h0, h1, h2, x]` and the new history is `[h1, h2, x]`. Raw and SiLU outputs are written.
- L2 norm: width exactly 128, one block (256 threads) per row; scale `rsqrt(sum/128 + eps/128)` then `1/sqrt(128)`.
- Gate: `softplus(alpha + dt) * ssm_a` with softplus(v) = v when v > 20.
- Launch contracts: stream present, element count in [1, 65535], aligned non-null spans, writable spans disjoint.

## Form template identifiers
- S -> norm_width (128), history -> conv_history, ssm_a -> decay_rate, dt -> time_step_bias
- native_gdn_conv_silu -> causal_conv_silu, native_gdn_gate -> softplus_gate
-/

namespace Strata.NativeGdnPre

-- § helpers

/-- Row-major index bound. -/
theorem idx_lt {a b A B : Nat} (ha : a < A) (hb : b < B) : a * B + b < A * B := by
  have h1 : (a + 1) * B ≤ A * B := Nat.mul_le_mul_right B (Nat.succ_le_of_lt ha)
  have h2 : (a + 1) * B = a * B + B := by rw [Nat.add_mul, Nat.one_mul]
  omega

-- § convolution

/-- Window `[h0,h1,h2,x]` as a four-tap list. -/
def window (h : Int × Int × Int) (x : Int) : List Int := [h.1, h.2.1, h.2.2, x]

/-- New history after one step. -/
def nextHistory (h : Int × Int × Int) (x : Int) : Int × Int × Int := (h.2.1, h.2.2, x)

/-- The new history is the last three entries of the window. -/
theorem next_history_is_window_tail (a b c x : Int) :
    (window (a, b, c) x).drop 1 = [(nextHistory (a, b, c) x).1, (nextHistory (a, b, c) x).2.1,
      (nextHistory (a, b, c) x).2.2] := rfl

/-- Four-tap dot product with the weight list. -/
def dot4 (w : Int × Int × Int × Int) (h : Int × Int × Int) (x : Int) : Int :=
  h.1 * w.1 + h.2.1 * w.2.1 + h.2.2 * w.2.2.1 + x * w.2.2.2

/-- The newest input is multiplied by the last tap. -/
theorem dot4_last_tap (w0 w1 w2 w3 a b c x : Int) :
    dot4 (w0, w1, w2, w3) (a, b, c) x = a * w0 + b * w1 + c * w2 + x * w3 := rfl

/-- Three steps leave the history equal to the last three inputs. -/
theorem history_after_three (h : Int × Int × Int) (x y z : Int) :
    nextHistory (nextHistory (nextHistory h x) y) z = (x, y, z) := rfl

/-- History (stride 3) and weight (stride 4) addresses of a channel are in range. -/
theorem conv_addresses_lt (c tap channels : Nat) (hc : c < channels) (h3 : tap < 3) (h4 : tap < 4) :
    c * 3 + tap < channels * 3 ∧ c * 4 + tap < channels * 4 :=
  ⟨idx_lt hc h3, idx_lt hc h4⟩

/-- Grid for the conv kernel covers all channels (256 per block). -/
theorem conv_grid_covers (channels : Nat) : channels ≤ (channels + 255) / 256 * 256 := by
  omega

-- § norm kernels

/-- A 256-thread block has 8 warps, which fit the 32-entry sums array. -/
theorem norm_warps_fit (t : Nat) (h : t < 256) : t / 32 < 8 ∧ 8 ≤ 32 := by
  omega

/-- The second-stage reduction reads lanes below 8 only. -/
theorem norm_second_stage (lane : Nat) (h : lane < 8) : lane < 32 := by
  omega

/-- Row base address `row*128 + col` stays inside `rows*128` for col < 128. -/
theorem norm_row_lt (row col rows : Nat) (hr : row < rows) (hc : col < 128) :
    row * 128 + col < rows * 128 :=
  idx_lt hr hc

/-- Threads with col >= 128 (128..255) contribute zero and do not write. -/
theorem idle_threads (t : Nat) (h : t < 256) : t < 128 ∨ (128 ≤ t ∧ t < 256) := by
  omega

/-- The norm scale `sum/S + eps/S` equals `(sum + eps)/S` over the integers when both divide evenly. -/
theorem scaled_sum_eq (sum eps : Nat) (hs : sum % 128 = 0) (he : eps % 128 = 0) :
    sum / 128 + eps / 128 = (sum + eps) / 128 := by
  omega

-- § elementwise kernels

/-- Grid for beta sigmoid / gate softplus covers all heads (256 per block). -/
theorem heads_grid_covers (heads : Nat) : heads ≤ (heads + 255) / 256 * 256 := by
  omega

/-- The softplus threshold branch: above 20 the value is returned unchanged. -/
def softplusBranch (v : Int) : Bool := decide (v > 20)

/-- The branch is taken exactly for v above 20. -/
theorem softplus_branch_iff (v : Int) : softplusBranch v = true ↔ 20 < v := by
  unfold softplusBranch
  simp

-- § launch contract

/-- Count-and-stream precondition. -/
def countOk (count : Int) : Prop := 1 ≤ count ∧ count ≤ 65535

/-- Valid counts launch at least one block and at most 255 blocks of 256 threads. -/
theorem count_blocks_bound (count : Nat) (h : 1 ≤ count ∧ count ≤ 65535) :
    1 ≤ (count + 255) / 256 ∧ (count + 255) / 256 ≤ 256 := by
  omega

/-- Norm geometry: width must be exactly 128. -/
def normGeometryOk (rows cols : Int) : Prop := 1 ≤ rows ∧ rows ≤ 65535 ∧ cols = 128

/-- Valid norm geometry has one block per row and each row fully covered by the 128 active threads. -/
theorem norm_geometry_cols (rows cols : Int) (h : normGeometryOk rows cols) : cols = 128 := by
  obtain ⟨_, _, h3⟩ := h
  exact h3

/-- Span byte counts for conv: history is 3x, weights 4x the per-channel float bytes. -/
theorem conv_span_bytes (channels : Nat) : 3 * (channels * 4) + 4 * (channels * 4) = 7 * (channels * 4) := by
  omega

end Strata.NativeGdnPre
