/-!
# MODULE: native_gdn_preprocess
Pinned-llama.cpp gated-delta-net preprocessing kernels: 4-tap causal conv + SiLU with history shift, per-row
L2 (RMS-style) norm with post-scale, beta sigmoid, alpha+dt softplus gate, and RMSNorm*gamma*sigmoid(z) output norm.

## Source semantics
* Conv: window `[h0, h1, h2, x]`, output = dot with 4 weights (`weights[c*4 + tap]`); the history becomes `[h1, h2, x]`.
* Norms run one block of 256 threads per 128-wide row; threads ≥ 128 contribute zero; the reduction has 8 warps
  and `sums[32]` has room for them.
* Counts must lie in [1, 65535]; geometry must be 128 wide; epsilon finite and non-negative.
* Writable spans must be aligned, valid, and pairwise disjoint from each other and from the inputs.
* L2 norm passes `epsilon/S` as the kernel epsilon so that `rsqrt(sum/S + eps/S)` is applied.

## Form template identifiers
`conv_silu` → causal_conv_silu ; `l2_norm` → row_l2_norm ; `beta_sigmoid` → write_strength ;
`gate_softplus` → decay_gate ; `out_norm` → gated_output_norm ; `norm_sum` → block_sum_8warps.
-/

namespace Strata.GdnPreprocess

-- § conv

/-- One-block-per-256 channels: the grid covers all channels. -/
theorem grid_covers (n : Nat) : n ≤ (n + 255) / 256 * 256 := by omega

/-- Block count for n in [1, 65535] is at most 256. -/
theorem grid_bound (n : Nat) (h : n ≤ 65535) : (n + 255) / 256 ≤ 256 := by omega

/-- Weight index c*4 + tap (tap < 4) stays inside 4*channels. -/
theorem weight_index_lt (c tap n : Nat) (hc : c < n) (ht : tap < 4) : c * 4 + tap < 4 * n := by omega

/-- History index c*3 + t (t < 3) stays inside 3*channels. -/
theorem history_index_lt (c t n : Nat) (hc : c < n) (ht : t < 3) : c * 3 + t < 3 * n := by omega

/-- History/weight cells of distinct channels are distinct. -/
theorem history_unique (c t c' t' : Nat) (ht : t < 3) (ht' : t' < 3) (h : c * 3 + t = c' * 3 + t') :
    c = c' ∧ t = t' := by omega

/-- Window shift: the new history is the old window without its oldest entry. -/
def shift (w : List Float) : List Float := w.drop 1

/-- The window has 4 entries; the shifted history has 3. -/
theorem shift_length (a b c x : Float) : (shift [a, b, c, x]).length = 3 := by simp [shift]

/-- The shifted history is exactly [b, c, x]. -/
theorem shift_value (a b c x : Float) : shift [a, b, c, x] = [b, c, x] := by simp [shift]

/-- Byte spans: history needs 3x, weights 4x the per-channel float bytes. -/
theorem conv_span_bytes (n : Nat) : 3 * (n * 4) + 4 * (n * 4) = 28 * n := by omega

-- § row norms

/-- The reduction uses 8 warps for 256 threads. -/
theorem norm_warps : 256 / 32 = 8 := rfl

/-- 8 partial sums fit in `sums[32]`. -/
theorem norm_sums_fit : 8 ≤ 32 := by decide

/-- Warp index of thread t < 256 is below 8. -/
theorem warp_index_lt (t : Nat) (h : t < 256) : t / 32 < 8 := by omega

/-- Active threads (col < 128) occupy warps 0..3 only; the other four warps add zero. -/
theorem active_warps (t : Nat) (h : t < 128) : t / 32 < 4 := by omega

/-- Row base offset blockIdx*128 + col is below rows*128. -/
theorem row_offset_lt (b col rows : Nat) (hb : b < rows) (hc : col < 128) : b * 128 + col < rows * 128 := by
  have h1 : (b + 1) * 128 ≤ rows * 128 := Nat.mul_le_mul_right 128 (Nat.succ_le_of_lt hb)
  omega

/-- Kernel epsilon is eps/S, so `sum/S + eps/S` is `(sum + eps)/S`: integer shadow with S | both. -/
theorem eps_fold (s e : Nat) (h : e % 128 = 0) (h2 : s % 128 = 0) : s / 128 + e / 128 = (s + e) / 128 := by omega

/-- Geometry check accepts only width 128. -/
def normGeometryOk (rows cols : Nat) : Bool := decide (1 ≤ rows ∧ rows ≤ 65535) && cols == 128

theorem normGeometry_iff (rows cols : Nat) :
    normGeometryOk rows cols = true ↔ (1 ≤ rows ∧ rows ≤ 65535) ∧ cols = 128 := by
  unfold normGeometryOk; simp

-- § beta / gate

/-- Elementwise kernels: thread i < count handles element i, and the grid covers all. -/
theorem elementwise_cover (i n : Nat) (h : i < n) : i / 256 < (n + 255) / 256 := by omega

/-- softplus branch: values above 20 pass through unchanged (threshold model). -/
def softplusBranchIsIdentity (v : Int) : Bool := decide (20 < v)

theorem softplus_branch_large (v : Int) (h : 20 < v) : softplusBranchIsIdentity v = true := by
  unfold softplusBranchIsIdentity; exact decide_eq_true h

theorem softplus_branch_small (v : Int) (h : v ≤ 20) : softplusBranchIsIdentity v = false := by
  unfold softplusBranchIsIdentity; exact decide_eq_false (by omega)

-- § span checks

/-- Overlap of spans. -/
def overlap (a an b bn : Nat) : Prop := a < b + bn ∧ b < a + an

theorem overlap_symm (a an b bn : Nat) : overlap a an b bn ↔ overlap b bn a an := by
  unfold overlap; omega

theorem overlap_end_to_end (a an b bn : Nat) (h : a + an ≤ b) : ¬ overlap a an b bn := by
  unfold overlap; omega

/-- A span is valid when aligned to 4 and its end does not wrap. -/
def validSpan (addr bytes : Nat) (ptrMax : Nat) : Prop := addr ≠ 0 ∧ addr % 4 = 0 ∧ bytes ≤ ptrMax - addr

theorem validSpan_nonnull (a b m : Nat) (h : validSpan a b m) : a ≠ 0 := h.1

end Strata.GdnPreprocess
