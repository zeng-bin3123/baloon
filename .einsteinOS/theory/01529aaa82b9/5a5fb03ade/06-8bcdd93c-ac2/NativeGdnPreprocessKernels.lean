/-!
# MODULE: native_gdn_preprocess_kernels
Conv+SiLU history update, 128-wide norms with padded 256-thread block reductions, and the span validation
of the native GDN preprocessing path.

## Source semantics
`conv_silu` reads `[h0,h1,h2,x]`, accumulates the four taps in order starting from 0, then stores history
`[h1,h2,x]`.  `l2_norm` and `out_norm` use 256 threads for a 128-wide row: threads `128..255` contribute 0
to the block sum, and the 8 warp partials land in `sums[tid/32]`.  Host checks: count in `[1, 65535]`,
a stream, width exactly 128, aligned non-null spans, pairwise disjoint writable spans.

## Form template identifiers
- `history`/`values` -> `win`, `newHist`
- `norm_sum`          -> `sumRange (padded f) 256`
- `Span`              -> `overlap`
-/

namespace Strata.Kernels.NativeGdnPre

-- § arithmetic helpers

/-- Row-major bound. -/
theorem mul_add_lt_aux {a b c n : Nat} (ha : a < b) (hc : c < n) : a * n + c < b * n := by
  have h : (a + 1) * n ≤ b * n := Nat.mul_le_mul_right n ha
  rw [Nat.add_mul, Nat.one_mul] at h
  omega

/-- Sequential sum of `f 0 .. f (n-1)`. -/
def sumRange (f : Nat → Int) : Nat → Int
  | 0 => 0
  | n + 1 => sumRange f n + f n

/-- A range sum splits at any point. -/
theorem sumRange_add (f : Nat → Int) (n m : Nat) :
    sumRange f (n + m) = sumRange f n + sumRange (fun i => f (n + i)) m := by
  induction m with
  | zero => simp [sumRange]
  | succ m ih =>
    show sumRange f (n + m) + f (n + m) =
      sumRange f n + (sumRange (fun i => f (n + i)) m + f (n + m))
    rw [ih]
    omega

/-- Pointwise-equal summands give equal sums. -/
theorem sumRange_congr (f g : Nat → Int) (n : Nat) (h : ∀ i, i < n → f i = g i) :
    sumRange f n = sumRange g n := by
  induction n with
  | zero => rfl
  | succ n ih =>
    show sumRange f n + f n = sumRange g n + g n
    have h1 := ih (fun i hi => h i (by omega))
    have h2 := h n (by omega)
    omega

/-- A sum of zeros is zero. -/
theorem sumRange_zero (f : Nat → Int) (n : Nat) (h : ∀ i, i < n → f i = 0) :
    sumRange f n = 0 := by
  induction n with
  | zero => rfl
  | succ n ih =>
    show sumRange f n + f n = 0
    have h1 := ih (fun i hi => h i (by omega))
    have h2 := h n (by omega)
    omega

-- § padded block reduction

/-- Thread `tid` contributes `f tid` when inside the 128-wide row and `0` otherwise. -/
def padded (f : Nat → Int) : Nat → Int := fun i => if i < 128 then f i else 0

/-- Running 256 threads over a 128-wide row computes exactly the 128-element sum. -/
theorem padded_norm_sum (f : Nat → Int) : sumRange (padded f) 256 = sumRange f 128 := by
  have h : sumRange (padded f) 256 =
      sumRange (padded f) 128 + sumRange (fun i => padded f (128 + i)) 128 :=
    sumRange_add (padded f) 128 128
  have h1 : sumRange (padded f) 128 = sumRange f 128 :=
    sumRange_congr _ _ 128 (fun i hi => by simp [padded, hi])
  have h2 : sumRange (fun i => padded f (128 + i)) 128 = 0 :=
    sumRange_zero _ 128 (fun i _ => by
      have hn : ¬ (128 + i < 128) := by omega
      simp [padded, hn])
  omega

/-- Warp partials of a 256-thread block fit the 8 slots read by lanes `< 8` and the 32-slot array. -/
theorem warp_partial_slot {tid : Nat} (h : tid < 256) : tid / 32 < 8 ∧ 8 ≤ 32 := by
  omega

-- § conv + silu window

/-- Integer dot product of two lists. -/
def dot : List Int → List Int → Int
  | a :: as, b :: bs => a * b + dot as bs
  | _, _ => 0

/-- Left-to-right tap accumulation from 0 equals the dot of the 4-wide window. -/
theorem conv_taps_sum (v0 v1 v2 v3 w0 w1 w2 w3 : Int) :
    ((0 + v0 * w0) + v1 * w1 + v2 * w2) + v3 * w3 = dot [v0, v1, v2, v3] [w0, w1, w2, w3] := by
  simp only [dot]
  omega

/-- The new history is the window without its oldest entry: `[h1,h2,x]`. -/
def newHist (h0 h1 h2 x : Int) : List Int := ([h0, h1, h2, x]).drop 1

/-- The new history has three rows and ends with the new input. -/
theorem newHist_shape (h0 h1 h2 x : Int) :
    newHist h0 h1 h2 x = [h1, h2, x] ∧ (newHist h0 h1 h2 x).length = 3 := by
  simp [newHist]

/-- History row `c*3 + tap` is inside `3*channels`; weights `c*4 + tap` inside `4*channels`. -/
theorem conv_indices {c channels t : Nat} (hc : c < channels) (ht : t < 3) :
    c * 3 + t < channels * 3 ∧ c * 4 + t < channels * 4 :=
  ⟨mul_add_lt_aux hc ht, mul_add_lt_aux hc (by omega)⟩

/-- Channel launch of 256-thread blocks covers `channels`. -/
theorem conv_launch_cover (channels : Nat) : channels ≤ ((channels + 255) / 256) * 256 := by
  omega

/-- Row offset `blockIdx*128 + col` stays inside `rows*128`. -/
theorem row_offset_lt {b col rows : Nat} (hb : b < rows) (hc : col < 128) :
    b * 128 + col < rows * 128 :=
  mul_add_lt_aux hb hc

/-- The host's count window `[1, 65535]` keeps the 1-D grids within the 16-bit bound. -/
theorem count_window {n : Nat} (h1 : 1 ≤ n) (h2 : n ≤ 65535) :
    0 < (n + 255) / 256 ∧ (n + 255) / 256 ≤ 256 := by
  omega

-- § span validation

/-- Interval overlap. -/
def overlap (a an b bn : Nat) : Prop := a < b + bn ∧ b < a + an

/-- Adjacent spans never overlap. -/
theorem adjacent_disjoint {a an b bn : Nat} (h : a + an ≤ b) : ¬ overlap a an b bn := by
  unfold overlap
  omega

/-- Overlap is symmetric, so checking `i < j` pairs suffices for pairwise disjointness. -/
theorem overlap_symm (a an b bn : Nat) : overlap a an b bn ↔ overlap b bn a an := by
  unfold overlap
  exact ⟨fun h => ⟨h.2, h.1⟩, fun h => ⟨h.2, h.1⟩⟩

/-- Span byte counts in the conv wrapper: history `3*bytes` and weights `4*bytes` are multiples of 4. -/
theorem conv_span_sizes (channels : Nat) :
    (3 * (channels * 4)) % 4 = 0 ∧ (4 * (channels * 4)) % 4 = 0 := by
  omega

end Strata.Kernels.NativeGdnPre
