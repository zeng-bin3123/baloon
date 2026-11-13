/-!
# MODULE: native_gdn_preprocess
SOURCES: src/kernels/cuda/native_gdn_preprocess.cu
The llama.cpp-pinned GDN preprocessing kernels: 4-tap causal conv producing both the raw and
the SiLU output, row L2 norm (expressed as an RMS norm with rescaled epsilon followed by a
`1/sqrt(S)` scale), beta sigmoid, softplus gate, and the gated output RMS norm, together with
their host-side span and geometry validation.

## Source semantics
* Conv history per channel is 3 floats, `values = [h0,h1,h2,x]`, raw output `Σ values*w`,
  history becomes `[h1,h2,x]`; `d_conv` must be 4.
* `l2_norm` calls RMS norm with `eps/S` and post-scale `1/sqrt(S)`:
  `x * rsqrt(Σx²/S + eps/S) / sqrt(S) = x * rsqrt(Σx² + eps)`, i.e. the L2 norm.
* `gate = softplus(alpha + dt) * ssm_a`, with the `> 20` identity branch.
* Validation: `1 ≤ count ≤ 65535`, stream non-null, norm width exactly 128, epsilon finite
  and `≥ 0`; writable spans pairwise disjoint and disjoint from inputs.

## Form template identifiers
conv window ↦ `convWindow`, history shift ↦ `shiftHist`, rescaled norm denominator ↦
`rmsDenom`, count check ↦ `countOk`, disjointness ↦ `disjoint`.
-/

namespace Ref.NativeGdnPre

-- § conv + silu

/-- The 4-entry window `[h0,h1,h2,x]` for one channel. -/
def convWindow (h0 h1 h2 x : Int) : List Int := [h0, h1, h2, x]

/-- Dot product of coefficient lists. -/
def dot : List Int → List Int → Int
  | a :: as, b :: bs => a * b + dot as bs
  | _, _ => 0

/-- The new history is the window without its oldest entry. -/
def shiftHist (h0 h1 h2 x : Int) : List Int := (convWindow h0 h1 h2 x).drop 1

/-- The shifted history is `[h1,h2,x]` and stays 3 entries long. -/
theorem shiftHist_eq (h0 h1 h2 x : Int) :
    shiftHist h0 h1 h2 x = [h1, h2, x] ∧ (shiftHist h0 h1 h2 x).length = 3 := by
  simp [shiftHist, convWindow]

/-- Raw conv output: tap 0 (oldest) pairs with `w[c*4]`, the new input with `w[c*4+3]`. -/
theorem conv_raw (h0 h1 h2 x w0 w1 w2 w3 : Int) :
    dot (convWindow h0 h1 h2 x) [w0, w1, w2, w3] = h0 * w0 + (h1 * w1 + (h2 * w2 + (x * w3 + 0))) :=
  rfl

/-- The pinned zero-bias add (`sum + 0`) is the identity in exact arithmetic. -/
theorem zero_bias_identity (s : Int) : s + 0 = s := Int.add_zero s

/-- Channel `c`'s history occupies `[3c, 3c+3)`, inside the `3*channels` buffer and disjoint
    from every other channel's. -/
theorem hist_slots {c ch t : Nat} (hc : c < ch) (ht : t < 3) : c * 3 + t < 3 * ch := by omega

/-- Distinct channels have distinct history slots. -/
theorem hist_slots_unique {c c' t t' : Nat} (ht : t < 3) (ht' : t' < 3)
    (h : c * 3 + t = c' * 3 + t') : c = c' ∧ t = t' := by omega

-- § l2 via rescaled rms

/-- Denominator inside `rsqrt` for the RMS formulation: `ss/S + eps/S`, scaled by `S`. -/
def rmsDenom (S ss eps : Rat) : Rat := ss / S + eps / S

/-- Multiplying the rescaled RMS denominator by `S` recovers the L2 denominator `ss + eps`;
    together with the post-scale `1/sqrt S` this is exactly the L2 norm. -/
theorem rmsDenom_times_S (ss eps : Rat) : rmsDenom 128 ss eps * 128 = ss + eps := by
  unfold rmsDenom
  rw [Rat.add_mul, Rat.div_mul_cancel (by decide), Rat.div_mul_cancel (by decide)]

/-- The l2 kernel is launched with 256 threads but only threads `< 128` carry data; all data
    columns are inside the block. -/
theorem l2_threads_cover {col : Nat} (h : col < 128) : col < 256 := by omega

-- § gates

/-- Softplus with the overflow branch, abstract over the smooth part `f`. -/
def softplus (f : Int → Int) (x : Int) : Int := if x > 20 then x else f x

/-- Gate formula: `softplus(alpha + dt) * ssm_a`. -/
def gate (f : Int → Int) (alpha dt a : Int) : Int := softplus f (alpha + dt) * a

/-- Above the threshold the gate is linear in the pre-activation. -/
theorem gate_large {f : Int → Int} {alpha dt a : Int} (h : alpha + dt > 20) :
    gate f alpha dt a = (alpha + dt) * a := by
  simp [gate, softplus, h]

/-- With `ssm_a = 0` the gate is zero whatever the branch (no decay applied). -/
theorem gate_zero_a (f : Int → Int) (alpha dt : Int) : gate f alpha dt 0 = 0 := by
  simp [gate]

-- § validation

/-- `count_and_stream`: a non-null stream and `1 ≤ count ≤ 65535`. -/
def countOk (stream : Bool) (count : Int) : Bool := stream && 0 < count && count ≤ 65535

/-- A passing count fits the grid's x dimension and the 256-thread blocks cover it. -/
theorem countOk_grid {s : Bool} {n : Int} (h : countOk s n = true) :
    1 ≤ n ∧ n ≤ 65535 ∧ s = true := by
  unfold countOk at h
  simp only [Bool.and_eq_true, decide_eq_true_eq] at h
  obtain ⟨⟨hs, h1⟩, h2⟩ := h
  exact ⟨by omega, h2, hs⟩

/-- `norm_geometry`: width exactly 128 and non-negative epsilon (finiteness abstracted). -/
def normOk (stream : Bool) (rows cols : Int) (epsNonneg : Bool) : Bool :=
  countOk stream rows && cols == 128 && epsNonneg

/-- A passing norm call has exactly 128 columns. -/
theorem normOk_width {s : Bool} {r c : Int} {e : Bool} (h : normOk s r c e = true) : c = 128 := by
  unfold normOk at h
  simp only [Bool.and_eq_true, beq_iff_eq] at h
  exact h.1.2

/-- Half-open span disjointness used by `disjoint(a, b)`. -/
def disjoint (a an b bn : Nat) : Bool := !(a < b + bn && b < a + an)

/-- Disjointness is symmetric. -/
theorem disjoint_symm (a an b bn : Nat) : disjoint a an b bn = disjoint b bn a an := by
  simp [disjoint, Bool.and_comm]

/-- Back-to-back buffers are disjoint. -/
theorem disjoint_adjacent (a an bn : Nat) : disjoint a an (a + an) bn = true := by
  simp [disjoint]

/-- The conv checks every ordered pair `j < i` of its three writable spans: that covers all
    three unordered pairs. -/
theorem conv_pairs_checked :
    ((List.range 3).flatMap (fun i => (List.range i).map (fun j => (i, j)))) =
      [(1, 0), (2, 0), (2, 1)] := by decide

end Ref.NativeGdnPre


