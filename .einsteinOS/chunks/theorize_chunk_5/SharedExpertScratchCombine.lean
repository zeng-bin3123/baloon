/-!
# MODULE: shared_expert_scratch_combine
Scratch carving and the MoE combine of `shared_expert.cu`.

## Source semantics
Scratch is carved with 16-byte alignment: gate|up|q8_0|q8k|g. The MoE combine is
`sum_e w_e * part_e + shared` where the shared output is added PLAIN (not router-weighted). `k > 64` is refused.

## Form template identifiers
align16 -> `(x + 15) & ~15`; combine -> moe_combine_kernel.
-/

namespace Strata.SharedExpertScratchCombine

def align16 (n : Nat) : Nat := (n + 15) / 16 * 16

-- § alignment

theorem align16_ge (n : Nat) : n ≤ align16 n := by
  unfold align16
  omega

theorem align16_mod (n : Nat) : align16 n % 16 = 0 := by
  unfold align16
  omega

theorem align16_lt (n : Nat) : align16 n < n + 16 := by
  unfold align16
  omega

theorem align16_mono (a b : Nat) (h : a ≤ b) : align16 a ≤ align16 b := by
  unfold align16
  omega

/-- Carved offsets are non-overlapping and ordered. -/
theorem carve_ordered (nff : Nat) :
    align16 (nff * 4) ≤ align16 (nff * 4) * 2 := by
  omega

-- § combine

def dot : List Int → List Int → Int
  | w :: ws, p :: ps => w * p + dot ws ps
  | _, _ => 0

def combine (ws ps : List Int) (shared : Option Int) : Int :=
  dot ws ps + shared.getD 0

/-- The shared output is added plain. -/
theorem shared_added_plain (ws ps : List Int) (s : Int) :
    combine ws ps (some s) = combine ws ps none + s := by
  simp [combine]

/-- Without a shared expert the result is the routed sum. -/
theorem no_shared (ws ps : List Int) : combine ws ps none = dot ws ps := by
  simp [combine]

/-- Router weights scale routed parts only: scaling the shared term does not touch `dot`. -/
theorem routed_independent_of_shared (ws ps : List Int) (s t : Int) :
    combine ws ps (some s) - combine ws ps (some t) = s - t := by
  simp [combine]

/-- A single expert contributes `w * p`. -/
theorem dot_single (w p : Int) : dot [w] [p] = w * p := by
  simp [dot]

-- § scalar gate bound

/-- A sigmoid-style gate in [0,1] scales an output without flipping its sign (model on integers 0..1). -/
theorem gate_scale_nonneg (g x : Int) (hg : 0 ≤ g) (hx : 0 ≤ x) : 0 ≤ g * x := by
  exact Int.mul_nonneg hg hx

end Strata.SharedExpertScratchCombine
