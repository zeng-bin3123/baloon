/-!
# MODULE: shared_expert_combine
Scratch carving and alignment for the shared expert, the per-token scalar gate scaling, and the MoE
combination (router-weighted routed parts plus a plain shared term).

## Source semantics
shared = silu(gate_proj x) * up_proj x, down-projected, scaled by one scalar gate per token. The MoE block
output is `sum_e w_e * part_e + shared` with the shared output added PLAIN (not router-weighted).
Scratch layout: gate | up | q8_0 | q8k | g, each 16-byte aligned.

## Form template identifiers
- `align16` ↔ `(x + 15) & ~15`; `scratchBytes` ↔ `shared_expert_scratch_bytes`
- `combine` ↔ `moe_combine_kernel`; `scaleRow` ↔ `out[i] *= g`
-/

namespace Strata.SharedExpertCombine

-- § alignment and scratch

/-- Round up to a multiple of 16. -/
def align16 (n : Nat) : Nat := (n + 15) / 16 * 16

/-- Alignment never shrinks. -/
theorem align16_ge (n : Nat) : n ≤ align16 n := by
  unfold align16; omega

/-- Result is a multiple of 16. -/
theorem align16_mod (n : Nat) : align16 n % 16 = 0 := by
  unfold align16; omega

/-- Alignment adds fewer than 16 bytes. -/
theorem align16_lt (n : Nat) : align16 n < n + 16 := by
  unfold align16; omega

/-- Alignment is idempotent. -/
theorem align16_idem (n : Nat) : align16 (align16 n) = align16 n := by
  unfold align16; omega

/-- Total scratch bytes for a given `n_ff`. -/
def scratchBytes (nFf : Nat) : Nat :=
  align16 (nFf * 4) * 2 + align16 (nFf / 32 * 34) + align16 (nFf / 256 * 292) + 32

/-- Offsets of the carved regions, in order: gate, up, q8_0, q8k, g. -/
def offsets (nFf : Nat) : Nat × Nat × Nat × Nat × Nat :=
  let a := align16 (nFf * 4)
  let q0 := align16 (nFf / 32 * 34)
  let qk := align16 (nFf / 256 * 292)
  (0, a, a * 2, a * 2 + q0, a * 2 + q0 + qk)

/-- Every region offset is 16-byte aligned. -/
theorem offsets_aligned (nFf : Nat) :
    (offsets nFf).2.1 % 16 = 0 ∧ (offsets nFf).2.2.1 % 16 = 0 ∧
    (offsets nFf).2.2.2.1 % 16 = 0 ∧ (offsets nFf).2.2.2.2 % 16 = 0 := by
  unfold offsets
  have h1 := align16_mod (nFf * 4)
  have h2 := align16_mod (nFf / 32 * 34)
  have h3 := align16_mod (nFf / 256 * 292)
  simp only
  omega

/-- The scalar gate slot ends inside the scratch buffer (room for 32 bytes after the q8k region). -/
theorem gate_slot_fits (nFf : Nat) : (offsets nFf).2.2.2.2 + 4 ≤ scratchBytes nFf := by
  unfold offsets scratchBytes
  simp only
  omega

/-- Region ordering: gate < up < q8_0 offset when `n_ff > 0`. -/
theorem offsets_ordered (nFf : Nat) (h : 0 < nFf) :
    (offsets nFf).2.1 < (offsets nFf).2.2.1 := by
  unfold offsets
  have h1 := align16_ge (nFf * 4)
  simp only
  omega

/-- For the Q2 expert width 640 the intermediate Q8_0 region is 680 bytes: 20 blocks of 34. -/
theorem q8_0_region_640 : 640 / 32 * 34 = 680 := by decide

-- § combination

/-- Router-weighted sum `Σ w_e * p_e` (integer model). -/
def weightedSum : List Int → List Int → Int
  | w :: ws, p :: ps => w * p + weightedSum ws ps
  | _, _ => 0

/-- MoE combine: routed weighted sum plus the shared output if present (added plain). -/
def combine (ws ps : List Int) (shared : Option Int) : Int :=
  weightedSum ws ps + (match shared with | some s => s | none => 0)

/-- Without a shared expert the output is exactly the routed sum. -/
theorem combine_no_shared (ws ps : List Int) : combine ws ps none = weightedSum ws ps := by
  simp [combine]

/-- The shared term enters with coefficient exactly one: no router weight, no renormalization. -/
theorem combine_shared_plain (ws ps : List Int) (s : Int) :
    combine ws ps (some s) = weightedSum ws ps + s := by
  simp [combine]

/-- Combining is additive in the shared term. -/
theorem combine_shared_shift (ws ps : List Int) (s t : Int) :
    combine ws ps (some (s + t)) = combine ws ps (some s) + t := by
  simp only [combine]; omega

/-- The weighted sum of an empty selection is zero. -/
theorem weightedSum_nil : weightedSum [] [] = 0 := by
  simp [weightedSum]

/-- Scaling a weighted sum by a constant scales every term (distributivity of the gate). -/
theorem weightedSum_scale (g : Int) :
    ∀ ws ps : List Int, weightedSum (ws.map (g * ·)) ps = g * weightedSum ws ps := by
  intro ws
  induction ws with
  | nil => intro ps; simp [weightedSum]
  | cons w ws ih =>
    intro ps
    cases ps with
    | nil => simp [weightedSum]
    | cons p ps =>
      simp only [List.map_cons, weightedSum, ih]
      rw [Int.mul_add, Int.mul_assoc]

-- § scalar gate scaling

/-- Elementwise scaling of a row by the token's gate scalar. -/
def scaleRow (g : Int) (row : List Int) : List Int := row.map (g * ·)

/-- Scaling preserves length. -/
theorem scaleRow_length (g : Int) (row : List Int) : (scaleRow g row).length = row.length := by
  simp [scaleRow]

/-- A unit gate leaves the row unchanged. -/
theorem scaleRow_one (row : List Int) : scaleRow 1 row = row := by
  simp [scaleRow]

/-- Row-major index of (token t, column i) in the multi-token scaling kernel stays in bounds. -/
theorem row_index_lt (t i nTok nEmbd : Nat) (ht : t < nTok) (hi : i < nEmbd) :
    t * nEmbd + i < nTok * nEmbd := by
  have : (t + 1) * nEmbd ≤ nTok * nEmbd := Nat.mul_le_mul_right _ ht
  rw [Nat.succ_mul] at this
  omega

/-- The multi-token path handles 1..8 tokens only. -/
def multiOk (nTok : Nat) : Prop := 1 ≤ nTok ∧ nTok ≤ 8

/-- Eight tokens is the largest accepted window. -/
theorem multiOk_max : multiOk 8 ∧ ¬ multiOk 9 := by
  unfold multiOk; decide

end Strata.SharedExpertCombine
