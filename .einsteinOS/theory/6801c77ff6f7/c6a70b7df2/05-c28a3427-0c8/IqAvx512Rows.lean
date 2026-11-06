/-!
# MODULE: iq_avx512_rows
SOURCES: src/kernels/cpu/iq_avx512.cpp
Layout facts for the i-quant AVX-512 row kernels (IQ2_XXS, IQ2_XS, IQ3_XXS, IQ3_S, IQ2_S): block byte sizes,
that every decode read of chunk `j < 4` stays inside its 256-value block and inside its own field, the sign
negation on int8 activations, the lane-to-scale mapping, and the batching of tokens in groups of four.

## Source semantics
* A 256-value block is decoded as four 64-value chunks `j = 0..3`; each chunk reads a fixed slice of the block.
* Block sizes (bytes): IQ2_XXS 66, IQ2_XS 74, IQ3_XXS 98, IQ3_S 110, IQ2_S 82. Type ids: 16, 17, 18, 21, 22.
* Signs are applied as a masked subtract `0 - y` on lanes whose mask bit is set; activations come from Q8_K
  and lie in [-127,127], so the negation cannot overflow int8.
* `scales4` assigns int16 lanes 0-7, 8-15, 16-23, 24-31 to scale 0, 1, 2, 3.
* `dot_rows_nt` handles 1..4 tokens directly and loops in groups of four otherwise.

## Form template identifiers
iq512_supported → vector_path_supported, blockBytes → quant_block_bytes, negIf → masked_sign_flip,
scaleLane → lane_scale_index, chunksK → token_groups.
-/

namespace Strata.Kernels.Cpu.IqAvx512Rows

-- § supported types

/-- Block byte size of each supported format (0 for unsupported ids). -/
def blockBytes (ty : Nat) : Nat :=
  if ty = 16 then 66 else if ty = 17 then 74 else if ty = 18 then 98
  else if ty = 21 then 110 else if ty = 22 then 82 else 0

/-- Whether the vector path handles the type id. -/
def iq512Supported (ty : Nat) : Bool :=
  ty == 16 || ty == 17 || ty == 18 || ty == 21 || ty == 22

/-- The supported ids are exactly 16, 17, 18, 21, 22. -/
theorem supported_iff (ty : Nat) :
    iq512Supported ty = true ↔ (ty = 16 ∨ ty = 17 ∨ ty = 18 ∨ ty = 21 ∨ ty = 22) := by
  simp [iq512Supported, or_assoc]

/-- Every supported type has a non-zero block size. -/
theorem supported_has_size (ty : Nat) (h : iq512Supported ty = true) : 0 < blockBytes ty := by
  rcases (supported_iff ty).mp h with rfl | rfl | rfl | rfl | rfl <;> decide

/-- IQ1_M (id 29) stays on ggml-cpu. -/
theorem iq1_m_unsupported : iq512Supported 29 = false := by
  decide

/-- Q2_0 (id 42) is not an i-quant and is not handled here. -/
theorem q2_0_unsupported : iq512Supported 42 = false := by
  decide

/-- The five block sizes. -/
theorem block_sizes :
    blockBytes 16 = 66 ∧ blockBytes 17 = 74 ∧ blockBytes 18 = 98 ∧ blockBytes 21 = 110 ∧ blockBytes 22 = 82 := by
  decide

/-- Four 64-value chunks make one 256-value block (QK_K). -/
theorem four_chunks_make_block : 4 * 64 = 256 := by
  decide

-- § the field layout of each format

/-- IQ2_XXS: scale (2) + 32 u16 = 64 bytes. -/
theorem iq2xxs_layout : 2 + 64 = blockBytes 16 := by
  decide

/-- IQ2_XS: scale (2) + 64 bytes of u16 indices + 8 scale bytes. -/
theorem iq2xs_layout : 2 + 64 + 8 = blockBytes 17 := by
  decide

/-- IQ3_XXS: scale (2) + 64 grid bytes + 32 bytes of sign/scale words. -/
theorem iq3xxs_layout : 2 + 64 + 32 = blockBytes 18 := by
  decide

/-- IQ3_S: scale (2) + 64 grid bytes + 8 qh + 32 signs + 4 scale bytes. -/
theorem iq3s_layout : 2 + 64 + 8 + 32 + 4 = blockBytes 21 := by
  decide

/-- IQ2_S: scale (2) + 32 grid bytes + 32 sign bytes + 8 qh + 8 scale bytes. -/
theorem iq2s_layout : 2 + 32 + 32 + 8 + 8 = blockBytes 22 := by
  decide

-- § every decode read stays inside the block

/-- IQ2_XXS: the 16-byte read at `2 + 16 j` ends inside the 66-byte block. -/
theorem iq2xxs_reads (j : Nat) (hj : j < 4) : 2 + 16 * j + 16 ≤ 66 := by
  omega

/-- IQ2_XS: the 16-byte index read and the two scale bytes at `66 + 2 j` stay inside 74 bytes, and do not overlap. -/
theorem iq2xs_reads (j : Nat) (hj : j < 4) :
    2 + 16 * j + 16 ≤ 66 ∧ 66 + 2 * j + 1 < 74 := by
  omega

/-- IQ2_S: grid bytes (8 at `2 + 8 j`), sign words (u64 at `34 + 8 j`), qh and scale bytes all fit in 82 bytes. -/
theorem iq2s_reads (j : Nat) (hj : j < 4) :
    2 + 8 * j + 8 ≤ 34 ∧ 34 + 8 * j + 8 ≤ 66 ∧ 66 + 2 * j + 1 < 74 ∧ 74 + 2 * j + 1 < 82 := by
  omega

/-- IQ3_XXS: the 16 grid bytes and the two u32 words (at `66 + 8 j` and `+4`) fit in 98 bytes. -/
theorem iq3xxs_reads (j : Nat) (hj : j < 4) :
    2 + 16 * j + 16 ≤ 66 ∧ 66 + 8 * j + 4 + 4 ≤ 98 := by
  omega

/-- IQ3_S: grid bytes, qh, the u64 sign word and the scale byte all fit in 110 bytes. -/
theorem iq3s_reads (j : Nat) (hj : j < 4) :
    2 + 16 * j + 16 ≤ 66 ∧ 66 + 2 * j + 1 < 74 ∧ 74 + 8 * j + 8 ≤ 106 ∧ 106 + j < 110 := by
  omega

/-- In IQ3_S the sign region ends exactly where the scale bytes begin: the regions do not overlap. -/
theorem iq3s_regions_disjoint : 74 + 32 = 106 := by
  decide

/-- In IQ2_S the sign region ends exactly where qh begins. -/
theorem iq2s_regions_disjoint : 34 + 32 = 66 := by
  decide

-- § sign application

/-- The masked subtract: negate the lane when its mask bit is set. -/
def negIf (b : Bool) (y : Int) : Int := if b then -y else y

/-- Negation keeps an int8 activation in [-127,127] (Q8_K never emits -128), so there is no overflow. -/
theorem negIf_range (b : Bool) (y : Int) (h1 : -127 ≤ y) (h2 : y ≤ 127) :
    -127 ≤ negIf b y ∧ negIf b y ≤ 127 := by
  unfold negIf
  split <;> omega

/-- Applying the sign twice restores the activation. -/
theorem negIf_involutive (b : Bool) (y : Int) : negIf b (negIf b y) = y := by
  unfold negIf
  cases b <;> simp

/-- An unset mask bit leaves the activation unchanged. -/
theorem negIf_false (y : Int) : negIf false y = y := by
  simp [negIf]

-- § lane to scale mapping

/-- The scale index assigned to int16 lane `l` by `scales4`: lanes 0-7 → 0, 8-15 → 1, 16-23 → 2, 24-31 → 3. -/
def scaleLane (l : Nat) : Nat := l / 8

/-- Every one of the 32 lanes maps to one of the four scales. -/
theorem scaleLane_lt (l : Nat) (hl : l < 32) : scaleLane l < 4 := by
  unfold scaleLane
  omega

/-- Each scale governs exactly a contiguous run of 8 lanes. -/
theorem scaleLane_run (l : Nat) (s : Nat) (hs : s < 4) :
    scaleLane l = s ↔ (8 * s ≤ l ∧ l < 8 * s + 8) := by
  unfold scaleLane
  omega

-- § token batching in groups of four

/-- Group sizes when walking `n` tokens in groups of at most `k` (fuel-bounded). -/
def chunksK (k : Nat) : Nat → Nat → List Nat
  | 0, _ => []
  | f + 1, n => if n = 0 then [] else min k n :: chunksK k f (n - min k n)

/-- The groups cover exactly `n` tokens. -/
theorem chunksK_sum (k : Nat) (f : Nat) :
    ∀ n : Nat, n ≤ k * f → (chunksK k f n).sum = n := by
  induction f with
  | zero =>
    intro n hn
    have h0 : n = 0 := by simpa using hn
    simp [chunksK, h0]
  | succ f ih =>
    intro n hn
    by_cases h0 : n = 0
    · rw [chunksK, if_pos h0]
      simp [h0]
    · rw [chunksK, if_neg h0, List.sum_cons]
      have e : k * (f + 1) = k * f + k := by rw [Nat.mul_add, Nat.mul_one]
      rw [e] at hn
      have := ih (n - min k n) (by omega)
      rw [this]
      omega

/-- Every group has at most `k` tokens. -/
theorem chunksK_le (k : Nat) (hk : 0 < k) (f : Nat) :
    ∀ n : Nat, ∀ x ∈ chunksK k f n, x ≤ k := by
  induction f with
  | zero =>
    intro n x hx
    simp [chunksK] at hx
  | succ f ih =>
    intro n x hx
    by_cases h0 : n = 0
    · rw [chunksK, if_pos h0] at hx
      simp at hx
    · rw [chunksK, if_neg h0] at hx
      rcases List.mem_cons.mp hx with h | h
      · rw [h]
        omega
      · exact ih _ x h

/-- Any number of tokens is processed in groups of four with fuel `n`, each token exactly once. -/
theorem token_groups_cover (n : Nat) : (chunksK 4 n n).sum = n := by
  apply chunksK_sum
  have := Nat.mul_le_mul_right n (show 1 ≤ 4 by decide)
  omega

end Strata.Kernels.Cpu.IqAvx512Rows
