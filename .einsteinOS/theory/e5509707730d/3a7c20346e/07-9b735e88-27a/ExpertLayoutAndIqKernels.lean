/-!
# MODULE: expert_layout_and_iq_kernels
SOURCES: src/kernels/cpu/expert_layout.cpp, src/kernels/cpu/iq_avx512.cpp
The per-layer expert offset table (contiguity validation, max blob) and the byte layouts the AVX-512 i-quant
row kernels decode (IQ2_XXS, IQ2_XS, IQ3_XXS, IQ3_S, IQ2_S).

## Source semantics
* Layer l starts at the sum of (layer bytes * n_expert) over earlier layers; a line is accepted only if its
  offset equals that running total. With no native table every layer is BLOB bytes.
* `max_blob` is the maximum of layer sizes.
* iq512 supports exactly types 16,17,18,21,22; IQ1_M (29) stays on ggml-cpu.
* Each decode of chunk j<4 reads only inside its 256-value block; scale nibbles give odd scales 2s+1.

## Form template identifiers
layerOff -> layer_offset, chk -> contiguity_check, iqBytes -> iq_block_bytes, iqSupported -> iq512_supported
-/

namespace Strata.ExpertLayout

-- § offset table
/-- Byte offset of layer l: sum of earlier layer sizes times the expert count. -/
def layerOff (bytes : Nat → Nat) (ne : Nat) : Nat → Nat
  | 0 => 0
  | l + 1 => layerOff bytes ne l + bytes l * ne

/-- One layer's span is its blob size times the expert count. -/
theorem layerOff_succ (bytes : Nat → Nat) (ne l : Nat) :
    layerOff bytes ne (l + 1) = layerOff bytes ne l + bytes l * ne := rfl

/-- Offsets never decrease. -/
theorem layerOff_mono (bytes : Nat → Nat) (ne l : Nat) :
    layerOff bytes ne l ≤ layerOff bytes ne (l + 1) := Nat.le_add_right _ _

/-- With uniform blobs (no native table) the offset is l * (BLOB * ne). -/
theorem layerOff_uniform (b ne : Nat) : ∀ l : Nat, layerOff (fun _ => b) ne l = l * (b * ne)
  | 0 => by simp [layerOff]
  | l + 1 => by
      have ih := layerOff_uniform b ne l
      simp only [layerOff, ih]
      rw [Nat.add_mul, Nat.one_mul]

/-- Contiguity checker: each (offset, bytes) entry must start where the previous one ended. -/
def chk (ne : Nat) : List (Nat × Nat) → Nat → Bool
  | [], _ => true
  | (o, b) :: t, at_ => (o == at_) && chk ne t (at_ + b * ne)

/-- An empty table is accepted. -/
theorem chk_nil (ne at_ : Nat) : chk ne [] at_ = true := rfl

/-- A table is accepted iff its head starts at the running offset and the tail is accepted after it. -/
theorem chk_cons (ne o b : Nat) (t : List (Nat × Nat)) (at_ : Nat) :
    chk ne ((o, b) :: t) at_ = true ↔ o = at_ ∧ chk ne t (at_ + b * ne) = true := by
  simp [chk]

/-- Largest blob over a list of layer sizes. -/
def maxBlob (l : List Nat) : Nat := l.foldl max 0

/-- The running maximum never falls below its accumulator. -/
theorem le_foldl_max (acc : Nat) : ∀ l : List Nat, acc ≤ l.foldl max acc
  | [] => Nat.le_refl _
  | x :: l => by
      simp only [List.foldl]
      exact Nat.le_trans (Nat.le_max_left acc x) (le_foldl_max (max acc x) l)

/-- Every element is bounded by the running maximum. -/
theorem mem_le_foldl_max : ∀ (l : List Nat) (acc x : Nat), x ∈ l → x ≤ l.foldl max acc
  | [], _, _, h => by simp at h
  | y :: l, acc, x, h => by
      simp only [List.foldl]
      rcases List.mem_cons.mp h with rfl | h'
      · exact Nat.le_trans (Nat.le_max_right acc x) (le_foldl_max _ l)
      · exact mem_le_foldl_max l _ x h'

/-- `max_blob` bounds every layer's blob. -/
theorem maxBlob_ge (l : List Nat) (x : Nat) (h : x ∈ l) : x ≤ maxBlob l :=
  mem_le_foldl_max l 0 x h

-- § i-quant block layouts
/-- Bytes per 256-value block of each supported i-quant type (0 if unsupported). -/
def iqBytes : Nat → Nat
  | 16 => 66
  | 17 => 74
  | 18 => 98
  | 21 => 110
  | 22 => 82
  | _ => 0

/-- Types handled by the AVX-512 i-quant kernels. -/
def iqSupported (t : Nat) : Bool := t == 16 || t == 17 || t == 18 || t == 21 || t == 22

/-- Every supported type has a positive block size. -/
theorem iqSupported_bytes_pos (t : Nat) (h : iqSupported t = true) : 0 < iqBytes t := by
  simp only [iqSupported, Bool.or_eq_true, beq_iff_eq] at h
  rcases h with ((((h | h) | h) | h) | h) <;> subst h <;> decide

/-- IQ1_M (29) is not handled by the AVX-512 path. -/
theorem iq1m_unsupported : iqSupported 29 = false := by decide

/-- Q2_0 (42) is not an i-quant type. -/
theorem q2_0_not_iq : iqSupported 42 = false := by decide

/-- IQ2_XXS: d + 32 u16. -/
theorem iq2xxs_total : 2 + 32 * 2 = iqBytes 16 := by decide
/-- IQ2_XS: d + 32 u16 + 8 scale bytes. -/
theorem iq2xs_total : 2 + 32 * 2 + 8 = iqBytes 17 := by decide
/-- IQ3_XXS: d + 64 grid bytes + 32 sign/scale bytes. -/
theorem iq3xxs_total : 2 + 64 + 32 = iqBytes 18 := by decide
/-- IQ3_S: d + qs 64 + qh 8 + signs 32 + scales 4. -/
theorem iq3s_total : 2 + 64 + 8 + 32 + 4 = iqBytes 21 := by decide
/-- IQ2_S: d + qs 64 (grid+signs) + qh 8 + scales 8. -/
theorem iq2s_total : 2 + 64 + 8 + 8 = iqBytes 22 := by decide

-- § decode stays in bounds for chunks j < 4
/-- IQ2_XXS chunk reads 16 bytes inside the qs array. -/
theorem iq2xxs_chunk_in_bounds (j : Nat) (h : j < 4) : 2 + 16 * j + 16 ≤ 66 := by omega

/-- IQ2_XS chunk reads and its scale bytes stay in the block. -/
theorem iq2xs_in_bounds (j : Nat) (h : j < 4) :
    2 + 16 * j + 16 ≤ 66 ∧ 66 + 2 * j + 1 < 74 := by omega

/-- IQ2_S grid bytes, sign words, qh bytes and scale bytes are disjoint and in bounds. -/
theorem iq2s_in_bounds (j : Nat) (h : j < 4) :
    2 + 8 * j + 8 ≤ 34 ∧ 34 + 8 * j + 8 ≤ 66 ∧ 66 + 2 * j + 1 < 74 ∧ 74 + 2 * j + 1 < 82 := by
  omega

/-- IQ3_XXS grid bytes and sign/scale words stay in the block. -/
theorem iq3xxs_in_bounds (j : Nat) (h : j < 4) :
    2 + 16 * j + 16 ≤ 66 ∧ 2 + 64 + 8 * j + 8 ≤ 98 := by omega

/-- IQ3_S qs, qh, signs and scale reads stay in the block. -/
theorem iq3s_in_bounds (j : Nat) (h : j < 4) :
    2 + 16 * j + 16 ≤ 66 ∧ 66 + 2 * j + 1 < 74 ∧ 74 + 8 * j + 8 ≤ 106 ∧ 106 + j < 110 := by
  omega

-- § scales
/-- A 4-bit scale field s gives the odd scale 2s+1 in [1,31]. -/
theorem scale_odd (s : Nat) (h : s < 16) : 1 ≤ 2 * s + 1 ∧ 2 * s + 1 ≤ 31 ∧ (2 * s + 1) % 2 = 1 := by
  omega

/-- The scale permute maps each of the 32 int16 lanes to one of the four scales. -/
theorem scales4_lane (i : Nat) (h : i < 32) : i / 8 < 4 := by omega

end Strata.ExpertLayout
