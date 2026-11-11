/-!
# MODULE: iquant_block_decode
SOURCES: src/kernels/cpu/iq_avx512.cpp, src/kernels/cpu/native_expert.cpp, src/kernels/cpu/q2_avx2.cpp
The i-quant AVX-512 row kernels: the five supported block formats (IQ2_XXS=16, IQ2_XS=17, IQ3_XXS=18,
IQ3_S=21, IQ2_S=22), their byte budgets, the in-bounds property of every field `Fmt<TY>::decode` reads
for each of the four 64-value chunks of a 256-value block, the odd scale encoding, the sign application,
and the dispatch rules (≥ 2 tokens go to the AVX-512 kernel; more than 4 tokens are split into groups of
at most 4 — the same split the AVX2 Q2_0 rows use).

## Source semantics
* A block holds QK_K = 256 values = 4 chunks of 64; the fp16 `d` is the first 2 bytes.
* Byte budgets: 66, 74, 98, 110, 82 for types 16, 17, 18, 21, 22.
* Scales are `2·s + 1` with s a 4-bit field: odd, 1..31.  Sign indices are 7-bit (< 128).
* Signs: a masked subtract replaces y by −y; `maddubs(g, ±y) = ±g·y`.
* `native_gu_rows` uses AVX-512 iff the CPU passes, the env does not opt out, nt ≥ 2 and the type is
  supported; otherwise ggml-cpu's `vec_dot` per token.

## Form template identifiers
blockBytes → formatBlockBytes, supported → iq512Supported, oddScale → nibbleScale, applySign → signedAct,
useAvx512 → iqDispatch, split4 → tokenGroups
-/

namespace Strata.Kernels.Cpu.IQ

-- § formats

/-- `iq512_supported`. -/
def supported (t : Nat) : Bool := t == 16 || t == 17 || t == 18 || t == 21 || t == 22

/-- Block bytes per format (`Fmt<TY>::bytes`). -/
def blockBytes : Nat → Nat
  | 16 => 66
  | 17 => 74
  | 18 => 98
  | 21 => 110
  | 22 => 82
  | _ => 0

/-- Exactly the five formats are supported. -/
theorem supported_iff (t : Nat) : supported t = true ↔ t = 16 ∨ t = 17 ∨ t = 18 ∨ t = 21 ∨ t = 22 := by
  simp [supported, beq_iff_eq, or_assoc]

/-- IQ1_M (29) and Q2_0 (42) are not AVX-512 i-quant formats. -/
theorem not_supported_examples : supported 29 = false ∧ supported 42 = false := by decide

/-- 4 chunks of 64 values make one 256-value super-block. -/
theorem chunks_per_block : 4 * 64 = 256 := by decide

/-- Byte budgets decompose into their fields (d, qs, extra high bits / signs / scales). -/
theorem budgets :
    blockBytes 16 = 2 + 64 ∧ blockBytes 17 = 2 + 64 + 8 ∧ blockBytes 22 = 2 + 64 + 8 + 8 ∧
      blockBytes 18 = 2 + 64 + 32 ∧ blockBytes 21 = 2 + 64 + 8 + 32 + 4 := by decide

-- § decode reads stay inside the block

/-- IQ2_XXS: chunk j reads qs bytes [2+16j, 2+16j+16). -/
theorem iq2xxs_in_bounds (j : Nat) (hj : j < 4) : 2 + 16 * j + 16 ≤ blockBytes 16 := by
  show 2 + 16 * j + 16 ≤ 66; omega

/-- IQ2_XS: qs [2+16j, +16) and scale bytes 66+2j, 66+2j+1. -/
theorem iq2xs_in_bounds (j : Nat) (hj : j < 4) :
    2 + 16 * j + 16 ≤ 66 ∧ 66 + 2 * j + 1 < blockBytes 17 := by
  show _ ∧ 66 + 2 * j + 1 < 74; omega

/-- IQ2_S: grid bytes, sign bytes, qh bytes and scale bytes of chunk j. -/
theorem iq2s_in_bounds (j : Nat) (hj : j < 4) :
    2 + 8 * j + 8 ≤ 34 ∧ 2 + 32 + 8 * j + 8 ≤ 66 ∧ 66 + 2 * j + 1 < 74 ∧
      74 + 2 * j + 1 < blockBytes 22 := by
  show _ ∧ _ ∧ _ ∧ 74 + 2 * j + 1 < 82; omega

/-- IQ3_XXS: grid bytes [2+16j, +16) and the two sign/scale words [66+8j, +8). -/
theorem iq3xxs_in_bounds (j : Nat) (hj : j < 4) :
    2 + 16 * j + 16 ≤ 66 ∧ 2 + 64 + 8 * j + 8 ≤ blockBytes 18 := by
  show _ ∧ 2 + 64 + 8 * j + 8 ≤ 98; omega

/-- IQ3_S: grid bytes, qh bytes, sign bytes [74+8j, +8) and scale byte 106+j. -/
theorem iq3s_in_bounds (j : Nat) (hj : j < 4) :
    2 + 16 * j + 16 ≤ 66 ∧ 66 + 2 * j + 1 < 74 ∧ 74 + 8 * j + 8 ≤ 106 ∧ 106 + j < blockBytes 21 := by
  show _ ∧ _ ∧ _ ∧ 106 + j < 110; omega

/-- IQ3_S grid index `q | ((h >> k) & 1) << 8` is a 9-bit index (< 512). -/
theorem iq3s_grid_index (q h k : Nat) (hq : q < 256) : q + ((h >>> k) % 2) * 256 < 512 := by
  have : (h >>> k) % 2 < 2 := Nat.mod_lt _ (by decide)
  omega

/-- IQ2_XS grid index `v & 511` and sign index `v >> 9` of a u16 are in their tables. -/
theorem iq2xs_indices (v : Nat) (hv : v < 65536) : v % 512 < 512 ∧ v / 512 < 128 := by omega

-- § scales and signs

/-- The scale encoding `2·s + 1` of a nibble. -/
def oddScale (s : Nat) : Nat := 2 * (s % 16) + 1

/-- Scales are odd and between 1 and 31: they fit an int16 `madd` operand trivially. -/
theorem oddScale_range (s : Nat) : 1 ≤ oddScale s ∧ oddScale s ≤ 31 ∧ oddScale s % 2 = 1 := by
  unfold oddScale; omega

/-- 7-bit sign-table indices `(a >> 7k) & 127` are always in the 128-entry table. -/
theorem sign_index_lt (a k : Nat) : (a >>> (7 * k)) % 128 < 128 := Nat.mod_lt _ (by decide)

/-- The masked subtract: a set sign bit negates the activation. -/
def applySign (m : Bool) (y : Int) : Int := if m then 0 - y else y

/-- `maddubs(g, ±y) = ±(g·y)`: the sign moves from the weight onto the product. -/
theorem sign_product (g y : Int) (m : Bool) :
    g * applySign m y = if m then -(g * y) else g * y := by
  cases m
  · rfl
  · simp only [applySign, ite_true, Int.zero_sub, Int.mul_neg]

-- § dispatch

/-- `native_gu_rows`: AVX-512 for ≥ 2 tokens on a supported type, otherwise ggml-cpu. -/
def useAvx512 (cpuOk optOut : Bool) (nt type : Nat) : Bool :=
  cpuOk && !optOut && decide (2 ≤ nt) && supported type

/-- One token always goes to ggml-cpu (it is as fast or faster there). -/
theorem one_token_ggml (cpuOk optOut : Bool) (type : Nat) : useAvx512 cpuOk optOut 1 type = false := by
  simp [useAvx512]

/-- Token groups of at most 4 (`dot_rows_nt` default case and `q2_0_gguf_rows_multi_avx2`). -/
def split4 : Nat → List Nat
  | n => if n ≤ 4 then [n] else 4 :: split4 (n - 4)
termination_by n => n
decreasing_by omega

/-- The groups cover every token. -/
theorem split4_sum (n : Nat) : (split4 n).foldr (· + ·) 0 = n := by
  induction n using Nat.strongRecOn with
  | ind n ih =>
    rw [split4]
    split
    · simp
    · rename_i h
      simp only [List.foldr_cons]
      rw [ih (n - 4) (by omega)]
      omega

/-- Every group is a supported template width (≤ 4). -/
theorem split4_le (n g : Nat) (hg : g ∈ split4 n) : g ≤ 4 := by
  induction n using Nat.strongRecOn with
  | ind n ih =>
    rw [split4] at hg
    split at hg
    · rename_i h; simp at hg; omega
    · rcases List.mem_cons.mp hg with h' | h'
      · omega
      · exact ih (n - 4) (by omega) h'

/-- Eight tokens (MAXT) run as two groups of four. -/
theorem split4_eight : split4 8 = [4, 4] := by
  rw [split4]; simp only [show ¬ (8 ≤ 4) from by decide, ite_false]
  rw [split4]; simp

end Strata.Kernels.Cpu.IQ
