/-!
# MODULE: cpu_iq_avx512_rows
i-quant (IQ2_XXS, IQ2_XS, IQ3_XXS, IQ3_S, IQ2_S) expert rows in 512-bit lanes for several tokens at once: the
byte layout each `Fmt<TY>::decode` reads, the grid-index and scale ranges, the row/block addressing, and the
token-count batching of `dot_rows_nt` / `gu_rows_nt`.

## Source semantics
* Each 256-value block is decoded as four 64-value chunks (j = 0..3). Every byte `decode` touches lies inside
  the format's block (66 / 74 / 82 / 98 / 110 bytes).
* Scale values are `2*s + 1` for a 4-bit s, so they are odd and in [1, 31].
* Grid indices stay inside their lookup table (256, 512 or 1024 entries); sign indices are 7-bit.
* `dot_rows_nt`: 1..4 tokens directly, otherwise in groups of 4; `gu_rows_nt`: 1..7 directly, else NT = 8.
* A row of nb blocks occupies `nb * bytes`; block i ends before the row ends.
* Types 16, 17, 18, 21, 22 are supported, nothing else.

## Form template identifiers
* `blockBytes` : `Fmt<TY>::bytes`
* `supported` : `iq512_supported`
* `chunks 4 nt` : the `t0 += 4` loop of `dot_rows_nt`
* `guNT` : the `switch (nt)` of `gu_rows_nt`
-/

namespace Strata.IqAvx512

-- § block layouts

/-- `Fmt<TY>::bytes` for the supported ggml type ids. -/
def blockBytes : Nat → Nat
  | 16 => 66
  | 17 => 74
  | 18 => 98
  | 21 => 110
  | 22 => 82
  | _ => 0

/-- `iq512_supported`. -/
def supported (t : Nat) : Bool := t == 16 || t == 17 || t == 18 || t == 21 || t == 22

/-- A type is supported iff it has a block size. -/
theorem supported_iff_bytes (t : Nat) : supported t = true ↔ 0 < blockBytes t := by
  unfold supported blockBytes
  split <;> simp_all

/-- IQ2_XXS: 2-byte d + 32 u16 = 66 bytes. -/
theorem bytes_iq2xxs : blockBytes 16 = 2 + 64 := by decide
/-- IQ2_XS: d + 32 u16 + 8 scale bytes = 74 bytes. -/
theorem bytes_iq2xs : blockBytes 17 = 2 + 64 + 8 := by decide
/-- IQ2_S: d + 64 qs + 8 qh + 8 scales = 82 bytes. -/
theorem bytes_iq2s : blockBytes 22 = 2 + 64 + 8 + 8 := by decide
/-- IQ3_XXS: d + 64 grid bytes + 32 sign/scale bytes = 98 bytes. -/
theorem bytes_iq3xxs : blockBytes 18 = 2 + 64 + 32 := by decide
/-- IQ3_S: d + 64 qs + 8 qh + 32 signs + 4 scales = 110 bytes. -/
theorem bytes_iq3s : blockBytes 21 = 2 + 64 + 8 + 32 + 4 := by decide

/-- Four 64-value chunks make one 256-value block (QK_K). -/
theorem chunks_per_block : 4 * 64 = 256 := by decide

/-- IQ2_XXS reads 16 bytes at `2 + 16 j`; all four chunks stay inside the block. -/
theorem xxs_qs_in_block (j : Nat) (hj : j < 4) : 2 + 16 * j + 16 ≤ blockBytes 16 := by
  have : blockBytes 16 = 66 := by decide
  omega

/-- IQ2_XS: qs at `2 + 16 j`, scale bytes at `66 + 2 j (+1)`, all below 74. -/
theorem xs_in_block (j : Nat) (hj : j < 4) :
    2 + 16 * j + 16 ≤ 66 ∧ 66 + 2 * j + 1 < blockBytes 17 := by
  have : blockBytes 17 = 74 := by decide
  omega

/-- IQ2_S: grid bytes `2 + 8 j`, sign bytes `34 + 8 j`, qh `66 + 2 j (+1)`, scales `74 + 2 j (+1)`. -/
theorem iq2s_in_block (j : Nat) (hj : j < 4) :
    2 + 8 * j + 8 ≤ 34 ∧ 34 + 8 * j + 8 ≤ 66 ∧ 66 + 2 * j + 1 < 74 ∧ 74 + 2 * j + 1 < blockBytes 22 := by
  have : blockBytes 22 = 82 := by decide
  omega

/-- IQ3_XXS: qs at `2 + 16 j`, sign/scale words at `66 + 8 j .. + 8`, ending exactly at 98 for j = 3. -/
theorem xxs3_in_block (j : Nat) (hj : j < 4) :
    2 + 16 * j + 16 ≤ 66 ∧ 66 + 8 * j + 8 ≤ blockBytes 18 := by
  have : blockBytes 18 = 98 := by decide
  omega

/-- IQ3_S: qs `2 + 16 j`, qh `66 + 2 j (+1)`, signs `74 + 8 j .. + 8`, scale `106 + j`. -/
theorem iq3s_in_block (j : Nat) (hj : j < 4) :
    2 + 16 * j + 16 ≤ 66 ∧ 66 + 2 * j + 1 < 74 ∧ 74 + 8 * j + 8 ≤ 106 ∧ 106 + j < blockBytes 21 := by
  have : blockBytes 21 = 110 := by decide
  omega

-- § scales and indices

/-- A decoded scale `2 s + 1` from a 4-bit field s lies in [1, 31] and is odd. -/
theorem scale_range (s : Nat) (h : s < 16) : 1 ≤ 2 * s + 1 ∧ 2 * s + 1 ≤ 31 ∧ (2 * s + 1) % 2 = 1 := by
  omega

/-- `scales4` assigns int16 lane i (0..31) to scale index `i / 8`, always one of the four scales. -/
theorem scales4_index (i : Nat) (h : i < 32) : i / 8 < 4 := by omega

/-- A 9-bit grid index `v & 511` stays inside a 512-entry table. -/
theorem grid9_lt (v : Nat) : v &&& 511 < 512 := by
  have := Nat.and_le_right (n := v) (m := 511)
  omega

/-- An 8-bit grid index `x & 255` stays inside a 256-entry table. -/
theorem grid8_lt (x : Nat) : x &&& 255 < 256 := by
  have := Nat.and_le_right (n := x) (m := 255)
  omega

/-- A 7-bit sign index `(a >> k) & 127` stays inside the 128-entry `ksigns_iq2xs`. -/
theorem sign7_lt (a k : Nat) : (a >>> k) &&& 127 < 128 := by
  have := Nat.and_le_right (n := a >>> k) (m := 127)
  omega

/-- The IQ2_XS sign index `v >> 9` of a 16-bit v is a 7-bit value. -/
theorem iq2xs_sign_lt (v : Nat) (h : v < 65536) : v >>> 9 < 128 := by
  rw [Nat.shiftRight_eq_div_pow]
  omega

/-- The IQ2_S grid index `qs | ((h << k) & 0x300)` is below 1024 for a byte qs. -/
theorem iq2s_grid_lt (qs h k : Nat) (hq : qs < 256) : qs ||| ((h <<< k) &&& 0x300) < 1024 := by
  have h1 : (h <<< k) &&& 0x300 ≤ 0x300 := Nat.and_le_right
  have h2 : qs ||| ((h <<< k) &&& 0x300) < 2 ^ 10 := by
    apply Nat.or_lt_two_pow
    · omega
    · omega
  omega

-- § row addressing

/-- Block i of an nb-block row ends no later than the row. -/
theorem block_in_row (i nb bytes : Nat) (h : i < nb) : (i + 1) * bytes ≤ nb * bytes :=
  Nat.mul_le_mul_right bytes h

/-- A row of n values has `n / 256` blocks; for a multiple of 256 they cover it exactly. -/
theorem row_blocks (n : Nat) (h : n % 256 = 0) : (n / 256) * 256 = n := by omega

-- § token batching

/-- `gu_rows_nt`: 1..7 direct, everything else NT = 8. -/
def guNT (nt : Nat) : Nat := if 1 ≤ nt ∧ nt ≤ 7 then nt else 8

/-- For 1..8 tokens (MAXT = 8) the gate/up kernel has exactly nt lanes. -/
theorem guNT_exact (nt : Nat) (h1 : 1 ≤ nt) (h2 : nt ≤ 8) : guNT nt = nt := by
  unfold guNT
  split <;> omega

/-- Chunks of at most k, as in the `t0 += 4` loop. -/
def chunks (k n : Nat) : List Nat :=
  if h : n = 0 ∨ k = 0 then [] else min k n :: chunks k (n - min k n)
termination_by n
decreasing_by omega

/-- Every token is processed exactly once by the 4-token loop. -/
theorem chunks_sum (k : Nat) (hk : 0 < k) : ∀ n, (chunks k n).sum = n := by
  intro n
  induction n using Nat.strong_induction_on with
  | _ n ih =>
    by_cases h0 : n = 0
    · subst h0
      simp [chunks]
    · rw [chunks, dif_neg (by omega), List.sum_cons, ih (n - min k n) (by omega)]
      omega

/-- Each group has between 1 and 4 tokens, matching the instantiated NT = 1..4 templates. -/
theorem chunks_bound (k : Nat) (hk : 0 < k) : ∀ n, ∀ c ∈ chunks k n, 0 < c ∧ c ≤ k := by
  intro n
  induction n using Nat.strong_induction_on with
  | _ n ih =>
    intro c hc
    by_cases h0 : n = 0
    · subst h0
      simp [chunks] at hc
    · rw [chunks, dif_neg (by omega)] at hc
      rcases List.mem_cons.mp hc with rfl | hc
      · omega
      · exact ih (n - min k n) (by omega) c hc

end Strata.IqAvx512
