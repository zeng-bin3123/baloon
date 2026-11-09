/-!
# MODULE: iq_avx512_rows
SOURCES: src/kernels/cpu/iq_avx512.cpp
The i-quant expert rows in 512-bit lanes for IQ2_XXS (16), IQ2_XS (17), IQ3_XXS (18), IQ3_S (21) and IQ2_S
(22): block layouts, decoded-field index ranges, 64-value chunk/scale lane maps, the multi-token chunking
loop, and the supported-type predicate.

## Source semantics
* Each format stores 256 values per block; the byte sizes are 66, 74, 98, 110 and 82, and every field the
  decoder touches for chunk `j < 4` lies inside its block and the regions tile the block exactly.
* A 32-bit word packs four 7-bit sign indices and a 4-bit scale: `4*7 + 4 = 32`, and the packing is lossless.
* Sign application is a masked negate: it preserves the magnitude of the activation.
* More than four tokens are processed in chunks of at most four; the chunk sizes sum to `nt`.
* Only the five listed type ids are handled here (IQ1_M stays on ggml-cpu).

## Form template identifiers
* `blockBytes`      : iq_block_bytes
* `word32_fields`   : sign_scale_word_layout
* `laneScale`       : scale_lane_map
* `chunks`          : token_chunking
* `supported`       : iq512_supported
-/

namespace Ref.IqAvx512

-- § block layouts

/-- Bytes per 256-value block for the supported format ids. -/
def blockBytes (ty : Nat) : Nat :=
  if ty = 16 then 66 else if ty = 17 then 74 else if ty = 18 then 98 else
  if ty = 21 then 110 else if ty = 22 then 82 else 0

/-- IQ2_XXS: `d` (2) + 32 u16 (64). -/
theorem layout_16 : 2 + 64 = blockBytes 16 := by
  decide

/-- IQ2_XS: `d` + 32 u16 + 8 scale bytes. -/
theorem layout_17 : 2 + 64 + 8 = blockBytes 17 := by
  decide

/-- IQ3_XXS: `d` + 64 grid bytes + 8 x u32. -/
theorem layout_18 : 2 + 64 + 32 = blockBytes 18 := by
  decide

/-- IQ3_S: `d` + qs 64 + qh 8 + signs 32 + scales 4. -/
theorem layout_21 : 2 + 64 + 8 + 32 + 4 = blockBytes 21 := by
  decide

/-- IQ2_S: `d` + qs 64 (32 grid + 32 sign) + qh 8 + scales 8. -/
theorem layout_22 : 2 + 64 + 8 + 8 = blockBytes 22 := by
  decide

/-- IQ2_XXS reads for chunk `j < 4` end inside the 66-byte block. -/
theorem iq2xxs_in_bounds (j : Nat) (hj : j < 4) : 2 + 16 * j + 15 < 66 := by
  omega

/-- IQ2_XS: codes `[2+16j, 2+16j+16)` and scales `66+2j+1` are inside the 74-byte block. -/
theorem iq2xs_in_bounds (j : Nat) (hj : j < 4) : 2 + 16 * j + 15 < 66 ∧ 66 + 2 * j + 1 < 74 := by
  refine ⟨?_, ?_⟩ <;> omega

/-- IQ2_S: grid bytes, sign bytes, `qh` and scales are inside the 82-byte block and the sign
bytes start where the grid bytes end. -/
theorem iq2s_in_bounds (j : Nat) (hj : j < 4) :
    2 + 8 * j + 7 < 34 ∧ 2 + 32 + 8 * j + 7 < 66 ∧ 66 + 2 * j + 1 < 74 ∧ 74 + 2 * j + 1 < 82 := by
  refine ⟨?_, ?_, ?_, ?_⟩ <;> omega

/-- IQ3_XXS: codes and the sign/scale words are inside the 98-byte block. -/
theorem iq3xxs_in_bounds (j : Nat) (hj : j < 4) : 2 + 16 * j + 15 < 66 ∧ 2 + 64 + 8 * j + 7 < 98 := by
  refine ⟨?_, ?_⟩ <;> omega

/-- IQ3_S: all four field reads are inside the 110-byte block. -/
theorem iq3s_in_bounds (j : Nat) (hj : j < 4) :
    2 + 16 * j + 15 < 66 ∧ 66 + 2 * j + 1 < 74 ∧ 74 + 8 * j + 7 < 106 ∧ 106 + j < 110 := by
  refine ⟨?_, ?_, ?_, ?_⟩ <;> omega

/-- Four 64-value chunks make a 256-value block, and a chunk read of `qs + 64*j` stays inside it. -/
theorem chunk_geometry : 4 * 64 = 256 := by
  decide

/-- The 64 int8 activations of chunk `j` are `[64j, 64j+64)`, below 256. -/
theorem chunk_act_in_block (j : Nat) (hj : j < 4) : 64 * j + 63 < 256 := by
  omega

-- § packed fields

/-- A 32-bit word of four 7-bit sign indices and a 4-bit scale is a lossless packing. -/
theorem word32_fields (a : Nat) :
    a % 128 + 128 * ((a / 128) % 128) + 16384 * ((a / 16384) % 128) +
      2097152 * ((a / 2097152) % 128) + 268435456 * (a / 268435456) = a := by
  omega

/-- The field widths add up to the word width. -/
theorem word32_width : 4 * 7 + 4 = 32 := by
  decide

/-- For a 32-bit word the scale nibble `a >> 28` is below 16. -/
theorem scale_nibble_lt (a : Nat) (h : a < 4294967296) : a / 268435456 < 16 := by
  omega

/-- The odd scale `2*s + 1` is odd and at most 31 for a nibble. -/
theorem odd_scale (s : Nat) (h : s < 16) : (2 * s + 1) % 2 = 1 ∧ 2 * s + 1 ≤ 31 := by
  refine ⟨?_, ?_⟩ <;> omega

/-- Each sign index is a 7-bit value, a valid index into the 128-entry sign table. -/
theorem sign_index_lt (a k : Nat) : (a / 2 ^ (7 * k)) % 128 < 128 :=
  Nat.mod_lt _ (by decide)

/-- Eight bytes pack into a 64-bit sign mask without overflow. -/
theorem mask64_pack (b0 b1 b2 b3 b4 b5 b6 b7 : Nat) (h0 : b0 < 256) (h1 : b1 < 256) (h2 : b2 < 256)
    (h3 : b3 < 256) (h4 : b4 < 256) (h5 : b5 < 256) (h6 : b6 < 256) (h7 : b7 < 256) :
    b0 + 256 * b1 + 65536 * b2 + 16777216 * b3 + 4294967296 * b4 + 1099511627776 * b5 +
      281474976710656 * b6 + 72057594037927936 * b7 < 18446744073709551616 := by
  omega

/-- The 9-bit grid index and 7-bit sign index of an IQ2_XS word are lossless: `v = idx + 512*sign`. -/
theorem iq2xs_word_split (v : Nat) (h : v < 65536) : v % 512 + 512 * (v / 512) = v ∧ v / 512 < 128 := by
  refine ⟨?_, ?_⟩ <;> omega

-- § lane maps

/-- Scale lane map: int16 lane `l < 32` of a `maddubs` result uses scale `l / 8`. -/
def laneScale (l : Nat) : Nat := l / 8

/-- Value `v < 64` sits in int16 lane `v/2`, whose scale index is `v/16`: four groups of 16 values. -/
theorem value_scale_group (v : Nat) (hv : v < 64) : laneScale (v / 2) = v / 16 ∧ v / 16 < 4 := by
  unfold laneScale
  refine ⟨?_, ?_⟩ <;> omega

/-- Sign application as a masked negate preserves magnitude. -/
theorem masked_negate_abs (y : Int) (b : Bool) : ((if b then -y else y)).natAbs = y.natAbs := by
  cases b <;> simp

-- § token chunking

/-- Sizes of the chunks (of at most `K`) a loop over `rem` tokens produces, with fuel `f`. -/
def chunks (K : Nat) : Nat → Nat → List Nat
  | 0, _ => []
  | f + 1, rem => if rem = 0 then [] else min K rem :: chunks K f (rem - min K rem)

/-- The chunk sizes sum to the token count when the fuel suffices. -/
theorem chunks_sum (K : Nat) (f : Nat) : ∀ rem, rem ≤ K * f → (chunks K f rem).sum = rem := by
  induction f with
  | zero =>
    intro rem h
    have : rem = 0 := by omega
    subst this
    rfl
  | succ f ih =>
    intro rem h
    rw [Nat.mul_succ] at h
    by_cases hr : rem = 0
    · subst hr
      simp [chunks]
    · have e : chunks K (f + 1) rem = min K rem :: chunks K f (rem - min K rem) := by
        simp [chunks, hr]
      rw [e, List.sum_cons, ih (rem - min K rem) (by omega)]
      omega

/-- Every chunk is at most `K`. -/
theorem chunks_le (K : Nat) (f : Nat) : ∀ rem, ∀ c ∈ chunks K f rem, c ≤ K := by
  induction f with
  | zero => intro rem c hc; simp [chunks] at hc
  | succ f ih =>
    intro rem c hc
    by_cases hr : rem = 0
    · subst hr
      simp [chunks] at hc
    · have e : chunks K (f + 1) rem = min K rem :: chunks K f (rem - min K rem) := by
        simp [chunks, hr]
      rw [e] at hc
      rcases List.mem_cons.mp hc with h | h
      · rw [h]; omega
      · exact ih _ c h

/-- A tail chunk `k = min 4 (nt - t0)` is non-empty and stays inside `nt`. -/
theorem tail_chunk_bounds (nt t0 : Nat) (h : t0 < nt) :
    1 ≤ min 4 (nt - t0) ∧ t0 + min 4 (nt - t0) ≤ nt := by
  refine ⟨?_, ?_⟩ <;> omega

-- § supported types

/-- The types handled by the 512-bit kernels. -/
def supported (ty : Nat) : Bool := ty == 16 || ty == 17 || ty == 18 || ty == 21 || ty == 22

/-- Exactly the five listed ids are supported. -/
theorem supported_iff (ty : Nat) : supported ty = true ↔ (ty = 16 ∨ ty = 17 ∨ ty = 18 ∨ ty = 21 ∨ ty = 22) := by
  simp [supported, or_assoc]

/-- Supported ids are exactly the ones with a known block size. -/
theorem supported_has_size (ty : Nat) (h : supported ty = true) : 0 < blockBytes ty := by
  rw [supported_iff] at h
  rcases h with h | h | h | h | h <;> subst h <;> decide

/-- Other ggml ids (IQ1_M and non-IQ types) are not claimed. -/
theorem unsupported_examples : [0, 1, 19, 20, 23, 29].all (fun t => !supported t) = true := by
  decide

end Ref.IqAvx512
