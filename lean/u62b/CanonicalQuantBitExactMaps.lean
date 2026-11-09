/-!
# MODULE: canonical_quant_bit_exact_maps
SOURCES: tools/canonical_xcheck.py
Integer-level model of the GGML-quant -> canonical-form (S2/S4/S8/P16/P32) mappings. The Python tool checks the
float decode bit-exactly against gguf-py; this theory captures the integer / layout contracts that make that
possible: block byte decompositions, code ranges, integer biases, nibble and bit-plane splits, mask walks.

## Source semantics
Each mapping is (block_bytes, block_elems, group_elems, code_bias, has_offset). The decode is
`(code + bias) * scale (+ offset)`, the bias applied to the INTEGER code before the single multiply.
Codes are unsigned; Q2_0 `-1`, Q3_K `-4`, Q4_0 `-8`, Q5_0 `-16`, Q6_K `-32`, Q8_0 `-128` are integer biases.
IQ4_NL / IQ4_XS use a non-affine 16-entry codebook. Floating point rounding is NOT modelled here.

## Form template identifiers
mapping registry `MAPPINGS` -> `Mapping`; `to_s*`/`from_s*` -> pack/unpack; `KV_IQ4NL` -> `kv`;
`get_scale_min_k4` -> `scaleMinLow`/`scaleMinHigh`; `BLOCK_BYTES`/`QK` -> `Mapping.blockBytes`/`blockElems`.
-/

namespace RefCanonicalXcheck

-- § mapping registry

/-- One row of the MAPPINGS registry (integer part). -/
structure Mapping where
  blockBytes : Nat
  blockElems : Nat
  groupElems : Nat
  codeBias : Int
  hasOffset : Bool
  deriving DecidableEq, Repr

/-- Q2_0 mapping row. -/
def q2_0 : Mapping := ⟨18, 64, 64, -1, false⟩
/-- Q3_K mapping row. -/
def q3_k : Mapping := ⟨110, 256, 16, -4, false⟩
/-- Q4_0 mapping row. -/
def q4_0 : Mapping := ⟨18, 32, 32, -8, false⟩
/-- Q4_K mapping row. -/
def q4_k : Mapping := ⟨144, 256, 32, 0, true⟩
/-- Q5_0 mapping row. -/
def q5_0 : Mapping := ⟨22, 32, 32, -16, false⟩
/-- Q5_K mapping row. -/
def q5_k : Mapping := ⟨176, 256, 32, 0, true⟩
/-- Q6_K mapping row. -/
def q6_k : Mapping := ⟨210, 256, 16, -32, false⟩
/-- Q8_0 mapping row. -/
def q8_0 : Mapping := ⟨34, 32, 32, -128, false⟩
/-- IQ4_NL mapping row. -/
def iq4_nl : Mapping := ⟨18, 32, 32, 0, false⟩
/-- IQ4_XS mapping row. -/
def iq4_xs : Mapping := ⟨136, 256, 32, 0, false⟩

/-- All integer-quant rows of the registry. -/
def allMappings : List Mapping := [q2_0, q3_k, q4_0, q4_k, q5_0, q5_k, q6_k, q8_0, iq4_nl, iq4_xs]

/-- Every group size divides its block size. -/
theorem groups_divide_blocks : ∀ m ∈ allMappings, m.blockElems % m.groupElems = 0 := by decide

/-- Q2_0 block = 2-byte fp16 scale + 64/4 bytes of 2-bit codes. -/
theorem q2_0_layout : q2_0.blockBytes = 2 + 64 / 4 := by decide
/-- Q3_K block = hmask 32 + qs 64 + scales 12 + d 2. -/
theorem q3_k_layout : q3_k.blockBytes = 32 + 64 + 12 + 2 := by decide
/-- Q4_K block = d 2 + dmin 2 + scales 12 + qs 128. -/
theorem q4_k_layout : q4_k.blockBytes = 2 + 2 + 12 + 128 := by decide
/-- Q5_K block = d 2 + dmin 2 + scales 12 + qh 32 + qs 128. -/
theorem q5_k_layout : q5_k.blockBytes = 2 + 2 + 12 + 32 + 128 := by decide
/-- Q6_K block = ql 128 + qh 64 + scales 16 + d 2. -/
theorem q6_k_layout : q6_k.blockBytes = 128 + 64 + 16 + 2 := by decide
/-- Q5_0 block = d 2 + qh 4 + qs 16. -/
theorem q5_0_layout : q5_0.blockBytes = 2 + 4 + 16 := by decide
/-- Q8_0 block = d 2 + 32 int8 codes. -/
theorem q8_0_layout : q8_0.blockBytes = 2 + 32 := by decide
/-- IQ4_XS block = d 2 + scales_h 2 + scales_l 4 + qs 128. -/
theorem iq4_xs_layout : iq4_xs.blockBytes = 2 + 2 + 4 + 128 := by decide

/-- Rows with an additive offset are exactly Q4_K and Q5_K. -/
theorem offset_rows : (allMappings.filter (fun m => m.hasOffset)) = [q4_k, q5_k] := by decide

-- § integer-bias decode

/-- Integer decode: the bias is applied to the integer code BEFORE the single multiply. -/
def decodeInt (c : Nat) (bias : Int) (scale : Int) : Int := ((c : Int) + bias) * scale

/-- Q2_0 codebook is {-1,0,1,2}, not {0,1,2,3}. -/
theorem q2_0_value_set : ∀ c : Fin 4, ((c.val : Int) + q2_0.codeBias) ∈ [-1, 0, 1, 2] := by decide

/-- Decoding Q2_0 codes as unsigned would be wrong at every code. -/
theorem q2_0_bias_matters : ∀ c : Fin 4, ((c.val : Int) + q2_0.codeBias) ≠ c.val := by decide

/-- With unit scale the decode is exactly the biased code. -/
theorem decodeInt_unit (c : Nat) (b : Int) : decodeInt c b 1 = (c : Int) + b := by
  unfold decodeInt; omega

/-- Q4_0 codes 0..15 decode to -8..7. -/
theorem q4_0_range (c : Nat) (h : c < 16) : -8 ≤ (c : Int) + q4_0.codeBias ∧ (c : Int) + q4_0.codeBias ≤ 7 := by
  have : q4_0.codeBias = -8 := rfl
  rw [this]; omega

/-- Q8_0: an int8 stored +128 is an unsigned byte and decodes back to the same int8. -/
theorem q8_0_bias_roundtrip (s : Int) (h1 : -128 ≤ s) (h2 : s ≤ 127) :
    ∃ c : Nat, c < 256 ∧ (c : Int) + q8_0.codeBias = s := by
  have : q8_0.codeBias = -128 := rfl
  rw [this]
  exact ⟨(s + 128).toNat, by omega, by omega⟩

/-- Q5_0 codes 0..31 decode to -16..15. -/
theorem q5_0_range (c : Nat) (h : c < 32) : -16 ≤ (c : Int) + q5_0.codeBias ∧ (c : Int) + q5_0.codeBias ≤ 15 := by
  have : q5_0.codeBias = -16 := rfl
  rw [this]; omega

-- § nibble splits (Q4_0, IQ4_NL, IQ4_XS, Q4_K, Q5_K)

/-- A byte splits into low nibble (first half of block) and high nibble (second half) and recombines. -/
theorem nibble_split (b : Nat) (h : b < 256) : b % 16 < 16 ∧ b / 16 < 16 ∧ b % 16 + 16 * (b / 16) = b := by
  omega

/-- Q4_0: element j+16 comes from the high nibble of byte j, so the halves are not interleaved. -/
theorem q4_0_halves (j : Nat) (hj : j < 16) : j < 32 ∧ j + 16 < 32 ∧ j ≠ j + 16 := by omega

-- § Q2_0 bit plane

/-- Byte index and shift of element j of a Q2_0 block. -/
theorem q2_0_pos (j : Nat) (hj : j < 64) : j / 4 < 16 ∧ (j % 4) * 2 < 8 ∧ 4 * (j / 4) + j % 4 = j := by
  omega

/-- Packing four 2-bit codes in one byte and reading them back (LSB first). -/
theorem q2_0_pack_roundtrip (c0 c1 c2 c3 : Nat) (h0 : c0 < 4) (h1 : c1 < 4) (h2 : c2 < 4) (h3 : c3 < 4) :
    (c0 + 4 * c1 + 16 * c2 + 64 * c3) < 256 ∧
    (c0 + 4 * c1 + 16 * c2 + 64 * c3) % 4 = c0 ∧
    (c0 + 4 * c1 + 16 * c2 + 64 * c3) / 4 % 4 = c1 ∧
    (c0 + 4 * c1 + 16 * c2 + 64 * c3) / 16 % 4 = c2 ∧
    (c0 + 4 * c1 + 16 * c2 + 64 * c3) / 64 % 4 = c3 := by
  omega

-- § Q3_K

/-- The Q3_K 6-bit scale: low nibble plus two high bits is below 64. -/
theorem q3_k_scale_bound (lo t : Nat) (hl : lo < 256) (ht : t < 256) : lo % 16 + 16 * (t % 4) < 64 := by
  omega

/-- Stored Q3_K code = 2 bits + 4*hmask-bit stays in 0..7 and decodes to -4..3. -/
theorem q3_k_code_range (c2 m : Nat) (hc : c2 < 4) (hm : m ≤ 1) :
    c2 + 4 * m < 8 ∧ -4 ≤ ((c2 + 4 * m : Nat) : Int) + q3_k.codeBias ∧
    ((c2 + 4 * m : Nat) : Int) + q3_k.codeBias ≤ 3 := by
  have : q3_k.codeBias = -4 := rfl
  rw [this]; omega

/-- The hmask bit `4*half + j` never resets between halves: all 8 (half, j) pairs hit distinct bits < 8. -/
theorem q3_k_mask_bit_injective (h1 j1 h2 j2 : Nat) (a : h1 < 2) (b : j1 < 4) (c : h2 < 2) (d : j2 < 4)
    (e : 4 * h1 + j1 = 4 * h2 + j2) : h1 = h2 ∧ j1 = j2 := by
  omega

/-- Q3_K output group index `8*half + 2*j + sub` is below 16 and injective. -/
theorem q3_k_group_injective (h1 j1 s1 h2 j2 s2 : Nat) (a : h1 < 2) (b : j1 < 4) (c : s1 < 2)
    (d : h2 < 2) (e : j2 < 4) (f : s2 < 2) (g : 8 * h1 + 2 * j1 + s1 = 8 * h2 + 2 * j2 + s2) :
    8 * h1 + 2 * j1 + s1 < 16 ∧ h1 = h2 ∧ j1 = j2 ∧ s1 = s2 := by
  omega

-- § get_scale_min_k4

/-- Scale of group j < 4: low six bits of byte j. -/
def scaleMinLow (s : Nat) : Nat := s % 64

/-- Scale of group j >= 4: low nibble of s[j+4] and top two bits of s[j-4] shifted by 4. -/
def scaleMinHigh (sj4 sjm4 : Nat) : Nat := sj4 % 16 + 16 * (sjm4 / 64)

/-- Both scale forms are 6-bit values. -/
theorem scale_min_k4_bound (a b : Nat) (ha : a < 256) (hb : b < 256) :
    scaleMinLow a < 64 ∧ scaleMinHigh a b < 64 := by
  unfold scaleMinLow scaleMinHigh; omega

-- § Q5_K

/-- The Q5_K qh masks u1 for the four 64-element steps. -/
theorem q5_k_u1 : (List.range 4).map (fun j => 1 <<< (2 * j)) = [1, 4, 16, 64] := by decide
/-- The Q5_K qh masks u2 for the four 64-element steps. -/
theorem q5_k_u2 : (List.range 4).map (fun j => 2 <<< (2 * j)) = [2, 8, 32, 128] := by decide
/-- u1 and u2 together cover every one of the 8 bits of qh exactly once. -/
theorem q5_k_masks_cover : [1, 4, 16, 64, 2, 8, 32, 128].foldl (· + ·) 0 = 255 := by decide

/-- A Q5_K code is a nibble plus 16 times a bit, so below 32. -/
theorem q5_k_code_bound (n b : Nat) (hn : n < 16) (hb : b ≤ 1) : n + 16 * b < 32 := by omega

-- § Q6_K

/-- Q6_K code = nibble + 16 * two bits, below 64, decoding to -32..31. -/
theorem q6_k_code_range (n h : Nat) (hn : n < 16) (hh : h < 4) :
    n + 16 * h < 64 ∧ -32 ≤ ((n + 16 * h : Nat) : Int) + q6_k.codeBias ∧
    ((n + 16 * h : Nat) : Int) + q6_k.codeBias ≤ 31 := by
  have : q6_k.codeBias = -32 := rfl
  rw [this]; omega

/-- Output element index of quad q (0..3), lane l (0..31) in half h is 128h+32q+l, and its scale group is element/16. -/
theorem q6_k_scale_group (h q l : Nat) (hh : h < 2) (hq : q < 4) (hl : l < 32) :
    (128 * h + 32 * q + l) / 16 = 8 * h + 2 * q + l / 16 ∧ 128 * h + 32 * q + l < 256 := by
  omega

-- § IQ4

/-- The IQ4 non-linear codebook `kvalues_iq4nl`. -/
def kv : List Int := [-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113]

/-- Codebook lookup (out of range reads 0). -/
def kvAt (i : Nat) : Int := kv.getD i 0

/-- The codebook has exactly 16 entries (a 4-bit index). -/
theorem kv_length : kv.length = 16 := by decide

/-- The codebook is strictly increasing. -/
theorem kv_strictMono : ∀ i j : Fin 16, i.val < j.val → kvAt i.val < kvAt j.val := by decide

/-- The codebook is not affine: consecutive differences are not constant, so no integer bias can express it. -/
theorem kv_not_affine : kvAt 1 - kvAt 0 ≠ kvAt 2 - kvAt 1 := by decide

/-- The first eight entries are negative and the last eight positive. -/
theorem kv_sign_split : (∀ i : Fin 8, kvAt i.val < 0) ∧ (∀ i : Fin 8, 0 < kvAt (i.val + 8)) := by decide

/-- IQ4_XS 6-bit scale: nibble + 16 * two bits, giving ls - 32 in -32..31. -/
theorem iq4_xs_ls_range (n h : Nat) (hn : n < 16) (hh : h < 4) :
    n + 16 * h < 64 ∧ -32 ≤ ((n + 16 * h : Nat) : Int) - 32 ∧ ((n + 16 * h : Nat) : Int) - 32 ≤ 31 := by
  omega

/-- IQ4_XS group ib indexes scales_l byte ib/2 (below 4) and scales_h bits 2ib, 2ib+1 (below 16). -/
theorem iq4_xs_scale_index (ib : Nat) (h : ib < 8) : ib / 2 < 4 ∧ 2 * ib + 1 < 16 := by omega

-- § float forms

/-- bf16 widening: shifting the 16 bits left by 16 gives a value whose low 16 bits are zero, recoverable. -/
theorem bf16_widen_exact (u : Nat) (h : u < 65536) : (u * 65536) % 65536 = 0 ∧ (u * 65536) / 65536 = u := by
  omega

-- § full-run bookkeeping

/-- The split `idx % N == I` assigns every tensor index to exactly one process below N. -/
theorem split_in_range (idx n : Nat) (h : 0 < n) : idx % n < n := Nat.mod_lt _ h

/-- Streaming chunks hold whole blocks and never exceed the chunk size. -/
theorem chunk_whole_blocks (chunk bb : Nat) : (chunk / bb) * bb ≤ chunk := Nat.div_mul_le_self _ _

/-- Data section offset: file length minus the end of the last tensor places that end exactly at the file end. -/
theorem data_offset_exact (filelen endOfData : Nat) (h : endOfData ≤ filelen) :
    (filelen - endOfData) + endOfData = filelen := by omega

end RefCanonicalXcheck
