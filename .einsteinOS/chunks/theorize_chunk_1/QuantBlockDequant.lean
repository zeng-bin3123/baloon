/-!
# MODULE: quant_block_dequant
Integer-level model of the scalar ggml block dequantizers (Q2_0, Q4_0, Q5_0, Q8_0, IQ4_NL, Q6_K,
Q4_K scale/min unpacking, IQ4_XS scales). Floating-point scaling is abstracted: each function yields
the integer symbol that is multiplied by the block scale.

## Source semantics
* Q2_0: 4 codes per byte, LSB-first; code c ∈ {0..3} maps to symbol c - 1 ∈ {-1,0,1,2}.
* Q4_0: low nibble is element j, high nibble element j+16, symbol = nibble - 8.
* Q5_0: a fifth bit from a 32-bit mask extends the nibble; symbol = x - 16 ∈ [-16, 15].
* IQ4_NL: non-linear 16-entry codebook (strictly increasing, in [-127, 113]).
* Q6_K: 6-bit value (4 low + 2 high bits) minus 32 ∈ [-32, 31].
* get_scale_min_k4: unpacked scale and min are 6-bit values (< 64).
* IQ4_XS: group scale ls ∈ [0, 63], effective scale ls - 32 ∈ [-32, 31].

## Form template identifiers
* `q2Code`         ← inner index math of `dequantize_q2_0`
* `q2Symbol`       ← `code - 1`
* `q4Lo`/`q4Hi`    ← nibble extraction in `dequantize_q4_0`
* `q5Val`          ← 5-bit assembly in `dequantize_q5_0`
* `kvalues`        ← `kvalues_iq4nl`
* `q6Val`          ← `q1..q4` in `dequantize_q6_K`
* `scaleMinK4`     ← `get_scale_min_k4`
* `iq4xsScale`     ← `ls` in `dequantize_iq4_xs`
-/

namespace Strata.Dequant

-- § Q2_0

/-- Code `j` (0..63) of a Q2_0 block, given its 16 code bytes. -/
def q2Code (qs : Nat → Nat) (j : Nat) : Nat := (qs (j / 4) / 2 ^ ((j % 4) * 2)) % 4

/-- Symbol for a code: code - 1. -/
def q2Symbol (c : Nat) : Int := (c : Int) - 1

/-- Every Q2_0 code is in 0..3. -/
theorem q2Code_lt (qs : Nat → Nat) (j : Nat) : q2Code qs j < 4 := by
  unfold q2Code
  exact Nat.mod_lt _ (by decide)

/-- Every Q2_0 symbol is one of -1, 0, 1, 2. -/
theorem q2Symbol_range (qs : Nat → Nat) (j : Nat) :
    q2Symbol (q2Code qs j) = -1 ∨ q2Symbol (q2Code qs j) = 0 ∨ q2Symbol (q2Code qs j) = 1 ∨
    q2Symbol (q2Code qs j) = 2 := by
  have h := q2Code_lt qs j
  unfold q2Symbol
  omega

/-- Symbols are bounded below by -1. -/
theorem q2Symbol_ge (qs : Nat → Nat) (j : Nat) : -1 ≤ q2Symbol (q2Code qs j) := by
  have h := q2Code_lt qs j
  unfold q2Symbol
  omega

/-- Symbols are bounded above by 2. -/
theorem q2Symbol_le (qs : Nat → Nat) (j : Nat) : q2Symbol (q2Code qs j) ≤ 2 := by
  have h := q2Code_lt qs j
  unfold q2Symbol
  omega

/-- The bitwise form `(b >> s) & 3` equals the arithmetic form for every byte and legal shift. -/
theorem q2_bitwise_eq : ∀ b, b < 256 → ∀ k, k < 4 → (b >>> (k * 2)) &&& 3 = (b / 2 ^ (k * 2)) % 4 := by
  decide +kernel

/-- Position `j` lives in byte `j / 4` which is in range 0..15 for `j < 64`. -/
theorem q2_byte_index_lt (j : Nat) (h : j < 64) : j / 4 < 16 := by
  omega

/-- The bit offset of a Q2_0 element is one of 0, 2, 4, 6. -/
theorem q2_bit_offset_cases (j : Nat) :
    (j % 4) * 2 = 0 ∨ (j % 4) * 2 = 2 ∨ (j % 4) * 2 = 4 ∨ (j % 4) * 2 = 6 := by
  omega

/-- The four codes in one byte, reassembled, give back the byte. -/
theorem q2_pack_roundtrip : ∀ b, b < 256 →
    (b % 4) + 4 * ((b / 4) % 4) + 16 * ((b / 16) % 4) + 64 * ((b / 64) % 4) = b := by
  decide +kernel

/-- Distinct elements `4i .. 4i+3` of a byte index the same byte. -/
theorem q2_same_byte (i k : Nat) (hk : k < 4) : (4 * i + k) / 4 = i := by
  omega

/-- And their within-byte slot equals `k`. -/
theorem q2_slot (i k : Nat) (hk : k < 4) : (4 * i + k) % 4 = k := by
  omega

-- § Q4_0 / Q8_0 / Q5_0

/-- Low nibble of a byte. -/
def q4Lo (b : Nat) : Nat := b % 16

/-- High nibble of a byte. -/
def q4Hi (b : Nat) : Nat := b / 16

/-- Bridge: `b & 0x0F = b % 16` for all bytes. -/
theorem q4Lo_bitwise : ∀ b, b < 256 → b &&& 0x0F = q4Lo b := by
  decide +kernel

/-- Bridge: `b >> 4 = b / 16` for all bytes. -/
theorem q4Hi_bitwise : ∀ b, b < 256 → b >>> 4 = q4Hi b := by
  decide +kernel

/-- Nibbles recombine to the byte. -/
theorem q4_nibbles_recombine (b : Nat) (h : b < 256) : q4Lo b + 16 * q4Hi b = b := by
  unfold q4Lo q4Hi
  omega

/-- The high nibble of a byte is below 16. -/
theorem q4Hi_lt (b : Nat) (h : b < 256) : q4Hi b < 16 := by
  unfold q4Hi
  omega

/-- The low nibble is below 16. -/
theorem q4Lo_lt (b : Nat) : q4Lo b < 16 := by
  unfold q4Lo
  omega

/-- Q4_0 symbol: nibble - 8. -/
def q4Symbol (n : Nat) : Int := (n : Int) - 8

/-- Q4_0 symbols are in [-8, 7]. -/
theorem q4Symbol_range (n : Nat) (h : n < 16) : -8 ≤ q4Symbol n ∧ q4Symbol n ≤ 7 := by
  unfold q4Symbol
  omega

/-- Q4_0 element mapping: element j (< 16) is low nibble of byte j, element j+16 high nibble. -/
def q4Element (qs : Nat → Nat) (i : Nat) : Int :=
  if i < 16 then q4Symbol (q4Lo (qs i)) else q4Symbol (q4Hi (qs (i - 16)))

/-- Every Q4_0 output symbol lies in [-8, 7] for byte-valued input. -/
theorem q4Element_range (qs : Nat → Nat) (hq : ∀ i, qs i < 256) (i : Nat) :
    -8 ≤ q4Element qs i ∧ q4Element qs i ≤ 7 := by
  unfold q4Element
  split
  · exact q4Symbol_range _ (q4Lo_lt _)
  · exact q4Symbol_range _ (q4Hi_lt _ (hq _))

/-- Q8_0 symbol is the signed byte itself; a signed byte lies in [-128, 127]. -/
theorem q8_signed_byte_range (b : Nat) (h : b < 256) :
    -128 ≤ (if b < 128 then (b : Int) else (b : Int) - 256) ∧
    (if b < 128 then (b : Int) else (b : Int) - 256) ≤ 127 := by
  split <;> omega

/-- Q5_0: 5-bit value from a nibble and the high bit. -/
def q5Val (nib hbit : Nat) : Nat := nib + 16 * hbit

/-- Q5_0 symbol is `q5Val - 16`. -/
def q5Symbol (nib hbit : Nat) : Int := (q5Val nib hbit : Int) - 16

/-- The 5-bit value is below 32. -/
theorem q5Val_lt (nib hbit : Nat) (hn : nib < 16) (hh : hbit < 2) : q5Val nib hbit < 32 := by
  unfold q5Val
  omega

/-- Q5_0 symbols are in [-16, 15]. -/
theorem q5Symbol_range (nib hbit : Nat) (hn : nib < 16) (hh : hbit < 2) :
    -16 ≤ q5Symbol nib hbit ∧ q5Symbol nib hbit ≤ 15 := by
  unfold q5Symbol q5Val
  omega

/-- Extracting bit `j` of a 32-bit mask is 0 or 1. -/
theorem q5_hbit_lt (h j : Nat) : (h / 2 ^ j) % 2 < 2 := by
  exact Nat.mod_lt _ (by decide)

/-- Q5_0 element j and element j+16 use mask bits j and j+16, which are distinct. -/
theorem q5_mask_bits_distinct (j : Nat) : j ≠ j + 16 := by
  omega

-- § IQ4_NL codebook

/-- The IQ4_NL codebook, verbatim from ggml. -/
def kvalues : List Int :=
  [-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113]

/-- The codebook has 16 entries. -/
theorem kvalues_length : kvalues.length = 16 := by
  rfl

/-- The codebook is strictly increasing. -/
theorem kvalues_sorted : kvalues.Pairwise (· < ·) := by
  decide

/-- The codebook is non-linear: successive gaps are not constant (gap 23 then 21). -/
theorem kvalues_nonlinear : (-104 : Int) - (-127) ≠ (-83 : Int) - (-104) := by
  decide

/-- Every codebook value lies in [-127, 113]. -/
theorem kvalues_range : ∀ v ∈ kvalues, -127 ≤ v ∧ v ≤ 113 := by
  decide

/-- Lookup with default zero. -/
def kv (i : Nat) : Int := kvalues.getD i 0

/-- Every nibble indexes inside the table (lookup never falls back to the default except by value). -/
theorem kv_in_table (i : Nat) (h : i < 16) : kv i ∈ kvalues := by
  unfold kv
  have : i < kvalues.length := by rw [kvalues_length]; exact h
  rw [List.getD_eq_getElem _ _ this]
  exact List.getElem_mem this

/-- kv is within [-127, 113] for any nibble. -/
theorem kv_range (i : Nat) (h : i < 16) : -127 ≤ kv i ∧ kv i ≤ 113 :=
  kvalues_range _ (kv_in_table i h)

/-- Spot checks of the codebook endpoints. -/
theorem kv_endpoints : kv 0 = -127 ∧ kv 15 = 113 ∧ kv 8 = 1 := by
  decide

-- § Q6_K

/-- 6-bit value: low 4 bits from `lo`, high 2 bits from `hi`, minus 32. -/
def q6Val (lo hi : Nat) : Int := ((lo + 16 * hi : Nat) : Int) - 32

/-- Q6_K values lie in [-32, 31]. -/
theorem q6Val_range (lo hi : Nat) (hl : lo < 16) (hh : hi < 4) :
    -32 ≤ q6Val lo hi ∧ q6Val lo hi ≤ 31 := by
  unfold q6Val
  omega

/-- The two-bit field extracted from `qh` at shift 0, 2, 4, 6 is below 4. -/
theorem q6_hi_bits_lt (b k : Nat) : (b / 2 ^ (2 * k)) % 4 < 4 := by
  exact Nat.mod_lt _ (by decide)

/-- The scale index `is = l / 16` is 0 or 1 for `l < 32`, so `is + {0,2,4,6}` is below 8. -/
theorem q6_scale_index_lt (l : Nat) (hl : l < 32) (c : Nat) (hc : c = 0 ∨ c = 2 ∨ c = 4 ∨ c = 6) :
    l / 16 + c < 8 := by
  omega

/-- Over both 128-element halves, the scale base advances by 8 and indices stay below 16. -/
theorem q6_scale_index_lt16 (half l : Nat) (hh : half < 2) (hl : l < 32) (c : Nat)
    (hc : c = 0 ∨ c = 2 ∨ c = 4 ∨ c = 6) : 8 * half + (l / 16 + c) < 16 := by
  omega

/-- Output index `n + l + 32 k` stays inside the 256-element block. -/
theorem q6_out_index_lt (half l k : Nat) (hh : half < 2) (hl : l < 32) (hk : k < 4) :
    128 * half + l + 32 * k < 256 := by
  omega

/-- Distinct (half, l, k) triples give distinct output indices: the output map is injective. -/
theorem q6_out_index_inj (h1 l1 k1 h2 l2 k2 : Nat) (hh1 : h1 < 2) (hl1 : l1 < 32) (hk1 : k1 < 4)
    (hh2 : h2 < 2) (hl2 : l2 < 32) (hk2 : k2 < 4)
    (e : 128 * h1 + l1 + 32 * k1 = 128 * h2 + l2 + 32 * k2) :
    h1 = h2 ∧ l1 = l2 ∧ k1 = k2 := by
  omega

/-- The q6 layout byte budget: 128 + 64 + 16 + 2 = 210. -/
theorem q6_K_bytes : 128 + 64 + 16 + 2 = 210 := by
  rfl

/-- The d field of Q6_K sits at offset 208, the last two bytes of the 210-byte block. -/
theorem q6_K_d_offset : 128 + 64 + 16 = 208 ∧ 208 + 2 = 210 := by
  exact ⟨rfl, rfl⟩

-- § Q4_K / Q5_K scale-min unpacking

/-- Unpack scale and min for sub-block `j` (0..7) from the 12 scale bytes `q`. -/
def scaleMinK4 (q : Nat → Nat) (j : Nat) : Nat × Nat :=
  if j < 4 then (q j % 64, q (j + 4) % 64)
  else ((q (j + 4) % 16) + (q (j - 4) / 64) * 16, (q (j + 4) / 16) + (q j / 64) * 16)

/-- For `j < 4` the scale is a 6-bit value. -/
theorem scaleMinK4_lt64_low (q : Nat → Nat) (j : Nat) (hj : j < 4) :
    (scaleMinK4 q j).1 < 64 ∧ (scaleMinK4 q j).2 < 64 := by
  unfold scaleMinK4
  rw [if_pos hj]
  simp only
  omega

/-- For `4 ≤ j < 8` with byte-valued input both outputs are 6-bit values. -/
theorem scaleMinK4_lt64_high (q : Nat → Nat) (hq : ∀ i, q i < 256) (j : Nat) (hj : 4 ≤ j) :
    (scaleMinK4 q j).1 < 64 ∧ (scaleMinK4 q j).2 < 64 := by
  unfold scaleMinK4
  rw [if_neg (by omega)]
  have h1 := hq (j + 4)
  have h2 := hq (j - 4)
  have h3 := hq j
  simp only
  omega

/-- Sub-block scales are always 6 bits for byte input and j < 8. -/
theorem scaleMinK4_lt64 (q : Nat → Nat) (hq : ∀ i, q i < 256) (j : Nat) :
    (scaleMinK4 q j).1 < 64 ∧ (scaleMinK4 q j).2 < 64 := by
  by_cases hj : j < 4
  · exact scaleMinK4_lt64_low q j hj
  · exact scaleMinK4_lt64_high q hq j (by omega)

/-- Q4_K: the two sub-blocks of 64-element group `g` use scale indices `2g` and `2g+1`, below 8. -/
theorem q4K_scale_indices (g : Nat) (hg : g < 4) : 2 * g < 8 ∧ 2 * g + 1 < 8 := by
  omega

/-- Q5_K masks u1 = 1,4,16,64 and u2 = 2,8,32,128 by shifting left 2 each group. -/
theorem q5K_masks : (1 <<< 0, 1 <<< 2, 1 <<< 4, 1 <<< 6) = (1, 4, 16, 64) ∧
    (2 <<< 0, 2 <<< 2, 2 <<< 4, 2 <<< 6) = (2, 8, 32, 128) := by
  decide

/-- The Q5_K masks never overlap: u1 and u2 select different bits for each group. -/
theorem q5K_masks_disjoint : ∀ g, g < 4 → (1 <<< (2 * g)) &&& (2 <<< (2 * g)) = 0 := by
  decide

/-- Block byte budgets: Q4_K is 4 + 12 + 128 = 144 and Q5_K is 4 + 12 + 32 + 128 = 176. -/
theorem q4K_q5K_bytes : 4 + 12 + 128 = 144 ∧ 4 + 12 + 32 + 128 = 176 := by
  exact ⟨rfl, rfl⟩

-- § IQ4_XS

/-- IQ4_XS 6-bit group scale from `scales_l` byte and 16-bit `scales_h`. -/
def iq4xsScale (sl : Nat → Nat) (sh ib : Nat) : Nat :=
  ((sl (ib / 2) / 2 ^ (4 * (ib % 2))) % 16) + ((sh / 2 ^ (2 * ib)) % 4) * 16

/-- The assembled group scale is a 6-bit value. -/
theorem iq4xsScale_lt64 (sl : Nat → Nat) (sh ib : Nat) : iq4xsScale sl sh ib < 64 := by
  unfold iq4xsScale
  have h1 : (sl (ib / 2) / 2 ^ (4 * (ib % 2))) % 16 < 16 := Nat.mod_lt _ (by decide)
  have h2 : (sh / 2 ^ (2 * ib)) % 4 < 4 := Nat.mod_lt _ (by decide)
  omega

/-- Signed effective scale `ls - 32` lies in [-32, 31]. -/
theorem iq4xsScale_signed_range (sl : Nat → Nat) (sh ib : Nat) :
    -32 ≤ (iq4xsScale sl sh ib : Int) - 32 ∧ (iq4xsScale sl sh ib : Int) - 32 ≤ 31 := by
  have := iq4xsScale_lt64 sl sh ib
  omega

/-- For ib < 8 the `scales_l` index `ib/2` is below 4. -/
theorem iq4xs_scales_l_index (ib : Nat) (h : ib < 8) : ib / 2 < 4 := by
  omega

/-- IQ4_XS byte budget: 2 + 2 + 4 + 128 = 136. -/
theorem iq4xs_bytes : 2 + 2 + 4 + 128 = 136 := by
  rfl

/-- Eight groups of 32 elements fill the 256-element super-block. -/
theorem iq4xs_groups : 8 * 32 = 256 := by
  rfl

-- § Q3_K

/-- Q3_K byte budget: 32 + 64 + 12 + 2 = 110. -/
theorem q3K_bytes : 32 + 64 + 12 + 2 = 110 := by
  rfl

/-- The hoisted mask `m` takes values 1,2,4,...,128 across the 8 (n, j) iterations: `m = 2^(4n' + j)`
for the iteration number `i = 4n' + j`, and it is a single set bit within a byte. -/
theorem q3K_mask_in_byte (i : Nat) (h : i < 8) : 2 ^ i < 256 := by
  have : 2 ^ i < 2 ^ 8 := Nat.pow_lt_pow_right (by decide) h
  simpa using this

/-- Q3_K high-bit correction: the signed 3-bit value lies in [-4, 3]. -/
theorem q3K_value_range (lo hbit : Nat) (hl : lo < 4) (hh : hbit < 2) :
    -4 ≤ (lo : Int) - (if hbit = 1 then 0 else 4) ∧ (lo : Int) - (if hbit = 1 then 0 else 4) ≤ 3 := by
  split <;> omega

-- § fp16 / bf16 field extraction

/-- fp16 exponent field. -/
def fp16Exp (h : Nat) : Nat := (h / 1024) % 32

/-- fp16 mantissa field. -/
def fp16Man (h : Nat) : Nat := h % 1024

/-- fp16 sign field. -/
def fp16Sign (h : Nat) : Nat := (h / 32768) % 2

/-- The three fp16 fields recompose the 16-bit pattern. -/
theorem fp16_fields_recompose (h : Nat) (hh : h < 65536) :
    fp16Man h + 1024 * fp16Exp h + 32768 * fp16Sign h = h := by
  unfold fp16Man fp16Exp fp16Sign
  omega

/-- The exponent field is below 32. -/
theorem fp16Exp_lt (h : Nat) : fp16Exp h < 32 := by
  unfold fp16Exp
  exact Nat.mod_lt _ (by decide)

/-- Subnormal path renormalisation: exp starts at 127 - 15 + 1 = 113. -/
theorem fp16_subnormal_start : 127 - 15 + 1 = 113 := by
  rfl

/-- Normal path rebias: 15 maps to 127, so fp16 1.0 (exp field 15) becomes fp32 exponent 127. -/
theorem fp16_rebias_one : 15 - 15 + 127 = 127 := by
  rfl

/-- Infinity/NaN exponent field 31 becomes 0xFF in fp32 (0x7F800000 has exponent bits 255). -/
theorem fp16_inf_exp : 0x7F800000 / 2 ^ 23 = 255 := by
  decide

/-- bf16 to fp32 is a left shift by 16: the low 16 bits of the fp32 are zero. -/
theorem bf16_low_bits_zero (h : Nat) : (h <<< 16) % 65536 = 0 := by
  rw [Nat.shiftLeft_eq]
  omega

/-- read_u16 is little endian: p0 + 256 * p1 for byte inputs is below 65536. -/
theorem read_u16_lt (p0 p1 : Nat) (h0 : p0 < 256) (h1 : p1 < 256) : p0 + 256 * p1 < 65536 := by
  omega

/-- read_u16 low byte recovery. -/
theorem read_u16_low (p0 p1 : Nat) (h0 : p0 < 256) : (p0 + 256 * p1) % 256 = p0 := by
  omega

/-- read_u16 high byte recovery. -/
theorem read_u16_high (p0 p1 : Nat) (h0 : p0 < 256) (h1 : p1 < 256) : (p0 + 256 * p1) / 256 = p1 := by
  omega

end Strata.Dequant
