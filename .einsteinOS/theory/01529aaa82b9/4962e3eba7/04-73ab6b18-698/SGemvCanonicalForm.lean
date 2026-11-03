/-!
# MODULE: s_gemv_canonical_form
The S-family GEMV's canonical weight form: code widths, packing density, group addressing,
activation block strides, strided work partitioning.

## Source semantics
value = cb[code] * scale + offset, parameterised by code_bits (2/4/8), code_bias, group_elems and
codebook. Row o of a [n_in, n_out] tensor is a contiguous run of n_in/group_elems groups. Q8_K
activation blocks are 292 bytes per 256 elements; Q8_0 blocks are 34 bytes per 32. A row split
across T threads gives thread t the strided subset i = t mod T.

## Form template identifiers
SForm -> SForm ; perByte -> codes_per_byte ; scaleIdx -> o*(n_in/g) + i/g ; stride -> threads_per_row
-/

set_option maxRecDepth 100000

namespace Strata.SGemv

/-- Canonical form attributes relevant to addressing. -/
structure SForm where
  codeBits : Nat
  codeBias : Int
  groupElems : Nat

/-- Q2_0 attributes. -/
def q2_0 : SForm := ⟨2, -1, 64⟩
/-- Q4_0 attributes. -/
def q4_0 : SForm := ⟨4, -8, 32⟩
/-- Q5_0 attributes (8-bit container, bias -16). -/
def q5_0 : SForm := ⟨8, -16, 32⟩

/-- Codes per byte. -/
def perByte (f : SForm) : Nat := 8 / f.codeBits

/-- 2, 4 and 8 bit codes pack 4, 2 and 1 per byte. -/
theorem per_byte_vals : perByte q2_0 = 4 ∧ perByte q4_0 = 2 ∧ perByte q5_0 = 1 := by decide

/-- Decode with an optional offset. -/
def decode (cb scale off : Int) : Int := cb * scale + off

/-- Without an offset the decode is a pure product. -/
theorem decode_no_offset (cb scale : Int) : decode cb scale 0 = cb * scale := by
  unfold decode; omega

/-- Biased integer value range of a code below m (m = 2^bits). -/
theorem biased_range (m : Nat) (bias : Int) (c : Nat) (h : c < m) :
    bias ≤ (c : Int) + bias ∧ (c : Int) + bias ≤ (m : Int) - 1 + bias := by omega

/-- Q2_0 symbols are -1..2, Q4_0 symbols are -8..7. -/
theorem sym_ranges : (0 : Int) + (-1) = -1 ∧ (3 : Int) + (-1) = 2 ∧ (0 : Int) + (-8) = -8 ∧
    (15 : Int) + (-8) = 7 := by decide

-- § Addressing

/-- Row-major bound. -/
theorem rowmajor_lt {i n j m : Nat} (hi : i < n) (hj : j < m) : i * m + j < n * m := by
  have h1 : (i + 1) * m ≤ n * m := Nat.mul_le_mul_right m hi
  rw [Nat.add_mul, Nat.one_mul] at h1
  omega

/-- Group index within a row stays below n_in / g when g divides n_in. -/
theorem group_in_row (i g nIn : Nat) (hg : 0 < g) (hd : g ∣ nIn) (hi : i < nIn) :
    i / g < nIn / g := by
  apply (Nat.div_lt_iff_lt_mul hg).2
  rw [Nat.div_mul_cancel hd]
  exact hi

/-- Scale/offset index of element i of row o stays inside the n_out * (n_in / g) array. -/
theorem scale_index_lt (o i nOut nIn g : Nat) (hg : 0 < g) (hd : g ∣ nIn) (ho : o < nOut)
    (hi : i < nIn) : o * (nIn / g) + i / g < nOut * (nIn / g) :=
  rowmajor_lt ho (group_in_row i g nIn hg hd hi)

/-- Four consecutive elements starting at a multiple of four share one code byte (S2 quads). -/
theorem quad_shares_byte (k d : Nat) (hd : d < 4) : (4 * k + d) / 4 = k := by omega

/-- Row byte count is exact when the code density divides n_in. -/
theorem row_code_bytes (nIn p : Nat) (hp : 0 < p) (hd : p ∣ nIn) : p * (nIn / p) = nIn :=
  Nat.mul_div_cancel' hd

-- § Strided row split

/-- Thread t = i mod T handles element i at iteration i / T: every element is covered. -/
theorem stride_cover (i T : Nat) : i % T + (i / T) * T = i := by
  have := Nat.mod_add_div i T
  rw [Nat.mul_comm (i / T) T]
  exact this

/-- The covering thread is below T (so it exists). -/
theorem stride_thread_lt (i T : Nat) (hT : 0 < T) : i % T < T := Nat.mod_lt i hT

/-- Thread t at iteration k handles element t + k*T, and that element's covering thread is t. -/
theorem stride_unique (t k T : Nat) (ht : t < T) : (t + k * T) % T = t := by
  rw [Nat.add_mul_mod_self_right]
  exact Nat.mod_eq_of_lt ht

-- § Activation block strides

/-- Q8_K block: d 4 + qs 256 + bsums 32 = 292; bsums at offset 260. -/
theorem q8k_block : 4 + 256 + 16 * 2 = 292 ∧ 4 + 256 = 260 := ⟨rfl, rfl⟩

/-- Q8_0 block: d 2 + qs 32 = 34. -/
theorem q8_0_block : 2 + 32 = 34 := rfl

/-- 2560 activations: 10 Q8_K blocks = 2920 B; 640 activations: 20 Q8_0 blocks = 680 B. -/
theorem act_bytes : 2560 / 256 * 292 = 2920 ∧ 640 / 32 * 34 = 680 := by decide

/-- ffn_down_shexp n_in = 640 is a multiple of 32 but not of 256: Q8_K is impossible. -/
theorem shexp_not_q8k : 640 % 32 = 0 ∧ 640 % 256 ≠ 0 := by decide

/-- Power-of-two test used for group_elems requirements. -/
def isPow2 (g : Nat) : Bool := decide (0 < g) && decide ((g &&& (g - 1)) = 0)

/-- 32, 64 and 256 qualify; 20 does not. -/
theorem pow2_cases : isPow2 32 = true ∧ isPow2 64 = true ∧ isPow2 256 = true ∧
    isPow2 20 = false := by decide

/-- Expert gate/up [2560 x 640]: 640 outputs over 48 SMs is 13 per SM with 16 left over. -/
theorem sm_occupancy : 640 / 48 = 13 ∧ 640 % 48 = 16 := by decide

end Strata.SGemv
