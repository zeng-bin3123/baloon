/-!
# MODULE: s2_canonical_dequant
SOURCES: src/kernels/cuda/dequant_s2.cu
The reference decode of the S2 canonical form (Q2_0's re-packing): 2-bit codes, four per byte,
LSB-first, one FP32 scale per 64-element group, value `(code - 1) * scale`.

## Source semantics
* Element `j` of a group lives at byte `j / 4`, bits `(j % 4) * 2 .. +2` of that group's 16-byte code run;
  group `b` starts at byte `16 b` and its outputs at `64 b`.
* `code = (byte >> ((j%4)*2)) & 3 ∈ {0,1,2,3}`; the bias `-1` is applied to the CODE in the integer
  domain, so the factor multiplying the scale is exactly one of `-1, 0, 1, 2` (bit-exact decode).
* The packing is lossless: a byte equals `Σ_k code_k · 4^k`.
* `n_blocks ≤ 0` is a no-op; grid `⌈n_blocks/256⌉` of 256 threads, one thread per group.

## Form template identifiers
* `QK = 64`, `CODES_PER_BYTE = 4`
* `code` ↔ `(c[j / CODES_PER_BYTE] >> ((j % CODES_PER_BYTE) * 2)) & 0x03`
* `factor` ↔ `code - 1` (the `code_bias`)
-/

namespace Strata.DequantS2

/-- `x & 3 = x mod 4`. -/
theorem and3 (x : Nat) : x &&& 3 = x % 4 := by simpa using Nat.and_two_pow_sub_one_eq_mod x 2

-- § code extraction

/-- The 2-bit code in slot `k` of byte `b`. -/
def code (b k : Nat) : Nat := (b >>> (k * 2)) &&& 3

/-- The integer factor applied to the scale. -/
def factor (b k : Nat) : Int := (code b k : Int) - 1

/-- Code extraction as arithmetic: slot `k` is base-4 digit `k`. -/
theorem code_eq_digit (b k : Nat) : code b k = b / 4 ^ k % 4 := by
  unfold code
  rw [Nat.shiftRight_eq_div_pow, and3, Nat.mul_comm k 2, Nat.pow_mul]

/-- Codes are 2-bit. -/
theorem code_lt (b k : Nat) : code b k < 4 := by
  rw [code_eq_digit]; exact Nat.mod_lt _ (by decide)

/-- **The integer-domain bias**: the factor is one of `-1, 0, 1, 2`, so `factor * d` is a single
rounding of an exact small integer times the scale. -/
theorem factor_range (b k : Nat) : -1 ≤ factor b k ∧ factor b k ≤ 2 := by
  have := code_lt b k
  unfold factor; omega

/-- The four codes of a byte determine it: packing four 2-bit codes per byte is lossless. -/
theorem byte_from_codes (b : Nat) (hb : b < 256) :
    code b 0 + 4 * code b 1 + 16 * code b 2 + 64 * code b 3 = b := by
  simp only [code_eq_digit]
  simp
  omega

-- § plane layout

/-- Element `j < 64` of a group reads one of that group's 16 code bytes. -/
theorem byte_in_group (j : Nat) (_hj : j < 64) : j / 4 < 16 := by omega

/-- Global consistency: element `e = 64 b + j` of the plane is byte `16 b + j/4`, slot `j % 4` — the
per-group pointer arithmetic (`codes + b*16`, `out + b*64`) agrees with the plane-wide convention
"element i at byte i/4, bits (i%4)*2". -/
theorem global_layout (b j : Nat) (_hj : j < 64) :
    (64 * b + j) / 4 = 16 * b + j / 4 ∧ (64 * b + j) % 4 = j % 4 := by omega

/-- Distinct (byte, slot) pairs are distinct elements: the layout is a bijection. -/
theorem layout_injective (B₁ B₂ k₁ k₂ : Nat) (h₁ : k₁ < 4) (h₂ : k₂ < 4)
    (h : 4 * B₁ + k₁ = 4 * B₂ + k₂) : B₁ = B₂ ∧ k₁ = k₂ := by omega

/-- Outputs of distinct groups are disjoint: group `b` writes `[64 b, 64 b + 64)`. -/
theorem outputs_disjoint (b₁ b₂ j₁ j₂ : Nat) (h₁ : j₁ < 64) (h₂ : j₂ < 64)
    (h : 64 * b₁ + j₁ = 64 * b₂ + j₂) : b₁ = b₂ := by omega

/-- The S2 plane costs `16` code bytes plus a 4-byte scale per 64 elements (2.5 bits/weight). -/
theorem plane_bytes_per_group : 64 * 2 / 8 + 4 = 20 ∧ 20 * 8 * 2 = 5 * 64 := by decide

-- § launch

/-- One thread per group: the grid `⌈n/256⌉ · 256` covers every group. -/
theorem grid_covers (n b : Nat) (hb : b < n) : b < (n + 255) / 256 * 256 := by omega

end Strata.DequantS2
