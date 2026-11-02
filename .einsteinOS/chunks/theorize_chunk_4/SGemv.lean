/-!
# MODULE: s_gemv
S-family quantized GEMV (2/4/8-bit codes, per-group scale, optional offset, optional non-linear IQ4_NL codebook)
with FP16, Q8_K and Q8_0 activations. Naive, split-row and warp-per-row variants.

## Source semantics
w = decode(code)*scale + offset, applied before multiplying by the activation. PER_BYTE = 8/bits.
A lane processes 16 consecutive elements per iteration; those lie inside one scale group (groups are 16/32/64)
and one activation block (Q8_K 256, Q8_0 32). Q8_K needs n_in % 256 = 0, Q8_0 needs n_in % 32 = 0.

## Form template identifiers
code_bits -> width; group_elems -> group; kIq4Nl -> nonlinear_codebook
-/
namespace Strata.SGemv

/-- Codes per byte. -/
def perByte (bits : Nat) : Nat := 8 / bits

/-- Per-byte counts for the three widths. -/
theorem per_byte_values : perByte 2 = 4 ∧ perByte 4 = 2 ∧ perByte 8 = 1 := by decide

/-- Bytes per four-code word: 1, 2, 4. -/
theorem quad_bytes : 4 * 2 / 8 = 1 ∧ 4 * 4 / 8 = 2 ∧ 4 * 8 / 8 = 4 := by decide

/-- Bytes per sixteen-code span: 4, 8, 16. -/
theorem octet16_bytes : 16 * 2 / 8 = 4 ∧ 16 * 4 / 8 = 8 ∧ 16 * 8 / 8 = 16 := by decide

/-- A 16-element lane span starting at a multiple of 16 stays within one group of 16. -/
theorem span_in_group16 (i : Nat) (h : i % 16 = 0) : (i + 15) / 16 = i / 16 := by omega

/-- Same for groups of 32. -/
theorem span_in_group32 (i : Nat) (h : i % 16 = 0) : (i + 15) / 32 = i / 32 := by omega

/-- Same for groups of 64. -/
theorem span_in_group64 (i : Nat) (h : i % 16 = 0) : (i + 15) / 64 = i / 64 := by omega

/-- A 16-element span stays inside one Q8_K block of 256. -/
theorem span_in_q8k (i : Nat) (h : i % 16 = 0) : i % 256 + 16 ≤ 256 := by omega

/-- A 16-element span stays inside one Q8_0 block of 32. -/
theorem span_in_q8_0 (i : Nat) (h : i % 16 = 0) : i % 32 + 16 ≤ 32 := by omega

/-- Lane starts are multiples of 16, and so are all strides of 32*16. -/
theorem lane_start (lane k : Nat) : (lane * 16 + k * (32 * 16)) % 16 = 0 := by omega

/-- Shared-memory codebook is indexed by a 4-bit value. -/
theorem code_index_lt (code : Nat) : code % 16 < 16 := by omega

/-- IQ4_NL codebook values are strictly increasing. -/
def iq4nl : List Int := [-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113]

/-- Codebook has 16 entries. -/
theorem iq4nl_len : iq4nl.length = 16 := by decide

/-- Codebook is sorted (adjacent differences positive). -/
theorem iq4nl_sorted : (List.zip iq4nl iq4nl.tail).all (fun p => decide (p.1 < p.2)) = true := by decide

/-- Q8_0 requires n_in multiple of 32; 640 qualifies but is not a multiple of 256. -/
theorem shexp_not_q8k : 640 % 32 = 0 ∧ 640 % 256 ≠ 0 := by decide

/-- Warps per block of 256 threads cover all output rows. -/
theorem warp_grid (n : Nat) : n ≤ ((n + 7) / 8) * 8 := by omega

end Strata.SGemv
