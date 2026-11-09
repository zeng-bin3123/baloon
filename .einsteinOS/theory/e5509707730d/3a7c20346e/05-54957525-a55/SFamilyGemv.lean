/-!
# MODULE: s_family_gemv
SOURCES: src/kernels/cuda/s_gemv.cu
S-family GEMV (2/4/8-bit packed codes with per-group fp32 scale and optional offset) against fp16, Q8_K or Q8_0
activations; naive, split (block per row) and warp-per-row kernels.

## Source semantics
A row of n_in codes of width b occupies n_in / (8/b) bytes with n_in / group_elems scales. The weight is
decoded as `decode(code) * scale + offset` before the activation multiply. Group size is a power of two (passed
as its logarithm). Hoisting the activation block out of a 16-element lane-iteration is valid because the
iteration start is a multiple of 16 and blocks are 256 (Q8_K) or 32 (Q8_0) wide. Q8_K needs n_in % 256 = 0;
Q8_0 only n_in % 32 = 0, which is why a Q8_0 kernel exists (n_in = 640).

## Form template identifiers
perByte -> "codes per byte"; quad lemmas -> "consecutive-element packing"; q8k/q8_0 constants -> "activation block".
-/
namespace Strata.SGemv

-- § layout
/-- Codes per byte for a code width. -/
def perByte (bits : Nat) : Nat := 8 / bits

/-- Bytes of one four-code word (`QW`): 1, 2 or 4. -/
def wordBytes (bits : Nat) : Nat := 4 * bits / 8

/-- The three supported widths pack 4, 2 and 1 codes per byte. -/
theorem perByte_values : perByte 2 = 4 ∧ perByte 4 = 2 ∧ perByte 8 = 1 := by
  decide

/-- Four-code words are 1, 2 and 4 bytes wide. -/
theorem wordBytes_values : wordBytes 2 = 1 ∧ wordBytes 4 = 2 ∧ wordBytes 8 = 4 := by
  decide

/-- Q8_K block geometry: 292 bytes per 256 elements. -/
def q8kBytes : Nat := 292

/-- Q8_0 block geometry: 34 bytes per 32 elements. -/
def q8_0Bytes : Nat := 34

/-- `ffn_down_shexp` (n_in = 640) is a Q8_0-shaped input but structurally cannot be Q8_K. -/
theorem q8k_impossible_640 : 640 % 32 = 0 ∧ 640 % 256 ≠ 0 := by
  decide

-- § quad packing and hoisting
/-- 4-bit codes: a quad starting at a multiple of 4 spans bytes i/2 and i/2+1 with in-byte slot k%2. -/
theorem quad_bytes_s4 (i k : Nat) (hi : i % 4 = 0) (hk : k < 4) :
    (i + k) / 2 = i / 2 + k / 2 ∧ (i + k) % 2 = k % 2 := by
  omega

/-- 2-bit codes: a quad is exactly one byte, slot k. -/
theorem quad_bytes_s2 (i k : Nat) (hi : i % 4 = 0) (hk : k < 4) :
    (i + k) / 4 = i / 4 ∧ (i + k) % 4 = k := by
  omega

/-- 8-bit codes: byte index is the element index. -/
theorem quad_bytes_s8 (i k : Nat) : (i + k) / 1 = i + k := by
  omega

/-- A quad inside one group shares its scale (group size 4m). -/
theorem quad_in_group (i k m : Nat) (hi : i % 4 = 0) (hk : k < 4) : (i + k) / (4 * m) = i / (4 * m) := by
  rw [← Nat.div_div_eq_div_mul, ← Nat.div_div_eq_div_mul]
  have : (i + k) / 4 = i / 4 := by omega
  rw [this]

/-- Q8_K hoisting: sixteen consecutive elements from a multiple of 16 stay in one 256-block. -/
theorem hoist_q8k (i k : Nat) (hi : i % 16 = 0) (hk : k < 16) :
    (i + k) / 256 = i / 256 ∧ (i + k) % 256 = i % 256 + k := by
  omega

/-- Q8_0 hoisting: sixteen consecutive elements from a multiple of 16 stay in one 32-block. -/
theorem hoist_q8_0 (i k : Nat) (hi : i % 16 = 0) (hk : k < 16) :
    (i + k) / 32 = i / 32 ∧ (i + k) % 32 = i % 32 + k := by
  omega

/-- Lane-iteration starts `lane*16 + 512*j` are multiples of 16. -/
theorem lane_start_mult16 (lane j : Nat) : (lane * 16 + 512 * j) % 16 = 0 := by
  omega

/-- Using a shift for the group index equals division by the (power-of-two) group size. -/
theorem group_shift_eq_div (i s : Nat) : i >>> s = i / 2 ^ s := by
  exact Nat.shiftRight_eq_div_pow i s

/-- Group sizes defined by the format are powers of two. -/
def isPow2 (n : Nat) : Bool := n != 0 && (n &&& (n - 1)) == 0

/-- 16, 32, 64 pass the power-of-two guard; 48 is refused. -/
theorem pow2_guard : isPow2 16 = true ∧ isPow2 32 = true ∧ isPow2 64 = true ∧ isPow2 48 = false := by
  decide

-- § launch coverage
/-- Thread-per-row launch (128 threads) covers all rows. -/
theorem grid_thread_per_row (n : Nat) : n ≤ ((n + 127) / 128) * 128 := by
  omega

/-- Warp-per-row launch (256 threads, 8 warps) covers all rows. -/
theorem grid_warp_per_row (n : Nat) : n ≤ ((n + 7) / 8) * 8 := by
  omega

-- § codebook
/-- The non-linear IQ4 codebook (kvalues_iq4nl). -/
def kIq4Nl : List Int := [-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113]

/-- The codebook has 16 entries (one per 4-bit code). -/
theorem kIq4Nl_length : kIq4Nl.length = 16 := by
  decide

/-- The codebook is strictly increasing. -/
theorem kIq4Nl_sorted : kIq4Nl.Pairwise (· < ·) := by
  decide

end Strata.SGemv
