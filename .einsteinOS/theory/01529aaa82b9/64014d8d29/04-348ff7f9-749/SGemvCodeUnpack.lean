/-!
# MODULE: s_gemv_code_unpack
SOURCES: src/kernels/cuda/s_gemv.cu
Index arithmetic of the S-family GEMV: Q8_K / Q8_0 activation addressing, hoisting of the activation block out
of the 16-element lane iteration, the bit order of packed codes (S2/S4/S8), group-shift division, alignment of
the code words, the same-group requirement, and launch-grid coverage.

## Source semantics
* Q8_K blocks are 292 bytes per 256 elements (`d`, 256 int8, 16 int16 bsums); Q8_0 blocks are 34 bytes per 32.
* A lane takes QE = 16 consecutive elements starting at a multiple of 16, so they never straddle a block.
* Element `i+k` is bits `[k*B, (k+1)*B)` of the little-endian word at byte `i/PER_BYTE`.
* The group index is `i >> log2(group_elems)`; sixteen elements share one scale only if 16 divides the group.

## Form template identifiers
q8k_at -> q8k_element_offset ; q8_0_at -> q8_0_element_offset ; group_shift -> log2_group_elems
QE -> elements_per_lane_iteration ; QW -> bytes_per_four_code_word
-/

namespace Strata.SGemv

-- § activation addressing

/-- Byte offset of element `i` inside a Q8_K activation buffer. -/
def q8kOff (i : Nat) : Nat := (i / 256) * 292 + 4 + i % 256

/-- Byte offset of element `i` inside a Q8_0 activation buffer. -/
def q80Off (i : Nat) : Nat := (i / 32) * 34 + 2 + i % 32

/-- Distinct elements map to distinct Q8_K bytes. -/
theorem q8kOff_inj (i j : Nat) (h : q8kOff i = q8kOff j) : i = j := by
  unfold q8kOff at h
  omega

/-- Distinct elements map to distinct Q8_0 bytes. -/
theorem q80Off_inj (i j : Nat) (h : q80Off i = q80Off j) : i = j := by
  unfold q80Off at h
  omega

/-- Every element byte lies strictly before the bsums tail of its Q8_K block. -/
theorem q8kOff_before_bsums (i : Nat) : q8kOff i + 32 < (i / 256 + 1) * 292 := by
  unfold q8kOff
  omega

/-- A buffer of `nb` blocks is never read past its end. -/
theorem q8kOff_in_buffer (i nb : Nat) (h : i < 256 * nb) : q8kOff i < 292 * nb := by
  unfold q8kOff
  omega

/-- Same for Q8_0. -/
theorem q80Off_in_buffer (i nb : Nat) (h : i < 32 * nb) : q80Off i < 34 * nb := by
  unfold q80Off
  omega

-- § hoisting the activation block out of the lane iteration

/-- With `i % 16 = 0`, sixteen consecutive elements share one Q8_K block and sit at consecutive offsets. -/
theorem hoist_q8k (i k : Nat) (h16 : i % 16 = 0) (hk : k < 16) :
    (i + k) / 256 = i / 256 ∧ (i + k) % 256 = i % 256 + k ∧ i % 256 + 16 ≤ 256 := by
  omega

/-- The same for Q8_0 blocks of 32. -/
theorem hoist_q8_0 (i k : Nat) (h16 : i % 16 = 0) (hk : k < 16) :
    (i + k) / 32 = i / 32 ∧ (i + k) % 32 = i % 32 + k ∧ i % 32 + 16 ≤ 32 := by
  omega

/-- Every lane iteration start `lane*16 + 512*t` is a multiple of 16. -/
theorem lane_iter_aligned (lane t : Nat) : (lane * 16 + 512 * t) % 16 = 0 := by omega

/-- Distinct (lane, iteration) pairs cover distinct 16-element octets. -/
theorem lane_iter_inj (lane t lane' t' : Nat) (hl : lane < 32) (hl' : lane' < 32)
    (h : lane * 16 + 512 * t = lane' * 16 + 512 * t') : lane = lane' ∧ t = t' := by
  omega

/-- The 16 int8 activations of a lane stay inside the int8 payload of the block (before bsums). -/
theorem lane_payload_bound (i : Nat) (h16 : i % 16 = 0) : 4 + i % 256 + 15 < 4 + 256 := by omega

/-- When the row length is a multiple of 16 the partial-tail loop never runs. -/
theorem tail_empty (i n : Nat) (hi : i % 16 = 0) (hn : n % 16 = 0) (h : ¬ (i + 16 ≤ n)) : ¬ (i < n) := by
  omega

-- § same-group requirement

/-- Elements `m*a .. m*a+m-1` all fall in group `a / b` of size `m*b`. -/
theorem same_group (m a b k : Nat) (hm : 0 < m) (hk : k < m) : (m * a + k) / (m * b) = a / b := by
  have h1 : (m * a + k) / m = a := by
    simp [Nat.mul_add_div hm, Nat.div_eq_of_lt hk]
  rw [← Nat.div_div_eq_div_mul, h1]

/-- A quad (four elements) shares a group whenever 4 divides the group size. -/
theorem quad_same_group (a b k : Nat) (hk : k < 4) : (4 * a + k) / (4 * b) = a / b :=
  same_group 4 a b k (by decide) hk

/-- A 16-element lane block shares a group whenever 16 divides the group size. -/
theorem octet16_same_group (a b k : Nat) (hk : k < 16) : (16 * a + k) / (16 * b) = a / b :=
  same_group 16 a b k (by decide) hk

/-- The Q8_0 split launcher only checks `group_elems % 4 == 0`, but the kernel reads ONE scale for sixteen
elements: with a group of 4 the sixteen elements span several groups.  Every real group (16/32/64) is safe. -/
theorem q8_0_split_group_hazard : (15 : Nat) / 4 ≠ 0 / 4 := by decide

/-- Group sizes 16, 32, 64 as shifts. -/
theorem group_shifts : (1 <<< 4 = 16) ∧ (1 <<< 5 = 32) ∧ (1 <<< 6 = 64) := by decide

/-- A right shift is division by the power of two. -/
theorem group_shift_div (i s : Nat) : i >>> s = i / 2 ^ s := Nat.shiftRight_eq_div_pow i s

-- § bit order of the packed codes

/-- S2: the four codes of a byte are its 2-bit fields, and the byte is recovered from them. -/
theorem s2_fields (b : Nat) (h : b < 256) :
    b = b % 4 + 4 * ((b / 4) % 4) + 16 * ((b / 16) % 4) + 64 * ((b / 64) % 4) := by omega

/-- S4: the 16-bit word of four codes yields the same nibbles as the scalar per-byte extraction. -/
theorem s4_word (b0 b1 : Nat) (h0 : b0 < 256) (h1 : b1 < 256) :
    ((b0 + 256 * b1) % 16 = (b0 / 1) % 16) ∧ (((b0 + 256 * b1) / 16) % 16 = (b0 / 16) % 16) ∧
    (((b0 + 256 * b1) / 256) % 16 = (b1 / 1) % 16) ∧ (((b0 + 256 * b1) / 4096) % 16 = (b1 / 16) % 16) := by
  omega

/-- S8: the 32-bit word of four codes yields the four bytes. -/
theorem s8_word (b0 b1 b2 b3 : Nat) (h0 : b0 < 256) (h1 : b1 < 256) (h2 : b2 < 256) (h3 : b3 < 256) :
    let v := b0 + 256 * b1 + 65536 * b2 + 16777216 * b3
    (v % 256 = b0) ∧ ((v / 256) % 256 = b1) ∧ ((v / 65536) % 256 = b2) ∧ ((v / 16777216) % 256 = b3) := by
  intro v
  simp only [v]
  omega

/-- Four lane words of QW bytes cover exactly 16 codes: `4 * (4*B/8) = 16*B/8` for B in {2,4,8}. -/
theorem word_bytes : ∀ B ∈ [2, 4, 8], 4 * (4 * B / 8) = 16 * B / 8 := by decide

/-- Alignment of the wide code loads: S8 words 4-aligned, S4 words 2-aligned, rows stay aligned. -/
theorem code_load_alignment (i n : Nat) (hi : i % 16 = 0) (hn : n % 16 = 0) :
    (i / 1) % 4 = 0 ∧ (i / 2) % 2 = 0 ∧ (n / 1) % 4 = 0 ∧ (n / 2) % 2 = 0 ∧ (n / 4) % 1 = 0 := by
  omega

-- § launch grids

/-- 128 threads per block, one row per thread: the grid covers all rows. -/
theorem grid_cover_128 (n : Nat) : n ≤ ((n + 127) / 128) * 128 := by omega

/-- 8 warps per block, one row per warp: the grid covers all rows. -/
theorem grid_cover_warps (n : Nat) : n ≤ ((n + 7) / 8) * 8 := by omega

end Strata.SGemv
