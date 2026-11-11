/-!
# MODULE: s_family_gemv
SOURCES: src/kernels/cuda/s_gemv.cu
The S-family GEMV (2/4/8-bit codes, per-group FP32 scale, optional per-group offset, affine or IQ4_NL
codebook) against FP16, Q8_K and Q8_0 activations: the naive thread-per-row kernels, the block-split
FP16 kernel, and the warp-per-row Q8 kernel with 16 elements per lane-iteration.

## Source semantics
* Codes are packed `PER_BYTE = 8 / CODE_BITS` per byte, little-endian, element `i` at byte
  `i / PER_BYTE`, bits `(i % PER_BYTE) * CODE_BITS`.  A run of four codes is `QW = 4·CODE_BITS/8`
  bytes, read as one little-endian word; code `k` of the word is bits `[k·CB, (k+1)·CB)`.
* Weight `w = decode(code)·d + b` (offset applied to the WEIGHT before the activation multiply);
  affine `decode(code) = code + bias`, IQ4_NL `decode(code) = kIq4Nl[code & 15]`.
* Group index: dense kernels divide by `group_elems`; split kernels shift by `group_shift`, computed as
  the least `s` with `2^s ≥ group_elems`.  Split guards: power of two, and the per-iteration span lies in
  one group: `%4` for the FP16 split (QE = 4), `%16` for the Q8_K split, but only `%4` for the Q8_0
  split although that kernel ALSO consumes QE = 16 elements per scale load.
* Q8 split kernel: lane `l` starts at `16 l` and advances by `512`; since `16 | i`, the 16 elements of a
  lane-iteration lie in one Q8_K (256) and one Q8_0 (32) activation block, which is what makes hoisting
  the block pointer exact.
* Q8_K entry points refuse `code_bits = 2` and need `n_in % 256 = 0`; `ffn_down_shexp` (`n_in = 640`)
  can therefore only use the Q8_0 path.

## Form template identifiers
* `perByte`, `quadBytes` ↔ `PER_BYTE`, `QW`/`QB`
* `gsLoop` ↔ `while ((1 << group_shift) < group_elems) ++group_shift`
* `pow2Guard` ↔ `(g & (g - 1)) == 0`;  `guardQ8_0Split`, `guardQ8KSplit` ↔ the host checks
* `laneStart` ↔ `i = lane * QE + it * 32 * QE`
* `weight` ↔ `decode_tbl(code) * d + b`
-/

namespace Strata.SGemv

-- § bit helpers

/-- `x & 3 = x mod 4`. -/
theorem and3 (x : Nat) : x &&& 3 = x % 4 := by simpa using Nat.and_two_pow_sub_one_eq_mod x 2
/-- `x & 15 = x mod 16`. -/
theorem and15 (x : Nat) : x &&& 15 = x % 16 := by simpa using Nat.and_two_pow_sub_one_eq_mod x 4
/-- `x & 255 = x mod 256`. -/
theorem and255 (x : Nat) : x &&& 255 = x % 256 := by simpa using Nat.and_two_pow_sub_one_eq_mod x 8

-- § form constants

/-- Codes per byte. -/
def perByte (cb : Nat) : Nat := 8 / cb

/-- Bytes per word of four codes. -/
def quadBytes (cb : Nat) : Nat := 4 * cb / 8

/-- The three widths and their packing constants: S2 `(4, 1)`, S4 `(2, 2)`, S8 `(1, 4)` — the
`QW` table whose S4/S8 entries the first version of the Q8 kernel got backwards. -/
theorem width_table :
    (perByte 2, quadBytes 2) = (4, 1) ∧ (perByte 4, quadBytes 4) = (2, 2) ∧
    (perByte 8, quadBytes 8) = (1, 4) := by decide

/-- Decoded weight (exact model). -/
def weight (code bias d b : Int) : Int := (code + bias) * d + b

/-- Applying the offset to the weight or distributing it over the activation is the same EXACT value;
the source's ordering matters only for FP32 rounding, which is why no-offset types could not reveal it. -/
theorem offset_placement (code bias d b x : Int) :
    weight code bias d b * x = (code + bias) * d * x + b * x := by
  unfold weight; rw [Int.add_mul]

-- § group shift

/-- The host loop computing `group_shift` (with fuel). -/
def gsLoop (ge : Nat) : Nat → Nat → Nat
  | 0, s => s
  | fuel + 1, s => if 2 ^ s < ge then gsLoop ge fuel (s + 1) else s

/-- Started at or below `k` with enough fuel, the loop stops exactly at `k` when `ge = 2^k`. -/
theorem gsLoop_pow (k : Nat) : ∀ fuel s, s ≤ k → k - s ≤ fuel → gsLoop (2 ^ k) fuel s = k
  | 0, s, h1, h2 => by
    have : s = k := by omega
    subst this; rfl
  | fuel + 1, s, h1, h2 => by
    simp only [gsLoop]
    by_cases hs : s < k
    · have hlt : 2 ^ s < 2 ^ k := (Nat.pow_lt_pow_iff_right (by decide)).2 hs
      rw [if_pos hlt]
      exact gsLoop_pow k fuel (s + 1) (by omega) (by omega)
    · have heq : s = k := by omega
      subst heq
      simp

/-- For every power-of-two group size up to `2^31` the host computes its exact logarithm. -/
theorem groupShift_exact (k : Nat) (hk : k ≤ 31) : gsLoop (2 ^ k) 32 0 = k :=
  gsLoop_pow k 32 0 (by omega) (by omega)

/-- The defined group sizes: 16, 32, 64 give shifts 4, 5, 6. -/
theorem groupShift_values : gsLoop 16 32 0 = 4 ∧ gsLoop 32 32 0 = 5 ∧ gsLoop 64 32 0 = 6 := by decide

/-- Once `1 << group_shift = group_elems` is confirmed (the Q8_K split guard), the shift IS the
division the naive kernel performs. -/
theorem shift_is_div (e s ge : Nat) (h : 1 <<< s = ge) : e >>> s = e / ge := by
  rw [Nat.shiftRight_eq_div_pow, ← h, Nat.shiftLeft_eq, Nat.one_mul]

-- § host guards

/-- The power-of-two test `(g & (g-1)) == 0` with `g > 0`. -/
def pow2Guard (g : Nat) : Bool := 0 < g && (g &&& (g - 1)) == 0

/-- Every power of two passes it. -/
theorem pow2Guard_pow (k : Nat) : pow2Guard (2 ^ k) = true := by
  have hpos : 0 < 2 ^ k := Nat.two_pow_pos k
  have hand : 2 ^ k &&& (2 ^ k - 1) = 0 := by rw [Nat.and_two_pow_sub_one_eq_mod, Nat.mod_self]
  simp [pow2Guard, hpos, hand]

/-- Non-powers of two are refused. -/
theorem pow2Guard_rejects : pow2Guard 0 = false ∧ pow2Guard 24 = false ∧ pow2Guard 48 = false := by decide

/-- Host guard of `s_gemv_q8_0_split` (group part). -/
def guardQ8_0Split (ge : Nat) : Bool := pow2Guard ge && ge % 4 == 0

/-- Host guard of `s_gemv_q8k_split` (group part). -/
def guardQ8KSplit (ge : Nat) : Bool := gsLoop ge 32 0 < 32 && 1 <<< gsLoop ge 32 0 == ge && ge % 16 == 0

/-- With the Q8_K split guard, sixteen consecutive elements from a 16-aligned start share one group, so the
kernel's single scale load per lane-iteration is exact. -/
theorem sixteen_in_one_group (i k ge : Nat) (hi : i % 16 = 0) (hk : k < 16) (hg : ge % 16 = 0)
    (hpos : 0 < ge) : (i + k) / ge = i / ge := by
  obtain ⟨m, rfl⟩ : ∃ m, ge = 16 * m := ⟨ge / 16, by omega⟩
  rw [← Nat.div_div_eq_div_mul, ← Nat.div_div_eq_div_mul]
  have : (i + k) / 16 = i / 16 := by omega
  rw [this]

/-- **GUARD GAP in `s_gemv_q8_0_split`.**  The guard admits `group_elems = 4` (and 8): a power of two and
a multiple of 4.  But the kernel takes ONE scale per 16-element lane-iteration (`QE = 16`), and with groups
of 4 element 4 already belongs to group 1.  (The kernel's own comment says "the launcher asserts
`group_elems % 8 == 0`"; this launcher asserts `% 4`.)  No real pack has such groups (16/32/64), so this is
latent, but the guard does not establish what the kernel assumes; `% 16`, as in `s_gemv_q8k_split`, would. -/
theorem q8_0_split_guard_gap :
    guardQ8_0Split 4 = true ∧ (0 + 4) / 4 ≠ 0 / 4 ∧
    guardQ8_0Split 8 = true ∧ (0 + 8) / 8 ≠ 0 / 8 := by decide

/-- The Q8_K split guard accepts exactly the defined sizes among these and rejects the ones the Q8_0 guard
lets through. -/
theorem q8k_split_guard_values :
    guardQ8KSplit 16 = true ∧ guardQ8KSplit 32 = true ∧ guardQ8KSplit 64 = true ∧
    guardQ8KSplit 4 = false ∧ guardQ8KSplit 8 = false ∧ guardQ8KSplit 48 = false := by decide

/-- FP16 split kernel (QE = 4): four consecutive elements from a 4-aligned start share a group when
`group_elems % 4 = 0` — its `% 4` guard is the right one for it. -/
theorem quad_in_one_group (i k ge : Nat) (hi : i % 4 = 0) (hk : k < 4) (hg : ge % 4 = 0)
    (hpos : 0 < ge) : (i + k) / ge = i / ge := by
  obtain ⟨m, rfl⟩ : ∃ m, ge = 4 * m := ⟨ge / 4, by omega⟩
  rw [← Nat.div_div_eq_div_mul, ← Nat.div_div_eq_div_mul]
  have : (i + k) / 4 = i / 4 := by omega
  rw [this]

/-- Q8_K activation contract: `n_in % 256 = 0`; the 640-wide shared-expert down projection is a
multiple of 32 but not 256, so only `s_gemv_q8_0_split` can serve it. -/
theorem q8k_contract_640 : 640 % 256 ≠ 0 ∧ 640 % 32 = 0 := by decide

-- § Q8 split kernel: lane schedule and activation hoisting

/-- Start of lane `l`'s `t`-th iteration: `16 l + 512 t`. -/
def laneStart (l t : Nat) : Nat := 16 * l + 512 * t

/-- Every element `e` belongs to lane `(e/16) % 32`, iteration `e / 512`, slot `e % 16`. -/
theorem lane_schedule_covers (e : Nat) :
    laneStart ((e / 16) % 32) (e / 512) + e % 16 = e ∧ (e / 16) % 32 < 32 ∧ e % 16 < 16 := by
  unfold laneStart; omega

/-- **Hoisting is exact**: the 16 elements of any lane-iteration lie inside ONE Q8_K block and ONE Q8_0
block, at consecutive offsets — so one `/` and one `%` per iteration replace sixteen of each. -/
theorem hoist_exact (l t k : Nat) (hk : k < 16) :
    let i := laneStart l t
    (i + k) / 256 = i / 256 ∧ (i + k) % 256 = i % 256 + k ∧
    (i + k) / 32 = i / 32 ∧ (i + k) % 32 = i % 32 + k := by
  simp only [laneStart]; omega

-- § bit order of the word loads

/-- S2 (`QW = 1`): element `i + 4m + k` (with `16 | i`) is byte `i/4 + m`, bit shift `2k`. -/
theorem s2_octet_index (i m k : Nat) (hi : i % 16 = 0) (hm : m < 4) (hk : k < 4) :
    (i + 4 * m + k) / 4 = i / 4 + m ∧ ((i + 4 * m + k) % 4) * 2 = 2 * k := by omega

/-- S4 (`QW = 2`): element `i + 4m + k` is byte `i/2 + 2m + k/2`, shift `4·(k % 2)` — i.e. inside the
`m`-th 16-bit word at `cp + 2m`. -/
theorem s4_octet_index (i m k : Nat) (hi : i % 16 = 0) (hm : m < 4) (hk : k < 4) :
    (i + 4 * m + k) / 2 = i / 2 + 2 * m + k / 2 ∧ ((i + 4 * m + k) % 2) * 4 = 4 * (k % 2) := by omega

/-- S8 (`QW = 4`): element `i + 4m + k` is byte `i + 4m + k` — the `k`-th byte of the `m`-th 32-bit word. -/
theorem s8_octet_index (i m k : Nat) (hm : m < 4) (hk : k < 4) :
    (i + 4 * m + k) / 1 = i + 4 * m + k ∧ ((i + 4 * m + k) % 1) * 8 = 0 := by omega

/-- S4 word extraction: `(v >> 4k) & 15` of the little-endian word `b₀ + 256 b₁` is the per-element
`(c[e/2] >> ((e%2)*4)) & 15`. -/
theorem s4_word_extract (b₀ b₁ k : Nat) (h₀ : b₀ < 256) (hk : k < 4) :
    ((b₀ + 256 * b₁) >>> (4 * k)) &&& 15 =
      ((if k / 2 = 0 then b₀ else b₁) >>> (4 * (k % 2))) &&& 15 := by
  have hk' : k = 0 ∨ k = 1 ∨ k = 2 ∨ k = 3 := by omega
  rcases hk' with rfl | rfl | rfl | rfl <;> simp [Nat.shiftRight_eq_div_pow, and15] <;> omega

/-- S8 word extraction: `(v >> 8k) & 255` of `b₀ + 2^8 b₁ + 2^16 b₂ + 2^24 b₃` is byte `k`. -/
theorem s8_word_extract (b₀ b₁ b₂ b₃ k : Nat) (h₀ : b₀ < 256) (h₁ : b₁ < 256) (h₂ : b₂ < 256)
    (h₃ : b₃ < 256) (hk : k < 4) :
    ((b₀ + 256 * b₁ + 65536 * b₂ + 16777216 * b₃) >>> (8 * k)) &&& 255 = [b₀, b₁, b₂, b₃].getD k 0 := by
  have hk' : k = 0 ∨ k = 1 ∨ k = 2 ∨ k = 3 := by omega
  rcases hk' with rfl | rfl | rfl | rfl <;> simp [Nat.shiftRight_eq_div_pow, and255] <;> omega

/-- S2 byte extraction: `(v >> 2k) & 3` of a byte is base-4 digit `k`. -/
theorem s2_byte_extract (b k : Nat) (hk : k < 4) :
    (b >>> (2 * k)) &&& 3 = [b % 4, b / 4 % 4, b / 16 % 4, b / 64 % 4].getD k 0 := by
  have hk' : k = 0 ∨ k = 1 ∨ k = 2 ∨ k = 3 := by omega
  rcases hk' with rfl | rfl | rfl | rfl <;> simp [Nat.shiftRight_eq_div_pow, and3] <;> omega

-- § codebook

/-- `kIq4Nl` is strictly increasing and inside int8; IQ4_NL indexing masks to the 16 entries. -/
theorem iq4nl_codebook :
    let t : List Int := [-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113]
    t.length = 16 ∧ (∀ i : Fin 15, t.getD i.val 0 < t.getD (i.val + 1) 0) ∧ (∀ c : Nat, c &&& 15 < 16) := by
  refine ⟨by decide, by decide, ?_⟩
  intro c; rw [and15]; exact Nat.mod_lt _ (by decide)

-- § launch shapes

/-- FP16 split: one block per row, `tpr ∈ [1, 1024]` a power of two, `tpr · 4` bytes of partials. -/
theorem split_tpr_smem (k : Nat) (hk : k ≤ 10) : pow2Guard (2 ^ k) = true ∧ 2 ^ k * 4 ≤ 4096 := by
  refine ⟨pow2Guard_pow k, ?_⟩
  have : 2 ^ k ≤ 2 ^ 10 := Nat.pow_le_pow_right (by decide) hk
  simp at this; omega

/-- Warp-per-row Q8 launches: `⌈n_out/8⌉` blocks of 256 threads reach every row. -/
theorem q8_split_rows_covered (n_out o : Nat) (ho : o < n_out) :
    o / 8 < (n_out + 7) / 8 ∧ (o / 8) * 8 + o % 8 = o := by omega

end Strata.SGemv
