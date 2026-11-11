/-!
# MODULE: ggml_block_dequant
SOURCES: src/kernels/cuda/dequant_bf16.cu
Dequantisation of ggml block formats (Q2_0, Q4_0, Q5_0, Q8_0, IQ4_NL, Q3_K, Q4_K, Q5_K, Q6_K, IQ4_XS)
to BF16 / FP16 / FP32, one thread per 32-element group, with the i-quant types routed elsewhere.

## Source semantics
* `geometry(type)` gives `(block_elems, block_bytes)` for the ten supported ggml types and refuses all
  others; `dequant_bf16_supported` is exactly `geometry`'s success.
* `launch` refuses unless the type is supported, `cols % block_elems == 0` and `rows > 0`; then
  `row_bytes = cols / block_elems * block_bytes`, `gpr = cols / 32`, one thread per group.
* Thread `g` handles row `r = g / gpr`, group `gi = g % gpr`, and writes outputs
  `[r*gpr*32 + gi*32, +32)`, i.e. `[32g, 32g+32)` — contiguous and disjoint.
* Each group reads only inside its block, and its block lies inside its row.
* Decoded integer factors: Q4_0 `nib-8 ∈ [-8,7]`; Q5_0 `(nib|xh)-16 ∈ [-16,15]`; Q6_K `q-32 ∈ [-32,31]`;
  K-quant 6-bit scales/mins `< 64`.
* `f2bf`: `(u + 0x7fff + ((u>>16)&1)) >> 16` — round to nearest, ties to even, on the bit pattern.
* `iq_only` types (16,17,18,21,22,29) are never in `geometry`; FP16/FP32 outputs route them to the
  i-quant kernels, BF16 output refuses them.

## Form template identifiers
* `geometry` ↔ `geometry(int, int&, int&)`;  `iqOnly` ↔ `iq_only`
* `launchOk`, `rowBytes`, `gpr` ↔ the checks and sizes in `launch<T>`
* `f2bf` ↔ `__device__ uint16_t f2bf(float)` on the raw bits
* `scaleMinK4` ↔ `scale_min_k4`
-/

namespace Strata.Dequant

-- § bit helpers

/-- `x & 3 = x mod 4`. -/
theorem and3 (x : Nat) : x &&& 3 = x % 4 := by simpa using Nat.and_two_pow_sub_one_eq_mod x 2
/-- `x & 15 = x mod 16`. -/
theorem and15 (x : Nat) : x &&& 15 = x % 16 := by simpa using Nat.and_two_pow_sub_one_eq_mod x 4
/-- `x & 63 = x mod 64`. -/
theorem and63 (x : Nat) : x &&& 63 = x % 64 := by simpa using Nat.and_two_pow_sub_one_eq_mod x 6

-- § geometry table

/-- `(block_elems, block_bytes)` per ggml type id. -/
def geometry : Nat → Option (Nat × Nat)
  | 2 => some (32, 18)
  | 6 => some (32, 22)
  | 8 => some (32, 34)
  | 20 => some (32, 18)
  | 11 => some (256, 110)
  | 12 => some (256, 144)
  | 13 => some (256, 176)
  | 14 => some (256, 210)
  | 23 => some (256, 136)
  | 42 => some (64, 18)
  | _ => none

/-- The supported ids, in the order of the `launch` switch. -/
def supported : List Nat := [2, 6, 8, 11, 12, 13, 14, 20, 23, 42]

/-- Every supported id has a geometry. -/
theorem supported_have_geometry : ∀ t ∈ supported, (geometry t).isSome = true := by decide

/-- Every geometry is a whole number of 32-element groups, with a positive byte size — so the
one-thread-per-group decomposition never splits a group across a block boundary. -/
theorem geometry_groups (t be bb : Nat) (h : geometry t = some (be, bb)) :
    be % 32 = 0 ∧ 0 < be ∧ 0 < bb := by
  unfold geometry at h
  split at h <;> simp at h <;> omega

/-- Bits per weight of each format, as `block_bytes * 8 * 32 = bpw*32 * block_elems` (bpw scaled by 32):
Q4_0 4.5, Q5_0 5.5, Q8_0 8.5, IQ4_NL 4.5, Q3_K 3.4375, Q4_K 4.5, Q5_K 5.5, Q6_K 6.5625, IQ4_XS 4.25,
Q2_0 2.25. -/
theorem bits_per_weight :
    geometry 2 = some (32, 18) ∧ 18 * 8 * 32 = 144 * 32 ∧
    geometry 11 = some (256, 110) ∧ 110 * 8 * 32 = 110 * 256 ∧
    geometry 14 = some (256, 210) ∧ 210 * 8 * 32 = 210 * 256 ∧
    geometry 42 = some (64, 18) ∧ 18 * 8 * 32 = 72 * 64 := by decide

/-- The i-quant ids handled by `iq_kernels.cu`. -/
def iqOnly (t : Nat) : Bool := t ∈ [16, 17, 18, 21, 22, 29]

/-- The i-quant ids and the block-dequant ids are disjoint: no type is decoded by both paths, and
`dequant_bf16` (which has no i-quant branch) refuses them through `geometry`. -/
theorem iqOnly_unsupported (t : Nat) (h : iqOnly t = true) : geometry t = none := by
  simp [iqOnly] at h
  rcases h with rfl | rfl | rfl | rfl | rfl | rfl <;> rfl

-- § launch geometry and thread indexing

/-- The host-side acceptance test of `launch`. -/
def launchOk (t rows cols : Nat) : Prop :=
  ∃ be bb, geometry t = some (be, bb) ∧ cols % be = 0 ∧ 0 < rows

/-- Groups per row. -/
def gpr (cols : Nat) : Nat := cols / 32

/-- Bytes per row. -/
def rowBytes (cols be bb : Nat) : Nat := cols / be * bb

/-- An accepted shape has an integral number of groups per row: `32 * gpr = cols`. -/
theorem launch_groups_exact (t rows cols : Nat) (h : launchOk t rows cols) : 32 * gpr cols = cols := by
  obtain ⟨be, bb, hg, hc, _⟩ := h
  obtain ⟨h32, hpos, _⟩ := geometry_groups t be bb hg
  have h1 : cols = be * (cols / be) := (Nat.mul_div_cancel' (Nat.dvd_of_mod_eq_zero hc)).symm
  have h2 : be = 32 * (be / 32) := (Nat.mul_div_cancel' (Nat.dvd_of_mod_eq_zero h32)).symm
  have h3 : cols % 32 = 0 := by
    rw [h1, h2, Nat.mul_assoc]; exact Nat.mul_mod_right 32 _
  unfold gpr; omega

/-- Thread `g` writes exactly the 32 outputs starting at `32 g`: row-major groups are contiguous and
distinct threads never write the same output. -/
theorem output_offset (g gp : Nat) : (g / gp) * gp * 32 + (g % gp) * 32 = 32 * g := by
  have h := Nat.div_add_mod g gp
  generalize g / gp = q at *
  generalize g % gp = m at *
  rw [← h, Nat.mul_comm gp q]
  omega

/-- Threads below `rows * gpr` address a valid row and group. -/
theorem thread_in_range (g rows gp : Nat) (hg : g < rows * gp) :
    g / gp < rows ∧ g % gp < gp := by
  have hgp : 0 < gp := by
    rcases Nat.eq_zero_or_pos gp with h | h
    · subst h; simp at hg
    · exact h
  exact ⟨(Nat.div_lt_iff_lt_mul hgp).2 hg, Nat.mod_lt _ hgp⟩

/-- Grid `⌈total/256⌉` of 256 threads reaches every group. -/
theorem grid_covers (total g : Nat) (h : g < total) : g / 256 < (total + 255) / 256 := by omega

-- § every group reads inside its own block, and the block inside its row

/-- 32-element formats (Q4_0, Q5_0, Q8_0, IQ4_NL): group `gi` is block `gi`. -/
theorem block32_in_row (gi cols bb : Nat) (hc : cols % 32 = 0) (hg : gi < cols / 32) :
    gi * bb + bb ≤ cols / 32 * bb := by
  have : gi + 1 ≤ cols / 32 := by omega
  calc gi * bb + bb = (gi + 1) * bb := by rw [Nat.add_mul, Nat.one_mul]
    _ ≤ cols / 32 * bb := Nat.mul_le_mul_right bb this

/-- Q2_0 (64 elements per 18-byte block): group `gi` is half of block `gi/2`. -/
theorem q2_0_block_in_row (gi cols : Nat) (hc : cols % 64 = 0) (hg : gi < cols / 32) :
    (gi / 2) * 18 + 18 ≤ cols / 64 * 18 := by omega

/-- 256-element K-quant formats: group `gi` is one eighth of block `gi/8`. -/
theorem kquant_block_in_row (gi cols bb : Nat) (hc : cols % 256 = 0) (hg : gi < cols / 32) :
    (gi / 8) * bb + bb ≤ cols / 256 * bb := by
  have : gi / 8 + 1 ≤ cols / 256 := by omega
  calc (gi / 8) * bb + bb = (gi / 8 + 1) * bb := by rw [Nat.add_mul, Nat.one_mul]
    _ ≤ cols / 256 * bb := Nat.mul_le_mul_right bb this

/-- In-block byte offsets read by each decoder stay below the block size. -/
theorem in_block_offsets :
    (∀ e, e < 64 → 2 + e / 4 < 18) ∧                                   -- Q2_0 codes
    (∀ j, j < 16 → 2 + j < 18 ∧ 6 + j < 22) ∧                          -- Q4_0 / IQ4_NL, Q5_0
    (∀ j, j < 32 → 2 + j < 34) ∧                                       -- Q8_0
    (∀ n t, n < 2 → t < 32 → 32 + n * 32 + t < 96 ∧ t < 32) ∧          -- Q3_K qs, hmask
    (∀ j l, j < 4 → l < 32 → 16 + 32 * j + l < 144) ∧                  -- Q4_K qs
    (∀ j l, j < 4 → l < 32 → 48 + 32 * j + l < 176 ∧ 16 + l < 48) ∧    -- Q5_K qs, qh
    (∀ n l, n < 2 → l < 32 → 64 * n + l + 32 < 128 ∧ 128 + 32 * n + l < 192) ∧  -- Q6_K ql, qh
    (∀ n i q, n < 2 → i < 2 → q < 4 → 192 + 8 * n + (i + 2 * q) < 208) ∧       -- Q6_K scales
    (∀ ib j, ib < 8 → j < 16 → 8 + 16 * ib + j < 136 ∧ 4 + ib / 2 < 8) := by  -- IQ4_XS
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩ <;> intros <;> omega

-- § decoded integer ranges

/-- Q4_0: a nibble minus 8 lies in `[-8, 7]`. -/
theorem q4_0_range (b : Nat) : -8 ≤ ((b &&& 15 : Nat) : Int) - 8 ∧ ((b &&& 15 : Nat) : Int) - 8 ≤ 7 := by
  rw [and15]; omega

/-- Q5_0: the fifth bit `xh ≤ 16` combined by OR with a nibble gives a 5-bit value, so the decoded
factor lies in `[-16, 15]`. -/
theorem q5_0_range (nib qh : Nat) (hn : nib < 16) :
    -16 ≤ (((nib ||| (qh &&& 16)) : Nat) : Int) - 16 ∧ (((nib ||| (qh &&& 16)) : Nat) : Int) - 16 ≤ 15 := by
  have hx : qh &&& 16 ≤ 16 := Nat.and_le_right
  have hor : nib ||| (qh &&& 16) < 2 ^ 5 := Nat.or_lt_two_pow (by omega) (by omega)
  have : nib ||| (qh &&& 16) < 32 := by simpa using hor
  omega

/-- `scale_min_k4`. -/
def scaleMinK4 (j : Nat) (q : Nat → Nat) : Nat × Nat :=
  if j < 4 then (q j &&& 63, q (j + 4) &&& 63)
  else ((q (j + 4) &&& 15) ||| ((q (j - 4) >>> 6) <<< 4), (q (j + 4) >>> 4) ||| ((q j >>> 6) <<< 4))

/-- K-quant scales and mins are 6-bit (`< 64`) for every sub-block, given byte inputs. -/
theorem scaleMinK4_six_bit (j : Nat) (q : Nat → Nat) (hq : ∀ i, q i < 256) :
    (scaleMinK4 j q).1 < 64 ∧ (scaleMinK4 j q).2 < 64 := by
  have hi : ∀ y, y < 256 → (y >>> 6) <<< 4 < 2 ^ 6 := by
    intro y hy
    rw [Nat.shiftRight_eq_div_pow, Nat.shiftLeft_eq]; simp; omega
  unfold scaleMinK4
  by_cases hj : j < 4
  · simp only [hj, if_true, and63]; constructor <;> omega
  · simp only [hj, if_false]
    have h1 : q (j + 4) &&& 15 < 2 ^ 6 := by rw [and15]; simp; omega
    have h2 : q (j + 4) >>> 4 < 2 ^ 6 := by
      rw [Nat.shiftRight_eq_div_pow]; have := hq (j + 4); simp; omega
    exact ⟨by simpa using Nat.or_lt_two_pow h1 (hi _ (hq _)),
           by simpa using Nat.or_lt_two_pow h2 (hi _ (hq _))⟩

/-- Q6_K: low nibble OR two high bits shifted by 4 is a 6-bit code, so `q - 32 ∈ [-32, 31]`. -/
theorem q6_K_range (ql qh s : Nat) :
    -32 ≤ ((((ql &&& 15) ||| (((qh >>> s) &&& 3) <<< 4)) : Nat) : Int) - 32 ∧
    ((((ql &&& 15) ||| (((qh >>> s) &&& 3) <<< 4)) : Nat) : Int) - 32 ≤ 31 := by
  have h1 : ql &&& 15 < 2 ^ 6 := by rw [and15]; simp; omega
  have h2 : ((qh >>> s) &&& 3) <<< 4 < 2 ^ 6 := by
    rw [and3, Nat.shiftLeft_eq]; simp; omega
  have h := Nat.or_lt_two_pow h1 h2
  simp at h
  omega

/-- `kvalues_iq4nl` is strictly increasing and fits int8 — a valid non-linear codebook. -/
theorem iq4nl_table :
    let t : List Int := [-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113]
    t.length = 16 ∧ (∀ i : Fin 15, t.getD i.val 0 < t.getD (i.val + 1) 0) ∧
      (∀ v ∈ t, -128 ≤ v ∧ v ≤ 127) := by
  decide

-- § BF16 rounding

/-- `f2bf` on the raw FP32 bit pattern `u` (no wrap; NaN payloads aside). -/
def f2bf (u : Nat) : Nat := (u + 0x7fff + ((u >>> 16) &&& 1)) >>> 16

/-- **Round to nearest, ties to even.**  With `u = 2^16 q + r`: below the half-way point the high half
is kept, above it it is incremented, and exactly at it the result is the even one of `q, q+1`. -/
theorem f2bf_rne (u : Nat) :
    let q := u / 65536
    let r := u % 65536
    (r < 32768 → f2bf u = q) ∧ (32768 < r → f2bf u = q + 1) ∧
    (r = 32768 → f2bf u = q + q % 2 ∧ (q + q % 2) % 2 = 0) := by
  have hand : (u >>> 16) &&& 1 = u / 65536 % 2 := by
    rw [Nat.shiftRight_eq_div_pow]; simpa using Nat.and_two_pow_sub_one_eq_mod (u / 2 ^ 16) 1
  simp only [f2bf, hand, Nat.shiftRight_eq_div_pow]
  refine ⟨?_, ?_, ?_⟩ <;> intro h <;> simp <;> omega

end Strata.Dequant
