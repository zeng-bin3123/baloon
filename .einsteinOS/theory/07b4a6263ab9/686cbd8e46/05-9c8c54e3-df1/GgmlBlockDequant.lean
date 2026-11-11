/-!
# MODULE: ggml_block_dequant
SOURCES: src/kernels/cuda/dequant_bf16.cu
Row-slice dequantization of ggml block formats (Q2_0, Q4_0, Q5_0, Q8_0, IQ4_NL, Q3_K, Q4_K, Q5_K, Q6_K,
IQ4_XS) to BF16 / FP16 / FP32, one thread per 32-element group.  This module models the format geometry table,
the group→block addressing, the bit-level unpacking rules, the round-to-nearest-even FP32→BF16 conversion, and
the dispatch to the i-quant path.

## Source semantics
* `geometry(type)` gives `(block_elems, block_bytes)`; unsupported types make `launch` exit.
* `launch` requires `cols % block_elems = 0`, `rows > 0`; `row_bytes = cols/block_elems*block_bytes`,
  `groups_per_row = cols/32`, one thread per group, output group `g` of row `r` at `r*gpr*32 + g*32`.
* `f2bf(u) = (u + 0x7fff + ((u >> 16) & 1)) >> 16` on the FP32 bit pattern: nearest, ties to even, NO NaN guard.
* Q4_0/IQ4_NL: element `j` = low nibble of byte `j`, element `j+16` = high nibble.
  Q5_0: fifth bit of element `j` is bit `j` of `qh`, of element `j+16` bit `j+16`.
  Q6_K: `q = low4 | high2 << 4 ∈ [0,64)`, decoded `q - 32`.  K-scale unpack `scale_min_k4` yields 6-bit values.
* `dequant_f16/f32` route the i-quant types {16,17,18,21,22,29} to `iq_kernels`; `dequant_bf16` does not.

## Form template identifiers
`geometry` → `geometry`; `dequant_bf16_supported` → `supported`; `iq_only` → `iqOnly`; `f2bf` → `f2bf`;
`kv_iq4nl` → `iq4nl`; `scale_min_k4` → `scaleMinK4`; per-group block index → `blockOfGroup`.
-/

namespace Strata.GgmlBlockDequant

-- § format geometry

/-- `geometry(type)`: `(block_elems, block_bytes)` for the supported ggml type ids. -/
def geometry : Nat → Option (Nat × Nat)
  | 2 => some (32, 18)     -- Q4_0
  | 6 => some (32, 22)     -- Q5_0
  | 8 => some (32, 34)     -- Q8_0
  | 20 => some (32, 18)    -- IQ4_NL
  | 11 => some (256, 110)  -- Q3_K
  | 12 => some (256, 144)  -- Q4_K
  | 13 => some (256, 176)  -- Q5_K
  | 14 => some (256, 210)  -- Q6_K
  | 23 => some (256, 136)  -- IQ4_XS
  | 42 => some (64, 18)    -- Q2_0
  | _ => none

/-- The supported type ids. -/
def supportedTypes : List Nat := [2, 6, 8, 20, 11, 12, 13, 14, 23, 42]

/-- `dequant_bf16_supported`. -/
def supported (t : Nat) : Bool := (geometry t).isSome

/-- The struct layout each block size is derived from (field byte counts, summed). -/
def layoutBytes : Nat → Nat
  | 2 => 2 + 16                 -- d, qs[16]
  | 6 => 2 + 4 + 16             -- d, qh[4], qs[16]
  | 8 => 2 + 32                 -- d, qs[32]
  | 20 => 2 + 16                -- d, qs[16]
  | 11 => 32 + 64 + 12 + 2      -- hmask, qs, scales, d
  | 12 => 2 + 2 + 12 + 128      -- d, dmin, scales, qs
  | 13 => 2 + 2 + 12 + 32 + 128 -- d, dmin, scales, qh, qs
  | 14 => 128 + 64 + 16 + 2     -- ql, qh, scales, d
  | 23 => 2 + 2 + 4 + 128       -- d, scales_h, scales_l, qs
  | 42 => 2 + 16                -- d, qs[16] (64 two-bit codes)
  | _ => 0

/-- Block byte sizes agree with the field layouts of the ggml structs. -/
theorem geometry_matches_layout :
    ∀ t ∈ supportedTypes, (geometry t).map Prod.snd = some (layoutBytes t) := by
  decide

/-- `supported` holds exactly on the listed types. -/
theorem supported_iff (t : Nat) : supported t = true ↔ t ∈ supportedTypes := by
  unfold supported geometry supportedTypes
  split <;> simp_all

/-- Every block holds a whole number of 32-element groups, so the thread-per-group grid is exact. -/
theorem blocks_whole_groups (t e b : Nat) (h : geometry t = some (e, b)) : e % 32 = 0 := by
  unfold geometry at h
  split at h <;> simp at h <;> omega

/-- Groups per block (`block_elems / 32`): 1 for the 32-blocks, 2 for Q2_0, 8 for the K-quants. -/
def groupsPerBlock (e : Nat) : Nat := e / 32

/-- The block holding group `gi` of a row and the group's index inside it. -/
def blockOfGroup (e gi : Nat) : Nat × Nat := (gi / groupsPerBlock e, gi % groupsPerBlock e)

/-- The kernel's hard-coded indices are the generic ones: `gi/2, gi%2` for Q2_0 and `gi/8, gi%8` for K. -/
theorem kernel_block_indices (gi : Nat) :
    blockOfGroup 64 gi = (gi / 2, gi % 2) ∧ blockOfGroup 256 gi = (gi / 8, gi % 8) ∧
    blockOfGroup 32 gi = (gi, 0) := by
  simp [blockOfGroup, groupsPerBlock, Nat.mod_one]

/-- Group `gi`'s 32 elements lie inside the block the kernel reads (for 2 and 8 groups per block). -/
theorem group_inside_block (gi j : Nat) (hj : j < 32) :
    (gi / 2) * 64 ≤ gi * 32 + j ∧ gi * 32 + j < (gi / 2) * 64 + 64 ∧
    (gi / 8) * 256 ≤ gi * 32 + j ∧ gi * 32 + j < (gi / 8) * 256 + 256 := by
  omega

/-- `launch`'s shape gate. -/
def shapeOk (t rows cols : Nat) : Prop :=
  ∃ e b, geometry t = some (e, b) ∧ cols % e = 0 ∧ 0 < rows

/-- Under the gate, `groups_per_row * 32 = cols`: the output tile is exactly `rows × cols`. -/
theorem gpr_exact (t rows cols : Nat) (h : shapeOk t rows cols) : cols / 32 * 32 = cols := by
  obtain ⟨e, b, hg, hc, _⟩ := h
  have he := blocks_whole_groups t e b hg
  have h32 : cols % 32 = 0 :=
    Nat.mod_eq_zero_of_dvd (Nat.dvd_trans (Nat.dvd_of_mod_eq_zero he) (Nat.dvd_of_mod_eq_zero hc))
  omega

/-- Output addressing `r*gpr*32 + gi*32 + j` is injective over the tile. -/
theorem out_index_inj (gpr r1 g1 j1 r2 g2 j2 : Nat) (hg1 : g1 < gpr) (hg2 : g2 < gpr)
    (hj1 : j1 < 32) (hj2 : j2 < 32)
    (h : r1 * gpr * 32 + g1 * 32 + j1 = r2 * gpr * 32 + g2 * 32 + j2) :
    r1 = r2 ∧ g1 = g2 ∧ j1 = j2 := by
  have hj : j1 = j2 := by omega
  subst hj
  have h' : (r1 * gpr + g1) * 32 = (r2 * gpr + g2) * 32 := by
    rw [Nat.add_mul, Nat.add_mul]; omega
  have h2 : r1 * gpr + g1 = r2 * gpr + g2 := Nat.eq_of_mul_eq_mul_right (by decide) h'
  have hr : r1 = r2 := by
    rcases Nat.lt_trichotomy r1 r2 with hl | he | hl
    · have : (r1 + 1) * gpr ≤ r2 * gpr := Nat.mul_le_mul_right _ hl
      rw [Nat.add_mul, Nat.one_mul] at this; omega
    · exact he
    · have : (r2 + 1) * gpr ≤ r1 * gpr := Nat.mul_le_mul_right _ hl
      rw [Nat.add_mul, Nat.one_mul] at this; omega
  subst hr
  exact ⟨rfl, by omega, rfl⟩

-- § i-quant routing

/-- `iq_only`: types served by `iq_kernels`. -/
def iqOnly (t : Nat) : Bool := t == 16 || t == 17 || t == 18 || t == 21 || t == 22 || t == 29

/-- The i-quant set and the block-format table are disjoint, so the routing is unambiguous. -/
theorem iq_disjoint : ∀ t ∈ [16, 17, 18, 21, 22, 29], supported t = false := by decide

/-- `dequant_bf16` has no i-quant branch: an i-quant type is unsupported there and `launch` would exit, while
`dequant_f16`/`dequant_f32` serve it. -/
theorem bf16_lacks_iq : iqOnly 16 = true ∧ supported 16 = false := by decide

-- § FP32 → BF16 rounding

/-- `f2bf` on the FP32 bit pattern `u < 2^32` (the C arithmetic wraps mod `2^32`, then truncates to 16 bits). -/
def f2bf (u : Nat) : Nat := (u + 0x7fff + u / 65536 % 2) % 4294967296 / 65536 % 65536

/-- Away from the NaN/Inf top of the range neither the `2^32` wrap nor the 16-bit truncation fires. -/
theorem f2bf_nowrap (u : Nat) (hu : u < 4294901760) : f2bf u = (u + 0x7fff + u / 65536 % 2) / 65536 := by
  unfold f2bf
  rw [Nat.mod_eq_of_lt (by omega : u + 0x7fff + u / 65536 % 2 < 4294967296)]
  exact Nat.mod_eq_of_lt (by omega)

/-- Below the halfway point the low half is dropped (round down in magnitude). -/
theorem f2bf_down (u : Nat) (hu : u < 4294901760) (hlo : u % 65536 < 32768) : f2bf u = u / 65536 := by
  rw [f2bf_nowrap u hu]; omega

/-- Above the halfway point the result rounds up by one ulp. -/
theorem f2bf_up (u : Nat) (hu : u < 4294901760) (hlo : 32768 < u % 65536) : f2bf u = u / 65536 + 1 := by
  rw [f2bf_nowrap u hu]; omega

/-- **Ties go to even**: at exactly half an ulp the result's last bit is 0. -/
theorem f2bf_tie_even (u : Nat) (hu : u < 4294901760) (hlo : u % 65536 = 32768) : f2bf u % 2 = 0 := by
  rw [f2bf_nowrap u hu]
  have hq : (u + 0x7fff + u / 65536 % 2) / 65536 = u / 65536 + u / 65536 % 2 := by omega
  rw [hq]; omega

/-- **No NaN guard.**  The rounding add carries NaN payloads across the exponent: the FP32 NaN `0x7FFFFFFF`
becomes BF16 `0x8000` (−0.0).  Only a non-finite block scale could reach this, but the conversion itself
does not preserve NaN. -/
theorem f2bf_nan_to_negzero : f2bf 0x7FFFFFFF = 0x8000 := by decide

/-- A NaN whose payload sits only in the low half rounds to the infinity pattern `0x7F80`. -/
theorem f2bf_nan_to_inf : f2bf 0x7F800001 = 0x7F80 := by decide

-- § bit unpacking

/-- Shift-and-mask is division-and-modulo by powers of two (the identity every unpacker relies on). -/
theorem extract_eq (x s m : Nat) : (x >>> s) &&& (2 ^ m - 1) = x / 2 ^ s % 2 ^ m := by
  rw [Nat.shiftRight_eq_div_pow, Nat.and_two_pow_sub_one_eq_mod]

/-- A byte splits into its two nibbles, low nibble first. -/
theorem nibbles (b : Nat) (hb : b < 256) : b &&& 15 = b % 16 ∧ b >>> 4 = b / 16 ∧ b = b % 16 + 16 * (b / 16) ∧
    b / 16 < 16 := by
  have h1 : b &&& 15 = b % 16 := by
    have := Nat.and_two_pow_sub_one_eq_mod b 4; simpa using this
  have h2 : b >>> 4 = b / 16 := by rw [Nat.shiftRight_eq_div_pow]
  exact ⟨h1, h2, by omega, by omega⟩

/-- Q4_0 code range: `nibble - 8 ∈ [-8, 7]`. -/
theorem q4_0_range (n : Nat) (h : n < 16) : -8 ≤ (n : Int) - 8 ∧ (n : Int) - 8 ≤ 7 := by omega

/-- Testing bit 4 of any value: `a & 0x10 = 16 * bit₄(a)`. -/
theorem and16 (a : Nat) : a &&& 16 = 16 * (a / 16 % 2) := by
  have hsplit : a &&& 16 = (a &&& 31) &&& 16 := by
    rw [Nat.and_assoc, show (31 : Nat) &&& 16 = 16 by decide]
  have h31 : a &&& 31 = a % 32 := by
    have := Nat.and_two_pow_sub_one_eq_mod a 5; simpa using this
  have tbl : ∀ r : Fin 32, r.val &&& 16 = 16 * (r.val / 16 % 2) := by decide
  have hr := tbl ⟨a % 32, Nat.mod_lt _ (by decide)⟩
  simp only at hr
  rw [hsplit, h31, hr]
  omega

/-- Q5_0, low element `j`: `((qh >> j) << 4) & 0x10 = 16 * bit_j(qh)`. -/
theorem q5_0_high_bit_lo (qh j : Nat) : ((qh >>> j) <<< 4) &&& 16 = 16 * (qh / 2 ^ j % 2) := by
  rw [and16, Nat.shiftLeft_eq, Nat.shiftRight_eq_div_pow]
  have : qh / 2 ^ j * 2 ^ 4 / 16 = qh / 2 ^ j := by simp
  rw [this]

/-- Q5_0, high element `j+16`: `(qh >> (j+12)) & 0x10 = 16 * bit_{j+16}(qh)`. -/
theorem q5_0_high_bit_hi (qh j : Nat) : (qh >>> (j + 12)) &&& 16 = 16 * (qh / 2 ^ (j + 16) % 2) := by
  rw [and16, Nat.shiftRight_eq_div_pow, Nat.div_div_eq_div_mul,
    show (16 : Nat) = 2 ^ 4 from rfl, ← Nat.pow_add]

/-- Six-bit assembly `low4 | high2 << 4` is plain addition (Q6_K codes and K-quant scales). -/
theorem or_shift4 : ∀ a : Fin 16, ∀ h : Fin 4, (a.val ||| (h.val <<< 4)) = a.val + 16 * h.val := by
  decide

/-- Q6_K code: `q - 32 ∈ [-32, 31]`. -/
theorem q6_k_range (lo hi : Nat) (hl : lo < 16) (hh : hi < 4) :
    (lo ||| (hi <<< 4)) < 64 ∧ -32 ≤ ((lo ||| (hi <<< 4) : Nat) : Int) - 32 := by
  have := or_shift4 ⟨lo, hl⟩ ⟨hi, hh⟩
  simp only at this
  omega

/-- `scale_min_k4`. -/
def scaleMinK4 (j : Nat) (q : Nat → Nat) : Nat × Nat :=
  if j < 4 then (q j &&& 63, q (j + 4) &&& 63)
  else ((q (j + 4) &&& 15) ||| ((q (j - 4) >>> 6) <<< 4), (q (j + 4) >>> 4) ||| ((q j >>> 6) <<< 4))

/-- **K-scales are 6-bit** for bytes in range, in both halves of the 12-byte scale field. -/
theorem scaleMinK4_lt64 (j : Nat) (q : Nat → Nat) (hq : ∀ i, q i < 256) :
    (scaleMinK4 j q).1 < 64 ∧ (scaleMinK4 j q).2 < 64 := by
  have m63 : ∀ x, x &&& 63 < 64 := fun x => by
    have := Nat.and_two_pow_sub_one_eq_mod x 6; simp at this; rw [this]; exact Nat.mod_lt _ (by decide)
  have m15 : ∀ x, x &&& 15 < 16 := fun x => by
    have := Nat.and_two_pow_sub_one_eq_mod x 4; simp at this; rw [this]; exact Nat.mod_lt _ (by decide)
  have s6 : ∀ x, x < 256 → x >>> 6 < 4 := fun x hx => by rw [Nat.shiftRight_eq_div_pow]; simp; omega
  have s4 : ∀ x, x < 256 → x >>> 4 < 16 := fun x hx => by rw [Nat.shiftRight_eq_div_pow]; simp; omega
  unfold scaleMinK4
  by_cases hj : j < 4
  · simp only [hj, ite_true]; exact ⟨m63 _, m63 _⟩
  · simp only [hj, ite_false]
    have e1 := or_shift4 ⟨_, m15 (q (j + 4))⟩ ⟨_, s6 _ (hq (j - 4))⟩
    have e2 := or_shift4 ⟨_, s4 _ (hq (j + 4))⟩ ⟨_, s6 _ (hq j)⟩
    simp only at e1 e2
    have := m15 (q (j + 4)); have := s6 _ (hq (j - 4)); have := s4 _ (hq (j + 4)); have := s6 _ (hq j)
    omega

/-- Q3_K's high-mask bit `1 << (n*4 + jj)` (n < 2, jj < 4) fits in the `uint8_t` it is stored in. -/
theorem q3_k_hmask_fits : ∀ n : Fin 2, ∀ jj : Fin 4, 1 <<< (n.val * 4 + jj.val) < 256 := by decide

-- § the IQ4_NL codebook

/-- `kv_iq4nl`. -/
def iq4nl : List Int := [-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113]

/-- The codebook has 16 entries, is strictly increasing and lies in `int8`. -/
theorem iq4nl_shape : iq4nl.length = 16 ∧ iq4nl.Pairwise (· < ·) ∧ ∀ v ∈ iq4nl, -128 ≤ v ∧ v ≤ 127 := by
  decide

end Strata.GgmlBlockDequant
