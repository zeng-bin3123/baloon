/-!
# MODULE: cuda_dequant_bf16
Launch geometry and indexing contracts of the GPU dequantizer (one thread per 32-element group): the type
geometry table, row byte sizes, grid coverage, per-format group indexing bounds, the bf16 rounding rule, and the
split between ggml types handled here and the i-quant types routed elsewhere.

## Source semantics
* `geometry` gives (elements per block, bytes per block) for types 2, 6, 8, 11, 12, 13, 14, 20, 23, 42; every
  block has a multiple of 32 elements, so a row is `cols/32` groups.
* `row_bytes = cols / be * bb`; thread `g` maps to row `g / gpr` and group `g % gpr`.
* `grid = ceil(total / 256)` threads-blocks cover all `total` groups.
* Group `gi` of a K-quant row reads block `gi / 8`, sub-group `gi % 8`; of Q2_0 block `gi / 2`, half `gi % 2`.
* `f2bf(u) = (u + 0x7fff + ((u >> 16) & 1)) >> 16` in 32-bit unsigned arithmetic rounds to nearest even, but the
  addition wraps for `u ≥ 0xFFFF8000` (large-mantissa negative NaNs).
* The i-quant types {16, 17, 18, 21, 22, 29} are disjoint from the geometry table.

## Form template identifiers
geometry → block_geometry, group32 → group_dequant, dequant_kernel → per_group_thread, f2bf → float_to_bf16,
iq_only → iquant_types, launch → kernel_launch.
-/

namespace Strata.Cuda.DequantBf16

-- § geometry table

/-- (block elements, block bytes) per ggml type id. -/
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

/-- Every supported block has a multiple of 32 elements (so groups of 32 never straddle a block). -/
theorem geometry_multiple_of_32 (t be bb : Nat) (h : geometry t = some (be, bb)) : be % 32 = 0 := by
  unfold geometry at h
  split at h <;> simp at h <;> omega

/-- The i-quant types routed to `iq_dequant_*`. -/
def iqOnly (t : Nat) : Bool := t == 16 || t == 17 || t == 18 || t == 21 || t == 22 || t == 29

/-- No type is both a table type and an i-quant type, so the two dispatch paths never overlap. -/
theorem iq_and_geometry_disjoint (t : Nat) (h : iqOnly t = true) : geometry t = none := by
  simp only [iqOnly, Bool.or_eq_true, beq_iff_eq] at h
  rcases h with ((((h | h) | h) | h) | h) | h <;> subst h <;> rfl

-- § row and grid arithmetic

/-- Groups per row is `cols / 32`; with `be ∈ {32,64,256}` dividing `cols`, it equals blocks-per-row times
groups-per-block. -/
theorem groups_per_row_32 (cols : Nat) (h : cols % 32 = 0) : cols / 32 * 32 = cols := by omega

/-- Same for 64-element blocks (Q2_0). -/
theorem groups_per_row_64 (cols : Nat) (h : cols % 64 = 0) : cols / 32 = (cols / 64) * 2 := by omega

/-- Same for 256-element blocks (K-quants, IQ4_XS). -/
theorem groups_per_row_256 (cols : Nat) (h : cols % 256 = 0) : cols / 32 = (cols / 256) * 8 := by omega

/-- Grid size `(total + 255) / 256` launches at least `total` threads. -/
theorem grid_covers (total : Nat) : total ≤ (total + 255) / 256 * 256 := by omega

/-- And not more than one extra block of threads. -/
theorem grid_tight (total : Nat) : (total + 255) / 256 * 256 < total + 256 := by omega

/-- Thread-to-(row, group) decode: `g < rows * gpr` gives `row < rows` and `group < gpr`. -/
theorem thread_decode (g rows gpr : Nat) (hg : 0 < gpr) (h : g < rows * gpr) : g / gpr < rows ∧ g % gpr < gpr := by
  constructor
  · exact Nat.div_lt_of_lt_mul (by rw [Nat.mul_comm] at h; exact h)
  · exact Nat.mod_lt _ hg

/-- Each output group of 32 lies inside its row: `gi * 32 + j < gpr * 32` for `gi < gpr`, `j < 32`. -/
theorem output_in_row (gi gpr j : Nat) (hgi : gi < gpr) (hj : j < 32) : gi * 32 + j < gpr * 32 := by omega

-- § per-format group indexing

/-- Q2_0: block `gi/2` is inside the row, and the 32 elements start at 0 or 32 within the block. -/
theorem q2_0_group_bounds (cols gi j : Nat) (hc : cols % 64 = 0) (hg : gi < cols / 32) (hj : j < 32) :
    gi / 2 < cols / 64 ∧ (gi % 2) * 32 + j < 64 ∧ 2 + ((gi % 2) * 32 + j) / 4 < 18 := by omega

/-- Q3_K: block `gi/8` is in the row; qs at `32 + 32n + t` and scale index `8n + 2jj + (t≥16)` stay in range. -/
theorem q3_k_group_bounds (cols gi t : Nat) (hc : cols % 256 = 0) (hg : gi < cols / 32) (ht : t < 32) :
    gi / 8 < cols / 256 ∧ 32 + (gi % 8 / 4) * 32 + t < 96 ∧
      (gi % 8 / 4) * 8 + (gi % 8 % 4) * 2 + (if t ≥ 16 then 1 else 0) < 16 := by
  split <;> omega

/-- Q4_K: sub-group `gi%8` selects 64-group `j64 = (gi%8)/2` and a nibble half; reads `16 + 32 j64 + l < 144`. -/
theorem q4_k_group_bounds (cols gi l : Nat) (hc : cols % 256 = 0) (hg : gi < cols / 32) (hl : l < 32) :
    gi / 8 < cols / 256 ∧ 16 + 32 * (gi % 8 / 2) + l < 144 := by omega

/-- Q5_K: the high-bit mask `1 << (2 j64 + hi)` equals `1 << (gi % 8)`, a byte value. -/
theorem q5_k_mask_shift (gi : Nat) : 2 * (gi % 8 / 2) + gi % 8 % 2 = gi % 8 := by omega

/-- Q5_K reads: `qh` at `16 + l`, `ql` at `48 + 32 j64 + l`, both inside the 176-byte block. -/
theorem q5_k_group_bounds (gi l : Nat) (hl : l < 32) :
    16 + l < 48 ∧ 48 + 32 * (gi % 8 / 2) + l < 176 := by omega

/-- Q6_K: `ql` at `64 n + l (+32)`, `qh` at `128 + 32 n + l`, `sc` at `192 + 8n + l/16 + 2 qu`, `n = (gi%8)/4`,
`qu = (gi%8)%4`. -/
theorem q6_k_group_bounds (gi l : Nat) (hl : l < 32) :
    64 * (gi % 8 / 4) + l + 32 < 128 ∧ 128 + 32 * (gi % 8 / 4) + l < 192 ∧
      192 + 8 * (gi % 8 / 4) + l / 16 + 2 * (gi % 8 % 4) < 208 := by omega

/-- IQ4_XS: `qs` at `8 + 16 ib + j` stays inside the 136-byte block. -/
theorem iq4_xs_group_bounds (gi j : Nat) (hj : j < 16) : 8 + 16 * (gi % 8) + j < 136 := by omega

/-- Q4_0/Q8_0/Q5_0/IQ4_NL: one block per group, so `gi * bb + 2 + j` is inside the row for `gi < gpr`. -/
theorem single_group_block_bounds (gi gpr bb j : Nat) (hgi : gi < gpr) (hj : j + 2 < bb) :
    gi * bb + 2 + j < gpr * bb := by
  have h : (gi + 1) * bb ≤ gpr * bb := Nat.mul_le_mul_right bb hgi
  have h2 : (gi + 1) * bb = gi * bb + bb := by rw [Nat.add_mul, Nat.one_mul]
  omega

-- § bf16 rounding

/-- `f2bf` in 32-bit unsigned arithmetic (the addition wraps modulo 2^32). -/
def f2bf (u : Nat) : Nat := ((u + 32767 + (u / 65536) % 2) % 4294967296) / 65536

/-- Without wrap-around the result fits 16 bits. -/
theorem f2bf_fits (u : Nat) (h : u < 4294934528) : f2bf u ≤ 65535 := by
  unfold f2bf
  rw [Nat.mod_eq_of_lt (by omega)]
  omega

/-- Below half: truncate. -/
theorem f2bf_round_down (u : Nat) (h : u < 4294934528) (hr : u % 65536 < 32768) : f2bf u = u / 65536 := by
  unfold f2bf
  rw [Nat.mod_eq_of_lt (by omega)]
  omega

/-- Above half: round up. -/
theorem f2bf_round_up (u : Nat) (h : u < 4294934528) (hr : u % 65536 > 32768) : f2bf u = u / 65536 + 1 := by
  unfold f2bf
  rw [Nat.mod_eq_of_lt (by omega)]
  omega

/-- Exact tie, even kept bit: stay. -/
theorem f2bf_tie_even (u : Nat) (h : u < 4294934528) (hr : u % 65536 = 32768) (he : (u / 65536) % 2 = 0) :
    f2bf u = u / 65536 := by
  unfold f2bf
  rw [Nat.mod_eq_of_lt (by omega)]
  omega

/-- Exact tie, odd kept bit: round up to even. -/
theorem f2bf_tie_odd (u : Nat) (h : u < 4294934528) (hr : u % 65536 = 32768) (ho : (u / 65536) % 2 = 1) :
    f2bf u = u / 65536 + 1 := by
  unfold f2bf
  rw [Nat.mod_eq_of_lt (by omega)]
  omega

/-- The wrap hazard: the all-ones NaN pattern `0xFFFFFFFF` is converted to bf16 bits `0x0000` (+0.0), not a NaN.
Recorded as an observed property of the arithmetic as written. -/
theorem f2bf_all_ones_wraps_to_zero : f2bf 4294967295 = 0 := by decide

/-- A finite, ordinary value such as 1.0f (`0x3F800000`) converts exactly to bf16 `0x3F80`. -/
theorem f2bf_one : f2bf 1065353216 = 16256 := by decide

end Strata.Cuda.DequantBf16
