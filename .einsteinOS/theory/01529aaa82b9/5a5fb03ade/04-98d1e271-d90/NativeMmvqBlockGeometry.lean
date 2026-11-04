/-!
# MODULE: native_mmvq_block_geometry
Native Q8_1 matrix-vector kernels (Q2_0, Q3_K, Q4_K, Q5_K, Q6_K, IQ4_XS, Q4_0, Q5_0, Q8_0, IQ4_NL) adapted
from llama.cpp, plus the multi-column (ncols = 2..8) generic kernel, the Q8_1 quantizer and the host
validation / weight-size helpers.

## Source semantics
* Each weight block struct has a fixed size (`static_assert`s); rows are `n_in / block_elems` blocks.
* A 4 x 32 thread block advances `BPI` weight blocks per iteration where thread `tid` starts at block
  `tid / T`; every format satisfies `T * BPI = 128`, so all blocks are visited exactly once per row.
* "Small K" (`blocks_per_row < BPI`) uses 4 rows per block.
* Q4_0/Q5_0 affine correction constants `8 * vdr / QI` are 4 and 8.
* Multi-column activation images are column-contiguous with `n_in / 32` Q8_1 blocks per column.

## Form template identifiers
- `Q5KBlock`..`IQ4NLBlock`         → `weight_block_struct`
- `Q81Block`                        → `act_block_q8_1`
- `native_*_mmvq_kernel`            → `row_mmvq_kernel`
- `native_mmvq_multi_kernel`        → `multi_column_kernel`
- `native_mmvq_weight_bytes`        → `weight_byte_count`
- `native_quantize_q8_1_kernel`     → `quantize_act_kernel`
-/

namespace Strata.NativeMmvq

-- § struct sizes

/-- Sizes of the weight and activation block structs (mirrors the `static_assert`s). -/
theorem struct_sizes :
    4 + 12 + 32 + 128 = 176 ∧          -- Q5_K
    4 + 32 = 36 ∧                       -- Q8_1
    2 + 16 = 18 ∧                       -- Q2_0, Q4_0, IQ4_NL
    32 + 64 + 12 + 2 = 110 ∧            -- Q3_K
    2 + 2 + 4 + 128 = 136 ∧             -- IQ4_XS
    4 + 12 + 128 = 144 ∧                -- Q4_K
    128 + 64 + 16 + 2 = 210 ∧           -- Q6_K
    2 + 4 + 16 = 22 ∧                   -- Q5_0
    2 + 32 = 34 := by                   -- Q8_0
  decide

/-- Field offsets of Q3_K and Q6_K (the `offsetof` asserts). -/
theorem struct_offsets :
    32 + 64 = 96 ∧ 96 + 12 = 108 ∧ 128 + 64 = 192 ∧ 192 + 16 = 208 := by decide

/-- Quantized activation block alignment: 36 is a multiple of 4 so `int` loads of `qs` (offset 4) are aligned. -/
theorem q8_1_alignment : 36 % 4 = 0 ∧ 4 % 4 = 0 := by decide

-- § iteration traits

/-- `(tag, T, BPI)` of each generic-kernel format: Q5_K=1, Q4_K=2, Q2_0=3, Q3_K=4, Q6_K=5, IQ4_XS=6,
Q4_0/Q5_0/IQ4_NL=7 (Qi = 4), Q8_0=8 (Qi = 8). -/
def traits : List (Nat × Nat × Nat) :=
  [(1, 16, 8), (2, 16, 8), (3, 2, 64), (4, 16, 8), (5, 32, 4), (6, 8, 16), (7, 2, 64), (8, 4, 32)]

/-- Each format's `T * BPI = 128 = WARPS * WARP`: one iteration of all threads tiles `BPI` blocks exactly. -/
theorem traits_tile_threadblock : ∀ x ∈ traits, x.2.1 * x.2.2 = 128 := by decide

/-- `BPI` formulas from the source: `VDR * WARPS * WARP / QI` etc. -/
theorem bpi_formulas :
    2 * 4 * 32 / 32 = 8 ∧ 4 * 32 / 2 = 64 ∧ 4 * 32 / 16 = 8 ∧ 4 * 32 / 32 = 4 ∧
    4 * 4 * 32 / 32 = 16 ∧ 2 * 4 * 32 / 4 = 64 ∧ 2 * 4 * 32 / 8 = 32 := by decide

/-- Thread `tid` of the 128 starts at block `tid / T`; two threads of one block-group share `kbx`. -/
theorem start_block_cover (tid T : Nat) (hT : 0 < T) : tid = (tid / T) * T + tid % T := by
  rw [Nat.mul_comm]
  exact (Nat.div_add_mod tid T).symm

/-- Strided loop `kbx = start, start + BPI, ...` covers every block `b` from start `b % BPI`. -/
theorem stride_cover (b BPI : Nat) : b = b % BPI + BPI * (b / BPI) :=
  (Nat.mod_add_div b BPI).symm

/-- Small-K rule: fewer blocks per row than `BPI` ⇒ four rows per block, grid `ceil(n_out / 4)`. -/
theorem small_k_grid_covers (n_out r : Nat) (h : r < n_out) : r / 4 < (n_out + 4 - 1) / 4 := by omega

/-- Affine correction constants of Q4_0 / Q5_0: `8 * vdr / QI` and `16 * vdr / QI` with vdr = 2, QI = 4. -/
theorem affine_correction_constants : 8 * 2 / 4 = 4 ∧ 16 * 2 / 4 = 8 := by decide

/-- Summing `(c - b) * x` equals `sum(c * x) - b * sum(x)`: the weight-bias correction through Q8_1's stored sum. -/
theorem bias_correction (b : Int) (l : List (Int × Int)) :
    (l.map (fun p => (p.1 - b) * p.2)).sum = (l.map (fun p => p.1 * p.2)).sum - b * (l.map (fun p => p.2)).sum := by
  induction l with
  | nil => simp
  | cons p l ih =>
    simp only [List.map_cons, List.sum_cons]
    rw [ih, Int.sub_mul, Int.mul_add]
    omega

-- § multi-column

/-- Multi-column grid in the EXACT layout: warps per block and rows per block. -/
theorem exact_layout_rows (n_out : Nat) : (n_out + 4 - 1) / 4 * 4 ≥ n_out := by omega

/-- Upstream layout uses two rows per block: grid `(n_out + 1) / 2` covers all rows. -/
theorem upstream_layout_rows (n_out : Nat) : (n_out + 1) / 2 * 2 ≥ n_out := by omega

/-- Column `j` activation offset in Q8_1 blocks: `j * (n_in / 32)`; with `n_in = 32 q` columns do not overlap. -/
theorem column_blocks_disjoint (q j j' k : Nat) (hjj : j < j') (hk : k < q) : j * q + k < j' * q := by
  have : (j + 1) * q ≤ j' * q := Nat.mul_le_mul_right q hjj
  rw [Nat.add_mul] at this
  omega

/-- Output layout `y[j * n_out + row]`: distinct (column, row) pairs hit distinct slots. -/
theorem output_slot_injective (n_out j r j' r' : Nat) (hr : r < n_out) (hr' : r' < n_out)
    (h : j * n_out + r = j' * n_out + r') : j = j' ∧ r = r' := by
  have hpos : 0 < n_out := by omega
  have h1 : (j * n_out + r) / n_out = j := by
    rw [Nat.add_comm, Nat.add_mul_div_right _ _ hpos, Nat.div_eq_of_lt hr, Nat.zero_add]
  have h2 : (j' * n_out + r') / n_out = j' := by
    rw [Nat.add_comm, Nat.add_mul_div_right _ _ hpos, Nat.div_eq_of_lt hr', Nat.zero_add]
  have hj : j = j' := by
    calc j = (j * n_out + r) / n_out := h1.symm
      _ = (j' * n_out + r') / n_out := by rw [h]
      _ = j' := h2
  subst hj
  exact ⟨rfl, by omega⟩

/-- The partial-sum shared array has `NW - 1` warp slices; thread rows `1 .. NW-1` map to slices `0 .. NW-2`. -/
theorem partial_slice_index (NW y : Nat) (h1 : 0 < y) (h2 : y < NW) : y - 1 < NW - 1 := by omega

-- § quantizer / host

/-- Quantizer: `n_in` multiple of 32 means whole warps; `i / 32`, `i % 32` recompose. -/
theorem quantize_blocks (i : Nat) : i = 32 * (i / 32) + i % 32 := by omega

/-- Quantizer grid with 256 threads covers `n_total`. -/
theorem quantize_grid_covers (n i : Nat) (h : i < n) : i / 256 < (n + 255) / 256 := by omega

/-- The Q8_1 scratch size `ncols * (n_in / 32) * 36`; bytes for one column divide into blocks. -/
theorem q8_1_bytes (n_in ncols : Nat) (h : n_in % 32 = 0) :
    ncols * (n_in / 32) * 36 = ncols * n_in / 32 * 36 := by
  have : ncols * n_in / 32 = ncols * (n_in / 32) := by
    obtain ⟨q, hq⟩ : ∃ q, n_in = 32 * q := ⟨n_in / 32, by omega⟩
    subst hq
    rw [Nat.mul_left_comm, Nat.mul_div_cancel_left _ (by decide), Nat.mul_div_cancel_left _ (by decide)]
  rw [this]

/-- `native_mmvq_weight_bytes` block table `(type, block_elems, block_bytes)`. -/
def weightTable : List (Nat × Nat × Nat) :=
  [(2, 32, 18), (6, 32, 22), (8, 32, 34), (20, 32, 18), (11, 256, 110), (12, 256, 144),
   (13, 256, 176), (14, 256, 210), (23, 256, 136), (42, 64, 18)]

/-- Weight byte count of a row-major matrix: `(n_in / block_elems) * block_bytes * n_out`. -/
def weightBytes (elems bytes n_in n_out : Nat) : Nat := (n_in / elems) * bytes * n_out

/-- The weight byte count is monotone in the number of output rows. -/
theorem weightBytes_mono (elems bytes n_in a b : Nat) (h : a ≤ b) :
    weightBytes elems bytes n_in a ≤ weightBytes elems bytes n_in b := by
  unfold weightBytes
  exact Nat.mul_le_mul_left _ h

/-- Every table entry has a block size dividing 256 or equal to 32/64, so `n_in = 2560` is valid for all. -/
theorem n_embd_valid_for_all : ∀ x ∈ weightTable, 2560 % x.2.1 = 0 := by decide

/-- Supported-type predicate matches the table plus the IQ types handled by `iq_mmvq`. -/
theorem table_types_supported :
    ∀ x ∈ weightTable, (x.1 = 2 ∨ x.1 = 6 ∨ x.1 = 8 ∨ x.1 = 11 ∨ x.1 = 12 ∨ x.1 = 13 ∨ x.1 = 14 ∨
      x.1 = 20 ∨ x.1 = 23 ∨ x.1 = 42) := by decide

end Strata.NativeMmvq
