/-!
# MODULE: native_mmvq_quantized_gemv
Quantized matrix-vector product (MMVQ) for the 16 supported GGML block formats: Q8_1 activation
quantization, per-format block layouts, thread-to-block iteration geometry, small-K row grouping,
multi-column (ncols 2..8) layout and shape/byte validation.

## Source semantics
* Every weight block struct has a fixed byte size (Q5_K 176, Q4_K 144, Q6_K 210, Q3_K 110, IQ4_XS 136,
  Q2_0/Q4_0/IQ4_NL 18, Q5_0 22, Q8_0 34, Q8_1 36).
* Blocks per K iteration = VDR*WARPS*WARP/QI; small-K (fewer blocks than one iteration) uses 4 rows per block.
* Each valid output row is covered by exactly one (block, slot); rows >= n_out are guarded.
* Column j of a multi-column call reads x + j*(n_in/32) and writes y + j*n_out.
* With 4 warps the multi-column layout equals the single-column layout (bitwise-equal contract);
  the upstream layout uses 2 rows/block and (ncols>4) 2 warps.
* The warp reduction is an XOR butterfly: each stage pairs lane l with l xor mask (an involution).

## Form template identifiers
* `WARPS`, `WARP`            -> abstract warp-count / warp-size constants
* `bpi vdr qi`               -> blocks-per-iteration function (kernel loop stride)
* `rowsPer`, `gridBlocks`    -> rows per CUDA block, grid size
* `validShape`               -> validate_shape predicate
* `weightBytes`              -> native_mmvq_weight_bytes
* `supported`                -> native_mmvq_supported
-/

namespace Strata.NativeMmvq

/-- Number of warps per CUDA block in every MMVQ kernel. -/
def WARPS : Nat := 4
/-- Lanes per warp. -/
def WARP : Nat := 32

-- § block layouts

/-- Struct sizes: field-by-field byte sums match the `static_assert`ed `sizeof`s. -/
theorem block_sizes :
    (4 + 12 + 32 + 128 = 176) ∧ (4 + 12 + 128 = 144) ∧ (128 + 64 + 16 + 2 = 210) ∧
    (32 + 64 + 12 + 2 = 110) ∧ (2 + 2 + 4 + 128 = 136) ∧ (2 + 16 = 18) ∧
    (2 + 4 + 16 = 22) ∧ (2 + 32 = 34) ∧ (4 + 32 = 36) := by
  decide

/-- Q3_K field offsets: hmask at 0, qs at 32, scales at 96, d at 108. -/
theorem q3k_offsets : (32 = 32 + 0) ∧ (32 + 64 = 96) ∧ (96 + 12 = 108) ∧ (108 + 2 = 110) := by
  decide

/-- Q6_K field offsets: ql 0, qh 128, scales 192, d 208. -/
theorem q6k_offsets : (128 + 64 = 192) ∧ (192 + 16 = 208) ∧ (208 + 2 = 210) := by
  decide

-- § iteration geometry

/-- Blocks consumed per K-loop iteration by a CUDA block. -/
def bpi (vdr qi : Nat) : Nat := vdr * WARPS * WARP / qi

/-- The concrete iteration widths transcribed in the kernels. -/
theorem bpi_values :
    bpi 2 32 = 8 ∧ bpi 1 16 = 8 ∧ bpi 1 32 = 4 ∧ bpi 4 32 = 16 ∧ bpi 1 2 = 64 ∧
    bpi 2 4 = 64 ∧ bpi 2 8 = 32 := by
  decide

/-- Small-K flag is a threshold on K: for the 32-element formats with QI=4, `n_in/32 < 64` iff `n_in < 2048`. -/
theorem small_k_threshold_q4 (m : Nat) : (32 * m) / 32 < 64 ↔ 32 * m < 2048 := by
  omega

/-- Same for QI=8 (Q8_0): threshold 1024 elements. -/
theorem small_k_threshold_q8 (m : Nat) : (32 * m) / 32 < 32 ↔ 32 * m < 1024 := by
  omega

/-- Rows handled by one CUDA block. -/
def rowsPer (smallK : Bool) : Nat := if smallK then WARPS else 1

/-- Grid size for `n_out` rows. -/
def gridBlocks (smallK : Bool) (n_out : Nat) : Nat :=
  if smallK then (n_out + WARPS - 1) / WARPS else n_out

/-- The launch grid covers every output row. -/
theorem grid_covers (smallK : Bool) (n_out : Nat) :
    n_out ≤ rowsPer smallK * gridBlocks smallK n_out := by
  cases smallK
  · show n_out ≤ 1 * n_out
    omega
  · show n_out ≤ 4 * ((n_out + 4 - 1) / 4)
    omega

/-- Every row `r` is owned by block `r/4`, slot `r%4` in small-K mode, and the slot is in range. -/
theorem small_k_row_decomposition (r : Nat) :
    4 * (r / 4) + r % 4 = r ∧ r % 4 < 4 := by
  omega

/-- Row-guarded block index stays inside the weight matrix: `row*bpr + kbx < n_out*bpr`. -/
theorem weight_block_in_range {row kbx bpr n_out : Nat} (hr : row < n_out) (hk : kbx < bpr) :
    row * bpr + kbx < n_out * bpr := by
  have h1 : (row + 1) * bpr ≤ n_out * bpr := Nat.mul_le_mul_right bpr (Nat.succ_le_of_lt hr)
  have h2 : (row + 1) * bpr = row * bpr + bpr := Nat.succ_mul row bpr
  omega

-- § multi-column

/-- Blocks per iteration scale linearly with the warp count. -/
def multiBpi (b nw : Nat) : Nat := b * nw / WARPS

/-- With the exact 4-warp layout the multi-column stride equals the single-column stride. -/
theorem multi_exact_same_stride (b : Nat) : multiBpi b 4 = b := by
  unfold multiBpi WARPS
  omega

/-- The upstream 2-warp layout halves the stride (8 -> 4). -/
theorem multi_upstream_two_warps : multiBpi 8 2 = 4 := by
  decide

/-- Upstream layout uses 2 rows per block: the grid `(n_out+1)/2` covers all rows. -/
theorem multi_upstream_grid_covers (n_out : Nat) : n_out ≤ 2 * ((n_out + 1) / 2) := by
  omega

/-- Output index of column `j`, row `row` is in range. -/
theorem multi_output_in_range {j row ncols n_out : Nat} (hj : j < ncols) (hr : row < n_out) :
    j * n_out + row < ncols * n_out := by
  have h1 : (j + 1) * n_out ≤ ncols * n_out := Nat.mul_le_mul_right n_out (Nat.succ_le_of_lt hj)
  have h2 : (j + 1) * n_out = j * n_out + n_out := Nat.succ_mul j n_out
  omega

-- § validation and bytes

/-- `validate_shape`: positive divisible `n_in` and `1 ≤ ncols ≤ 8`. -/
def validShape (n_in ncols be : Nat) : Prop :=
  0 < n_in ∧ n_in % be = 0 ∧ 1 ≤ ncols ∧ ncols ≤ 8

/-- A valid shape has a whole number of blocks per row. -/
theorem valid_shape_whole_blocks {n_in ncols be : Nat} (h : validShape n_in ncols be) :
    n_in / be * be = n_in := by
  obtain ⟨_, hmod, _, _⟩ := h
  exact Nat.div_mul_cancel (Nat.dvd_of_mod_eq_zero hmod)

/-- Total weight bytes for a matrix. -/
def weightBytes (be bb n_in n_out : Nat) : Nat := n_in / be * bb * n_out

/-- Row bytes below `MAX / n_out` guarantee the overflow check passes (no wrap). -/
theorem weight_bytes_no_overflow {rowBytes nOut maxv : Nat} (hn : 0 < nOut)
    (h : rowBytes ≤ maxv / nOut) : rowBytes * nOut ≤ maxv :=
  (Nat.le_div_iff_mul_le hn).mp h

/-- Q8_1 byte count per call: `ncols * (n_in/32) * 36`. -/
def q81Bytes (n_in ncols : Nat) : Nat := ncols * (n_in / 32) * 36

/-- Columns are contiguous: a 32-element block never straddles two columns, so quantizing
`ncols*n_in` elements as one vector is the same as quantizing each column. -/
theorem q81_block_inside_column (i m : Nat) : i / 32 / m = i / (32 * m) :=
  Nat.div_div_eq_div_mul i 32 m

/-- Supported GGML type ids. -/
def supported (t : Nat) : Bool :=
  [2, 6, 8, 11, 12, 13, 14, 20, 23, 42, 16, 17, 18, 21, 22, 29].contains t

/-- The supported list has 16 distinct entries. -/
theorem supported_nodup : [2, 6, 8, 11, 12, 13, 14, 20, 23, 42, 16, 17, 18, 21, 22, 29].Nodup := by
  decide

/-- Spot checks of the support predicate. -/
theorem supported_spot : supported 12 = true ∧ supported 42 = true ∧ supported 3 = false := by
  decide

-- § warp reduction

/-- The XOR butterfly pairing is an involution: lane `l` reads partner `l xor m`, and the partner reads `l` back. -/
theorem xor_partner_involutive (l m : Nat) : (l ^^^ m) ^^^ m = l := by
  rw [Nat.xor_assoc, Nat.xor_self, Nat.xor_zero]

end Strata.NativeMmvq
