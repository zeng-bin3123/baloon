/-!
# MODULE: iq_kernels
Row sizes, grouped-expert blob layout and launch arithmetic for the IQ-format matrix-vector kernels.

## Source semantics
A weight row of n values in a format with `elems` values per block of `bytes` bytes occupies (n/elems)*bytes. An expert blob is [gate rows | up rows | down rows], with up_off = n_ff*gu_row and down_off = 2*up_off. The q8_1 quantizer maps flat element i to block i/32, lane i%32. Gate/up fused rows are 2*n_ff, the upper half being the up projection.

## Form template identifiers
iq_row_bytes -> row_bytes, native_expert_layout -> expert_layout, quantize_q8_1_kernel -> q8_1_quantize
-/
namespace Strata.IqKernels

/-- Bytes of one weight row. -/
def rowBytes (n elems bytes : Nat) : Nat := n / elems * bytes

/-- Expert blob layout. -/
structure Layout where
  nEmbd : Nat
  nFf : Nat
  guRow : Nat
  dRow : Nat

/-- Offset of the up matrix. -/
def Layout.upOff (L : Layout) : Nat := L.nFf * L.guRow
/-- Offset of the down matrix. -/
def Layout.downOff (L : Layout) : Nat := 2 * L.upOff
/-- Total blob bytes. -/
def Layout.bytes (L : Layout) : Nat := L.downOff + L.nEmbd * L.dRow

-- § layout

/-- The up and gate matrices have equal size, so the regions are adjacent. -/
theorem down_after_up (L : Layout) : L.downOff = L.upOff + L.upOff := by unfold Layout.downOff; omega

/-- The blob contains the up region. -/
theorem up_in_blob (L : Layout) : L.upOff ≤ L.bytes := by unfold Layout.bytes Layout.downOff; omega

/-- The down region ends at the blob end. -/
theorem down_end (L : Layout) : L.downOff + L.nEmbd * L.dRow = L.bytes := rfl

/-- A row of zero columns is empty. -/
theorem rowBytes_zero (elems bytes : Nat) : rowBytes 0 elems bytes = 0 := by simp [rowBytes]

/-- Row bytes at block-multiple widths are exact. -/
theorem rowBytes_exact (k elems bytes : Nat) (h : 0 < elems) : rowBytes (k * elems) elems bytes = k * bytes := by simp [rowBytes, Nat.mul_div_cancel _ h]

-- § 256-byte alignment of scratch

/-- Round up to a multiple of 256. -/
def align256 (x : Nat) : Nat := (x + 255) / 256 * 256

/-- Alignment never shrinks. -/
theorem align256_ge (x : Nat) : x ≤ align256 x := by unfold align256; omega

/-- Result is 256-aligned. -/
theorem align256_mod (x : Nat) : align256 x % 256 = 0 := by unfold align256; omega

-- § row selection in the fused gate/up kernel

/-- Rebased up rows are valid rows of the up matrix. -/
theorem up_row_rebase (n_ff row : Nat) (h : row < 2 * n_ff) (hu : n_ff ≤ row) : row - n_ff < n_ff := by omega

/-- The gate/up grid covers 2*n_ff rows. -/
theorem gu_grid (n_ff : Nat) : 2 * n_ff ≤ (2 * n_ff + 7) / 8 * 8 := by omega

/-- The mmvq grid (4 rows per block) covers n_out. -/
theorem mmvq_grid (n_out : Nat) : n_out ≤ (n_out + 3) / 4 * 4 := by omega

-- § q8_1 quantizer

/-- Block and lane recompose the element index. -/
theorem q8_decomp (i : Nat) : (i / 32) * 32 + i % 32 = i := by omega

/-- The quantizer grid covers n elements. -/
theorem quant_grid (n : Nat) : n ≤ (n + 255) / 256 * 256 := by omega

end Strata.IqKernels
