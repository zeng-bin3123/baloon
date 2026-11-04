/-!
# MODULE: native_qsa_norm_gate
Weighted RMS norm (block-size-dispatched, in-place safe) and the sigmoid output gate for QSA attention.

## Source semantics
* RMS norm uses a 256-thread block when `n_cols < 1024`, otherwise 1024 threads; each thread strides by the
  block size so every column is visited by exactly thread `col % BlockSize`.
* The gate reads `q_full[head*2*head_dim + head_dim + channel]` (the SECOND half of each head's row).
* Buffers must be aligned, bounded, disjoint, or an exact input/output alias.

## Form template identifiers
* `blockSize`       -> norm block size dispatch
* `gateIndex`       -> index of the gate half in q_full
* `overlap`         -> byte-span overlap
-/

namespace Strata.NativeQsa

/-- Block size dispatch of the norm kernel. -/
def blockSize (n_cols : Nat) : Nat := if n_cols < 1024 then 256 else 1024

/-- Dispatch gives one of the two instantiated sizes. -/
theorem blockSize_cases (n : Nat) : blockSize n = 256 ∨ blockSize n = 1024 := by
  unfold blockSize
  split <;> simp

/-- The warp-partials array (32 entries) is large enough: `blockSize / 32 ≤ 32`. -/
theorem warps_fit (n : Nat) : blockSize n / 32 ≤ 32 := by
  unfold blockSize
  split <;> omega

/-- Strided loop coverage: column `col` is visited by thread `col % BS` at iteration `col / BS`. -/
theorem strided_coverage (col BS : Nat) (h : 0 < BS) :
    col % BS < BS ∧ col % BS + BS * (col / BS) = col :=
  ⟨Nat.mod_lt col h, Nat.mod_add_div col BS⟩

/-- Row offset `row*n_cols + col` is inside `n_rows*n_cols`. -/
theorem row_offset_in_range {row col n_rows n_cols : Nat} (hr : row < n_rows) (hc : col < n_cols) :
    row * n_cols + col < n_rows * n_cols := by
  have h1 : (row + 1) * n_cols ≤ n_rows * n_cols := Nat.mul_le_mul_right n_cols (Nat.succ_le_of_lt hr)
  have h2 : (row + 1) * n_cols = row * n_cols + n_cols := Nat.succ_mul row n_cols
  omega

/-- Gate index: element `i` belongs to head `i/head_dim`, channel `i%head_dim`, reassembling to `i`. -/
theorem gate_decomposition (i head_dim : Nat) : (i / head_dim) * head_dim + i % head_dim = i :=
  Nat.div_add_mod' i head_dim

/-- Position of the gate half in `q_full`. -/
def gateIndex (head head_dim channel : Nat) : Nat := head * (2 * head_dim) + head_dim + channel

/-- The gate half lies strictly inside head `head`'s `2*head_dim` row, at or above its midpoint. -/
theorem gate_index_second_half {head head_dim channel : Nat} (hc : channel < head_dim) :
    head * (2 * head_dim) + head_dim ≤ gateIndex head head_dim channel ∧
    gateIndex head head_dim channel < (head + 1) * (2 * head_dim) := by
  unfold gateIndex
  have h2 : (head + 1) * (2 * head_dim) = head * (2 * head_dim) + 2 * head_dim := Nat.succ_mul head _
  omega

/-- `q_full` has `2*n_head*head_dim` elements; every gate index is in range. -/
theorem gate_index_in_range {head n_head head_dim channel : Nat} (hh : head < n_head)
    (hc : channel < head_dim) : gateIndex head head_dim channel < n_head * (2 * head_dim) := by
  have h := (gate_index_second_half (head := head) hc).2
  have h1 : (head + 1) * (2 * head_dim) ≤ n_head * (2 * head_dim) :=
    Nat.mul_le_mul_right _ (Nat.succ_le_of_lt hh)
  omega

/-- Byte span overlap. -/
def overlap (ap an bp bn : Nat) : Prop := ap < bp + bn ∧ bp < ap + an

/-- Overlap is symmetric. -/
theorem overlap_symm {ap an bp bn : Nat} (h : overlap ap an bp bn) : overlap bp bn ap an :=
  ⟨h.2, h.1⟩

/-- An input/output exact alias is a genuine overlap, so it must be special-cased explicitly. -/
theorem alias_is_overlap (p n : Nat) (hn : 0 < n) : overlap p n p n := by
  unfold overlap
  omega

/-- Alignment check: a 4-aligned pointer plus a multiple-of-4 size stays 4-aligned. -/
theorem aligned_end (p n : Nat) (hp : p % 4 = 0) (hn : n % 4 = 0) : (p + n) % 4 = 0 := by
  omega

end Strata.NativeQsa
