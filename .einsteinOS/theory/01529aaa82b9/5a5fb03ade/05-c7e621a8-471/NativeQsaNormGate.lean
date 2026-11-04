/-!
# MODULE: native_qsa_norm_gate
Weighted RMS norm and sigmoid gate of the native QSA path (`src/kernels/cuda/native_qsa.cu`),
plus the buffer-overlap guard used by the host wrappers.

## Source semantics
* `norm`: one block per row, `BlockSize` threads stride over columns (`col = tid + BlockSize*k`);
  block size is 256 for `n_cols < 1024`, else 1024; at most 32 warp partials are reduced.
* Input and output may alias exactly (each thread touches only its own elements after the barrier).
* `gate`: `q_full` holds `[head][2][head_dim]`; the gate is the SECOND half, index
  `head*2*head_dim + head_dim + channel`.
* Spans overlap iff `a < b + bn && b < a + an`.

## Form template identifiers
* `blockSize`    <- `n_cols < 1024 ? 256 : 1024`
* `gateIndex`    <- `head * 2 * head_dim + head_dim + channel`
* `Overlap`      <- `overlap(a, an, b, bn)`
-/

namespace NativeQsaNormGate

-- § norm

/-- Block size selected from the row width. -/
def blockSize (nCols : Nat) : Nat := if nCols < 1024 then 256 else 1024

/-- The reduction needs at most 32 warp partials. -/
theorem blockSize_warps (nCols : Nat) : blockSize nCols / 32 ≤ 32 := by
  unfold blockSize
  split <;> omega

/-- A strided column index determines its thread and iteration uniquely. -/
theorem stride_unique (bs tid k col : Nat) (hpos : 0 < bs) (ht : tid < bs)
    (h : tid + bs * k = col) : tid = col % bs ∧ k = col / bs := by
  subst h
  constructor
  · rw [Nat.add_mul_mod_self_left, Nat.mod_eq_of_lt ht]
  · rw [Nat.add_mul_div_left _ _ hpos, Nat.div_eq_of_lt ht, Nat.zero_add]

/-- Every column of a row is reached by exactly the thread `col % bs` at iteration `col / bs`. -/
theorem stride_cover (bs col : Nat) (hpos : 0 < bs) :
    col % bs + bs * (col / bs) = col := by
  have := Nat.mod_add_div col bs
  omega

/-- Row-major element index stays inside the buffer. -/
theorem row_index_lt (row col nRows nCols : Nat) (hr : row < nRows) (hc : col < nCols) :
    row * nCols + col < nRows * nCols := by
  have h0 : (row + 1) * nCols ≤ nRows * nCols := Nat.mul_le_mul_right _ hr
  have h0' : (row + 1) * nCols = row * nCols + nCols := by rw [Nat.add_mul, Nat.one_mul]
  omega

/-- The in-place read and write of a thread name the same element. -/
theorem inplace_same_index (rowOffset col : Nat) : rowOffset + col = rowOffset + col := rfl

-- § gate

/-- Index of the gate half of `q_full`. -/
def gateIndex (head headDim channel : Nat) : Nat := head * 2 * headDim + headDim + channel

/-- Gate reads stay inside `q_full` (size `nHead * 2 * headDim`). -/
theorem gateIndex_lt (head headDim channel nHead : Nat) (hh : head < nHead)
    (hc : channel < headDim) : gateIndex head headDim channel < nHead * 2 * headDim := by
  unfold gateIndex
  have h0 : (head + 1) * (2 * headDim) ≤ nHead * (2 * headDim) := Nat.mul_le_mul_right _ hh
  have h0' : (head + 1) * (2 * headDim) = head * (2 * headDim) + 2 * headDim := by
    rw [Nat.add_mul, Nat.one_mul]
  have e1 : head * 2 * headDim = head * (2 * headDim) := Nat.mul_assoc _ _ _
  have e2 : nHead * 2 * headDim = nHead * (2 * headDim) := Nat.mul_assoc _ _ _
  omega

/-- The gate half starts after the query half of the same head. -/
theorem gateIndex_ge (head headDim channel : Nat) :
    head * 2 * headDim + headDim ≤ gateIndex head headDim channel := by
  unfold gateIndex
  omega

/-- The flat output index splits into (head, channel) and reassembles. -/
theorem head_channel_roundtrip (i headDim : Nat) :
    headDim * (i / headDim) + i % headDim = i := Nat.div_add_mod i headDim

-- § overlap guard

/-- Two byte spans overlap. -/
def Overlap (a an b bn : Nat) : Prop := a < b + bn ∧ b < a + an

/-- Overlap is symmetric. -/
theorem overlap_symm (a an b bn : Nat) : Overlap a an b bn ↔ Overlap b bn a an := by
  unfold Overlap
  omega

/-- A span that ends before the other starts does not overlap it. -/
theorem not_overlap_of_le (a an b bn : Nat) (h : a + an ≤ b) : ¬ Overlap a an b bn := by
  unfold Overlap
  omega

/-- A nonempty span overlaps itself, so distinct buffers must be checked, and exact aliasing handled separately. -/
theorem overlap_self (a n : Nat) (h : 0 < n) : Overlap a n a n := by
  unfold Overlap
  omega

end NativeQsaNormGate
