/-!
# MODULE: native_qsa_buffer_contracts
SOURCES: src/kernels/cuda/native_qsa.cu, src/kernels/cuda/native_rope.cu
Host-side contracts of the native (llama.cpp-pinned) QSA helpers: `native_qsa_rms_norm_weighted`,
`native_qsa_gate_apply`, `native_rope_apply`, and their buffer checks.

## Source semantics
* `elements(cols, rows)` requires both positive and `cols*rows ≤ INT_MAX`.
* `buffers` accepts disjoint spans or an EXACT input/output alias, never a partial overlap.
* `native_rope`'s `overlaps` (`x ≤ y ? y - x < an : x - y < bn`) agrees with the half-open interval
  test whenever both spans are non-empty.
* The gate reads the SECOND half of each head's `2*head_dim` query row; every such read is inside the
  `2 * n_head * head_dim` floats of `q_full`.
* The norm block size is 256 for rows narrower than 1024 and 1024 otherwise; either way the strided
  column loop covers every column exactly once.

## Form template identifiers
`overlap` → `overlapStd`, native_rope `overlaps` → `overlapRope`, `buffers` → `buffersOk`,
`elements` → `elementsOk`, gate read index → `gateIdx`, `BlockSize` → `normBlock`.
-/

namespace Strata.NativeQsa

/-- Standard half-open overlap (`native_qsa.cu` `overlap`). -/
def overlapStd (a an b bn : Nat) : Prop := a < b + bn ∧ b < a + an

/-- `native_rope.cu`'s ordered overlap test. -/
def overlapRope (x an y bn : Nat) : Prop := if x ≤ y then y - x < an else x - y < bn

/-- The two overlap tests agree for non-empty spans. -/
theorem overlapRope_iff_std (x an y bn : Nat) (ha : 0 < an) (hb : 0 < bn) :
    overlapRope x an y bn ↔ overlapStd x an y bn := by
  unfold overlapRope overlapStd
  split <;> omega

/-- They differ only for an empty span at the same address — impossible for `rows ≥ 1`. -/
theorem overlap_empty_differs : overlapRope 0 1 0 0 ∧ ¬ overlapStd 0 1 0 0 := by
  unfold overlapRope overlapStd
  decide

/-- The `buffers` acceptance rule for input/output: identical, or disjoint. -/
def buffersOk (inp out n : Nat) : Prop := inp = out ∨ ¬ overlapStd inp n out n

/-- An exact in-place alias is accepted. -/
theorem buffersOk_alias (p n : Nat) : buffersOk p p n := Or.inl rfl

/-- A partial overlap of non-empty buffers is rejected. -/
theorem buffersOk_rejects_partial (inp out n : Nat) (hne : inp ≠ out) (hov : overlapStd inp n out n) :
    ¬ buffersOk inp out n := by
  intro h
  rcases h with h | h
  · exact hne h
  · exact h hov

/-- `elements`: positive dimensions with an `int`-bounded product. -/
def elementsOk (cols rows : Nat) : Prop := 0 < cols ∧ 0 < rows ∧ cols * rows ≤ 2147483647

/-- Accepted shapes give a non-empty, int-addressable element count. -/
theorem elementsOk_count (cols rows : Nat) (h : elementsOk cols rows) :
    0 < cols * rows ∧ cols * rows ≤ 2147483647 :=
  ⟨Nat.mul_pos h.1 h.2.1, h.2.2⟩

/-- The gate's read index into `q_full`: the second half of head `h`. -/
def gateIdx (hd h c : Nat) : Nat := h * 2 * hd + hd + c

/-- Every gate read is inside `q_full` (`2 * n_head * head_dim` floats). -/
theorem gateIdx_lt (hd nh h c : Nat) (hh : h < nh) (hc : c < hd) : gateIdx hd h c < nh * 2 * hd := by
  unfold gateIdx
  have h1 : (h + 1) * (2 * hd) ≤ nh * (2 * hd) := Nat.mul_le_mul_right (2 * hd) (Nat.succ_le_of_lt hh)
  rw [Nat.succ_mul] at h1
  rw [Nat.mul_assoc, Nat.mul_assoc]
  omega

/-- The gate reads the second half, never the first: it is at least `h*2*hd + hd`. -/
theorem gateIdx_second_half (hd h c : Nat) : h * 2 * hd + hd ≤ gateIdx hd h c := by
  unfold gateIdx; omega

/-- The norm launch's block size. -/
def normBlock (nCols : Nat) : Nat := if nCols < 1024 then 256 else 1024

/-- Thread `col % B` covers column `col` at step `col / B` of the strided loop. -/
theorem norm_stride_covers (nCols col : Nat) (_hc : col < nCols) :
    col % normBlock nCols < normBlock nCols ∧
      col = col / normBlock nCols * normBlock nCols + col % normBlock nCols := by
  unfold normBlock
  split <;> omega

end Strata.NativeQsa
