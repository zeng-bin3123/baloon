-- NativeQsaBufferContracts.lean
-- Formal model of src/kernels/cuda/native_qsa.cu buffer and element contracts.
namespace Baloon.NativeQsa

def overlapStd (a an b bn : Nat) : Prop := a < b + bn ∧ b < a + an

def overlapRope (x an y bn : Nat) : Prop := if x ≤ y then y - x < an else x - y < bn

theorem overlapRope_iff_std (x an y bn : Nat) (ha : 0 < an) (hb : 0 < bn) :
    overlapRope x an y bn ↔ overlapStd x an y bn := by
  unfold overlapRope overlapStd; split <;> omega

theorem overlap_empty_differs : overlapRope 0 1 0 0 ∧ ¬ overlapStd 0 1 0 0 := by
  unfold overlapRope overlapStd; decide

def buffersOk (inp out n : Nat) : Prop := inp = out ∨ ¬ overlapStd inp n out n

theorem buffersOk_alias (p n : Nat) : buffersOk p p n := Or.inl rfl

theorem buffersOk_rejects_partial (inp out n : Nat) (hne : inp ≠ out) (hov : overlapStd inp n out n) :
    ¬ buffersOk inp out n := by
  intro h; rcases h with h | h; · exact hne h; · exact h hov

def elementsOk (cols rows : Nat) : Prop := 0 < cols ∧ 0 < rows ∧ cols * rows ≤ 2147483647

theorem elementsOk_count (cols rows : Nat) (h : elementsOk cols rows) :
    0 < cols * rows ∧ cols * rows ≤ 2147483647 :=
  ⟨Nat.mul_pos h.1 h.2.1, h.2.2⟩

def gateIdx (hd h c : Nat) : Nat := h * 2 * hd + hd + c

theorem gateIdx_lt (hd nh h c : Nat) (hh : h < nh) (hc : c < hd) : gateIdx hd h c < nh * 2 * hd := by
  unfold gateIdx
  have h1 : (h + 1) * (2 * hd) ≤ nh * (2 * hd) := Nat.mul_le_mul_right (2 * hd) (Nat.succ_le_of_lt hh)
  rw [Nat.succ_mul] at h1; rw [Nat.mul_assoc, Nat.mul_assoc]; omega

theorem gateIdx_second_half (hd h c : Nat) : h * 2 * hd + hd ≤ gateIdx hd h c := by
  unfold gateIdx; omega

def normBlock (nCols : Nat) : Nat := if nCols < 1024 then 256 else 1024

theorem norm_stride_covers (nCols col : Nat) (_hc : col < nCols) :
    col % normBlock nCols < normBlock nCols ∧
      col = col / normBlock nCols * normBlock nCols + col % normBlock nCols := by
  unfold normBlock; split <;> omega

-- warp_sum function referenced in the native_qsa contract
def warp_sum_spec (vals : List Float) : Float := vals.foldl (· + ·) 0.0

-- valid function referenced in native_qsa.cu
def valid (addr bytes : Nat) : Prop := addr ≠ 0 ∧ bytes ≠ 0 ∧ addr % 4 = 0

-- elements function referenced in native_qsa.cu
def elements (cols rows : Nat) : Bool :=
  decide (0 < cols) && decide (0 < rows) && decide (cols * rows ≤ 2147483647)

theorem elements_iff_ok (cols rows : Nat) : elements cols rows = true ↔ elementsOk cols rows := by
  unfold elements elementsOk; simp [Bool.and_eq_true, decide_eq_true_eq]; tauto

-- check_launch function: validates element count before kernel launch
def check_launch (cols rows : Nat) : Bool := elements cols rows

theorem check_launch_safe (cols rows : Nat) (h : check_launch cols rows = true) :
    0 < cols * rows ∧ cols * rows ≤ 2147483647 := by
  rw [check_launch, elements_iff_ok] at h; exact elementsOk_count cols rows h

-- overlap function: checks if two buffers overlap
def overlap (a an b bn : Nat) : Bool := decide (overlapStd a an b bn)

theorem overlap_iff (a an b bn : Nat) : overlap a an b bn = true ↔ overlapStd a an b bn := by
  unfold overlap; exact decide_eq_true_eq

end Baloon.NativeQsa
