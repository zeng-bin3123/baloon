-- QsaStepState.lean
-- Formal model of src/kernels/cuda/qsa.cu step buffer arithmetic.
namespace Baloon.QsaStep

structure Shapes where
  idxBlock : Nat
  topK : Nat

def selectionWidth (s : Shapes) (nKv : Nat) : Nat := min nKv (s.topK + s.idxBlock - 1)

structure Step where
  pos : Nat
  nKv : Nat
  nBid : Nat
  width : Nat
deriving DecidableEq

def stepFill (s : Shapes) (pos : Nat) : Step :=
  { pos := pos, nKv := pos + 1, nBid := (pos + 1) / s.idxBlock,
    width := selectionWidth s (pos + 1) }

def hintAccepted (s : Shapes) (pos hint : Nat) : Bool := hint == (stepFill s pos).nKv

theorem stepFill_nKv (s : Shapes) (pos : Nat) : (stepFill s pos).nKv = pos + 1 := rfl

theorem stepFill_nKv_pos (s : Shapes) (pos : Nat) : 0 < (stepFill s pos).nKv := by
  simp [stepFill]

theorem stepFill_width_le (s : Shapes) (pos : Nat) : (stepFill s pos).width ≤ (stepFill s pos).nKv := by
  simp only [stepFill, selectionWidth]; exact Nat.min_le_left _ _

theorem stepFill_width_le_cap (s : Shapes) (pos : Nat) :
    (stepFill s pos).width ≤ s.topK + s.idxBlock - 1 := by
  simp only [stepFill, selectionWidth]; exact Nat.min_le_right _ _

theorem stepFill_width_identity (s : Shapes) (pos : Nat) (h : pos + 1 ≤ s.topK + s.idxBlock - 1) :
    (stepFill s pos).width = (stepFill s pos).nKv := by
  simp only [stepFill, selectionWidth]; exact Nat.min_eq_left h

theorem stepFill_width_pos (s : Shapes) (pos : Nat) (hr : 2 ≤ s.idxBlock) :
    0 < (stepFill s pos).width := by
  simp only [stepFill, selectionWidth]
  exact Nat.lt_min.mpr ⟨Nat.succ_pos _, by omega⟩

theorem stepFill_nBid_mul_le (s : Shapes) (pos : Nat) :
    (stepFill s pos).nBid * s.idxBlock ≤ (stepFill s pos).nKv := by
  simp only [stepFill]; exact Nat.div_mul_le_self _ _

theorem stepFill_nKv_lt_next (s : Shapes) (pos : Nat) (hr : 0 < s.idxBlock) :
    (stepFill s pos).nKv < ((stepFill s pos).nBid + 1) * s.idxBlock := by
  simp only [stepFill]
  have h1 := Nat.div_add_mod (pos + 1) s.idxBlock
  have h2 := Nat.mod_lt (pos + 1) hr
  rw [Nat.succ_mul, Nat.mul_comm ((pos + 1) / s.idxBlock) s.idxBlock]; omega

theorem hintAccepted_iff (s : Shapes) (pos hint : Nat) :
    hintAccepted s pos hint = true ↔ hint = pos + 1 := by
  simp [hintAccepted, stepFill]

theorem stepFill_injective (s : Shapes) (p q : Nat) : stepFill s p = stepFill s q ↔ p = q := by
  constructor
  · intro h; have := congrArg Step.pos h; simpa [stepFill] using this
  · intro h; subst h; rfl

structure GeomValid (nHead nHeadKv headDim : Nat) : Prop where
  kv_pos : 0 < nHeadKv
  gqa : nHead % nHeadKv = 0
  dim4 : headDim % 4 = 0

theorem GeomValid.per_exact {a b c : Nat} (h : GeomValid a b c) : (c / 4) * 4 = c := by
  have := h.dim4; omega

theorem GeomValid.group_exact {a b c : Nat} (h : GeomValid a b c) : (a / b) * b = a := by
  have h1 := Nat.div_add_mod a b
  have h2 := h.gqa
  rw [Nat.mul_comm]; omega

end Baloon.QsaStep
