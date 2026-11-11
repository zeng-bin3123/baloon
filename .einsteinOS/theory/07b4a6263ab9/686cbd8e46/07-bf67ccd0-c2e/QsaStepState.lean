/-!
# MODULE: qsa_step_state
SOURCES: src/kernels/cuda/qsa.cu
The per-token QSA step buffer (`qsa_step_fill`, `step_upload`, `validate`): every per-token count
(`n_kv`, `n_bid`, `width`) is derived from one position, so a captured CUDA graph only needs one H2D upload.

## Source semantics
* `n_kv = pos + 1`, `n_bid = n_kv / idx_block`, `width = min(n_kv, top_k + idx_block - 1)`.
* `width ≤ n_kv`; below the selection bound the selection is the identity (`width = n_kv`).
* `n_bid` completed blocks cover at most `n_kv` cells and the next block would overshoot.
* `step_upload` refuses an `n_kv` hint disagreeing with `pos + 1`.
* `validate` requires `head_dim % 4 = 0` (uint2 gather) and `n_head % n_head_kv = 0`.

## Form template identifiers
`qsa_step_fill` → `stepFill`, `qsa_selection_width` → `selectionWidth`, `kStepPos/NKv/NBid/Width` →
`Step.pos/nKv/nBid/width`, `step_upload` hint check → `hintAccepted`, `validate` → `GeomValid`.
-/

namespace Strata.QsaStep

/-- Indexer geometry relevant to the step state. -/
structure Shapes where
  idxBlock : Nat
  topK : Nat

/-- `qsa_selection_width`: the selection keeps `top_k` blocks' worth plus the incomplete tail. -/
def selectionWidth (s : Shapes) (nKv : Nat) : Nat := min nKv (s.topK + s.idxBlock - 1)

/-- The four step entries uploaded per token. -/
structure Step where
  pos : Nat
  nKv : Nat
  nBid : Nat
  width : Nat
deriving DecidableEq

/-- `qsa_step_fill`: the ONE writer of all four step entries. -/
def stepFill (s : Shapes) (pos : Nat) : Step :=
  { pos := pos, nKv := pos + 1, nBid := (pos + 1) / s.idxBlock,
    width := selectionWidth s (pos + 1) }

/-- `step_upload` accepts a caller hint only if it equals the derived `n_kv`. -/
def hintAccepted (s : Shapes) (pos hint : Nat) : Bool := hint == (stepFill s pos).nKv

-- § derived counts

/-- `n_kv` is always the position plus one. -/
theorem stepFill_nKv (s : Shapes) (pos : Nat) : (stepFill s pos).nKv = pos + 1 := rfl

/-- The cache is never empty after a fill. -/
theorem stepFill_nKv_pos (s : Shapes) (pos : Nat) : 0 < (stepFill s pos).nKv := by
  simp [stepFill]

/-- The selection never exceeds the number of cached cells. -/
theorem stepFill_width_le (s : Shapes) (pos : Nat) : (stepFill s pos).width ≤ (stepFill s pos).nKv := by
  simp only [stepFill, selectionWidth]
  exact Nat.min_le_left _ _

/-- The selection never exceeds `top_k + r - 1` (the capacity `topk_512_step` checks). -/
theorem stepFill_width_le_cap (s : Shapes) (pos : Nat) :
    (stepFill s pos).width ≤ s.topK + s.idxBlock - 1 := by
  simp only [stepFill, selectionWidth]
  exact Nat.min_le_right _ _

/-- Below the selection bound the selection is the identity. -/
theorem stepFill_width_identity (s : Shapes) (pos : Nat) (h : pos + 1 ≤ s.topK + s.idxBlock - 1) :
    (stepFill s pos).width = (stepFill s pos).nKv := by
  simp only [stepFill, selectionWidth]
  exact Nat.min_eq_left h

/-- With `idx_block ≥ 2` (what `validate` enforces) the selection is nonempty, so `qsa_attend`'s
zero path is unreachable for a filled step. -/
theorem stepFill_width_pos (s : Shapes) (pos : Nat) (hr : 2 ≤ s.idxBlock) :
    0 < (stepFill s pos).width := by
  simp only [stepFill, selectionWidth]
  exact Nat.lt_min.mpr ⟨Nat.succ_pos _, by omega⟩

/-- Completed blocks never cover more cells than are cached. -/
theorem stepFill_nBid_mul_le (s : Shapes) (pos : Nat) :
    (stepFill s pos).nBid * s.idxBlock ≤ (stepFill s pos).nKv := by
  simp only [stepFill]
  exact Nat.div_mul_le_self _ _

/-- One more block would overshoot: the incomplete tail has fewer than `r` cells. -/
theorem stepFill_nKv_lt_next (s : Shapes) (pos : Nat) (hr : 0 < s.idxBlock) :
    (stepFill s pos).nKv < ((stepFill s pos).nBid + 1) * s.idxBlock := by
  simp only [stepFill]
  have h1 := Nat.div_add_mod (pos + 1) s.idxBlock
  have h2 := Nat.mod_lt (pos + 1) hr
  rw [Nat.succ_mul, Nat.mul_comm ((pos + 1) / s.idxBlock) s.idxBlock]
  omega

/-- The hint check accepts exactly the derived `n_kv`. -/
theorem hintAccepted_iff (s : Shapes) (pos hint : Nat) :
    hintAccepted s pos hint = true ↔ hint = pos + 1 := by
  simp [hintAccepted, stepFill]

/-- A fill is determined by its position: two fills agree iff the positions agree. -/
theorem stepFill_injective (s : Shapes) (p q : Nat) : stepFill s p = stepFill s q ↔ p = q := by
  constructor
  · intro h
    have := congrArg Step.pos h
    simpa [stepFill] using this
  · intro h; subst h; rfl

-- § geometry validation

/-- The part of `validate` that the kernels' index arithmetic depends on. -/
structure GeomValid (nHead nHeadKv headDim : Nat) : Prop where
  kv_pos : 0 < nHeadKv
  gqa : nHead % nHeadKv = 0
  dim4 : headDim % 4 = 0

/-- `head_dim % 4 = 0` makes the uint2 gather cover the row exactly. -/
theorem GeomValid.per_exact {a b c : Nat} (h : GeomValid a b c) : (c / 4) * 4 = c := by
  have := h.dim4
  omega

/-- `n_head % n_head_kv = 0` makes the group size exact. -/
theorem GeomValid.group_exact {a b c : Nat} (h : GeomValid a b c) : (a / b) * b = a := by
  have h1 := Nat.div_add_mod a b
  have h2 := h.gqa
  rw [Nat.mul_comm]
  omega

end Strata.QsaStep
