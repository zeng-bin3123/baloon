/-!
# MODULE: qsa_step_geometry
SOURCES: include/strata/kernels/qsa.hpp, src/kernels/cuda/qsa.cu
The device-resident per-token step state (pos, n_kv, n_bid, width) of the QSA layer, the selection
width, the GQA head map and the paged-KV row addressing.

## Source semantics
* `qsa_step_fill` derives all four entries from `pos`: n_kv = pos+1, n_bid = n_kv / r,
  width = min(n_kv, top_k + r - 1).
* Below the bound the selection is the identity; the width never exceeds n_kv.
* q -> kv head map is integer division (`h / (n_head / n_head_kv)`), not `h % n_head_kv`.
* Physical KV row = (page * n_head_kv + h) * page_size + slot; one head's cells are contiguous in a page.

## Form template identifiers
step.pos <- kStepPos, step.nKv <- kStepNKv, step.nBid <- kStepNBid, step.width <- kStepWidth,
selWidth <- qsa_selection_width, physRow <- kv_append/kv_gather row formula, kvHead <- iv2 = iq2 / rv2.
-/

namespace Strata.QsaStep

-- § selection width and step state

/-- The selection width `min(n_kv, top_k + r - 1)`. -/
def selWidth (nKv topK r : Nat) : Nat := min nKv (topK + r - 1)

/-- The four per-token values held in the device step buffer. -/
structure Step where
  pos : Nat
  nKv : Nat
  nBid : Nat
  width : Nat

/-- `qsa_step_fill`: all four entries computed from the single position. -/
def stepFill (pos topK r : Nat) : Step :=
  { pos := pos, nKv := pos + 1, nBid := (pos + 1) / r, width := selWidth (pos + 1) topK r }

/-- The selection never returns more cells than are cached. -/
theorem selWidth_le_nKv (nKv topK r : Nat) : selWidth nKv topK r ≤ nKv := by
  unfold selWidth; omega

/-- The selection never exceeds the geometry's bound `top_k + r - 1`. -/
theorem selWidth_le_bound (nKv topK r : Nat) : selWidth nKv topK r ≤ topK + r - 1 := by
  unfold selWidth; omega

/-- Below the bound the selection is the identity. -/
theorem selWidth_identity {nKv topK r : Nat} (h : nKv ≤ topK + r - 1) :
    selWidth nKv topK r = nKv := by
  unfold selWidth; omega

/-- The selection width is monotone in the number of cached cells. -/
theorem selWidth_mono {a b : Nat} (topK r : Nat) (h : a ≤ b) :
    selWidth a topK r ≤ selWidth b topK r := by
  unfold selWidth; omega

/-- `n_kv` is `pos + 1` in every filled step. -/
theorem stepFill_nKv (pos topK r : Nat) : (stepFill pos topK r).nKv = pos + 1 := rfl

/-- The filled step is internally consistent: width is derived from its own n_kv. -/
theorem stepFill_consistent (pos topK r : Nat) :
    (stepFill pos topK r).nKv = (stepFill pos topK r).pos + 1 ∧
    (stepFill pos topK r).width = selWidth (stepFill pos topK r).nKv topK r := ⟨rfl, rfl⟩

/-- Completed blocks never cover more cells than are cached. -/
theorem stepFill_blocks_le (pos topK r : Nat) :
    (stepFill pos topK r).nBid * r ≤ (stepFill pos topK r).nKv := by
  simp only [stepFill]
  exact Nat.div_mul_le_self _ _

/-- The incomplete tail is shorter than one block. -/
theorem stepFill_tail_lt {r : Nat} (pos topK : Nat) (hr : 0 < r) :
    (stepFill pos topK r).nKv < (stepFill pos topK r).nBid * r + r := by
  simp only [stepFill]
  have h1 := Nat.div_add_mod (pos + 1) r
  have h2 := Nat.mod_lt (pos + 1) hr
  rw [Nat.mul_comm] at h1
  omega

/-- The step buffer is four int32. -/
theorem stepBytes_eq : 4 * 4 = 16 := by decide

-- § GQA head map

/-- Query head to key/value head: integer division. -/
def kvHead (q group : Nat) : Nat := q / group

/-- The integer-division map lands inside the kv heads. -/
theorem kvHead_lt {q group nKvH : Nat} (hq : q < group * nKvH) : kvHead q group < nKvH :=
  Nat.div_lt_of_lt_mul hq

/-- For the real artifact (24 query heads, 2 kv heads) every map value is a valid kv head. -/
theorem real_gqa_map : ∀ h : Nat, h < 24 → h / 12 < 2 := by
  intro h hh; omega

/-- The real map is `[0 x12, 1 x12]`. -/
theorem real_gqa_split :
    (List.range 24).map (· / 12) = List.replicate 12 0 ++ List.replicate 12 1 := by
  decide

/-- The rival modulo reading differs from the division reading (head 1). -/
theorem div_ne_mod : (1 : Nat) / 12 ≠ 1 % 2 := by decide

-- § paged KV addressing

/-- Physical row of logical (page, kv head, slot). -/
def physRow (page h nKvH pageSize slot : Nat) : Nat := (page * nKvH + h) * pageSize + slot

/-- Consecutive cells of one head are contiguous inside a page. -/
theorem physRow_contiguous (page h nKvH pageSize slot : Nat) :
    physRow page h nKvH pageSize (slot + 1) = physRow page h nKvH pageSize slot + 1 := by
  unfold physRow; omega

/-- A cell index splits into page index and slot. -/
theorem cell_split (pos pageSize : Nat) : pos / pageSize * pageSize + pos % pageSize = pos :=
  Nat.div_add_mod' pos pageSize

/-- The page-table index of a cell is in range. -/
theorem page_index_lt {pos pageSize nPages : Nat} (h : pos < pageSize * nPages) :
    pos / pageSize < nPages := Nat.div_lt_of_lt_mul h

/-- The slot is inside the page. -/
theorem slot_lt {pos pageSize : Nat} (h : 0 < pageSize) : pos % pageSize < pageSize :=
  Nat.mod_lt _ h

end Strata.QsaStep
