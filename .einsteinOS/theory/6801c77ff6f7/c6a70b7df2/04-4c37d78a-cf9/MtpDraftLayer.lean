/-!
# MODULE: mtp_draft_layer
SOURCES: include/strata/core/mtp.hpp, src/core/mtp.cpp
The MTP (multi-token-prediction) draft layer: a bump-carved device arena, per-cell step records, the batched
prompt prefill, the capture-per-step draft chain with an early-stop rule, and the draft-head vocabulary subset.

## Source semantics
* `Bump::take` rounds every carve up to 256 bytes; the counting pass and the real pass run the same sequence, so
  the arena size is the final `used`.
* A cell `c` is described by the step record `(c, c+1, (c+1)/4, c+1)`; the position of every head is `c`.
* `prefill` walks the prompt in batches of at most `max_t` rows, skipping batches wholly before
  `prompt_len - window - 64`.
* `draft` always produces one draft, then continues the chain for steps `j = 1 .. max_t-2` while the last
  draft's probability is at least `min_p`; the unused tail of `drafts` is zeroed.
* The full layer runs one row at a time (`full -> T = 1`); `step_row0 < 0` encodes "K/V only from row `-1-step_row0`".
* Step rows: `0..T-1` catch-up, `max_t + j - 1` chain step `j`, `2*max_t - 1` the last-accepted-row slot.

## Form template identifiers
role mapping (source names -> abstract names):
  Bump::take            -> arena_take
  alignUp               -> align256
  MtpDrafter::prefill   -> prompt_batch_walk
  MtpDrafter::draft     -> draft_chain
  record_forward        -> layer_forward
  step_exec_/round_exec_/prefill_exec_ -> captured_graph_cache
  draft_vocab.bin       -> draft_head_subset
-/

namespace Strata.Mtp

-- § arena carving

/-- A bump take of `b` bytes occupies `b` rounded up to a multiple of 256. -/
def alignUp (b : Nat) : Nat := (b + 255) / 256 * 256

/-- The rounded size never shrinks a request. -/
theorem alignUp_ge (b : Nat) : b ≤ alignUp b := by
  unfold alignUp; omega

/-- The rounded size is a multiple of 256. -/
theorem alignUp_mod (b : Nat) : alignUp b % 256 = 0 := by
  unfold alignUp; omega

/-- The rounding wastes less than 256 bytes. -/
theorem alignUp_lt (b : Nat) : alignUp b < b + 256 := by
  unfold alignUp; omega

/-- Offset of take number `i` in a carve sequence of `(count, element size)` pairs. -/
def offsetAt : List (Nat × Nat) → Nat → Nat
  | [], _ => 0
  | _ :: _, 0 => 0
  | p :: ps, i + 1 => alignUp (p.1 * p.2) + offsetAt ps i

/-- Every carve offset is 256-byte aligned. -/
theorem offsetAt_mod (l : List (Nat × Nat)) : ∀ i, offsetAt l i % 256 = 0 := by
  induction l with
  | nil => intro i; simp [offsetAt]
  | cons p ps ih =>
    intro i
    cases i with
    | zero => simp [offsetAt]
    | succ j =>
      simp only [offsetAt]
      have h1 := alignUp_mod (p.1 * p.2)
      have h2 := ih j
      omega

/-- Offsets never decrease from one take to the next. -/
theorem offsetAt_mono (l : List (Nat × Nat)) : ∀ i, offsetAt l i ≤ offsetAt l (i + 1) := by
  induction l with
  | nil => intro i; simp [offsetAt]
  | cons p ps ih =>
    intro i
    cases i with
    | zero => simp [offsetAt]
    | succ j =>
      simp only [offsetAt]
      have h := ih j
      omega

/-- Offset 0 is always 0. -/
theorem offsetAt_zero (l : List (Nat × Nat)) : offsetAt l 0 = 0 := by
  cases l <;> rfl

/-- The first take starts at offset 0 and the second right after its rounded size. -/
theorem offsetAt_second (p : Nat × Nat) (ps : List (Nat × Nat)) :
    offsetAt (p :: ps) 0 = 0 ∧ offsetAt (p :: ps) 1 = alignUp (p.1 * p.2) := by
  simp [offsetAt, offsetAt_zero]

/-- The region of the first take is contained in the space before the second take. -/
theorem first_take_fits (p : Nat × Nat) (ps : List (Nat × Nat)) :
    p.1 * p.2 ≤ offsetAt (p :: ps) 1 := by
  have h := offsetAt_second p ps
  rw [h.2]
  exact alignUp_ge _

-- § step records

/-- The per-cell step record handed to the QSA append/attention kernels. -/
structure StepRec where
  cell : Nat
  len : Nat
  block : Nat
  seq : Nat

/-- Step record of cell `c` (`put` in `draft`, the loop body of `prefill`). -/
def stepOf (c : Nat) : StepRec := ⟨c, c + 1, (c + 1) / 4, c + 1⟩

/-- The visible length is one past the cell. -/
theorem stepOf_len (c : Nat) : (stepOf c).len = c + 1 := rfl

/-- The block number rounds the length down to a multiple of 4. -/
theorem stepOf_block_le (c : Nat) : (stepOf c).block * 4 ≤ c + 1 := by
  show (c + 1) / 4 * 4 ≤ c + 1
  omega

/-- The block number is the largest one that fits. -/
theorem stepOf_block_lt (c : Nat) : c + 1 < ((stepOf c).block + 1) * 4 := by
  show c + 1 < ((c + 1) / 4 + 1) * 4
  omega

-- § prompt prefill batching

/-- Rows processed by the batch starting at row `c` of a prompt of `n` cells. -/
def batchSize (maxT n c : Nat) : Nat := min maxT (n - c)

/-- A batch inside the prompt is never empty. -/
theorem batchSize_pos (maxT n c : Nat) (h1 : 1 ≤ maxT) (h2 : c < n) : 1 ≤ batchSize maxT n c := by
  unfold batchSize; omega

/-- A batch never exceeds the staging capacity. -/
theorem batchSize_le (maxT n c : Nat) : batchSize maxT n c ≤ maxT := by
  unfold batchSize; omega

/-- A batch never runs past the end of the prompt. -/
theorem batchSize_fits (maxT n c : Nat) : c + batchSize maxT n c ≤ n ∨ n < c := by
  unfold batchSize; omega

/-- The first prompt cell whose K/V the attention window can still reach. -/
def firstNeeded (window promptLen : Nat) : Nat :=
  if 0 < window ∧ 0 < promptLen then promptLen - window - 64 else 0

/-- A batch is skipped when it ends at or before `firstNeeded`. -/
def skipBatch (cell0 c T firstNeed : Nat) : Bool := decide (cell0 + c + T ≤ firstNeed)

/-- No batch is skipped when no window bound applies. -/
theorem no_skip_without_window (cell0 c T : Nat) (hT : 1 ≤ T) :
    skipBatch cell0 c T (firstNeeded 0 0) = false := by
  simp [skipBatch, firstNeeded]
  omega

/-- Every cell of a skipped batch is more than `window + 64` behind the prompt end. -/
theorem skipped_cells_unreachable (window promptLen cell0 c T cell : Nat)
    (hw : 0 < window) (hp : 0 < promptLen)
    (hs : skipBatch cell0 c T (firstNeeded window promptLen) = true)
    (hcell : cell < cell0 + c + T) :
    cell < promptLen - window - 64 := by
  have h : cell0 + c + T ≤ firstNeeded window promptLen := of_decide_eq_true hs
  have h2 : firstNeeded window promptLen = promptLen - window - 64 := by
    simp [firstNeeded, hw, hp]
  omega

-- § window and attention span

/-- The window actually applied: 0 means attention over every cell. -/
def effWindow (window maxCells : Nat) : Nat := if 0 < window ∧ window < maxCells then window else 0

/-- Number of cells the attention reads at most. -/
def attnSpan (window maxCells : Nat) : Nat :=
  if 0 < effWindow window maxCells then effWindow window maxCells else maxCells

/-- Capacity of the identity selection table: the span rounded up to 64. -/
def capCells (window maxCells : Nat) : Nat := (attnSpan window maxCells + 63) / 64 * 64

/-- The attention span never exceeds the cache. -/
theorem attnSpan_le (window maxCells : Nat) : attnSpan window maxCells ≤ maxCells := by
  by_cases h : 0 < window ∧ window < maxCells
  · have h1 : effWindow window maxCells = window := by simp [effWindow, h]
    have h2 : attnSpan window maxCells = window := by simp [attnSpan, h1, h.1]
    omega
  · have h1 : effWindow window maxCells = 0 := by simp [effWindow, h]
    simp [attnSpan, h1]

/-- The selection table holds the whole span. -/
theorem capCells_ge (window maxCells : Nat) : attnSpan window maxCells ≤ capCells window maxCells := by
  unfold capCells; omega

/-- The selection table size is a multiple of 64. -/
theorem capCells_mod (window maxCells : Nat) : capCells window maxCells % 64 = 0 := by
  unfold capCells; omega

-- § the draft chain

/-- Number of chain steps taken after the first draft: step `j` runs when `cont j` holds. -/
def loopLen (cont : Nat → Bool) : Nat → Nat → Nat
  | _, 0 => 0
  | j, fuel + 1 => if cont j = true then 1 + loopLen cont (j + 1) fuel else 0

/-- The chain loop never exceeds its iteration budget. -/
theorem loopLen_le (cont : Nat → Bool) (fuel : Nat) : ∀ j, loopLen cont j fuel ≤ fuel := by
  induction fuel with
  | zero => intro j; simp [loopLen]
  | succ f ih =>
    intro j
    simp only [loopLen]
    split
    · have := ih (j + 1)
      omega
    · omega

/-- A chain whose continuation test always fails takes no steps. -/
theorem loopLen_false (j f : Nat) : loopLen (fun _ => false) j f = 0 := by
  cases f <;> simp [loopLen]

/-- Drafts produced in one round: the first draft plus the chain steps. -/
def draftCount (maxT : Nat) (cont : Nat → Bool) : Nat := 1 + loopLen cont 1 (maxT - 2)

/-- A round yields between 1 and `max_t - 1` drafts. -/
theorem draftCount_bounds (maxT : Nat) (cont : Nat → Bool) (h : 2 ≤ maxT) :
    1 ≤ draftCount maxT cont ∧ draftCount maxT cont ≤ maxT - 1 := by
  have := loopLen_le cont (maxT - 2) 1
  unfold draftCount
  omega

/-- A chain that stops immediately yields exactly the first draft. -/
theorem draftCount_stop_early (maxT : Nat) : draftCount maxT (fun _ => false) = 1 := by
  unfold draftCount
  rw [loopLen_false]

/-- Number of zero-filled draft slots after the chain. -/
def padCount (maxT n : Nat) : Nat := (maxT - 1) - n

/-- Produced plus padded slots fill exactly `max_t - 1`. -/
theorem draft_slots_full (maxT n : Nat) (h : n ≤ maxT - 1) : n + padCount maxT n = maxT - 1 := by
  unfold padCount; omega

/-- Chain-step rows lie above the catch-up rows and below the last-accepted slot. -/
theorem chain_row_range (maxT j : Nat) (h1 : 1 ≤ j) (h2 : j + 2 ≤ maxT) :
    maxT ≤ maxT + j - 1 ∧ maxT + j - 1 < 2 * maxT - 1 := by
  omega

/-- Catch-up rows (`< T <= max_t`) never collide with chain rows. -/
theorem catchup_row_below_chain (maxT T t j : Nat) (hT : T ≤ maxT) (ht : t < T) (hj : 1 ≤ j) :
    t < maxT + j - 1 := by
  omega

/-- The last-accepted slot is the final row of the `2 * max_t` staging area. -/
theorem last_slot_in_range (maxT : Nat) (h : 1 ≤ maxT) : 2 * maxT - 1 < 2 * maxT := by
  omega

/-- Argument check of `draft`: the accepted row lies in the window. -/
def draftArgsOk (T maxT a : Nat) : Bool := decide (1 ≤ T ∧ T ≤ maxT ∧ a < T)

/-- A valid call has a valid accepted row and a window within staging capacity. -/
theorem draftArgsOk_sound (T maxT a : Nat) (h : draftArgsOk T maxT a = true) :
    a < maxT ∧ 1 ≤ T := by
  have h' : 1 ≤ T ∧ T ≤ maxT ∧ a < T := of_decide_eq_true h
  omega

/-- `draft_first` always uses accepted row 0, valid whenever the window is non-empty. -/
theorem draft_first_row_valid (T maxT : Nat) (h1 : 1 ≤ T) (h2 : T ≤ maxT) :
    draftArgsOk T maxT 0 = true := by
  simp [draftArgsOk]
  omega

-- § full layer versus K/V-only encoding

/-- Encoding of "K/V only starting at row `r`" in the `step_row0` argument. -/
def encKvOnly (r : Nat) : Int := -1 - (r : Int)

/-- Decoding recovers the row (`row0 = -1 - step_row0`). -/
def decRow (s : Int) : Int := -1 - s

/-- Encode then decode is the identity. -/
theorem dec_enc (r : Nat) : decRow (encKvOnly r) = (r : Int) := by
  unfold decRow encKvOnly; omega

/-- A K/V-only encoding is always negative, so `full` is false. -/
theorem enc_is_kv_only (r : Nat) : encKvOnly r < 0 := by
  unfold encKvOnly; omega

/-- A non-negative `step_row0` is a full-layer request and must have `T = 1`. -/
def fullLayerOk (s : Int) (T : Nat) : Prop := 0 ≤ s → T = 1

/-- K/V-only requests are valid for any batch size. -/
theorem kv_only_any_T (r T : Nat) : fullLayerOk (encKvOnly r) T := by
  intro h
  have := enc_is_kv_only r
  omega

-- § draft vocabulary subset

/-- Number of vocabulary ids in `draft_vocab.bin` (4 bytes each). -/
def nDvocab (rawBytes : Nat) : Nat := rawBytes / 4

/-- The file size is `4 * n` when it is a multiple of 4, as the loader requires. -/
theorem nDvocab_exact (rawBytes : Nat) (h : rawBytes % 4 = 0) : nDvocab rawBytes * 4 = rawBytes := by
  unfold nDvocab; omega

/-- A loadable subset file holds at least one id. -/
theorem nDvocab_pos (rawBytes : Nat) (h : 4 ≤ rawBytes) : 1 ≤ nDvocab rawBytes := by
  unfold nDvocab; omega

end Strata.Mtp
