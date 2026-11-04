/-!
# MODULE: mtp_drafter
SOURCES: src/core/mtp.cpp
The multi-token-prediction draft layer: window/capacity sizing, step-row staging, the probability-gated draft
chain, chunked prefill and tensor lookup by name and kind.

## Source semantics
* `window_ = window` only when `0 < window < max_cells`, else 0 (dense attention over every cell).
* The attention capacity is `cap = ceil(len / 64) * 64` with `len = window_` or `max_cells`.
* Step/pos staging has `2*T` rows: T catch-up rows, chain steps in rows `T + j - 1` for `j = 1..T-2`, and the
  full-layer row `2*T - 1`.
* The chain continues while `j < max_t - 1` and the last draft's probability is at least `min_p`;
  the remaining draft slots are zero-filled.
* Prefill walks the prompt in chunks of `max_t`, skipping chunks entirely below the first needed cell.
* Tensors are looked up by name AND kind; a kind mismatch yields null.

## Form template identifiers
- effWindow → window_ ; capOf → cap_ ; chainFuel → the `for (j = 1; ...)` loop ; findTensor → f32/bf16/q8
-/

namespace Strata.MtpDrafter

-- § window and capacity

/-- Effective window: the requested one only when strictly inside the context. -/
def effWindow (window maxCells : Nat) : Nat := if 0 < window ∧ window < maxCells then window else 0

/-- Cells attended per row. -/
def attendLen (window maxCells : Nat) : Nat :=
  if effWindow window maxCells > 0 then effWindow window maxCells else maxCells

/-- Selection capacity, rounded up to a multiple of 64. -/
def capOf (window maxCells : Nat) : Nat := (attendLen window maxCells + 63) / 64 * 64

/-- The effective window never exceeds the context. -/
theorem effWindow_le (w m : Nat) : effWindow w m ≤ m := by
  unfold effWindow
  split <;> omega

/-- The attended length never exceeds the context. -/
theorem attendLen_le (w m : Nat) : attendLen w m ≤ m := by
  unfold attendLen
  have := effWindow_le w m
  split <;> omega

/-- The capacity covers every attended cell. -/
theorem cap_covers (w m : Nat) : attendLen w m ≤ capOf w m := by
  unfold capOf; omega

/-- The capacity is a multiple of 64. -/
theorem cap_mod (w m : Nat) : capOf w m % 64 = 0 := by
  unfold capOf; omega

/-- Rounding adds fewer than 64 cells. -/
theorem cap_lt (w m : Nat) : capOf w m < attendLen w m + 64 := by
  unfold capOf; omega

-- § step-row staging

/-- Row index of chain step `j`. -/
def chainRow (T j : Nat) : Nat := T + j - 1

/-- Row of the full-layer cell of the catch-up. -/
def fullRow (T : Nat) : Nat := 2 * T - 1

/-- Chain rows lie above the catch-up rows. -/
theorem chainRow_above_catchup (T j : Nat) (hj : 1 ≤ j) : T ≤ chainRow T j := by
  unfold chainRow; omega

/-- Chain rows lie strictly below the full-layer row and inside the 2T rows. -/
theorem chainRow_below_full (T j : Nat) (hj : j ≤ T - 2) (hT : 3 ≤ T) :
    chainRow T j < fullRow T ∧ fullRow T < 2 * T := by
  unfold chainRow fullRow; omega

/-- Distinct chain steps use distinct rows. -/
theorem chainRow_inj (T j k : Nat) (hj : 1 ≤ j) (hk : 1 ≤ k) (h : chainRow T j = chainRow T k) :
    j = k := by
  unfold chainRow at h; omega

/-- The step-row block count of a cell: `(cell + 1) / 4` is monotone in the cell. -/
theorem step_block_mono (a b : Nat) (h : a ≤ b) : (a + 1) / 4 ≤ (b + 1) / 4 := by
  omega

-- § the draft chain

/-- Draft-chain loop with fuel: continue while `j < maxT - 1` and the previous probability `p (j-1)` is high
enough; returns the final `j`, which equals the number of drafts produced. -/
def chainFuel (maxT minP : Nat) (p : Nat → Nat) : Nat → Nat → Nat
  | 0, j => j
  | fuel + 1, j => if j < maxT - 1 ∧ p (j - 1) ≥ minP then chainFuel maxT minP p fuel (j + 1) else j

/-- The chain never produces fewer drafts than it started with. -/
theorem chain_ge (maxT minP : Nat) (p : Nat → Nat) :
    ∀ fuel j, j ≤ chainFuel maxT minP p fuel j
  | 0, j => by simp [chainFuel]
  | fuel + 1, j => by
    unfold chainFuel
    split
    · have := chain_ge maxT minP p fuel (j + 1)
      omega
    · omega

/-- Starting inside the bound, the chain never exceeds `max_t - 1` drafts. -/
theorem chain_le (maxT minP : Nat) (p : Nat → Nat) :
    ∀ fuel j, j ≤ maxT - 1 → chainFuel maxT minP p fuel j ≤ maxT - 1
  | 0, j, h => by simpa [chainFuel] using h
  | fuel + 1, j, h => by
    unfold chainFuel
    split
    · rename_i hc
      exact chain_le maxT minP p fuel (j + 1) (by omega)
    · exact h

/-- A draft whose probability is below the threshold stops the chain at once. -/
theorem chain_stops_on_low (maxT minP : Nat) (p : Nat → Nat) (fuel j : Nat) (h : p (j - 1) < minP) :
    chainFuel maxT minP p (fuel + 1) j = j := by
  unfold chainFuel
  split
  · rename_i hc; omega
  · rfl

-- § argument validity of `draft`

/-- `draft` accepts `1 ≤ T ≤ max_t` and `0 ≤ a < T`. -/
def draftArgsOk (T maxT a : Nat) : Prop := 1 ≤ T ∧ T ≤ maxT ∧ a < T

/-- The caught-up row `a` is a real window row. -/
theorem draftArgs_row (T maxT a : Nat) (h : draftArgsOk T maxT a) : a < maxT := by
  obtain ⟨_, h2, h3⟩ := h
  omega

-- § chunked prefill

/-- Chunk length at offset `c` of `n` prompt cells. -/
def prefillChunk (maxT n c : Nat) : Nat := min maxT (n - c)

/-- While cells remain, a chunk is non-empty. -/
theorem prefillChunk_pos (maxT n c : Nat) (hm : 0 < maxT) (hc : c < n) : 0 < prefillChunk maxT n c := by
  unfold prefillChunk; omega

/-- A chunk never runs past the prompt. -/
theorem prefillChunk_end (maxT n c : Nat) (hc : c ≤ n) : c + prefillChunk maxT n c ≤ n := by
  unfold prefillChunk; omega

/-- A chunk fits the K/V-only graph table (size `max_t`). -/
theorem prefillChunk_le (maxT n c : Nat) : prefillChunk maxT n c ≤ maxT := by
  unfold prefillChunk; omega

/-- A skipped chunk lies entirely below the first needed cell (integer arithmetic). -/
theorem skipped_chunk_unneeded (cell0 c T firstNeeded : Int) (h : cell0 + c + T ≤ firstNeeded)
    (t : Int) (_ht0 : 0 ≤ t) (ht : t < T) : cell0 + c + t < firstNeeded := by
  omega

-- § tensor lookup by name and kind

/-- A dense tensor entry. -/
structure Tensor where
  name : String
  kind : String
  off : Nat

/-- Lookup of `f32`/`bf16`/`q8`: first entry with this name and kind. -/
def findTensor (ts : List Tensor) (name kind : String) : Option Tensor :=
  ts.find? (fun t => t.name == name && t.kind == kind)

/-- A hit always has the requested name and kind. -/
theorem findTensor_sound (ts : List Tensor) (name kind : String) (t : Tensor)
    (h : findTensor ts name kind = some t) : t.name = name ∧ t.kind = kind := by
  have := List.find?_some h
  simpa using this

/-- An empty table finds nothing. -/
theorem findTensor_nil (name kind : String) : findTensor [] name kind = none := rfl

end Strata.MtpDrafter
