/-!
# MODULE: mtp_drafter_window
Row bookkeeping of the multi-token-prediction draft layer: staging-row layout for catch-up and chain steps,
attention capacity rounding, bump-allocator alignment, prefill chunking, and draft-count bounds.

## Source semantics
With window `T ≤ max_t` there are `2*max_t` step/pos rows. Catch-up rows are `0..T-1`; the full-layer row for the
accepted cell is `2*max_t-1`; chain step `j ∈ [1, max_t-2]` uses row `max_t+j-1`. These sets never collide. The
attention capacity is rounded up to a multiple of 64. Every bump allocation is rounded up to 256 bytes.

## Form template identifiers
- stepRowCatch / stepRowAccept / stepRowChain ← step row indices in `capture_round`/`capture_step`,
- capRound ← `cap_`, bumpTake ← `Bump::take`, chunkLen ← `T = min(max_t_, n - c)` in `prefill`.
-/

namespace Strata.Mtp

-- § step rows

/-- Row of the full layer for the accepted cell. -/
def stepRowAccept (maxT : Nat) : Nat := 2 * maxT - 1

/-- Row of chain step `j`. -/
def stepRowChain (maxT j : Nat) : Nat := maxT + j - 1

/-- The accepted-cell row lies inside the `2*max_t` rows allocated. -/
theorem accept_row_lt (maxT : Nat) (h : 1 ≤ maxT) : stepRowAccept maxT < 2 * maxT := by
  unfold stepRowAccept; omega

/-- Chain rows are inside the allocation. -/
theorem chain_row_lt (maxT j : Nat) (hj : 1 ≤ j) (hjm : j + 2 ≤ maxT) :
    stepRowChain maxT j < 2 * maxT - 1 := by
  unfold stepRowChain; omega

/-- Chain rows never collide with the accepted-cell row. -/
theorem chain_ne_accept (maxT j : Nat) (hj : 1 ≤ j) (hjm : j + 2 ≤ maxT) :
    stepRowChain maxT j ≠ stepRowAccept maxT := by
  unfold stepRowChain stepRowAccept; omega

/-- Catch-up rows `t < T ≤ max_t` never collide with chain rows. -/
theorem catchup_ne_chain (maxT T t j : Nat) (hT : T ≤ maxT) (ht : t < T) (hj : 1 ≤ j) :
    t ≠ stepRowChain maxT j := by
  unfold stepRowChain; omega

/-- Catch-up rows never collide with the accepted-cell row. -/
theorem catchup_ne_accept (maxT T t : Nat) (hT : T ≤ maxT) (ht : t < T) :
    t ≠ stepRowAccept maxT := by
  unfold stepRowAccept; omega

/-- Distinct chain steps use distinct rows. -/
theorem chain_injective (maxT j k : Nat) (hj : 1 ≤ j) (hk : 1 ≤ k)
    (h : stepRowChain maxT j = stepRowChain maxT k) : j = k := by
  unfold stepRowChain at h; omega

/-- The chain loop condition `j < max_t - 1` keeps the draft count at most `max_t - 1`. -/
theorem draft_count_le (maxT j : Nat) (hj : j < maxT - 1) : j + 1 ≤ maxT - 1 := by
  omega

-- § capacity and allocation

/-- The cells the attention covers: the window when it is positive and below `max_cells`, else all cells. -/
def coveredCells (window maxCells : Nat) : Nat :=
  if 0 < window ∧ window < maxCells then window else maxCells

/-- Attention capacity `cap_`: covered cells rounded up to a multiple of 64. -/
def capRound (window maxCells : Nat) : Nat := (coveredCells window maxCells + 63) / 64 * 64

/-- Capacity is a multiple of 64. -/
theorem capRound_mod (w m : Nat) : capRound w m % 64 = 0 := by
  unfold capRound; omega

/-- Capacity covers every cell that attention may select. -/
theorem capRound_ge (w m : Nat) : coveredCells w m ≤ capRound w m := by
  unfold capRound; omega

/-- Capacity wastes fewer than 64 cells. -/
theorem capRound_lt (w m : Nat) : capRound w m < coveredCells w m + 64 := by
  unfold capRound; omega

/-- Bytes consumed by one bump allocation of `b = n * sizeof(T)` bytes. -/
def bumpTake (b : Nat) : Nat := (b + 255) / 256 * 256

/-- A bump slot holds its request. -/
theorem bumpTake_ge (b : Nat) : b ≤ bumpTake b := by
  unfold bumpTake; omega

/-- Bump slots keep the arena 256-byte aligned. -/
theorem bumpTake_mod (b : Nat) : bumpTake b % 256 = 0 := by
  unfold bumpTake; omega

/-- A bump slot wastes fewer than 256 bytes. -/
theorem bumpTake_lt (b : Nat) : bumpTake b < b + 256 := by
  unfold bumpTake; omega

-- § prefill chunking

/-- Length of the chunk starting at `c`. -/
def chunkLen (maxT n c : Nat) : Nat := min maxT (n - c)

/-- A chunk never runs past the prompt. -/
theorem chunk_within (maxT n c : Nat) : c + chunkLen maxT n c ≤ max c n := by
  unfold chunkLen; omega

/-- While cells remain, a chunk is nonempty (so the loop makes progress). -/
theorem chunk_pos (maxT n c : Nat) (hT : 1 ≤ maxT) (hc : c < n) : 0 < chunkLen maxT n c := by
  unfold chunkLen; omega

/-- A chunk fits the captured graph's width. -/
theorem chunk_le (maxT n c : Nat) : chunkLen maxT n c ≤ maxT := by
  unfold chunkLen; omega

/-- The staged step record `(cell+1)/4` never exceeds `cell+1`. -/
theorem step_block_le (cell : Nat) : (cell + 1) / 4 ≤ cell + 1 := by
  omega

end Strata.Mtp
