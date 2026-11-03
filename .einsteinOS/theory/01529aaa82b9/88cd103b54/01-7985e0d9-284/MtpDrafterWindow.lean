/-!
# MODULE: mtp_drafter_window
Index arithmetic of the multi-token-prediction drafter: the 256-byte bump allocator, the attention window
capacity, the staged step-row layout of the captured round and chain graphs, and the prefill chunking.

## Source semantics
- The bump allocator rounds every take up to 256 bytes; a first pass counts, a second pass carves.
- The attention window `cap` is the effective window (the whole context when no window or window >= context)
  rounded up to a multiple of 64.
- Step rows: `T` catch-up rows `0..T-1`, chain step `j` (1..T-2) at row `T + j - 1`, and the full layer's
  row `a` at the dedicated row `2T - 1`; all below `2T`, pairwise distinct.
- Prefill walks the prompt in chunks of at most `max_t`; a chunk wholly before `first_needed` is skipped.
- The residual-mixing GEMV is issued in chunks of 8 columns.

## Form template identifiers
- `bumpAdvance`  <- `Bump::take`
- `effWindow`, `capOf` <- `window_` and `cap_` in `MtpDrafter::load`
- `chainRow`, `raRow`  <- `max_t_ + j - 1` and `2 * max_t_ - 1`
- `chunkLen`     <- `std::min(8, T*HC - c0)` and `std::min(max_t_, n - c)`
-/

namespace Strata.Mtp

/-- Bump-allocator advance: `n * size` bytes rounded up to 256. -/
def bumpAdvance (used n size : Nat) : Nat := used + (n * size + 255) / 256 * 256

/-- The effective window: the context when the window is absent or not smaller than it. -/
def effWindow (w maxCells : Nat) : Nat := if 0 < w ∧ w < maxCells then w else maxCells

/-- Attention capacity: effective window rounded up to 64. -/
def capOf (w maxCells : Nat) : Nat := (effWindow w maxCells + 63) / 64 * 64

/-- Step row of chain step `j`. -/
def chainRow (maxT j : Nat) : Nat := maxT + j - 1

/-- Step row reserved for the full layer's row `a`. -/
def raRow (maxT : Nat) : Nat := 2 * maxT - 1

/-- Chunk length of a stride-`c` walk over `total` items. -/
def chunkLen (stride total c : Nat) : Nat := min stride (total - c)

/-- The bump allocator keeps every region 256-aligned. -/
theorem bumpAdvance_mod (used n size : Nat) (h : used % 256 = 0) : bumpAdvance used n size % 256 = 0 := by
  unfold bumpAdvance
  omega

/-- A take covers the requested bytes. -/
theorem bumpAdvance_covers (used n size : Nat) : used + n * size ≤ bumpAdvance used n size := by
  unfold bumpAdvance
  omega

/-- The effective window never exceeds the context. -/
theorem effWindow_le (w maxCells : Nat) : effWindow w maxCells ≤ maxCells := by
  unfold effWindow
  split <;> omega

/-- The capacity covers the effective window. -/
theorem effWindow_le_cap (w maxCells : Nat) : effWindow w maxCells ≤ capOf w maxCells := by
  unfold capOf
  omega

/-- The capacity is a multiple of 64. -/
theorem capOf_mod (w maxCells : Nat) : capOf w maxCells % 64 = 0 := by
  unfold capOf
  omega

/-- Chain step rows lie above the catch-up rows. -/
theorem chainRow_ge (maxT j : Nat) (h : 1 ≤ j) : maxT ≤ chainRow maxT j := by
  unfold chainRow
  omega

/-- Chain step rows stay strictly below the full-layer row, so they never collide with it. -/
theorem chainRow_lt_ra (maxT j : Nat) (h1 : 1 ≤ j) (h2 : j + 2 ≤ maxT) :
    chainRow maxT j < raRow maxT := by
  unfold chainRow raRow
  omega

/-- Two distinct chain steps use distinct rows. -/
theorem chainRow_inj (maxT j j' : Nat) (h1 : 1 ≤ j) (h2 : 1 ≤ j') (h : chainRow maxT j = chainRow maxT j') :
    j = j' := by
  unfold chainRow at h
  omega

/-- The reserved row is inside the `2T`-row staging buffers. -/
theorem raRow_lt (maxT : Nat) (h : 1 ≤ maxT) : raRow maxT < 2 * maxT := by
  unfold raRow
  omega

/-- A chunk is non-empty and bounded by the stride while items remain. -/
theorem chunkLen_bounds (stride total c : Nat) (hs : 0 < stride) (hc : c < total) :
    0 < chunkLen stride total c ∧ chunkLen stride total c ≤ stride := by
  unfold chunkLen
  omega

/-- A chunk never overruns the items. -/
theorem chunkLen_fits (stride total c : Nat) (hc : c ≤ total) : c + chunkLen stride total c ≤ total := by
  unfold chunkLen
  omega

/-- A chunk wholly before `first_needed` skips every cell it contains. -/
theorem skip_chunk_cells (cell0 c T first : Nat) (h : cell0 + c + T ≤ first) :
    ∀ t, t < T → cell0 + c + t < first := by
  intro t ht
  omega

/-- The committed catch-up count: row `a` is a valid catch-up row when `a < T`. -/
theorem catchup_row_valid (a T : Nat) (h : a < T) : a < 2 * T := by
  omega

end Strata.Mtp
