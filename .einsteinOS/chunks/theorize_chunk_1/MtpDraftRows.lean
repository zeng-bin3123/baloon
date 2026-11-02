/-!
# MODULE: mtp_draft_rows
Row bookkeeping of the MTP draft layer: arena alignment, window rounding, step-row layout, prefill chunking.

## Source semantics
Staging has `2*T` step rows: rows `[0,T)` catch-up, row `2*max_t - 1` the full-layer row, and chain step `j`
(1..max_t-2) uses row `max_t + j - 1`; these ranges are disjoint. Bump allocations round to 256 bytes. The window rounds
up to 64. Prefill walks the prompt in chunks of at most `max_t` and always makes progress.

## Form template identifiers
- Bump::take            -> bumpUsed
- cap_                  -> cap64
- put(row, cell) rows   -> chainRow / fullRow
- MtpDrafter::prefill   -> chunkT
-/
namespace Strata.MtpDraftRows

def bumpUsed (used n sz : Nat) : Nat := used + (n * sz + 255) / 256 * 256

/-- A bump allocation never overlaps earlier ones. -/
theorem bump_ge (used n sz : Nat) : used + n * sz ≤ bumpUsed used n sz := by unfold bumpUsed; omega
/-- A bump allocation keeps 256-byte alignment. -/
theorem bump_mod (used n sz : Nat) (h : used % 256 = 0) : bumpUsed used n sz % 256 = 0 := by
  unfold bumpUsed; omega

def cap64 (w : Nat) : Nat := (w + 63) / 64 * 64
/-- The attention capacity covers the window. -/
theorem cap64_ge (w : Nat) : w ≤ cap64 w := by unfold cap64; omega
/-- The capacity is a multiple of 64. -/
theorem cap64_mod (w : Nat) : cap64 w % 64 = 0 := by unfold cap64; omega
/-- The capacity wastes under 64 cells. -/
theorem cap64_lt (w : Nat) : cap64 w < w + 64 := by unfold cap64; omega

def fullRow (maxT : Nat) : Nat := 2 * maxT - 1
def chainRow (maxT j : Nat) : Nat := maxT + j - 1

/-- The full-layer row is inside the `2*max_t` staged rows. -/
theorem fullRow_lt {maxT : Nat} (h : 1 ≤ maxT) : fullRow maxT < 2 * maxT := by unfold fullRow; omega

/-- Chain rows are inside the staged rows. -/
theorem chainRow_lt {maxT j : Nat} (h1 : 1 ≤ j) (h2 : j + 2 ≤ maxT) : chainRow maxT j < 2 * maxT := by
  unfold chainRow; omega

/-- Chain rows sit above the catch-up rows `[0, max_t)`. -/
theorem chainRow_ge {maxT j : Nat} (h1 : 1 ≤ j) : maxT ≤ chainRow maxT j := by unfold chainRow; omega

/-- Chain rows never collide with the full-layer row. -/
theorem chainRow_ne_fullRow {maxT j : Nat} (h1 : 1 ≤ j) (h2 : j + 2 ≤ maxT) : chainRow maxT j ≠ fullRow maxT := by
  unfold chainRow fullRow; omega

/-- Distinct chain steps use distinct rows. -/
theorem chainRow_inj {maxT j j' : Nat} (h1 : 1 ≤ j) (h2 : 1 ≤ j') (h : chainRow maxT j = chainRow maxT j') : j = j' := by
  unfold chainRow at h; omega

/-- The full-layer row is above every catch-up row `t < T ≤ max_t`. -/
theorem fullRow_above_catchup {maxT T t : Nat} (hT : T ≤ maxT) (ht : t < T) : t < fullRow maxT := by
  unfold fullRow; omega

def chunkT (maxT n c : Nat) : Nat := min maxT (n - c)

/-- A prefill chunk is non-empty whenever cells remain. -/
theorem chunkT_pos {maxT n c : Nat} (h1 : 1 ≤ maxT) (h2 : c < n) : 1 ≤ chunkT maxT n c := by
  unfold chunkT; omega

/-- A prefill chunk never runs past the prompt. -/
theorem chunkT_le {maxT n c : Nat} : c + chunkT maxT n c ≤ max c n := by unfold chunkT; omega

/-- A chunk never exceeds the staged `max_t` rows. -/
theorem chunkT_le_max (maxT n c : Nat) : chunkT maxT n c ≤ maxT := by unfold chunkT; omega

/-- The draft-vocabulary file is whole 4-byte ids. -/
theorem dvocab_count {size : Nat} (h : size % 4 = 0) : size / 4 * 4 = size := by omega

end Strata.MtpDraftRows
