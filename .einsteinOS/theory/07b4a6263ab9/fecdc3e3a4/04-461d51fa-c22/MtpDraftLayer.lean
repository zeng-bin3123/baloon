/-!
# MODULE: mtp_draft_layer
SOURCES: src/core/mtp.cpp
The multi-token-prediction draft layer: one attention + MoE block with its own K/V cache, captured as a
prefill graph (K/V only), a round graph (catch-up K/V for the window plus one full row) and one graph per
chain step.  The host stages step rows `[cell, cell+1, (cell+1)/4, cell+1]` and positions; the chain stops
as soon as a draft's probability falls below `min_p`.

## Source semantics
* The device arena is carved by a bump allocator rounding every region up to 256 bytes.
* Step rows: rows `0..T-1` are the catch-up window, rows `max_t + j - 1` (j = 1..max_t-2) the chain steps,
  and row `2*max_t - 1` the round's full row; these three groups never collide and all fit the
  `2 * max_t` rows allocated.
* The attention width `cap` is the window (or the context) rounded up to a multiple of 64; the window is
  used only when `0 < window < max_cells`.
* The chain emits 1 + (the number of leading steps whose previous probability passed `min_p`) drafts,
  never more than `max_t - 1`; unused draft slots are zeroed.
* `prefill` skips chunks the window can never reach again (`cell0 + c + T ≤ first_needed`).

## Form template identifiers
bumpTake ↔ Bump::take · stepRow ↔ put()/h_step_ fill · catchupRow/chainRow/fullRow ↔ t, max_t+j-1,
2*max_t-1 · capOf ↔ cap_ · windowOf ↔ window_ · chainLen ↔ the `pj >= min_p` loop
-/

namespace MtpDraftLayer

-- § the bump allocator

/-- Bytes a region of `n` bytes consumes from the bump arena. -/
def bumpTake (n : Nat) : Nat := (n + 255) / 256 * 256

/-- Region offsets of a sequence of requests, starting from `used`. -/
def bumpOffsets : List Nat → Nat → List Nat
  | [], _ => []
  | n :: ns, used => used :: bumpOffsets ns (used + bumpTake n)

/-- Final `used` after a sequence of requests (the counted pass, `count.used`). -/
def bumpUsed : List Nat → Nat → Nat
  | [], used => used
  | n :: ns, used => bumpUsed ns (used + bumpTake n)

/-- Every region is at least as large as requested. -/
theorem bumpTake_ge (n : Nat) : n ≤ bumpTake n := by
  unfold bumpTake; omega

/-- Every region is a multiple of 256 bytes. -/
theorem bumpTake_mod (n : Nat) : bumpTake n % 256 = 0 := by
  unfold bumpTake; omega

/-- All offsets are 256-aligned when the arena base is. -/
theorem offsets_aligned :
    ∀ (ns : List Nat) (u : Nat), u % 256 = 0 → ∀ x ∈ bumpOffsets ns u, x % 256 = 0
  | [], _, _, x, hx => by simp [bumpOffsets] at hx
  | n :: ns, u, hu, x, hx => by
    simp only [bumpOffsets, List.mem_cons] at hx
    rcases hx with rfl | hx
    · exact hu
    · have := bumpTake_mod n
      exact offsets_aligned ns (u + bumpTake n) (by omega) x hx

/-- Every region of the real carve ends inside the arena sized by the counted carve: the two passes
walk the same sequence, so `cudaMalloc(count.used)` holds them all. -/
theorem regions_in_arena :
    ∀ (ns : List Nat) (u : Nat), ∀ x ∈ bumpOffsets ns u, u ≤ x ∧ x ≤ bumpUsed ns u
  | [], _, x, hx => by simp [bumpOffsets] at hx
  | n :: ns, u, x, hx => by
    simp only [bumpOffsets, List.mem_cons] at hx
    simp only [bumpUsed]
    have hmono : ∀ (ms : List Nat) (v : Nat), v ≤ bumpUsed ms v := by
      intro ms
      induction ms with
      | nil => intro v; exact Nat.le_refl v
      | cons m ms ih => intro v; simp only [bumpUsed]; have := ih (v + bumpTake m); omega
    rcases hx with rfl | hx
    · have := hmono ns (x + bumpTake n)
      omega
    · have := regions_in_arena ns (u + bumpTake n) x hx
      omega

-- § step rows

/-- The four step-state integers the host stages for `cell`. -/
def stepRow (cell : Nat) : List Nat := [cell, cell + 1, (cell + 1) / 4, cell + 1]

/-- The K/V count of a staged row is one past the cell: the new cell is attended to. -/
theorem stepRow_kv (cell : Nat) : (stepRow cell)[1]? = some (cell + 1) := rfl

/-- The pooled-block count is the K/V count divided by the 4-cell index block. -/
theorem stepRow_blocks (cell : Nat) : (stepRow cell)[2]? = some ((cell + 1) / 4) := rfl

/-- Catch-up rows (`t < T ≤ max_t`) never collide with chain rows (`max_t + j - 1`, `j ≥ 1`). -/
theorem catchup_below_chain (t T maxT j : Nat) (ht : t < T) (hT : T ≤ maxT) (hj : 1 ≤ j) :
    t < maxT + j - 1 := by
  omega

/-- Chain rows stay below the round's full row `2 * max_t - 1`. -/
theorem chain_below_full (maxT j : Nat) (hj : 1 ≤ j) (hj2 : j + 2 ≤ maxT) :
    maxT + j - 1 < 2 * maxT - 1 := by
  omega

/-- The full row fits the `2 * max_t` rows the mapped staging holds. -/
theorem full_row_fits (maxT : Nat) (h : 1 ≤ maxT) : 2 * maxT - 1 < 2 * maxT := by
  omega

-- § attention width and window

/-- The draft attention's selection width: window or context, rounded up to 64. -/
def capOf (cells : Nat) : Nat := (cells + 63) / 64 * 64

/-- The width covers every cell the layer may attend to. -/
theorem cap_covers (cells : Nat) : cells ≤ capOf cells := by
  unfold capOf; omega

/-- The width is a whole number of 64-cell tiles. -/
theorem cap_tiles (cells : Nat) : capOf cells % 64 = 0 := by
  unfold capOf; omega

/-- The effective window: the requested one if strictly inside the context, else 0 (every cell). -/
def windowOf (window maxCells : Nat) : Nat :=
  if 0 < window ∧ window < maxCells then window else 0

/-- An effective window is always smaller than the context. -/
theorem window_lt_context (w m : Nat) (hm : 0 < m) : windowOf w m < m := by
  unfold windowOf
  split <;> omega

/-- The attention width never exceeds the context's own width. -/
theorem cap_le_context_cap (w m : Nat) (hm : 0 < m) :
    capOf (if windowOf w m > 0 then windowOf w m else m) ≤ capOf m := by
  have hw := window_lt_context w m hm
  unfold capOf
  split <;> omega

-- § the draft chain

/-- Leading steps whose gate passed (`pj >= min_p`), stopping at the first failure. -/
def leading : List Bool → Nat
  | [] => 0
  | b :: bs => if b then 1 + leading bs else 0

/-- Drafts produced: the round's first plus the leading chain steps. -/
def chainLen (gates : List Bool) : Nat := 1 + leading gates

/-- The leading count never exceeds the number of chain steps. -/
theorem leading_le : ∀ bs : List Bool, leading bs ≤ bs.length
  | [] => Nat.le_refl 0
  | b :: bs => by
    simp only [leading, List.length_cons]
    have := leading_le bs
    split <;> omega

/-- With at most `max_t - 2` chain steps the chain yields at most `max_t - 1` drafts, the size of `drafts`. -/
theorem chain_fits (gates : List Bool) (maxT : Nat) (h : gates.length + 2 ≤ maxT) :
    chainLen gates ≤ maxT - 1 := by
  unfold chainLen
  have := leading_le gates
  omega

/-- A first gate that fails stops the chain at one draft. -/
theorem first_fail_one (bs : List Bool) : chainLen (false :: bs) = 1 := by
  simp [chainLen, leading]

/-- Every chain yields at least one draft. -/
theorem chain_pos (gates : List Bool) : 1 ≤ chainLen gates := by
  unfold chainLen; omega

-- § prefill skipping

/-- A prefill chunk is skipped when every cell in it is below the first cell the window needs. -/
def skipChunk (cell0 c T firstNeeded : Nat) : Prop := cell0 + c + T ≤ firstNeeded

/-- A skipped chunk holds no cell the window will read. -/
theorem skipped_unneeded (cell0 c T f t : Nat) (h : skipChunk cell0 c T f) (ht : t < T) :
    cell0 + c + t < f := by
  unfold skipChunk at h; omega

end MtpDraftLayer
