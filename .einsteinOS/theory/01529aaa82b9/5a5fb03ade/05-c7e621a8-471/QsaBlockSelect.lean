/-!
# MODULE: qsa_block_select
Block-level scoring and radix-select top-k for the batched QSA path (`src/kernels/cuda/qsa_select.cu`).
Scores are per pooled block of 4 cells; the selection is made over blocks weighted by their cell count
and then expanded to cell ids.

## Source semantics
* Block `b < n_bid` has weight 4; the last block `b = n_bid` has weight `n_kv - 4 n_bid` (possibly 0).
* The radix select fixes 8 key bits at a time (shift 24, 16, 8, 0); each pass scans digits from 255 down
  and stops at the first digit whose cumulative weight reaches `width`.
* Equal-to-threshold cells are shared out lowest index first: each thread takes
  `clamp (budget - eq_before) 0 eq`.

## Form template identifiers
* `weight`     <- lambda `weight(b)`
* `sumW`       <- total weight of a block prefix
* `pick`       <- the digit-descent loop of one radix pass
* `eqShare`    <- `my_eq`
-/

namespace QsaBlockSelect

-- § block weights

/-- Weight (cell count) of block `b`. -/
def weight (nKv nBid b : Nat) : Nat := if b < nBid then 4 else nKv - nBid * 4

/-- Total weight of the first `m` blocks. -/
def sumW (nKv nBid : Nat) : Nat → Nat
  | 0 => 0
  | m + 1 => sumW nKv nBid m + weight nKv nBid m

/-- Full blocks weigh 4 each. -/
theorem sumW_prefix (nKv nBid m : Nat) (h : m ≤ nBid) : sumW nKv nBid m = 4 * m := by
  induction m with
  | zero => simp [sumW]
  | succ m ih =>
    have hm : m < nBid := by omega
    simp only [sumW]
    rw [ih (by omega)]
    unfold weight
    rw [if_pos hm]
    omega

/-- Blocks `0 .. n_kv/4` together weigh exactly `n_kv`. -/
theorem sumW_total (nKv : Nat) : sumW nKv (nKv / 4) (nKv / 4 + 1) = nKv := by
  simp only [sumW]
  rw [sumW_prefix nKv (nKv / 4) (nKv / 4) (Nat.le_refl _)]
  unfold weight
  rw [if_neg (by omega)]
  omega

/-- Every expanded cell id `4b + c` is a valid cell. -/
theorem cell_lt (nKv b c : Nat) (hb : b ≤ nKv / 4) (hc : c < weight nKv (nKv / 4) b) :
    b * 4 + c < nKv := by
  unfold weight at hc
  split at hc <;> omega

/-- Every cell belongs to some block and offset within that block's weight. -/
theorem cell_covered (nKv j : Nat) (hj : j < nKv) :
    j / 4 ≤ nKv / 4 ∧ j % 4 < weight nKv (nKv / 4) (j / 4) := by
  unfold weight
  split <;> omega

/-- Cell ids are recovered from (block, offset). -/
theorem cell_roundtrip (j : Nat) : j / 4 * 4 + j % 4 = j := by
  omega

/-- The `+1e9` bonus applies exactly when the last block is a non-empty incomplete tail. -/
theorem tail_nonempty_iff (nKv : Nat) : 0 < weight nKv (nKv / 4) (nKv / 4) ↔ nKv % 4 ≠ 0 := by
  unfold weight
  rw [if_neg (by omega)]
  omega

-- § radix select

/-- The four fixed digits assembled into the 32-bit threshold. -/
def prefix4 (d3 d2 d1 d0 : Nat) : Nat := d3 * 16777216 + d2 * 65536 + d1 * 256 + d0

/-- The assembled threshold fits in 32 bits. -/
theorem prefix4_lt (d3 d2 d1 d0 : Nat) (h3 : d3 < 256) (h2 : d2 < 256) (h1 : d1 < 256)
    (h0 : d0 < 256) : prefix4 d3 d2 d1 d0 < 4294967296 := by
  unfold prefix4
  omega

/-- Each pass's digit is recoverable from the assembled threshold. -/
theorem prefix4_digits (d3 d2 d1 d0 : Nat) (h3 : d3 < 256) (h2 : d2 < 256) (h1 : d1 < 256)
    (h0 : d0 < 256) :
    prefix4 d3 d2 d1 d0 / 16777216 = d3 ∧
    prefix4 d3 d2 d1 d0 / 65536 % 256 = d2 ∧
    prefix4 d3 d2 d1 d0 / 256 % 256 = d1 ∧
    prefix4 d3 d2 d1 d0 % 256 = d0 := by
  unfold prefix4
  omega

/-- One radix pass: descend from digit `d`, accumulating weight; stop when the target is reached. -/
def pick (hist : Nat → Nat) (width : Nat) : Nat → Nat → Nat × Nat
  | 0, cum => (0, cum)
  | d + 1, cum =>
    if width ≤ cum + hist (d + 1) then (d + 1, cum)
    else pick hist width d (cum + hist (d + 1))

/-- The weight strictly above the chosen digit is always below `width`. -/
theorem pick_below (hist : Nat → Nat) (width : Nat) :
    ∀ d cum, cum < width → (pick hist width d cum).2 < width := by
  intro d
  induction d with
  | zero => intro cum h; simpa [pick] using h
  | succ d ih =>
    intro cum h
    by_cases hc : width ≤ cum + hist (d + 1)
    · simpa [pick, hc] using h
    · have := ih (cum + hist (d + 1)) (by omega)
      simpa [pick, hc] using this

/-- The chosen digit never exceeds the start digit. -/
theorem pick_le (hist : Nat → Nat) (width : Nat) :
    ∀ d cum, (pick hist width d cum).1 ≤ d := by
  intro d
  induction d with
  | zero => intro cum; simp [pick]
  | succ d ih =>
    intro cum
    by_cases hc : width ≤ cum + hist (d + 1)
    · simp [pick, hc]
    · have := ih (cum + hist (d + 1))
      simp only [pick, if_neg hc]
      omega

-- § equal-cell sharing

/-- Equal-to-threshold cells granted to a thread: budget left after earlier threads, capped by its own. -/
def eqShare (budget before eq : Nat) : Nat := min eq (budget - before)

/-- A thread never takes more than it owns. -/
theorem eqShare_le (budget before eq : Nat) : eqShare budget before eq ≤ eq := by
  unfold eqShare
  omega

/-- Earlier plus own share never overshoots the budget (unless earlier threads already did). -/
theorem eqShare_budget (budget before eq : Nat) :
    before + eqShare budget before eq ≤ max budget before := by
  unfold eqShare
  omega

/-- When the budget covers everything, the thread takes all of its equal cells. -/
theorem eqShare_full (budget before eq : Nat) (h : before + eq ≤ budget) :
    eqShare budget before eq = eq := by
  unfold eqShare
  omega

end QsaBlockSelect
