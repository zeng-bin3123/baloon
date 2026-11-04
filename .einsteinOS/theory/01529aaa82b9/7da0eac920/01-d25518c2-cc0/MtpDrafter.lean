/-!
# MODULE: mtp_drafter
The multi-token-prediction draft layer: batching of prefill rows, draft counts per round, attention-window
size, and the VRAM budget of the resident experts.

## Source semantics
* A round of window size `T` produces `T - 1` drafts; the graph tables are indexed by `T <= 8`.
* Prefill runs in batches of up to `max_t` rows; the batch count covers all rows.
* Attention is dense over every cell seen, or over the last `window` cells when `window > 0`; it never
  attends more cells than exist.
* All 512 routed experts live in VRAM: 512 blobs of 1,382,400 bytes.

## Form template identifiers
* `MtpDrafter::draft` -> `draftCount`
* `MtpDrafter::prefill` -> `batches`
* `window_` -> `attended`
* `step_exec_[9]` -> `execIndexOk`
-/

namespace Strata.MtpDrafter

-- § draft rounds

/-- Number of drafts a round of window size `T` produces. -/
def draftCount (T : Nat) : Nat := T - 1

/-- A round with `T ≥ 1` produces strictly fewer drafts than window rows. -/
theorem draftCount_lt (T : Nat) (h : 1 ≤ T) : draftCount T < T := by
  unfold draftCount
  omega

/-- The graph tables have 9 entries, so any `T` from 1 to 8 is a valid index. -/
def execIndexOk (T : Nat) : Bool := decide (1 ≤ T ∧ T < 9)

/-- Window sizes 1 through 8 index inside the 9-entry executable tables. -/
theorem exec_index_ok_range (T : Nat) (h1 : 1 ≤ T) (h2 : T ≤ 8) : execIndexOk T = true := by
  unfold execIndexOk
  exact decide_eq_true (by omega)

/-- The first draft cell is `p + a + 1`, and it is never beyond one past the window when `a < T`. -/
theorem first_draft_cell (p a T : Nat) (ha : a < T) : p + a + 1 ≤ p + T := by
  omega

/-- The catch-up covers `T` cells from `p`, so the last cell is `p + T - 1`. -/
theorem catchup_last_cell (p T : Nat) (h : 1 ≤ T) : p + (T - 1) + 1 = p + T := by
  omega

/-- Mapped step records hold `2 * max_t` rows: any row index for `t < T ≤ max_t` is inside. -/
theorem step_rows_fit (t T maxT : Nat) (ht : t < T) (hT : T ≤ maxT) : t < 2 * maxT := by
  omega

-- § prefill batching

/-- Number of batches needed to push `n` rows through `max_t` rows at a time. -/
def batches (n m : Nat) : Nat := (n + m - 1) / m

/-- The batches cover every row. -/
theorem batches_cover (n m : Nat) (hm : 0 < m) : n ≤ batches n m * m := by
  unfold batches
  have h := Nat.div_add_mod (n + m - 1) m
  have h2 := Nat.mod_lt (n + m - 1) hm
  have h3 : (n + m - 1) / m * m = m * ((n + m - 1) / m) := Nat.mul_comm _ _
  omega

/-- The batches overshoot by less than one batch width: no wholly empty batch is issued. -/
theorem batches_tight (n m : Nat) (hm : 0 < m) : batches n m * m < n + m := by
  unfold batches
  have h := Nat.div_add_mod (n + m - 1) m
  have h3 : (n + m - 1) / m * m = m * ((n + m - 1) / m) := Nat.mul_comm _ _
  omega

/-- No rows, no batches. -/
theorem batches_zero (m : Nat) (hm : 0 < m) : batches 0 m = 0 := by
  unfold batches
  exact Nat.div_eq_of_lt (by omega)

-- § attention window

/-- Cells attended: all cells when `window = 0`, otherwise the most recent `window` cells. -/
def attended (n w : Nat) : Nat := if w = 0 then n else min n w

/-- The draft layer never attends more cells than it has seen. -/
theorem attended_le (n w : Nat) : attended n w ≤ n := by
  unfold attended
  split <;> omega

/-- With a window, it never attends more than the window. -/
theorem attended_le_window (n w : Nat) (h : 0 < w) : attended n w ≤ w := by
  unfold attended
  have : ¬ (w = 0) := by omega
  simp only [this, if_false]
  omega

/-- When the sequence is shorter than the window, attention is dense over all cells. -/
theorem attended_dense (n w : Nat) (h : n ≤ w) : attended n w = n := by
  unfold attended
  split <;> omega

/-- The default window of 32768 cells exceeds the 2,051-cell point where sparse selection starts. -/
theorem default_window_dense_below_sparse : 2051 ≤ 32768 := by
  decide

-- § accepted drafts

/-- Number of leading drafts that agree with the verified greedy tokens. -/
def acceptLen : List Nat → List Nat → Nat
  | d :: ds, g :: gs => if d = g then 1 + acceptLen ds gs else 0
  | _, _ => 0

/-- Acceptance never exceeds the number of drafts offered. -/
theorem acceptLen_le (ds gs : List Nat) : acceptLen ds gs ≤ ds.length := by
  induction ds generalizing gs with
  | nil => simp [acceptLen]
  | cons d ds ih =>
    cases gs with
    | nil => simp [acceptLen]
    | cons g gs =>
      have h := ih gs
      simp only [acceptLen]
      split <;> simp only [List.length_cons] <;> omega

/-- Acceptance never exceeds the number of verified tokens. -/
theorem acceptLen_le_right (ds gs : List Nat) : acceptLen ds gs ≤ gs.length := by
  induction ds generalizing gs with
  | nil => simp [acceptLen]
  | cons d ds ih =>
    cases gs with
    | nil => simp [acceptLen]
    | cons g gs =>
      have h := ih gs
      simp only [acceptLen]
      split <;> simp only [List.length_cons] <;> omega

-- § VRAM budget

/-- All 512 routed experts resident: 707,788,800 bytes, about 708 MB. -/
theorem experts_vram_bytes : 512 * 1382400 = 707788800 := by
  decide

/-- That is below the 0.9 GB the drafter's own K/V and buffers are quoted at, plus the 708 MB. -/
theorem experts_below_one_gb : 512 * 1382400 < 1000000000 := by
  decide

end Strata.MtpDrafter
