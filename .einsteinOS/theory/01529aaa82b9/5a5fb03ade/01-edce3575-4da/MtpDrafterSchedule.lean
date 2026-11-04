/-!
# MODULE: mtp_drafter_schedule
The multi-token-prediction draft layer's scheduling arithmetic (`mtp.cpp`): staging row layout for the round
graph and chain steps, the attention window capacity, prompt prefill chunking, the draft chain length bound,
and the 256-byte bump allocator.

## Source semantics
* Staging rows: catch-up rows are `[0, T)`, chain step `j` uses row `max_t + j - 1`, and the full-layer row for
  catch-up cell `a` is `2*max_t - 1`; the three groups are disjoint and all below `2*max_t`.
* `cap = ((window or max_cells) + 63) / 64 * 64` covers the window and is a multiple of 64.
* Prefill walks the prompt in chunks of at most `max_t`; chunk sizes are positive and sum to `n`.
* A draft chain produces at most `max_t - 1` drafts.
* The bump allocator aligns each region to 256 B and its counting pass equals the carving pass.

## Form template identifiers
* `h_step_` / `step_` rows -> `catchupRow`, `chainRow`, `fullRow`; `R2 = 2*T` -> `rows2`
* `cap_` -> `capOf`; `prefill` loop -> `chunks`; `draft` loop -> `chain`; `Bump::take` -> `bumpTake`
-/

namespace MtpDrafterSchedule

-- § staging rows

/-- Chain step `j` (1 <= j <= max_t - 2) uses row `max_t + j - 1`. -/
def chainRow (maxT j : Nat) : Nat := maxT + j - 1

/-- The full-layer row for the catch-up cell. -/
def fullRow (maxT : Nat) : Nat := 2 * maxT - 1

/-- Chain rows start at `max_t`, so they never collide with catch-up rows `[0, T)` for `T <= max_t`. -/
theorem chain_row_ge (maxT j : Nat) (hj : 1 ≤ j) : maxT ≤ chainRow maxT j := by
  unfold chainRow; omega

/-- Chain rows stay strictly below the full-layer row. -/
theorem chain_row_lt_full (maxT j : Nat) (hj : j + 1 ≤ maxT - 1) (hm : 2 ≤ maxT) :
    chainRow maxT j < fullRow maxT := by
  unfold chainRow fullRow; omega

/-- The full-layer row is inside the `2*max_t` rows allocated for steps and positions. -/
theorem full_row_lt (maxT : Nat) (hm : 1 ≤ maxT) : fullRow maxT < 2 * maxT := by
  unfold fullRow; omega

/-- Catch-up rows `t < T <= max_t` are below the first chain row. -/
theorem catchup_lt_chain (maxT T t j : Nat) (hT : T ≤ maxT) (ht : t < T) (hj : 1 ≤ j) :
    t < chainRow maxT j := by
  unfold chainRow; omega

-- § window capacity

/-- The attention capacity: the effective window rounded up to a multiple of 64. -/
def capOf (eff : Nat) : Nat := (eff + 63) / 64 * 64

/-- Capacity covers the window. -/
theorem capOf_ge (eff : Nat) : eff ≤ capOf eff := by unfold capOf; omega

/-- Capacity is a multiple of 64. -/
theorem capOf_mod (eff : Nat) : capOf eff % 64 = 0 := by unfold capOf; omega

/-- Capacity wastes fewer than 64 cells. -/
theorem capOf_lt (eff : Nat) : capOf eff < eff + 64 := by unfold capOf; omega

-- § prefill chunking

/-- Chunk sizes of the prefill loop: `min max_t remaining`, with fuel to make recursion structural. -/
def chunks (m : Nat) : Nat → Nat → List Nat
  | 0, _ => []
  | f + 1, n => if n = 0 then [] else min m n :: chunks m f (n - min m n)

/-- With enough fuel and `max_t > 0`, the chunks sum to the number of prompt rows. -/
theorem chunks_sum (m : Nat) (hm : 0 < m) : ∀ (f n : Nat), n ≤ f → (chunks m f n).sum = n
  | 0, n, h => by
    have : n = 0 := by omega
    subst this
    simp [chunks]
  | f + 1, n, h => by
    unfold chunks
    by_cases hn : n = 0
    · simp [hn]
    · rw [if_neg hn]
      have h1 : 0 < min m n := by omega
      have h2 : n - min m n ≤ f := by omega
      rw [List.sum_cons, chunks_sum m hm f (n - min m n) h2]
      omega

/-- Every chunk is at most `max_t` and positive. -/
theorem chunks_bound (m : Nat) (hm : 0 < m) : ∀ (f n : Nat) (c : Nat), c ∈ chunks m f n → 0 < c ∧ c ≤ m
  | 0, n, c, h => by simp [chunks] at h
  | f + 1, n, c, h => by
    unfold chunks at h
    by_cases hn : n = 0
    · simp [hn] at h
    · rw [if_neg hn] at h
      rw [List.mem_cons] at h
      rcases h with h | h
      · subst h; omega
      · exact chunks_bound m hm f _ c h

-- § draft chain

/-- Chain bookkeeping: `j` is the next step, `n` the drafts so far; stops at `max_t - 1` or when unlikely. -/
def chain (maxT : Nat) (lik : Nat → Bool) : Nat → Nat → Nat → Nat
  | 0, _, n => n
  | f + 1, j, n => if j < maxT - 1 && lik (j - 1) then chain maxT lik f (j + 1) (n + 1) else n

/-- Starting from one draft at step 1, the chain never yields more than `max_t - 1` drafts. -/
theorem chain_le (maxT : Nat) (lik : Nat → Bool) :
    ∀ (f j : Nat), chain maxT lik f j j ≤ max j (maxT - 1)
  | 0, j => by simp [chain]
  | f + 1, j => by
    unfold chain
    by_cases hc : (j < maxT - 1 && lik (j - 1)) = true
    · rw [if_pos hc]
      have hj : j < maxT - 1 := by
        have := (Bool.and_eq_true _ _).mp hc
        exact of_decide_eq_true this.1
      have := chain_le maxT lik f (j + 1)
      omega
    · rw [if_neg hc]
      omega

/-- Starting at step 1 with one draft, `n_drafts <= max_t - 1` (for `max_t >= 2`). -/
theorem chain_from_one_le (maxT : Nat) (lik : Nat → Bool) (hm : 2 ≤ maxT) (f : Nat) :
    chain maxT lik f 1 1 ≤ maxT - 1 := by
  have := chain_le maxT lik f 1
  omega

-- § the bump allocator

/-- One bump step: advance the cursor by the request rounded up to a multiple of 256. -/
def bumpTake (used n : Nat) : Nat := used + (n + 255) / 256 * 256

/-- A region is aligned to 256 and covers its request. -/
theorem bumpTake_covers (used n : Nat) : used + n ≤ bumpTake used n := by
  unfold bumpTake; omega

/-- The step is a multiple of 256 away from the old cursor. -/
theorem bumpTake_mod (used n : Nat) (h : used % 256 = 0) : bumpTake used n % 256 = 0 := by
  unfold bumpTake; omega

/-- Cursor after a sequence of requests. -/
def bumpAll (used : Nat) : List Nat → Nat
  | [] => used
  | n :: l => bumpAll (bumpTake used n) l

/-- The total covers the sum of requests, so the counting pass sizes the arena for the carving pass. -/
theorem bumpAll_ge : ∀ (l : List Nat) (used : Nat), used + l.sum ≤ bumpAll used l
  | [], used => by simp [bumpAll]
  | n :: l, used => by
    have h1 := bumpTake_covers used n
    have h2 := bumpAll_ge l (bumpTake used n)
    simp only [bumpAll, List.sum_cons]
    omega

/-- Starting aligned, the total stays aligned to 256. -/
theorem bumpAll_mod : ∀ (l : List Nat) (used : Nat), used % 256 = 0 → bumpAll used l % 256 = 0
  | [], used, h => by simpa [bumpAll] using h
  | n :: l, used, h => by
    simp only [bumpAll]
    exact bumpAll_mod l _ (bumpTake_mod used n h)

end MtpDrafterSchedule
