/-!
# MODULE: mtp_draft_chain
SOURCES: src/core/mtp.cpp
The multi-token-prediction drafter: a bump allocator for its device arena, the step-row staging that
places each drafted cell, and the draft chain that continues while the last draft is likely enough.

## Source semantics
* `Bump::take` rounds every region to 256 bytes; counted and real carves follow the same sequence.
* A step row for cell `c` is `[c, c+1, (c+1)/4, c+1]`; positions for every head are `c`.
* There are `2*max_t` step rows: catch-up rows `0..T-1` (`T ≤ max_t`), chain rows `max_t + j - 1`
  for `1 ≤ j ≤ max_t - 2`, and the round's row `ra = 2*max_t - 1`.  They never collide.
* The chain writes draft 0, then keeps going while `j < max_t - 1` and the previous draft's
  probability is ≥ `min_p`; the count is between 1 and `max_t - 1` when `max_t ≥ 2`.
* With a window the attention capacity is the window rounded up to 64.
-/

namespace Ref.Mtp

-- § bump allocation

/-- `Bump::take`'s rounding `(n + 255) & ~255`. -/
def bump (n : Nat) : Nat := 256 * ((n + 255) / 256)

/-- A region is never smaller than requested. -/
theorem bump_ge (n : Nat) : n ≤ bump n := by simp only [bump]; exact (by omega)

/-- Every region size is 256-aligned, so every carved pointer is too. -/
theorem bump_mod (n : Nat) : bump n % 256 = 0 := by simp only [bump]; exact (by omega)

/-- Offsets of a counted carve. -/
def carve (base : Nat) : List Nat → List Nat
  | [] => []
  | n :: rest => base :: carve (base + bump n) ns

/-- The counted pass and the real pass agree, because both only depend on the request sizes:
carving from base `b` is carving from 0 shifted by `b`. -/
theorem carve_shift (b : Nat) (ns : List Nat) : carve b ns = (carve 0 ns).map (· + b) := by
  induction ns generalizing b with
  | nil => rfl
  | cons n ns ih =>
    simp only [carve, List.map_cons, Nat.zero_add]
    rw [ih (b + bump n), ih (bump n), List.map_map]
    congr 1
    · apply List.map_congr_left
      intro x _
      simp only [Function.comp]
      exact (by omega)

/-- From an aligned base every carved region is 256-aligned. -/
theorem carve_aligned (b : Nat) (ns : List Nat) (hb : b % 256 = 0) : ∀ o ∈ carve b ns, o % 256 = 0 := by
  induction ns generalizing b with
  | nil => intro o h; simp only [carve] at h
  | cons n ns ih =>
    intro o h
    simp only [carve, List.mem_cons] at h
    rcases h with rfl | h
    · exact hb
    · have := bump_mod n
      exact ih _ (by omega) o h

-- § step rows

/-- The four step counts staged for cell `c`. -/
def stepRow (i : Nat) : List Nat := [i, i + 1, (i + 1) / 4, i + 1]

/-- Row shape: the count after appending is `c + 1`, and the block count is its quarter. -/
theorem stepRow_shape (i : Nat) :
    stepRow i = [i, i + 1, (i + 1) / 4, i + 1] ∧ 4 * ((i + 1) / 4) ≤ i + 1 ∧ i + 1 < 4 * ((i + 1) / 4) + 4 := by
  refine ⟨rfl, ?_, ?_⟩ <;> omega

/-- The round's own row. -/
def raRow (mt : Nat) : Nat := 2 * mt - 1

/-- Chain step `j`'s row. -/
def chainRow (mt j : Nat) : Nat := mt + j - 1

/-- All rows lie in the `2*max_t` staged rows and the three kinds never collide. -/
theorem rows_disjoint (maxT T t j : Nat) (hT : T ≤ maxT) (ht : t < T) (hj1 : 1 ≤ j) (hj : j ≤ maxT - 2) :
    t < chainRow maxT j ∧ chainRow maxT j < raRow maxT ∧ raRow maxT < 2 * maxT := by
  simp only [chainRow] raRow
  exact (by omega)

/-- Two chain steps never share a row. -/
theorem chainRow_injective (maxT i j : Nat) (hi : 1 ≤ i) (hj : 1 ≤ j) (h : chainRow maxT i = chainRow maxT j) :
    i = j := by
  simp only [chainRow] at h; exact (by omega)

-- § the draft chain

/-- Extra drafts from step `j` on, given whether draft `j-1` passes `min_p`, with `fuel` steps left. -/
def chainFrom (pred : Nat → Bool) : Nat → Nat → Nat
  | _, 0 => 0
  | j, f + 1 => if pred (j - 1) then 1 + chainFrom pred (j + 1) f else 0

/-- Total drafts: draft 0 always, then the chain over `j = 1 .. max_t - 2`. -/
def chainLen (pred : Nat → Bool) (maxT : Nat) : Nat := 1 + chainFrom pred 1 (maxT - 2)

/-- The chain never takes more steps than it has fuel. -/
theorem chainFrom_le (pred : Nat → Bool) (j f : Nat) : chainFrom pred j f ≤ f := by
  induction f generalizing j with
  | zero => simp only [chainFrom]
  | succ f ih =>
    simp only [chainFrom]
    split
    · have := ih (j + 1); exact (by omega)
    · exact (by omega)

/-- When every draft passes, the chain uses all its fuel. -/
theorem chainFrom_all (pred : Nat → Bool) (hp : ∀ i, pred i = true) (j f : Nat) : chainFrom pred j f = f := by
  induction f generalizing j with
  | zero => rfl
  | succ f ih => simp only [chainFrom, hp, ih]; exact (by omega)

/-- The number of drafts is between 1 and `max_t - 1`. -/
theorem chainLen_bounds (pred : Nat → Bool) (maxT : Nat) (h : 2 ≤ maxT) :
    1 ≤ chainLen pred maxT ∧ chainLen pred maxT ≤ maxT - 1 := by
  have := chainFrom_le pred 1 (maxT - 2)
  simp only [chainLen]
  exact (by omega)

/-- With every draft confident, the chain fills all `max_t - 1` slots. -/
theorem chainLen_full (pred : Nat → Bool) (hp : ∀ i, pred i = true) (maxT : Nat) (h : 2 ≤ maxT) :
    chainLen pred maxT = maxT - 1 := by
  simp only [chainLen]
  rw [chainFrom_all pass hp]
  exact (by omega)

/-- A rejected first draft stops the chain at one draft. -/
theorem chainLen_stop (pred : Nat → Bool) (h0 : pred 0 = false) (maxT : Nat) (h : 3 ≤ maxT) :
    chainLen pred maxT = 1 := by
  simp only [chainLen]
  obtain ⟨f, hf⟩ : ∃ f, maxT - 2 = f + 1 := ⟨maxT - 3, by omega⟩
  rw [hf]
  simp only [chainFrom, h0]

/-- Edge case: `max_t = 1` (allowed by `load`) still writes draft 0, one more than `max_t - 1`. -/
theorem chainLen_maxT_one (pred : Nat → Bool) : chainLen pred 1 = 1 := by
  simp only [chainLen, chainFrom]

-- § the attention capacity

/-- `cap_` rounds the window (or the context) up to 64. -/
def cap64 (n : Nat) : Nat := (n + 63) / 64 * 64

/-- The capacity covers the window and is 64-aligned. -/
theorem cap64_spec (n : Nat) : n ≤ cap64 n ∧ cap64 n % 64 = 0 ∧ cap64 n < n + 64 := by
  simp only [cap64]; exact (by omega)

end Ref.Mtp

