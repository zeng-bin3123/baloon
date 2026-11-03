/-!
# MODULE: session_state_layout
Session arena layout and per-layer state addressing (`session.cpp`): the GDN state index, per-GDN-layer state slices,
the QSA state arena (one RoPE-owning state, the rest borrow), alignment, ring sequence numbers and the five-stage
differencing.

## Source semantics
* A GDN layer's index is the number of non-QSA layers before it; each owns `F = state + conv` floats.
* QSA state `i` sits at offset `0` for `i = 0` and `first + (i-1)*rest` otherwise; all fit in `first + (n-1)*rest`.
* `session_loop` waits for ring `l+1` at layer `l`; the token graph waits for ring `l+1` too.
* The five stage times are consecutive differences of prefix times, so they telescope to the full prefix.

## Form template identifiers
`gdn_point_at` -> gdnIndex, `gdn_state_floats` -> stateFloats, `qsa_state_arena` -> qsaOff, `align_up` -> alignUp256.
-/

namespace Strata.Session

-- § alignment

/-- 256-byte alignment of the session arena regions. -/
def alignUp256 (n : Nat) : Nat := (n + 255) / 256 * 256

/-- Aligned size covers the request. -/
theorem alignUp256_ge (n : Nat) : n ≤ alignUp256 n := by unfold alignUp256; omega

/-- Aligned size is a multiple of 256. -/
theorem alignUp256_mod (n : Nat) : alignUp256 n % 256 = 0 := by unfold alignUp256; omega

-- § GDN state indexing

/-- Index of a GDN layer: the count of non-QSA layers before it. -/
def gdnIndex (l : Nat) : Nat := ((List.range l).filter (fun i => !(i % 4 == 3))).length

/-- Closed form of the index on the 48-layer geometry. -/
theorem gdnIndex_closed : ∀ l, l < 49 → gdnIndex l = l - l / 4 := by decide

/-- A GDN layer's index is below the 36 states. -/
theorem gdnIndex_lt_36 (l : Nat) (h : l < 48) (hq : l % 4 ≠ 3) : gdnIndex l < 36 := by
  have := gdnIndex_closed l (by omega)
  omega

/-- Consecutive GDN layers get consecutive indices (no state is shared). -/
theorem gdnIndex_succ (l : Nat) (h : l < 47) (hq : l % 4 ≠ 3) : gdnIndex (l + 1) = gdnIndex l + 1 := by
  have h1 := gdnIndex_closed l (by omega)
  have h2 := gdnIndex_closed (l + 1) (by omega)
  omega

/-- The conv state starts where the recurrent state ends, and ends at the slice's end. -/
theorem gdn_slice_tiles (idx S1 S2 : Nat) : idx * (S1 + S2) + S1 + S2 = (idx + 1) * (S1 + S2) := by
  rw [Nat.add_mul, Nat.one_mul]; omega

/-- A valid GDN index's slice lies inside the `n_gdn * F` allocation. -/
theorem gdn_slice_in_arena (idx n F : Nat) (h : idx < n) : idx * F + F ≤ n * F := by
  have h1 := Nat.mul_le_mul_right F (show idx + 1 ≤ n by omega)
  rw [Nat.add_mul, Nat.one_mul] at h1
  exact h1

-- § QSA state arena

/-- Offset of QSA state `i` in the arena. -/
def qsaOff (first rest i : Nat) : Nat := if i = 0 then 0 else first + (i - 1) * rest

/-- State `i`'s extent: `first` for the RoPE owner, `rest` for the borrowers. -/
def qsaSize (first rest i : Nat) : Nat := if i = 0 then first else rest

/-- Every QSA state fits in the `first + (n-1)*rest` arena. -/
theorem qsa_state_fits (first rest n i : Nat) (h : i < n) :
    qsaOff first rest i + qsaSize first rest i ≤ first + (n - 1) * rest := by
  unfold qsaOff qsaSize
  by_cases h0 : i = 0
  · simp [h0]
  · simp only [h0, if_false]
    have h1 := Nat.mul_le_mul_right rest (show i ≤ n - 1 by omega)
    have h2 : (i - 1 + 1) * rest = (i - 1) * rest + rest := by rw [Nat.add_mul, Nat.one_mul]
    have h3 : i - 1 + 1 = i := by omega
    rw [h3] at h2
    omega

-- § ring sequence

/-- At layer `l` the loop waits for ring number `l + 1`; the counter is monotone. -/
theorem ring_monotone (l : Nat) : l + 1 < (l + 1) + 1 := by omega

/-- The per-layer wait value is positive, so a zeroed doorbell is never mistaken for a ring. -/
theorem ring_positive (l : Nat) : 0 < l + 1 := by omega

-- § five-stage differencing

/-- The five stage times are the differences of the prefix times (`t0 = 0`). -/
def stageTimes (t1 t2 t3 t4 t5 : Int) : List Int := [t1, t2 - t1, t3 - t2, t4 - t3, t5 - t4]

/-- The five stage times sum to the full prefix time. -/
theorem stageTimes_sum (t1 t2 t3 t4 t5 : Int) : (stageTimes t1 t2 t3 t4 t5).sum = t5 := by
  simp [stageTimes]

end Strata.Session
