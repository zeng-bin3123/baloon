/-!
# MODULE: spec_controller
SOURCES: src/spec/controller.cpp
The speculative-decoding controller: clamped window index into the cost tables, expected committed tokens of a
draft window, the EMA acceptance learner, and the bookkeeping of which drafted positions an observation touches.

## Source semantics
* `step_ms(k)` reads table entry `clamp(k + 1, 1, K_MAX + 1) - 1`, always in range.
* The expected tokens of a window is `1 + sum of cumulative acceptance products`; with probabilities in `[0, 1]`
  it lies in `[1, k + 1]`.
* `observe` clamps `accepted` to `0..k`, updates positions `0 .. min(k, accepted + 1) - 1`, and after a fully
  accepted window pulls the untested positions toward the deepest tested one.
* The EMA `p += e * (hit - p)` keeps `p` in `[0, 1]` (fixed-point model: scale `S`).

## Form template identifiers
`std::clamp` -> clampN; `expected` -> runs (fixed point, scale S); EMA -> emaUpdate; `seen` -> seenCount.
-/

namespace Strata.SpecController

/-- `std::clamp(v, lo, hi)` for `lo ≤ hi`. -/
def clampN (v lo hi : Nat) : Nat := max lo (min v hi)

/-- The clamp result is inside `[lo, hi]`. -/
theorem clampN_bounds (v lo hi : Nat) (h : lo ≤ hi) : lo ≤ clampN v lo hi ∧ clampN v lo hi ≤ hi := by
  unfold clampN
  constructor <;> omega

/-- The table index `n - 1` with `n = clamp(k+1, 1, K_MAX+1)` is below `K_MAX + 1`. -/
theorem table_index_lt (k kmax : Nat) : clampN (k + 1) 1 (kmax + 1) - 1 < kmax + 1 := by
  unfold clampN
  omega

/-- The clamped `n` is at least 1 (so `n - 1` never underflows). -/
theorem table_n_pos (k kmax : Nat) : 1 ≤ clampN (k + 1) 1 (kmax + 1) := by
  unfold clampN
  omega

-- § expected committed tokens (fixed point, scale S)

/-- Sum over the window of cumulative acceptance (`run` is the running product, scale S). -/
def runs (S : Nat) : Nat → List Nat → Nat
  | _, [] => 0
  | run, p :: ps => (run * p / S) + runs S (run * p / S) ps

/-- Multiplying by a probability `p ≤ S` never increases a quantity `≤ S`. -/
theorem step_le (S run p : Nat) (hS : 0 < S) (hp : p ≤ S) (hr : run ≤ S) : run * p / S ≤ S := by
  apply Nat.div_le_of_le_mul
  have h1 : run * p ≤ run * S := Nat.mul_le_mul_left run hp
  have h2 : run * S ≤ S * S := Nat.mul_le_mul_right S hr
  omega

/-- The accumulated acceptance of `n` drafted positions is at most `n` whole tokens (`S * n`). -/
theorem runs_le (S : Nat) (hS : 0 < S) (ps : List Nat) :
    ∀ run, run ≤ S → (∀ p ∈ ps, p ≤ S) → runs S run ps ≤ S * ps.length := by
  induction ps with
  | nil => intro run _ _; simp [runs]
  | cons p ps ih =>
    intro run hr hp
    have hp0 : p ≤ S := hp p (List.mem_cons_self ..)
    have hstep := step_le S run p hS hp0 hr
    have hrest := ih (run * p / S) hstep (fun q hq => hp q (List.mem_cons_of_mem _ hq))
    have hlen : S * (ps.length + 1) = S * ps.length + S := by rw [Nat.mul_add, Nat.mul_one]
    simp only [runs, List.length_cons]
    omega

-- § EMA learner (fixed point)

/-- `p' = (p (S - e) + hit e) / S`, i.e. `p += e (hit - p)` with `e = e'/S`. -/
def emaUpdate (S e p hit : Nat) : Nat := (p * (S - e) + hit * e) / S

/-- The EMA of values in `[0, S]` stays in `[0, S]`. -/
theorem emaUpdate_le (S e p hit : Nat) (he : e ≤ S) (hp : p ≤ S) (hh : hit ≤ S) : emaUpdate S e p hit ≤ S := by
  unfold emaUpdate
  apply Nat.div_le_of_le_mul
  have h1 : p * (S - e) ≤ S * (S - e) := Nat.mul_le_mul_right _ hp
  have h2 : hit * e ≤ S * e := Nat.mul_le_mul_right _ hh
  have h3 : S * (S - e) + S * e = S * S := by rw [← Nat.mul_add, Nat.sub_add_cancel he]
  omega

-- § observe bookkeeping

/-- Positions updated by an observation: `min k (accepted + 1)` after clamping `accepted` into `0..k`. -/
def seenCount (k accepted : Nat) : Nat := min k (min accepted k + 1)

/-- Never more positions than were drafted. -/
theorem seenCount_le (k accepted : Nat) : seenCount k accepted ≤ k := by unfold seenCount; omega

/-- A rejected draft (accepted < k) is itself observed: `accepted + 1` positions. -/
theorem seenCount_rejected (k accepted : Nat) (h : accepted < k) : seenCount k accepted = accepted + 1 := by
  unfold seenCount; omega

/-- A fully accepted window observes exactly `k` positions. -/
theorem seenCount_full (k : Nat) : seenCount k k = k := by unfold seenCount; omega

/-- The deepest tested position `k - 1` exists when `k > 0`, and the pulled positions `k .. K_MAX - 1` are valid. -/
theorem pull_positions_valid (k kmax i : Nat) (hk : 0 < k) (hkm : k ≤ kmax) (hi : k ≤ i) (hi2 : i < kmax) :
    k - 1 < kmax ∧ i < kmax := by
  constructor <;> omega

/-- Lookup candidates are bounded by `K_MAX`. -/
theorem lookup_len_le (avail kmax : Nat) : min avail kmax ≤ kmax := by omega

end Strata.SpecController
