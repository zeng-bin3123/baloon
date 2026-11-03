import Mathlib

/-!
# MODULE: session_loop_contract
The structural contracts of the per-token host loop (`session.hpp`): layers run in order, the host pool sits
between `pre[l]` and `post[l]` without changing device arithmetic, graph capture is idempotent, the C1 dump
buffer has one disjoint slot per layer, and prefix-differenced stage timings telescope.

## Source semantics
* `session_loop` runs layers `0..n-1` in order; `post[l] ∘ pool ∘ pre[l]` equals `post[l] ∘ pre[l]` when the pool
  leaves device state alone (it writes the host staging only), which is why the loop is a change in WHEN the host
  runs, not in what is computed.
* `session_capture` is idempotent: a second call is a no-op.
* `dump_layers` holds `(n_layers + 1)` slots of `hc * n_embd` floats; slot `l + 1` is `R` after layer `l`.
* `session_replay_stage_prefixes` differences prefix timings `P_1 .. P_5`; the stage times sum to `P_5 - P_0`.
* A verify window of `T` tokens indexes `exec_[T]`, `T <= 8`.

## Form template identifiers
* `runLayers`       <- the layer loop of `session_loop` / `session_token`
* `pre`, `post`, `pool` <- `block_layer_pre`, `block_layer_post`, `PoolFn`
* `capture`         <- `session_capture` (`SessionGraphs::captured`)
* `slotBase`        <- slot `l` of `dump_layers`
* `P`               <- the prefix graph times `preP[k]`
-/

namespace Strata.Session

-- § the layer loop

/-- Run layers `0 .. n-1` in order, threading the state. -/
def runLayers {σ : Type} (step : Nat → σ → σ) (n : Nat) (s : σ) : σ :=
  (List.range n).foldl (fun acc l => step l acc) s

/-- Zero layers leave the state alone. -/
theorem runLayers_zero {σ : Type} (step : Nat → σ → σ) (s : σ) : runLayers step 0 s = s := by
  simp [runLayers]

/-- The loop is a left-to-right composition: layer `n` runs after layers `0 .. n-1`. -/
theorem runLayers_succ {σ : Type} (step : Nat → σ → σ) (n : Nat) (s : σ) :
    runLayers step (n + 1) s = step n (runLayers step n s) := by
  simp [runLayers, List.range_succ]

/-- If the pool leaves device state untouched, the loop with the pool in the middle of each layer computes the
same final state as the loop without it. -/
theorem loop_eq_token {σ : Type} (pre post : Nat → σ → σ) (pool : Nat → σ → σ)
    (hp : ∀ l, pool l = id) (n : Nat) (s : σ) :
    runLayers (fun l => post l ∘ pool l ∘ pre l) n s = runLayers (fun l => post l ∘ pre l) n s := by
  have hfun : (fun l => post l ∘ pool l ∘ pre l) = (fun l => post l ∘ pre l) := by
    funext l
    simp [hp l]
  rw [hfun]

-- § capture

/-- Capture state: whether it has happened, and the graphs held. A second capture changes nothing. -/
def capture {α : Type} (s : Bool × α) (g : α) : Bool × α := if s.1 then s else (true, g)

/-- `session_capture` is idempotent. -/
theorem capture_idem {α : Type} (s : Bool × α) (g g' : α) :
    capture (capture s g) g' = capture s g := by
  rcases s with ⟨c, a⟩
  cases c <;> simp [capture]

/-- After a capture the flag is set. -/
theorem capture_sets {α : Type} (s : Bool × α) (g : α) : (capture s g).1 = true := by
  rcases s with ⟨c, a⟩
  cases c <;> simp [capture]

-- § the dump buffer

/-- Slot `l + 1` of the dump (an element `j` of it) lies inside `(n_layers + 1)` slots of width `a = hc * n_embd`. -/
theorem dump_slot_lt (a nl l j : Nat) (hl : l < nl) (hj : j < a) : (l + 1) * a + j < (nl + 1) * a := by
  have h1 : (l + 2) * a ≤ (nl + 1) * a := Nat.mul_le_mul_right _ (by omega)
  have h2 : (l + 2) * a = (l + 1) * a + a := by ring
  omega

/-- Distinct layers' dump slots do not overlap. -/
theorem dump_slots_disjoint (a l1 l2 : Nat) (h : l1 < l2) : (l1 + 1) * a + a ≤ (l2 + 1) * a := by
  have h1 : (l1 + 2) * a ≤ (l2 + 1) * a := Nat.mul_le_mul_right _ (by omega)
  have h2 : (l1 + 2) * a = (l1 + 1) * a + a := by ring
  omega

-- § prefix-differenced timings

/-- The stage times obtained by differencing prefix times sum to the whole prefix: the table accounts for all of it. -/
theorem stage_times_sum (P : Nat → ℤ) (n : Nat) :
    ∑ k ∈ Finset.range n, (P (k + 1) - P k) = P n - P 0 := by
  exact Finset.sum_range_sub P n

/-- The five-stage table: its stage times sum to the full-graph time minus the empty prefix. -/
theorem five_stage_total (P : Nat → ℤ) :
    (P 1 - P 0) + (P 2 - P 1) + (P 3 - P 2) + (P 4 - P 3) + (P 5 - P 4) = P 5 - P 0 := by
  ring

-- § the verify window's graph table

/-- A window of `T <= max_t <= 8` tokens indexes inside `exec_[9]`. -/
theorem exec_index_lt (maxT T : Nat) (hmax : maxT ≤ 8) (hT : T ≤ maxT) : T < 9 := by
  omega

end Strata.Session
