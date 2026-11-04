/-!
# MODULE: graph_registry
The captured-graph registry (`graph.cpp`): record-once-per-key, find, and a bounded wait.

## Source semantics
* `record(type, n)` captures a body at most once per `(type, n)` key; a second record of the same key is a no-op
  and does not count as a capture.
* `captures_` equals the number of stored graphs.
* A capture with zero nodes is refused.
* `wait_ms` is bounded: it reports success only if completion was observed before the deadline, and it does
  observe completion if it happens within the bound.

## Form template identifiers
* `GraphRegistry::record` -> `record`; `graphs_` -> `Reg.graphs`; `captures_` -> `Reg.captures`
* `GraphRegistry::find` -> `Reg.find`; `CapturedGraph::wait_ms` -> `waitFuel`
* `nodes_ == 0` check in `end` -> `endOk`
-/

namespace GraphRegistry

-- § registry

/-- Registry state: stored graphs keyed by `(layer type, n_tokens)` and the capture counter. -/
structure Reg where
  graphs : List ((Nat × Nat) × Nat)
  captures : Nat

/-- Look up a graph by key. -/
def Reg.find (r : Reg) (k : Nat × Nat) : Option Nat :=
  (r.graphs.find? (fun p => p.1 == k)).map (·.2)

/-- Record a graph unless the key is already present (the point of a registry). -/
def record (r : Reg) (k : Nat × Nat) (id : Nat) : Reg :=
  match r.find k with
  | some _ => r
  | none => { graphs := (k, id) :: r.graphs, captures := r.captures + 1 }

/-- Recording an already-recorded key changes nothing. -/
theorem record_found (r : Reg) (k : Nat × Nat) (id v : Nat) (h : r.find k = some v) :
    record r k id = r := by
  simp [record, h]

/-- After `record`, the key is findable. -/
theorem find_record_some (r : Reg) (k : Nat × Nat) (id : Nat) :
    ∃ v, (record r k id).find k = some v := by
  cases h : r.find k with
  | some v => exact ⟨v, by rw [record_found r k id v h]; exact h⟩
  | none =>
    have hr : record r k id = { graphs := (k, id) :: r.graphs, captures := r.captures + 1 } := by
      simp [record, h]
    rw [hr]
    exact ⟨id, by simp [Reg.find]⟩

/-- Recording the same key twice is the same as recording it once. -/
theorem record_idempotent (r : Reg) (k : Nat × Nat) (i j : Nat) :
    record (record r k i) k j = record r k i := by
  obtain ⟨v, hv⟩ := find_record_some r k i
  exact record_found _ k j v hv

/-- The capture counter equals the number of stored graphs, and `record` preserves that. -/
theorem record_inv (r : Reg) (k : Nat × Nat) (id : Nat)
    (hinv : r.captures = r.graphs.length) :
    (record r k id).captures = (record r k id).graphs.length := by
  cases h : r.find k with
  | some v => simpa [record, h] using hinv
  | none => simp [record, h, hinv]

/-- A capture increments the counter by at most one. -/
theorem record_captures_le (r : Reg) (k : Nat × Nat) (id : Nat) :
    (record r k id).captures ≤ r.captures + 1 := by
  cases h : r.find k with
  | some v => simp [record, h]
  | none => simp [record, h]

/-- A found key is not re-captured: the counter is unchanged. -/
theorem record_found_captures (r : Reg) (k : Nat × Nat) (id v : Nat) (h : r.find k = some v) :
    (record r k id).captures = r.captures := by
  rw [record_found r k id v h]

-- § capture validity

/-- A capture that recorded nothing is a wiring mistake and is refused. -/
def endOk (nodes : Nat) : Bool := decide (nodes ≠ 0)

/-- Zero nodes is refused. -/
theorem endOk_zero : endOk 0 = false := by
  simp [endOk]

/-- Any positive node count is accepted. -/
theorem endOk_pos (n : Nat) (h : 0 < n) : endOk n = true := by
  simp [endOk]
  omega

-- § bounded wait

/-- A bounded poll: `n` attempts at times `t, t+1, ..`; true iff some poll sees completion. -/
def waitFuel : Nat → (Nat → Bool) → Nat → Bool
  | 0, _, _ => false
  | n + 1, p, t => if p t then true else waitFuel n p (t + 1)

/-- Soundness: a reported success means completion was observed within the bound. -/
theorem waitFuel_sound : ∀ (n : Nat) (p : Nat → Bool) (t : Nat), waitFuel n p t = true →
    ∃ i, t ≤ i ∧ i < t + n ∧ p i = true
  | 0, p, t, h => by simp [waitFuel] at h
  | n + 1, p, t, h => by
    unfold waitFuel at h
    by_cases hp : p t = true
    · exact ⟨t, le_refl _, by omega, hp⟩
    · rw [if_neg hp] at h
      obtain ⟨i, h1, h2, h3⟩ := waitFuel_sound n p (t + 1) h
      exact ⟨i, by omega, by omega, h3⟩

/-- Completeness: completion within the bound is always observed. -/
theorem waitFuel_complete (p : Nat → Bool) (i : Nat) (hi : p i = true) :
    ∀ (n t : Nat), t ≤ i → i < t + n → waitFuel n p t = true
  | 0, t, h1, h2 => by omega
  | n + 1, t, h1, h2 => by
    unfold waitFuel
    by_cases hp : p t = true
    · rw [if_pos hp]
    · rw [if_neg hp]
      have hne : t ≠ i := fun h => hp (by subst h; exact hi)
      exact waitFuel_complete p i hi n (t + 1) (by omega) (by omega)

/-- A zero bound never reports success: a timeout that does not time out would report a hang as a pass. -/
theorem waitFuel_zero (p : Nat → Bool) (t : Nat) : waitFuel 0 p t = false := rfl

end GraphRegistry
