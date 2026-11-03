import Mathlib

/-!
# MODULE: graph_registry
The captured-graph registry (`CapturedGraph`, `GraphRegistry` in `graph.cpp`): graphs are recorded once per
`(layer type, n_tokens)` key, a second `record` for the same key is a no-op, `find` returns what was recorded, and
a capture that recorded zero nodes is refused.

## Source semantics
* `record(type, n, ...)` returns early when the key exists ("already recorded: the point of a registry"); only a
  new key adds a graph and bumps `captures_`.
* `find` returns the graph recorded for exactly that key, or none; recording one key never disturbs another.
* `CapturedGraph::end` refuses a capture with zero nodes.
* `wait_ms` is bounded by a real-clock deadline (not modelled; no arithmetic content).

## Form template identifiers
* `find`   <- `GraphRegistry::find`
* `record` <- `GraphRegistry::record` (the `graphs_.count(k)` early return)
* `endOk`  <- the `nodes_ == 0` check in `CapturedGraph::end`
* key `(Nat × Nat)` <- `Key{(int) type, n_tokens}`; value `Nat` <- the `CapturedGraph`
-/

namespace Strata.Graph

/-- Look a key up in the registry (association list; the first match wins, as in a map). -/
def find : List ((Nat × Nat) × Nat) → Nat × Nat → Option Nat
  | [], _ => none
  | (k', v) :: m, k => if k' = k then some v else find m k

/-- Record a graph for a key unless one is already there. -/
def record (m : List ((Nat × Nat) × Nat)) (k : Nat × Nat) (v : Nat) : List ((Nat × Nat) × Nat) :=
  if (find m k).isSome = true then m else (k, v) :: m

/-- A capture is accepted only if it recorded at least one node. -/
def endOk (nodes : Nat) : Bool := nodes != 0

/-- After `record`, the key is findable. -/
theorem find_record_isSome (m : List ((Nat × Nat) × Nat)) (k : Nat × Nat) (v : Nat) :
    (find (record m k v) k).isSome = true := by
  by_cases h : (find m k).isSome = true
  · simp [record, h]
  · simp [record, h, find]

/-- Recording an existing key changes nothing. -/
theorem record_existing (m : List ((Nat × Nat) × Nat)) (k : Nat × Nat) (v : Nat)
    (h : (find m k).isSome = true) : record m k v = m := by
  simp [record, h]

/-- `record` is idempotent: a second record for the same key is a no-op. -/
theorem record_idem (m : List ((Nat × Nat) × Nat)) (k : Nat × Nat) (v v' : Nat) :
    record (record m k v) k v' = record m k v := by
  by_cases h : (find m k).isSome = true
  · simp [record, h]
  · have hr : record m k v = (k, v) :: m := by simp [record, h]
    rw [hr]
    simp [record, find]

/-- Recording a key never disturbs another key's graph. -/
theorem find_record_other (m : List ((Nat × Nat) × Nat)) (k k' : Nat × Nat) (v : Nat) (hne : k' ≠ k) :
    find (record m k v) k' = find m k' := by
  have hne' : k ≠ k' := fun e => hne e.symm
  by_cases h : (find m k).isSome = true
  · simp [record, h]
  · simp [record, h, find, hne']

/-- A new key adds exactly one graph (`captures_` goes up by one). -/
theorem record_length_new (m : List ((Nat × Nat) × Nat)) (k : Nat × Nat) (v : Nat)
    (h : (find m k).isSome = false) : (record m k v).length = m.length + 1 := by
  simp [record, h]

/-- An existing key adds nothing. -/
theorem record_length_old (m : List ((Nat × Nat) × Nat)) (k : Nat × Nat) (v : Nat)
    (h : (find m k).isSome = true) : (record m k v).length = m.length := by
  simp [record, h]

/-- The registry never loses graphs. -/
theorem record_length_ge (m : List ((Nat × Nat) × Nat)) (k : Nat × Nat) (v : Nat) :
    m.length ≤ (record m k v).length := by
  unfold record
  split_ifs
  · exact Nat.le_refl _
  · simp

/-- A capture with zero nodes is refused. -/
theorem endOk_zero : endOk 0 = false := by
  decide

/-- A capture with at least one node is accepted. -/
theorem endOk_pos (n : Nat) (h : 0 < n) : endOk (n) = true := by
  unfold endOk
  simp
  omega

end Strata.Graph
