/-!
# MODULE: graph_capture_registry
Capture-once, replay-many registry keyed by `(layer_type, n_tokens)`.

## Source semantics
* Recording a key twice is a no-op and not an error; the capture body runs exactly once per key.
* The number of captures equals the number of registered graphs (`captures() == size()`).
* A capture that produced zero nodes is an error, not an empty graph.
* The model has 48 layers: 36 GDN and 12 QSA, QSA at indices 3, 7, ..., 47.

## Form template identifiers
* `GraphRegistry::record` -> `record`
* `GraphRegistry::size` / `captures` -> `Reg.keys.length` / `Reg.captures`
* `CapturedGraph::end` -> `endOk`
* `LayerType` -> `isQsa`
-/

namespace Strata.GraphCaptureRegistry

-- § registry model

/-- Registry state: the recorded keys and the number of bodies actually captured. -/
structure Reg where
  keys : List (Nat × Nat)
  captures : Nat

/-- Record a key: capture only if it is not already recorded. -/
def record (s : Reg) (k : Nat × Nat) : Reg :=
  if k ∈ s.keys then s else { keys := k :: s.keys, captures := s.captures + 1 }

/-- The registry invariant: captures equals size and keys are distinct. -/
def Inv (s : Reg) : Prop := s.captures = s.keys.length ∧ s.keys.Nodup

/-- The empty registry satisfies the invariant. -/
theorem inv_empty : Inv { keys := [], captures := 0 } := by
  simp [Inv]

/-- Recording the same key twice is a no-op: the body runs exactly once. -/
theorem record_idempotent (s : Reg) (k : Nat × Nat) : record (record s k) k = record s k := by
  by_cases h : k ∈ s.keys
  · simp [record, h]
  · simp [record, h]

/-- Recording preserves the invariant: `captures() == size()` always holds. -/
theorem record_inv (s : Reg) (k : Nat × Nat) (h : Inv s) : Inv (record s k) := by
  obtain ⟨h1, h2⟩ := h
  by_cases hk : k ∈ s.keys
  · simp [record, hk, Inv, h1, h2]
  · simp [record, hk, Inv, h1, h2]

/-- Recording an already-recorded key leaves the capture count unchanged. -/
theorem record_known_no_capture (s : Reg) (k : Nat × Nat) (h : k ∈ s.keys) :
    (record s k).captures = s.captures := by
  simp [record, h]

/-- Recording a new key captures exactly once. -/
theorem record_new_captures (s : Reg) (k : Nat × Nat) (h : k ∉ s.keys) :
    (record s k).captures = s.captures + 1 := by
  simp [record, h]

/-- A recorded key stays findable after any later record. -/
theorem record_keeps_keys (s : Reg) (k k' : Nat × Nat) (h : k' ∈ s.keys) : k' ∈ (record s k).keys := by
  by_cases hk : k ∈ s.keys
  · simp [record, hk, h]
  · simp [record, hk, h]

/-- After recording, the key is present. -/
theorem record_mem (s : Reg) (k : Nat × Nat) : k ∈ (record s k).keys := by
  by_cases hk : k ∈ s.keys
  · simp [record, hk]
  · simp [record, hk]

-- § capture end

/-- A capture is accepted only if it produced at least one node. -/
def endOk (nodes : Nat) : Bool := decide (0 < nodes)

/-- A capture with zero nodes is reported as an error. -/
theorem empty_capture_rejected : endOk 0 = false := by
  decide

/-- Any non-empty capture is accepted. -/
theorem nonempty_capture_accepted (n : Nat) (h : 0 < n) : endOk n = true := by
  unfold endOk
  exact decide_eq_true h

-- § layer types

/-- A layer is QSA exactly at indices 3, 7, ..., 47. -/
def isQsa (l : Nat) : Bool := decide (l % 4 = 3)

/-- 12 of the 48 layers are QSA. -/
theorem qsa_count : (List.range 48).countP isQsa = 12 := by
  first | decide | rfl

/-- 36 of the 48 layers are GDN. -/
theorem gdn_count : (List.range 48).countP (fun l => !isQsa l) = 36 := by
  first | decide | rfl

/-- The last layer (47) is QSA and the first three are not. -/
theorem qsa_endpoints : isQsa 47 = true ∧ isQsa 0 = false ∧ isQsa 1 = false ∧ isQsa 2 = false := by
  decide

/-- Per-token launch overhead: two graph launches per layer, 96 per token. -/
theorem launches_per_token : 2 * 48 = 96 := by
  decide

end Strata.GraphCaptureRegistry
