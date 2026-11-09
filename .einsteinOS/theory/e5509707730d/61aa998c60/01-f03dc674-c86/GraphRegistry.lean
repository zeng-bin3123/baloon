/-!
# MODULE: graph_registry
SOURCES: src/core/graph.cpp
Captured CUDA graph registry: a graph is recorded once per `(layer type, n_tokens)` key; a capture that records
zero nodes is refused; a bounded wait never succeeds after its deadline.

## Source semantics
* `record` of an existing key is a no-op returning success.
* `launch` requires a recorded key.
* `wait_ms` loops on a real clock, returning false on timeout or on a real error.

## Form template identifiers
Key → Nat key ; graphs_ → List Nat ; record → recordKey ; end() node check → endOk
-/

namespace Strata.GraphRegistry

-- § registry

/-- Record a key into the registry (already-present keys are unchanged). -/
def recordKey (ks : List Nat) (k : Nat) : List Nat := if k ∈ ks then ks else k :: ks

/-- After recording, the key is found (so `launch` can succeed). -/
theorem mem_recordKey (ks : List Nat) (k : Nat) : k ∈ recordKey ks k := by
  unfold recordKey
  by_cases h : k ∈ ks <;> simp [h]

/-- Recording twice equals recording once: the point of a registry. -/
theorem recordKey_idem (ks : List Nat) (k : Nat) : recordKey (recordKey ks k) k = recordKey ks k := by
  unfold recordKey
  by_cases h : k ∈ ks <;> simp [h]

/-- Recording never loses an existing key. -/
theorem recordKey_mono (ks : List Nat) (k j : Nat) (h : j ∈ ks) : j ∈ recordKey ks k := by
  unfold recordKey
  by_cases hk : k ∈ ks <;> simp [hk, h]

/-- The number of captures grows by at most one per `record`. -/
theorem recordKey_length (ks : List Nat) (k : Nat) : (recordKey ks k).length ≤ ks.length + 1 := by
  unfold recordKey
  by_cases h : k ∈ ks <;> simp [h]

-- § capture validity

/-- `end()` accepts a capture only when it recorded at least one node. -/
def endOk (nodes : Nat) : Bool := decide (0 < nodes)

/-- An empty capture is refused. -/
theorem endOk_zero : endOk 0 = false := by
  simp [endOk]

/-- A non-empty capture is accepted. -/
theorem endOk_pos (n : Nat) (h : 0 < n) : endOk n = true := by
  simp [endOk, h]

-- § bounded wait

/-- Outcome of one poll: success, real error, or still running past/before the deadline. -/
def pollResult (done err : Bool) (now deadline : Nat) : Option Bool :=
  if done then some true
  else if err then some false
  else if deadline ≤ now then some false
  else none

/-- Completion wins over everything else. -/
theorem pollResult_done (err : Bool) (now deadline : Nat) : pollResult true err now deadline = some true := by
  simp [pollResult]

/-- Past the deadline an unfinished, error-free poll reports failure rather than spinning on. -/
theorem pollResult_timeout (now deadline : Nat) (h : deadline ≤ now) :
    pollResult false false now deadline = some false := by
  simp [pollResult, h]

/-- Before the deadline an unfinished, error-free poll keeps waiting. -/
theorem pollResult_wait (now deadline : Nat) (h : now < deadline) :
    pollResult false false now deadline = none := by
  have : ¬ deadline ≤ now := by omega
  simp [pollResult, this]

end Strata.GraphRegistry
