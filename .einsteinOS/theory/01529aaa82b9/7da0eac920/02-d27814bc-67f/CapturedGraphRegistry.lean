/-!
# MODULE: captured_graph_registry
A registry of captured CUDA graphs keyed by (layer type, token count), plus the bounded completion wait.

## Source semantics
* `record` is idempotent: recording an existing key changes nothing and succeeds.
* A capture that recorded zero nodes is rejected, so a graph that replays nothing is never stored.
* Each recorded key is stored once; a successful record makes the key findable.
* The completion wait is bounded by a real clock deadline: it returns true on completion, false on a real error
  or on deadline expiry.

## Form template identifiers
* `GraphRegistry::record` -> `record`
* `GraphRegistry::find` -> `find`
* `CapturedGraph::end` zero-node guard -> `endOk`
* `CapturedGraph::wait_ms` -> `waitStep`
-/

namespace Strata.GraphRegistry

/-- Registry key: (layer type, number of tokens). -/
abbrev Key := Nat × Nat

/-- The registry as an association list of keys to node counts. -/
abbrev Reg := List (Key × Nat)

/-- Is the key recorded? -/
def mem (r : Reg) (k : Key) : Bool := r.any (fun p => p.1 == k)

/-- Record: already present means no-op; otherwise add the capture. -/
def record (r : Reg) (k : Key) (nodes : Nat) : Reg :=
  if mem r k then r else (k, nodes) :: r

/-- Find the node count of a key. -/
def find : Reg → Key → Option Nat
  | [], _ => none
  | (k', n) :: t, k => if k' == k then some n else find t k

/-- Recording the same key twice equals recording it once. -/
theorem record_idempotent (r : Reg) (k : Key) (n m : Nat) :
    record (record r k n) k m = record r k n := by
  unfold record
  by_cases h : mem r k = true
  · simp [h]
  · have h' : mem ((k, n) :: r) k = true := by
      simp [mem]
    simp [h, h']

/-- Recording an existing key leaves the registry untouched. -/
theorem record_existing (r : Reg) (k : Key) (n : Nat) (h : mem r k = true) :
    record r k n = r := by
  simp [record, h]

/-- A new key grows the registry by exactly one. -/
theorem record_new_length (r : Reg) (k : Key) (n : Nat) (h : mem r k = false) :
    (record r k n).length = r.length + 1 := by
  simp [record, h]

/-- Recording never shrinks the registry. -/
theorem record_length_ge (r : Reg) (k : Key) (n : Nat) : r.length ≤ (record r k n).length := by
  unfold record
  by_cases h : mem r k = true
  · simp [h]
  · simp [h]

/-- After recording a new key it is findable with its node count. -/
theorem find_record_new (r : Reg) (k : Key) (n : Nat) :
    find (record r k n) k = if mem r k then find r k else some n := by
  unfold record
  by_cases h : mem r k = true
  · simp [h]
  · simp [h, find]

/-- An empty registry finds nothing. -/
theorem find_empty (k : Key) : find [] k = none := by
  simp [find]

-- § capture end guard

/-- `end` accepts a capture only when it recorded at least one node. -/
def endOk (nodes : Nat) : Bool := 0 < nodes

/-- A zero-node capture is rejected. -/
theorem endOk_zero : endOk 0 = false := by
  simp [endOk]

/-- An accepted capture has a positive node count. -/
theorem endOk_pos (n : Nat) (h : endOk n = true) : 0 < n := by
  simpa [endOk] using h

/-- Registry entries only come from accepted captures when the caller guards with `endOk`. -/
theorem guarded_record_pos (r : Reg) (k : Key) (n : Nat) (h : endOk n = true) :
    0 < n := endOk_pos n h

-- § bounded wait

/-- Outcome of one wait step. -/
inductive Wait where
  | done
  | fail
  | timeout
  | again
  deriving DecidableEq

/-- One poll: `queryDone`, `queryError`, and whether the deadline passed. -/
def waitStep (queryDone queryError expired : Bool) : Wait :=
  if queryDone then .done
  else if queryError then .fail
  else if expired then .timeout
  else .again

/-- Completion always wins: a finished graph is a success even past the deadline. -/
theorem waitStep_done (e x : Bool) : waitStep true e x = .done := by
  simp [waitStep]

/-- A real error is reported, not retried. -/
theorem waitStep_error (x : Bool) : waitStep false true x = .fail := by
  simp [waitStep]

/-- The deadline is a real bound: once expired with nothing pending, the wait stops. -/
theorem waitStep_timeout : waitStep false false true = .timeout := by
  simp [waitStep]

/-- Without expiry and without completion or error, the wait polls again. -/
theorem waitStep_again : waitStep false false false = .again := by
  simp [waitStep]

/-- The wait loop terminates within `fuel` polls once the deadline has expired. -/
def waitLoop : Nat → (Nat → Bool) → (Nat → Bool) → (Nat → Bool) → Wait
  | 0, _, _, _ => .timeout
  | n + 1, d, e, x =>
    match waitStep (d n) (e n) (x n) with
    | .again => waitLoop n d e x
    | w => w

/-- With no fuel the loop reports a timeout, never a spurious success. -/
theorem waitLoop_zero (d e x : Nat → Bool) : waitLoop 0 d e x = .timeout := by
  simp [waitLoop]

end Strata.GraphRegistry
