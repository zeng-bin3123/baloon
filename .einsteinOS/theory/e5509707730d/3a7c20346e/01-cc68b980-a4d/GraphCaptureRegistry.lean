/-!
# MODULE: graph_capture_registry
SOURCES: include/strata/core/graph.hpp, src/core/graph.cpp
Capture-once / replay-many registry of CUDA graphs keyed by (layer_type, n_tokens), plus the bounded
event-polling wait.

## Source semantics
* `record` on an existing key is a no-op and does not run the body; `captures()` equals `size()`.
* A capture with zero nodes is an error, never an empty graph.
* `wait_ms` polls an event query: success -> true, not-ready -> keep polling until the deadline,
  any other status -> false. It never reports completion that did not occur.

## Form template identifiers
GraphRegistry -> graph_cache; CapturedGraph -> recorded_graph; record -> capture_once;
launch -> replay_and_wait; wait_ms -> bounded_poll; LayerType -> kind_tag.
-/

namespace GraphCaptureRegistry

-- § registry

/-- Abstract registry state: the recorded keys and the capture counter. -/
structure Reg where
  keys : List (Nat × Nat)
  captures : Nat

/-- `record`: do nothing if the key exists; otherwise capture it and count the capture. -/
def Reg.record (r : Reg) (k : Nat × Nat) : Reg :=
  if k ∈ r.keys then r else { keys := k :: r.keys, captures := r.captures + 1 }

/-- Invariant: `captures()` equals `size()`. -/
def Inv (r : Reg) : Prop := r.captures = r.keys.length

/-- Recording the same key twice equals recording it once (the body runs exactly once). -/
theorem record_idem (r : Reg) (k : Nat × Nat) : (r.record k).record k = r.record k := by
  by_cases h : k ∈ r.keys <;> simp [Reg.record, h]

/-- After `record`, the key is present. -/
theorem record_mem (r : Reg) (k : Nat × Nat) : k ∈ (r.record k).keys := by
  by_cases h : k ∈ r.keys <;> simp [Reg.record, h]

/-- Recording never forgets an existing key. -/
theorem record_preserves {r : Reg} {k k' : Nat × Nat} (h : k' ∈ r.keys) : k' ∈ (r.record k).keys := by
  by_cases hk : k ∈ r.keys <;> simp [Reg.record, hk, h]

/-- `captures = size` is preserved by `record`. -/
theorem record_inv {r : Reg} (k : Nat × Nat) (hi : Inv r) : Inv (r.record k) := by
  unfold Inv at *
  by_cases h : k ∈ r.keys <;> simp [Reg.record, h, hi]

/-- The two layer types never share a key: GDN (0) and QSA (1) at the same token count are distinct. -/
theorem layer_keys_distinct (n : Nat) : ((0, n) : Nat × Nat) ≠ (1, n) := by
  simp

-- § capture validity

/-- `end` succeeds only if the capture recorded at least one node. -/
def endOk (nodes : Nat) : Bool := decide (0 < nodes)

/-- A zero-node capture is rejected. -/
theorem endOk_zero : endOk 0 = false := by decide

/-- Any non-empty capture is accepted. -/
theorem endOk_pos {n : Nat} (h : 0 < n) : endOk n = true := by
  simp [endOk, h]

-- § bounded wait

/-- Outcomes of one `cudaEventQuery`. -/
inductive Q where
  | success
  | notReady
  | err

/-- `wait_ms` over a fuel bound (the deadline) and a stream of query outcomes. -/
def wait : Nat → List Q → Bool
  | 0, _ => false
  | _ + 1, [] => false
  | _ + 1, Q.success :: _ => true
  | n + 1, Q.notReady :: t => wait n t
  | _ + 1, Q.err :: _ => false

/-- An expired deadline reports failure, never success. -/
theorem wait_zero (qs : List Q) : wait 0 qs = false := by
  cases qs <;> rfl

/-- A first-poll success returns true. -/
theorem wait_success (n : Nat) (t : List Q) : wait (n + 1) (Q.success :: t) = true := rfl

/-- A real error (not "not ready") returns false immediately. -/
theorem wait_err (n : Nat) (t : List Q) : wait (n + 1) (Q.err :: t) = false := rfl

/-- Not-ready consumes one unit of the deadline and continues. -/
theorem wait_notReady (n : Nat) (t : List Q) : wait (n + 1) (Q.notReady :: t) = wait n t := rfl

/-- `wait` returns true only if some poll actually observed completion. -/
theorem wait_true_has_success : ∀ (n : Nat) (qs : List Q), wait n qs = true → Q.success ∈ qs := by
  intro n
  induction n with
  | zero => intro qs h; cases qs <;> simp [wait] at h
  | succ n ih =>
    intro qs h
    cases qs with
    | nil => simp [wait] at h
    | cons q t =>
      cases q with
      | success => exact List.mem_cons_self
      | notReady => exact List.mem_cons_of_mem _ (ih t h)
      | err => simp [wait] at h

end GraphCaptureRegistry
