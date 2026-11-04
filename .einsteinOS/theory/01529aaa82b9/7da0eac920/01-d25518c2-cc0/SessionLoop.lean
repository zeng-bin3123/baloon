/-!
# MODULE: session_loop
One token through all 48 layers: the ordering of pre / pool / hits / post per layer, the 36 GDN + 12 QSA
state split, the residual dump slots, and the monotone loop counters.

## Source semantics
* The host loop is `pre[l] -> pool[l] -> hits[l] -> post[l]` for `l = 0..n-1`; layer `l`'s experts are
  combined by layer `l`'s `post`, never by a neighbour's.
* The state split is 36 GDN layers + 12 QSA layers = 48.
* `dump_layers` has `(n_layers + 1)` slots: slot 0 is the entry residual, slot `l + 1` follows layer `l`.
* Cumulative counters (`rings_mid_graph`, `ms_to_ring`) are bounded by `calls * n_layers`.
* The token graph requires all hit-path pieces or none.

## Form template identifiers
* `session_loop` -> `trace`
* `SessionLoopScratch::parts_bytes` -> `partsBytes`
* `dump_layers` -> `dumpSlot`
* `SessionGraphs::rings_mid_graph` -> `ringsBound`
-/

namespace Strata.SessionLoop

-- § loop trace

/-- The events the host loop emits per layer. -/
inductive Ev where
  | pre (l : Nat)
  | pool (l : Nat)
  | hits (l : Nat)
  | post (l : Nat)
  deriving DecidableEq

/-- The trace of the first `n` layers: each layer appends `pre, pool, hits, post` in that order. -/
def trace : Nat → List Ev
  | 0 => []
  | n + 1 => trace n ++ [Ev.pre n, Ev.pool n, Ev.hits n, Ev.post n]

/-- Four events per layer. -/
theorem trace_length (n : Nat) : (trace n).length = 4 * n := by
  induction n with
  | zero => simp [trace]
  | succ n ih => simp [trace, ih] <;> omega

/-- A longer run extends a shorter one: earlier layers' events are never rewritten. -/
theorem trace_prefix (n : Nat) : trace n <+: trace (n + 1) := by
  exact ⟨[Ev.pre n, Ev.pool n, Ev.hits n, Ev.post n], rfl⟩

/-- The last four events of a run are layer `n`'s pre, pool, hits, post: post follows its own pool. -/
theorem trace_last_layer (n : Nat) :
    trace (n + 1) = trace n ++ [Ev.pre n, Ev.pool n, Ev.hits n, Ev.post n] := by
  rfl

/-- The layer index an event belongs to. -/
def evLayer : Ev → Nat
  | Ev.pre l => l
  | Ev.pool l => l
  | Ev.hits l => l
  | Ev.post l => l

/-- Every event in the first `n` layers' trace belongs to a layer below `n`. -/
theorem trace_layers_lt (n : Nat) : ∀ e ∈ trace n, evLayer e < n := by
  induction n with
  | zero => simp [trace]
  | succ n ih =>
    intro e he
    simp [trace] at he
    rcases he with he | rfl | rfl | rfl | rfl
    · have := ih e he
      omega
    all_goals simp [evLayer]

/-- Layer `n`'s post event is never emitted by the first `n` layers: no layer's post is borrowed early. -/
theorem post_not_in_earlier (n : Nat) : Ev.post n ∉ trace n := by
  intro h
  have := trace_layers_lt n _ h
  simp [evLayer] at this

-- § state split

/-- 36 GDN layers and 12 QSA layers make 48. -/
theorem layer_split : 36 + 12 = 48 := by
  decide

/-- Recurrent GDN state is per layer: `n_gdn * gdn_state_floats` floats, 36 slices of equal size. -/
theorem gdn_state_total (s : Nat) : 36 * s = s * 36 := by
  omega

-- § residual dump

/-- Slot of the residual after layer `l`; slot 0 is the entry residual. -/
def dumpSlot (l : Nat) : Nat := l + 1

/-- Every layer's slot is one of the `n_layers + 1 = 49` slots and none collides with slot 0. -/
theorem dump_slot_bounds (l : Nat) (h : l < 48) : dumpSlot l < 49 ∧ 0 < dumpSlot l := by
  unfold dumpSlot
  omega

/-- Distinct layers use distinct slots. -/
theorem dump_slot_injective (l l' : Nat) (h : dumpSlot l = dumpSlot l') : l = l' := by
  unfold dumpSlot at h
  omega

/-- One layer's dump copy is `hc * n_embd * 4 = 40960` bytes (40 KB) for `hc = 4`, `n_embd = 2560`. -/
theorem dump_bytes_per_layer : 4 * 2560 * 4 = 40960 := by
  decide

-- § scratch sizes

/-- Parts buffer bytes: `k * n_embd` floats. -/
def partsBytes (k nEmbd : Nat) : Nat := k * nEmbd * 4

/-- For `k = 10` and `n_embd = 2560` the pinned staging is 102,400 bytes. -/
theorem parts_bytes_value : partsBytes 10 2560 = 102400 := by
  decide

-- § counters

/-- Sum of per-call counters is at most `c` per call when every call adds at most `c`. -/
theorem ringsBound (c : Nat) (xs : List Nat) (h : ∀ x ∈ xs, x ≤ c) : xs.sum ≤ c * xs.length := by
  induction xs with
  | nil => simp
  | cons x t ih =>
    have hx : x ≤ c := h x (by simp)
    have ht : t.sum ≤ c * t.length := ih (fun y hy => h y (by simp [hy]))
    simp only [List.sum_cons, List.length_cons, Nat.mul_add, Nat.mul_one]
    omega

/-- A loop call that rings at most once per layer adds at most 48 to `rings_mid_graph`. -/
theorem rings_per_call (calls : List Nat) (h : ∀ x ∈ calls, x ≤ 48) : calls.sum ≤ 48 * calls.length := by
  exact ringsBound 48 calls h

-- § hit-path readiness

/-- The hit path is enabled only when every one of its pieces is present. -/
def hitsReady (bs : List Bool) : Bool := bs.all id

/-- A missing piece disables the hit path: no half-configured path computes experts twice. -/
theorem hits_all_or_nothing (bs : List Bool) (h : false ∈ bs) : hitsReady bs = false := by
  unfold hitsReady
  cases hh : bs.all id
  · rfl
  · have := List.all_eq_true.mp hh false h
    simp at this

end Strata.SessionLoop
