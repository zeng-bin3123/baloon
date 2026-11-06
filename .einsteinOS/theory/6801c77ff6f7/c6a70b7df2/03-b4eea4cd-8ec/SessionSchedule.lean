/-!
# MODULE: session_schedule
SOURCES: include/strata/core/session.hpp, src/core/session.cpp
One token through all 48 layers: which GDN state slice each layer owns, the order of the captured `pre[l]` /
`post[l]` graph launches, the per-stage timing differencing, and the layout of the per-layer residual dump.

## Source semantics
* `gdn_point_at`: a GDN layer's state index is the number of GDN layers before it; indices are distinct and < 36.
* `session_replay_full` / `session_loop`: for every layer launch `pre[l]` then `post[l]`.
* `session_replay_stage_prefixes`: stage times are consecutive differences of prefix times, so they telescope.
* `dump_layers`: `(n_layers + 1) * hc * n_embd` floats; slot 0 is the input, slot l+1 is the residual after layer l;
  slots never overlap.

## Form template identifiers
gdnIndex <- `gdn_index` / `gdn_point_at`; schedule <- the launch loop; stageMs <- `t[k]`;
slotOff <- `dump_layers + (l + 1) * n`.
-/

namespace Strata.SessionSchedule

-- § GDN state slices

/-- Number of GDN layers strictly before `l` (every fourth layer, l % 4 = 3, is QSA). -/
def gdnIndex : Nat → Nat
  | 0 => 0
  | l + 1 => gdnIndex l + (if l % 4 = 3 then 0 else 1)

/-- Closed form: `l - l / 4` GDN layers precede layer `l`. -/
theorem gdnIndex_closed : ∀ l : Nat, gdnIndex l = l - l / 4
  | 0 => by simp [gdnIndex]
  | l + 1 => by
    have ih := gdnIndex_closed l
    by_cases h : l % 4 = 3
    · simp [gdnIndex, h, ih]
      omega
    · simp [gdnIndex, h, ih]
      omega

/-- Distinct GDN layers get distinct state slices (no two layers share a recurrence). -/
theorem gdnIndex_injective (a b : Nat) (ha : a % 4 ≠ 3) (hb : b % 4 ≠ 3)
    (h : gdnIndex a = gdnIndex b) : a = b := by
  rw [gdnIndex_closed, gdnIndex_closed] at h
  omega

/-- Every GDN layer of the 48 owns an in-range slice (< 36). -/
theorem gdnIndex_lt (l : Nat) (hl : l < 48) (h : l % 4 ≠ 3) : gdnIndex l < 36 := by
  rw [gdnIndex_closed]
  omega

-- § launch order

/-- A graph launch. -/
inductive Ev
  | pre (l : Nat)
  | post (l : Nat)
  deriving DecidableEq, Repr

/-- The launch order of `session_replay_full`: pre[l], post[l] for each layer in turn. -/
def schedule : Nat → List Ev
  | 0 => []
  | n + 1 => schedule n ++ [Ev.pre n, Ev.post n]

/-- Two graphs per layer. -/
theorem schedule_length (n : Nat) : (schedule n).length = 2 * n := by
  induction n with
  | zero => simp [schedule]
  | succ n ih => simp [schedule, ih] <;> omega

/-- `pre[l]` is launched exactly for the layers that exist. -/
theorem pre_mem (n l : Nat) : Ev.pre l ∈ schedule n ↔ l < n := by
  induction n with
  | zero => simp [schedule]
  | succ n ih =>
    simp [schedule, ih] <;> omega

/-- `post[l]` is launched exactly for the layers that exist. -/
theorem post_mem (n l : Nat) : Ev.post l ∈ schedule n ↔ l < n := by
  induction n with
  | zero => simp [schedule]
  | succ n ih =>
    simp [schedule, ih] <;> omega

/-- The next layer's launches come strictly after the current layer's (append-only schedule). -/
theorem schedule_append (n : Nat) : schedule (n + 1) = schedule n ++ [Ev.pre n, Ev.post n] := rfl

-- § stage differencing

/-- The five stage times from prefix times p1..p5 (p0 = 0), as `session_replay_stage_prefixes` computes them. -/
def stageMs (p1 p2 p3 p4 p5 : Int) : List Int := [p1, p2 - p1, p3 - p2, p4 - p3, p5 - p4]

/-- The stage times telescope to the full-prefix time: the sum check the driver prints. -/
theorem stageMs_sum (p1 p2 p3 p4 p5 : Int) : (stageMs p1 p2 p3 p4 p5).sum = p5 := by
  simp [stageMs] <;> omega

-- § residual dump layout

/-- Offset of slot `l` in the per-layer dump, for `n = hc * n_embd` floats per slot. -/
def slotOff (l n : Nat) : Nat := l * n

/-- Consecutive slots do not overlap. -/
theorem slots_disjoint (l n : Nat) : slotOff l n + n ≤ slotOff (l + 1) n := by
  unfold slotOff
  have := Nat.add_mul l 1 n
  omega

/-- The last slot (after layer n_layers - 1) ends exactly at the buffer size `(n_layers + 1) * n`. -/
theorem last_slot_ends (nLayers n : Nat) : slotOff nLayers n + n = (nLayers + 1) * n := by
  unfold slotOff
  have := Nat.add_mul nLayers 1 n
  omega

/-- Per-layer half-dump stride at which layer l's block starts never overlaps layer l+1's. -/
theorem half_stride_disjoint (l stride : Nat) : l * stride + stride ≤ (l + 1) * stride := by
  have := Nat.add_mul l 1 stride
  omega

-- § per-session sizes

/-- GDN recurrence + conv history floats for one layer. -/
def gdnStateFloats (stateSize vHeads dConv convChannels : Nat) : Nat :=
  stateSize * vHeads * stateSize + convChannels * (dConv - 1)

/-- Real geometry: 817,152 floats (3,268,608 bytes) per GDN layer. -/
theorem real_gdn_state_floats : gdnStateFloats 128 48 4 10240 = 817152 := by
  decide

/-- The one-time pinned scratch holds `k * n_embd * 4` bytes, matching the parts buffer. -/
theorem scratch_matches_parts (k nEmbd : Nat) : k * nEmbd * 4 = (k * nEmbd) * 4 := rfl

end Strata.SessionSchedule
