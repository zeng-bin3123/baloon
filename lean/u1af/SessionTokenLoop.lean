/-!
# MODULE: session_token_loop
SOURCES: include/ref/core/session.hpp
One token through 48 layers: per-layer state indexing (12 QSA / 36 GDN), the per-layer graph pair
(`pre`/`post`), the `dump_layers` snapshot buffer layout, the prefix-graph indexing, and scratch teardown.

## Source semantics
* `qsa_states` has one entry per QSA layer, `gdn_state` one per GDN layer.
* The token path replays `pre[l]` and `post[l]` per layer: 96 graph launches, `session_replay` only 48.
* `dump_layers` holds `(n_layers + 1) * hc * n_embd` floats: slot 0 is the entry residual, slot `l+1` is after layer `l`.
* `preP[k-1]` (k = 1..5) are the five prefix graphs.
* `SessionLoopScratch::free` is safe to call twice.

## Form template identifiers
SessionState -> sequence_state; SessionGraphs -> layer_graph_set; session_loop -> host_layer_loop;
SessionLoopScratch -> loop_scratch; TokenGraph -> whole_token_graph; dump_layers -> residual_ladder.
-/

namespace SessionTokenLoop

-- § per-kind state indices

/-- Whether layer `l` is QSA. -/
def isQsa (l : Nat) : Prop := l % 4 = 3

/-- Index of a GDN layer among the 36 GDN layers (QSA layers below it are `l / 4`). -/
def gdnIdx (l : Nat) : Nat := l - l / 4

/-- Index of a QSA layer among the 12 QSA layers. -/
def qsaIdx (l : Nat) : Nat := l / 4

/-- GDN state indices are within the 36 states. -/
theorem gdnIdx_lt {l : Nat} (hl : l < 48) (hg : l % 4 ≠ 3) : gdnIdx l < 36 := by
  unfold gdnIdx
  omega

/-- QSA state indices are within the 12 states. -/
theorem qsaIdx_lt {l : Nat} (hl : l < 48) : qsaIdx l < 12 := by
  unfold qsaIdx
  omega

/-- Distinct GDN layers get distinct state slots (no two layers share a recurrence). -/
theorem gdnIdx_injective {l1 l2 : Nat} (h1 : l1 % 4 ≠ 3) (h2 : l2 % 4 ≠ 3)
    (h : gdnIdx l1 = gdnIdx l2) : l1 = l2 := by
  unfold gdnIdx at h
  omega

/-- Distinct QSA layers get distinct KV states. -/
theorem qsaIdx_injective {l1 l2 : Nat} (h1 : l1 % 4 = 3) (h2 : l2 % 4 = 3)
    (h : qsaIdx l1 = qsaIdx l2) : l1 = l2 := by
  unfold qsaIdx at h
  omega

-- § graph launches

/-- Graph launches per token: `pre` and `post` for each of `n` layers. -/
def launchesFull (n : Nat) : Nat := 2 * n

/-- `session_replay` launches only the `pre` graphs. -/
def launchesPreOnly (n : Nat) : Nat := n

/-- The full token is 96 launches for 48 layers. -/
theorem full_is_96 : launchesFull 48 = 96 := by decide

/-- `session_replay` under-counts the GPU work: it omits all `post` graphs. -/
theorem replay_omits_post {n : Nat} (h : 0 < n) : launchesPreOnly n < launchesFull n := by
  unfold launchesPreOnly launchesFull
  omega

/-- `preP[k-1]` for `k` in 1..5 is within the five-entry prefix array. -/
theorem prefix_index_lt {k : Nat} (h1 : 1 ≤ k) (h2 : k ≤ 5) : k - 1 < 5 := by omega

-- § dump_layers layout

/-- Start offset of snapshot slot `s` in the dump buffer. -/
def slotOff (stride s : Nat) : Nat := s * stride

/-- Slot `l + 1` (after layer `l`) ends inside the `(n_layers + 1) * stride` buffer. -/
theorem dump_slot_in_bounds {stride n l : Nat} (hl : l < n) :
    slotOff stride (l + 1) + stride ≤ (n + 1) * stride := by
  unfold slotOff
  have h1 : (l + 1 + 1) * stride ≤ (n + 1) * stride := Nat.mul_le_mul_right _ (by omega)
  have e : (l + 1 + 1) * stride = (l + 1) * stride + stride := Nat.succ_mul _ _
  rw [e] at h1
  exact h1

/-- Slots do not overlap: slot `s + 1` starts where slot `s` ends. -/
theorem dump_slots_abut (stride s : Nat) : slotOff stride s + stride = slotOff stride (s + 1) := by
  unfold slotOff
  rw [Nat.add_mul, Nat.one_mul]

-- § scratch teardown

/-- Whether the scratch currently owns resources (the host thread is pinned). -/
structure Scratch where
  pinned : Bool
  allocated : Bool

/-- `free` releases everything. -/
def Scratch.free (_ : Scratch) : Scratch := { pinned := false, allocated := false }

/-- Calling `free` twice equals calling it once. -/
theorem free_idempotent (s : Scratch) : s.free.free = s.free := rfl

end SessionTokenLoop
