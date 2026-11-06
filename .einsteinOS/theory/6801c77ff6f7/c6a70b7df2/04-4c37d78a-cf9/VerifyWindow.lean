/-!
# MODULE: verify_window
SOURCES: include/strata/core/verify.hpp, src/core/verify.cpp
The speculative verify window: T tokens through all layers in one captured graph, optionally split in two token
groups whose CPU expert work is pipelined against the GPU work of the other group; `commit(n_keep)` makes a prefix
permanent.

## Source semantics
* With `G = 2` the tokens split into A = `[0, (T+1)/2)` and B = `[(T+1)/2, T)`; with `G = 1` one group holds all.
* Ring `l*G + grp + 1` numbers every (layer, group) hand-off; the host serves them in order `k = l*G + grp`.
* The GPU plan block is `counts(4) | start(cap+1) | dst(cap) | tok(cap) | pad | ptr(cap u64) | ptr2(cap u64) |
  start2(cap+1) | 1`, one block per group.
* Staging holds 16 expert blobs, halved between the groups of a split window.
* `commit` writes `[n_keep, n_keep-1, pos0 .. pos0+T-1]` with `-1` for rejected rows, and advances the PLE history
  so that its newest entry is the last kept token.
* Flag B only rises (`raise_flag` is a monotone compare-and-set).

## Form template identifiers
role mapping (source names -> abstract names):
  Verifier::run         -> window_run
  Verifier::commit      -> window_commit
  record_window pre/post-> group_pipeline
  raise_flag            -> monotone_flag
  plan block / sink_    -> gpu_plan_block
  kStagingBlobs         -> staging_slots
  ple_prev              -> token_history_pair
-/

namespace Strata.Verify

-- § token groups

/-- Group count of a window: two when splitting is on and the window has at least two tokens. -/
def groupsOf (split : Bool) (T : Nat) : Nat := if split = true ∧ 2 ≤ T then 2 else 1

/-- First token of group `grp`. -/
def tb (T : Nat) : Nat → Nat
  | 0 => 0
  | _ + 1 => (T + 1) / 2

/-- One past the last token of group `grp`. -/
def te (G T : Nat) : Nat → Nat
  | 0 => (if G = 2 then (T + 1) / 2 else T)
  | _ + 1 => T

/-- The group count is 1 or 2. -/
theorem groupsOf_cases (split : Bool) (T : Nat) : groupsOf split T = 1 ∨ groupsOf split T = 2 := by
  unfold groupsOf
  split <;> simp

/-- Without splitting, the single group is the whole window. -/
theorem single_group_whole (T : Nat) : tb T 0 = 0 ∧ te 1 T 0 = T := by
  simp [tb, te]

/-- In a split window group B starts exactly where group A ends. -/
theorem groups_adjacent (T : Nat) : te 2 T 0 = tb T 1 := by
  simp [te, tb]

/-- Both groups of a split window are non-empty when `T >= 2`. -/
theorem group_sizes_pos (T : Nat) (hT : 2 ≤ T) :
    1 ≤ te 2 T 0 - tb T 0 ∧ 1 ≤ te 2 T 1 - tb T 1 := by
  show 1 ≤ (T + 1) / 2 - 0 ∧ 1 ≤ T - (T + 1) / 2
  omega

/-- The two groups of a split window partition the window. -/
theorem group_sizes_sum (T : Nat) :
    (te 2 T 0 - tb T 0) + (te 2 T 1 - tb T 1) = T := by
  show ((T + 1) / 2 - 0) + (T - (T + 1) / 2) = T
  omega

/-- Group A is at least as large as group B. -/
theorem groupA_ge_groupB (T : Nat) : te 2 T 1 - tb T 1 ≤ te 2 T 0 - tb T 0 := by
  show T - (T + 1) / 2 ≤ (T + 1) / 2 - 0
  omega

-- § ring numbering

/-- Ring (sequence) number of the hand-off of layer `l`, group `grp`. -/
def ringOf (l G grp : Nat) : Nat := l * G + grp + 1

/-- Decomposing a global index `k` into (layer, group) and re-ringing gives `k + 1`. -/
theorem ring_of_index (k G : Nat) : ringOf (k / G) G (k % G) = k + 1 := by
  unfold ringOf
  have h := Nat.div_add_mod k G
  rw [Nat.mul_comm (k / G) G]
  omega

/-- Distinct hand-offs get distinct ring numbers. -/
theorem ring_of_index_inj (k1 k2 G : Nat)
    (h : ringOf (k1 / G) G (k1 % G) = ringOf (k2 / G) G (k2 % G)) : k1 = k2 := by
  rw [ring_of_index, ring_of_index] at h
  omega

/-- The serving loop reaches every layer index below `n_layers` and no other. -/
theorem layer_of_index_lt (k L G : Nat) (hG : 0 < G) (h : k < L * G) : k / G < L :=
  (Nat.div_lt_iff_lt_mul hG).2 h

/-- The group index of a hand-off is a valid group. -/
theorem group_of_index_lt (k G : Nat) (hG : 0 < G) : k % G < G :=
  Nat.mod_lt k hG

-- § GPU plan block layout

/-- Words before the pointer arrays: counts, start, dst, tok. -/
def planBase (cap : Nat) : Nat := 4 + (cap + 1) + cap + cap

/-- Offset of the 8-byte-aligned pointer arrays (`(i32 + 1) & ~1`). -/
def ptrOff (cap : Nat) : Nat := (planBase cap + 1) / 2 * 2

/-- Total 32-bit words of one plan block. -/
def planWords (cap : Nat) : Nat := ptrOff cap + 4 * cap + (cap + 1) + 1

/-- The pointer arrays start at an even word offset. -/
theorem ptrOff_even (cap : Nat) : ptrOff cap % 2 = 0 := by
  unfold ptrOff; omega

/-- The pointer arrays begin at or after the index arrays (no overlap). -/
theorem ptrOff_ge (cap : Nat) : planBase cap ≤ ptrOff cap := by
  unfold ptrOff; omega

/-- Alignment padding is at most one word. -/
theorem ptrOff_le (cap : Nat) : ptrOff cap ≤ planBase cap + 1 := by
  unfold ptrOff; omega

/-- `ptr` and `ptr2` (each `cap` 64-bit entries) fit before `start2`. -/
theorem ptr_arrays_before_start2 (cap : Nat) : ptrOff cap + 2 * cap + 2 * cap = ptrOff cap + 4 * cap := by
  omega

/-- `start2` and the trailing pad word end exactly at the end of the block. -/
theorem start2_end (cap : Nat) : (ptrOff cap + 4 * cap) + (cap + 1) + 1 = planWords cap := by
  unfold planWords; omega

/-- Group `grp`'s plan block lies inside the two-block region. -/
theorem plan_slot_in_range (cap grp : Nat) (h : grp < 2) :
    grp * planWords cap + planWords cap ≤ 2 * planWords cap := by
  have : grp * planWords cap ≤ 1 * planWords cap := Nat.mul_le_mul_right _ (by omega)
  omega

-- § expert staging

/-- Staging blobs given to each group. -/
def stagePer (G : Nat) : Nat := if G = 2 then 16 / 2 else 16

/-- Each group's staging slice lies inside the 16 blobs. -/
theorem stage_slice_in_range (G grp : Nat) (hG : G = 1 ∨ G = 2) (hgrp : grp < G) :
    grp * stagePer G + stagePer G ≤ 16 := by
  rcases hG with h | h
  · subst h
    have : grp = 0 := by omega
    subst this
    simp [stagePer]
  · subst h
    unfold stagePer
    simp
    omega

/-- The two slices of a split window are disjoint. -/
theorem stage_slices_disjoint : 0 * stagePer 2 + stagePer 2 ≤ 1 * stagePer 2 := by
  simp [stagePer]

-- § monotone flag

/-- `raise_flag`: compare-and-set that only ever increases the flag. -/
def raiseFlag (cur v : Nat) : Nat := max cur v

/-- The flag never decreases. -/
theorem raiseFlag_ge_cur (cur v : Nat) : cur ≤ raiseFlag cur v := by
  unfold raiseFlag; omega

/-- The flag reaches the raised value. -/
theorem raiseFlag_ge_val (cur v : Nat) : v ≤ raiseFlag cur v := by
  unfold raiseFlag; omega

/-- Late and early raisers commute: the order of host functions does not matter. -/
theorem raiseFlag_comm (c a b : Nat) : raiseFlag (raiseFlag c a) b = raiseFlag (raiseFlag c b) a := by
  unfold raiseFlag; omega

/-- Raising twice with the same value is the same as raising once. -/
theorem raiseFlag_idem (c a : Nat) : raiseFlag (raiseFlag c a) a = raiseFlag c a := by
  unfold raiseFlag; omega

-- § commit

/-- Entry `t` of the commit array: the position when kept, `-1` when rejected. -/
def commitEntry (pos0 : Int) (nKeep t : Nat) : Int := if t < nKeep then pos0 + (t : Int) else -1

/-- With a non-negative base position, entry `t` is valid exactly for kept rows. -/
theorem commitEntry_valid (pos0 : Int) (nKeep t : Nat) (h : 0 ≤ pos0) :
    0 ≤ commitEntry pos0 nKeep t ↔ t < nKeep := by
  unfold commitEntry
  split <;> omega

/-- The last kept row (`n_keep - 1`) carries its true position. -/
theorem commitEntry_last (pos0 : Int) (nKeep : Nat) (h : 1 ≤ nKeep) :
    commitEntry pos0 nKeep (nKeep - 1) = pos0 + ((nKeep - 1 : Nat) : Int) := by
  unfold commitEntry
  split
  · rfl
  · omega

/-- `n_keep - 1` indexes a kept row. -/
theorem last_kept_lt (nKeep : Nat) (h : 1 ≤ nKeep) : nKeep - 1 < nKeep := by
  omega

/-- Commit argument check: `1 <= n_keep <= T of the last window`. -/
def commitOk (nKeep lastT : Nat) : Bool := decide (1 ≤ nKeep ∧ nKeep ≤ lastT)

/-- An accepted commit keeps at least one token (the verify pass always commits one). -/
theorem commitOk_pos (nKeep lastT : Nat) (h : commitOk nKeep lastT = true) : 1 ≤ nKeep := by
  have h' : 1 ≤ nKeep ∧ nKeep ≤ lastT := of_decide_eq_true h
  omega

-- § PLE history after a commit

/-- One token enters the two-token history `ple_prev`. -/
def pleStep (p : Nat × Nat) (tok : Nat) : Nat × Nat := (p.2, tok)

/-- Advance the history over a list of kept tokens. -/
def pleAdvance (p : Nat × Nat) (toks : List Nat) : Nat × Nat := toks.foldl pleStep p

/-- Appending a token shifts the previous newest token into the older slot. -/
theorem pleAdvance_append (p : Nat × Nat) (l : List Nat) (x : Nat) :
    pleAdvance p (l ++ [x]) = ((pleAdvance p l).2, x) := by
  simp [pleAdvance, List.foldl_append, pleStep]

/-- After a commit the newest history entry is the last kept token. -/
theorem pleAdvance_snd (p : Nat × Nat) (l : List Nat) (x : Nat) :
    (pleAdvance p (l ++ [x])).2 = x := by
  simp [pleAdvance_append]

/-- Committing no token leaves the history untouched. -/
theorem pleAdvance_nil (p : Nat × Nat) : pleAdvance p [] = p := rfl

-- § window argument checks

/-- `init` accepts a window of `2 .. kMax` tokens that the CPU pool also supports. -/
def initOk (maxT kMax cpuMax : Nat) : Bool := decide (2 ≤ maxT ∧ maxT ≤ kMax ∧ maxT ≤ cpuMax)

/-- A valid window is at least two tokens and fits the graph caches indexed `0 .. 8`. -/
theorem initOk_bounds (maxT kMax cpuMax : Nat) (hk : kMax ≤ 8) (h : initOk maxT kMax cpuMax = true) :
    2 ≤ maxT ∧ maxT < 9 := by
  have h' : 2 ≤ maxT ∧ maxT ≤ kMax ∧ maxT ≤ cpuMax := of_decide_eq_true h
  omega

/-- `run` accepts `1 <= T <= max_t` with the window inside the cache. -/
def runOk (T maxT pos0 maxCells : Nat) : Bool := decide (1 ≤ T ∧ T ≤ maxT ∧ pos0 + T ≤ maxCells)

/-- Every position of an accepted window is a valid cache cell. -/
theorem runOk_positions (T maxT pos0 maxCells t : Nat) (h : runOk T maxT pos0 maxCells = true)
    (ht : t < T) : pos0 + t < maxCells := by
  have h' : 1 ≤ T ∧ T ≤ maxT ∧ pos0 + T ≤ maxCells := of_decide_eq_true h
  omega

end Strata.Verify
