/-!
# MODULE: speculative_verify_window
SOURCES: src/core/verify.cpp
The verify window of speculative decoding: T tokens (the last accepted one plus drafts) run through all
48 layers in one captured graph, optionally split into two token groups A and B so the CPU computes one
group's experts while the GPU runs the other's mixer.  Each (layer, group) rings the doorbell with a
unique sequence number; the host serves the pool in that order.  After the window the argmax prefix
that agrees with the drafts is accepted and `commit` replays the state updates of the kept tokens.

## Source semantics
* Groups: `G = 2` iff split and `T ≥ 2`; A = [0, ⌈T/2⌉), B = [⌈T/2⌉, T); otherwise one group [0, T).
* Ring number of (layer l, group g) is `l*G + g + 1`; the host's k-th wait is `k + 1` with
  `l = k / G`, `g = k % G`, so host and graph agree on every ring.
* Accepted count `a`: the longest prefix with `window[i+1] = out[i]`; the round emits `a + 1 ≤ T` tokens.
* `commit` stages `[n_keep, n_keep-1, pos0, pos0+1, …, -1, …]`.
* Flag B only rises (`raise_flag` is a compare-exchange max).
* The GPU plan's u64 pointer block starts at an even i32 offset (8-byte aligned) and the plan's regions
  (counts, start, dst, tok, ptr, ptr2, start2) fit `plan_i32`.
* A window may not run past the context: `pos0 + T ≤ max_cells`.

## Form template identifiers
groupBegin/groupEnd ↔ tb_/te_ · ringOf ↔ `l * G + grp + 1` · acceptLen ↔ the `while (a < T-1 && …)` loop ·
commitRow ↔ h_commit_ · raise ↔ raise_flag · ptrOff/planI32 ↔ ptr_off / plan_i32_ · windowFits ↔ run guard
-/

namespace SpeculativeVerifyWindow

-- § token groups

/-- Number of groups for a window of `T` tokens. -/
def groups (split : Bool) (T : Nat) : Nat := if split = true ∧ 2 ≤ T then 2 else 1

/-- First token of group `g`. -/
def groupBegin (T g : Nat) : Nat := if g = 0 then 0 else (T + 1) / 2

/-- One past the last token of group `g`. -/
def groupEnd (G T g : Nat) : Nat := if g = 0 then (if G = 2 then (T + 1) / 2 else T) else T

/-- A split window's groups partition `[0, T)`: A starts at 0, B starts where A ends, B ends at T. -/
theorem split_partition (T : Nat) :
    groupBegin T 0 = 0 ∧ groupEnd 2 T 0 = groupBegin T 1 ∧ groupEnd 2 T 1 = T := by
  simp [groupBegin, groupEnd]

/-- Both groups of a split window are non-empty, so each rings. -/
theorem split_nonempty (T : Nat) (hT : 2 ≤ T) :
    groupBegin T 0 < groupEnd 2 T 0 ∧ groupBegin T 1 < groupEnd 2 T 1 := by
  simp only [groupBegin, groupEnd, ite_true]
  simp
  omega

/-- An unsplit window is a single group covering every token. -/
theorem unsplit_whole (T : Nat) : groupBegin T 0 = 0 ∧ groupEnd 1 T 0 = T := by
  simp [groupBegin, groupEnd]

/-- A single-token window is never split. -/
theorem one_token_one_group (s : Bool) : groups s 1 = 1 := by
  simp [groups]

-- § ring numbering

/-- The doorbell value the graph publishes for (layer, group). -/
def ringOf (G l g : Nat) : Nat := l * G + g + 1

/-- The host's k-th wait is exactly the graph's ring for (k / G, k % G). -/
theorem host_matches_graph (G k : Nat) : ringOf G (k / G) (k % G) = k + 1 := by
  unfold ringOf
  have := Nat.div_add_mod k G
  rw [Nat.mul_comm] at this
  omega

/-- Rings increase along the stream order (l, g) lexicographically, so `seq ≥ want` is monotone. -/
theorem ring_increasing (G l g l' g' : Nat) (hg : g < G) (_hg' : g' < G)
    (h : l < l' ∨ (l = l' ∧ g < g')) : ringOf G l g < ringOf G l' g' := by
  unfold ringOf
  rcases h with h | ⟨rfl, h⟩
  · have : (l + 1) * G ≤ l' * G := Nat.mul_le_mul_right G h
    rw [Nat.add_mul, Nat.one_mul] at this
    omega
  · omega

/-- The last ring of a window is `n_layers * G`: the host performs exactly that many waits. -/
theorem last_ring (G L : Nat) (hG : 0 < G) (hL : 0 < L) : ringOf G (L - 1) (G - 1) = L * G := by
  unfold ringOf
  have h1 : (L - 1) * G + G = L * G := by
    have : L - 1 + 1 = L := by omega
    calc (L - 1) * G + G = (L - 1 + 1) * G := by rw [Nat.add_mul, Nat.one_mul]
      _ = L * G := by rw [this]
  omega

-- § acceptance

/-- Longest prefix of drafts confirmed by the window's argmax (`window[i+1] = out[i]`). -/
def acceptLen : List (Nat × Nat) → Nat
  | [] => 0
  | (d, o) :: ps => if d = o then 1 + acceptLen ps else 0

/-- At most every draft is accepted. -/
theorem accept_le : ∀ ps : List (Nat × Nat), acceptLen ps ≤ ps.length
  | [] => Nat.le_refl 0
  | (d, o) :: ps => by
    simp only [acceptLen, List.length_cons]
    have := accept_le ps
    split <;> omega

/-- A window of `T` tokens compares `T - 1` drafts and emits `a + 1 ≤ T` tokens. -/
theorem emitted_le_window (ps : List (Nat × Nat)) (T : Nat) (h : ps.length + 1 = T) :
    acceptLen ps + 1 ≤ T := by
  have := accept_le ps
  omega

/-- Every round emits at least one token: the window's first argmax is always kept. -/
theorem round_emits (ps : List (Nat × Nat)) : 1 ≤ acceptLen ps + 1 := by
  omega

/-- A rejected first draft keeps exactly one token. -/
theorem reject_first (d o : Nat) (ps : List (Nat × Nat)) (h : d ≠ o) : acceptLen ((d, o) :: ps) = 0 := by
  simp [acceptLen, h]

-- § commit

/-- Entry `t` of the commit's position list: a kept position or `-1`. -/
def commitRow (pos0 : Int) (nKeep t : Nat) : Int := if t < nKeep then pos0 + t else -1

/-- Kept entries are consecutive positions from `pos0`. -/
theorem commit_consecutive (pos0 : Int) (n t : Nat) (h : t + 1 < n) :
    commitRow pos0 n (t + 1) = commitRow pos0 n t + 1 := by
  unfold commitRow
  have h1 : t + 1 < n := h
  have h2 : t < n := by omega
  simp only [h1, h2, ite_true]
  push_cast
  omega

/-- Entries past the kept count are the `-1` sentinel the commit kernels skip. -/
theorem commit_sentinel (pos0 : Int) (n t : Nat) (h : n ≤ t) : commitRow pos0 n t = -1 := by
  unfold commitRow
  have : ¬ t < n := by omega
  simp [this]

-- § the monotone flag

/-- `raise_flag`: the compare-exchange loop leaves the larger of the current value and the new one. -/
def raise (cur v : Nat) : Nat := max cur v

/-- Flag B never decreases, so a late host function cannot undo a later layer's raise. -/
theorem raise_monotone (cur v : Nat) : cur ≤ raise cur v := by
  unfold raise; omega

/-- After a raise the flag satisfies the waiter for `v`. -/
theorem raise_reaches (cur v : Nat) : v ≤ raise cur v := by
  unfold raise; omega

/-- Raises commute: the order host functions complete in does not matter. -/
theorem raise_comm (cur a b : Nat) : raise (raise cur a) b = raise (raise cur b) a := by
  unfold raise; omega

-- § the GPU plan layout (in i32 units)

/-- Offset of the u64 pointer block: the i32 header rounded up to even. -/
def ptrOff (cap : Nat) : Nat := (4 + (cap + 1) + cap + cap + 1) / 2 * 2

/-- Total i32 words of one plan slot. -/
def planI32 (cap : Nat) : Nat := ptrOff cap + 4 * cap + (cap + 1) + 1

/-- The pointer block is 8-byte aligned (an even number of i32 words). -/
theorem ptrOff_even (cap : Nat) : ptrOff cap % 2 = 0 := by
  unfold ptrOff; omega

/-- The pointer block starts after counts, start, dst and tok. -/
theorem ptrOff_after_header (cap : Nat) : 4 + (cap + 1) + cap + cap ≤ ptrOff cap := by
  unfold ptrOff; omega

/-- `ptr` (cap u64) and `ptr2` (cap u64) end where `start2` begins, and `start2` (cap + 1) fits the slot. -/
theorem plan_regions_fit (cap : Nat) :
    ptrOff cap + 2 * cap + 2 * cap = ptrOff cap + 4 * cap ∧
      ptrOff cap + 4 * cap + (cap + 1) < planI32 cap := by
  unfold planI32; omega

-- § the context guard

/-- `run` refuses a window that would run past the context. -/
def windowFits (pos0 T maxCells : Nat) : Prop := pos0 + T ≤ maxCells

/-- Every position the window stages is a valid cell. -/
theorem window_cells_valid (pos0 T m t : Nat) (h : windowFits pos0 T m) (ht : t < T) : pos0 + t < m := by
  unfold windowFits at h; omega

end SpeculativeVerifyWindow
