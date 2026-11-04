/-!
# MODULE: verify_window
SOURCES: src/core/verify.cpp, include/strata/core/verify.hpp
The speculative verify window: T tokens through all layers in one captured graph, optionally split into two
token groups pipelined against the CPU experts, with a commit step that keeps the first `n_keep` tokens.

## Source semantics
* Split window: groups A = [0, (T+1)/2) and B = [(T+1)/2, T); with one group the range is [0, T).
* Ring/flag value for (layer l, group g) is `l*G + g + 1`; the host loop recovers (l, g) = (k / G, k % G).
* The stream order is pre(0,A) pre(0,B) | post(0,A) pre(1,A) | post(0,B) pre(1,B) | post(1,A) pre(2,A) ...
* The GPU plan block lays out counts | start | dst | tok | pad | ptr | ptr2 | start2 in int32 words.
* Flag B only rises (`raise_flag` is an atomic max).
* `commit(n_keep)` writes `[n_keep, n_keep-1, pos0..pos0+n_keep-1, -1...]` and advances the PLE predecessors.

## Form template identifiers
- groupBegin/groupEnd → tb_/te_ ; ringValue → `ring` ; preIdx/postIdx → position in the captured stream
- planPtrOff → ptr_off ; raiseFlag → Verifier::raise_flag ; advance → ss.ple_prev update
-/

namespace Strata.VerifyWindow

-- § token groups

/-- First token of group `grp` for a window of `T` tokens. -/
def groupBegin (T grp : Nat) : Nat := if grp = 0 then 0 else (T + 1) / 2

/-- End (exclusive) of group `grp`, with `G` groups. -/
def groupEnd (T G grp : Nat) : Nat := if grp = 0 then (if G = 2 then (T + 1) / 2 else T) else T

/-- With two groups the two ranges tile [0, T). -/
theorem groups_tile (T : Nat) :
    groupBegin T 0 = 0 ∧ groupEnd T 2 0 = groupBegin T 1 ∧ groupEnd T 2 1 = T := by
  simp [groupBegin, groupEnd]

/-- With two groups the sizes add up to T. -/
theorem groups_sizes (T : Nat) :
    (groupEnd T 2 0 - groupBegin T 0) + (groupEnd T 2 1 - groupBegin T 1) = T := by
  simp only [groupBegin, groupEnd]
  simp
  omega

/-- For a window of at least two tokens both groups are non-empty. -/
theorem groups_nonempty (T : Nat) (hT : 2 ≤ T) :
    groupBegin T 0 < groupEnd T 2 0 ∧ groupBegin T 1 < groupEnd T 2 1 := by
  simp only [groupBegin, groupEnd]
  simp
  omega

/-- With one group the single range is [0, T). -/
theorem single_group (T : Nat) : groupBegin T 0 = 0 ∧ groupEnd T 1 0 = T := by
  simp [groupBegin, groupEnd]

-- § ring numbering

/-- Ring/flag value for (layer, group). -/
def ringValue (l G grp : Nat) : Nat := l * G + grp + 1

/-- The host's `want = k + 1` with `k = l*G + grp` equals the graph's ring value. -/
theorem ring_host_matches (l G grp : Nat) : (l * G + grp) + 1 = ringValue l G grp := by
  unfold ringValue; rfl

/-- With two groups the host recovers the layer as `k / 2`. -/
theorem ring_recover_layer (l grp : Nat) (h : grp < 2) : (l * 2 + grp) / 2 = l := by
  omega

/-- With two groups the host recovers the group as `k % 2`. -/
theorem ring_recover_group (l grp : Nat) (h : grp < 2) : (l * 2 + grp) % 2 = grp := by
  omega

/-- Ring values are strictly increasing in (layer, group) order for two groups. -/
theorem ring_strict (l grp l' grp' : Nat) (h : grp < 2) (h' : grp' < 2) (hl : l < l') :
    ringValue l 2 grp < ringValue l' 2 grp' := by
  unfold ringValue; omega

-- § stream order of the split window

/-- Position of `pre(l, g)` in the captured stream (two groups). -/
def preIdx (l g : Nat) : Nat := if l = 0 then g else 4 * l - 1 + 2 * g

/-- Position of `post(l, g)` in the captured stream (two groups). -/
def postIdx (l g : Nat) : Nat := 2 + 4 * l + 2 * g

/-- `post(l,g)` comes after `pre(l,g)`. -/
theorem pre_before_post (l g : Nat) (_hg : g < 2) : preIdx l g < postIdx l g := by
  unfold preIdx postIdx
  split <;> omega

/-- `pre(l+1,g)` comes after `post(l,g)`. -/
theorem post_before_next_pre (l g : Nat) (_hg : g < 2) : postIdx l g < preIdx (l + 1) g := by
  unfold preIdx postIdx
  split <;> omega

/-- Group B's `pre` follows group A's `pre` of the same layer. -/
theorem preB_after_preA (l : Nat) : preIdx l 0 < preIdx l 1 := by
  unfold preIdx
  split <;> omega

/-- The overlap property: A's next `pre` is queued before B's `post` of this layer. -/
theorem nextPreA_before_postB (l : Nat) : preIdx (l + 1) 0 < postIdx l 1 := by
  unfold preIdx postIdx
  split <;> omega

/-- `post(l,A)` precedes `post(l,B)`. -/
theorem postA_before_postB (l : Nat) : postIdx l 0 < postIdx l 1 := by
  unfold postIdx; omega

-- § GPU plan block layout

/-- int32 words before the pad: counts(4) | start(cap+1) | dst(cap) | tok(cap). -/
def planI32 (cap : Nat) : Nat := 4 + (cap + 1) + cap + cap

/-- `ptr_off`: the 64-bit arrays start on an even word. -/
def planPtrOff (cap : Nat) : Nat := (planI32 cap + 1) / 2 * 2

/-- Total int32 words: ptr_off + 4*cap (ptr and ptr2) + (cap+1) (start2) + 1. -/
def planTotal (cap : Nat) : Nat := planPtrOff cap + 4 * cap + (cap + 1) + 1

/-- The pointer arrays are 8-byte aligned (even word offset). -/
theorem planPtrOff_even (cap : Nat) : planPtrOff cap % 2 = 0 := by
  unfold planPtrOff; omega

/-- The pad is at most one word. -/
theorem planPtrOff_bounds (cap : Nat) : planI32 cap ≤ planPtrOff cap ∧ planPtrOff cap ≤ planI32 cap + 1 := by
  unfold planPtrOff; omega

/-- `start2` fits inside the block. -/
theorem plan_start2_fits (cap : Nat) : planPtrOff cap + 4 * cap + (cap + 1) ≤ planTotal cap := by
  unfold planTotal; omega

/-- The `post` lambda's formula for `ptr_off` is the same as the init formula. -/
theorem plan_post_formula (cap : Nat) : ((4 + (cap + 1) + 2 * cap) + 1) / 2 * 2 = planPtrOff cap := by
  unfold planPtrOff planI32; omega

-- § staging split

/-- Staging blobs per group: half each with two groups. -/
def stagingPer (G : Nat) : Nat := if G = 2 then 16 / 2 else 16

/-- Group staging ranges are disjoint and fit in the 16 staging blobs. -/
theorem staging_fits (G grp : Nat) (hG : G = 1 ∨ G = 2) (hg : grp < G) :
    grp * stagingPer G + stagingPer G ≤ 16 := by
  unfold stagingPer
  rcases hG with rfl | rfl
  · have : grp = 0 := by omega
    subst this
    simp
  · simp
    omega

-- § flag B is a monotone max

/-- `raise_flag`: store `v` only if it is larger than the current value. -/
def raiseFlag (cur v : Nat) : Nat := max cur v

/-- Raising never lowers the flag. -/
theorem raiseFlag_ge_cur (cur v : Nat) : cur ≤ raiseFlag cur v := by
  unfold raiseFlag; omega

/-- After raising to `v` the flag is at least `v`. -/
theorem raiseFlag_ge_val (cur v : Nat) : v ≤ raiseFlag cur v := by
  unfold raiseFlag; omega

/-- A late raise of an older layer cannot lower the flag. -/
theorem raiseFlag_late (cur v : Nat) (h : v ≤ cur) : raiseFlag cur v = cur := by
  unfold raiseFlag; omega

/-- Raising is order-insensitive. -/
theorem raiseFlag_comm (c a b : Nat) : raiseFlag (raiseFlag c a) b = raiseFlag (raiseFlag c b) a := by
  unfold raiseFlag; omega

-- § commit and PLE predecessors

/-- Predecessor window update of `commit`: shift in one committed token. -/
def advance (p : Int × Int) (tok : Int) : Int × Int := (p.2, tok)

/-- After committing at least two tokens, the predecessors are exactly the last two. -/
theorem advance_last_two (xs : List Int) (a b y z : Int) :
    (xs ++ [y, z]).foldl advance (a, b) = (y, z) := by
  simp [List.foldl_append, advance]

/-- Committing one token shifts the window by one. -/
theorem advance_one (a b z : Int) : [z].foldl advance (a, b) = (b, z) := by
  simp [advance]

/-- Exactly `n_keep` of the 8 commit slots carry a position. -/
theorem commit_slots_count : ∀ k, k ≤ 8 →
    ((List.range 8).filter (fun t => decide (t < k))).length = k := by
  decide

/-- Window sizes within the constructor bound index the 9-entry graph table safely. -/
theorem window_index_ok (T maxT kMax : Nat) (h1 : T ≤ maxT) (h2 : maxT ≤ kMax) (h3 : kMax ≤ 8) :
    T < 9 := by
  omega

end Strata.VerifyWindow
