/-!
# MODULE: verify_window_protocol
The speculative verify window (`Verifier`): token-group split, ring numbering, staging split, mapped plan layout,
monotone flag raising and the commit vector.

## Source semantics
* With two groups, A = tokens `[0, (T+1)/2)` and B = the rest; the groups cover the window and are non-empty for `T ≥ 2`.
* The host's `k`-th ring maps to layer `k / G`, group `k % G`; the graph's ring number for `(l, grp)` is `l*G + grp + 1`.
* Staging blobs are split between groups without overlap.
* The mapped plan buffer holds `dst/tok/ptr/ptr2/start2` regions that do not overlap; two slots fit the allocation.
* Flag B only rises (`raise_flag` is a monotone max).
* `commit` stages `[n_keep, n_keep-1, pos0+t for t<n_keep else -1]`.

## Form template identifiers
`tb_`/`te_` -> groupBegin/groupEnd, `ring` -> ringNo, `kStagingBlobs` -> k, `plan_i32_` -> planTotal,
`raise_flag` -> raise, `h_commit_` -> commitEntry.
-/

namespace Strata.Verify

-- § token groups

/-- Size of group A in a split window. -/
def half (T : Nat) : Nat := (T + 1) / 2

/-- For `T ≥ 2` both groups are non-empty and together cover the window. -/
theorem groups_cover (T : Nat) (h : 2 ≤ T) :
    0 < half T ∧ half T < T ∧ half T + (T - half T) = T := by
  unfold half; omega

-- § ring numbering

/-- The graph's ring number for layer `l`, group `grp`, with `G` groups. -/
def ringNo (G l grp : Nat) : Nat := l * G + grp + 1

/-- Two groups: the host's `k`-th ring is layer `k/2`, group `k%2`. -/
theorem ring_decode_two (l grp : Nat) (hg : grp < 2) :
    (l * 2 + grp) / 2 = l ∧ (l * 2 + grp) % 2 = grp := by omega

/-- One group: the `k`-th ring is layer `k`, group 0. -/
theorem ring_decode_one (l : Nat) : (l * 1 + 0) / 1 = l ∧ (l * 1 + 0) % 1 = 0 := by omega

/-- Ring numbers are consecutive across the group boundary (two groups). -/
theorem ring_consecutive_two (l : Nat) : ringNo 2 (l + 1) 0 = ringNo 2 l 1 + 1 := by
  unfold ringNo; omega

/-- Within a layer the two groups' rings are consecutive. -/
theorem ring_consecutive_in_layer (l : Nat) : ringNo 2 l 1 = ringNo 2 l 0 + 1 := by
  unfold ringNo; omega

/-- Ring numbers are positive (a zeroed flag is never mistaken for a ring). -/
theorem ring_pos (G l grp : Nat) : 0 < ringNo G l grp := by unfold ringNo; omega

/-- The total number of rings is `n_layers * G`; the last one is `n_layers*G` for `grp = G-1`. -/
theorem ring_last_two (nl : Nat) (h : 0 < nl) : ringNo 2 (nl - 1) 1 = nl * 2 := by
  unfold ringNo; omega

-- § staging split

/-- Staging blobs per group. -/
def per (G k : Nat) : Nat := if G = 2 then k / 2 else k

/-- Group regions of staging are disjoint and inside `kStagingBlobs`. -/
theorem staging_disjoint (G k grp : Nat) (hG : G = 1 ∨ G = 2) (hg : grp < G) :
    grp * per G k + per G k ≤ k := by
  rcases hG with rfl | rfl
  · have : grp = 0 := by omega
    subst this
    simp [per]
  · rcases (by omega : grp = 0 ∨ grp = 1) with rfl | rfl <;> simp [per] <;> omega

-- § mapped plan layout (int32 words)

/-- Words before the u64 pointers: counts(4) | start(cap+1) | dst(cap) | tok(cap). -/
def planI32 (cap : Nat) : Nat := 4 + (cap + 1) + cap + cap

/-- Word offset of the u64 pointer array, rounded to an even word (8-byte alignment). -/
def ptrOff (cap : Nat) : Nat := (planI32 cap + 1) / 2 * 2

/-- Words per plan slot. -/
def planTotal (cap : Nat) : Nat := ptrOff cap + 4 * cap + (cap + 1) + 1

/-- The i32 prefix fits before the pointers, the pointers are 8-byte aligned. -/
theorem plan_prefix_fits (cap : Nat) : planI32 cap ≤ ptrOff cap ∧ ptrOff cap % 2 = 0 := by
  unfold ptrOff planI32; omega

/-- `ptr` (2*cap words), `ptr2` (2*cap words) and `start2` (cap+1 words) tile the tail without overlap. -/
theorem plan_tail_tiles (cap : Nat) :
    ptrOff cap + 2 * cap + 2 * cap = ptrOff cap + 4 * cap ∧
    ptrOff cap + 4 * cap + (cap + 1) ≤ planTotal cap := by
  unfold planTotal; omega

/-- The dst/tok regions end before the pointer array. -/
theorem plan_tok_before_ptr (cap : Nat) : 4 + (cap + 1) + cap + cap ≤ ptrOff cap := by
  unfold ptrOff planI32; omega

/-- Two plan slots fit the allocation of `2 * words + 16` words. -/
theorem two_slots_fit (N grp : Nat) (hg : grp < 2) : grp * N + N ≤ 2 * N := by
  rcases (by omega : grp = 0 ∨ grp = 1) with rfl | rfl <;> omega

-- § flag B only rises

/-- `raise_flag`: store `value` only if it is larger than the current value. -/
def raise (cur value : Nat) : Nat := if cur < value then value else cur

/-- The result dominates both inputs. -/
theorem raise_ge (cur value : Nat) : cur ≤ raise cur value ∧ value ≤ raise cur value := by
  unfold raise; split <;> omega

/-- Raising is order independent: a late host function of an earlier layer cannot lower the flag. -/
theorem raise_comm (cur a b : Nat) : raise (raise cur a) b = raise (raise cur b) a := by
  unfold raise
  split_ifs <;> omega

/-- Raising to a smaller value is a no-op. -/
theorem raise_noop (cur value : Nat) (h : value ≤ cur) : raise cur value = cur := by
  unfold raise; split <;> omega

-- § commit vector

/-- The staged position of cell `t`: `pos0 + t` if kept, `-1` otherwise. -/
def commitEntry (pos0 t nKeep : Int) : Int := if t < nKeep then pos0 + t else -1

/-- Kept cells carry their real (non-negative) position. -/
theorem commit_kept (pos0 t nKeep : Int) (h0 : 0 ≤ pos0) (ht : 0 ≤ t) (h : t < nKeep) :
    commitEntry pos0 t nKeep = pos0 + t ∧ 0 ≤ commitEntry pos0 t nKeep := by
  unfold commitEntry; rw [if_pos h]; omega

/-- Dropped cells are marked `-1`. -/
theorem commit_dropped (pos0 t nKeep : Int) (h : nKeep ≤ t) : commitEntry pos0 t nKeep = -1 := by
  unfold commitEntry; rw [if_neg (by omega)]

/-- The commit counts are in range: `1 ≤ n_keep ≤ last_t` gives `0 ≤ n_keep - 1 < last_t`. -/
theorem commit_index_range (nKeep lastT : Nat) (h1 : 1 ≤ nKeep) (h2 : nKeep ≤ lastT) :
    nKeep - 1 < lastT := by omega

-- § acceptance

/-- Accepted drafts `a` satisfy `a ≤ T-1`, so emitted tokens `a+1 ≤ T`. -/
theorem accepted_bound (T a : Nat) (h : a ≤ T - 1) (hT : 1 ≤ T) : a + 1 ≤ T := by omega

end Strata.Verify
