-- Baloon.PrefillOrchestration: batched prefill orchestration invariants.
-- Sources: src/prefill/prefill.cpp.
-- Chunk partition, bump allocator, staging/dequant ring slot distinctness,
-- expert grouping cursors, and GDN state layout.
-- Independent implementation proving the same mathematical facts as Ref.Prefill.

namespace Baloon.PrefillOrchestration

-- § chunk partition (the for c0 = 0; c0 < n; c0 += T loop) ——————————————

def chunksAux : Nat → Nat → Nat → List Nat
  | 0, _, _ => []
  | gas + 1, T, rem => if rem = 0 then [] else min T rem :: chunksAux gas T (rem - min T rem)

def chunks (T : Nat) (n : Nat) : List Nat := chunksAux n T n

/-- Chunk lengths sum to the number of positions processed. -/
theorem sum_chunksAux (T : Nat) (hT : 0 < T) :
    ∀ fuel n, n ≤ fuel → (chunksAux fuel T n).sum = n := by
  intro fuel
  induction fuel with
  | zero =>
    intro n h
    have hn : n = 0 := Nat.le_zero.mp h
    subst hn; simp [chunksAux]
  | succ f ih =>
    intro n h
    by_cases hn : n = 0
    · subst hn; simp [chunksAux]
    · have hm : 0 < min T n := by omega
      have h2 := ih (n - min T n) (by omega)
      rw [chunksAux, if_neg hn, List.sum_cons, h2]; omega

/-- The full chunk list covers the prompt exactly. -/
theorem sum_chunks (T n : Nat) (hT : 0 < T) : (chunks T n).sum = n :=
  sum_chunksAux T hT n n (Nat.le_refl _)

/-- Every chunk is non-empty and at most T tokens. -/
theorem chunksAux_bounds (T : Nat) (hT : 0 < T) :
    ∀ fuel n, ∀ x ∈ chunksAux fuel T n, 0 < x ∧ x ≤ T := by
  intro fuel
  induction fuel with
  | zero => intro n x hx; simp [chunksAux] at hx
  | succ f ih =>
    intro n x hx
    by_cases hn : n = 0
    · subst hn; simp [chunksAux] at hx
    · rw [chunksAux, if_neg hn] at hx
      rcases List.mem_cons.mp hx with h | hx'
      · rw [h]; omega
      · exact ih _ x hx'

-- § bump allocator ————————————————————————————————————————————————————————

/-- Bytes charged for a request of b bytes: (b + 511) / 256 * 256 (256-aligned, 256-byte slack). -/
def roundBytes (b : Nat) : Nat := (b + 511) / 256 * 256

/-- Every allocation is 256-byte aligned. -/
theorem roundBytes_aligned (b : Nat) : roundBytes b % 256 = 0 := by
  unfold roundBytes; omega

/-- Every allocation leaves at least the 256 byte slack. -/
theorem roundBytes_ge (b : Nat) : b + 256 ≤ roundBytes b := by
  unfold roundBytes; omega

/-- Rounding wastes less than one alignment unit. -/
theorem roundBytes_lt (b : Nat) : roundBytes b < b + 256 + 256 := by
  unfold roundBytes; omega

/-- Sequential bump allocation of sizes into a region of capacity cap. -/
def allocAll (cap : Nat) : List Nat → Nat → Option Nat
  | [], u0 => some u0
  | b :: bs, u0 => if u0 + b ≤ cap then allocAll cap bs (u0 + b) else none

/-- A successful allocation sequence never exceeds the region. -/
theorem allocAll_le (cap : Nat) : ∀ (bs : List Nat) (used u : Nat),
    used ≤ cap → allocAll cap bs used = some u → u ≤ cap := by
  intro bs
  induction bs with
  | nil =>
    intro used u hu h; simp [allocAll] at h; omega
  | cons b bs ih =>
    intro used u hu h
    by_cases hb : used + b ≤ cap
    · rw [allocAll, if_pos hb] at h; exact ih _ u hb h
    · rw [allocAll, if_neg hb] at h; simp at h

/-- A single request fits exactly when the region has room. -/
theorem alloc_single_iff (cap used b : Nat) :
    (allocAll cap [b] used).isSome = true ↔ used + b ≤ cap := by
  by_cases hb : used + b ≤ cap <;> simp [allocAll, hb]

-- § staging rings ————————————————————————————————————————————————————————

/-- Two staged blobs within one lookahead window (< 8 apart) never share a staging slot. -/
theorem stage_slots_distinct (i k : Nat) (hlt : i < k) (hwin : k < i + 8) : i % 8 ≠ k % 8 := by
  { omega }

/-- Lookahead 7 is strictly below the ring size 8, so the window fits the ring. -/
theorem lookahead_lt_stage : (7 : Nat) < 8 := by { omega }

/-- Two experts in flight on the 2-slot dequantised ring never share a slot. -/
theorem dq_slots_distinct (j k : Nat) (hlt : j < k) (hwin : k < j + 2) : j % 2 ≠ k % 2 := by
  { omega }

-- § expert grouping (counting sort by expert id) ————————————————————————

/-- A token-expert pair's destination row lies strictly inside its expert's block. -/
theorem group_slot_in_block (off cnt c : Nat) (h : c < cnt) : off + c < off + cnt := by
  { omega }

/-- The next block starts where this block ends (off[e+1] = off[e] + cnt[e]). -/
theorem group_blocks_disjoint (off cnt c off' : Nat) (h : c < cnt) (h' : off' = off + cnt) :
    off + c < off' := by subst h'; omega

-- § GDN state layout ———————————————————————————————————————————————————

/-- Floats of GDN state per layer: 48*128*128 recurrent state plus 3*10240 conv history. -/
def gdnFloats : Nat := 48 * 128 * 128 + 3 * 10240

/-- The per-layer GDN state is 817152 floats. -/
theorem gdnFloats_value : gdnFloats = 817152 := by unfold gdnFloats; omega

/-- The conv history starts after the recurrent state inside one layer's block. -/
theorem gdn_conv_offset (idx : Nat) :
    idx * gdnFloats + 128 * 48 * 128 < (idx + 1) * gdnFloats := by
  unfold gdnFloats; omega

end Baloon.PrefillOrchestration

