-- Baloon.PrefillOrchestration: independent implementation; proves the same
-- invariants as Ref.Prefill with different definitions and proof strategies.

namespace Baloon.PrefillOrchestration

-- § chunk partition

/-- Chunk lengths for n tokens with size T (fuel-bounded). -/
def chunksAux (fuel T n : Nat) : List Nat :=
  match fuel with
  | 0 => []
  | fuel + 1 =>
    if n = 0 then []
    else Nat.min T n :: chunksAux fuel T (n - Nat.min T n)

/-- Main entry: fuel = n (each step reduces n by at least 1). -/
def chunks (T n : Nat) : List Nat := let fuel := n; chunksAux fuel T n

theorem sum_chunksAux (T : Nat) (hT : 0 < T) :
    ∀ fuel n, n ≤ fuel → (chunksAux fuel T n).sum = n := by
  intro fuel
  induction fuel with
  | zero => intro n h; simp [chunksAux, Nat.le_zero.mp h]
  | succ f ih =>
    intro n h
    by_cases hn : n = 0
    · simp [chunksAux, hn]
    · simp only [chunksAux, hn, ite_false]
      have hm : 0 < Nat.min T n := by
        unfold Nat.min; rw [min_def]; split_ifs with h; exact hT; omega
      have hle : Nat.min T n ≤ n := min_le_right T n
      rw [List.sum_cons, ih _ (by omega)]; omega

theorem sum_chunks (T n : Nat) (hT : 0 < T) : (chunks T n).sum = n :=
  sum_chunksAux T hT n n (le_refl n)

theorem chunksAux_bounds (T : Nat) (hT : 0 < T) :
    ∀ fuel n, ∀ x ∈ chunksAux fuel T n, 0 < x ∧ x ≤ T := by
  intro fuel
  induction fuel with
  | zero => intro n x hx; simp [chunksAux] at hx
  | succ f ih =>
    intro n x hx
    by_cases hn : n = 0
    · simp [chunksAux, hn] at hx
    · simp only [chunksAux, hn, ite_false, List.mem_cons] at hx
      rcases hx with rfl | hx'
      · have hm : 0 < Nat.min T n := by
          unfold Nat.min; rw [min_def]; split_ifs with h; exact hT; omega
        exact ⟨hm, min_le_left T n⟩
      · exact ih _ x hx'

-- § bump allocator

/-- Round b to 256-byte alignment with 256 bytes of slack: (b+511)/256*256. -/
def roundBytes (b : Nat) : Nat := (b + 511) / 256 * 256

theorem roundBytes_aligned (b : Nat) : roundBytes b % 256 = 0 := by
  unfold roundBytes; omega

theorem roundBytes_ge (b : Nat) : b + 256 ≤ roundBytes b := by
  unfold roundBytes; omega

theorem roundBytes_lt (b : Nat) : roundBytes b < b + 512 := by
  unfold roundBytes; omega

/-- Sequential bump allocation: accumulate used; fail if over cap. -/
def allocAll (cap : Nat) : List Nat → Nat → Option Nat
  | [], used => some used
  | b :: bs, used => if b + used ≤ cap then allocAll cap bs (b + used) else none

theorem allocAll_le (cap : Nat) : ∀ (bs : List Nat) (used u : Nat),
    used ≤ cap → allocAll cap bs used = some u → u ≤ cap := by
  intro bs
  induction bs with
  | nil => intro used u _ h; simp [allocAll] at h; omega
  | cons b bs ih =>
    intro used u hu h
    simp only [allocAll] at h
    by_cases hb : b + used ≤ cap
    · rw [if_pos hb] at h; exact ih _ u (by omega) h
    · rw [if_neg hb] at h; exact absurd h (by simp)

theorem alloc_single_iff (cap used b : Nat) :
    (allocAll cap [b] used).isSome = true ↔ used + b ≤ cap := by
  constructor
  · intro h
    simp only [allocAll] at h
    split_ifs at h with hb
    · omega
    · simp at h
  · intro h
    simp [allocAll, show b + used ≤ cap from by omega]

-- § staging rings (8-slot stage, 2-slot DQ)

theorem stage_slots_distinct (i k : Nat) (hlt : i < k) (hwin : k < i + 8) : i % 8 ≠ k % 8 := by
  intro h; omega

theorem lookahead_lt_stage : (7 : Nat) < 8 := by norm_num

theorem dq_slots_distinct (j k : Nat) (hlt : j < k) (hwin : k < j + 2) : j % 2 ≠ k % 2 := by
  intro h; omega

-- § expert grouping (counting sort by expert id)

theorem group_slot_in_block (off cnt c : Nat) (h : c < cnt) : off + c < off + cnt :=
  Nat.add_lt_add_left h off

theorem group_blocks_disjoint (off cnt c off' : Nat) (h : c < cnt) (h' : off' = off + cnt) :
    off + c < off' := by subst h'; omega

-- § GDN state layout

/-- Floats per layer: recurrent state (48*128*128) + conv history (3*10240). -/
def gdnFloats : Nat := 48 * 128 * 128 + 3 * 10240

theorem gdnFloats_value : gdnFloats = 817152 := by unfold gdnFloats; decide

theorem gdn_conv_offset (idx : Nat) :
    idx * gdnFloats + 128 * 48 * 128 < (idx + 1) * gdnFloats := by
  unfold gdnFloats; omega

end Baloon.PrefillOrchestration
