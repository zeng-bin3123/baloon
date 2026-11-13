-- PagedKvLayout.lean
-- Formal model of src/kernels/cuda/qsa.cu paged KV pool index arithmetic.
namespace Baloon.PagedKv

theorem mul_add_lt {a b c n : Nat} (hb : b < c) (ha : a < n) : a * c + b < n * c := by
  have h1 : (a + 1) * c ≤ n * c := Nat.mul_le_mul_right c (Nat.succ_le_of_lt ha)
  rw [Nat.succ_mul] at h1; omega

theorem decomp_unique {a b a' b' c : Nat} (hb : b < c) (hb' : b' < c)
    (h : a * c + b = a' * c + b') : a = a' ∧ b = b' := by
  rcases Nat.lt_trichotomy a a' with hlt | heq | hgt
  · have h1 : (a + 1) * c ≤ a' * c := Nat.mul_le_mul_right c (Nat.succ_le_of_lt hlt)
    rw [Nat.succ_mul] at h1; omega
  · subst heq; omega
  · have h1 : (a' + 1) * c ≤ a * c := Nat.mul_le_mul_right c (Nat.succ_le_of_lt hgt)
    rw [Nat.succ_mul] at h1; omega

def poolRow (table : Nat → Nat) (K ps h pos : Nat) : Nat :=
  (table (pos / ps) * K + h) * ps + pos % ps

theorem poolRow_lt (table : Nat → Nat) (K ps h pos nPages : Nat) (hps : 0 < ps) (hh : h < K)
    (hpage : table (pos / ps) < nPages) : poolRow table K ps h pos < nPages * K * ps := by
  have h1 : table (pos / ps) * K + h < nPages * K := mul_add_lt hh hpage
  exact mul_add_lt (Nat.mod_lt pos hps) h1

theorem row_injective {p p' h h' s s' K ps : Nat} (hh : h < K) (hh' : h' < K) (hs : s < ps)
    (hs' : s' < ps) (heq : (p * K + h) * ps + s = (p' * K + h') * ps + s') :
    p = p' ∧ h = h' ∧ s = s' := by
  obtain ⟨h1, h2⟩ := decomp_unique hs hs' heq
  obtain ⟨h3, h4⟩ := decomp_unique hh hh' h1
  exact ⟨h3, h4, h2⟩

def gatherId (K per i : Nat) : Nat := i / (K * per)

def scratchIdx (K per i : Nat) : Nat :=
  (i / (K * per) * K + i % (K * per) / per) * per + i % (K * per) % per

theorem scratchIdx_eq (K per i : Nat) : scratchIdx K per i = i := by
  unfold scratchIdx
  have e1 := Nat.div_add_mod i (K * per)
  have e2 := Nat.div_add_mod (i % (K * per)) per
  rw [Nat.add_mul, Nat.mul_assoc, Nat.mul_comm (i / (K * per)) (K * per),
    Nat.mul_comm (i % (K * per) / per) per]
  omega

theorem gatherId_lt (K per i nIds : Nat) (hK : 0 < K * per) (hi : i < nIds * (K * per)) :
    gatherId K per i < nIds := by
  unfold gatherId; exact (Nat.div_lt_iff_lt_mul hK).mpr hi

theorem gather_head_lt (K per i : Nat) (hK : 0 < K * per) : i % (K * per) / per < K := by
  have hm := Nat.mod_lt i hK
  have hper : 0 < per := Nat.pos_of_mul_pos_left hK
  exact (Nat.div_lt_iff_lt_mul hper).mpr hm

end Baloon.PagedKv
