-- Baloon.ExpertSource: expert blob addressing (flat index with alias hazard), hit/miss
-- job decomposition and expert file size constants.
-- Independent implementation proving the same mathematical facts as Ref.ExpertSource.

namespace Baloon.ExpertSource

def flatIdx (layer expert nExpert : Nat) : Nat := layer * nExpert + expert

theorem unchecked_flat_alias (E : Nat) : flatIdx 0 E E = flatIdx 1 0 E := by
  unfold flatIdx; omega

theorem alias_512 : flatIdx 0 512 512 = flatIdx 1 0 512 := by decide

theorem flatIdx_recover (l e E : Nat) (hE : 0 < E) (he : e < E) :
    flatIdx l e E / E = l ∧ flatIdx l e E % E = e := by
  have hcomm : flatIdx l e E = e + E * l := by
    unfold flatIdx; rw [Nat.mul_comm, Nat.add_comm]
  rw [hcomm]
  constructor
  · rw [Nat.add_mul_div_left e l hE, Nat.div_eq_of_lt he, Nat.zero_add]
  · rw [Nat.add_mul_mod_self_left, Nat.mod_eq_of_lt he]

theorem flatIdx_inj (l e l' e' E : Nat) (hE : 0 < E) (he : e < E) (he' : e' < E)
    (h : flatIdx l e E = flatIdx l' e' E) : l = l' ∧ e = e' := by
  have h1 := flatIdx_recover l e E hE he
  have h2 := flatIdx_recover l' e' E hE he'
  rw [h] at h1
  exact ⟨h1.1.symm.trans h2.1, h1.2.symm.trans h2.2⟩

theorem flatIdx_lt (l e L E : Nat) (hl : l < L) (he : e < E) : flatIdx l e E < L * E := by
  have h1 : (l + 1) * E ≤ L * E := Nat.mul_le_mul_right E (by omega)
  simp only [Nat.add_mul, Nat.one_mul] at h1
  unfold flatIdx; omega

theorem experts_file_size : 48 * 512 * 1382400 = 33973862400 := by decide
theorem blob_count : 48 * 512 = 24576 := by decide

def nHit : List Bool → Nat
  | [] => 0
  | b :: bs => (if b then 1 else 0) + nHit bs

def nMiss : List Bool → Nat
  | [] => 0
  | b :: bs => (if b then 0 else 1) + nMiss bs

theorem jobs_plus_hits : ∀ l : List Bool, nHit l + nMiss l = l.length
  | [] => by simp [nHit, nMiss]
  | b :: bs => by
    have ih := jobs_plus_hits bs
    cases b <;> simp [nHit, nMiss, List.length_cons] <;> omega

theorem nHit_le (l : List Bool) : nHit l ≤ l.length := by
  have := jobs_plus_hits l; omega

end Baloon.ExpertSource
