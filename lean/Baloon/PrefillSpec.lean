-- Baloon.PrefillSpec: batched-prefill chunk partition (sum coverage, bounds),
-- bump allocator rounding and position accounting.
-- Independent implementation proving the same mathematical facts as Ref.Prefill.

namespace Baloon.PrefillSpec

def chunksAux : Nat → Nat → Nat → List Nat
  | 0, _, _ => ([])
  | fuel + 1, T, n => (if n = 0 then [] else min T n :: chunksAux fuel T (n - min T n))

def chunks (T n : Nat) : List Nat := (chunksAux n T n)

theorem sum_chunksAux (T : Nat) (hT : 0 < T) :
    ∀ fuel n, n ≤ fuel → (chunksAux fuel T n).sum = n := by
  intro fuel
  induction fuel with
  | zero => intro n h; have hn : n = 0 := Nat.le_zero.mp h; subst hn; simp [chunksAux]
  | succ f ih =>
    intro n h
    by_cases hn : n = 0
    · subst hn; simp [chunksAux]
    · have hm : 0 < min T n := by omega
      have h2 := ih (n - min T n) (by omega)
      rw [chunksAux, if_neg hn, List.sum_cons, h2]; omega

theorem sum_chunks (T n : Nat) (hT : 0 < T) : (chunks T n).sum = n := by exact sum_chunksAux T hT n n (Nat.le_refl n)

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

/-- Bytes charged for a request of b bytes: (b + 256 + 255) & ~255. -/
def roundBytes (b : Nat) : Nat := ((b + 256 + 255) / 256 * 256)

/-- The charge is at least b + 1 (the header byte is included). -/
theorem roundBytes_gt (b : Nat) : b < roundBytes b := by unfold roundBytes; omega

/-- The charge is a multiple of 256. -/
theorem roundBytes_mod (b : Nat) : roundBytes b % 256 = 0 := by unfold roundBytes; omega

/-- Position of the first generated token: n_prompt - 1 (the last conditioning step). -/
def firstGenPos (n : Nat) : Nat := (n - 1)

/-- After the prompt run the generated tokens start at position n - 1. -/
theorem firstGenPos_lt (n : Nat) (hn : 1 ≤ n) : firstGenPos n < n := by
  unfold firstGenPos; omega

end Baloon.PrefillSpec
