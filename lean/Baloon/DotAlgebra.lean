-- Baloon.DotAlgebra: dot product algebra used by the Q2_0 / dp4a kernels:
-- chunked dot products, the bias identity, and 2-bit code expansion.
-- Independent implementation proving the same mathematical facts as Ref.DotAlgebra.

namespace Baloon.DotAlgebra

def dot : List Int → List Int → Int
  | a :: as, b :: bs => a * b + dot as bs
  | _, _ => 0

theorem dot_append (as as' bs bs' : List Int) (h : as.length = bs.length) :
    dot (as ++ as') (bs ++ bs') = dot as bs + dot as' bs' := by
  induction as generalizing bs with
  | nil => cases bs with | nil => simp [dot] | cons _ _ => simp at h
  | cons a as ih =>
    cases bs with
    | nil => simp at h
    | cons b bs =>
      have h' : as.length = bs.length := by simpa using h
      simp only [List.cons_append, dot]; rw [ih bs h']; omega

theorem dot_bias (cs xs : List Int) (h : cs.length = xs.length) :
    dot (cs.map (· - 1)) xs = dot cs xs - xs.sum := by
  induction cs generalizing xs with
  | nil => cases xs with | nil => simp [dot] | cons _ _ => simp at h
  | cons c cs ih =>
    cases xs with
    | nil => simp at h
    | cons x xs =>
      have h' : cs.length = xs.length := by simpa using h
      simp only [List.map_cons, dot, List.sum_cons]
      rw [ih xs h', Int.sub_mul, Int.one_mul]; omega

theorem pack4 (a b c d : Nat) (ha : a < 4) (hb : b < 4) (hc : c < 4) (hd : d < 4) :
    (a + 256 * b + 65536 * c + 16777216 * d) % 256 = a ∧
    ((a + 256 * b + 65536 * c + 16777216 * d) / 256) % 256 = b ∧
    ((a + 256 * b + 65536 * c + 16777216 * d) / 65536) % 256 = c ∧
    (a + 256 * b + 65536 * c + 16777216 * d) / 16777216 = d ∧
    a + 256 * b + 65536 * c + 16777216 * d < 4294967296 := by omega

end Baloon.DotAlgebra
