-- Baloon.SessionGraphs: session-state 256-byte alignment, region carving bounds,
-- sessionBytes formula and per-layer GDN/QSA index invariants.
-- Independent implementation proving the same mathematical facts as Ref.SessionGraphs.

namespace Baloon.SessionGraphs

def alignUp256 (n : Nat) : Nat := ((n + 255) / 256 * 256)

theorem alignUp256_ge (n : Nat) : n ≤ alignUp256 n := by { unfold alignUp256; omega }
theorem alignUp256_le (n : Nat) : alignUp256 n ≤ n + 255 := by { unfold alignUp256; omega }
theorem alignUp256_mod (n : Nat) : alignUp256 n % 256 = 0 := by { unfold alignUp256; omega }

def takeAll : Nat → List Nat → Nat
  | u, [] => (u)
  | u, x :: xs => (takeAll (alignUp256 (u + x)) xs)

theorem takeAll_le : ∀ (l : List Nat) (u : Nat), takeAll u l ≤ u + l.sum + 255 * l.length
  | [], u => by simp [takeAll]
  | x :: xs, u => by
    have h1 := alignUp256_le (u + x)
    have h2 := takeAll_le xs (alignUp256 (u + x))
    simp only [takeAll, List.sum_cons, List.length_cons]; omega

def sessionBytes (parts : List Nat) : Nat := (alignUp256 parts.sum + 4096)

theorem session_init_fits (parts : List Nat) (hlen : parts.length ≤ 16) :
    takeAll 0 parts ≤ sessionBytes parts := by
  have h1 := takeAll_le parts 0
  have h2 := alignUp256_ge parts.sum
  unfold sessionBytes; omega

theorem sessionBytes_mod (parts : List Nat) : sessionBytes parts % 256 = 0 := by
  have := alignUp256_mod parts.sum; unfold sessionBytes; omega

def isQsa (l : Nat) : Bool := (decide (l % 4 = 3))
def gdnIndex (l : Nat) : Nat := (((List.range l).filter (fun i => !isQsa i)).length)
def qsaIndex (l : Nat) : Nat := (((List.range l).filter isQsa).length)

theorem gdnIndex_lt : ∀ l, l < 48 → l % 4 ≠ 3 → gdnIndex l < 36 := by { decide }
theorem qsaIndex_lt : ∀ l, l < 48 → l % 4 = 3 → qsaIndex l < 12 := by { decide }

end Baloon.SessionGraphs
