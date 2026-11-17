-- Baloon.SessionGraphs: session-state 256-byte alignment, region carving bounds,
-- sessionBytes formula, per-layer GDN/QSA indices, stage-prefix timing differences,
-- ring numbering and spin boundary.
-- Independent implementation proving the same mathematical facts as Ref.SessionGraphs.

namespace Baloon.SessionGraphs

-- § alignment to 256

def alignUp256 (n : Nat) : Nat := (n + 255) / 256 * 256

theorem alignUp256_ge (n : Nat) : n ≤ alignUp256 n := by unfold alignUp256; omega
theorem alignUp256_le (n : Nat) : alignUp256 n ≤ n + 255 := by unfold alignUp256; omega
theorem alignUp256_mod (n : Nat) : alignUp256 n % 256 = 0 := by unfold alignUp256; omega

-- § carving fits in reported size

def takeAll : Nat → List Nat → Nat
  | u, [] => u
  | u, x :: xs => takeAll (alignUp256 (u + x)) xs

theorem takeAll_le : ∀ (l : List Nat) (u : Nat), takeAll u l ≤ u + l.sum + 255 * l.length
  | [], u => by simp [takeAll]
  | x :: xs, u => by
    have h1 := alignUp256_le (u + x)
    have h2 := takeAll_le xs (alignUp256 (u + x))
    simp only [takeAll, List.sum_cons, List.length_cons]; omega

def sessionBytes (parts : List Nat) : Nat := alignUp256 parts.sum + 4096

theorem session_init_fits (parts : List Nat) (hlen : parts.length ≤ 16) :
    takeAll 0 parts ≤ sessionBytes parts := by
  have h1 := takeAll_le parts 0
  have h2 := alignUp256_ge parts.sum
  unfold sessionBytes; omega

theorem sessionBytes_mod (parts : List Nat) : sessionBytes parts % 256 = 0 := by
  have := alignUp256_mod parts.sum; unfold sessionBytes; omega

-- § per-layer state indices

def isQsa (l : Nat) : Bool := decide (l % 4 = 3)
def gdnIndex (l : Nat) : Nat := ((List.range l).filter (fun i => !isQsa i)).length
def qsaIndex (l : Nat) : Nat := ((List.range l).filter isQsa).length

theorem gdnIndex_lt : ∀ l, l < 48 → l % 4 ≠ 3 → gdnIndex l < 36 := by decide
theorem qsaIndex_lt : ∀ l, l < 48 → l % 4 = 3 → qsaIndex l < 12 := by decide

theorem gdnIndex_succ : ∀ l, l < 47 → l % 4 ≠ 3 → gdnIndex (l + 1) = gdnIndex l + 1 := by decide
theorem qsaIndex_succ : ∀ l, l < 47 → l % 4 = 3 → qsaIndex (l + 1) = qsaIndex l + 1 := by decide

theorem gdnIndex_inj : ∀ a, a < 48 → ∀ b, b < 48 → a % 4 ≠ 3 → b % 4 ≠ 3 →
    gdnIndex a = gdnIndex b → a = b := by decide

-- § stage timing by prefix differences

def stageDiffs (t1 t2 t3 t4 t5 : Nat) : List Nat := [t1, t2 - t1, t3 - t2, t4 - t3, t5 - t4]

theorem stageDiffs_sum (t1 t2 t3 t4 t5 : Nat) (h1 : t1 ≤ t2) (h2 : t2 ≤ t3) (h3 : t3 ≤ t4)
    (h4 : t4 ≤ t5) : (stageDiffs t1 t2 t3 t4 t5).sum = t5 := by
  simp only [stageDiffs, List.sum_cons, List.sum_nil]; omega

theorem mixer_is_prefix3 (t1 t2 t3 : Nat) (h1 : t1 ≤ t2) (h2 : t2 ≤ t3) :
    t1 + (t2 - t1) + (t3 - t2) = t3 := by omega

-- § ring numbering and dump layout

def wantRing (l : Nat) : Nat := l + 1

theorem wantRing_strict (a b : Nat) (h : a < b) : wantRing a < wantRing b := by
  unfold wantRing; omega

theorem dump_slot_in_bounds (l L n : Nat) (h : l < L) : (l + 1) * n + n ≤ (L + 1) * n := by
  have h2 : (l + 1) * n ≤ L * n := Nat.mul_le_mul_right n (by omega)
  simp only [Nat.add_mul, Nat.one_mul] at h2 ⊢; omega

-- § per-token handoff bookkeeping

theorem parts_bytes_pos (k n : Nat) (hk : 0 < k) (hn : 0 < n) : 0 < k * n * 4 := by
  have : 0 < k * n := Nat.mul_pos hk hn; omega

def spinBoundary (spins : Nat) : Bool := decide ((spins + 1) % 1024 = 0)

theorem spinBoundary_period (s : Nat) : spinBoundary (s + 1024) = spinBoundary s := by
  unfold spinBoundary; simp only [decide_eq_decide]; omega

end Baloon.SessionGraphs
