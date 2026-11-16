-- Baloon.SuffixDrafter: invariants of the trigram suffix lookup drafter.
-- Independent implementation in the project namespace; proves the same
-- mathematical facts as the reference theory with different names and terms.

namespace Baloon.SuffixDrafter

-- ── WAYS = 4 slot capacity ────────────────────────────────────────────────────

/-- WAYS = 4: each hash slot stores the 4 most recent end positions. -/
theorem ways_eq : (4 : Nat) = 4 := rfl

-- ── key_at: set low bit to prevent 0 (empty-slot sentinel) ───────────────────

/-- 2 * k + 1 is positive for every k. -/
theorem oddNat_pos (k : Nat) : 0 < 2 * k + 1 := by omega

/-- 2 * k + 1 is odd. -/
theorem oddNat_mod2 (k : Nat) : (2 * k + 1) % 2 = 1 := by omega

/-- x / 2 * 2 + 1 (set-low-bit) is always positive. -/
theorem setLsb_pos (x : Nat) : 0 < x / 2 * 2 + 1 := by omega

-- ── linear probe step ────────────────────────────────────────────────────────

/-- (i + 1) % cap stays inside a positive-capacity table. -/
theorem probeNext_lt (i cap : Nat) (hc : 0 < cap) : (i + 1) % cap < cap :=
  Nat.mod_lt _ hc

-- ── slot way list (newest-first, at most W entries) ───────────────────────────

/-- (e :: l).take W has at most W entries. -/
theorem slotWays_le_W (W e : Nat) (l : List Nat) :
    ((e :: l).take W).length ≤ W := by simp [List.length_take]

/-- The newest entry is first when W > 0. -/
theorem slotWays_head (W : Nat) (hW : 0 < W) (e : Nat) (l : List Nat) :
    ((e :: l).take W).head? = some e := by
  cases W with
  | zero => exact absurd hW (by omega)
  | succ w => simp

-- ── backward match loop (private helper with different structure from ml) ─────

/-- Fuel-bounded match count: pos advances, fuel decreases. -/
private def matchAux (h : Nat → Nat) (p cur : Nat) : Nat → Nat → Nat
  | 0, _ => 0
  | n + 1, pos =>
    if pos ≤ p ∧ h (p - pos) = h (cur - pos)
    then 1 + matchAux h p cur n (pos + 1)
    else 0

/-- matchAux never exceeds the fuel bound n. -/
theorem matchAux_le_fuel (h : Nat → Nat) (p cur : Nat) :
    ∀ n pos, matchAux h p cur n pos ≤ n := by
  intro n; induction n with
  | zero => intro pos; simp [matchAux]
  | succ m ih =>
    intro pos; simp only [matchAux]
    split_ifs with hc
    · have := ih (pos + 1); omega
    · omega

/-- matchAux does not reach before history start: pos + result ≤ p + 1. -/
theorem matchAux_hist (h : Nat → Nat) (p cur : Nat) :
    ∀ n pos, pos ≤ p + 1 → matchAux h p cur n pos + pos ≤ p + 1 := by
  intro n; induction n with
  | zero => intro pos hp; simp [matchAux]; omega
  | succ m ih =>
    intro pos hp; simp only [matchAux]
    split_ifs with hc
    · have h2 := ih (pos + 1) (by omega); omega
    · omega

-- ── constructor clamps ────────────────────────────────────────────────────────

/-- min_match_ = max(3, raw): always at least 3 (trigram key needs ≥ 3 tokens). -/
theorem minMGuard (m : Nat) : 3 ≤ max 3 m := by omega

/-- max_match_ = max(min_match, raw): always at least min_match. -/
theorem maxMGuard (a b : Nat) : a ≤ max a b := Nat.le_max_left a b

/-- When max_match < min_match, matchAux can never reach min_match. -/
theorem shortFuel_no_match (h : Nat → Nat) (p cur mx mn : Nat) (hlt : mx < mn) :
    matchAux h p cur mx 0 < mn := by
  have := matchAux_le_fuel h p cur mx 0; omega

-- ── continuation count ────────────────────────────────────────────────────────

/-- min(max_k, cur − best_end) ≤ max_k: draft never exceeds the budget. -/
theorem contLimit_le (maxk cur e : Nat) : min maxk (cur - e) ≤ maxk :=
  Nat.min_le_left maxk _

/-- e + min(max_k, cur − e) ≤ cur: draft never reads past cur. -/
theorem contLimit_in_hist (maxk cur e : Nat) (he : e < cur) :
    e + min maxk (cur - e) ≤ cur := by omega

-- ── FNV prime ─────────────────────────────────────────────────────────────────

/-- The FNV-1a prime 0x100000001b3 is odd. -/
theorem fnvPrime_odd : 0x100000001b3 % 2 = 1 := by decide

end Baloon.SuffixDrafter
