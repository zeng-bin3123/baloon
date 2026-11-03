/-!
# MODULE: gdn_layer_contracts
Shape and indexing contracts of the gated delta-net layer interface (`gdn.hpp`): channel count, head
pairing `h % h_k`, the (S, H_v, S) state layout, and why the kernel's decay-then-update recurrence agrees
with the reference.

## Source semantics
S = 128, h_k = 16, h_v = 48, C = 2*S*h_k + S*h_v = 10240. State layout is (i, h, j) with j fastest, so the
flat index is `(i*h_v + h)*S + j`, injective and bounded by `S*h_v*S`. The head pairing is MODULO; the
interleaved pairing is a different map. The step contracts the state against k with the decayed state; the
kernel forms `g * dot(state, k)`, which equals `dot(g * state, k)`.

## Form template identifiers
- S -> `S`, h_k -> `hk`, h_v -> `hv`, d_conv -> `dconv`
- state layout -> `stateIdx`; head pairing -> `h % hk`
-/

namespace Strata.Gdn

def S : Nat := 128
def hk : Nat := 16
def hv : Nat := 48
def C : Nat := 2 * S * hk + S * hv

/-- The real artifact has 10240 conv channels. -/
theorem C_eq : C = 10240 := by decide

/-- h_v is a multiple of h_k at the real geometry. -/
theorem hv_multiple : hv % hk = 0 := by decide

-- § head pairing

/-- The modulo pairing always names a valid k/q head. -/
theorem pair_lt (h hk : Nat) (hpos : 0 < hk) : h % hk < hk := Nat.mod_lt h hpos

/-- Modulo and interleaved pairings are different maps: head 1 pairs with k-head 1 under modulo but 0 under interleave. -/
theorem modulo_ne_interleave : ∃ h, h < 48 ∧ h % 16 ≠ h / 3 := by
  exact ⟨1, by decide, by decide⟩

/-- Each k head serves exactly h_v/h_k = 3 value heads under modulo pairing (checked for head 0). -/
theorem pair_fanout_real : ∀ kh, kh < 16 → ((List.range 48).filter (fun h => h % 16 = kh)).length = 3 := by
  decide

-- § state layout

/-- Mixed-radix digits are unique. -/
theorem digit_inj {S a a' j j' : Nat} (hS : 0 < S) (hj : j < S) (hj' : j' < S)
    (h : a * S + j = a' * S + j') : a = a' ∧ j = j' := by
  have key : ∀ x y p q : Nat, p < S → x < y → x * S + p < y * S + q := by
    intro x y p q hp hxy
    have h1 := Nat.mul_le_mul_right S (show x + 1 ≤ y from hxy)
    rw [Nat.add_mul, Nat.one_mul] at h1
    omega
  rcases Nat.lt_trichotomy a a' with h1 | h1 | h1
  · have := key a a' j j' hj h1
    omega
  · subst h1
    exact ⟨rfl, by omega⟩
  · have := key a' a j' j hj' h1
    omega

/-- A digit pair stays below the radix product. -/
theorem digit_lt {a m S j : Nat} (ha : a < m) (hj : j < S) : a * S + j < m * S := by
  have h := Nat.mul_le_mul_right S (show a + 1 ≤ m from ha)
  rw [Nat.add_mul, Nat.one_mul] at h
  omega

def stateIdx (hv S i h j : Nat) : Nat := (i * hv + h) * S + j

/-- State indices stay inside the (S, h_v, S) buffer. -/
theorem stateIdx_lt (hv S i h j : Nat) (hi : i < S) (hh : h < hv) (hj : j < S) :
    stateIdx hv S i h j < S * hv * S := by
  unfold stateIdx
  exact digit_lt (digit_lt hi hh) hj

/-- Distinct (i, h, j) triples address distinct state cells. -/
theorem stateIdx_inj (hv S i h j i' h' j' : Nat) (hS : 0 < S) (hH : 0 < hv)
    (hh : h < hv) (hh' : h' < hv) (hj : j < S) (hj' : j' < S)
    (e : stateIdx hv S i h j = stateIdx hv S i' h' j') : i = i' ∧ h = h' ∧ j = j' := by
  unfold stateIdx at e
  obtain ⟨e1, e2⟩ := digit_inj hS hj hj' e
  obtain ⟨e3, e4⟩ := digit_inj hH hh hh' e1
  exact ⟨e3, e4, e2⟩

/-- For fixed (i, h), consecutive j are consecutive addresses (the coalescing property). -/
theorem stateIdx_succ (hv S i h j : Nat) : stateIdx hv S i h (j + 1) = stateIdx hv S i h j + 1 := by
  unfold stateIdx; omega

-- § kernel row-group split

/-- A step block splits the 128 state rows into 4 groups of 32. -/
theorem rowgroup_cover (r : Nat) (hr : r < 128) : ∃ rg k, rg < 4 ∧ k < 32 ∧ r = rg * 32 + k := by
  exact ⟨r / 32, r % 32, by omega, by omega, by omega⟩

-- § decay-then-update

def dot : List Int → List Int → Int
  | a :: as, b :: bs => a * b + dot as bs
  | _, _ => 0

/-- Scaling the state before contracting equals scaling after, so the kernel's `g * dot(state, k)` is the
reference's contraction of the decayed state. -/
theorem dot_smul (g : Int) (as bs : List Int) : dot (as.map (fun a => g * a)) bs = g * dot as bs := by
  induction as generalizing bs with
  | nil => simp [dot]
  | cons a as ih =>
    cases bs with
    | nil => simp [dot]
    | cons b bs => simp [dot, ih, Int.mul_add, Int.mul_assoc]

/-- The delta term built from the undecayed contraction equals the one built from the decayed state. -/
theorem delta_decay_equiv (g beta v : Int) (st k : List Int) :
    (v - g * dot st k) * beta = (v - dot (st.map (fun a => g * a)) k) * beta := by
  rw [dot_smul]

end Strata.Gdn
