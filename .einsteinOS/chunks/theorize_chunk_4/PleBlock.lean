/-!
# MODULE: ple_block
GPU half of the PLE block: grouped RMS norms, signed-sqrt sigmoid gate, depthwise causal dilated conv + SiLU,
three-way add, and history shift. Caller-owned scratch workspace.

## Source semantics
Conv tap k reads row `nhist - (kern-1-k)*dil` of [history | new]; history is row-fastest (`row + nhist*c`).
Scratch regions are 16-byte aligned. Output buffers must not overlap scratch.

## Form template identifiers
ple_block -> gated_conv_block; hist -> conv_state; normalized -> conv_input
-/
namespace Strata.Ple

/-- Round up to a multiple of 16. -/
def round16 (n : Nat) : Nat := (n + 15) / 16 * 16

/-- Rounded size is not smaller. -/
theorem round16_ge (n : Nat) : n ≤ round16 n := by unfold round16; omega

/-- Rounded size is 16-byte aligned. -/
theorem round16_mod (n : Nat) : round16 n % 16 = 0 := by unfold round16; omega

/-- Row read by tap `k`. -/
def convRow (nhist kern dil k : Nat) : Int := (nhist : Int) - ((kern : Int) - 1 - k) * dil

/-- Real geometry (kern 4, dil 3, hist 9): taps read rows 0, 3, 6, 9. -/
theorem conv_rows_real : [0, 1, 2, 3].map (fun k => convRow 9 4 3 k) = [0, 3, 6, 9] := by decide

/-- Row-fastest flat index of the conv state. -/
def idx (nhist row c : Nat) : Nat := row + nhist * c

/-- Row-fastest indexing recovers the channel. -/
theorem idx_div (nhist row c : Nat) (hp : 0 < nhist) (hr : row < nhist) : idx nhist row c / nhist = c := by
  unfold idx
  rw [Nat.add_mul_div_left _ _ hp, Nat.div_eq_of_lt hr]
  simp

/-- Row-fastest indexing recovers the row. -/
theorem idx_mod (nhist row c : Nat) (hr : row < nhist) : idx nhist row c % nhist = row := by
  unfold idx
  rw [Nat.add_mul_mod_self_left]
  exact Nat.mod_eq_of_lt hr

/-- One column of the history shifts left and appends the new entry. -/
def advance {α : Type} (col : List α) (x : α) : List α := col.tail ++ [x]

/-- History advance preserves column length for a non-empty column. -/
theorem advance_length {α : Type} (a : α) (l : List α) (x : α) :
    (advance (a :: l) x).length = (a :: l).length := by
  simp [advance]

/-- Pointer-range overlap test of `overlap()` in ple.cu. -/
def ovl (aa an bb bn : Nat) : Bool :=
  if aa < bb then decide (bb - aa < an) else decide (aa - bb < bn)

/-- For non-empty spans the test is exactly interval overlap. -/
theorem ovl_iff (aa an bb bn : Nat) (ha : 0 < an) (hb : 0 < bn) :
    ovl aa an bb bn = true ↔ (aa < bb + bn ∧ bb < aa + an) := by
  unfold ovl
  split <;> simp <;> omega

/-- Sign function used by the gate (zero maps to zero). -/
def sgn (s : Int) : Int := if s > 0 then 1 else if s < 0 then -1 else 0

/-- Sign is odd, which keeps the gate symmetric about 0.5. -/
theorem sgn_neg (s : Int) : sgn (-s) = - sgn s := by
  unfold sgn
  rcases lt_trichotomy s 0 with h | h | h
  · have h1 : ¬ (s > 0) := by omega
    have h2 : -s > 0 := by omega
    simp only [if_pos h2, if_neg h1, if_pos h] <;> omega
  · subst h
    simp
  · have h2 : ¬ (-s > 0) := by omega
    have h3 : -s < 0 := by omega
    have h1 : s > 0 := h
    simp only [if_neg h2, if_pos h3, if_pos h1] <;> omega

end Strata.Ple
