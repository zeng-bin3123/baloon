/-!
# MODULE: ple_conv_history
Conv tap indexing, history advance and scratch alignment of the PLE block (`ple.cu`).

## Source semantics
Tap `k` of the dilated causal conv reads row `nhist - (kern-1-k)*dil` of `[history | new]`; for kern 4,
dil 3, nhist 9 the rows are 0,3,6,9 and only row 9 is the new input. History is a shift register: drop
the oldest row, append the new one. The conv state is ROW-FASTEST: flat index `row + nhist*channel`.
Scratch regions are 16-byte aligned.

## Form template identifiers
tapRow -> `row`; advance -> `history_advance_kernel`; flat -> `hist[row + nhist*c]`.
-/

namespace Strata.PleConvHistory

def tapRow (nhist kern dil k : Nat) : Nat := nhist - (kern - 1 - k) * dil

-- § taps

theorem real_geometry_rows :
    tapRow 9 4 3 0 = 0 ∧ tapRow 9 4 3 1 = 3 ∧ tapRow 9 4 3 2 = 6 ∧ tapRow 9 4 3 3 = 9 := by
  decide

/-- Only the last tap reads the new row. -/
theorem only_last_tap_is_new (k : Nat) (hk : k < 4) : tapRow 9 4 3 k = 9 → k = 3 := by
  interval_cases k <;> simp [tapRow]

/-- Every tap row is within `[0, nhist]`. -/
theorem tap_row_le (nhist kern dil k : Nat) : tapRow nhist kern dil k ≤ nhist := by
  unfold tapRow
  omega

-- § row-fastest flat index

def flat (nhist row c : Nat) : Nat := row + nhist * c

/-- Distinct (row, channel) pairs with row < nhist map to distinct flat slots. -/
theorem flat_injective (nhist r1 c1 r2 c2 : Nat) (h1 : r1 < nhist) (h2 : r2 < nhist)
    (h : flat nhist r1 c1 = flat nhist r2 c2) : r1 = r2 ∧ c1 = c2 := by
  unfold flat at h
  have e1 : (r1 + nhist * c1) % nhist = r1 := by
    rw [Nat.add_mul_mod_self_left]; exact Nat.mod_eq_of_lt h1
  have e2 : (r2 + nhist * c2) % nhist = r2 := by
    rw [Nat.add_mul_mod_self_left]; exact Nat.mod_eq_of_lt h2
  have hr : r1 = r2 := by rw [← e1, ← e2, h]
  subst hr
  refine ⟨rfl, ?_⟩
  have hpos : 0 < nhist := by omega
  have : nhist * c1 = nhist * c2 := by omega
  exact Nat.eq_of_mul_eq_mul_left hpos this

-- § history advance

def advance (h : List Nat) (x : Nat) : List Nat := h.tail ++ [x]

/-- A non-empty history keeps its length. -/
theorem advance_length (h : List Nat) (x : Nat) (hne : h ≠ []) : (advance h x).length = h.length := by
  unfold advance
  cases h with
  | nil => exact absurd rfl hne
  | cons a t => simp

/-- The newest row is last. -/
theorem advance_last (h : List Nat) (x : Nat) : (advance h x).getLast? = some x := by
  unfold advance
  simp

-- § scratch

def align16 (n : Nat) : Nat := (n + 15) / 16 * 16

theorem align16_ok (n : Nat) : n ≤ align16 n ∧ align16 n % 16 = 0 := by
  unfold align16
  omega

/-- The PLE key geometry: hc_dim = hc * n_embd. -/
theorem hc_dim_is_product : 4 * 2560 = 10240 := by decide

end Strata.PleConvHistory
