-- Baloon.PleGeometry: PLE (prefix-length encoder) convolution geometry:
-- tap-row addressing, kernel weight indices, state indexing, history advance
-- and stream/workspace layout.
-- Independent implementation proving the same mathematical facts as Ref.PleGeometry.

namespace Baloon.PleGeometry

/-- Source row of tap k. -/
def tapRow (nhist kern dil k : Nat) : Nat := (nhist - (kern - 1 - k) * dil)

theorem real_tap_rows : (List.range 4).map (tapRow 9 4 3) = [0, 3, 6, 9] := by { decide }
theorem only_last_tap_is_new : ∀ k < 4, (tapRow 9 4 3 k = 9 ↔ k = 3) := by { decide }

theorem new_row_only_last (nhist kern dil k : Nat) (hd : 0 < dil) (hk : k < kern)
    (hle : (kern - 1 - k) * dil ≤ nhist) (hn : 0 < nhist) :
    (tapRow nhist kern dil k = nhist ↔ k = kern - 1) := by
  unfold tapRow
  have hz : (kern - 1 - k) * dil = 0 ↔ kern - 1 - k = 0 := by
    constructor
    · intro h
      rcases Nat.mul_eq_zero.mp h with h | h
      · exact h
      · omega
    · intro h; rw [h]; simp
  generalize (kern - 1 - k) * dil = p at hle hz ⊢
  omega

theorem kernel_weight_index (k c : Nat) (hk : k < 4) (hc : c < 10240) : k + 4 * c < 40960 := by { omega }
theorem state_index (row c : Nat) (hr : row < 9) (hc : c < 10240) : row + 9 * c < 9 * 10240 := by { omega }
theorem layout_not_transposable : 1 + 9 * 0 ≠ 1 * 10240 + 0 := by { decide }

def advance {α : Type} (h : List α) (x : α) : List α := (h.tail ++ [x])

theorem advance_length {α : Type} (h : List α) (x : α) (hne : h ≠ []) :
    (advance h x).length = h.length := by
  cases h with
  | nil => exact (absurd rfl hne)
  | cons a t => simp [advance]

theorem advance_last {α : Type} (h : List α) (x : α) : (advance h x).getLast? = some x := by { simp [advance] }

theorem stream_in_range (c : Nat) (h : c < 4) : c * 2560 + 2560 ≤ 10240 := by { omega }
theorem bcast_indices (i : Nat) (h : i < 10240) : i % 2560 < 2560 ∧ i / 2560 < 4 := by { omega }
theorem hc_dim_factor : 4 * 2560 = 10240 := by { decide }

def r16 (n : Nat) : Nat := ((n + 15) / 16 * 16)
theorem r16_ge (n : Nat) : n ≤ r16 n := by unfold r16; omega
theorem r16_aligned (n : Nat) : r16 n % 16 = 0 := by unfold r16; omega

end Baloon.PleGeometry
