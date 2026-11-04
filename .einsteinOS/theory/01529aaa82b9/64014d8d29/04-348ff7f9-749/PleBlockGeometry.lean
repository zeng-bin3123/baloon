/-!
# MODULE: ple_block_geometry
SOURCES: src/kernels/cuda/ple.cu
Index geometry of the PLE block's GPU half: the depthwise causal dilated convolution taps, the conv-state
layout (row-fastest), history advance, the broadcast/norm stream layout, workspace carving, and the
buffer-overlap predicate.

## Source semantics
* The conv reads rows `nhist - (kern-1-k)*dil` of `[history | new]`; for kern 4, dil 3, nhist 9 these are
  0, 3, 6, 9 and only the last tap reads the NEW row.
* The state is row-fastest: flat index `row + nhist * channel` (not `row * hc_dim + channel`).
* `ple_history_advance` shifts each channel column left by one and appends the new value.
* The scratch is carved from the caller's workspace at 16-byte aligned offsets.
* Two byte ranges overlap iff each starts before the other ends.

## Form template identifiers
NG_HIST -> history_rows ; NG_HC_DIM -> stream_dim ; NG_HC -> stream_count ; PLE_CONV_KERNEL -> conv_taps
-/

namespace Strata.PleGeometry

-- § conv taps

/-- Source row of tap `k`. -/
def tapRow (nhist kern dil k : Nat) : Nat := nhist - (kern - 1 - k) * dil

/-- The real geometry reads rows 0, 3, 6, 9. -/
theorem real_tap_rows : (List.range 4).map (tapRow 9 4 3) = [0, 3, 6, 9] := by decide

/-- Only the last tap reads the new row for the real geometry. -/
theorem only_last_tap_is_new : ∀ k < 4, (tapRow 9 4 3 k = 9 ↔ k = 3) := by decide

/-- In general (positive dilation, history long enough) only tap `kern-1` reads the new row. -/
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
    · intro h
      rw [h]
      simp
  generalize (kern - 1 - k) * dil = p at hle hz ⊢
  omega

/-- Kernel weight index `k + 4c` stays inside `4 * hc_dim` for the real geometry. -/
theorem kernel_weight_index (k c : Nat) (hk : k < 4) (hc : c < 10240) : k + 4 * c < 40960 := by omega

/-- Row-fastest state indexing stays in the state buffer for the history rows. -/
theorem state_index (row c : Nat) (hr : row < 9) (hc : c < 10240) : row + 9 * c < 9 * 10240 := by omega

/-- Row-fastest and channel-slowest indexing disagree, so mixing them up reads a transposed state. -/
theorem layout_not_transposable : 1 + 9 * 0 ≠ 1 * 10240 + 0 := by decide

-- § history advance

/-- Shift a column left by one and append the new value. -/
def advance {α : Type} (h : List α) (x : α) : List α := h.tail ++ [x]

/-- Advancing preserves the history length. -/
theorem advance_length {α : Type} (h : List α) (x : α) (hne : h ≠ []) :
    (advance h x).length = h.length := by
  cases h with
  | nil => exact absurd rfl hne
  | cons a t => simp [advance]

/-- The newest entry is the appended value. -/
theorem advance_last {α : Type} (h : List α) (x : α) : (advance h x).getLast? = some x := by
  simp [advance]

-- § stream layout

/-- Streams partition `hc_dim`: stream `c < 4` occupies `[c*2560, (c+1)*2560)` inside 10240. -/
theorem stream_in_range (c : Nat) (h : c < 4) : c * 2560 + 2560 ≤ 10240 := by omega

/-- Broadcast indices: element `i < 10240` reads value `i % 2560` and gate `i / 2560 < 4`. -/
theorem bcast_indices (i : Nat) (h : i < 10240) : i % 2560 < 2560 ∧ i / 2560 < 4 := by omega

/-- `hc * n_embd = hc_dim`. -/
theorem hc_dim_factor : 4 * 2560 = 10240 := by decide

-- § workspace carving

/-- Round up to 16. -/
def r16 (n : Nat) : Nat := (n + 15) / 16 * 16

/-- Total scratch of `ple_block_scratch_bytes`. -/
def scratchBytes : Nat :=
  r16 ((5 * 10240 + 2560 + 4) * 4) + r16 (2560 / 32 * 34) + 2560 * 2 + 256

/-- The concrete scratch size. -/
theorem scratch_value : scratchBytes = 223152 := by decide

/-- The activation image and the bf16 copy start at 16-aligned offsets and fit the total. -/
theorem scratch_regions (f q : Nat) :
    r16 f % 16 = 0 ∧ (r16 f + r16 q) % 16 = 0 ∧ r16 f + q ≤ r16 f + r16 q := by
  unfold r16
  omega

/-- The float workspace holds 5 * hc_dim + n_embd + hc floats, ending exactly at the gate buffer. -/
theorem float_workspace : 5 * 10240 + 2560 + 4 = 53764 := by decide

-- § overlap predicate

/-- Byte ranges `[a, a+ab)` and `[b, b+bb)` overlap (non-empty ranges only). -/
def overlap (a ab b bb : Nat) : Prop :=
  if a < b then b - a < ab else a - b < bb

/-- The overlap test is exactly interval intersection. -/
theorem overlap_iff (a ab b bb : Nat) (ha : 0 < ab) (hb : 0 < bb) :
    overlap a ab b bb ↔ (a < b + bb ∧ b < a + ab) := by
  unfold overlap
  split <;> omega

/-- The overlap test is symmetric. -/
theorem overlap_symm (a ab b bb : Nat) (ha : 0 < ab) (hb : 0 < bb) :
    overlap a ab b bb ↔ overlap b bb a ab := by
  rw [overlap_iff a ab b bb ha hb, overlap_iff b bb a ab hb ha]
  omega

-- § gate sign

/-- Sign function of the gate. -/
def sgn (s : Int) : Int := if s > 0 then 1 else if s < 0 then -1 else 0

/-- The gate sign is odd, so the gate is symmetric about 0.5. -/
theorem sgn_neg (s : Int) : sgn (-s) = -sgn s := by
  simp only [sgn]
  split_ifs <;> omega

end Strata.PleGeometry
