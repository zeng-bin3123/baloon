/-!
# MODULE: gdn_reference
Launch geometry and sliding-window semantics of the reference gated-delta-net kernels
(`gdn_step`, `gdn_conv_step`, `gdn_l2_norm`, `gdn_beta_gate`, `gdn_out_norm`).

## Source semantics
One thread per (h, j) state column; no barriers because each column is independent.
The conv state holds d_conv-1 taps; kernel tap 0 reads the OLDEST tap, the new input lands in the last tap.
The state slides: drop the oldest, append the newest.

## Form template identifiers
S -> state width (<=128); h_v/h_k -> value/key heads; MAX_H -> heads per block (1); JTHREADS -> 32;
dc -> d_conv; st -> conv state; x -> new input.
-/
namespace Strata.Gdn

/-- Conv state after one step: shift left, append x. -/
def slide {α : Type} (st : Nat → α) (x : α) (dc i : Nat) : α := if i + 2 < dc then st (i + 1) else x

/-- Taps seen by the dot product: the d_conv-1 stored taps then the new input. -/
def window {α : Type} (st : Nat → α) (x : α) (dc i : Nat) : α := if i < dc - 1 then st i else x

/-- Non-last slots take the next older value. -/
theorem slide_shift {α : Type} (st : Nat → α) (x : α) (dc i : Nat) (h : i + 2 < dc) :
    slide st x dc i = st (i + 1) := by
  unfold slide; rw [if_pos h]

/-- The last slot (dc-2) receives the new input. -/
theorem slide_last {α : Type} (st : Nat → α) (x : α) (dc : Nat) (h : 2 ≤ dc) :
    slide st x dc (dc - 2) = x := by
  unfold slide; rw [if_neg (by omega)]

/-- The next step's window is this step's window shifted by one. -/
theorem window_slide {α : Type} (st : Nat → α) (x : α) (dc i : Nat) (h : i + 1 < dc) :
    window (slide st x dc) x dc i = window st x dc (i + 1) := by
  unfold window slide
  by_cases h1 : i + 2 < dc
  · have h2 : i < dc - 1 := by omega
    have h3 : i + 1 < dc - 1 := by omega
    simp [h1, h2, h3]
  · have h2 : i < dc - 1 := by omega
    have h3 : ¬ (i + 1 < dc - 1) := by omega
    simp [h1, h2, h3]

/-- Step grid: ceil(S/32) column blocks cover all S columns. -/
theorem col_cover (S j : Nat) (h : j < S) : j / 32 < (S + 31) / 32 := by omega

/-- With MAX_H = 1 the head grid is exactly h_v blocks. -/
theorem head_grid (hv : Nat) : (hv + 1 - 1) / 1 = hv := by omega

/-- Block column index recovered from j = bx*32 + tx. -/
theorem col_decomp (bx tx : Nat) (h : tx < 32) : (bx * 32 + tx) / 32 = bx ∧ (bx * 32 + tx) % 32 = tx := by
  omega

/-- Modulo head pairing yields a valid key head. -/
theorem src_head_lt (h hk : Nat) (hp : 0 < hk) : h % hk < hk := Nat.mod_lt _ hp

/-- Conv grid: ceil(C/256) blocks of 256 threads cover all channels. -/
theorem conv_cover (C c : Nat) (h : c < C) : c / 256 < (C + 255) / 256 := by omega

/-- Beta-gate grid: ceil(n/128) blocks of 128 threads cover all heads. -/
theorem beta_cover (n i : Nat) (h : i < n) : i / 128 < (n + 127) / 128 := by omega

/-- The l2/out-norm warp reduction handles at most 32 lanes times 32 strided steps = 1024 columns. -/
theorem l2_width_bound : 32 * 32 = 1024 := by decide

/-- Strided loop `i = tid; i < n; i += 32` visits residue class tid mod 32 only. -/
theorem stride_class (t k : Nat) (h : t < 32) : (t + 32 * k) % 32 = t := by omega

/-- Every index below n is visited by exactly one lane of the strided loop. -/
theorem stride_cover (i : Nat) : ∃ t k, t < 32 ∧ i = t + 32 * k := ⟨i % 32, i / 32, by omega, by omega⟩

/-- The staged shared rows ks/qs hold S <= 128 entries, so the S guard is sufficient. -/
theorem staged_bound (S i : Nat) (hS : S ≤ 128) (hi : i < S) : i < 128 := by omega

end Strata.Gdn
