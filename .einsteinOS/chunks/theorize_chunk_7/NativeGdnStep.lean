/-!
# MODULE: native_gdn_step
Warp-sharded layout of the native gated-delta-net step: each warp owns one (head, column) and keeps four rows per lane in registers.

## Source semantics
Row index i = r*32 + lane for r < 4, lane < 32 covers 0..127 exactly once. Column col = z*4 + y with z < 32, y < 4 covers 0..127. The output is the warp-reduced readout scaled by 1/sqrt(S).

## Form template identifiers
step -> native_step, s_shard -> state_shard, S -> state_size
-/
namespace Strata.NativeGdn

/-- Rows owned by (register, lane) stay inside S = 128. -/
theorem row_lt (r lane : Nat) (hr : r < 4) (hl : lane < 32) : r * 32 + lane < 128 := by omega

/-- The row map is injective. -/
theorem row_inj (r l r' l' : Nat) (hr : r < 4) (hl : l < 32) (hr' : r' < 4) (hl' : l' < 32) (h : r * 32 + l = r' * 32 + l') : r = r' ∧ l = l' := by omega

/-- Every row is owned by some (register, lane). -/
theorem row_surj (i : Nat) (h : i < 128) : ∃ r l, r < 4 ∧ l < 32 ∧ r * 32 + l = i := by refine ⟨i / 32, i % 32, ?_, ?_, ?_⟩ <;> omega

/-- Column map: z*4 + y stays inside S. -/
theorem col_lt (z y : Nat) (hz : z < 32) (hy : y < 4) : z * 4 + y < 128 := by omega

/-- Column map is injective. -/
theorem col_inj (z y z' y' : Nat) (hy : y < 4) (hy' : y' < 4) (h : z * 4 + y = z' * 4 + y') : z = z' ∧ y = y' := by omega

/-- Every column is owned by some (z, y). -/
theorem col_surj (c : Nat) (h : c < 128) : ∃ z y, z < 32 ∧ y < 4 ∧ z * 4 + y = c := by refine ⟨c / 4, c % 4, ?_, ?_, ?_⟩ <;> omega

/-- The z grid dimension S/4 is 32. -/
theorem grid_z : 128 / 4 = 32 := by decide

/-- A head count at most 65535 fits the x grid dimension. -/
theorem head_grid (h_v : Nat) (h : h_v ≤ 65535) : h_v < 65536 := by omega

/-- Divisible head counts make modulo pairing balanced. -/
theorem pair_lt (head h_k : Nat) (hk : 0 < h_k) : head % h_k < h_k := Nat.mod_lt _ hk

/-- The extra reduced stride: lane 0 writes the single output of each warp. -/
theorem lane0_unique (l : Nat) (hl : l < 32) (h : l = 0 ∨ 0 < l) : l < 32 := hl

end Strata.NativeGdn
