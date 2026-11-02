/-!
# MODULE: fused_gdn_decode
Index/launch arithmetic of the fused gated-delta-net decode kernels (conv+l2, alpha/beta projection, step+norm).

## Source semantics
`gdn_step_norm_kernel`: block = head, thread (col, rg) owns 32 state rows `rg*32 .. rg*32+31` of column `col`; row groups tile the 128 rows exactly.
`gdn_conv_l2_kernel`: a 3-deep history per channel slides by one and the 4-tap dot uses (hist0, hist1, hist2, x).
`gdn_ab_kernel`: row < h_v is alpha (gate), row >= h_v is beta; grid rounds 2*h_v rows up to 8 warps per block.

## Form template identifiers
S -> state_size, RG -> row_groups, RPG -> rows_per_thread, fused_gdn_step_norm -> step_norm_launch
-/
namespace Strata.FusedGdn

def S : Nat := 128
def RG : Nat := 4
def RPG : Nat := 32

-- § row group tiling

/-- Row groups tile the state exactly. -/
theorem rpg_cover : RG * RPG = S := by decide

/-- Every (row group, local row) pair addresses a row inside the state. -/
theorem row_lt (rg r : Nat) (hg : rg < RG) (hr : r < RPG) : rg * RPG + r < S := by (try simp only [S, RG, RPG] at *) <;> omega

/-- The row map is injective, so no two registers alias a state row. -/
theorem row_inj (rg r rg' r' : Nat) (hg : rg < RG) (hr : r < RPG) (hg' : rg' < RG) (hr' : r' < RPG) (h : rg * RPG + r = rg' * RPG + r') : rg = rg' ∧ r = r' := by (try simp only [S, RG, RPG] at *) <;> omega

/-- Every row below S is hit by some thread register. -/
theorem row_surj (i : Nat) (hi : i < S) : ∃ rg r, rg < RG ∧ r < RPG ∧ rg * RPG + r = i := by refine ⟨i / 32, i % 32, ?_, ?_, ?_⟩ <;> (try simp only [S, RG, RPG] at *) <;> omega

-- § thread ids

/-- `tid = rg*S + col` is below S exactly for row group 0 (the threads that load k, q and write y). -/
theorem tid_lt_iff (rg col : Nat) (hc : col < S) : rg * S + col < S ↔ rg = 0 := by (try simp only [S] at *) <;> omega

/-- The per-warp partial-sum slot `tid/32` fits the 16-entry `wsum` array. -/
theorem wsum_slot (tid : Nat) (h : tid < S * RG) : tid / 32 < S * RG / 32 := by (try simp only [S, RG] at *) <;> omega

/-- Row group 0 owns warps 0..3, which are exactly the four partials summed for the RMS. -/
theorem rms_warps (col : Nat) (hc : col < S) : col / 32 < 4 := by (try simp only [S] at *) <;> omega

-- § alpha/beta kernel

/-- The alpha/beta grid covers all 2*h_v rows. -/
theorem ab_grid_cover (h_v : Nat) : 2 * h_v ≤ (2 * h_v + 7) / 8 * 8 := by omega

/-- A beta row, rebased by h_v, is a valid head index. -/
theorem ab_beta_index (h_v row : Nat) (h : row < 2 * h_v) (hb : ¬ row < h_v) : row - h_v < h_v := by omega

/-- An alpha row is already a valid head index. -/
theorem ab_alpha_index (h_v row : Nat) (hb : row < h_v) : row < h_v := by omega

-- § conv kernel

/-- Channels divisible by S make the conv grid exact. -/
theorem conv_grid_exact (channels : Nat) (h : channels % S = 0) : channels / S * S = channels := by (try simp only [S] at *) <;> omega

/-- The 4-tap convolution. -/
def conv4 (v0 v1 v2 x w0 w1 w2 w3 : Int) : Int := v0 * w0 + v1 * w1 + v2 * w2 + x * w3

/-- The history slide. -/
def slide3 {α : Type} (h : α × α × α) (x : α) : α × α × α := (h.2.1, h.2.2, x)

/-- After sliding, the newest input sits in the last history slot. -/
theorem slide3_newest {α : Type} (a b c x : α) : slide3 (a, b, c) x = (b, c, x) := rfl

/-- The next token's convolution reads the previous input in tap 2 (the oldest slot is dropped). -/
theorem conv4_next (v0 v1 v2 x x' w0 w1 w2 w3 : Int) : conv4 v1 v2 x x' w0 w1 w2 w3 = v1 * w0 + v2 * w1 + x * w2 + x' * w3 := rfl

end Strata.FusedGdn
