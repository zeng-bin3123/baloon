/-!
# MODULE: fused_gdn_decode
SOURCES: src/kernels/cuda/fused_gdn.cu
Single-token fused gated-delta-net decode: causal-conv + SiLU + L2 norm, the alpha/beta projections, and the
state-update + RMS-norm + gate kernel. Thread/row layouts and the exact-arithmetic algebra are formalised.

## Source semantics
* `gdn_step_norm_kernel`: block = head, thread (col, rg); 4 row groups of 32 rows own the 128 state rows; kv_col is the
  sum of the four row-group partials; delta = (v - g*kv_col)*beta; state' = g*state + k*delta.
* `gdn_conv_l2_kernel`: one thread per channel, history of 3 slides by one per token.
* `gdn_ab_kernel`: one warp per row of [alpha ; beta] weight rows, 8 warps per block.

## Form template identifiers
S -> state_size, RG -> row_groups, RPG -> rows_per_group, tid -> flat_thread_id, stateRow -> owned_state_row,
sumTo -> exact_reduction, histStep -> conv_history_shift
-/
namespace Strata.FusedGdnDecode

-- § exact-arithmetic reduction library
/-- Sum of `f 0 .. f (n-1)` over the integers (exact model of a reduction). -/
def sumTo (f : Nat → Int) : Nat → Int
  | 0 => 0
  | n+1 => sumTo f n + f n

/-- Splitting a reduction at an arbitrary point. -/
theorem sumTo_add (f : Nat → Int) (n m : Nat) :
    sumTo f (n+m) = sumTo f n + sumTo (fun r => f (n+r)) m := by
  induction m with
  | zero => simp [sumTo]
  | succ m ih =>
    have e : sumTo f (n + (m+1)) = sumTo f (n+m) + f (n+m) := rfl
    have e2 : sumTo (fun r => f (n+r)) (m+1) = sumTo (fun r => f (n+r)) m + f (n+m) := rfl
    rw [e, ih, e2]
    omega

/-- A reduction over `a*b` items equals the reduction of `a` block reductions of `b` items. -/
theorem sum_blocks (f : Nat → Int) (b a : Nat) :
    sumTo f (a*b) = sumTo (fun r => sumTo (fun c => f (r*b+c)) b) a := by
  induction a with
  | zero => simp [sumTo]
  | succ a ih =>
    have h : (a+1)*b = a*b + b := Nat.succ_mul a b
    rw [h, sumTo_add f (a*b) b, ih]
    try rfl

/-- Scalar factors pull out of a reduction. -/
theorem sum_scale (g : Int) (f : Nat → Int) (n : Nat) :
    sumTo (fun i => g * f i) n = g * sumTo f n := by
  induction n with
  | zero => simp [sumTo]
  | succ n ih => simp [sumTo, ih, Int.mul_add]

/-- Decaying the state before contracting with k equals decaying the contraction. -/
theorem decay_commutes (g : Int) (s k : Nat → Int) (n : Nat) :
    sumTo (fun i => g * s i * k i) n = g * sumTo (fun i => s i * k i) n := by
  have h : (fun i => g * s i * k i) = (fun i => g * (s i * k i)) :=
    funext fun i => Int.mul_assoc _ _ _
  rw [h, sum_scale]

-- § constants and thread layout
/-- State size (rows = cols). -/
def S : Nat := 128
/-- Row groups per column. -/
def RG : Nat := 4
/-- Rows per group. -/
def RPG : Nat := S / RG

/-- 32 rows per thread. -/
theorem rpg_eq : RPG = 32 := by decide

/-- The row groups tile the state rows exactly. -/
theorem rg_mul_rpg : RG * RPG = S := by decide

/-- Block size is 512 threads. -/
theorem block_threads : S * RG = 512 := by decide

/-- Flat thread id. -/
def tid (rg col : Nat) : Nat := rg * S + col
/-- State row owned by register slot r of row group rg. -/
def stateRow (rg r : Nat) : Nat := rg * RPG + r

/-- Flat thread ids are within the block. -/
theorem tid_lt (rg col : Nat) (h1 : rg < RG) (h2 : col < S) : tid rg col < S * RG := by
  simp only [tid, S, RG] at *
  omega

/-- The row group is recovered from the flat id. -/
theorem tid_div (rg col : Nat) (h2 : col < S) : tid rg col / S = rg := by
  simp only [tid, S] at *
  omega

/-- The column is recovered from the flat id. -/
theorem tid_mod (rg col : Nat) (h2 : col < S) : tid rg col % S = col := by
  simp only [tid, S] at *
  omega

/-- Distinct (rg, col) give distinct flat ids. -/
theorem tid_inj (a b c d : Nat) (hb : b < S) (hd : d < S) (h : tid a b = tid c d) : a = c ∧ b = d := by
  simp only [tid, S] at *
  omega

/-- Owned rows are inside the state. -/
theorem stateRow_lt (rg r : Nat) (h1 : rg < RG) (h2 : r < RPG) : stateRow rg r < S := by
  simp only [stateRow, RG, RPG, S] at *
  omega

/-- Row ownership is injective: no two (group, slot) pairs own the same row. -/
theorem stateRow_inj (a r c s : Nat) (hr : r < RPG) (hs : s < RPG)
    (h : stateRow a r = stateRow c s) : a = c ∧ r = s := by
  simp only [stateRow, RPG, S, RG] at *
  omega

/-- Row ownership is onto: every state row is owned. -/
theorem stateRow_surj (i : Nat) (hi : i < S) :
    ∃ rg r, rg < RG ∧ r < RPG ∧ stateRow rg r = i := by
  refine ⟨i / 32, i % 32, ?_, ?_, ?_⟩ <;> simp only [stateRow, RG, RPG, S] at * <;> omega

-- § reduction semantics
/-- kv_col as the sum of the four row-group partials equals the full contraction. -/
theorem kv_col_partition (f : Nat → Int) :
    sumTo f S = sumTo (fun rg => sumTo (fun r => f (rg * RPG + r)) RPG) RG :=
  sum_blocks f RPG RG

/-- The fused delta (decay folded in) equals the two-pass delta. -/
theorem fused_delta_eq (g beta v : Int) (s k : Nat → Int) (n : Nat) :
    (v - sumTo (fun i => g * s i * k i) n) * beta = (v - g * sumTo (fun i => s i * k i) n) * beta := by
  rw [decay_commutes]

/-- Thread-id to warp index for row group 0 stays within the four RMS partial slots. -/
theorem wsum_slot_lt (t : Nat) (h : t < S) : t / 32 < 4 := by
  simp only [S] at *
  omega

-- § conv history
/-- History of the three previous inputs. -/
structure Hist where
  a : Int
  b : Int
  c : Int

/-- The history slides by one per token. -/
def histStep (h : Hist) (x : Int) : Hist := ⟨h.b, h.c, x⟩

/-- Four-tap convolution output. -/
def convOut (h : Hist) (x w0 w1 w2 w3 : Int) : Int := h.a * w0 + h.b * w1 + h.c * w2 + x * w3

/-- After three tokens the history is exactly the last three inputs, independent of the old history. -/
theorem hist_three (h : Hist) (x y z : Int) :
    histStep (histStep (histStep h x) y) z = ⟨x, y, z⟩ := rfl

-- § launch geometry
/-- Channel index of a thread is in range. -/
theorem conv_channel_lt (blk tx channels : Nat) (hb : blk < channels / 128) (ht : tx < 128) :
    blk * 128 + tx < channels := by omega

/-- A thread's block is the channel's 128-wide head group. -/
theorem conv_head_of_channel (blk tx : Nat) (ht : tx < 128) : (blk * 128 + tx) / 128 = blk := by omega

/-- The ab grid covers all 2*h_v weight rows. -/
theorem ab_grid_cover (hv : Nat) : 2 * hv ≤ ((2 * hv + 7) / 8) * 8 := by omega

/-- Row split: alpha rows come first, beta rows are offset by h_v and stay in range. -/
theorem ab_row_split (row hv : Nat) (h : row < 2 * hv) :
    (row < hv → row < hv) ∧ (¬ row < hv → row - hv < hv) := by
  constructor <;> intro hh <;> omega

/-- Softplus selection branch: large inputs pass through unchanged. -/
theorem softplus_large (v : Int) (f : Int → Int) (h : 20 < v) : (if 20 < v then v else f v) = v := by
  simp [h]

/-- Head pairing is modulo h_k and in range. -/
theorem qhead_lt (head hk : Nat) (h : 0 < hk) : head % hk < hk := Nat.mod_lt _ h

end Strata.FusedGdnDecode
