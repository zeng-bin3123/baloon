/-!
# MODULE: qsa_block_select
Block-level scoring and radix-select top-k over pooled indexer blocks (block width R = 4).

## Source semantics
Block b < n_bid has weight 4; the last block n_bid has weight n_kv - 4*n_bid (= n_kv % 4, possibly 0).
The radix select fixes the 32-bit threshold key 8 bits at a time using weighted histograms.
Selected cell ids are b*4 + c for c < weight, ascending, with ties broken by lowest index.

## Form template identifiers
R -> 4; TOPK_T -> 256 threads; n_bid -> completed blocks; width -> selection width; thr -> threshold.
-/
namespace Strata.QsaSelect

/-- Weight of block b. -/
def weight (nkv nbid b : Nat) : Nat := if b < nbid then 4 else nkv - nbid * 4

/-- Completed blocks plus tail weight account for all cells. -/
theorem weights_total (nkv : Nat) : (nkv / 4) * 4 + (nkv - (nkv / 4) * 4) = nkv := by omega

/-- The tail weight equals n_kv % 4. -/
theorem tail_weight (nkv : Nat) : nkv - (nkv / 4) * 4 = nkv % 4 := by omega

/-- The tail weight is below the block width. -/
theorem tail_weight_lt (nkv : Nat) : nkv - (nkv / 4) * 4 < 4 := by omega

/-- Full block weight. -/
theorem weight_full (nkv nbid b : Nat) (h : b < nbid) : weight nkv nbid b = 4 := by
  unfold weight; rw [if_pos h]

/-- Tail block weight. -/
theorem weight_tail (nkv nbid : Nat) : weight nkv nbid nbid = nkv - nbid * 4 := by
  unfold weight; rw [if_neg (by omega)]

/-- Cell ids b*4 + c with c < 4 are unique per (block, offset). -/
theorem cell_id_inj (b c b' c' : Nat) (hc : c < 4) (hc' : c' < 4) (e : b * 4 + c = b' * 4 + c') :
    b = b' ∧ c = c' := by omega

/-- A cell id of block b < n_bid is below n_kv. -/
theorem cell_id_lt (b c nbid nkv : Nat) (hb : b < nbid) (hc : c < 4) (h : nbid * 4 ≤ nkv) : b * 4 + c < nkv := by
  omega

/-- A tail cell id c < weight is below n_kv. -/
theorem tail_cell_lt (nbid nkv c : Nat) (hc : c < nkv - nbid * 4) (h : nbid * 4 ≤ nkv) : nbid * 4 + c < nkv := by
  omega

/-- Identity branch: if n_kv <= width, all n_kv cells are selected. -/
theorem identity_branch (nkv w : Nat) (h : nkv ≤ w) : min nkv w = nkv := by omega

/-- Digit extraction: a 32-bit key is the concatenation of its four bytes. -/
theorem key_bytes (k : Nat) (h : k < 4294967296) :
    k = (k / 16777216 % 256) * 16777216 + (k / 65536 % 256) * 65536 + (k / 256 % 256) * 256 + k % 256 := by
  omega

/-- Each radix digit is below 256. -/
theorem digit_lt (k s : Nat) : k / s % 256 < 256 := by omega

/-- Prefix assembled from the four selected digits is below 2^32. -/
theorem prefix_lt (d3 d2 d1 d0 : Nat) (h3 : d3 < 256) (h2 : d2 < 256) (h1 : d1 < 256) (h0 : d0 < 256) :
    d3 * 16777216 + d2 * 65536 + d1 * 256 + d0 < 4294967296 := by omega

/-- Digit-assembly is injective on byte digits. -/
theorem prefix_inj (a3 a2 a1 a0 b3 b2 b1 b0 : Nat)
    (h3 : a3 < 256) (h2 : a2 < 256) (h1 : a1 < 256) (h0 : a0 < 256)
    (g3 : b3 < 256) (g2 : b2 < 256) (g1 : b1 < 256) (g0 : b0 < 256)
    (e : a3 * 16777216 + a2 * 65536 + a1 * 256 + a0 = b3 * 16777216 + b2 * 65536 + b1 * 256 + b0) :
    a3 = b3 ∧ a2 = b2 ∧ a1 = b1 ∧ a0 = b0 := by omega

/-- Digit descent: the first digit d with cum + hist[d] >= width keeps `above = cum` below width. -/
theorem descent_invariant (cum h w : Nat) (hlt : cum < w) (hge : w ≤ cum + h) : w - cum ≤ h ∧ cum < w := by
  omega

/-- Equal budget: cells equal to thr that fit are width - above, positive after the descent. -/
theorem eq_budget_pos (above w : Nat) (h : above < w) : 0 < w - above := by omega

/-- Per-thread equal quota is clamped between 0 and its own equal count. -/
theorem quota_bounds (budget before eq : Nat) : min eq (budget - before) ≤ eq := by omega

/-- Total selection when quotas are consumed in thread order: gt + min eq (budget) <= width. -/
theorem total_selected (gt eq w : Nat) (h : gt ≤ w) : gt + min eq (w - gt) ≤ w := by omega

/-- Block-score grid: ceil(max_blocks/8) blocks of 8 warps cover all blocks. -/
theorem score_grid_cover (b mb : Nat) (h : b < mb) : b / 8 < (mb + 7) / 8 := by omega

/-- Block index from (blockIdx.x, warp). -/
theorem score_block_decomp (bx w : Nat) (hw : w < 8) : (bx * 8 + w) / 8 = bx ∧ (bx * 8 + w) % 8 = w := by
  omega

/-- Lane dimension slices: 32 lanes x 4 floats cover the 128 indexer dims. -/
theorem idx_dim_partition (d : Nat) (h : d < 128) : ∃ l j, l < 32 ∧ j < 4 ∧ d = l * 4 + j :=
  ⟨d / 4, d % 4, by omega, by omega, by omega⟩

/-- Per-thread block range [t*per, t*per+per) covers all nb blocks for 256 threads. -/
theorem thread_range_cover (b nb : Nat) (h : b < nb) (hn : 0 < nb) :
    b / ((nb + 255) / 256) < 256 := by
  have hc : 0 < (nb + 255) / 256 := by omega
  exact Nat.div_lt_of_lt_mul (by omega)

end Strata.QsaSelect
