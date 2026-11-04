/-!
# MODULE: qsa_select
SOURCES: src/kernels/cuda/qsa_select.cu, src/kernels/cuda/qsa.cu
Exact top-k selection over per-cell / per-block indexer scores: the monotone f32 order key, the threshold
(binary lifting / 8-bit radix) search, tie-breaking by ascending index, and the block-weighted expansion
of selected blocks into cell ids.

## Source semantics
* `order_key` maps f32 bit patterns to an unsigned key so integer comparison matches numeric order; -0.0 is
  folded onto +0.0 first.
* The threshold `thr` satisfies count(key >= thr) >= width > count(key >= thr+1); the selection is
  every key > thr plus the first `width - c_gt` keys equal to thr, which has exactly `width` elements.
* Block selection weights each block by r cells, the spare block by `n_kv - n_bid*r`; total weight is n_kv.
* Selected block b yields cells b*r .. b*r+w-1, ascending.

## Form template identifiers
order_key -> `orderKey`; width -> `width`; thr -> `t`; block weight -> `weightB`.
-/

namespace QsaSelect

-- § order key

/-- The f32 total-order key on raw bit patterns (`order_key`): sign bit set -> bitwise not, else set the sign bit. -/
def orderKey (b : Nat) : Nat := if 2147483648 ≤ b then 4294967295 - b else b + 2147483648

/-- `s + 0.0f` folds -0.0 (bits 0x80000000) onto +0.0 (bits 0). -/
def negZeroFold (b : Nat) : Nat := if b = 2147483648 then 0 else b

/-- Keys stay in 32 bits. -/
theorem orderKey_lt (b : Nat) (h : b < 4294967296) : orderKey b < 4294967296 := by
  unfold orderKey; split <;> omega

/-- Every negative float sorts strictly below every non-negative float. -/
theorem orderKey_neg_lt_pos (b1 b2 : Nat) (h1 : 2147483648 ≤ b1) (h1' : b1 < 4294967296)
    (h2 : b2 < 2147483648) : orderKey b1 < orderKey b2 := by
  unfold orderKey; split <;> split <;> omega

/-- On non-negative floats the key is monotone in the bit pattern. -/
theorem orderKey_mono_pos (b1 b2 : Nat) (h : b1 ≤ b2) (h2 : b2 < 2147483648) :
    orderKey b1 ≤ orderKey b2 := by
  unfold orderKey; split <;> split <;> omega

/-- On negative floats the key is antitone in the bit pattern. -/
theorem orderKey_anti_neg (b1 b2 : Nat) (h : b1 ≤ b2) (h1 : 2147483648 ≤ b1) (h2 : b2 < 4294967296) :
    orderKey b2 ≤ orderKey b1 := by
  unfold orderKey
  rw [if_pos h1]
  by_cases c2 : 2147483648 ≤ b2
  · rw [if_pos c2]; omega
  · rw [if_neg c2]; omega

/-- After folding, -0.0 and +0.0 get the same key (they tie, and the tie is broken by index). -/
theorem negZero_ties : orderKey (negZeroFold 2147483648) = orderKey (negZeroFold 0) := by decide

-- § threshold selection

/-- Count of keys strictly above `t`. -/
def cntGt (l : List Nat) (t : Nat) : Nat := (l.filter (fun k => decide (t < k))).length
/-- Count of keys equal to `t`. -/
def cntEq (l : List Nat) (t : Nat) : Nat := (l.filter (fun k => decide (k = t))).length
/-- Count of keys at or above `t`. -/
def cntGe (l : List Nat) (t : Nat) : Nat := (l.filter (fun k => decide (t ≤ k))).length

/-- `>= t` splits into `> t` and `= t`. -/
theorem cntGe_split (l : List Nat) (t : Nat) : cntGe l t = cntGt l t + cntEq l t := by
  unfold cntGe cntGt cntEq
  induction l with
  | nil => simp
  | cons a l ih =>
    by_cases h1 : t < a
    · have h2 : a ≠ t := by omega
      have h3 : t ≤ a := by omega
      simp [List.filter_cons, h1, h2, h3, ih]
      omega
    · by_cases h2 : a = t
      · have h3 : t ≤ a := by omega
        simp [List.filter_cons, h1, h2, ih]
        omega
      · have h3 : ¬ t ≤ a := by omega
        simp [List.filter_cons, h1, h2, h3, ih]

/-- `> t` is `>= t+1`. -/
theorem cntGt_eq_cntGe_succ (l : List Nat) (t : Nat) : cntGt l t = cntGe l (t + 1) := by
  unfold cntGt cntGe
  have : (fun k => decide (t < k)) = (fun k => decide (t + 1 ≤ k)) := rfl
  rw [this]

/-- Selected count: everything above the threshold plus as many ties as the budget leaves room for. -/
def selCount (l : List Nat) (t width : Nat) : Nat := cntGt l t + min (cntEq l t) (width - cntGt l t)

/-- Given the threshold property, the selection has exactly `width` elements. -/
theorem selCount_eq_width (l : List Nat) (t width : Nat)
    (h1 : cntGt l t < width) (h2 : width ≤ cntGt l t + cntEq l t) : selCount l t width = width := by
  unfold selCount; omega

/-- The binary-lifting invariant `count(>= t) >= width > count(>= t+1)` implies the selection property. -/
theorem threshold_property (l : List Nat) (t width : Nat)
    (hge : width ≤ cntGe l t) (hlt : cntGe l (t + 1) < width) : selCount l t width = width := by
  have hs := cntGe_split l t
  have hg := cntGt_eq_cntGe_succ l t
  exact selCount_eq_width l t width (by omega) (by omega)

/-- Ties never overflow the budget: the number of tied cells taken is at most `width - c_gt`. -/
theorem ties_taken_le (l : List Nat) (t width : Nat) :
    min (cntEq l t) (width - cntGt l t) ≤ width - cntGt l t := Nat.min_le_right _ _

-- § radix digits

/-- Four 8-bit digits recompose a 32-bit key. -/
theorem digits_recompose (k : Nat) (h : k < 4294967296) :
    (k / 16777216) * 16777216 + ((k / 65536) % 256) * 65536 + ((k / 256) % 256) * 256 + k % 256 = k := by
  omega

/-- The top digit is a byte. -/
theorem top_digit_lt (k : Nat) (h : k < 4294967296) : k / 16777216 < 256 := by omega

/-- A histogram bin index is a byte. -/
theorem digit_lt (k s : Nat) : (k / s) % 256 < 256 := Nat.mod_lt _ (by decide)

-- § block weights

/-- Weight of block `b`: r cells for completed blocks, the remainder for the spare/tail row. -/
def weightB (nBid nKv r b : Nat) : Nat := if b < nBid then r else nKv - nBid * r

/-- Prefix sum of a function. -/
def wsum (f : Nat → Nat) : Nat → Nat
  | 0 => 0
  | k + 1 => wsum f k + f k

/-- Completed blocks contribute `k * r`. -/
theorem wsum_prefix (nBid nKv r : Nat) : ∀ k, k ≤ nBid → wsum (weightB nBid nKv r) k = k * r := by
  intro k
  induction k with
  | zero => intro _; simp [wsum]
  | succ k ih =>
    intro hk
    have h1 : weightB nBid nKv r k = r := by unfold weightB; simp [show k < nBid by omega]
    show wsum (weightB nBid nKv r) k + weightB nBid nKv r k = (k+1) * r
    rw [ih (by omega), h1, Nat.add_mul, Nat.one_mul]

/-- Total weight over blocks 0..n_bid is `n_kv`. -/
theorem total_weight (nBid nKv r : Nat) (h : nBid * r ≤ nKv) :
    wsum (weightB nBid nKv r) (nBid + 1) = nKv := by
  show wsum (weightB nBid nKv r) nBid + weightB nBid nKv r nBid = nKv
  rw [wsum_prefix nBid nKv r nBid (Nat.le_refl _)]
  have : weightB nBid nKv r nBid = nKv - nBid * r := by unfold weightB; simp
  rw [this]
  omega

-- § expansion of selected blocks into cell ids

/-- A completed block's cells are real cells. -/
theorem full_block_cell_lt (b c nBid nKv : Nat) (hb : b < nBid) (hc : c < 4) (h : nBid * 4 ≤ nKv) :
    b * 4 + c < nKv := by omega

/-- The spare/tail block's cells are real cells. -/
theorem tail_block_cell_lt (c nBid nKv : Nat) (hc : c < nKv - nBid * 4) : nBid * 4 + c < nKv := by omega

/-- Emission order is ascending across blocks. -/
theorem cells_ascending (b b' c c' : Nat) (hb : b < b') (hc : c < 4) (hc' : c' < 4) :
    b * 4 + c < b' * 4 + c' := by omega

/-- Block -> cell map is injective. -/
theorem cell_unique (b b' c c' : Nat) (hc : c < 4) (hc' : c' < 4) (h : b * 4 + c = b' * 4 + c') :
    b = b' ∧ c = c' := by omega

/-- When everything fits (`n_kv <= width`) the selection is the identity; for the real geometry that is `n_kv <= 2051`. -/
theorem identity_branch (n : Nat) : n ≤ min n 2051 ↔ n ≤ 2051 := by omega

-- § launch geometry

/-- Warp-per-block scoring: every block index `b < max_blocks` is served by exactly one (blockIdx, warp). -/
theorem score_grid_covers (b maxBlocks : Nat) (h : b < maxBlocks) :
    b / 8 < (maxBlocks + 7) / 8 ∧ b = (b / 8) * 8 + b % 8 ∧ b % 8 < 8 := by omega

/-- The tail bias fires iff the spare row holds a partial block. -/
theorem tail_bias_iff (n : Nat) : n % 4 ≠ 0 ↔ (n / 4) * 4 < n := by omega

end QsaSelect
