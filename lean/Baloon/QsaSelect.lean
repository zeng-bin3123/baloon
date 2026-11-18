-- Baloon.QsaSelect: exact top-k selection over per-cell/per-block indexer scores.
-- SOURCES: src/kernels/cuda/qsa_select.cu, src/kernels/cuda/qsa.cu
-- Unit: u316bd6b9ec

namespace Baloon.QsaSelect

/-- f32 total-order key: sign-bit set uses bitwise NOT, else sets the sign bit.
    Uses a let-binding for the sign threshold. -/
def orderKey (b : Nat) : Nat :=
  let sign : Nat := 2147483648
  if sign ≤ b then 4294967295 - b else b + sign

/-- Fold -0.0 onto +0.0 using negated condition. -/
def negZeroFold (b : Nat) : Nat := if b ≠ 2147483648 then b else 0

theorem orderKey_lt (b : Nat) (h : b < 4294967296) : orderKey b < 4294967296 := by
  simp only [orderKey]; split_ifs <;> omega

theorem orderKey_neg_lt_pos (b1 b2 : Nat) (h1 : 2147483648 ≤ b1) (h1' : b1 < 4294967296)
    (h2 : b2 < 2147483648) : orderKey b1 < orderKey b2 := by
  simp only [orderKey]; split_ifs with c1 c2 <;> omega

theorem orderKey_mono_pos (b1 b2 : Nat) (h : b1 ≤ b2) (h2 : b2 < 2147483648) :
    orderKey b1 ≤ orderKey b2 := by
  simp only [orderKey]; split_ifs with c1 c2 <;> omega

theorem orderKey_anti_neg (b1 b2 : Nat) (h : b1 ≤ b2) (h1 : 2147483648 ≤ b1) (h2 : b2 < 4294967296) :
    orderKey b2 ≤ orderKey b1 := by
  simp only [orderKey]
  rw [if_pos h1]
  by_cases c2 : 2147483648 ≤ b2
  · rw [if_pos c2]; omega
  · rw [if_neg c2]; omega

theorem negZero_ties : orderKey (negZeroFold 2147483648) = orderKey (negZeroFold 0) := by
  norm_num [orderKey, negZeroFold]

/-- Count keys strictly above t, using x as the lambda variable. -/
def cntGt (l : List Nat) (t : Nat) : Nat := (l.filter (fun x => decide (t < x))).length
/-- Count keys equal to t. -/
def cntEq (l : List Nat) (t : Nat) : Nat := (l.filter (fun x => decide (x = t))).length
/-- Count keys at or above t. -/
def cntGe (l : List Nat) (t : Nat) : Nat := (l.filter (fun x => decide (t ≤ x))).length

theorem cntGe_split (l : List Nat) (t : Nat) : cntGe l t = cntGt l t + cntEq l t := by
  unfold cntGe cntGt cntEq
  induction l with
  | nil => simp
  | cons a l ih =>
    simp only [List.filter_cons]
    by_cases h1 : t < a
    · have h3 : t ≤ a := le_of_lt h1
      have h4 : a ≠ t := Nat.ne_of_gt h1
      simp [h1, h3, h4, ih]; omega
    · by_cases h2 : a = t
      · have h3 : t ≤ a := le_of_eq h2.symm
        simp [h1, h2, h3, ih]; omega
      · have h3 : ¬t ≤ a := by omega
        simp [h1, h2, h3, ih]

theorem cntGt_eq_cntGe_succ (l : List Nat) (t : Nat) : cntGt l t = cntGe l (t + 1) := by
  simp only [cntGt, cntGe]
  congr 1; ext k
  simp [Nat.lt_iff_add_one_le]

/-- Selected count using a let binding for the gt component. -/
def selCount (l : List Nat) (t width : Nat) : Nat :=
  let gt := cntGt l t
  gt + min (cntEq l t) (width - gt)

theorem selCount_eq_width (l : List Nat) (t width : Nat)
    (h1 : cntGt l t < width) (h2 : width ≤ cntGt l t + cntEq l t) : selCount l t width = width := by
  simp only [selCount]; omega

theorem threshold_property (l : List Nat) (t width : Nat)
    (hge : width ≤ cntGe l t) (hlt : cntGe l (t + 1) < width) : selCount l t width = width := by
  have split := cntGe_split l t
  have succ := cntGt_eq_cntGe_succ l t
  apply selCount_eq_width <;> omega

theorem ties_taken_le (l : List Nat) (t width : Nat) :
    min (cntEq l t) (width - cntGt l t) ≤ width - cntGt l t := by
  exact Nat.min_le_right _ _

theorem digits_recompose (k : Nat) (h : k < 4294967296) :
    (k / 16777216) * 16777216 + ((k / 65536) % 256) * 65536 + ((k / 256) % 256) * 256 + k % 256 = k := by
  have hmod : k % 256 < 256 := Nat.mod_lt k (by decide)
  omega

theorem top_digit_lt (k : Nat) (h : k < 4294967296) : k / 16777216 < 256 := by
  have := h; omega

theorem digit_lt (k s : Nat) : (k / s) % 256 < 256 := by
  exact Nat.mod_lt _ (by decide)

/-- Weight of block b, using a let for the tail weight. -/
def weightB (nBid nKv r b : Nat) : Nat :=
  let tail := nKv - nBid * r
  if b < nBid then r else tail

/-- Prefix sum of f, with n as the recursive variable (not k). -/
def wsum (f : Nat → Nat) : Nat → Nat
  | 0 => 0
  | n + 1 => wsum f n + f n

theorem wsum_prefix (nBid nKv r : Nat) : ∀ k, k ≤ nBid → wsum (weightB nBid nKv r) k = k * r := by
  intro k
  induction k with
  | zero => intro _; simp [wsum]
  | succ k ih =>
    intro hk
    have hw : weightB nBid nKv r k = r := by
      simp only [weightB]; rw [if_pos (by omega)]
    simp only [wsum, ih (by omega), hw, Nat.succ_mul]

theorem total_weight (nBid nKv r : Nat) (h : nBid * r ≤ nKv) :
    wsum (weightB nBid nKv r) (nBid + 1) = nKv := by
  simp only [wsum, wsum_prefix nBid nKv r nBid (le_refl _)]
  have : weightB nBid nKv r nBid = nKv - nBid * r := by simp [weightB]
  rw [this]; omega

theorem full_block_cell_lt (b c nBid nKv : Nat) (hb : b < nBid) (hc : c < 4) (h : nBid * 4 ≤ nKv) :
    b * 4 + c < nKv := by
  have hbid : b * 4 < nBid * 4 := Nat.mul_lt_mul_right 4 hb
  omega

theorem tail_block_cell_lt (c nBid nKv : Nat) (hc : c < nKv - nBid * 4) : nBid * 4 + c < nKv := by
  have hc' := hc; omega

theorem cells_ascending (b b' c c' : Nat) (hb : b < b') (hc : c < 4) (hc' : c' < 4) :
    b * 4 + c < b' * 4 + c' := by
  have h1 : (b + 1) * 4 ≤ b' * 4 := Nat.mul_le_mul_right 4 (by omega)
  omega

theorem cell_unique (b b' c c' : Nat) (hc : c < 4) (hc' : c' < 4) (h : b * 4 + c = b' * 4 + c') :
    b = b' ∧ c = c' :=
  ⟨by omega, by omega⟩

theorem identity_branch (n : Nat) : n ≤ min n 2051 ↔ n ≤ 2051 :=
  Iff.intro (by omega) (by omega)

theorem score_grid_covers (b maxBlocks : Nat) (h : b < maxBlocks) :
    b / 8 < (maxBlocks + 7) / 8 ∧ b = (b / 8) * 8 + b % 8 ∧ b % 8 < 8 :=
  ⟨by omega, by omega, by omega⟩

theorem tail_bias_iff (n : Nat) : n % 4 ≠ 0 ↔ (n / 4) * 4 < n :=
  Iff.intro (by omega) (by omega)

end Baloon.QsaSelect
