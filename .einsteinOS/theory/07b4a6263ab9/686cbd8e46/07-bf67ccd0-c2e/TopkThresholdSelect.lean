/-!
# MODULE: topk_threshold_select
SOURCES: src/kernels/cuda/qsa.cu, src/kernels/cuda/qsa_select.cu
The exact top-k used by `topk_512` (binary lifting) and `block_topk_kernel` (radix select): find a
threshold key, then emit every cell strictly above it plus the first `width - c_gt` cells equal to it
in ONE ascending walk, reproducing `ggml_top_k` with the ascending-index tie rule.

## Source semantics
* `order_key` maps an f32 bit pattern to an unsigned key: negatives are bit-inverted, non-negatives get
  the sign bit set; `-0.0` is normalised to `+0.0` first; NaN maps to key 0 (never preferred).
* `count(key ≥ v)` is antitone in `v`; the threshold `thr` satisfies
  `count(key ≥ thr) ≥ width > count(key ≥ thr + 1)`.
* The walk emits exactly `c_gt + min(c_eq, budget)` ids, strictly ascending; with the threshold
  postcondition and `budget = width - c_gt` it emits exactly `width` ids (the bug the source comment
  "THE RANK VS THE COUNT" records was a wrong budget here).

## Form template identifiers
`order_key` → `orderKey`, `-0.0` normalisation → `normZero`, `thr` → `thr`, `c_gt` → `cgt`,
equal count → `ceq`, `count(key >= cand)` → `cntGe`, `eq_budget` → `budget`, emit loop → `emit`.
-/

namespace Strata.Topk

-- § order key on 32-bit patterns

/-- The sign bit `0x80000000`. -/
def signBit : Nat := 2147483648

/-- `order_key` on a raw 32-bit pattern `b < 2^32`. -/
def orderKey (b : Nat) : Nat := if signBit ≤ b then 4294967295 - b else b + signBit

/-- `s + 0.0f` turns the pattern of `-0.0` into the pattern of `+0.0`. -/
def normZero (b : Nat) : Nat := if b = signBit then 0 else b

/-- Keys stay 32-bit. -/
theorem orderKey_lt (b : Nat) (hb : b < 4294967296) : orderKey b < 4294967296 := by
  unfold orderKey signBit
  split <;> omega

/-- Distinct patterns get distinct keys, so the key is a total order on non-NaN floats. -/
theorem orderKey_injective (b b' : Nat) (hb : b < 4294967296) (hb' : b' < 4294967296)
    (h : orderKey b = orderKey b') : b = b' := by
  unfold orderKey signBit at h
  split at h <;> split at h <;> omega

/-- Every negative float orders below every non-negative float. -/
theorem orderKey_neg_lt_nonneg (b b' : Nat) (hb : signBit ≤ b) (hb4 : b < 4294967296)
    (hb' : b' < signBit) : orderKey b < orderKey b' := by
  unfold orderKey
  simp only [hb, if_true, Nat.not_le.mpr hb', if_false]
  unfold signBit at *
  omega

/-- Larger magnitude negatives (larger bit patterns) get smaller keys. -/
theorem orderKey_neg_antitone (b b' : Nat) (hb : signBit ≤ b) (hbb : b < b')
    (hb4 : b' < 4294967296) : orderKey b' < orderKey b := by
  have hb' : signBit ≤ b' := Nat.le_trans hb (Nat.le_of_lt hbb)
  unfold orderKey
  simp only [hb, hb', if_true]
  omega

/-- Non-negative floats keep their bit order. -/
theorem orderKey_pos_monotone (b b' : Nat) (hbb : b < b') (hb' : b' < signBit) :
    orderKey b < orderKey b' := by
  have hb : ¬ signBit ≤ b := by omega
  unfold orderKey
  simp only [hb, Nat.not_le.mpr hb', if_false]
  omega

/-- `-0.0` and `+0.0` get the same key, as numpy treats them as equal. -/
theorem orderKey_negZero : orderKey (normZero signBit) = orderKey (normZero 0) := by
  decide

-- § counts

/-- Number of keys strictly above `t`. -/
def cgt : List Nat → Nat → Nat
  | [], _ => 0
  | k :: ks, t => (if t < k then 1 else 0) + cgt ks t

/-- Number of keys equal to `t`. -/
def ceq : List Nat → Nat → Nat
  | [], _ => 0
  | k :: ks, t => (if k = t then 1 else 0) + ceq ks t

/-- `count(key >= v)`, the predicate the binary lifting tests. -/
def cntGe : List Nat → Nat → Nat
  | [], _ => 0
  | k :: ks, v => (if v ≤ k then 1 else 0) + cntGe ks v

/-- `count(key ≥ v)` is antitone in `v`: this is what makes binary lifting correct. -/
theorem cntGe_antitone (ks : List Nat) {v v' : Nat} (h : v ≤ v') : cntGe ks v' ≤ cntGe ks v := by
  induction ks with
  | nil => simp [cntGe]
  | cons k ks ih =>
    simp only [cntGe]
    by_cases h1 : v' ≤ k
    · have h2 : v ≤ k := Nat.le_trans h h1
      simp only [h1, h2, if_true]
      omega
    · by_cases h2 : v ≤ k
      · simp only [h1, h2, if_true, if_false]
        omega
      · simp only [h1, h2, if_false]
        omega

/-- Strictly-above is `≥ t + 1`. -/
theorem cgt_eq_cntGe_succ (ks : List Nat) (t : Nat) : cgt ks t = cntGe ks (t + 1) := by
  induction ks with
  | nil => rfl
  | cons k ks ih =>
    simp only [cgt, cntGe, ih]
    by_cases h : t < k
    · have h' : t + 1 ≤ k := h
      simp [h, h']
    · have h' : ¬ t + 1 ≤ k := h
      simp [h, h']

/-- At-or-above splits into strictly-above plus equal. -/
theorem cntGe_split (ks : List Nat) (t : Nat) : cntGe ks t = cgt ks t + ceq ks t := by
  induction ks with
  | nil => rfl
  | cons k ks ih =>
    simp only [cntGe, cgt, ceq, ih]
    by_cases h1 : t < k
    · have h2 : t ≤ k := Nat.le_of_lt h1
      have h3 : k ≠ t := by omega
      simp only [h1, h2, h3, if_true, if_false]
      omega
    · by_cases h3 : k = t
      · subst h3
        simp
        omega
      · have h2 : ¬ t ≤ k := by omega
        simp only [h1, h2, h3, if_false]
        omega

-- § the ascending emission walk

/-- One ascending walk: emit index `j` for keys above `thr`, and for keys equal to `thr` while the
remaining equal budget `e` is positive. -/
def emit : List Nat → Nat → Nat → Nat → List Nat
  | [], _, _, _ => []
  | k :: ks, thr, e, j =>
    if thr < k then j :: emit ks thr e (j + 1)
    else if k = thr ∧ 0 < e then j :: emit ks thr (e - 1) (j + 1)
    else emit ks thr e (j + 1)

/-- The walk emits `c_gt + min(c_eq, budget)` ids. -/
theorem emit_length (ks : List Nat) (thr e j : Nat) :
    (emit ks thr e j).length = cgt ks thr + min (ceq ks thr) e := by
  induction ks generalizing e j with
  | nil => simp [emit, cgt, ceq]
  | cons k ks ih =>
    by_cases h1 : thr < k
    · have h2 : k ≠ thr := by omega
      simp only [emit, cgt, ceq, h1, h2, if_true, if_false, List.length_cons, ih]
      omega
    · by_cases h2 : k = thr
      · subst h2
        by_cases h3 : 0 < e
        · simp [emit, cgt, ceq, h3, ih] <;> omega
        · simp [emit, cgt, ceq, h3, ih] <;> omega
      · simp only [emit, cgt, ceq, h1, h2, false_and, if_false, ih]
        omega

/-- Every emitted id lies in `[j, j + length)`: ids are real cell indices. -/
theorem emit_bounds (ks : List Nat) (thr e j : Nat) :
    ∀ x ∈ emit ks thr e j, j ≤ x ∧ x < j + ks.length := by
  induction ks generalizing e j with
  | nil => simp [emit]
  | cons k ks ih =>
    intro x hx
    simp only [List.length_cons]
    unfold emit at hx
    split at hx
    · rcases List.mem_cons.mp hx with h | h
      · omega
      · have := ih e (j + 1) x h; omega
    · split at hx
      · rcases List.mem_cons.mp hx with h | h
        · omega
        · have := ih (e - 1) (j + 1) x h; omega
      · have := ih e (j + 1) x hx; omega

/-- The emitted ids are strictly ascending, so the tie rule "first by index" is the walk order. -/
theorem emit_sorted (ks : List Nat) (thr e j : Nat) : (emit ks thr e j).Pairwise (· < ·) := by
  induction ks generalizing e j with
  | nil => simp [emit]
  | cons k ks ih =>
    unfold emit
    split
    · refine List.pairwise_cons.mpr ⟨fun x hx => ?_, ih e (j + 1)⟩
      have := emit_bounds ks thr e (j + 1) x hx; omega
    · split
      · refine List.pairwise_cons.mpr ⟨fun x hx => ?_, ih (e - 1) (j + 1)⟩
        have := emit_bounds ks thr (e - 1) (j + 1) x hx; omega
      · exact ih e (j + 1)

/-- Given the binary-lifting postcondition on `thr`, the walk with `budget = width - c_gt` emits
exactly `width` ids. -/
theorem emit_exact_width (ks : List Nat) (thr w : Nat) (hge : w ≤ cntGe ks thr)
    (hlt : cntGe ks (thr + 1) < w) : (emit ks thr (w - cgt ks thr) 0).length = w := by
  have h1 := cgt_eq_cntGe_succ ks thr
  have h2 := cntGe_split ks thr
  rw [emit_length]
  omega

/-- The source's first (buggy) budget compared the LAYOUT position `c_gt + rank` to the budget; with
distinct scores (`c_eq = 1`, `c_gt = w - 1`) it is short by one whenever `c_gt > 0`. -/
theorem layout_budget_short (ks : List Nat) (thr w : Nat) (hgt : cgt ks thr = w - 1)
    (heq : ceq ks thr = 1) (hw : 2 ≤ w) :
    (emit ks thr (w - cgt ks thr - cgt ks thr) 0).length = w - 1 := by
  rw [emit_length]
  omega

end Strata.Topk
