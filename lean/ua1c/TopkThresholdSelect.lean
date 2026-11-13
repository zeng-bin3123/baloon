-- TopkThresholdSelect.lean
-- Formal model of src/kernels/cuda/qsa.cu and qsa_select.cu top-k threshold selection.
namespace Baloon.Topk

def signBit : Nat := 2147483648

def orderKey (b : Nat) : Nat := if signBit ≤ b then 4294967295 - b else b + signBit

def normZero (b : Nat) : Nat := if b = signBit then 0 else b

theorem orderKey_lt (b : Nat) (hb : b < 4294967296) : orderKey b < 4294967296 := by
  unfold orderKey signBit; split <;> omega

theorem orderKey_injective (b b' : Nat) (hb : b < 4294967296) (hb' : b' < 4294967296)
    (h : orderKey b = orderKey b') : b = b' := by
  unfold orderKey signBit at h; split at h <;> split at h <;> omega

theorem orderKey_neg_lt_nonneg (b b' : Nat) (hb : signBit ≤ b) (hb4 : b < 4294967296)
    (hb' : b' < signBit) : orderKey b < orderKey b' := by
  unfold orderKey; simp only [hb, if_true, Nat.not_le.mpr hb', if_false]; unfold signBit at *; omega

theorem orderKey_neg_antitone (b b' : Nat) (hb : signBit ≤ b) (hbb : b < b')
    (hb4 : b' < 4294967296) : orderKey b' < orderKey b := by
  have hb' : signBit ≤ b' := Nat.le_trans hb (Nat.le_of_lt hbb)
  unfold orderKey; simp only [hb, hb', if_true]; omega

theorem orderKey_pos_monotone (b b' : Nat) (hbb : b < b') (hb' : b' < signBit) :
    orderKey b < orderKey b' := by
  have hb : ¬ signBit ≤ b := by omega
  unfold orderKey; simp only [hb, Nat.not_le.mpr hb', if_false]; omega

theorem orderKey_negZero : orderKey (normZero signBit) = orderKey (normZero 0) := by decide

def cgt : List Nat → Nat → Nat
  | [], _ => 0
  | k :: ks, t => (if t < k then 1 else 0) + cgt ks t

def ceq : List Nat → Nat → Nat
  | [], _ => 0
  | k :: ks, t => (if k = t then 1 else 0) + ceq ks t

def cntGe : List Nat → Nat → Nat
  | [], _ => 0
  | k :: ks, v => (if v ≤ k then 1 else 0) + cntGe ks v

theorem cntGe_antitone (ks : List Nat) {v v' : Nat} (h : v ≤ v') : cntGe ks v' ≤ cntGe ks v := by
  induction ks with
  | nil => simp [cntGe]
  | cons k ks ih =>
    simp only [cntGe]
    by_cases h1 : v' ≤ k
    · have h2 : v ≤ k := Nat.le_trans h h1; simp only [h1, h2, if_true]; omega
    · by_cases h2 : v ≤ k
      · simp only [h1, h2, if_true, if_false]; omega
      · simp only [h1, h2, if_false]; omega

theorem cgt_eq_cntGe_succ (ks : List Nat) (t : Nat) : cgt ks t = cntGe ks (t + 1) := by
  induction ks with
  | nil => rfl
  | cons k ks ih =>
    simp only [cgt, cntGe, ih]
    by_cases h : t < k
    · have h' : t + 1 ≤ k := h; simp [h, h']
    · have h' : ¬ t + 1 ≤ k := h; simp [h, h']

theorem cntGe_split (ks : List Nat) (t : Nat) : cntGe ks t = cgt ks t + ceq ks t := by
  induction ks with
  | nil => rfl
  | cons k ks ih =>
    simp only [cntGe, cgt, ceq, ih]
    by_cases h1 : t < k
    · have h2 : t ≤ k := Nat.le_of_lt h1; have h3 : k ≠ t := by omega
      simp only [h1, h2, h3, if_true, if_false]; omega
    · by_cases h3 : k = t
      · subst h3; simp; omega
      · have h2 : ¬ t ≤ k := by omega; simp only [h1, h2, h3, if_false]; omega

def emit : List Nat → Nat → Nat → Nat → List Nat
  | [], _, _, _ => []
  | k :: ks, thr, e, j =>
    if thr < k then j :: emit ks thr e (j + 1)
    else if k = thr ∧ 0 < e then j :: emit ks thr (e - 1) (j + 1)
    else emit ks thr e (j + 1)

theorem emit_length (ks : List Nat) (thr e j : Nat) :
    (emit ks thr e j).length = cgt ks thr + min (ceq ks thr) e := by
  induction ks generalizing e j with
  | nil => simp [emit, cgt, ceq]
  | cons k ks ih =>
    by_cases h1 : thr < k
    · have h2 : k ≠ thr := by omega
      simp only [emit, cgt, ceq, h1, h2, if_true, if_false, List.length_cons, ih]; omega
    · by_cases h2 : k = thr
      · subst h2
        by_cases h3 : 0 < e
        · simp [emit, cgt, ceq, h3, ih] <;> omega
        · simp [emit, cgt, ceq, h3, ih] <;> omega
      · simp only [emit, cgt, ceq, h1, h2, false_and, if_false, ih]; omega

theorem emit_bounds (ks : List Nat) (thr e j : Nat) :
    ∀ x ∈ emit ks thr e j, j ≤ x ∧ x < j + ks.length := by
  induction ks generalizing e j with
  | nil => simp [emit]
  | cons k ks ih =>
    intro x hx; simp only [List.length_cons]
    unfold emit at hx; split at hx
    · rcases List.mem_cons.mp hx with h | h
      · omega
      · have := ih e (j + 1) x h; omega
    · split at hx
      · rcases List.mem_cons.mp hx with h | h
        · omega
        · have := ih (e - 1) (j + 1) x h; omega
      · have := ih e (j + 1) x hx; omega

theorem emit_sorted (ks : List Nat) (thr e j : Nat) : (emit ks thr e j).Pairwise (· < ·) := by
  induction ks generalizing e j with
  | nil => simp [emit]
  | cons k ks ih =>
    unfold emit; split
    · refine List.pairwise_cons.mpr ⟨fun x hx => ?_, ih e (j + 1)⟩
      have := emit_bounds ks thr e (j + 1) x hx; omega
    · split
      · refine List.pairwise_cons.mpr ⟨fun x hx => ?_, ih (e - 1) (j + 1)⟩
        have := emit_bounds ks thr (e - 1) (j + 1) x hx; omega
      · exact ih e (j + 1)

theorem emit_exact_width (ks : List Nat) (thr w : Nat) (hge : w ≤ cntGe ks thr)
    (hlt : cntGe ks (thr + 1) < w) : (emit ks thr (w - cgt ks thr) 0).length = w := by
  have h1 := cgt_eq_cntGe_succ ks thr
  have h2 := cntGe_split ks thr
  rw [emit_length]; omega

theorem layout_budget_short (ks : List Nat) (thr w : Nat) (hgt : cgt ks thr = w - 1)
    (heq : ceq ks thr = 1) (hw : 2 ≤ w) :
    (emit ks thr (w - cgt ks thr - cgt ks thr) 0).length = w - 1 := by
  rw [emit_length]; omega

end Baloon.Topk
