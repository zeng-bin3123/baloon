-- Baloon.QsaCacheStep: per-token device step state, paged FP16 KV cache row mapping,
-- indexer block pooling schedule and cell_block / tail-bias mapping.
-- SOURCES: include/ref/kernels/qsa.hpp, src/kernels/cuda/qsa.cu
-- Unit: u316bd6b9ec

namespace Baloon.QsaCacheStep

theorem decode_div_mod (a b n : Nat) (hb : b < n) : (a*n+b)/n = a ∧ (a*n+b)%n = b := by
  have hn : 0 < n := by omega
  have swap : a * n + b = b + a * n := by ring
  constructor
  · rw [swap]
    have h := Nat.add_mul_div_right b a hn
    rw [Nat.div_eq_of_lt hb] at h
    simpa using h
  · rw [swap, Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt hb]

theorem idx2_lt (a b A B : Nat) (ha : a < A) (hb : b < B) : a*B+b < A*B := by
  calc a * B + b < a * B + B := by omega
    _ = (a + 1) * B := by ring
    _ ≤ A * B := Nat.mul_le_mul_right B (by omega)

theorem div_mul_bounds (n r : Nat) (hr : 0 < r) : (n/r)*r ≤ n ∧ n < (n/r+1)*r := by
  have eqn := Nat.div_add_mod n r
  have bnd := Nat.mod_lt n hr
  have expand : (n / r + 1) * r = n / r * r + r := Nat.succ_mul (n / r) r
  rw [Nat.mul_comm r (n / r)] at eqn
  constructor <;> omega

/-- Selection width as min of n_kv and the capacity top_k + r - 1. -/
def selWidth (nKv topK r : Nat) : Nat := min nKv (topK + r - 1)

/-- Step buffer: pos, n_kv = pos+1, n_bid = n_kv/r, width = selWidth. -/
structure Step where
  pos   : Nat := 0
  nKv   : Nat := 0
  nBid  : Nat := 0
  width : Nat := 0

/-- Fill all four entries from one position using anonymous constructor. -/
def stepFill (pos topK r : Nat) : Step :=
  ⟨pos, pos + 1, (pos + 1) / r, selWidth (pos + 1) topK r⟩

theorem selWidth_le (n k r : Nat) : selWidth n k r ≤ n := by
  simp only [selWidth]; exact Nat.min_le_left n _

theorem selWidth_le_budget (n k r : Nat) : selWidth n k r ≤ k + r - 1 := by
  simp only [selWidth]; exact Nat.min_le_right _ _

theorem selWidth_identity (n k r : Nat) (h : n < k + r - 1) : selWidth n k r = n :=
  Nat.min_eq_left (by omega)

theorem selWidth_real (n : Nat) : selWidth n 2048 4 = min n 2051 := rfl

theorem selWidth_max_cap : selWidth 32768 2048 4 = 2051 := by
  simp [selWidth, Nat.min_def]

theorem attend_smem_fits : (2051 + 32) * 4 ≤ 49152 := by norm_num

theorem step_bytes : 4 * 4 = 16 := by rfl

theorem stepFill_consistent (pos topK r : Nat) (hr : 0 < r) :
    (stepFill pos topK r).nKv = (stepFill pos topK r).pos + 1 ∧
    (stepFill pos topK r).nBid * r ≤ (stepFill pos topK r).nKv ∧
    (stepFill pos topK r).nKv < ((stepFill pos topK r).nBid + 1) * r ∧
    (stepFill pos topK r).width ≤ (stepFill pos topK r).nKv := by
  obtain ⟨h1, h2⟩ := div_mul_bounds (pos+1) r hr
  refine ⟨rfl, h1, h2, ?_⟩
  exact selWidth_le (pos+1) topK r

/-- Physical KV row using a let binding for the intermediate index. -/
def kvRow (page H h P o : Nat) : Nat :=
  let mid := page * H + h
  mid * P + o

theorem kvRow_lt (page nPages H h P o : Nat) (hp : page < nPages) (hh : h < H) (ho : o < P) :
    kvRow page H h P o < nPages * H * P := by
  simp only [kvRow]
  have h1 : page * H + h < nPages * H := idx2_lt page h nPages H hp hh
  exact idx2_lt (page * H + h) o (nPages * H) P h1 ho

theorem kvRow_injective (p1 p2 H h1 h2 P o1 o2 : Nat) (hh1 : h1 < H) (hh2 : h2 < H)
    (ho1 : o1 < P) (ho2 : o2 < P) (heq : kvRow p1 H h1 P o1 = kvRow p2 H h2 P o2) :
    p1 = p2 ∧ h1 = h2 ∧ o1 = o2 := by
  simp only [kvRow] at heq
  obtain ⟨d1, m1⟩ := decode_div_mod (p1*H+h1) o1 P ho1
  obtain ⟨d2, m2⟩ := decode_div_mod (p2*H+h2) o2 P ho2
  rw [heq] at d1 m1
  have e1 : p1*H+h1 = p2*H+h2 := d1.symm.trans d2
  have e2 : o1 = o2 := m1.symm.trans m2
  obtain ⟨a1, b1⟩ := decode_div_mod p1 h1 H hh1
  obtain ⟨a2, b2⟩ := decode_div_mod p2 h2 H hh2
  rw [e1] at a1 b1
  exact ⟨a1.symm.trans a2, b1.symm.trans b2, e2⟩

theorem page_index_lt (pos P nPages : Nat) (hP : 0 < P) (hpos : pos < nPages * P) :
    pos / P < nPages := by
  exact (Nat.div_lt_iff_lt_mul hP).mpr hpos

theorem block_completes_iff (pos : Nat) : pos % 4 = 3 ↔ (pos + 1) % 4 = 0 :=
  Iff.intro (by omega) (by omega)

theorem nbid_grows_on_completion (pos : Nat) (h : pos % 4 = 3) : (pos + 1) / 4 = pos / 4 + 1 := by
  have h4 : (pos + 1) % 4 = 0 := by omega
  omega

theorem nbid_stable_off_completion (pos : Nat) (h : pos % 4 ≠ 3) : (pos + 1) / 4 = pos / 4 := by
  have h4 : (pos + 1) % 4 ≠ 0 := by omega
  omega

theorem tail_slot_in_range (pos : Nat) (h : pos % 4 ≠ 3) : pos % 4 < 3 := by
  have hlt : pos % 4 ≤ 3 := by omega
  omega

theorem block_first_cell (pos r : Nat) (hr : 0 < r) :
    (pos / r) * r ≤ pos ∧ pos < (pos / r) * r + r := by
  obtain ⟨h1, h2⟩ := div_mul_bounds pos r hr
  exact ⟨h1, by linarith [h2, (by ring : (pos / r + 1) * r = (pos / r) * r + r)]⟩

theorem cell_block_eq (j nKv r : Nat) (hj : j < nKv) : min (j / r) (nKv / r) = j / r := by
  apply Nat.min_eq_left
  exact Nat.div_le_div_right (Nat.le_of_lt hj)

theorem tail_cell_iff (j nKv r : Nat) (hr : 0 < r) (hj : j < nKv) :
    (nKv / r) * r ≤ j ↔ j / r = nKv / r :=
  Iff.intro
    (fun h => by
      have h1 : nKv / r ≤ j / r := (Nat.le_div_iff_mul_le hr).2 h
      have h2 : j / r ≤ nKv / r := Nat.div_le_div_right (Nat.le_of_lt hj)
      omega)
    (fun h => by
      have h3 := Nat.div_mul_le_self j r
      rw [h] at h3; exact h3)

theorem tail_exists_iff (n : Nat) : n % 4 ≠ 0 ↔ (n / 4) * 4 < n :=
  Iff.intro (by omega) (by omega)

end Baloon.QsaCacheStep
