/-!
# MODULE: qsa_cache_step
SOURCES: include/strata/kernels/qsa.hpp, src/kernels/cuda/qsa.cu
Per-token device step state (pos, n_kv, n_bid, width), the paged FP16 KV cache row mapping, the indexer's
block pooling schedule and the cell_block / tail-bias mapping of the QSA decode layer.

## Source semantics
* `qsa_step_fill` derives n_kv = pos+1, n_bid = n_kv / r, width = min(n_kv, top_k + r - 1) from `pos` alone.
* Below the selection bound the selection is the identity.
* Physical KV row = (page * n_head_kv + h) * page_size + (t % page_size); it must be in range and injective.
* A block completes exactly when pos % r = r-1; the tail holds r-1 raw rows.
* Cell j maps to pooled row min(j / r, n_bid); cells of the incomplete tail get the +1e9 bias.

## Form template identifiers
step buffer -> `Step`; selection width -> `selWidth`; page/row mapping -> `kvRow`; QsaShapes.idx_block -> `r`.
-/

namespace QsaCacheStep

-- § arithmetic helpers

/-- Decoding a mixed-radix pair: `(a*n+b)/n = a` and `(a*n+b)%n = b` when `b < n`. -/
theorem decode_div_mod (a b n : Nat) (hb : b < n) : (a*n+b)/n = a ∧ (a*n+b)%n = b := by
  have hn : 0 < n := by omega
  have e : a*n+b = b + a*n := Nat.add_comm _ _
  have h1 := Nat.add_mul_div_right b a hn
  rw [Nat.div_eq_of_lt hb] at h1
  constructor
  · rw [e]; omega
  · rw [e, Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt hb]

/-- Row-major index bound. -/
theorem idx2_lt (a b A B : Nat) (ha : a < A) (hb : b < B) : a*B+b < A*B := by
  have h1 : (a+1)*B ≤ A*B := Nat.mul_le_mul_right B (by omega)
  rw [Nat.add_mul, Nat.one_mul] at h1
  omega

/-- Quotient/remainder bracket for the block count. -/
theorem div_mul_bounds (n r : Nat) (hr : 0 < r) : (n/r)*r ≤ n ∧ n < (n/r+1)*r := by
  have hd := Nat.div_add_mod n r
  have hm := Nat.mod_lt n hr
  have e : (n/r+1)*r = (n/r)*r + r := by rw [Nat.add_mul, Nat.one_mul]
  rw [Nat.mul_comm r (n/r)] at hd
  constructor <;> omega

-- § selection width and step state

/-- `qsa_selection_width`: min(n_kv, top_k + r - 1). -/
def selWidth (nKv topK r : Nat) : Nat := if nKv < topK + r - 1 then nKv else topK + r - 1

/-- The step buffer contents. -/
structure Step where
  pos : Nat
  nKv : Nat
  nBid : Nat
  width : Nat

/-- `qsa_step_fill`: all four entries from one position. -/
def stepFill (pos topK r : Nat) : Step :=
  { pos := pos, nKv := pos + 1, nBid := (pos + 1) / r, width := selWidth (pos + 1) topK r }

/-- The selection never exceeds the number of cached cells. -/
theorem selWidth_le (n k r : Nat) : selWidth n k r ≤ n := by
  unfold selWidth; split <;> omega

/-- The selection never exceeds the budget `top_k + r - 1`. -/
theorem selWidth_le_budget (n k r : Nat) : selWidth n k r ≤ k + r - 1 := by
  unfold selWidth; split <;> omega

/-- Below the bound the selection is the identity (all cells). -/
theorem selWidth_identity (n k r : Nat) (h : n < k + r - 1) : selWidth n k r = n := by
  unfold selWidth; simp [h]

/-- For the real geometry the width is `min n 2051`. -/
theorem selWidth_real (n : Nat) : selWidth n 2048 4 = min n 2051 := by
  unfold selWidth; split <;> omega

/-- The largest selection width the one-block top-k can ever write is 2051 (the required `cap`). -/
theorem selWidth_max_cap : selWidth 32768 2048 4 = 2051 := by decide

/-- The attention shared buffer for the largest selection fits the default 48 KiB shared window. -/
theorem attend_smem_fits : (2051 + 32) * 4 ≤ 49152 := by decide

/-- The step buffer is four int32 = 16 bytes. -/
theorem step_bytes : 4 * 4 = 16 := by decide

/-- The four step entries are mutually consistent by construction. -/
theorem stepFill_consistent (pos topK r : Nat) (hr : 0 < r) :
    (stepFill pos topK r).nKv = (stepFill pos topK r).pos + 1 ∧
    (stepFill pos topK r).nBid * r ≤ (stepFill pos topK r).nKv ∧
    (stepFill pos topK r).nKv < ((stepFill pos topK r).nBid + 1) * r ∧
    (stepFill pos topK r).width ≤ (stepFill pos topK r).nKv := by
  obtain ⟨h1, h2⟩ := div_mul_bounds (pos+1) r hr
  have h3 := selWidth_le (pos+1) topK r
  exact ⟨rfl, h1, h2, h3⟩

-- § paged KV cache

/-- Physical row of logical cell offset `o` in a page for kv head `h`. -/
def kvRow (page H h P o : Nat) : Nat := (page * H + h) * P + o

/-- A physical row stays inside the pool. -/
theorem kvRow_lt (page nPages H h P o : Nat) (hp : page < nPages) (hh : h < H) (ho : o < P) :
    kvRow page H h P o < nPages * H * P := by
  have h1 : page * H + h < nPages * H := idx2_lt page h nPages H hp hh
  exact idx2_lt (page * H + h) o (nPages * H) P h1 ho

/-- Distinct (page, head, slot) triples give distinct physical rows. -/
theorem kvRow_injective (p1 p2 H h1 h2 P o1 o2 : Nat) (hh1 : h1 < H) (hh2 : h2 < H)
    (ho1 : o1 < P) (ho2 : o2 < P) (heq : kvRow p1 H h1 P o1 = kvRow p2 H h2 P o2) :
    p1 = p2 ∧ h1 = h2 ∧ o1 = o2 := by
  unfold kvRow at heq
  obtain ⟨d1, m1⟩ := decode_div_mod (p1*H+h1) o1 P ho1
  obtain ⟨d2, m2⟩ := decode_div_mod (p2*H+h2) o2 P ho2
  rw [heq] at d1 m1
  have e1 : p1*H+h1 = p2*H+h2 := d1.symm.trans d2
  have e2 : o1 = o2 := m1.symm.trans m2
  obtain ⟨a1, b1⟩ := decode_div_mod p1 h1 H hh1
  obtain ⟨a2, b2⟩ := decode_div_mod p2 h2 H hh2
  rw [e1] at a1 b1
  exact ⟨a1.symm.trans a2, b1.symm.trans b2, e2⟩

/-- The page-table index of a cell is in range whenever the cell is in the pool. -/
theorem page_index_lt (pos P nPages : Nat) (hP : 0 < P) (hpos : pos < nPages * P) :
    pos / P < nPages := (Nat.div_lt_iff_lt_mul hP).2 hpos

-- § indexer pooling schedule and cell_block

/-- A block completes exactly when `pos % 4 = 3`, i.e. when `(pos+1) % 4 = 0`. -/
theorem block_completes_iff (pos : Nat) : pos % 4 = 3 ↔ (pos + 1) % 4 = 0 := by omega

/-- On completion the completed-block count grows by exactly one. -/
theorem nbid_grows_on_completion (pos : Nat) (h : pos % 4 = 3) : (pos + 1) / 4 = pos / 4 + 1 := by
  omega

/-- Off completion the completed-block count is unchanged. -/
theorem nbid_stable_off_completion (pos : Nat) (h : pos % 4 ≠ 3) : (pos + 1) / 4 = pos / 4 := by
  omega

/-- Only slots 0..r-2 are stored in the raw tail: three rows suffice for r = 4. -/
theorem tail_slot_in_range (pos : Nat) (h : pos % 4 ≠ 3) : pos % 4 < 3 := by omega

/-- The block's first cell does not exceed `pos` and is within `r` cells of it. -/
theorem block_first_cell (pos r : Nat) (hr : 0 < r) :
    (pos / r) * r ≤ pos ∧ pos < (pos / r) * r + r := by
  obtain ⟨h1, h2⟩ := div_mul_bounds pos r hr
  have e : (pos/r+1)*r = (pos/r)*r + r := by rw [Nat.add_mul, Nat.one_mul]
  omega

/-- `cell_block`: a cached cell never maps past the spare slot row `n_bid`. -/
theorem cell_block_eq (j nKv r : Nat) (hj : j < nKv) : min (j / r) (nKv / r) = j / r := by
  have h : j / r ≤ nKv / r := Nat.div_le_div_right (Nat.le_of_lt hj)
  omega

/-- The +1e9 tail bias applies exactly to cells that map onto the spare row. -/
theorem tail_cell_iff (j nKv r : Nat) (hr : 0 < r) (hj : j < nKv) :
    (nKv / r) * r ≤ j ↔ j / r = nKv / r := by
  constructor
  · intro h
    have h1 : nKv / r ≤ j / r := (Nat.le_div_iff_mul_le hr).2 h
    have h2 : j / r ≤ nKv / r := Nat.div_le_div_right (Nat.le_of_lt hj)
    omega
  · intro h
    have h3 := Nat.div_mul_le_self j r
    rw [h] at h3
    exact h3

/-- A completed block has no tail bias: tail cells exist only when `n_kv % r ≠ 0` (r = 4). -/
theorem tail_exists_iff (n : Nat) : n % 4 ≠ 0 ↔ (n / 4) * 4 < n := by omega

end QsaCacheStep
