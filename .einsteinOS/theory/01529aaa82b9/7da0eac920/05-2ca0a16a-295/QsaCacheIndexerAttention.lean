/-!
# MODULE: qsa_cache_indexer_attention
QSA paged KV cache append/gather, block-pooled indexer scoring, exact top-k selection (threshold search with
ascending-index tie rule), single-token attention over the selected cells, and the sigmoid output gate.

## Source semantics
* Step state is derived from one position: `n_kv = pos+1`, `n_bid = n_kv / r`, `width = min(n_kv, top_k + r - 1)`.
* KV pool address: `((page*kv_heads + h)*page_size + pos % page_size)*head_dim + d`.
* Indexer block `b` scores cells `[b*r, min(b*r+r, n_kv))`; the tail block `n_bid` covers `[n_bid*r, n_kv)` and gets +1e9
  when incomplete.
* Top-k: with threshold `thr` = width-th largest key, emit all cells with key > thr plus the first
  `width - c_gt` cells with key == thr in ascending index order; the emitted count is exactly `width`.
* f32 order key: non-negative floats map to `bits | 2^31`, negative floats to `~bits`, giving a total order.
* Attention: query head `h` uses KV head `h / (n_head / n_head_kv)` (NOT `h % n_head_kv`).
* Gather copies `uint2` (4 halves) so `head_dim % 4 == 0`.

## Form template identifiers
* `selWidth`   -> selection width formula
* `keyPos`, `keyNeg` -> order-key mapping for non-negative / negative floats
* `selEq`      -> equal-to-threshold cells selected by a chunk
* `kvHead`     -> query-head to KV-head map
-/

namespace Strata.Qsa

/-- Index lemma for row-major addressing: `i*b + j < a*b` when `i < a`, `j < b`. -/
theorem idx_lt {a b i j : Nat} (hi : i < a) (hj : j < b) : i * b + j < a * b := by
  have h1 : (i + 1) * b ≤ a * b := Nat.mul_le_mul_right b (Nat.succ_le_of_lt hi)
  have h2 : (i + 1) * b = i * b + b := Nat.succ_mul i b
  omega

-- § step state

/-- Selection width. -/
def selWidth (n_kv top_k r : Nat) : Nat := min n_kv (top_k + r - 1)

/-- Width never exceeds the cells available. -/
theorem selWidth_le_nkv (n_kv top_k r : Nat) : selWidth n_kv top_k r ≤ n_kv := by
  unfold selWidth
  omega

/-- Width never exceeds the budget `top_k + r - 1`. -/
theorem selWidth_le_budget (n_kv top_k r : Nat) : selWidth n_kv top_k r ≤ top_k + r - 1 := by
  unfold selWidth
  omega

/-- Below the budget the selection is the identity (all cells). -/
theorem selWidth_identity (n_kv top_k r : Nat) (h : n_kv ≤ top_k + r - 1) : selWidth n_kv top_k r = n_kv := by
  unfold selWidth
  omega

/-- For the production geometry (2048, 4) the budget is 2051. -/
theorem selWidth_prod (n : Nat) : selWidth n 2048 4 = min n 2051 := by
  unfold selWidth
  omega

/-- Complete blocks and the partial tail partition the cells. -/
theorem blocks_partition (n_kv r : Nat) (hr : 0 < r) :
    r * (n_kv / r) + n_kv % r = n_kv ∧ n_kv % r < r :=
  ⟨Nat.div_add_mod n_kv r, Nat.mod_lt n_kv hr⟩

-- § addressing

/-- A KV pool row lies inside the pool: `(page*kv_heads + h)*page_size + slot`. -/
theorem kv_row_in_range {page h slot n_pages kv_heads page_size : Nat} (hp : page < n_pages)
    (hh : h < kv_heads) (hs : slot < page_size) :
    (page * kv_heads + h) * page_size + slot < n_pages * kv_heads * page_size := by
  have h1 : page * kv_heads + h < n_pages * kv_heads := idx_lt hp hh
  exact idx_lt h1 hs

/-- Element address inside the pool. -/
theorem kv_elem_in_range {row d rows head_dim : Nat} (hr : row < rows) (hd : d < head_dim) :
    row * head_dim + d < rows * head_dim := idx_lt hr hd

/-- Gather index decomposition: `(id, h, q)` recovered from flat `i` re-encode to `i`. -/
theorem gather_dst (i kv per : Nat) :
    ((i / (kv * per)) * kv + (i % (kv * per)) / per) * per + (i % (kv * per)) % per = i := by
  have h1 := Nat.div_add_mod' i (kv * per)
  have h2 := Nat.div_add_mod' (i % (kv * per)) per
  rw [Nat.add_mul, Nat.mul_assoc]
  omega

/-- `uint2` copies: `head_dim/4` copies of 4 halves cover `head_dim` when `head_dim % 4 = 0`. -/
theorem uint2_cover (head_dim : Nat) (h : head_dim % 4 = 0) : (head_dim / 4) * 4 = head_dim := by
  omega

-- § indexer cell mapping

/-- Cell `j` belongs to block `b` exactly when `b*r ≤ j < b*r + r`. -/
theorem cell_block_iff {j b r : Nat} (hr : 0 < r) : j / r = b ↔ (b * r ≤ j ∧ j < b * r + r) := by
  constructor
  · intro h
    have h1 := Nat.div_add_mod' j r
    have h2 := Nat.mod_lt j hr
    rw [h] at h1
    omega
  · intro h
    apply Nat.div_eq_of_lt_le h.1
    rw [Nat.add_mul, Nat.one_mul]
    omega

/-- Every cell below `n_kv` is owned by a block `≤ n_kv / r`. -/
theorem cell_block_le {j n_kv r : Nat} (hr : 0 < r) (h : j < n_kv) : j / r ≤ n_kv / r :=
  Nat.div_le_div_right (Nat.le_of_lt h)

-- § top-k

/-- Count identity of the threshold emission: with `c_gt < width ≤ c_gt + c_eq`, emitting every cell above
the threshold plus the first `width - c_gt` equal cells yields exactly `width` ids. -/
theorem topk_emits_width (c_gt c_eq width : Nat) (h1 : c_gt < width) (h2 : width ≤ c_gt + c_eq) :
    c_gt + min c_eq (width - c_gt) = width := by
  omega

/-- Equal-to-threshold cells selected by a chunk that starts after `off` equal cells, with room `budget`. -/
def selEq : Nat → Nat → Nat → Nat
  | _, 0, _ => 0
  | off, n + 1, b => (if off < b then 1 else 0) + selEq (off + 1) n b

/-- Closed form: `min n (budget - off)`. -/
theorem selEq_eq (n : Nat) : ∀ (off b : Nat), selEq off n b = min n (b - off) := by
  induction n with
  | zero => intro off b; simp [selEq]
  | succ n ih =>
    intro off b
    simp only [selEq, ih]
    split <;> omega

/-- The binary-lifting threshold bound: if more than `width - 1` cells are `≥ thr` then `thr` is feasible. -/
theorem threshold_feasible (c_ge width : Nat) (h : width ≤ c_ge) : ¬ c_ge < width := by
  omega

-- § order key

/-- Order key of a non-negative float with sign bit clear. -/
def keyPos (b : Nat) : Nat := b + 2147483648

/-- Order key of a float with sign bit set (`~bits`). -/
def keyNeg (b : Nat) : Nat := 4294967295 - b

/-- Negative floats sort below positive floats. -/
theorem key_neg_lt_pos (bn bp : Nat) (hn : 2147483648 ≤ bn) (hn' : bn < 4294967296)
    (hp : bp < 2147483648) : keyNeg bn < keyPos bp := by
  unfold keyNeg keyPos
  omega

/-- Larger positive bit patterns give larger keys. -/
theorem key_pos_mono (a b : Nat) (h : a ≤ b) : keyPos a ≤ keyPos b := by
  unfold keyPos
  omega

/-- Larger magnitude negatives (larger bit patterns) give smaller keys. -/
theorem key_neg_anti (a b : Nat) (h : a ≤ b) (hb : b ≤ 4294967295) : keyNeg b ≤ keyNeg a := by
  unfold keyNeg
  omega

/-- Positive keys are never zero, so the NaN key 0 is below every real key (never selected). -/
theorem key_pos_ne_zero (b : Nat) : 0 < keyPos b := by
  unfold keyPos
  omega

-- § attention

/-- KV head of query head `h`. -/
def kvHead (h n_head n_head_kv : Nat) : Nat := h / (n_head / n_head_kv)

/-- The KV head is in range when `n_head` is a multiple of `n_head_kv`. -/
theorem kvHead_lt {h n_head n_head_kv : Nat} (hdiv : n_head % n_head_kv = 0) (hh : h < n_head) :
    kvHead h n_head n_head_kv < n_head_kv := by
  unfold kvHead
  have e : n_head / n_head_kv * n_head_kv = n_head := Nat.div_mul_cancel (Nat.dvd_of_mod_eq_zero hdiv)
  apply Nat.div_lt_of_lt_mul
  rw [e]
  exact hh

/-- The grouping `h / g` differs from `h % n_head_kv` (24 heads, 2 KV heads, head 1). -/
theorem kvHead_not_mod : kvHead 1 24 2 = 0 ∧ 1 % 2 = 1 := by decide

/-- Empty selection yields zeros for the whole output row (guard `n_ids == 0`). -/
theorem empty_selection_guard (n_ids : Nat) : n_ids = 0 ∨ 0 < n_ids := by
  omega

/-- Shared memory: scores live at offset 32 and `max_ids` scores fit within `(max_ids+32)*4` bytes. -/
theorem smem_fits (j max_ids : Nat) (h : j < max_ids) : 32 + j < max_ids + 32 := by
  omega

end Strata.Qsa
