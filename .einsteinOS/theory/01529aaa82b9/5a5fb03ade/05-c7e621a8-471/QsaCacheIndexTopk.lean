/-!
# MODULE: qsa_cache_index_topk
The QSA paged KV cache, the pooled-key indexer and the exact threshold top-k selection
(`src/kernels/cuda/qsa.cu`): page/row addressing, the f32 order key, the ascending
"greater-than plus first equals" walk, the single-writer step state, and the gather decomposition.

## Source semantics
* A pool row is `[page][kv_head][page_size][head_dim]`, row = `(page*kvHeads + h)*pageSize + slot`.
* `order_key` is a total order on f32 bit patterns as unsigned ints (negative: `~b`, else `b | 2^31`).
* `topk_kernel` emits every cell strictly above the threshold key and the first
  `width - c_gt` cells equal to it, walking ascending, so the result has exactly `width` ids
  whenever `c_gt <= width <= c_gt + c_eq`.
* `qsa_step_fill` derives `n_kv = pos+1`, `n_bid = n_kv / r`, `width = min n_kv (topK + r - 1)` from one position.
* `kv_gather` splits a flat thread index into `(id, head, quad)` and writes `dst = i`.

## Form template identifiers
* `poolRow`      <- pool row index in `kv_append_kernel` / `kv_gather_kernel`
* `orderKey`     <- `order_key`
* `walk`         <- the emit loop of `topk_kernel`
* `stepFill`     <- `qsa_step_fill`
* `kvIndex`      <- `h / (n_head / n_head_kv)` (NOT `h % n_head_kv`)
-/

namespace QsaCacheIndexTopk

-- § pool addressing

/-- Row of a cell inside the paged pool. -/
def poolRow (page kvHeads h pageSize slot : Nat) : Nat :=
  (page * kvHeads + h) * pageSize + slot

/-- Every in-range (page, head, slot) maps to a row inside the pool. -/
theorem poolRow_lt (page kvHeads h pageSize slot nPages : Nat)
    (hp : page < nPages) (hh : h < kvHeads) (hs : slot < pageSize) :
    poolRow page kvHeads h pageSize slot < nPages * kvHeads * pageSize := by
  unfold poolRow
  have h0 : (page + 1) * kvHeads ≤ nPages * kvHeads := Nat.mul_le_mul_right _ hp
  have h0' : (page + 1) * kvHeads = page * kvHeads + kvHeads := by
    rw [Nat.add_mul, Nat.one_mul]
  have h1 : page * kvHeads + h + 1 ≤ nPages * kvHeads := by omega
  have h2 : (page * kvHeads + h + 1) * pageSize ≤ nPages * kvHeads * pageSize :=
    Nat.mul_le_mul_right _ h1
  have h2' : (page * kvHeads + h + 1) * pageSize = (page * kvHeads + h) * pageSize + pageSize := by
    rw [Nat.add_mul, Nat.one_mul]
  omega

/-- The query head to KV head map is a division by the group size, and lands in range. -/
theorem kvIndex_lt (nHead nHeadKv h : Nat) (hd : nHead % nHeadKv = 0) (hh : h < nHead) :
    h / (nHead / nHeadKv) < nHeadKv := by
  apply Nat.div_lt_of_lt_mul
  have e := Nat.div_add_mod nHead nHeadKv
  have c := Nat.mul_comm nHeadKv (nHead / nHeadKv)
  omega

-- § order key

/-- The unsigned order key of an f32 bit pattern `b < 2^32`. -/
def orderKey (b : Nat) : Nat :=
  if 2147483648 ≤ b then 4294967295 - b else b + 2147483648

/-- The key fits in 32 bits. -/
theorem orderKey_lt (b : Nat) (hb : b < 4294967296) : orderKey b < 4294967296 := by
  unfold orderKey
  split <;> omega

/-- Distinct bit patterns get distinct keys. -/
theorem orderKey_inj (a b : Nat) (ha : a < 4294967296) (hb : b < 4294967296)
    (h : orderKey a = orderKey b) : a = b := by
  unfold orderKey at h
  split at h <;> split at h <;> omega

/-- On non-negative floats the key is monotone. -/
theorem orderKey_mono_nonneg (a b : Nat) (hab : a ≤ b) (hb : b < 2147483648) :
    orderKey a ≤ orderKey b := by
  unfold orderKey
  split <;> split <;> omega

/-- On negative floats the key is antitone in the raw bits (bigger magnitude, smaller key). -/
theorem orderKey_anti_neg (a b : Nat) (hab : a ≤ b) (ha : 2147483648 ≤ a)
    (hb : b < 4294967296) : orderKey b ≤ orderKey a := by
  unfold orderKey
  split <;> omega

/-- Every negative key is below every non-negative key. -/
theorem orderKey_neg_lt_nonneg (a b : Nat) (ha : 2147483648 ≤ a) (ha2 : a < 4294967296)
    (hb : b < 2147483648) : orderKey a < orderKey b := by
  unfold orderKey
  split <;> split <;> omega

-- § the ascending walk

/-- Number of keys strictly above the threshold. -/
def countGt (thr : Nat) : List Nat → Nat
  | [] => 0
  | k :: ks => (if thr < k then 1 else 0) + countGt thr ks

/-- Number of keys equal to the threshold. -/
def countEq (thr : Nat) : List Nat → Nat
  | [] => 0
  | k :: ks => (if k = thr then 1 else 0) + countEq thr ks

/-- The emit loop: index `i`, `e` equal cells already taken, ascending. -/
def walk (thr budget : Nat) : Nat → Nat → List Nat → List Nat
  | _, _, [] => []
  | i, e, k :: ks =>
    if thr < k then i :: walk thr budget (i + 1) e ks
    else if k = thr ∧ e < budget then i :: walk thr budget (i + 1) (e + 1) ks
    else walk thr budget (i + 1) e ks

/-- The walk emits all strictly-greater cells plus as many equal cells as the budget allows. -/
theorem walk_length (thr budget : Nat) (ks : List Nat) :
    ∀ i e, (walk thr budget i e ks).length =
      countGt thr ks + min (countEq thr ks) (budget - e) := by
  induction ks with
  | nil => intro i e; simp [walk, countGt, countEq]
  | cons k ks ih =>
    intro i e
    by_cases h1 : thr < k
    · have h2 : ¬ k = thr := by omega
      simp [walk, countGt, countEq, h1, h2, ih] <;> omega
    · by_cases h2 : k = thr
      · by_cases h3 : e < budget
        · simp [walk, countGt, countEq, h1, h2, h3, ih] <;> omega
        · simp [walk, countGt, countEq, h1, h2, h3, ih] <;> omega
      · simp [walk, countGt, countEq, h1, h2, ih]

/-- With the right threshold the selection has exactly `width` ids. -/
theorem walk_length_width (thr width : Nat) (ks : List Nat) (i : Nat)
    (hgt : countGt thr ks ≤ width) (heq : width ≤ countGt thr ks + countEq thr ks) :
    (walk thr (width - countGt thr ks) i 0 ks).length = width := by
  rw [walk_length]
  omega

/-- The first kernel version compared `c_gt + prefix_eq` to the budget; with distinct scores
(`c_gt = width`, budget 0) that test can never succeed, which is exactly the observed bug. -/
theorem buggy_rank_never_selects (width cgt pe : Nat) (h : cgt = width) :
    ¬ (cgt + pe < width - cgt) := by
  omega

-- § step state

/-- The four per-token values, all derived from one position. -/
structure Step where
  pos : Nat
  nKv : Nat
  nBid : Nat
  width : Nat

/-- `qsa_step_fill`. -/
def stepFill (pos r topK : Nat) : Step :=
  ⟨pos, pos + 1, (pos + 1) / r, min (pos + 1) (topK + r - 1)⟩

/-- The selection never exceeds the cache. -/
theorem stepFill_width_le (pos r topK : Nat) :
    (stepFill pos r topK).width ≤ (stepFill pos r topK).nKv := by
  show min (pos + 1) (topK + r - 1) ≤ pos + 1
  omega

/-- Below the selection bound the selection is the identity. -/
theorem stepFill_identity (pos r topK : Nat) (h : pos + 1 ≤ topK + r - 1) :
    (stepFill pos r topK).width = (stepFill pos r topK).nKv := by
  show min (pos + 1) (topK + r - 1) = pos + 1
  omega

/-- Completed blocks fit inside the cache. -/
theorem stepFill_nBid_mul_le (pos r topK : Nat) :
    (stepFill pos r topK).nBid * r ≤ (stepFill pos r topK).nKv :=
  Nat.div_mul_le_self _ _

/-- The incomplete tail is shorter than one block. -/
theorem stepFill_nKv_lt (pos r topK : Nat) (hr : 0 < r) :
    (stepFill pos r topK).nKv < ((stepFill pos r topK).nBid + 1) * r := by
  show pos + 1 < ((pos + 1) / r + 1) * r
  have e := Nat.div_add_mod (pos + 1) r
  have m := Nat.mod_lt (pos + 1) hr
  have e2 : ((pos + 1) / r + 1) * r = r * ((pos + 1) / r) + r := by
    rw [Nat.add_mul, Nat.one_mul, Nat.mul_comm]
  omega

/-- For `idx_block = 4`, a cell completes a block exactly when the next count is a multiple of 4. -/
theorem block_completes_iff (pos : Nat) : pos % 4 = 3 ↔ (pos + 1) % 4 = 0 := by
  omega

-- § gather decomposition

/-- Arithmetic core of the gather index decomposition. -/
theorem gather_decomp_aux (i kv per A R B C : Nat) (e1 : kv * per * A + R = i)
    (e2 : per * B + C = R) : (A * kv + B) * per + C = i := by
  have e3 : (A * kv + B) * per = kv * per * A + per * B := by
    rw [Nat.add_mul, Nat.mul_assoc, Nat.mul_comm A (kv * per), Nat.mul_comm B per]
  omega

/-- A flat thread index is recovered from its `(id, head, quad)` split. -/
theorem gather_decomp (i kv per : Nat) :
    ((i / (kv * per)) * kv + (i % (kv * per)) / per) * per + (i % (kv * per)) % per = i :=
  gather_decomp_aux i kv per _ _ _ _ (Nat.div_add_mod i (kv * per))
    (Nat.div_add_mod (i % (kv * per)) per)

/-- The head and quad parts are in range. -/
theorem gather_parts_lt (i kv per : Nat) (hk : 0 < kv) (hp : 0 < per) :
    (i % (kv * per)) / per < kv ∧ (i % (kv * per)) % per < per := by
  have hm : 0 < kv * per := Nat.mul_pos hk hp
  have h1 : i % (kv * per) < kv * per := Nat.mod_lt _ hm
  have h2 : kv * per = per * kv := Nat.mul_comm kv per
  constructor
  · apply Nat.div_lt_of_lt_mul
    omega
  · exact Nat.mod_lt _ hp

end QsaCacheIndexTopk
