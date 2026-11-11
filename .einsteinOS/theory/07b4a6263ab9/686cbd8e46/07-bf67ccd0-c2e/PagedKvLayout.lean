/-!
# MODULE: paged_kv_layout
SOURCES: src/kernels/cuda/qsa.cu, src/kernels/cuda/kv_q8.cu, src/kernels/cuda/qsa_decode_attn.cu
Index arithmetic of the paged KV pool `[page][kv_head][page_size][head_dim]` shared by `kv_append`,
`kv_gather`, `kv_append_q8`, `kv_gather_q8` and the decode attention's `srow`.

## Source semantics
* A cell at position `pos` of KV head `h` lives in pool row
  `(table[pos / page_size] * kv_heads + h) * page_size + pos % page_size`.
* Rows stay inside a pool of `n_pages * kv_heads * page_size` rows when the page table is in range,
  and distinct (page, head, slot) triples never share a row.
* The gather's flat thread index `i` decomposes as `((id * kv_heads + h) * per + q)` with
  `per = head_dim / 4`; threads with `i < n_ids * kv_heads * per` touch only ids `< n_ids`.

## Form template identifiers
`row`/`src` → `poolRow`, gather `dst` → `scratchIdx`, `i / (kv_heads*per)` → `gatherId`,
`page_table` → `table`, `kv_heads` → `K`, `page_size` → `ps`.
-/

namespace Strata.PagedKv

-- § helpers

/-- `a*c + b` with `b < c` stays below `n*c` whenever `a < n`. -/
theorem mul_add_lt {a b c n : Nat} (hb : b < c) (ha : a < n) : a * c + b < n * c := by
  have h1 : (a + 1) * c ≤ n * c := Nat.mul_le_mul_right c (Nat.succ_le_of_lt ha)
  rw [Nat.succ_mul] at h1
  omega

/-- Mixed-radix decompositions with digit `< c` are unique. -/
theorem decomp_unique {a b a' b' c : Nat} (hb : b < c) (hb' : b' < c)
    (h : a * c + b = a' * c + b') : a = a' ∧ b = b' := by
  rcases Nat.lt_trichotomy a a' with hlt | heq | hgt
  · have h1 : (a + 1) * c ≤ a' * c := Nat.mul_le_mul_right c (Nat.succ_le_of_lt hlt)
    rw [Nat.succ_mul] at h1
    omega
  · subst heq
    omega
  · have h1 : (a' + 1) * c ≤ a * c := Nat.mul_le_mul_right c (Nat.succ_le_of_lt hgt)
    rw [Nat.succ_mul] at h1
    omega

-- § pool rows

/-- The pool row of cell `pos` for KV head `h` (`kv_append_kernel`, `kv_gather_kernel`). -/
def poolRow (table : Nat → Nat) (K ps h pos : Nat) : Nat :=
  (table (pos / ps) * K + h) * ps + pos % ps

/-- With an in-range page table every row is inside the pool. -/
theorem poolRow_lt (table : Nat → Nat) (K ps h pos nPages : Nat) (hps : 0 < ps) (hh : h < K)
    (hpage : table (pos / ps) < nPages) : poolRow table K ps h pos < nPages * K * ps := by
  have h1 : table (pos / ps) * K + h < nPages * K := mul_add_lt hh hpage
  exact mul_add_lt (Nat.mod_lt pos hps) h1

/-- Distinct (page, head, slot) triples map to distinct rows. -/
theorem row_injective {p p' h h' s s' K ps : Nat} (hh : h < K) (hh' : h' < K) (hs : s < ps)
    (hs' : s' < ps) (heq : (p * K + h) * ps + s = (p' * K + h') * ps + s') :
    p = p' ∧ h = h' ∧ s = s' := by
  obtain ⟨h1, h2⟩ := decomp_unique hs hs' heq
  obtain ⟨h3, h4⟩ := decomp_unique hh hh' h1
  exact ⟨h3, h4, h2⟩

/-- Two positions of the same page and head never collide inside the page. -/
theorem poolRow_slot_injective (table : Nat → Nat) (K ps h pos pos' : Nat)
    (hpage : pos / ps = pos' / ps) (heq : poolRow table K ps h pos = poolRow table K ps h pos') :
    pos = pos' := by
  unfold poolRow at heq
  rw [hpage] at heq
  have hm : pos % ps = pos' % ps := by omega
  have e1 := Nat.div_add_mod pos ps
  have e2 := Nat.div_add_mod pos' ps
  rw [hpage, hm] at e1
  omega

-- § gather thread decomposition

/-- The selection entry a gather thread handles. -/
def gatherId (K per i : Nat) : Nat := i / (K * per)

/-- The scratch element a gather thread writes: `(id * K + h) * per + q`. -/
def scratchIdx (K per i : Nat) : Nat :=
  (i / (K * per) * K + i % (K * per) / per) * per + i % (K * per) % per

/-- Every gather thread writes exactly its own flat index: the scratch write is a bijection. -/
theorem scratchIdx_eq (K per i : Nat) : scratchIdx K per i = i := by
  unfold scratchIdx
  have e1 := Nat.div_add_mod i (K * per)
  have e2 := Nat.div_add_mod (i % (K * per)) per
  rw [Nat.add_mul, Nat.mul_assoc, Nat.mul_comm (i / (K * per)) (K * per),
    Nat.mul_comm (i % (K * per) / per) per]
  omega

/-- Threads inside `n_ids * K * per` only read selection entries `< n_ids`. -/
theorem gatherId_lt (K per i nIds : Nat) (hK : 0 < K * per) (hi : i < nIds * (K * per)) :
    gatherId K per i < nIds := by
  unfold gatherId
  exact (Nat.div_lt_iff_lt_mul hK).mpr hi

/-- The head index of a gather thread is a valid KV head. -/
theorem gather_head_lt (K per i : Nat) (hK : 0 < K * per) : i % (K * per) / per < K := by
  have hm := Nat.mod_lt i hK
  have hper : 0 < per := Nat.pos_of_mul_pos_left hK
  exact (Nat.div_lt_iff_lt_mul hper).mpr hm

end Strata.PagedKv
