/-!
# MODULE: kv_cache_q8
Addressing and quantisation contracts of the paged int8 KV cache: the append kernel (one block per 64-value
group of one KV head of K or V) and the gather kernel (one thread per four consecutive values).

## Source semantics
- Cache row of (page, head h, slot s) is `(page*kv_heads + h)*page_size + s`; values of a row are `head_dim` int8,
  scales are one fp16 per 64-value group, `groups = head_dim/64`.
- Append: position `pos` maps to page `table[pos/page_size]` and slot `pos % page_size`; codes are clamped to
  [-127, 127] and quantised against the STORED (fp16-rounded) scale.
- Gather: thread i decomposes as (id, head, q4) with `per = head_dim/4`; destination index equals i, i.e. the
  scratch is the dense (id, head, d) layout.

## Form template identifiers
- KV_Q8_GROUP -> quant_group (64), kv_heads -> n_kv_heads, page_size -> page_slots, per -> quads_per_head
-/

namespace Strata.KvQ8

/-- Quantisation group width. -/
def KV_Q8_GROUP : Nat := 64

-- § helpers

/-- Row-major index bound. -/
theorem idx_lt {a b A B : Nat} (ha : a < A) (hb : b < B) : a * B + b < A * B := by
  have h1 : (a + 1) * B ≤ A * B := Nat.mul_le_mul_right B (Nat.succ_le_of_lt ha)
  have h2 : (a + 1) * B = a * B + B := by rw [Nat.add_mul, Nat.one_mul]
  omega

-- § quantisation

/-- Code clamp to the symmetric int8 range. -/
def clamp127 (q : Int) : Int := max (-127) (min 127 q)

/-- Clamped codes lie in [-127, 127]. -/
theorem clamp127_bounds (q : Int) : -127 ≤ clamp127 q ∧ clamp127 q ≤ 127 := by
  unfold clamp127
  omega

/-- Clamping is the identity inside the range. -/
theorem clamp127_id (q : Int) (h1 : -127 ≤ q) (h2 : q ≤ 127) : clamp127 q = q := by
  unfold clamp127
  omega

/-- The int8 cast of a clamped code does not wrap: it fits in [-128, 127]. -/
theorem clamp127_fits_int8 (q : Int) : -128 ≤ clamp127 q ∧ clamp127 q ≤ 127 := by
  unfold clamp127
  omega

-- § append addressing

/-- A 64-value group inside a head row stays inside the row (head_dim = groups * 64). -/
theorem group_offset_lt (g t groups : Nat) (hg : g < groups) (ht : t < 64) :
    g * 64 + t < groups * 64 :=
  idx_lt hg ht

/-- Cache row index bound for page < n_pages, head < kv_heads, slot < page_size. -/
theorem cache_row_lt (page h s n_pages kv ps : Nat) (hp : page < n_pages) (hh : h < kv)
    (hs : s < ps) : (page * kv + h) * ps + s < n_pages * kv * ps := by
  have h1 : page * kv + h < n_pages * kv := idx_lt hp hh
  exact idx_lt h1 hs

/-- Slot of a position is below page_size and recombines with the page index. -/
theorem pos_split (pos ps : Nat) : ps * (pos / ps) + pos % ps = pos :=
  Nat.div_add_mod pos ps

/-- Slot of a position is below page_size. -/
theorem slot_lt (pos ps : Nat) (h : 0 < ps) : pos % ps < ps := Nat.mod_lt _ h

/-- Value offset of (row, g, t) is below `(row+1)*head_dim` when head_dim = groups*64. -/
theorem value_offset_lt (row g t groups hd : Nat) (hd' : hd = groups * 64) (hg : g < groups)
    (ht : t < 64) : row * hd + (g * 64 + t) < (row + 1) * hd := by
  have h1 : g * 64 + t < groups * 64 := idx_lt hg ht
  have h2 : (row + 1) * hd = row * hd + hd := by rw [Nat.add_mul, Nat.one_mul]
  omega

/-- Scale index of (row, g) is below `(row+1)*groups`. -/
theorem scale_index_lt (row g groups : Nat) (hg : g < groups) :
    row * groups + g < (row + 1) * groups :=
  idx_lt (Nat.lt_succ_self row) hg

/-- Append grid has one block per (head, group, K/V) and 64 threads per block (two warps). -/
theorem append_block_warps (t : Nat) (h : t < 64) : t / 32 < 2 := by
  omega

-- § gather decomposition

/-- Thread index decomposition: id from the quotient, remainder `rem < kv*per`. -/
theorem gather_rem_lt (i kv per : Nat) (hk : 0 < kv * per) : i % (kv * per) < kv * per :=
  Nat.mod_lt _ hk

/-- Head index of the remainder is below kv_heads. -/
theorem gather_head_lt (rem kv per : Nat) (h : rem < kv * per) : rem / per < kv := by
  apply Nat.div_lt_of_lt_mul
  rw [Nat.mul_comm]
  exact h

/-- The quad index is below `per`. -/
theorem gather_quad_lt (rem per : Nat) (hp : 0 < per) : rem % per < per := Nat.mod_lt _ hp

/-- Destination index (id*kv + h)*per + q4 equals the thread index: the scratch is dense. -/
theorem gather_dst_eq (id h q4 kv per : Nat) :
    (id * kv + h) * per + q4 = id * (kv * per) + (h * per + q4) := by
  simp only [Nat.add_mul, Nat.mul_assoc]
  omega

/-- Total thread count bound: dst below `n_ids * kv * per`. -/
theorem gather_dst_lt (id h q4 n_ids kv per : Nat) (hi : id < n_ids) (hh : h < kv) (hq : q4 < per) :
    (id * kv + h) * per + q4 < n_ids * kv * per :=
  idx_lt (idx_lt hi hh) hq

/-- A quad starts at d = 4*q4 below head_dim when q4 < head_dim/4. -/
theorem quad_start_lt (q4 hd : Nat) (hd4 : hd % 4 = 0) (h : q4 < hd / 4) : q4 * 4 + 3 < hd := by
  omega

/-- The four values of a quad stay in one 64-group (head_dim a multiple of 64). -/
theorem quad_in_one_group (q4 : Nat) : (q4 * 4 + 3) / 64 = (q4 * 4) / 64 := by
  omega

/-- The scale group of a quad is below `groups` when head_dim = groups*64. -/
theorem quad_group_lt (q4 groups hd : Nat) (hd' : hd = groups * 64) (h : q4 < hd / 4) :
    (q4 * 4) / 64 < groups := by
  omega

/-- Row-level pointer arithmetic: char4 element q4 of row sits at byte `row*hd + 4*q4`. -/
theorem char4_in_row (row q4 hd : Nat) (h : q4 * 4 + 3 < hd) :
    row * hd + q4 * 4 + 3 < (row + 1) * hd := by
  have h2 : (row + 1) * hd = row * hd + hd := by rw [Nat.add_mul, Nat.one_mul]
  omega

-- § launch contract

/-- `validate`: head_dim a multiple of 64 with positive head count and page size. -/
def validShapes (hd nkv ps : Nat) : Prop := hd % KV_Q8_GROUP = 0 ∧ 0 < nkv ∧ 0 < ps

/-- Valid shapes give a whole number of groups and quads. -/
theorem valid_shapes_quads (hd nkv ps : Nat) (h : validShapes hd nkv ps) : hd % 4 = 0 := by
  obtain ⟨h1, _, _⟩ := h
  unfold KV_Q8_GROUP at h1
  omega

/-- Gather grid covers all threads with 256 per block. -/
theorem gather_grid_covers (total : Nat) : total ≤ (total + 255) / 256 * 256 := by
  omega

end Strata.KvQ8
