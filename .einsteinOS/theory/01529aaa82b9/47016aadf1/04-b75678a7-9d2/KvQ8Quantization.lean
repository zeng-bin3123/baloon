/-!
# MODULE: kv_q8_quantization
Quantization range and addressing contracts of the Q8 KV-cache kernels in `kv_q8.cu`.

## Source semantics
* Each 64-value group stores an fp16 scale `amax/127` and int8 codes clamped to [-127, 127].
* Append writes row `(page*kv_heads + h)*page_size + pos % page_size` with `page = table[pos / page_size]`.
* Gather decomposes a flat index `i` into (id, head, quad) via `kv_heads*per` and `per = head_dim/4`.
* A 64-thread block reduces its max in two warps.

## Form template identifiers
KV_Q8_GROUP -> group_size (64); per -> quads_per_head; groups -> groups_per_head; page_size -> page_cells.
-/

namespace KvQ8Quantization

-- § code range

/-- Clamp into [-127, 127]. -/
def clampQ (q : Int) : Int := if q < -127 then -127 else if q > 127 then 127 else q

/-- The clamped code is at least -127. -/
theorem clampQ_ge (q : Int) : -127 ≤ clampQ q := by unfold clampQ; split_ifs <;> omega

/-- The clamped code is at most 127. -/
theorem clampQ_le (q : Int) : clampQ q ≤ 127 := by unfold clampQ; split_ifs <;> omega

/-- In-range codes are unchanged. -/
theorem clampQ_id (q : Int) (h1 : -127 ≤ q) (h2 : q ≤ 127) : clampQ q = q := by
  unfold clampQ; split_ifs <;> omega

/-- The clamped code fits int8 (no wrap on the cast). -/
theorem clampQ_int8 (q : Int) : -128 ≤ clampQ q ∧ clampQ q ≤ 127 := by
  have a := clampQ_ge q; have b := clampQ_le q; omega

-- § append addressing

/-- A position decomposes into (page index, cell in page). -/
theorem pos_decomp (pos ps : Nat) : ps * (pos / ps) + pos % ps = pos := Nat.div_add_mod pos ps

/-- The cell in page is below the page size. -/
theorem cell_in_page_lt (pos ps : Nat) (h : 0 < ps) : pos % ps < ps := Nat.mod_lt _ h

/-- A row stays inside its (page, head) slab. -/
theorem row_in_slab (A ps off : Nat) (h : off < ps) : A * ps + off < (A + 1) * ps := by
  rw [Nat.add_mul, Nat.one_mul]; omega

/-- Slabs of different (page, head) pairs with different slab indices cannot collide. -/
theorem slabs_disjoint (A B ps off off' : Nat) (hab : A < B) (h : off < ps) (h' : off' < ps) :
    A * ps + off < B * ps + off' := by
  have h1 : (A + 1) * ps ≤ B * ps := Nat.mul_le_mul_right _ hab
  rw [Nat.add_mul, Nat.one_mul] at h1
  omega

-- § groups and warps

/-- A thread index of a 64-thread block belongs to warp 0 or 1. -/
theorem warp_of_thread (t : Nat) (h : t < 64) : t / 32 < 2 := by omega

/-- The group of a value is below the groups per head when `head_dim = 64*groups`. -/
theorem group_lt (groups d head_dim : Nat) (hh : head_dim = 64 * groups) (hd : d < head_dim) :
    d / 64 < groups := by omega

/-- Value index inside the head. -/
theorem value_index_lt (g t groups : Nat) (hg : g < groups) (ht : t < 64) : g * 64 + t < 64 * groups := by
  omega

-- § gather decomposition

/-- The quad within a head is below `per`. -/
theorem quad_lt (rem per : Nat) (hp : 0 < per) : rem % per < per := Nat.mod_lt _ hp

/-- The flat index recomposes. -/
theorem gather_decomp (i m : Nat) : m * (i / m) + i % m = i := Nat.div_add_mod i m

/-- The id of a flat index is below `n_ids`. -/
theorem gather_id_lt (i n_ids m : Nat) (h : i < m * n_ids) : i / m < n_ids := Nat.div_lt_of_lt_mul h

/-- A quad covers values `4q .. 4q+3`, all inside the head when `q < head_dim/4`. -/
theorem quad_values_lt (q head_dim : Nat) (h : q < head_dim / 4) : 4 * q + 3 < head_dim := by omega

/-- The group of the first value of a quad also contains the whole quad. -/
theorem quad_in_group (q : Nat) : (4 * q) / 64 = (4 * q + 3) / 64 := by omega

end KvQ8Quantization
