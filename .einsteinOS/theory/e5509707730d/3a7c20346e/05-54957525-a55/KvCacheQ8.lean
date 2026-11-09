/-!
# MODULE: kv_cache_q8
SOURCES: src/kernels/cuda/kv_q8.cu
Append (quantize) and gather (dequantize) steps for a paged int8 KV cache with one fp16 scale per 64-value group.

## Source semantics
Row of (page, head, in-page position) is (page*kv_heads + head)*page_size + pos % page_size, where page is the
page-table entry for pos / page_size. A row holds head_dim int8 values and head_dim/64 scales; head_dim must be
a multiple of 64. Quantization uses the STORED (fp16-rounded) scale, ints clamped to [-127,127]. Gather reads four
values per thread; thread index decomposes as (id, head, quad) and writes the scratch slot with the same linear
index.

## Form template identifiers
rowIdx -> "paged row index"; scaleIdx -> "group scale index"; gather decomposition -> "thread to (id, head, quad)".
-/
namespace Strata.KvQ8

-- § paged row index
/-- Row of an (page, head, offset) triple. -/
def rowIdx (page kvh h ps off : Nat) : Nat := ps * (page * kvh + h) + off

/-- The page-major cell is recovered by division. -/
theorem rowIdx_div (page kvh h ps off : Nat) (hps : 0 < ps) (hoff : off < ps) :
    rowIdx page kvh h ps off / ps = page * kvh + h := by
  unfold rowIdx
  rw [Nat.mul_add_div hps, Nat.div_eq_of_lt hoff]
  simp

/-- The in-page position is recovered by remainder. -/
theorem rowIdx_mod (page kvh h ps off : Nat) (hps : 0 < ps) (hoff : off < ps) :
    rowIdx page kvh h ps off % ps = off := by
  have hd := rowIdx_div page kvh h ps off hps hoff
  have h1 := Nat.div_add_mod (rowIdx page kvh h ps off) ps
  rw [hd] at h1
  unfold rowIdx at h1 ⊢
  omega

/-- Distinct (cell, offset) pairs get distinct rows. -/
theorem rowIdx_injective (a b ps off off' : Nat) (hps : 0 < ps) (ho : off < ps) (ho' : off' < ps)
    (e : ps * a + off = ps * b + off') : a = b ∧ off = off' := by
  have h1 := Nat.mul_add_div hps a off
  have h2 := Nat.mul_add_div hps b off'
  rw [Nat.div_eq_of_lt ho] at h1
  rw [Nat.div_eq_of_lt ho'] at h2
  rw [e] at h1
  have hab : a = b := by omega
  subst hab
  exact ⟨rfl, by omega⟩

-- § group scales
/-- Scale slot of group g in a row. -/
def scaleIdx (row groups g : Nat) : Nat := groups * row + g

/-- Scale slots of different rows never collide (g < groups). -/
theorem scaleIdx_injective (r r' groups g g' : Nat) (hg : 0 < groups) (h1 : g < groups) (h2 : g' < groups)
    (e : scaleIdx r groups g = scaleIdx r' groups g') : r = r' ∧ g = g' := by
  unfold scaleIdx at e
  have := rowIdx_injective r r' groups g g' hg h1 h2 e
  exact ⟨this.1, this.2⟩

/-- head_dim a multiple of 64 gives whole 4-value quads. -/
theorem head_dim_quads (hd : Nat) (h : hd % 64 = 0) : hd % 4 = 0 := by
  omega

-- § gather index decomposition
/-- The destination slot (id, head, quad) equals the source thread index. -/
theorem gather_dst (id kvh per h q4 : Nat) :
    (id * kvh + h) * per + q4 = id * (kvh * per) + (h * per + q4) := by
  rw [Nat.add_mul, Nat.mul_assoc]
  omega

/-- The remainder splits into head and quad. -/
theorem gather_rem (rem per : Nat) : per * (rem / per) + rem % per = rem := by
  exact Nat.div_add_mod rem per

-- § saturation
/-- Clamp to the symmetric int8 range used by the cache. -/
def clamp127 (v : Int) : Int := if v < -127 then -127 else if v > 127 then 127 else v

/-- The stored code is always within [-127,127]. -/
theorem clamp127_bounds (v : Int) : -127 ≤ clamp127 v ∧ clamp127 v ≤ 127 := by
  unfold clamp127
  split_ifs <;> omega

/-- In-range codes pass through unchanged. -/
theorem clamp127_id (v : Int) (h1 : -127 ≤ v) (h2 : v ≤ 127) : clamp127 v = v := by
  unfold clamp127
  split_ifs <;> omega

/-- A 256-thread grid covers every gather thread. -/
theorem gather_grid_covers (total : Nat) : total ≤ (total + 255) / 256 * 256 := by
  omega

end Strata.KvQ8
