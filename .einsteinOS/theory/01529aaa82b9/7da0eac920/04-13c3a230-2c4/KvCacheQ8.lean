/-!
# MODULE: kv_cache_q8
Paged int8 KV-cache append and gather with an fp16 scale per 64-value group.

## Source semantics
* Append: one 64-thread block per (kv head h, group g, K/V). The scale is stored as fp16 of amax/127 and the value
  is quantised against the stored scale, clamped to [-127, 127]. The row is
  `(page*kv_heads + h)*page_size + pos % page_size`, with `page = table[pos / page_size]`.
* Gather: one thread per 4 consecutive values; index i splits into (id, h, q4) and the destination index equals i.
* head_dim must be a multiple of 64.

## Form template identifiers
KV_Q8_GROUP → quant_group (64) ; `kv_append_q8_kernel` → append_quantised ; `kv_gather_q8_kernel` →
gather_dequantised ; page_table → logical_to_physical_page.
-/

namespace Strata.KvQ8

-- § quantisation

/-- Clamp to the symmetric int8 range [-127, 127]. -/
def clamp127 (q : Int) : Int := if q < -127 then -127 else if q > 127 then 127 else q

/-- The clamp result lies in [-127, 127]. -/
theorem clamp127_range (q : Int) : -127 ≤ clamp127 q ∧ clamp127 q ≤ 127 := by
  unfold clamp127; split <;> (try split) <;> omega

/-- The clamp is the identity inside the range. -/
theorem clamp127_id (q : Int) (h1 : -127 ≤ q) (h2 : q ≤ 127) : clamp127 q = q := by
  unfold clamp127; split <;> (try split) <;> omega

/-- The clamp is idempotent. -/
theorem clamp127_idem (q : Int) : clamp127 (clamp127 q) = clamp127 q := by
  have := clamp127_range q
  exact clamp127_id _ this.1 this.2

/-- The clamp is monotone. -/
theorem clamp127_mono (a b : Int) (h : a ≤ b) : clamp127 a ≤ clamp127 b := by
  unfold clamp127; split <;> (try split) <;> (try split) <;> (try split) <;> omega

-- § geometry

/-- Group count per head. -/
theorem groups_split (head_dim : Nat) (h : head_dim % 64 = 0) : head_dim / 64 * 64 = head_dim := by omega

/-- Input element index h*head_dim + g*64 + t is below kv_heads*head_dim. -/
theorem append_input_lt (h g t kvh hd : Nat) (hh : h < kvh) (hg : g < hd / 64) (ht : t < 64) (hm : hd % 64 = 0) :
    h * hd + g * 64 + t < kvh * hd := by
  have h1 : (h + 1) * hd ≤ kvh * hd := Nat.mul_le_mul_right hd (Nat.succ_le_of_lt hh)
  have h2 : (h + 1) * hd = h * hd + hd := Nat.succ_mul h hd
  omega

/-- Position splits into page index and in-page offset. -/
theorem pos_split (pos ps : Nat) (_h : 0 < ps) : pos = (pos / ps) * ps + pos % ps := by
  have := Nat.div_add_mod' pos ps
  omega

/-- In-page offset is below page size. -/
theorem pos_off_lt (pos ps : Nat) (h : 0 < ps) : pos % ps < ps := Nat.mod_lt _ h

/-- A physical row (page*kvh + h)*ps + off is below (pages*kvh)*ps for page < pages. -/
theorem row_lt (page h off kvh ps pages : Nat) (hp : page < pages) (hh : h < kvh) (ho : off < ps) :
    (page * kvh + h) * ps + off < pages * kvh * ps := by
  have h1 : (page + 1) * kvh ≤ pages * kvh := Nat.mul_le_mul_right kvh (Nat.succ_le_of_lt hp)
  have h2 : (page + 1) * kvh = page * kvh + kvh := Nat.succ_mul page kvh
  have h3 : page * kvh + h < pages * kvh := by omega
  have h4 : (page * kvh + h + 1) * ps ≤ pages * kvh * ps := Nat.mul_le_mul_right ps (Nat.succ_le_of_lt h3)
  have h5 : (page * kvh + h + 1) * ps = (page * kvh + h) * ps + ps := Nat.succ_mul _ ps
  omega

/-- Quant element address within a row. -/
theorem elem_in_row (g t hd : Nat) (hg : g < hd / 64) (ht : t < 64) (hm : hd % 64 = 0) : g * 64 + t < hd := by omega

/-- Scale address within a row: group g < groups. -/
theorem scale_in_row (g hd : Nat) (hg : g < hd / 64) : g < hd / 64 := hg

-- § gather

/-- Gather: with per = head_dim/4, K = kv*per, an index i < n*K splits into (id, h, q4) and
the destination index `(id*kv + h)*per + q4` equals i. -/
theorem gather_dst_eq (i kv per : Nat) (hp : 0 < per) :
    ((i / (kv * per)) * kv + (i % (kv * per)) / per) * per + (i % (kv * per)) % per = i := by
  generalize hq : i / (kv * per) = q
  generalize hm : i % (kv * per) = m
  have h1 : kv * per * q + m = i := by
    rw [← hq, ← hm]; exact Nat.div_add_mod i (kv * per)
  have h2 : per * (m / per) + m % per = m := Nat.div_add_mod m per
  have h3 : (q * kv + m / per) * per = kv * per * q + per * (m / per) := by
    rw [Nat.add_mul, Nat.mul_assoc q kv per, Nat.mul_comm q (kv * per), Nat.mul_comm (m / per) per]
  omega

/-- The 4 values loaded by one thread (d..d+3, d = 4*q4) stay inside the head row and one scale group. -/
theorem gather_vec_in_group (q4 hd : Nat) (hq : q4 < hd / 4) (hm : hd % 64 = 0) :
    (q4 * 4) / 64 < hd / 64 ∧ q4 * 4 + 3 < hd ∧ (q4 * 4) / 64 = (q4 * 4 + 3) / 64 := by omega

/-- Gather grid covers all work items. -/
theorem gather_grid_covers (total : Nat) : total ≤ (total + 255) / 256 * 256 := by omega

/-- Dequantised value is the stored int times the scale: sign symmetric (integer shadow). -/
theorem dequant_neg (q : Int) (s : Int) : (-q) * s = -(q * s) := by
  rw [Int.neg_mul]

/-- A zero scale quantises everything to zero. -/
theorem zero_scale_zero (q s : Int) (h : s = 0) : q * s = 0 := by subst h; simp

end Strata.KvQ8
