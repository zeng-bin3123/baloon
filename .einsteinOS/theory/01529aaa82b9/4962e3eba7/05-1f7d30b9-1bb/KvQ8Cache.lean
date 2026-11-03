/-!
# MODULE: kv_q8_cache
Int8 KV-cache append (per-64-value-group absmax scale stored as f16, codes clamped to [-127, 127]) and the
gather that dequantises selected cells back to f16.  This file proves the block/thread addressing of the
append, the code range, the scale-group addressing of the gather (a 4-value run never straddles a
group), and the memory saving of the layout.

## Source semantics
* One append block per (kv head, 64-value group, K or V); 64 threads, one value each, two warps whose
  maxima are combined in fixed order.
* Codes are `clamp(round(x / stored_scale), -127, 127)`, and 0 when the stored scale is 0.
* The gather thread handles 4 consecutive values (`d = q4*4`) of one cell and head.

## Form template identifiers
- `kv_append_q8_kernel` -> `q8_append`; `kv_gather_q8_kernel` -> `q8_gather`
- `KV_Q8_GROUP` -> `group_size` (64); `sbits` -> `scale_bits`; `row` -> `pool_row`
-/

namespace Strata.KvQ8

-- § arithmetic helper

/-- Row-major linearisation stays in bounds. -/
theorem lin2_lt {i a c d : Nat} (hi : i < a) (hc : c < d) : i * d + c < a * d := by
  have h := Nat.mul_le_mul_right d (show i + 1 ≤ a by omega)
  rw [Nat.add_mul, Nat.one_mul] at h
  omega

-- § append addressing

/-- Within a group, `g*64 + t` is below `head_dim = 64 * groups`. -/
theorem group_value_lt {g t groups hd : Nat} (hhd : hd = groups * 64) (hg : g < groups) (ht : t < 64) :
    g * 64 + t < hd := by omega

/-- The source element `h*head_dim + g*64 + t` is inside the `kv_heads * head_dim` input. -/
theorem input_index_lt {h g t groups hd kvh : Nat} (hhd : hd = groups * 64) (hh : h < kvh)
    (hg : g < groups) (ht : t < 64) : h * hd + (g * 64 + t) < kvh * hd :=
  lin2_lt hh (group_value_lt hhd hg ht)

/-- Thread `t < 64` belongs to warp `t / 32 < 2`, the two warps whose maxima are combined. -/
theorem warp_slot_lt {t : Nat} (ht : t < 64) : t / 32 < 2 := by omega

/-- The scale index `row*groups + g` is inside the scale buffer of `rows * groups` entries. -/
theorem scale_index_lt {row g rows groups : Nat} (hr : row < rows) (hg : g < groups) :
    row * groups + g < rows * groups := lin2_lt hr hg

/-- The pool row `(page*kv_heads + h)*page_size + slot` is inside the pool. -/
theorem pool_row_lt {page h slot pages kvh ps : Nat} (hp : page < pages) (hh : h < kvh) (hs : slot < ps) :
    (page * kvh + h) * ps + slot < pages * kvh * ps :=
  lin2_lt (lin2_lt hp hh) hs

-- § code range

/-- Clamp to the symmetric int8 range used by the quantiser. -/
def clamp127 (q : Int) : Int := if q < -127 then -127 else if q > 127 then 127 else q

/-- The code is within `[-127, 127]`, so `-128` is never produced. -/
theorem clamp127_range (q : Int) : -127 ≤ clamp127 q ∧ clamp127 q ≤ 127 := by
  unfold clamp127
  split_ifs <;> omega

/-- Codes already in range are unchanged. -/
theorem clamp127_id {q : Int} (h1 : -127 ≤ q) (h2 : q ≤ 127) : clamp127 q = q := by
  unfold clamp127
  split_ifs <;> omega

/-- The code is exactly representable as a signed byte. -/
theorem code_fits_int8 (q : Int) : -128 ≤ clamp127 q ∧ clamp127 q ≤ 127 := by
  have := clamp127_range q
  omega

-- § gather addressing

/-- A thread's 4-value run `q4*4 .. q4*4+3` stays inside the head row of `head_dim = 64 * groups`. -/
theorem run_in_row {q4 groups hd : Nat} (hhd : hd = groups * 64) (hq : q4 < hd / 4) : q4 * 4 + 3 < hd := by
  omega

/-- The scale group of a thread's run is below the number of groups. -/
theorem run_group_lt {q4 groups hd : Nat} (hhd : hd = groups * 64) (hq : q4 < hd / 4) :
    (q4 * 4) / 64 < groups := by omega

/-- A 4-value run starting at a multiple of 4 never crosses a 64-value group boundary,
so one scale applies to all four values. -/
theorem run_in_one_group (q4 : Nat) : (q4 * 4) / 64 = (q4 * 4 + 3) / 64 := by omega

/-- The gather destination `(id*kv_heads + h)*per + q4` is inside the scratch of `n_ids` cells. -/
theorem gather_dst_lt {id h q4 n_ids kvh per : Nat} (hid : id < n_ids) (hh : h < kvh) (hq : q4 < per) :
    (id * kvh + h) * per + q4 < n_ids * kvh * per :=
  lin2_lt (lin2_lt hid hh) hq

/-- The gather's flat index splits into (cell, head, quad) exactly. -/
theorem gather_split {i kvh per : Nat} (hper : 0 < per) (hk : 0 < kvh) :
    i % (kvh * per) / per < kvh ∧ i % (kvh * per) % per < per := by
  constructor
  · exact Nat.div_lt_of_lt_mul (by
      have := Nat.mod_lt i (Nat.mul_pos hk hper)
      rw [Nat.mul_comm per kvh]
      exact this)
  · exact Nat.mod_lt _ hper

-- § memory layout

/-- Per row of 256 values the int8 layout needs 256 code bytes plus 4 two-byte scales: 264 bytes. -/
theorem int8_row_bytes : 256 + 4 * 2 = 264 := by decide

/-- That is less than the 512 bytes of the f16 layout. -/
theorem int8_smaller_than_f16 : 256 + 4 * 2 < 256 * 2 := by decide

end Strata.KvQ8
