/-!
# MODULE: quant_dequant_kernels
Dequantisation of ggml block-quantised weights (Q2_0, Q4_0, Q5_0, Q8_0, Q3_K, Q4_K, Q5_K, Q6_K, IQ4_NL, IQ4_XS)
to bf16 / fp16 / fp32, one thread per 32-element group.

## Source semantics
* `geometry` maps a ggml type id to (elements per block, bytes per block); unsupported ids are rejected.
* Thread `g` handles row `g / groups_per_row`, group `g % groups_per_row`, and writes the 32 outputs at
  `row * groups_per_row * 32 + gi * 32`; the grid covers `rows * groups_per_row` threads exactly or more (guarded).
* Every byte offset read inside a block (scales, qs, qh, hmask, d) stays inside the block's byte size.
* Quantised integer values lie in known signed ranges before the float scale is applied.
* `f2bf` rounds an fp32 bit pattern to bf16 bits, round-to-nearest-even.

## Form template identifiers
dequant_bf16.cu `geometry` → block_geometry ; `group32<TYPE>` → group_decoder ; `dequant_kernel` → dequant_grid ;
`f2bf` → bf16_round_nearest_even ; `scale_min_k4` → k4_scale_min_unpack ; `launch` → dequant_launch.
-/

namespace Strata.QuantDequant

-- § block geometry

/-- Block geometry table (elements, bytes) per ggml type id. -/
def geometry (t : Nat) : Option (Nat × Nat) :=
  match t with
  | 2 => some (32, 18)
  | 6 => some (32, 22)
  | 8 => some (32, 34)
  | 20 => some (32, 18)
  | 11 => some (256, 110)
  | 12 => some (256, 144)
  | 13 => some (256, 176)
  | 14 => some (256, 210)
  | 23 => some (256, 136)
  | 42 => some (64, 18)
  | _ => none

/-- Bytes per row for `cols` elements. -/
def rowBytes (be bb cols : Nat) : Nat := cols / be * bb

/-- Q4_K has 144-byte blocks. -/
theorem geometry_q4k : geometry 12 = some (256, 144) := rfl

/-- An unlisted type id is unsupported. -/
theorem geometry_unsupported : geometry 3 = none := rfl

/-- Every supported block holds 32, 64 or 256 elements. -/
theorem geometry_elems (t be bb : Nat) (h : geometry t = some (be, bb)) :
    be = 32 ∨ be = 64 ∨ be = 256 := by
  unfold geometry at h
  split at h <;> simp at h <;> omega

/-- Every supported block has positive element and byte counts. -/
theorem geometry_pos (t be bb : Nat) (h : geometry t = some (be, bb)) : 0 < be ∧ 0 < bb := by
  unfold geometry at h
  split at h <;> simp at h <;> omega

/-- Block elements always divide into whole 32-element groups. -/
theorem geometry_groups (t be bb : Nat) (h : geometry t = some (be, bb)) : be % 32 = 0 := by
  have := geometry_elems t be bb h
  omega

/-- Layout sums: Q5_0 = d(2) + qh(4) + qs(16). -/
theorem q5_0_layout : 2 + 4 + 16 = 22 := by omega
/-- Layout sums: Q8_0 = d(2) + 32 quants. -/
theorem q8_0_layout : 2 + 32 = 34 := by omega
/-- Layout sums: Q3_K = hmask(32) + qs(64) + scales(12) + d(2). -/
theorem q3k_layout : 32 + 64 + 12 + 2 = 110 := by omega
/-- Layout sums: Q4_K = d + dmin + scales(12) + qs(128). -/
theorem q4k_layout : 2 + 2 + 12 + 128 = 144 := by omega
/-- Layout sums: Q5_K = d + dmin + scales(12) + qh(32) + qs(128). -/
theorem q5k_layout : 2 + 2 + 12 + 32 + 128 = 176 := by omega
/-- Layout sums: Q6_K = ql(128) + qh(64) + scales(16) + d(2). -/
theorem q6k_layout : 128 + 64 + 16 + 2 = 210 := by omega
/-- Layout sums: IQ4_XS = d(2) + scales_h(2) + scales_l(4) + qs(128). -/
theorem iq4xs_layout : 2 + 2 + 4 + 128 = 136 := by omega

-- § grid decomposition

/-- The thread index splits into (row, group) and recombines. -/
theorem thread_decompose (g gpr : Nat) (_h : 0 < gpr) : g = (g / gpr) * gpr + g % gpr := by
  have := Nat.div_add_mod' g gpr
  omega

/-- The group-in-row index is below groups per row. -/
theorem group_in_row_lt (g gpr : Nat) (h : 0 < gpr) : g % gpr < gpr := Nat.mod_lt _ h

/-- Row-major output offset of (row, gi) equals group number times 32. -/
theorem out_offset (r gpr gi : Nat) : r * gpr * 32 + gi * 32 = (r * gpr + gi) * 32 := by
  omega

/-- For gi below gpr, the row recovered from `r*gpr+gi` is r. -/
theorem row_recover (r gpr gi : Nat) (h : gi < gpr) : (r * gpr + gi) / gpr = r := by
  have hp : 0 < gpr := by omega
  rw [Nat.mul_comm r gpr, Nat.mul_add_div hp, Nat.div_eq_of_lt h]
  try omega

/-- For gi below gpr, the group recovered from `r*gpr+gi` is gi. -/
theorem group_recover (r gpr gi : Nat) (h : gi < gpr) : (r * gpr + gi) % gpr = gi := by
  rw [Nat.mul_comm r gpr, Nat.mul_add_mod]
  exact Nat.mod_eq_of_lt h

/-- The grid of 256-thread blocks has at least `total` threads. -/
theorem grid_covers (total : Nat) : total ≤ (total + 255) / 256 * 256 := by omega

/-- Row bytes for K-quants: cols/256 blocks. -/
theorem rowBytes_q4k (cols : Nat) (h : cols % 256 = 0) : rowBytes 256 144 cols = cols / 256 * 144 := rfl

/-- 32-element groups per row equals blocks per row times groups per block. -/
theorem groups_per_row_kquant (cols : Nat) (h : cols % 256 = 0) : cols / 32 = cols / 256 * 8 := by
  omega

-- § in-block group indexing

/-- K-quant group index splits into block and in-block group. -/
theorem kquant_split (gi : Nat) : gi = (gi / 8) * 8 + gi % 8 := by omega

/-- Q2_0 element index inside the 16-byte qs array stays below 18-byte block size. -/
theorem q2_0_qs_in_block (gi j : Nat) (hj : j < 32) : 2 + ((gi % 2) * 32 + j) / 4 < 18 := by omega

/-- Q2_0 stored code minus 1 lies in [-1, 2]. -/
theorem q2_0_range (q : Nat) (h : q < 4) : -1 ≤ (q : Int) - 1 ∧ (q : Int) - 1 ≤ 2 := by omega

/-- Q4_0 low nibble offset by 8 lies in [-8, 7]. -/
theorem q4_0_lo_range (b : Nat) : -8 ≤ ((b % 16 : Nat) : Int) - 8 ∧ ((b % 16 : Nat) : Int) - 8 ≤ 7 := by omega

/-- Q4_0 high nibble offset by 8 lies in [-8, 7] for a byte. -/
theorem q4_0_hi_range (b : Nat) (h : b < 256) : -8 ≤ ((b / 16 : Nat) : Int) - 8 ∧ ((b / 16 : Nat) : Int) - 8 ≤ 7 := by omega

/-- Q5_0 fifth bit selected from qh is 0 or 16. -/
theorem q5_0_bit (m : Nat) : m % 2 * 16 = 0 ∨ m % 2 * 16 = 16 := by omega

/-- Q5_0 value (nibble + fifth bit) − 16 lies in [-16, 15]. -/
theorem q5_0_range (nib bit : Nat) (hn : nib < 16) (hb : bit = 0 ∨ bit = 16) :
    -16 ≤ ((nib + bit : Nat) : Int) - 16 ∧ ((nib + bit : Nat) : Int) - 16 ≤ 15 := by omega

/-- Q8_0 reinterpretation of a byte as int8 lies in [-128, 127]. -/
def s8 (b : Nat) : Int := if b < 128 then (b : Int) else (b : Int) - 256

theorem s8_range (b : Nat) (h : b < 256) : -128 ≤ s8 b ∧ s8 b ≤ 127 := by
  unfold s8; split <;> omega

/-- Q3_K: 6-bit scale minus 32 lies in [-32, 31]. -/
theorem q3k_scale_range (us : Nat) (h : us < 64) : -32 ≤ (us : Int) - 32 ∧ (us : Int) - 32 ≤ 31 := by omega

/-- Q3_K: 2-bit value minus (0 or 4) lies in [-4, 3]. -/
theorem q3k_value_range (q2 off : Nat) (hq : q2 < 4) (ho : off = 0 ∨ off = 4) :
    -4 ≤ (q2 : Int) - off ∧ (q2 : Int) - off ≤ 3 := by omega

/-- Q3_K scale index stays in the 16 unpacked scales. -/
theorem q3k_scale_index (gi t : Nat) (hgi : gi < 8) (ht : t < 32) :
    (gi / 4) * 8 + (gi % 4) * 2 + (if t ≥ 16 then 1 else 0) < 16 := by
  split <;> omega

/-- Q3_K qs byte offset stays below the 96-byte hmask+qs region. -/
theorem q3k_qs_bound (gi t : Nat) (hgi : gi < 8) (ht : t < 32) : 32 + (gi / 4) * 32 + t < 96 := by omega

/-- Q3_K scale byte indices read by the 12-byte packed scales are below 12 (four branches a–d below). -/
theorem q3k_scales_pack_bound_a (is : Nat) (h : is < 4) : is + 8 < 12 := by omega
theorem q3k_scales_pack_bound_b (is : Nat) (h1 : 4 ≤ is) (h2 : is < 8) : is + 4 < 12 := by omega
theorem q3k_scales_pack_bound_c (is : Nat) (h1 : 8 ≤ is) (h2 : is < 12) : is - 8 < 12 ∧ is < 12 := by omega
theorem q3k_scales_pack_bound_d (is : Nat) (h1 : 12 ≤ is) (h2 : is < 16) : is - 4 < 12 ∧ is - 8 < 12 := by omega

/-- `scale_min_k4` first branch (j<4) yields 6-bit values. -/
theorem scale_min_k4_lo (a b : Nat) : a % 64 < 64 ∧ b % 64 < 64 := by omega

/-- `scale_min_k4` second branch combines a nibble with two high bits: still 6-bit. -/
theorem scale_min_k4_hi (a b c : Nat) (hc : c < 256) (hb : b < 256) :
    a % 16 + (c / 64) * 16 < 64 ∧ b / 16 + (c / 64) * 16 < 64 := by omega

/-- Q4_K qs byte offset stays within the 144-byte block. -/
theorem q4k_qs_bound (gi l : Nat) (hgi : gi < 8) (hl : l < 32) : 16 + 32 * (gi / 2) + l < 144 := by omega

/-- Q5_K qh and qs offsets stay within the 176-byte block. -/
theorem q5k_bounds (gi l : Nat) (hgi : gi < 8) (hl : l < 32) :
    16 + l < 48 ∧ 48 + 32 * (gi / 2) + l < 176 ∧ 2 * (gi / 2) + gi % 2 < 8 := by omega

/-- Q6_K offsets (ql, qh, scales) stay inside their regions. -/
theorem q6k_bounds (gi l : Nat) (hgi : gi < 8) (hl : l < 32) :
    64 * (gi / 4) + l + 32 < 128 ∧ 128 + 32 * (gi / 4) + l < 192 ∧
    192 + 8 * (gi / 4) + l / 16 + 2 * (gi % 4) < 208 := by omega

/-- Q6_K 6-bit value minus 32 lies in [-32, 31]. -/
theorem q6k_value_range (lo hi : Nat) (hl : lo < 16) (hh : hi < 4) :
    -32 ≤ ((lo + hi * 16 : Nat) : Int) - 32 ∧ ((lo + hi * 16 : Nat) : Int) - 32 ≤ 31 := by omega

/-- IQ4_XS offsets stay inside the 136-byte block. -/
theorem iq4xs_bounds (ib j : Nat) (hib : ib < 8) (hj : j < 16) :
    4 + ib / 2 < 8 ∧ 8 + 16 * ib + j < 136 := by omega

/-- IQ4_XS 6-bit scale (4 low bits + 2 high bits) minus 32 lies in [-32, 31]. -/
theorem iq4xs_scale_range (lo hi : Nat) (hl : lo < 16) (hh : hi < 4) :
    -32 ≤ ((lo + hi * 16 : Nat) : Int) - 32 ∧ ((lo + hi * 16 : Nat) : Int) - 32 ≤ 31 := by omega

-- § bf16 rounding

/-- fp32 bits → bf16 bits, round-to-nearest-even (uint32 arithmetic wraps). -/
def f2bf (u : Nat) : Nat := ((u + 0x7fff + (u / 65536) % 2) % 4294967296) / 65536

/-- The result always fits in 16 bits. -/
theorem f2bf_lt (u : Nat) : f2bf u < 65536 := by unfold f2bf; omega

/-- The rounding addend never wraps the uint32 when the value is below the top binade or below halfway. -/
theorem f2bf_nowrap (u : Nat) (hu : u < 4294967296) (h : u / 65536 < 65535 ∨ u % 65536 < 32768) :
    u + 0x7fff + (u / 65536) % 2 < 4294967296 := by omega

/-- Values already representable in bf16 round to themselves. -/
theorem f2bf_exact (h : Nat) (hh : h < 65536) : f2bf (h * 65536) = h := by
  have hw := f2bf_nowrap (h * 65536) (by omega) (Or.inr (by omega))
  unfold f2bf
  rw [Nat.mod_eq_of_lt hw]
  omega

/-- Below the halfway point the value truncates. -/
theorem f2bf_round_down (u : Nat) (hu : u < 4294967296) (hl : u % 65536 < 32768) : f2bf u = u / 65536 := by
  have hw := f2bf_nowrap u hu (Or.inr hl)
  unfold f2bf
  rw [Nat.mod_eq_of_lt hw]
  omega

/-- Above the halfway point the value rounds up. -/
theorem f2bf_round_up (u : Nat) (hu : u < 4294967296) (hl : 32768 < u % 65536) (hh : u / 65536 < 65535) :
    f2bf u = u / 65536 + 1 := by
  have hw := f2bf_nowrap u hu (Or.inl hh)
  unfold f2bf
  rw [Nat.mod_eq_of_lt hw]
  omega

/-- A tie rounds to an even mantissa. -/
theorem f2bf_tie_even (u : Nat) (hu : u < 4294967296) (hl : u % 65536 = 32768) (hh : u / 65536 < 65535) :
    f2bf u % 2 = 0 := by
  have hw := f2bf_nowrap u hu (Or.inl hh)
  obtain ⟨k, hk⟩ : ∃ k, u = 65536 * k + 32768 := ⟨u / 65536, by omega⟩
  subst hk
  have e1 : (65536 * k + 32768) / 65536 = k := by omega
  have e2 : (65536 * k + 32768 + 32767 + k % 2) / 65536 = k + k % 2 := by omega
  unfold f2bf
  rw [Nat.mod_eq_of_lt hw, e1, e2]
  omega

/-- 1.0f rounds to bf16 0x3f80. -/
theorem f2bf_one : f2bf 0x3f800000 = 0x3f80 := by decide

end Strata.QuantDequant
