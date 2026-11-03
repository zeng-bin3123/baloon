/-!
# MODULE: ggml_block_dequant
Scalar reference dequantizers for the ggml block formats (Q2_0, Q4_0, Q5_0, Q6_K, Q4_K, Q5_K, Q3_K,
IQ4_NL, IQ4_XS). We model the index/bit arithmetic and value ranges, not float rounding.

## Source semantics
* Q2_0: block = fp16 d ++ 16 bytes = 64 elements / 18 bytes; code c in {0..3} maps to symbol c-1 in {-1,0,1,2};
  4 codes per byte, LSB first: byte j/4, bit offset (j%4)*2.
* Q4_0/IQ4_NL: low nibble is element j, high nibble element j+16.
* Q5_0: 5th bit of element e is bit e of the 32-bit qh.
* Q6_K: two 128-halves; scale index is 8*half + l/16 + {0,2,4,6}.
* K-quant 6-bit scale/min unpacking (get_scale_min_k4); Q5_K masks shift by 2 per 64-group;
  Q3_K mask m is hoisted (does not reset between halves) and the correction is -4 when the high bit is clear.
* IQ4_XS: 6-bit scale from scales_l nibble plus 2 bits of scales_h, then -32.

## Form template identifiers
* block / blk -> `block` (byte array of one quantization block)
* out / y -> `out` (float output row)
* kvalues_iq4nl -> `codebook`
* qs / qh / ql -> `codes`, `highbits`, `lowbits`
-/

namespace Strata.Dequant

-- § Q2_0

/-- The 2-bit code of element `j` inside the byte `b` that holds it. -/
def q2Code (b j : Nat) : Nat := (b / 2 ^ ((j % 4) * 2)) % 4

/-- Symbol decode for Q2_0: code minus one. -/
def q2Sym (c : Nat) : Int := (c : Int) - 1

/-- Every extracted Q2_0 code is below four. -/
theorem q2Code_lt (b j : Nat) : q2Code b j < 4 := by
  unfold q2Code
  exact Nat.mod_lt _ (by decide)

/-- Symbols lie in {-1..2}. -/
theorem q2Sym_range (c : Nat) (hc : c < 4) : -1 ≤ q2Sym c ∧ q2Sym c ≤ 2 := by
  unfold q2Sym
  omega

/-- The structural invariant of `--check`: every decoded value is one of -d, 0, d, 2d. -/
theorem q2_value_set (c : Nat) (d : Int) (hc : c < 4) :
    q2Sym c * d = -d ∨ q2Sym c * d = 0 ∨ q2Sym c * d = d ∨ q2Sym c * d = 2 * d := by
  have h : c = 0 ∨ c = 1 ∨ c = 2 ∨ c = 3 := by omega
  unfold q2Sym
  rcases h with rfl | rfl | rfl | rfl <;> omega

/-- Byte index and bit offset stay inside the 16-byte payload / one byte. -/
theorem q2_index_bounds (j : Nat) (hj : j < 64) : j / 4 < 16 ∧ (j % 4) * 2 ≤ 6 := by
  omega

/-- (byte, slot) determines the element: the layout is a bijection of 64 elements onto 16x4 slots. -/
theorem q2_layout_inverse (j : Nat) : 4 * (j / 4) + j % 4 = j := by
  omega

/-- Distinct elements never share a (byte, slot) pair. -/
theorem q2_layout_injective (j k : Nat) (h1 : j / 4 = k / 4) (h2 : j % 4 = k % 4) : j = k := by
  have a := q2_layout_inverse j
  have b := q2_layout_inverse k
  omega

-- § Block sizes

/-- Q2_0: 2-byte scale plus 16 payload bytes is 18 bytes for 64 elements. -/
theorem q2_0_block_bytes : 2 + 64 / 4 = 18 := by decide

/-- Q4_0 and IQ4_NL: 2 + 16 = 18 bytes. -/
theorem q4_0_block_bytes : 2 + 32 / 2 = 18 := by decide

/-- Q5_0: 2 + 4 + 16 = 22 bytes. -/
theorem q5_0_block_bytes : 2 + 4 + 32 / 2 = 22 := by decide

/-- Q8_0: 2 + 32 = 34 bytes. -/
theorem q8_0_block_bytes : 2 + 32 = 34 := by decide

/-- Q6_K: ql 128, qh 64, scales 16, d 2 = 210 bytes. -/
theorem q6_K_block_bytes : 128 + 64 + 16 + 2 = 210 := by decide

/-- Q4_K: d, dmin, 12 scale bytes, 128 payload bytes = 144 bytes. -/
theorem q4_K_block_bytes : 2 + 2 + 12 + 128 = 144 := by decide

/-- Q5_K: adds the 32-byte 5th bit plane = 176 bytes. -/
theorem q5_K_block_bytes : 2 + 2 + 12 + 32 + 128 = 176 := by decide

/-- Q3_K: hmask 32, qs 64, scales 12, d 2 = 110 bytes. -/
theorem q3_K_block_bytes : 32 + 64 + 12 + 2 = 110 := by decide

/-- IQ4_XS: d 2, scales_h 2, scales_l 4, qs 128 = 136 bytes. -/
theorem iq4_xs_block_bytes : 2 + 2 + 4 + 128 = 136 := by decide

-- § Nibble layouts (Q4_0, Q5_0, IQ4_NL)

/-- Split-half layout: byte j holds elements j and j+16, covering all 32 elements. -/
theorem splitHalf_cover (e : Nat) (he : e < 32) :
    ∃ j, j < 16 ∧ (e = j ∨ e = j + 16) := by
  refine ⟨e % 16, by omega, by omega⟩

/-- The interleaved reading (2j, 2j+1) is a different layout: at j = 1 it names elements 2 and 3, not 1 and 17. -/
theorem splitHalf_ne_interleaved : (1 : Nat) ≠ 2 * 1 ∧ (1 + 16 : Nat) ≠ 2 * 1 + 1 := by
  decide

/-- Q4_0 signed range: a nibble minus 8 lies in [-8, 7]. -/
theorem q4_0_range (b : Nat) (hb : b < 256) :
    -8 ≤ ((b % 16 : Nat) : Int) - 8 ∧ ((b % 16 : Nat) : Int) - 8 ≤ 7 ∧
    -8 ≤ ((b / 16 : Nat) : Int) - 8 ∧ ((b / 16 : Nat) : Int) - 8 ≤ 7 := by
  omega

/-- Q5_0 signed range: nibble plus 16*highbit minus 16 lies in [-16, 15]. -/
theorem q5_0_range (lo bit : Nat) (hlo : lo < 16) (hbit : bit < 2) :
    -16 ≤ ((lo + 16 * bit : Nat) : Int) - 16 ∧ ((lo + 16 * bit : Nat) : Int) - 16 ≤ 15 := by
  omega

-- § Q6_K

/-- Q6_K 6-bit value minus 32 lies in [-32, 31]. -/
theorem q6_K_value_range (lo hi : Nat) (hlo : lo < 16) (hhi : hi < 4) :
    -32 ≤ ((lo + 16 * hi : Nat) : Int) - 32 ∧ ((lo + 16 * hi : Nat) : Int) - 32 ≤ 31 := by
  omega

/-- Q6_K scale indices `8*half + l/16 + 2k` stay among the 16 scales. -/
theorem q6_K_scale_index (h l k : Nat) (hh : h < 2) (hl : l < 32) (hk : k ≤ 3) :
    8 * h + l / 16 + 2 * k < 16 := by
  omega

/-- Q6_K output positions `128*half + l + 32*k` stay below 256. -/
theorem q6_K_out_pos (h l k : Nat) (hh : h < 2) (hl : l < 32) (hk : k ≤ 3) :
    128 * h + l + 32 * k < 256 := by
  omega

/-- Q6_K output positions are injective in (half, l, k), so the 256 outputs are written exactly once. -/
theorem q6_K_out_injective (h l k h' l' k' : Nat) (hh : h < 2) (hl : l < 32) (hk : k ≤ 3)
    (hh' : h' < 2) (hl' : l' < 32) (hk' : k' ≤ 3)
    (e : 128 * h + l + 32 * k = 128 * h' + l' + 32 * k') :
    h = h' ∧ l = l' ∧ k = k' := by
  omega

-- § K-quant scale/min (Q4_K, Q5_K)

/-- `get_scale_min_k4` on a byte accessor `q`. -/
def getScaleMin (q : Nat → Nat) (j : Nat) : Nat × Nat :=
  if j < 4 then (q j % 64, q (j + 4) % 64)
  else ((q (j + 4) % 16) + (q (j - 4) / 64) * 16, (q (j + 4) / 16) + (q j / 64) * 16)

/-- Both unpacked values are 6-bit. -/
theorem getScaleMin_lt (q : Nat → Nat) (hq : ∀ i, q i < 256) (j : Nat) :
    (getScaleMin q j).1 < 64 ∧ (getScaleMin q j).2 < 64 := by
  have h1 := hq j
  have h2 := hq (j + 4)
  have h3 := hq (j - 4)
  unfold getScaleMin
  split
  · dsimp only
    omega
  · dsimp only
    omega

/-- Q5_K: the eight 5th-bit masks (u1, u2 per 64-group) are exactly the eight distinct single bits. -/
theorem q5_K_masks_are_all_bits :
    (List.range 4).flatMap (fun g => [1 <<< (2 * g), 1 <<< (2 * g + 1)]) =
      (List.range 8).map (fun i => 1 <<< i) := by
  decide

/-- Q5_K masks are pairwise distinct. -/
theorem q5_K_masks_nodup : ((List.range 8).map (fun i => 1 <<< i)).Nodup := by
  decide

-- § Q3_K

/-- Q3_K: the hoisted mask exponent `4*half + j` ranges over 0..7 injectively. -/
theorem q3_K_mask_exponent (h j h' j' : Nat) (hh : h < 2) (hj : j < 4) (hh' : h' < 2) (hj' : j' < 4)
    (e : 4 * h + j = 4 * h' + j') : h = h' ∧ j = j' := by
  omega

/-- Q3_K exponents stay in a byte. -/
theorem q3_K_mask_lt (h j : Nat) (hh : h < 2) (hj : j < 4) : 4 * h + j < 8 := by
  omega

/-- Q3_K scale index `8*half + 2*j + (0|1)` stays below 16. -/
theorem q3_K_scale_index (h j s : Nat) (hh : h < 2) (hj : j < 4) (hs : s < 2) :
    8 * h + 2 * j + s < 16 := by
  omega

/-- Q3_K value: 2 low bits minus 4 when the high bit is clear lies in [-4, 3]. -/
theorem q3_K_value_range (lo : Nat) (hbit : Bool) (hlo : lo < 4) :
    -4 ≤ (lo : Int) - (if hbit then 0 else 4) ∧ (lo : Int) - (if hbit then 0 else 4) ≤ 3 := by
  cases hbit <;> simp <;> omega

-- § IQ4_NL / IQ4_XS

/-- The IQ4_NL codebook, verbatim from `kvalues_iq4nl`. -/
def kvaluesIq4nl : List Int :=
  [-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113]

/-- The codebook has 16 entries (one per nibble). -/
theorem kvalues_length : kvaluesIq4nl.length = 16 := by decide

/-- The codebook is strictly increasing. -/
theorem kvalues_strictly_increasing : kvaluesIq4nl.Pairwise (· < ·) := by decide

/-- All codebook values are in [-127, 113]. -/
theorem kvalues_range : kvaluesIq4nl.all (fun x => decide (-127 ≤ x ∧ x ≤ 113)) = true := by decide

/-- There is no `-8` shift: code 8 decodes to 1 and code 0 to -127. -/
theorem kvalues_no_offset : kvaluesIq4nl[8]? = some 1 ∧ kvaluesIq4nl[0]? = some (-127) := by decide

/-- IQ4_XS 6-bit scale assembled from low nibble and two high bits. -/
def iq4xsScale (lowByte hi16 ib : Nat) : Nat :=
  ((lowByte / 2 ^ (4 * (ib % 2))) % 16) + ((hi16 / 2 ^ (2 * ib)) % 4) * 16

/-- The assembled scale is below 64, so `ls - 32` lies in [-32, 31]. -/
theorem iq4xsScale_lt (lowByte hi16 ib : Nat) : iq4xsScale lowByte hi16 ib < 64 := by
  have h1 : (lowByte / 2 ^ (4 * (ib % 2))) % 16 < 16 := Nat.mod_lt _ (by decide)
  have h2 : (hi16 / 2 ^ (2 * ib)) % 4 < 4 := Nat.mod_lt _ (by decide)
  unfold iq4xsScale
  omega

/-- IQ4_XS group indexing: `ib/2 < 4`, nibble shift 0 or 4, high shift at most 14. -/
theorem iq4xs_group_index (ib : Nat) (hib : ib < 8) :
    ib / 2 < 4 ∧ (4 * (ib % 2) = 0 ∨ 4 * (ib % 2) = 4) ∧ 2 * ib ≤ 14 := by
  omega

end Strata.Dequant
