/-!
# MODULE: gguf_writer_q2_contract
SOURCES: tools/gguf_writer.py
Integer model of the minimal GGUF v3 writer and the Q2_0 quantiser contract: type ids, alignment of the
tensor directory and data section, the Q2_0 code clamp and bit packing.

## Source semantics
Q2_0 block = 64 elements, fp16 scale d = amax, code = clamp(round(w/d)+1, 0, 3), symbol = code-1 in {-1,0,1,2};
codes are packed 4 per byte, LSB first (element j -> byte j/4, shift 2*(j%4)). Tensor data offsets are
rounded up to the alignment (32) after each tensor; the data section starts at an aligned offset.
Metadata and tensor shape order follow GGUF (dim 0 varies fastest).

## Form template identifiers
`TYPE_IDS` -> `typeIds`; `_META_IDS` -> `metaIds`; `DEFAULT_ALIGNMENT` -> 32; `quantize_q2_0` -> `clamp`/`packByte`;
`GGUFWriter.write` offset loop -> `nextOffset`/`offsetAfter`.
-/

namespace StrataGgufWriter

-- § type ids

/-- ggml type ids the writer can emit. -/
def typeIds : List (String × Nat) := [("F32", 0), ("F16", 1), ("Q2_0", 42)]

/-- Type ids are pairwise distinct. -/
theorem type_ids_nodup : (typeIds.map Prod.snd).Nodup := by decide

/-- Metadata value type ids of the GGUF spec. -/
def metaIds : List Nat := [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

/-- Metadata type ids are pairwise distinct. -/
theorem meta_ids_nodup : metaIds.Nodup := by decide

-- § Q2_0 quantiser

/-- Integer clamp used for the code. -/
def clamp (lo hi x : Int) : Int := max lo (min hi x)

/-- The clamp lands in [lo, hi]. -/
theorem clamp_range (lo hi x : Int) (h : lo ≤ hi) : lo ≤ clamp lo hi x ∧ clamp lo hi x ≤ hi := by
  unfold clamp; omega

/-- A Q2_0 code `clamp(r+1, 0, 3)` is always a valid 2-bit code. -/
theorem q2_0_code_valid (r : Int) : 0 ≤ clamp 0 3 (r + 1) ∧ clamp 0 3 (r + 1) ≤ 3 :=
  clamp_range 0 3 (r + 1) (by omega)

/-- With d = amax, |w/d| <= 1, so rounded values are in {-1,0,1} and the clamp is the identity: the +2 level is never produced. -/
theorem q2_0_amax_never_plus2 (r : Int) (h1 : -1 ≤ r) (h2 : r ≤ 1) :
    clamp 0 3 (r + 1) = r + 1 ∧ clamp 0 3 (r + 1) ≤ 2 := by
  unfold clamp; omega

/-- Symbol = code - 1 lies in {-1,0,1,2}. -/
theorem q2_0_symbol_range (r : Int) : -1 ≤ clamp 0 3 (r + 1) - 1 ∧ clamp 0 3 (r + 1) - 1 ≤ 2 := by
  unfold clamp; omega

/-- Four 2-bit codes in one byte, element 4b+t at shift 2t. -/
def packByte (c0 c1 c2 c3 : Nat) : Nat := c0 + 4 * c1 + 16 * c2 + 64 * c3

/-- A packed byte fits 8 bits and each code is recovered by shift and mask. -/
theorem packByte_roundtrip (c0 c1 c2 c3 : Nat) (h0 : c0 < 4) (h1 : c1 < 4) (h2 : c2 < 4) (h3 : c3 < 4) :
    packByte c0 c1 c2 c3 < 256 ∧ packByte c0 c1 c2 c3 % 4 = c0 ∧ packByte c0 c1 c2 c3 / 4 % 4 = c1 ∧
    packByte c0 c1 c2 c3 / 16 % 4 = c2 ∧ packByte c0 c1 c2 c3 / 64 % 4 = c3 := by
  unfold packByte; omega

/-- Element j of a block lives in byte j/4 < 16 at shift (j%4)*2 < 8. -/
theorem element_position (j : Nat) (hj : j < 64) : j / 4 < 16 ∧ (j % 4) * 2 < 8 ∧ 4 * (j / 4) + j % 4 = j := by
  omega

/-- A block is an 18-byte unit: 2 scale bytes + 16 code bytes. -/
theorem block_bytes : 2 + 64 / 4 = 18 := by decide

/-- Whole-block requirement: n multiple of 64 gives n/64 blocks covering n elements, in (n/64)*18 bytes. -/
theorem blocks_cover (n : Nat) (h : n % 64 = 0) : (n / 64) * 64 = n := by omega

-- § alignment

/-- Padding to a 32-byte boundary. -/
def pad32 (n : Nat) : Nat := (32 - n % 32) % 32

/-- Round up to a multiple of 32. -/
def alignUp32 (n : Nat) : Nat := (n + 31) / 32 * 32

/-- Padding is below 32 and makes the length aligned. -/
theorem pad32_spec (n : Nat) : pad32 n < 32 ∧ (n + pad32 n) % 32 = 0 := by
  unfold pad32; omega

/-- Padding and rounding up agree. -/
theorem pad32_eq_alignUp (n : Nat) : n + pad32 n = alignUp32 n := by
  unfold pad32 alignUp32; omega

/-- Next tensor offset: current offset plus the data, rounded up to the alignment. -/
def nextOffset (o len : Nat) : Nat := alignUp32 (o + len)

/-- Every tensor offset after the first is aligned. -/
theorem nextOffset_aligned (o len : Nat) : nextOffset o len % 32 = 0 := by
  unfold nextOffset alignUp32; omega

/-- A tensor never overlaps the next one. -/
theorem nextOffset_ge (o len : Nat) : o + len ≤ nextOffset o len := by
  unfold nextOffset alignUp32; omega

/-- Padding wastes less than 32 bytes. -/
theorem nextOffset_lt (o len : Nat) : nextOffset o len < o + len + 32 := by
  unfold nextOffset alignUp32; omega

/-- Offset after a whole list of tensors. -/
def offsetAfter (lens : List Nat) : Nat := lens.foldl nextOffset 0

/-- Appending a tensor advances the offset by `nextOffset`. -/
theorem offsetAfter_append (lens : List Nat) (l : Nat) :
    offsetAfter (lens ++ [l]) = nextOffset (offsetAfter lens) l := by
  simp [offsetAfter, List.foldl_append]

-- § shapes

/-- A fused expert tensor [in, out, experts] has in*out*experts elements; rows = out*experts. -/
theorem fused_rows (i o e : Nat) : i * (o * e) = i * o * e := (Nat.mul_assoc i o e).symm

/-- The concrete case from the source: [64, 256, 16] has 4096 rows of 64. -/
theorem fused_example : 64 * (256 * 16) = 64 * 4096 := by decide

end StrataGgufWriter
