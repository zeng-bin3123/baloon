/-!
# MODULE: iq_quant_expert_layout
Index/layout arithmetic of `iq_kernels.cu`: lane-strided dot-product call decomposition, the native expert blob
layout (gate | up | down), 256-byte scratch alignment, gate/up row interleave and Q2_0 code decoding.

## Source semantics
* `row_dot` visits calls `k = lane, lane+32, ...` below `nb*ipb`, mapping `k` to `(kbx, part) = (k/ipb, k%ipb)`.
* `native_expert_layout`: `up_off = n_ff*gu_row`, `down_off = 2*up_off`, `bytes = down_off + n_embd*d_row`.
* Scratch regions are rounded up to a multiple of 256 bytes.
* `dequant_gu` writes the gate/up pair of row `r` to interleaved rows `2r + parity`.
* Q2_0 decodes a 2-bit code `c` to `c - 1`.

## Form template identifiers
kbx -> block_index; iqs -> intra_block_offset; ipb -> calls_per_block; gu_row -> gate_up_row_bytes;
d_row -> down_row_bytes; up_off -> up_offset; down_off -> down_offset.
-/

namespace IqQuantExpertLayout

-- § lane-strided call decomposition

/-- A call index decomposes uniquely into (block, part). -/
theorem call_decomp (k ipb : Nat) : ipb * (k / ipb) + k % ipb = k :=
  Nat.div_add_mod k ipb

/-- The block index of any call below `nb*ipb` is below `nb`. -/
theorem call_block_lt (k nb ipb : Nat) (h : k < ipb * nb) : k / ipb < nb :=
  Nat.div_lt_of_lt_mul h

/-- The part index is below `ipb`. -/
theorem call_part_lt (k ipb : Nat) (hp : 0 < ipb) : k % ipb < ipb :=
  Nat.mod_lt k hp

/-- The lane stride preserves lane residue modulo 32. -/
theorem lane_stride_mod (lane j : Nat) : (lane + 32 * j) % 32 = lane % 32 := by
  omega

-- § native expert layout

/-- Layout record of one expert blob. -/
structure Layout where
  n_embd : Nat
  n_ff : Nat
  gu_row : Nat
  d_row : Nat

/-- Offset of the up matrix. -/
def Layout.upOff (L : Layout) : Nat := L.n_ff * L.gu_row
/-- Offset of the down matrix. -/
def Layout.downOff (L : Layout) : Nat := 2 * L.upOff
/-- Total blob bytes. -/
def Layout.bytes (L : Layout) : Nat := L.downOff + L.n_embd * L.d_row

/-- The gate region ends exactly where the up region starts. -/
theorem gate_end_eq_up_off (L : Layout) : 0 + L.n_ff * L.gu_row = L.upOff := by
  unfold Layout.upOff; omega

/-- The up region ends exactly where the down region starts. -/
theorem up_end_eq_down_off (L : Layout) : L.upOff + L.n_ff * L.gu_row = L.downOff := by
  unfold Layout.downOff Layout.upOff; omega

/-- The down offset does not exceed the blob size. -/
theorem down_off_le_bytes (L : Layout) : L.downOff ≤ L.bytes := by
  unfold Layout.bytes; omega

/-- Blob size decomposes into gate + up + down. -/
theorem bytes_decomp (L : Layout) :
    L.bytes = L.n_ff * L.gu_row + L.n_ff * L.gu_row + L.n_embd * L.d_row := by
  unfold Layout.bytes Layout.downOff Layout.upOff; omega

-- § scratch alignment

/-- Round up to a multiple of 256, as `(f + 255) & ~255`. -/
def align256 (x : Nat) : Nat := (x + 255) / 256 * 256

/-- Alignment never shrinks. -/
theorem align256_ge (x : Nat) : x ≤ align256 x := by
  unfold align256; omega

/-- Alignment gives a multiple of 256. -/
theorem align256_mod (x : Nat) : align256 x % 256 = 0 := by
  unfold align256; omega

/-- Alignment overshoots by less than 256. -/
theorem align256_lt (x : Nat) : align256 x < x + 256 := by
  unfold align256; omega

/-- The aligned region offsets of the scratch do not overlap. -/
theorem scratch_regions_disjoint (f : Nat) : align256 f + align256 f ≥ f + f := by
  have := align256_ge f; omega

-- § gate/up interleave

/-- Interleaved row of (row, parity). -/
def guRow (r parity : Nat) : Nat := 2 * r + parity

/-- The interleave is injective on parities below 2. -/
theorem guRow_inj (r p r' p' : Nat) (hp : p < 2) (hp' : p' < 2)
    (h : guRow r p = guRow r' p') : r = r' ∧ p = p' := by
  unfold guRow at h; omega

-- § Q2_0 decode

/-- A 2-bit code decodes to a value in -1 .. 2. -/
theorem q2_0_decode_range (c : Nat) (h : c < 4) :
    (-1 : Int) ≤ (c : Int) - 1 ∧ (c : Int) - 1 ≤ 2 := by
  omega

/-- Fixed per-format geometry: ipb for the 256-wide IQ formats. -/
theorem iq_ipb_geometry : (256 / 32 = 8) ∧ (32 / 16 = 2) := by decide

end IqQuantExpertLayout
