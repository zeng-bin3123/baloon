/-!
# MODULE: iq_kernels_geometry
The IQ-family / Q3_K / Q2_0 CUDA kernels (`iq_kernels.cu`): the lane-strided dot-call enumeration of
`row_dot`, the grouped native expert layout and scratch carving, the gate/up row mapping, the dequantiser
output positions, the interleaved gate/up dequant layout and the q8_1 quantiser indexing.

## Source semantics
* A format has `qk` values per block, `ipb` dot calls per block and `step` between `iqs` values.
  Dot call `k` is `(kbx, iqs) = (k / ipb, step * (k % ipb))`, lane-strided by 32, against activation blocks
  `x + kbx * (qk / 32)`.
* Expert blob: gate rows, then up rows, then down rows; `up_off = n_ff*gu_row`, `down_off = 2*up_off`.
* Dequantiser thread `tid < 32` of a 256-value superblock writes disjoint positions covering all 256.
* `dequant_gu`: output block of `(r, c, parity)` is `(2r+parity)*per_row + c`.

## Form template identifiers
- `Fmt<TY>::qk/ipb/step` → `qk/ipb/step`; `NativeExpertLayout` fields → `Layout`
- `GU_ROWS` → `guRows`; `kvalues_iq4nl` → (module `SGemvQuadDecode.kIq4Nl`)
- `block_q8_1` → 36 bytes
-/

namespace Strata.IqKernels

-- § Dot-call enumeration

/-- Call index decomposes into block and part: `k = kbx*ipb + part`. -/
theorem call_decompose (k ipb : Nat) (h : 0 < ipb) : (k / ipb) * ipb + k % ipb = k := by
  have := Nat.div_add_mod k ipb
  rw [Nat.mul_comm] at this
  omega

/-- Part index is below `ipb`. -/
theorem part_lt (k ipb : Nat) (h : 0 < ipb) : k % ipb < ipb := Nat.mod_lt k h

/-- The `iqs` values of one block are distinct for distinct parts (step > 0). -/
theorem iqs_distinct (step p1 p2 : Nat) (h : 0 < step) (heq : step * p1 = step * p2) : p1 = p2 :=
  Nat.eq_of_mul_eq_mul_left h heq

/-- Format table values: `(qk, ipb, step)`; iqs max stays within the block's `qi` ints. -/
theorem iqs_max :
    2 * (8 - 1) < 16 ∧ 1 * (8 - 1) < 8 ∧ 2 * (2 - 1) < 4 ∧ 1 * (2 - 1) < 2 := by decide

/-- Activation q8_1 blocks per weight block: 8 for 256-blocks, 2 for Q2_0, 1 for IQ4_NL. -/
theorem q8_blocks_per_block : 256 / 32 = 8 ∧ 64 / 32 = 2 ∧ 32 / 32 = 1 := by decide

/-- Calls per row: `nb * ipb`; every call below that count has a valid block. -/
theorem call_block_lt (k nb ipb : Nat) (h : k < nb * ipb) (hp : 0 < ipb) : k / ipb < nb :=
  Nat.div_lt_of_lt_mul (by rw [Nat.mul_comm]; exact h)

/-- Lane stride: call `k` belongs to lane `k % 32`, iteration `k / 32`; unique. -/
theorem lane_call_unique (k : Nat) : k = k % 32 + 32 * (k / 32) := by omega

/-- Warp-per-row blocks: 4 rows per block, `threadIdx.y` selects the row; grid covers all rows. -/
theorem mmvq_grid (n_out : Nat) : n_out ≤ 4 * ((n_out + 3) / 4) := by omega

/-- Row of `(blockIdx.x, threadIdx.y)` is injective on `ty < 4`. -/
theorem mmvq_row_inj (b1 y1 b2 y2 : Nat) (h1 : y1 < 4) (h2 : y2 < 4) (h : b1 * 4 + y1 = b2 * 4 + y2) :
    b1 = b2 ∧ y1 = y2 := by omega

/-- Output index of column `c`, row `r`: `c*n_out + r`. -/
theorem y_index_lt (c r n_out ncols : Nat) (hc : c < ncols) (hr : r < n_out) :
    c * n_out + r < ncols * n_out := by
  have : (c + 1) * n_out ≤ ncols * n_out := Nat.mul_le_mul_right n_out hc
  rw [Nat.add_mul] at this
  omega

-- § Block sizes (components of the ggml structs)

/-- iq2_xxs: d + 32 u16 = 66. -/
theorem iq2xxs_size : 2 + 2 * 32 = 66 := by decide
/-- iq2_xs: d + 32 u16 + 8 scales = 74. -/
theorem iq2xs_size : 2 + 2 * 32 + 8 = 74 := by decide
/-- iq2_s: d + 64 qs (incl. signs) + 8 qh + 8 scales = 82. -/
theorem iq2s_size : 2 + 64 + 8 + 8 = 82 := by decide
/-- iq3_xxs: d + 3*256/8 = 98. -/
theorem iq3xxs_size : 2 + 3 * 256 / 8 = 98 := by decide
/-- iq3_s: d + qs 64 + qh 8 + signs 32 + scales 4 = 110. -/
theorem iq3s_size : 2 + 64 + 8 + 32 + 4 = 110 := by decide
/-- iq1_m: qs 32 + qh 16 + scales 8 = 56. -/
theorem iq1m_size : 32 + 16 + 8 = 56 := by decide
/-- q3_K = 110, iq4_xs = 136, iq4_nl = 18, q2_0 = 18. -/
theorem other_sizes : 32 + 64 + 12 + 2 = 110 ∧ 2 + 2 + 4 + 128 = 136 ∧ 2 + 16 = 18 := by decide

-- § Native expert layout

/-- Alignment to 256 bytes. -/
def align256 (x : Nat) : Nat := (x + 255) / 256 * 256

/-- Rounding up to 256 is above, within 255, and aligned. -/
theorem align256_spec (x : Nat) : x ≤ align256 x ∧ align256 x < x + 256 ∧ align256 x % 256 = 0 := by
  unfold align256; omega

/-- Expert blob offsets from the per-row byte counts. -/
structure Layout where
  guRow : Nat
  dRow : Nat
  nEmbd : Nat
  nFF : Nat

/-- Up-projection offset. -/
def Layout.upOff (L : Layout) : Nat := L.nFF * L.guRow
/-- Down-projection offset. -/
def Layout.downOff (L : Layout) : Nat := 2 * L.upOff
/-- Total bytes. -/
def Layout.bytes (L : Layout) : Nat := L.downOff + L.nEmbd * L.dRow

/-- The three regions are ordered. -/
theorem layout_ordered (L : Layout) : L.upOff ≤ L.downOff ∧ L.downOff ≤ L.bytes := by
  unfold Layout.bytes Layout.downOff
  omega

/-- Gate and up regions have equal size, so the down region starts at twice the up offset. -/
theorem layout_gate_up_equal (L : Layout) : L.downOff - L.upOff = L.upOff := by
  unfold Layout.downOff; omega

/-- A gate row `r < n_ff` lies entirely below `up_off`. -/
theorem gate_row_in_region (L : Layout) (r : Nat) (h : r < L.nFF) : r * L.guRow + L.guRow ≤ L.upOff := by
  have : (r + 1) * L.guRow ≤ L.nFF * L.guRow := Nat.mul_le_mul_right L.guRow h
  rw [Nat.add_mul] at this
  unfold Layout.upOff
  omega

/-- An up row lies between `up_off` and `down_off`. -/
theorem up_row_in_region (L : Layout) (r : Nat) (h : r < L.nFF) :
    L.upOff + r * L.guRow + L.guRow ≤ L.downOff := by
  have := gate_row_in_region L r h
  unfold Layout.downOff
  omega

/-- A down row lies between `down_off` and `bytes`. -/
theorem down_row_in_region (L : Layout) (r : Nat) (h : r < L.nEmbd) :
    L.downOff + r * L.dRow + L.dRow ≤ L.bytes := by
  have : (r + 1) * L.dRow ≤ L.nEmbd * L.dRow := Nat.mul_le_mul_right L.dRow h
  rw [Nat.add_mul] at this
  unfold Layout.bytes
  omega

/-- Scratch: three float planes (gate, up, h), each aligned to 256, then the q8_1 image. -/
def scratchBytes (planeBytes q8Bytes : Nat) : Nat := 3 * align256 planeBytes + align256 q8Bytes

/-- The carved regions are in order and the total covers them: offsets `0, fa, 2fa, 3fa`. -/
theorem scratch_covers (plane q8 : Nat) :
    plane ≤ align256 plane ∧ q8 ≤ align256 q8 ∧
      3 * align256 plane + q8 ≤ scratchBytes plane q8 := by
  have h1 := align256_spec plane
  have h2 := align256_spec q8
  unfold scratchBytes
  omega

/-- A q8_1 block is 36 bytes, so every block is 4-byte aligned from a 256-aligned base. -/
theorem q81_block_aligned (b : Nat) : (b * 36) % 4 = 0 := by omega

/-- Number of q8_1 blocks of `cap` entries of `32*m` floats: `cap*m` (n_ff multiple of 32). -/
theorem q81_blocks_per_entries (cap m : Nat) : cap * (32 * m) / 32 = cap * m := by
  have : cap * (32 * m) = 32 * (cap * m) := by rw [Nat.mul_left_comm]
  rw [this]
  exact Nat.mul_div_cancel_left _ (by decide)

-- § Gate/up row mapping inside a group

/-- Row `row ∈ [0, 2*n_ff)` is gate when `row < n_ff` else up with `r = row - n_ff`. -/
theorem row_split_cover (row nFF : Nat) (h : row < 2 * nFF) :
    (row < nFF ∧ row < nFF) ∨ (nFF ≤ row ∧ row - nFF < nFF) := by omega

/-- The `(plane, r)` pair recomposes the row, so distinct rows write distinct slots. -/
theorem row_recompose (a nFF : Nat) (h : a < 2 * nFF) :
    a = (if a < nFF then 0 else nFF) + (if a < nFF then a else a - nFF) := by
  split_ifs <;> omega

/-- The per-entry output slot `e*n_ff + r` stays below `cap*n_ff`. -/
theorem entry_slot_lt (e r nFF cap : Nat) (he : e < cap) (hr : r < nFF) : e * nFF + r < cap * nFF := by
  have : (e + 1) * nFF ≤ cap * nFF := Nat.mul_le_mul_right nFF he
  rw [Nat.add_mul] at this
  omega

/-- Gate/up grid: eight warps per block, grid `ceil(2*n_ff/8)`. -/
theorem gu_grid (nFF : Nat) : 2 * nFF ≤ 8 * ((2 * nFF + 7) / 8) := by omega

/-- Down grid: `r = blockIdx.x*8 + warp` covers `n_embd` rows. -/
theorem down_grid (nEmbd : Nat) : nEmbd ≤ 8 * ((nEmbd + 7) / 8) := by omega

-- § Dequantiser output positions

/-- Position written by thread `tid` (0..31), element `j < 8` of an 8-wide format. -/
def pos8 (tid j : Nat) : Nat := 32 * (tid % 8) + 8 * (tid / 8) + j

/-- All 8-wide positions are inside the 256-value superblock. -/
theorem pos8_lt (tid j : Nat) (ht : tid < 32) (hj : j < 8) : pos8 tid j < 256 := by
  unfold pos8; omega

/-- Distinct `(tid, j)` write distinct positions: the dequantiser has no write conflicts. -/
theorem pos8_inj (t j t' j' : Nat) (ht : t < 32) (ht' : t' < 32) (hj : j < 8) (hj' : j' < 8)
    (h : pos8 t j = pos8 t' j') : t = t' ∧ j = j' := by
  unfold pos8 at h; omega

/-- 32 threads x 8 elements = 256 values: the writes are a bijection onto the superblock. -/
theorem pos8_cover : 32 * 8 = 256 := by decide

/-- Position for the IQ4 formats: `32*ib + 4*il + j` and `+16`, `j < 4`, `half < 2`. -/
def pos4 (tid j half : Nat) : Nat := 32 * (tid % 8) + 4 * (tid / 8) + j + 16 * half

/-- IQ4 positions are in range. -/
theorem pos4_lt (tid j half : Nat) (ht : tid < 32) (hj : j < 4) (hh : half < 2) : pos4 tid j half < 256 := by
  unfold pos4; omega

/-- IQ4 positions are collision-free. -/
theorem pos4_inj (t j h t' j' h' : Nat) (ht : t < 32) (ht' : t' < 32) (hj : j < 4) (hj' : j' < 4)
    (hh : h < 2) (hh' : h' < 2) (heq : pos4 t j h = pos4 t' j' h') : t = t' ∧ j = j' ∧ h = h' := by
  unfold pos4 at heq
  have ha : 4 * (t / 8) + j + 16 * h < 32 := by omega
  have hb : 4 * (t' / 8) + j' + 16 * h' < 32 := by omega
  have e1 : t % 8 = t' % 8 := by omega
  have e2 : 4 * (t / 8) + j + 16 * h = 4 * (t' / 8) + j' + 16 * h' := by omega
  have hc : 4 * (t / 8) + j < 16 := by omega
  have hd : 4 * (t' / 8) + j' < 16 := by omega
  have e3 : h = h' := by omega
  subst e3
  have e4 : 4 * (t / 8) + j = 4 * (t' / 8) + j' := by omega
  have e5 : t / 8 = t' / 8 := by omega
  have e6 : j = j' := by omega
  have e7 : t = t' := by omega
  exact ⟨e7, e6, rfl⟩

/-- Q2_0 dequantiser: block `tid/8` of four 64-blocks, part `tid%8` of 8 values. -/
def posQ2 (tid j : Nat) : Nat := (tid / 8) * 64 + (tid % 8) * 8 + j

/-- Q2_0 positions in range. -/
theorem posQ2_lt (tid j : Nat) (ht : tid < 32) (hj : j < 8) : posQ2 tid j < 256 := by
  unfold posQ2; omega

/-- Q2_0 positions collision-free. -/
theorem posQ2_inj (t j t' j' : Nat) (ht : t < 32) (ht' : t' < 32) (hj : j < 8) (hj' : j' < 8)
    (h : posQ2 t j = posQ2 t' j') : t = t' ∧ j = j' := by
  unfold posQ2 at h; omega

/-- Value `i` of a Q2_0 block: byte `i/4`, field `i%4`: they recompose the index. -/
theorem q2_field_index (i : Nat) : (i / 4) * 4 + i % 4 = i := by omega

-- § Interleaved gate/up dequant layout

/-- Stride uniqueness (offset below the period fixes the iteration). -/
theorem stride_inj (a b P m1 m2 : Nat) (hP : 0 < P) (ha : a < P) (hb : b < P)
    (h : a + P * m1 = b + P * m2) : a = b ∧ m1 = m2 := by
  have h1 : (a + P * m1) / P = m1 := by
    rw [Nat.add_mul_div_left _ _ hP, Nat.div_eq_of_lt ha, Nat.zero_add]
  have h2 : (b + P * m2) / P = m2 := by
    rw [Nat.add_mul_div_left _ _ hP, Nat.div_eq_of_lt hb, Nat.zero_add]
  rw [h] at h1
  have hm : m1 = m2 := h1.symm.trans h2
  subst hm
  omega

/-- `dequant_gu` writes superblock `c` of row `r` with parity `q` at block `c + per_row*(2r+q)`; injective. -/
theorem gu_interleave_inj (p r q c r' q' c' : Nat) (hp : 0 < p) (hc : c < p) (hc' : c' < p)
    (hq : q < 2) (hq' : q' < 2) (h : c + p * (2 * r + q) = c' + p * (2 * r' + q')) :
    r = r' ∧ q = q' ∧ c = c' := by
  have := stride_inj c c' p _ _ hp hc hc' h
  omega

/-- Input superblock `i` splits into `(r, c)` with `i = r*per_row + c`, `c < per_row`. -/
theorem superblock_split (i p : Nat) (hp : 0 < p) : i = p * (i / p) + i % p ∧ i % p < p := by
  have := Nat.div_add_mod i p
  exact ⟨by omega, Nat.mod_lt i hp⟩

-- § q8_1 quantiser indexing

/-- Block and in-block index recompose the element index. -/
theorem q81_index (i : Nat) : (i / 32) * 32 + i % 32 = i := by omega

/-- Quantiser grid covers all `n` elements with 256-thread blocks. -/
theorem q81_grid (n : Nat) : n ≤ 256 * ((n + 255) / 256) := by omega

/-- `iq_dequant_*` grids: one block per 256-value superblock; `n % 256 = 0` gives `n/256` whole superblocks. -/
theorem superblocks (n : Nat) (h : n % 256 = 0) : n / 256 * 256 = n := by omega

/-- Embedding rows: token `t`, block `b` writes at `t*n_embd + b*256`, inside the output. -/
theorem embed_write_lt (t b nEmbd nTok : Nat) (ht : t < nTok) (hb : b * 256 + 256 ≤ nEmbd) :
    t * nEmbd + b * 256 + 255 < nTok * nEmbd := by
  have : (t + 1) * nEmbd ≤ nTok * nEmbd := Nat.mul_le_mul_right nEmbd ht
  rw [Nat.add_mul] at this
  omega

end Strata.IqKernels
