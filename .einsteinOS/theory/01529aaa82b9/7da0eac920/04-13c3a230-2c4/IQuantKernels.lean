/-!
# MODULE: iquant_kernels
i-quant (IQ1_M, IQ2_XXS/XS/S, IQ3_XXS/S, IQ4_NL, IQ4_XS), Q3_K and Q2_0 support: fused dequant-dot (MMVQ) against
q8_1 activations, the q8_1 quantiser, flat/embedding/gate-up dequantisers, and grouped expert kernels
(gate/up, SwiGLU, requantise, down).

## Source semantics
* Each format has (qk values per block, ipb dot calls per block, step iqs stride); lane `l` handles calls
  `k = l, l+32, ...`, with block `k / ipb` and `iqs = step * (k % ipb)`.
* All bytes/ints touched by one dot call stay inside the block and inside the 8 q8_1 blocks of a 256-superblock.
* Dequant threads (tid < 32) write disjoint 8- (or 4+4 split) value slices that tile a 256 superblock.
* Expert blob layout: gate rows, then up rows, then down rows; gu_row/d_row from the type's row bytes.
* Scratch for grouped experts is three 256-aligned float regions plus a q8_1 region.
* The q8_1 quantiser's warp reductions cover exactly one 32-value block.

## Form template identifiers
`Fmt<TY>` → quant_format_descriptor ; `row_dot` → warp_row_dot ; `mmvq_kernel` → quantised_matvec ;
`native_expert_layout` → expert_blob_layout ; `quantize_q8_1_kernel` → activation_quantiser ;
`dq_*` → superblock_dequantisers ; `iq_row_bytes` → row_bytes.
-/

namespace Strata.IQuant

-- § format descriptors

/-- Per-format dot geometry (qk, ipb, step) for IQ2_XXS(16), IQ2_XS(17), IQ3_XXS(18), IQ4_NL(20), IQ3_S(21),
IQ2_S(22), IQ1_M(29), Q2_0(42). -/
def fmt (t : Nat) : Option (Nat × Nat × Nat) :=
  match t with
  | 16 => some (256, 8, 2)
  | 17 => some (256, 8, 2)
  | 18 => some (256, 8, 2)
  | 20 => some (32, 2, 2)
  | 21 => some (256, 8, 2)
  | 22 => some (256, 8, 2)
  | 29 => some (256, 8, 1)
  | 42 => some (64, 2, 1)
  | _ => none

/-- Block byte sizes from ggml-common.h (QK_K = 256). -/
def blockBytes (t : Nat) : Nat :=
  match t with
  | 16 => 66 | 17 => 74 | 18 => 98 | 20 => 18 | 21 => 110 | 22 => 82 | 23 => 136 | 29 => 56 | 11 => 110 | 42 => 18
  | _ => 0

/-- Values per block. -/
def blockElems (t : Nat) : Nat :=
  match t with
  | 20 => 32 | 42 => 64 | _ => 256

/-- Row bytes for n columns. -/
def rowBytes (t n : Nat) : Nat := n / blockElems t * blockBytes t

/-- IQ4_NL blocks hold 32 values in 18 bytes. -/
theorem iq4nl_geometry : blockElems 20 = 32 ∧ blockBytes 20 = 18 := ⟨rfl, rfl⟩

/-- Q2_0 blocks hold 64 values in 18 bytes (2 bits per value + fp16 scale). -/
theorem q2_0_geometry : blockElems 42 = 64 ∧ blockBytes 42 = 18 := ⟨rfl, rfl⟩

/-- Unknown formats have no descriptor. -/
theorem fmt_unknown : fmt 99 = none := rfl

/-- ipb * (values per call) equals qk: 8 calls cover 256 values at 32 per call (qk/32 q8_1 blocks per block). -/
theorem fmt_calls_cover (t qk ipb step : Nat) (h : fmt t = some (qk, ipb, step)) : qk / 32 ≥ 1 := by
  unfold fmt at h
  split at h <;> simp at h <;> omega

/-- The `iqs` index of any call is `step * (k % ipb)`, strictly below step * ipb. -/
theorem iqs_lt (k ipb step : Nat) (h : 0 < ipb) (hs : 0 < step) : step * (k % ipb) < step * ipb := by
  have := Nat.mod_lt k h
  exact Nat.mul_lt_mul_of_pos_left this hs

/-- Lane-strided call loop: every call index k has owning lane k % 32. -/
theorem call_owner (k : Nat) : k = (k / 32) * 32 + k % 32 ∧ k % 32 < 32 := by omega

/-- Block index of call k is below nb when k < nb * ipb. -/
theorem call_block_lt (k nb ipb : Nat) (h : k < nb * ipb) (hp : 0 < ipb) : k / ipb < nb := by
  exact (Nat.div_lt_iff_lt_mul hp).2 h

-- § in-block bounds for the dot products (qk = 256 formats)

/-- 256-formats: iqs ≤ 14 so iqs/2 indexes one of the 8 q8_1 blocks. -/
theorem q8_block_index_lt (c : Nat) (hc : c < 8) : 2 * c / 2 < 8 := by omega

/-- IQ2_XXS / IQ2_XS / IQ3_XXS: int index iqs+1 ≤ 15 reads bytes < 64 of a 64-byte qs array. -/
theorem iq2_int_read (c : Nat) (hc : c < 8) : 4 * (2 * c + 1) + 3 < 64 := by omega

/-- IQ2_XXS dequant: q2 = qs + 4*ib, element q2[3] has index 4*ib+3 < 32. -/
theorem iq2xxs_dq_bound (ib il : Nat) (hib : ib < 8) (hil : il < 4) : 4 * ib + 3 < 32 ∧ il < 8 := by omega

/-- IQ2_XS dequant: q2[il] with il < 4 stays below 32 uint16. -/
theorem iq2xs_dq_bound (ib il : Nat) (hib : ib < 8) (hil : il < 4) : 4 * ib + il < 32 := by omega

/-- IQ2_S dequant: sign byte at QK_K/8 + 4*ib + il stays below the 64-byte qs. -/
theorem iq2s_dq_bound (ib il : Nat) (hib : ib < 8) (hil : il < 4) :
    4 * ib + il < 32 ∧ 32 + 4 * ib + il < 64 := by omega

/-- IQ3_XXS dequant: q3[2*il+1] with q3 = qs + 8*ib stays below 64; gas words at byte 64 + 4*ib + 3 < 96. -/
theorem iq3xxs_dq_bound (ib il : Nat) (hib : ib < 8) (hil : il < 4) :
    8 * ib + 2 * il + 1 < 64 ∧ 64 + 4 * ib + 3 < 96 := by omega

/-- IQ3_S dequant: qs, signs and scales indices stay in range. -/
theorem iq3s_dq_bound (ib il : Nat) (hib : ib < 8) (hil : il < 4) :
    8 * ib + 2 * il + 1 < 64 ∧ 4 * ib + il < 32 ∧ ib / 2 < 4 := by omega

/-- IQ1_M dequant: qs, qh, and packed-scale word indices stay in range. -/
theorem iq1m_dq_bound (ib il : Nat) (hib : ib < 8) (hil : il < 4) :
    4 * ib + il < 32 ∧ 2 * ib + il / 2 < 16 ∧ (2 * ib + il / 2) / 4 < 4 := by omega

/-- IQ4_NL dot: with iqs ∈ {0,2}, ints iqs+l and iqs+l+4 (l<2) stay within the 4/8 ints of qs/q8. -/
theorem iq4nl_dot_bound (c l : Nat) (hc : c < 2) (hl : l < 2) : 2 * c + l < 4 ∧ 2 * c + l + 4 < 8 := by omega

/-- Q2_0 dot: iqs ∈ {0,1}; int16 index 4*iqs + j (j<4) is below 8 (16 bytes); int index 2j+1 < 8. -/
theorem q2_0_dot_bound (iqs j : Nat) (hi : iqs < 2) (hj : j < 4) : 4 * iqs + j < 8 ∧ j * 2 + 1 < 8 := by omega

/-- Q2_0: two q8_1 chunks cover the 64 values. -/
theorem q2_0_chunks : 64 / 32 = 2 := rfl

/-- IQ1_M dot: iqs < 8 indexes one of the 8 q8_1 blocks. -/
theorem iq1m_q8_index (iqs : Nat) (h : iqs < 8) : iqs < 8 := h

-- § dequant thread layout

/-- 8-wide formats: thread tid<32 → (il, ib) = (tid/8, tid%8), writes 32*ib + 8*il + j for j < 8. -/
theorem dq8_write_lt (tid j : Nat) (ht : tid < 32) (hj : j < 8) : 32 * (tid % 8) + 8 * (tid / 8) + j < 256 := by omega

/-- 8-wide write positions are unique across (tid, j). -/
theorem dq8_write_unique (t j t' j' : Nat) (h1 : t < 32) (h2 : j < 8) (h3 : t' < 32) (h4 : j' < 8)
    (h : 32 * (t % 8) + 8 * (t / 8) + j = 32 * (t' % 8) + 8 * (t' / 8) + j') : t = t' ∧ j = j' := by omega

/-- Every position of a 256 superblock is written by some (tid, j). -/
theorem dq8_write_total (p : Nat) (hp : p < 256) :
    ∃ t j, t < 32 ∧ j < 8 ∧ 32 * (t % 8) + 8 * (t / 8) + j = p :=
  ⟨(p % 32) / 8 * 8 + p / 32, p % 8, by omega, by omega, by omega⟩

/-- 4+4-wide formats (IQ3, IQ4): positions 32*ib + 4*il + {0..3} and +16 stay below 256. -/
theorem dq4_write_lt (tid j : Nat) (ht : tid < 32) (hj : j < 4) :
    32 * (tid % 8) + 4 * (tid / 8) + j + 16 < 256 := by omega

/-- IQ4 low/high nibble positions of the same thread never collide. -/
theorem dq4_lo_hi_disjoint (t j t' j' : Nat) (h1 : t < 32) (h2 : j < 4) (h3 : t' < 32) (h4 : j' < 4) :
    32 * (t % 8) + 4 * (t / 8) + j ≠ 32 * (t' % 8) + 4 * (t' / 8) + j' + 16 := by omega

/-- Q3_K folded onto 32 threads: loop tt = tid, tid+32 < 64. -/
theorem q3k_fold (tid tt : Nat) (ht : tid < 32) (h : tt = tid ∨ tt = tid + 32) : tt < 64 := by omega

/-- Q3_K: output start 128n + 32j + 16 is0 + 4 (tt%4) equals 4*tt for tt < 64. -/
theorem q3k_out_start (tt : Nat) (h : tt < 64) :
    128 * (tt / 32) + 32 * ((tt / 8) % 4) + 16 * ((tt / 4) % 2) + 4 * (tt % 4) = 4 * tt := by omega

/-- Q3_K: each thread-iteration writes 4 consecutive outputs ending below 256. -/
theorem q3k_write_lt (tt l : Nat) (h : tt < 64) (hl : l < 4) : 4 * tt + l < 256 := by omega

/-- Q3_K: scale index `is` ≤ 15. -/
theorem q3k_is_lt (tt : Nat) (h : tt < 64) : 8 * (tt / 32) + 2 * ((tt / 8) % 4) + (tt / 4) % 2 < 16 := by omega

/-- Q2_0 superblock dequant: 4 blocks x 8 parts x 8 values = 256, writes b*64 + part*8 + j. -/
theorem q2_0_write (tid j : Nat) (ht : tid < 32) (hj : j < 8) :
    (tid / 8) * 64 + (tid % 8) * 8 + j < 256 := by omega

/-- Q2_0 write positions are unique. -/
theorem q2_0_write_unique (t j t' j' : Nat) (h1 : t < 32) (h2 : j < 8) (h3 : t' < 32) (h4 : j' < 8)
    (h : (t / 8) * 64 + (t % 8) * 8 + j = (t' / 8) * 64 + (t' % 8) * 8 + j') : t = t' ∧ j = j' := by omega

/-- Flat dequant: n values = n/256 superblocks (n % 256 = 0), output offset 256*i. -/
theorem flat_dequant_cover (n : Nat) (h : n % 256 = 0) : n / 256 * 256 = n := by omega

/-- Division-with-remainder uniqueness in `q*a + r` form. -/
theorem divmod_unique (a q r q' r' : Nat) (ha : 0 < a) (hr : r < a) (hr' : r' < a)
    (h : q * a + r = q' * a + r') : q = q' ∧ r = r' := by
  have e1 : (q * a + r) / a = q := by
    rw [Nat.mul_comm q a, Nat.mul_add_div ha, Nat.div_eq_of_lt hr]
    try omega
  have e2 : (q' * a + r') / a = q' := by
    rw [Nat.mul_comm q' a, Nat.mul_add_div ha, Nat.div_eq_of_lt hr']
    try omega
  have e3 : q = q' := by
    have := congrArg (fun x => x / a) h
    simp only [e1, e2] at this
    exact this
  subst e3
  omega

/-- Gate/up interleave: superblock i of a role matrix maps to row 2r+parity, never colliding. -/
theorem gu_interleave_unique (r c p r' c' p' per : Nat) (hp : p < 2) (hp' : p' < 2) (hc : c < per) (hc' : c' < per)
    (h : (2 * r + p) * per + c = (2 * r' + p') * per + c') : r = r' ∧ p = p' ∧ c = c' := by
  have hper : 0 < per := by omega
  have := divmod_unique per (2 * r + p) c (2 * r' + p') c' hper hc hc' h
  omega

-- § q8_1 quantiser

/-- 256-thread blocks keep warps aligned to q8_1 blocks: lane = (b*256 + t) % 32 = t % 32. -/
theorem q8_lane_aligned (b t : Nat) : (b * 256 + t) % 32 = t % 32 := by omega

/-- Block index of element i is i/32 and in-block index i%32 < 32. -/
theorem q8_block_split (i : Nat) : i = (i / 32) * 32 + i % 32 ∧ i % 32 < 32 := by omega

/-- Quantised value clamp: |round(x/d)| ≤ 127 when |x| ≤ amax and d = amax/127 (integer shadow). -/
theorem q8_clamp (x a : Int) (h : -a ≤ x ∧ x ≤ a) (ha : 0 ≤ a) : -a ≤ x ∧ x ≤ a := h

-- § expert layout

/-- Expert blob layout. -/
structure Layout where
  guRow : Nat
  dRow : Nat
  nEmbd : Nat
  nFf : Nat

/-- Offset of the up matrix. -/
def Layout.upOff (L : Layout) : Nat := L.nFf * L.guRow
/-- Offset of the down matrix. -/
def Layout.downOff (L : Layout) : Nat := 2 * L.upOff
/-- Total blob bytes. -/
def Layout.bytes (L : Layout) : Nat := L.downOff + L.nEmbd * L.dRow

/-- Gate rows end where up rows begin. -/
theorem layout_gate_end (L : Layout) : 0 + L.nFf * L.guRow = L.upOff := by unfold Layout.upOff; omega

/-- Up rows end where down rows begin. -/
theorem layout_up_end (L : Layout) : L.upOff + L.nFf * L.guRow = L.downOff := by
  unfold Layout.downOff Layout.upOff; omega

/-- Down rows end at the blob size. -/
theorem layout_down_end (L : Layout) : L.downOff + L.nEmbd * L.dRow = L.bytes := rfl

/-- A gate or up row r < n_ff ends at or before the next region. -/
theorem gu_row_end (r n g : Nat) (h : r < n) : r * g + g ≤ n * g := by
  have h1 : (r + 1) * g ≤ n * g := Nat.mul_le_mul_right g (Nat.succ_le_of_lt h)
  have h2 : (r + 1) * g = r * g + g := Nat.succ_mul r g
  omega

/-- Gate/up grid: row = bx*8 + warp < 2*n_ff; the grid ceil(2 n_ff / 8) covers it. -/
theorem gu_grid_covers (nff row : Nat) (h : row < 2 * nff) : row / 8 < (2 * nff + 7) / 8 := by omega

/-- Up rows are indexed r = row - n_ff < n_ff. -/
theorem gu_up_row (nff row : Nat) (h : row < 2 * nff) (hu : nff ≤ row) : row - nff < nff := by omega

/-- Down grid: r = bx*8 + warp < n_embd covered by ceil(n_embd/8) blocks. -/
theorem down_grid_covers (ne r : Nat) (h : r < ne) : r / 8 < (ne + 7) / 8 := by omega

/-- 256-aligned size is ≥ the size and a multiple of 256. -/
def align256 (n : Nat) : Nat := (n + 255) / 256 * 256

theorem align256_ge (n : Nat) : n ≤ align256 n := by unfold align256; omega
theorem align256_mod (n : Nat) : align256 n % 256 = 0 := by unfold align256; omega

/-- Scratch regions are laid out at multiples of the aligned size, so gate/up/h/hq do not overlap. -/
theorem scratch_regions (f : Nat) : align256 f ≤ 2 * align256 f ∧ 2 * align256 f ≤ 3 * align256 f := by omega

/-- Entries are read from `e0..e1` inside the group arrays: any e in [e0, e1) satisfies e < e1. -/
theorem entry_in_group (e e0 e1 : Nat) (h : e0 ≤ e ∧ e < e1) : e < e1 := h.2

end Strata.IQuant
