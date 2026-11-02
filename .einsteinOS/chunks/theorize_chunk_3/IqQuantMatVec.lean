/-!
# MODULE: iq_quant_mat_vec
Layout and index contracts of the IQ-family (and Q2_0/Q3_K) quantized matrix-vector kernels transcribed from
llama.cpp: per-format block traits, the warp-strided row dot, the grouped native expert layout (gate | up | down
blobs), the q8_1 activation quantiser, and the thread mapping of the dequantisers.

## Source semantics
- A format has `qk` values per block, `ipb` dot calls per block and an `iqs` stride `step`; call k of a row maps to
  block `k / ipb`, part `k % ipb`, and the activation q8_1 blocks start at `kbx * (qk/32)`.
- Expert blob layout: gate rows [0, n_ff*gu_row), up rows [up_off, down_off), down rows [down_off, bytes) with
  `up_off = n_ff*gu_row`, `down_off = 2*up_off`, `bytes = down_off + n_embd*d_row`.
- q8_1 quantiser: block of 32, scale `d = amax/127`, code 0 when amax = 0.
- Dequantisers: 32 threads per 256-value superblock, thread tid writes 8 (or 4+4 split) values.

## Form template identifiers
- qk -> block_values, ipb -> calls_per_block, step -> iqs_step, gu_row -> gate_up_row_bytes, d_row -> down_row_bytes
- up_off -> up_offset, down_off -> down_offset, ent_tok -> entry_token, ent_dst -> entry_dest
-/

namespace Strata.IqQuant

-- § helpers

/-- Row-major index bound. -/
theorem idx_lt {a b A B : Nat} (ha : a < A) (hb : b < B) : a * B + b < A * B := by
  have h1 : (a + 1) * B ≤ A * B := Nat.mul_le_mul_right B (Nat.succ_le_of_lt ha)
  have h2 : (a + 1) * B = a * B + B := by rw [Nat.add_mul, Nat.one_mul]
  omega

-- § format table

/-- Bytes per block for each supported GGML type code (sizes from ggml-common.h). -/
def blockBytes : Nat → Nat
  | 16 => 66 | 17 => 74 | 18 => 98 | 20 => 18 | 21 => 110
  | 22 => 82 | 29 => 56 | 23 => 136 | 11 => 110 | 42 => 18
  | _ => 0

/-- Values per block for each supported type code. -/
def blockValues : Nat → Nat
  | 20 => 32 | 42 => 64 | 16 => 256 | 17 => 256 | 18 => 256 | 21 => 256
  | 22 => 256 | 29 => 256 | 23 => 256 | 11 => 256
  | _ => 0

/-- Types accepted by `is_iq`. -/
def iqTypes : List Nat := [16, 17, 18, 20, 21, 22, 23, 29, 42, 11]

/-- Every accepted type has a nonzero block size and nonzero block width. -/
theorem iq_types_have_layout : ∀ t ∈ iqTypes, 0 < blockBytes t ∧ 0 < blockValues t := by
  decide

/-- The accepted type list has no duplicates. -/
theorem iq_types_nodup : iqTypes.Nodup := by decide

/-- Row bytes: whole blocks per row times block bytes. -/
def rowBytes (t n : Nat) : Nat := n / blockValues t * blockBytes t

-- § row dot call enumeration

/-- Call k decomposes into (block, part). -/
theorem call_split (k ipb : Nat) : (k / ipb) * ipb + k % ipb = k := by
  have := Nat.div_add_mod k ipb
  rw [Nat.mul_comm] at this
  exact this

/-- Block index of a call below `nb*ipb` is below `nb`. -/
theorem call_block_lt (k nb ipb : Nat) (h : k < nb * ipb) (hp : 0 < ipb) : k / ipb < nb :=
  Nat.div_lt_of_lt_mul (by rw [Nat.mul_comm]; exact h)

/-- Part index is below `ipb`. -/
theorem call_part_lt (k ipb : Nat) (hp : 0 < ipb) : k % ipb < ipb := Nat.mod_lt _ hp

/-- The q8_1 blocks consumed by block kbx end inside the activation: (kbx+1)*m ≤ nb*m. -/
theorem q8_window_in_range (kbx nb m : Nat) (h : kbx < nb) : (kbx + 1) * m ≤ nb * m :=
  Nat.mul_le_mul_right m (Nat.succ_le_of_lt h)

/-- `nb` weight blocks of `32*m` values consume `nb*m` q8_1 blocks of 32 values. -/
theorem q8_blocks_per_row (nb m : Nat) : (nb * (32 * m)) / 32 = nb * m := by
  rw [Nat.mul_left_comm]
  exact Nat.mul_div_cancel_left _ (by decide)

/-- Per-format iqs ranges: ipb=8, step=2 gives iqs ≤ 14. -/
theorem iqs_range_step2 (part : Nat) (h : part < 8) : 2 * part ≤ 14 := by
  omega

/-- IQ1_M: ipb=8, step=1 gives iqs ≤ 7. -/
theorem iqs_range_step1 (part : Nat) (h : part < 8) : 1 * part ≤ 7 := by
  omega

/-- IQ4_NL: ipb=2, step=2 gives iqs ≤ 2 (two 4-int halves). -/
theorem iqs_range_iq4nl (part : Nat) (h : part < 2) : 2 * part ≤ 2 := by
  omega

-- § mmvq grid

/-- Four rows per block cover all n_out rows. -/
theorem mmvq_grid_covers (n_out : Nat) : n_out ≤ (n_out + 3) / 4 * 4 := by
  omega

/-- Row of thread (bx, ty) with ty < 4 is bx*4+ty. -/
theorem mmvq_row_split (row : Nat) : (row / 4) * 4 + row % 4 = row := by
  omega

/-- Output address for column c of row r is below `ncols*n_out`. -/
theorem mmvq_out_lt (c r ncols n_out : Nat) (hc : c < ncols) (hr : r < n_out) :
    c * n_out + r < ncols * n_out :=
  idx_lt hc hr

-- § grouped expert layout

/-- Expert blob layout. -/
structure Layout where
  nEmbd : Nat
  nFf : Nat
  guRow : Nat
  dRow : Nat

/-- Offset of the up matrix. -/
def Layout.upOff (L : Layout) : Nat := L.nFf * L.guRow
/-- Offset of the down matrix. -/
def Layout.downOff (L : Layout) : Nat := 2 * L.upOff
/-- Total blob size. -/
def Layout.bytes (L : Layout) : Nat := L.downOff + L.nEmbd * L.dRow

/-- Gate occupies [0, upOff) and up occupies [upOff, downOff): no overlap and no gap. -/
theorem gate_up_partition (L : Layout) : L.upOff + L.nFf * L.guRow = L.downOff := by
  unfold Layout.downOff Layout.upOff
  omega

/-- Down matrix starts exactly after up. -/
theorem down_after_up (L : Layout) : L.upOff + (L.downOff - L.upOff) = L.downOff := by
  unfold Layout.downOff Layout.upOff
  omega

/-- Every gate row ends inside the gate region. -/
theorem gate_row_in_region (L : Layout) (r : Nat) (h : r < L.nFf) :
    r * L.guRow + L.guRow ≤ L.upOff := by
  unfold Layout.upOff
  have this : (r + 1) * L.guRow ≤ L.nFf * L.guRow := Nat.mul_le_mul_right L.guRow (Nat.succ_le_of_lt h)
  have e : (r + 1) * L.guRow = r * L.guRow + L.guRow := by rw [Nat.add_mul, Nat.one_mul]
  omega

/-- Every up row ends inside the up region. -/
theorem up_row_in_region (L : Layout) (r : Nat) (h : r < L.nFf) :
    L.upOff + r * L.guRow + L.guRow ≤ L.downOff := by
  have := gate_row_in_region L r h
  unfold Layout.downOff
  omega

/-- Every down row ends inside the blob. -/
theorem down_row_in_blob (L : Layout) (r : Nat) (h : r < L.nEmbd) :
    L.downOff + r * L.dRow + L.dRow ≤ L.bytes := by
  unfold Layout.bytes
  have this : (r + 1) * L.dRow ≤ L.nEmbd * L.dRow := Nat.mul_le_mul_right L.dRow (Nat.succ_le_of_lt h)
  have e : (r + 1) * L.dRow = r * L.dRow + L.dRow := by rw [Nat.add_mul, Nat.one_mul]
  omega

/-- Gate/up kernel: row index in [0, 2*n_ff) splits into matrix selector and local row. -/
theorem gu_row_split (row n_ff : Nat) (h : row < 2 * n_ff) :
    (row < n_ff ∧ row < n_ff) ∨ (n_ff ≤ row ∧ row - n_ff < n_ff) := by
  omega

/-- Gate/up grid covers 2*n_ff rows with 8 rows per block. -/
theorem gu_grid_covers (n_ff : Nat) : 2 * n_ff ≤ (2 * n_ff + 7) / 8 * 8 := by
  omega

/-- Down grid covers n_embd rows with 8 rows per block. -/
theorem down_grid_covers (n_embd : Nat) : n_embd ≤ (n_embd + 7) / 8 * 8 := by
  omega

/-- Entry output `e*n_ff + r` stays inside `cap*n_ff`. -/
theorem entry_out_lt (e r cap n_ff : Nat) (he : e < cap) (hr : r < n_ff) :
    e * n_ff + r < cap * n_ff :=
  idx_lt he hr

-- § scratch layout

/-- Round up to a multiple of 256. -/
def alignUp256 (n : Nat) : Nat := (n + 255) / 256 * 256

/-- Alignment never shrinks. -/
theorem alignUp256_ge (n : Nat) : n ≤ alignUp256 n := by
  unfold alignUp256
  omega

/-- Aligned value is a multiple of 256. -/
theorem alignUp256_mod (n : Nat) : alignUp256 n % 256 = 0 := by
  unfold alignUp256
  omega

/-- Scratch regions gate/up/h each get `fa` bytes and `hq` starts at `3*fa`: the four regions are disjoint. -/
theorem scratch_regions (f : Nat) : f ≤ alignUp256 f ∧ alignUp256 f + alignUp256 f ≤ 2 * alignUp256 f ∧
    2 * alignUp256 f + alignUp256 f = 3 * alignUp256 f := by
  have := alignUp256_ge f
  omega

-- § q8_1 quantiser

/-- Element i belongs to block i/32 at position i%32. -/
theorem q8_block_split (i : Nat) : (i / 32) * 32 + i % 32 = i := by
  omega

/-- The position within the block is below 32. -/
theorem q8_pos_lt (i : Nat) : i % 32 < 32 := by
  omega

/-- Quantiser grid covers n elements with 256 threads per block. -/
theorem q8_grid_covers (n : Nat) : n ≤ (n + 255) / 256 * 256 := by
  omega

-- § dequant thread mapping

/-- 8-value formats: thread tid writes offsets `32*(tid%8) + 8*(tid/8) + j`, all below 256. -/
theorem dq8_offset_lt (tid j : Nat) (ht : tid < 32) (hj : j < 8) :
    32 * (tid % 8) + 8 * (tid / 8) + j < 256 := by
  omega

/-- 8-value formats: distinct (thread, j) pairs write distinct offsets. -/
theorem dq8_offset_inj (t j t' j' : Nat) (ht : t < 32) (hj : j < 8) (ht' : t' < 32) (hj' : j' < 8)
    (h : 32 * (t % 8) + 8 * (t / 8) + j = 32 * (t' % 8) + 8 * (t' / 8) + j') : t = t' ∧ j = j' := by
  omega

/-- 4+4 formats (iq4_nl/iq4_xs): offsets `32*ib + 4*il + j (+16)` are below 256. -/
theorem dq4_offset_lt (ib il j h : Nat) (hb : ib < 8) (hi : il < 4) (hj : j < 4) (hh : h < 2) :
    32 * ib + 4 * il + j + 16 * h < 256 := by
  omega

/-- 4+4 formats: distinct (ib, il, j, h) write distinct offsets. -/
theorem dq4_offset_inj (ib il j h ib' il' j' h' : Nat) (hb : ib < 8) (hi : il < 4) (hj : j < 4)
    (hh : h < 2) (hb' : ib' < 8) (hi' : il' < 4) (hj' : j' < 4) (hh' : h' < 2)
    (e : 32 * ib + 4 * il + j + 16 * h = 32 * ib' + 4 * il' + j' + 16 * h') :
    ib = ib' ∧ il = il' ∧ j = j' ∧ h = h' := by
  omega

/-- Q2_0 dequant: block b (0..3) and part p (0..7) write offsets `b*64 + p*8 + j` below 256. -/
theorem dq_q2_0_offset_lt (b p j : Nat) (hb : b < 4) (hp : p < 8) (hj : j < 8) :
    b * 64 + (p * 8 + j) < 256 := by
  omega

/-- Embedding rows: output address `t*n_embd + b*256 + o` stays inside `n_tok * n_embd`. -/
theorem embed_out_lt (t b o n_tok n_embd : Nat) (ht : t < n_tok) (hn : n_embd % 256 = 0)
    (hb : b < n_embd / 256) (ho : o < 256) : t * n_embd + (b * 256 + o) < n_tok * n_embd := by
  have h : b * 256 + o < n_embd := by omega
  have h2 := idx_lt ht h
  exact h2

/-- Gate/up interleaved dequant: row `2r+parity` with column block c stays inside `2*n_ff` rows. -/
theorem gu_interleave_lt (r parity c n_ff per_row : Nat) (hr : r < n_ff) (hp : parity < 2)
    (hc : c < per_row) : (2 * r + parity) * per_row + c < 2 * n_ff * per_row := by
  have h : 2 * r + parity < 2 * n_ff := by omega
  exact idx_lt h hc

-- § supported types

/-- Types accepted by `iq_mmvq` / grouped gate-up kernels route to a template instance. -/
def mmvqTypes : List Nat := [16, 17, 18, 20, 21, 22, 29, 42]

/-- The mmvq set is a subset of the accepted IQ set. -/
theorem mmvq_types_subset : ∀ t ∈ mmvqTypes, t ∈ iqTypes := by
  decide

/-- Down projection of the grouped experts accepts only types 20 and 42. -/
def downTypes : List Nat := [20, 42]

/-- Down types are mmvq types. -/
theorem down_types_subset : ∀ t ∈ downTypes, t ∈ mmvqTypes := by
  decide

end Strata.IqQuant
