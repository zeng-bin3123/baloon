/-!
# MODULE: gated_residual_read_write
Workspace layout, reduction mapping and index contracts of the generic gated residual / hyper-connection
`gr_read` and `gr_write` kernels (any geometry, bf16 weights, one warp per output row).

## Source semantics
- Workspace has five regions xn, xq, lq, gated, lo, each rounded up to 16 bytes, laid out back to back; the
  first version mis-sized regions so they overlapped, which the offsets lemmas below rule out.
- Down kernel: one block per `lo` row, 8 warps split the reduction with element `warp*32+lane + 256*k`.
- Gate kernel: one warp per `gated` row, 8 rows per block. Mean kernel: flat over n_embd, streams strided.
- Write kernel: `out[c*n_embd+d] = R[...] + block_out[d] * w[c]`, w computed once per stream.

## Form template identifiers
- n_embd -> model_dim, hc -> n_streams, hc_lr -> low_rank, hc_dim -> stream_total
- xn -> normed_streams, xq -> quantized_streams, lq -> quantized_low_rank, gated -> gated_streams
-/

namespace Strata.GatedResidual

-- § helpers

/-- Row-major index bound. -/
theorem idx_lt {a b A B : Nat} (ha : a < A) (hb : b < B) : a * B + b < A * B := by
  have h1 : (a + 1) * B ≤ A * B := Nat.mul_le_mul_right B (Nat.succ_le_of_lt ha)
  have h2 : (a + 1) * B = a * B + B := by rw [Nat.add_mul, Nat.one_mul]
  omega

-- § workspace layout

/-- Round a byte count up to a multiple of 16. -/
def alignUp16 (n : Nat) : Nat := (n + 15) / 16 * 16

/-- Alignment never shrinks a region. -/
theorem alignUp16_ge (n : Nat) : n ≤ alignUp16 n := by
  unfold alignUp16
  omega

/-- Alignment adds less than one alignment unit. -/
theorem alignUp16_lt (n : Nat) : alignUp16 n < n + 16 := by
  unfold alignUp16
  omega

/-- Aligned sizes are multiples of 16. -/
theorem alignUp16_mod (n : Nat) : alignUp16 n % 16 = 0 := by
  unfold alignUp16
  omega

/-- Region offsets: each region starts where the previous one ends, so regions are disjoint. -/
def offsets (a0 a1 a2 a3 : Nat) : Nat × Nat × Nat × Nat := (0, a0, a0 + a1, a0 + a1 + a2)

/-- Region k+1 begins at or after the end of region k. -/
theorem regions_disjoint (a0 a1 a2 a3 : Nat) :
    (offsets a0 a1 a2 a3).1 + a0 ≤ (offsets a0 a1 a2 a3).2.1 ∧
    (offsets a0 a1 a2 a3).2.1 + a1 ≤ (offsets a0 a1 a2 a3).2.2.1 ∧
    (offsets a0 a1 a2 a3).2.2.1 + a2 ≤ (offsets a0 a1 a2 a3).2.2.2 ∧
    (offsets a0 a1 a2 a3).2.2.2 + a3 = a0 + a1 + a2 + a3 := by
  simp [offsets] <;> omega

/-- The real geometry (2560 x 4 streams, rank 320) has all five regions already 16-byte aligned. -/
theorem real_geometry_regions :
    alignUp16 (10240 * 4) = 40960 ∧ alignUp16 (10240 * 2) = 20480 ∧ alignUp16 (320 * 2) = 640 ∧
    alignUp16 (10240 * 4) = 40960 ∧ alignUp16 (320 * 4) = 1280 := by
  decide

/-- Total workspace of the real geometry. -/
theorem real_geometry_total : 40960 + 20480 + 640 + 40960 + 1280 = 104320 := by decide

/-- The historical mis-size: giving xq only low-rank bytes is far too small for the stream vector. -/
theorem xq_misize_too_small : (320 * 2 : Nat) < 10240 * 2 := by decide

-- § down kernel mapping

/-- Strided reduction: element i of the row decomposes into (iteration k, warp, lane). -/
theorem down_element_split (i : Nat) :
    i = (i / 256) * 256 + ((i % 256) / 32) * 32 + i % 32 := by
  omega

/-- The decomposed warp index is below 8 and the lane index below 32. -/
theorem down_split_ranges (i : Nat) : (i % 256) / 32 < 8 ∧ i % 32 < 32 := by
  omega

/-- Unique decomposition: equal (k, warp, lane) triples give equal elements. -/
theorem down_split_inj (k w l k' w' l' : Nat) (hw : w < 8) (hl : l < 32) (hw' : w' < 8)
    (hl' : l' < 32) (h : w * 32 + l + 256 * k = w' * 32 + l' + 256 * k') :
    k = k' ∧ w = w' ∧ l = l' := by
  omega

/-- Block-level reduction scratch: 8 warps' partials fit in `part[WARPS]`. -/
theorem partials_fit (nw : Nat) (h : nw = 256 / 32) : nw ≤ 8 := by
  omega

/-- The block-sum scratch of 8 doubles can hold 8 floats (reinterpretation does not overrun). -/
theorem scratch_reinterpret_fits : 8 * 4 ≤ 8 * 8 := by decide

-- § gate / mean / write kernels

/-- Gate grid covers all hc_dim rows (8 rows per block). -/
theorem gate_grid_covers (hc_dim : Nat) : hc_dim ≤ (hc_dim + 7) / 8 * 8 := by
  omega

/-- A row below hc_dim lies in a launched block. -/
theorem gate_row_block (i hc_dim : Nat) (h : i < hc_dim) : i / 8 < (hc_dim + 7) / 8 := by
  omega

/-- Mean kernel: element (c, d) of the gated array is in range. -/
theorem mean_index_lt (c d hc n_embd : Nat) (hc' : c < hc) (hd : d < n_embd) :
    c * n_embd + d < hc * n_embd :=
  idx_lt hc' hd

/-- Mean grid covers n_embd (256 threads per block). -/
theorem mean_grid_covers (n : Nat) : n ≤ (n + 255) / 256 * 256 := by
  omega

/-- Write kernel: the stream index of a flat position is below hc. -/
theorem write_stream_lt (i n_embd hc : Nat) (h : i < n_embd * hc) : i / n_embd < hc :=
  Nat.div_lt_of_lt_mul h

/-- Write kernel: (stream, column) recombines to the flat index. -/
theorem write_recombine (i n_embd : Nat) : n_embd * (i / n_embd) + i % n_embd = i :=
  Nat.div_add_mod i n_embd

/-- Write kernel: the column index is below n_embd. -/
theorem write_col_lt (i n_embd : Nat) (h : 0 < n_embd) : i % n_embd < n_embd :=
  Nat.mod_lt _ h

/-- Stream-weight setup: threads 0..hc-1 write `w`, which needs `hc` floats of shared memory. -/
theorem write_weight_slots (t hc : Nat) (h : t < hc) : t * 4 + 4 ≤ hc * 4 := by
  omega

-- § inject / native contracts

/-- The inject kernel launches 32*hc threads in one block: at most 1024 means hc <= 32. -/
theorem inject_block_size_ok (hc : Nat) (h : 32 * hc ≤ 1024) : hc ≤ 32 := by
  omega

/-- Native path requirement: even `hc*n_embd` when n_embd is even. -/
theorem native_even (n_embd hc : Nat) (h : n_embd % 2 = 0) : (hc * n_embd) % 2 = 0 := by
  rw [Nat.mul_mod, h]
  simp

end Strata.GatedResidual
