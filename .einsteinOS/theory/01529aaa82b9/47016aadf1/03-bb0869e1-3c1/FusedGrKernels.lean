/-!
# MODULE: fused_gr_kernels
Launch geometry and index decomposition of the fused gated-residual (hyper-connection) read kernels,
single-token and T-token variants, at the fixed geometry n_embd=2560, hc=4, hc_lr=320.

## Source semantics
Streams are stored stream-major: index i = c*N + d. `down` uses one warp per output row (8 rows/block,
40 blocks + 1 inject block); `up` handles 32 columns x 4 streams per block (80 blocks); the multi-token
variants use 16 columns per block (160 blocks) and 10 weight chunks per lane per tile.

## Form template identifiers
N -> n_embd 2560; HC -> streams 4; D -> N*HC; LR -> hc_lr 320; THREADS -> 256; WARPS -> 8; T -> tokens.
-/
namespace Strata.FusedGr

/-- D = N * HC. -/
theorem dim_eq : 2560 * 4 = 10240 := by decide

/-- Stream/column decomposition: i = c*2560 + d recovers (c, d). -/
theorem stream_decomp (c d : Nat) (hd : d < 2560) :
    (c * 2560 + d) / 2560 = c ∧ (c * 2560 + d) % 2560 = d := by omega

/-- Every flat index below D has a stream and column. -/
theorem stream_cover (i : Nat) (h : i < 10240) : ∃ c d, c < 4 ∧ d < 2560 ∧ i = c * 2560 + d :=
  ⟨i / 2560, i % 2560, by omega, by omega, by omega⟩

/-- float4 loop: indices i = 4t stay within one stream (4 divides N), so c is uniform over the vector. -/
theorem vec4_same_stream (t : Nat) (h : 4 * t + 3 < 10240) : (4 * t) / 2560 = (4 * t + 3) / 2560 := by
  omega

/-- 40 blocks of 8 warps give exactly the 320 down rows. -/
theorem down_rows : 40 * 8 = 320 := by decide

/-- A down row (block, warp) is below LR. -/
theorem down_row_lt (b w : Nat) (hb : b < 40) (hw : w < 8) : b * 8 + w < 320 := by omega

/-- The inject block uses warps 0..3 for the 4 streams. -/
theorem inject_rows (w : Nat) (h : w < 4) : w < 4 := h

/-- Down rows are distinct per (block, warp). -/
theorem down_row_inj (b w b' w' : Nat) (hw : w < 8) (hw' : w' < 8) (e : b * 8 + w = b' * 8 + w') :
    b = b' ∧ w = w' := by omega

/-- Row chunks: 10240 bf16 = 1280 chunks of 8, 40 per lane. -/
theorem down_chunks : 10240 / 8 = 1280 ∧ 1280 / 32 = 40 := by decide

/-- Chunk indices j = lane + 32 q (q<10) cover a 320-chunk tile uniquely. -/
theorem tile_chunk_unique (l q l' q' : Nat) (hl : l < 32) (hl' : l' < 32)
    (e : l + 32 * q = l' + 32 * q') : l = l' ∧ q = q' := by omega

/-- Every chunk of a tile below 320 is held by some (lane, q) with q < 10. -/
theorem tile_chunk_cover (j : Nat) (h : j < 320) : ∃ l q, l < 32 ∧ q < 10 ∧ j = l + 32 * q :=
  ⟨j % 32, j / 32, by omega, by omega, by omega⟩

/-- TILE = 2560 floats = 320 chunks of 8 = 10 per lane; 4 tiles make D. -/
theorem tile_sizes : 2560 / 8 / 32 = 10 ∧ 10240 / 2560 = 4 := by decide

/-- up: 80 blocks of 32 columns cover N = 2560 columns. -/
theorem up_cover : 80 * 32 = 2560 := by decide

/-- up (multi-token): 160 blocks of 16 columns cover N. -/
theorem upm_cover : 160 * 16 = 2560 := by decide

/-- up row r < 128 decomposes into stream r/32 and column offset r%32. -/
theorem up_row_decomp (r : Nat) (h : r < 128) : r / 32 < 4 ∧ r % 32 < 32 ∧ r = (r / 32) * 32 + r % 32 := by
  omega

/-- Global weight row i = c*2560 + d0 + dd stays below D. -/
theorem up_index_lt (c blk dd : Nat) (hc : c < 4) (hb : blk < 80) (hdd : dd < 32) :
    c * 2560 + blk * 32 + dd < 10240 := by omega

/-- Warp r-loop: 8 warps with stride 8 visit rows w, w+8, ..., i.e. 16 rows per warp out of 128. -/
theorem warp_rows (r : Nat) (h : r < 128) : ∃ w k, w < 8 ∧ k < 16 ∧ r = w + 8 * k :=
  ⟨r % 8, r / 8, by omega, by omega, by omega⟩

/-- LR = 40 chunks: 32 in the first pass and 8 in the second (lanes below 8). -/
theorem lr_chunks : 320 / 8 = 40 ∧ 40 - 32 = 8 := by decide

/-- Second-pass chunk 32+lane exists only for lane < 8. -/
theorem second_pass_bound (lane : Nat) (h : lane < 40 - 32) : 32 + lane < 40 := by omega

/-- Multi-token epilogue: lane k < T <= 32 is a valid lane. -/
theorem token_lane (k T : Nat) (hk : k < T) (hT : T ≤ 32) : k < 32 := by omega

/-- Shared tile size T*TILE*4 bytes grows linearly in T. -/
theorem tile_bytes (T : Nat) : T * 2560 * 4 = T * 10240 := by omega

/-- Mean over streams divides the stream sum by HC = 4: sum of four equal parts. -/
theorem mean_of_equal (x : Nat) : (x + x + x + x) / 4 = x := by omega

end Strata.FusedGr
