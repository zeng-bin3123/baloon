/-!
# MODULE: gated_residual
SOURCES: src/kernels/cuda/gr.cu, src/kernels/cuda/fused_gr.cu
Gated residual / hyper-connection read and write (hc = 4 streams of n_embd = 2560, low-rank hc_lr = 320):
per-stream RMSNorm, down projection (10240 -> 320) + silu, up projection (320 -> 10240) + sigmoid gate, mean over
streams, injection dot products, the residual write, the workspace layout, and the fused / multi-token variants.

## Source semantics
* hc_dim = hc * n_embd = 10240; down has 320 rows, one 8-warp block per row, the 8 warps splitting the reduction.
* Gate: one warp per output row, 1280 blocks of 8 warps cover 10240 rows; each lane does hc_lr/32 = 10 terms.
* Write: out[i] = R[i] + block_out[i % n_embd] * w[i / n_embd] (every stream adds the same block output).
* Workspace: five regions each rounded up to 16 bytes, laid out consecutively without overlap.
* Fused up: block covers 32 columns x 4 streams = 128 rows; a (block, row) pair addresses exactly one of 10240 entries.

## Form template identifiers
N -> 2560; HC -> 4; D -> 10240; LR -> 320; WARPS -> 8.
-/

namespace GatedResidual

-- § helpers

/-- Row-major index bound. -/
theorem idx2_lt (a b A B : Nat) (ha : a < A) (hb : b < B) : a*B+b < A*B := by
  have h1 : (a+1)*B ≤ A*B := Nat.mul_le_mul_right B (by omega)
  rw [Nat.add_mul, Nat.one_mul] at h1
  omega

-- § geometry constants

/-- `hc * n_embd = 10240`. -/
theorem hc_dim_val : 2560 * 4 = 10240 := by decide

/-- Down projection: 40 blocks of 8 warps are exactly the 320 rows. -/
theorem down_blocks_exact : 320 / 8 = 40 ∧ 40 * 8 = 320 := by decide

/-- The up kernel's 80 blocks of 32 columns are exactly n_embd; the multi variant's 160 x 16 as well. -/
theorem up_blocks_exact : 80 * 32 = 2560 ∧ 160 * 16 = 2560 := by decide

/-- 320 hidden values are 40 chunks of 8: 32 lanes take one, 8 lanes take a second. -/
theorem lr_chunks : 320 / 8 = 40 ∧ 40 - 32 = 8 := by decide

/-- One warp covers a 10240-row reduction as 1280 chunks, 40 per lane. -/
theorem down_chunks_per_lane : 10240 / 8 = 1280 ∧ 1280 / 32 = 40 := by decide

/-- Fused multi-token tiling: 4 tiles of 2560, each 320 chunks = 10 per lane. -/
theorem tiling : 10240 / 2560 = 4 ∧ 2560 / 8 / 32 = 10 := by decide

-- § work distribution

/-- Down rows: every row `k < 320` is (block, warp) = (k/8, k%8). -/
theorem down_row_cover (k : Nat) (h : k < 320) : k / 8 < 40 ∧ k = (k / 8) * 8 + k % 8 ∧ k % 8 < 8 := by omega

/-- Block-wide split of the 10240-element reduction among 8 warps x 32 lanes: each element has one owner. -/
theorem reduction_owner (i : Nat) (h : i < 10240) :
    i = (i / 256) * 256 + ((i % 256) / 32) * 32 + i % 32 ∧ (i % 256) / 32 < 8 ∧ i % 32 < 32 := by omega

/-- Gate rows: 1280 blocks of 8 warps cover 10240 rows. -/
theorem gate_row_cover (i : Nat) (h : i < 10240) : i / 8 < (10240 + 7) / 8 ∧ i = (i / 8) * 8 + i % 8 := by omega

/-- Hidden-vector lane coverage inside a 320-wide reduction. -/
theorem lane_cover_320 (j : Nat) (h : j < 320) : j = (j / 32) * 32 + j % 32 ∧ j / 32 < 10 := by omega

/-- Tile chunk coverage: each of the 320 chunks per tile is owned by one (lane, q). -/
theorem tile_chunk_cover (j : Nat) (h : j < 320) : j = j % 32 + 32 * (j / 32) ∧ j % 32 < 32 ∧ j / 32 < 10 := by
  omega

-- § up kernel addressing

/-- Fused up kernel: (block, row r in 0..127) addresses stream `r/32`, column `blk*32 + r%32`; index in range. -/
theorem up_index_lt (blk r : Nat) (hb : blk < 80) (hr : r < 128) :
    (r / 32) * 2560 + blk * 32 + r % 32 < 10240 := by omega

/-- Every entry of the 10240-wide residual is produced by exactly one (block, row) pair of the up kernel. -/
theorem up_index_surjective (i : Nat) (h : i < 10240) :
    ∃ blk r, blk < 80 ∧ r < 128 ∧ i = (r / 32) * 2560 + blk * 32 + r % 32 := by
  refine ⟨(i % 2560) / 32, (i / 2560) * 32 + (i % 2560) % 32, by omega, by omega, by omega⟩

/-- ...and injective. -/
theorem up_index_injective (b1 b2 r1 r2 : Nat) (hb1 : b1 < 80) (hb2 : b2 < 80) (hr1 : r1 < 128) (hr2 : r2 < 128)
    (h : (r1 / 32) * 2560 + b1 * 32 + r1 % 32 = (r2 / 32) * 2560 + b2 * 32 + r2 % 32) :
    b1 = b2 ∧ r1 = r2 := by omega

-- § residual write

/-- `out[i]` uses block output element `i % n` and stream weight `i / n`; both in range. -/
theorem write_index_decode (i n hc : Nat) (hn : 0 < n) (h : i < hc * n) :
    i / n < hc ∧ i % n < n ∧ (i / n) * n + i % n = i := by
  refine ⟨(Nat.div_lt_iff_lt_mul hn).2 h, Nat.mod_lt _ hn, ?_⟩
  have := Nat.div_add_mod i n
  rw [Nat.mul_comm n (i / n)] at this
  exact this

/-- Mean over streams reads `gated[c*n + d]` with c < hc, d < n: in range. -/
theorem mean_index_lt (c d hc n : Nat) (hc' : c < hc) (hd : d < n) : c * n + d < hc * n :=
  idx2_lt c d hc n hc' hd

-- § workspace layout

/-- Round a byte size up to 16. -/
def align16 (s : Nat) : Nat := (s + 15) / 16 * 16

/-- Alignment never shrinks, adds less than 16, and yields a multiple of 16. -/
theorem align16_props (s : Nat) : s ≤ align16 s ∧ align16 s < s + 16 ∧ align16 s % 16 = 0 := by
  unfold align16; omega

/-- Consecutive regions do not overlap: next offset >= previous offset + previous size. -/
theorem region_disjoint (off sz : Nat) : off + sz ≤ off + align16 sz := by
  have := (align16_props sz).1
  omega

/-- Real geometry: xn + xq + lq + gated + lo = 104320 bytes. -/
theorem workspace_total :
    align16 (10240 * 4) + align16 (10240 * 2) + align16 (320 * 2) + align16 (10240 * 4) + align16 (320 * 4)
      = 104320 := by decide

/-- The original bug: giving `xq` 640 bytes where 20480 are needed is strictly too small. -/
theorem xq_size_needed : 640 < 10240 * 2 := by decide

/-- Block-sum scratch: 8 floats, and a double scratch of 8 is at least as large. -/
theorem scratch_cast_safe : 8 * 4 ≤ 8 * 8 := by decide

end GatedResidual
