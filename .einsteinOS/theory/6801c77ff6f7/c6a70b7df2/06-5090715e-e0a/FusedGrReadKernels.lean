/-!
# MODULE: fused_gr_read_kernels
SOURCES: src/kernels/cuda/fused_gr.cu
Index-geometry theory of the fused gated-residual (hyper-connection) read: the `down` and `up` kernels for one
token and the T-token weight-sharing variants.

## Source semantics
* Geometry is fixed: n_embd 2560, 4 streams, D = 10240, low-rank 320, 256 threads / 8 warps per block.
* `down`: 40 blocks x 8 warps = 320 output rows, plus one extra block for the 4 inject rows.
* `up`: 80 blocks x 32 columns (x 4 streams = 128 rows, 16 per warp).
* multi-token: tiles of 2560 floats per token; `UPM` blocks of 16 columns.

## Form template identifiers
gr_down_kernel -> gated_residual_down ; gr_up_kernel -> gated_residual_up ; GrMulti -> multi_token_batch ;
N -> embed_dim ; HC -> stream_count ; LR -> low_rank
-/

namespace Strata.FusedGr

-- § helpers

/-- Row-major bound `t*h + r < T*h`. -/
theorem rowmajor_lt {t T h r : Nat} (ht : t < T) (hr : r < h) : t * h + r < T * h := by
  have h1 : (t + 1) * h ≤ T * h := Nat.mul_le_mul_right h (Nat.succ_le_of_lt ht)
  rw [Nat.add_mul, Nat.one_mul] at h1
  omega

-- § constants

/-- Embedding width. -/
def N : Nat := 2560
/-- Streams. -/
def HC : Nat := 4
/-- Flattened width. -/
def D : Nat := N * HC
/-- Low rank. -/
def LR : Nat := 320
/-- Threads per block. -/
def THREADS : Nat := 256
/-- Warps per block. -/
def WARPS : Nat := THREADS / 32
/-- Down blocks (one more handles inject). -/
def DOWN_BLOCKS : Nat := LR / WARPS
/-- Columns per `up` block. -/
def UP_COLS : Nat := 32
/-- `up` blocks. -/
def UP_BLOCKS : Nat := N / UP_COLS

/-- The constants have the values stated in the source comments. -/
theorem constants_values :
    D = 10240 ∧ WARPS = 8 ∧ DOWN_BLOCKS = 40 ∧ UP_BLOCKS = 80 := by decide

/-- 40 blocks of 8 warps give exactly one warp per low-rank row. -/
theorem down_rows_exact : DOWN_BLOCKS * WARPS = LR := by decide

/-- 80 blocks of 32 columns cover every embedding column exactly once. -/
theorem up_cols_exact : UP_BLOCKS * UP_COLS = N := by decide

/-- Each `up` block handles 4 x 32 = 128 rows, 16 per warp. -/
theorem up_rows_per_warp : HC * UP_COLS = 128 ∧ 128 / WARPS = 16 := by decide

-- § stream decomposition of the flattened index

/-- A flat index splits into stream `c < 4` and column `d < N`. -/
theorem stream_split (i : Nat) (h : i < 10240) :
    i / 2560 < 4 ∧ i % 2560 < 2560 ∧ i = (i / 2560) * 2560 + i % 2560 := by omega

/-- A float4 chunk starting at a multiple of 4 never straddles two streams. -/
theorem float4_no_straddle (i : Nat) (h4 : i % 4 = 0) (h : i < 10240) :
    i % 2560 + 3 < 2560 ∧ i / 2560 = (i + 3) / 2560 := by omega

/-- The `t*4` stride loop covers D in exactly 10 iterations of 1024 floats. -/
theorem norm_loop_trip_count : 10240 / (256 * 4) = 10 ∧ 10240 % (256 * 4) = 0 := by decide

/-- The element-wise tail loop visits each index once: `i` is reached by thread `i % 256`. -/
theorem tail_loop_owner (i : Nat) (h : i < 10240) :
    i % 256 < 256 ∧ i = i % 256 + 256 * (i / 256) := by omega

-- § down kernel rows

/-- Regular rows `blk*8 + warp` are in range for the 40 regular blocks. -/
theorem down_row_in_range {blk warp : Nat} (hb : blk < 40) (hw : warp < 8) : blk * 8 + warp < 320 := by
  omega

/-- Row assignment is injective across (block, warp). -/
theorem down_row_injective {b w b' w' : Nat} (hw : w < 8) (hw' : w' < 8)
    (h : b * 8 + w = b' * 8 + w') : b = b' ∧ w = w' := by omega

/-- Dot product chunks: 1280 chunks of 8, 40 per lane. -/
theorem down_chunks_per_lane : 10240 / 8 = 1280 ∧ 1280 / 32 = 40 ∧ 1280 % 32 = 0 := by decide

/-- The inject block uses only warps below 4, each a distinct row index. -/
theorem inject_rows_distinct {w w' : Nat} (h : w < 4) (h' : w' < 4) (e : w = w') : w = w' := e

-- § up kernel rows

/-- Output row `c*N + d0 + dd` of the `up` kernel stays inside D. -/
theorem up_row_lt {c blk dd : Nat} (hc : c < 4) (hb : blk < 80) (hd : dd < 32) :
    c * 2560 + blk * 32 + dd < 10240 := by omega

/-- The (stream, block, column) → row map is injective. -/
theorem up_row_injective {c b d c' b' d' : Nat} (hb : b < 80) (hb' : b' < 80) (hd : d < 32) (hd' : d' < 32)
    (h : c * 2560 + b * 32 + d = c' * 2560 + b' * 32 + d') : c = c' ∧ b = b' ∧ d = d' := by omega

/-- Low-rank reduction chunks: 40 = 32 (first load) + 8 (second load by lanes < 8). -/
theorem up_chunk_split : 320 / 8 = 40 ∧ 320 / 8 - 32 = 8 := by decide

/-- Lane `l < 32` loads chunk `l`; lane `l < 8` loads chunk `32 + l`: together all 40 chunks, once. -/
theorem up_chunk_cover (j : Nat) (h : j < 40) :
    (j < 32 ∧ j % 32 = j) ∨ (32 ≤ j ∧ j - 32 < 8) := by omega

-- § multi-token variants

/-- The tile covers D in four steps and holds 320 chunks, 10 per lane. -/
theorem tile_geometry : 10240 / 2560 = 4 ∧ 10240 % 2560 = 0 ∧ 2560 / 8 / 32 = 10 := by decide

/-- Multi-token `up` blocks of 16 columns tile N and give 64 rows, 8 per warp. -/
theorem upm_geometry : 2560 / 16 = 160 ∧ 4 * 16 = 64 ∧ 64 / 8 = 8 := by decide

/-- Dynamic shared memory for `T` tokens never exceeds the `kFusedGrMaxT` attribute set once. -/
theorem dyn_smem_le {T M : Nat} (h : T ≤ M) : T * 2560 * 4 ≤ M * 2560 * 4 := by
  exact Nat.mul_le_mul_right 4 (Nat.mul_le_mul_right 2560 h)

/-- Token/offset split used when staging activations is exact. -/
theorem stage_split (i T : Nat) (h : i < T * 640) : i / 640 < T ∧ i = (i / 640) * 640 + i % 640 := by
  have : i / 640 < T := Nat.div_lt_of_lt_mul (by rw [Nat.mul_comm] at h; exact h)
  omega

/-- Source float4 offsets are integral: both `k*D` and the tile base are multiples of 4. -/
theorem stage_src_aligned (k base : Nat) (hb : base % 2560 = 0) : (k * 10240 + base) % 4 = 0 := by omega

/-- Lane `k` writes token `k`: only valid while `T ≤ 32`. -/
theorem lane_writes_token {k T : Nat} (hk : k < T) (hT : T ≤ 32) : k < 32 := by omega

/-- Token-batched lo buffer index `t*LR + k` is in bounds. -/
theorem lo_index_lt {t T k : Nat} (ht : t < T) (hk : k < 320) : t * 320 + k < T * 320 :=
  rowmajor_lt ht hk

/-- Token-batched `xn` scratch index `t*D + i` is in bounds. -/
theorem xn_index_lt {t T i : Nat} (ht : t < T) (hi : i < 10240) : t * 10240 + i < T * 10240 :=
  rowmajor_lt ht hi

end Strata.FusedGr
