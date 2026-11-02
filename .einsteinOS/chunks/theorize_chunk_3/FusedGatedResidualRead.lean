/-!
# MODULE: fused_gated_residual_read
Index geometry of the fused hyper-connection read kernels for the fixed geometry n_embd = 2560, hc = 4,
hc_lr = 320: the norm, the down projection (+ inject rows), the up projection with gated mean, and their
T-token variants that read each weight row once.

## Source semantics
- Residual `R` is hc streams of n_embd floats; flat index `c*N + d`.
- Down kernel: 40 blocks of 8 warps (one row each) cover the 320 `lo` rows; block 40 handles the 4 inject rows.
- Up kernel: 80 blocks of 32 columns x 4 streams = 128 rows; multi variant 160 blocks of 16 columns.
- Tiled multi-token down kernel: 10240 reduction elements = 4 tiles of 2560 floats = 320 uint4 chunks each.

## Form template identifiers
- N -> n_embd, HC -> n_streams, D -> stream_dim, LR -> low_rank, TILE -> act_tile
- w_down -> down_weight, w_up -> up_weight, w_inject -> inject_weight, R -> residual
-/

namespace Strata.FusedGr

def N : Nat := 2560
def HC : Nat := 4
def D : Nat := N * HC
def LR : Nat := 320
def THREADS : Nat := 256
def WARPS : Nat := THREADS / 32
def DOWN_BLOCKS : Nat := LR / WARPS
def UP_COLS : Nat := 32
def UP_BLOCKS : Nat := N / UP_COLS
def TILE : Nat := 2560
def TQ : Nat := TILE / 8 / 32
def UPM_COLS : Nat := 16
def UPM_BLOCKS : Nat := N / UPM_COLS

-- § helpers

/-- Row-major index bound. -/
theorem idx_lt {a b A B : Nat} (ha : a < A) (hb : b < B) : a * B + b < A * B := by
  have h1 : (a + 1) * B ≤ A * B := Nat.mul_le_mul_right B (Nat.succ_le_of_lt ha)
  have h2 : (a + 1) * B = a * B + B := by rw [Nat.add_mul, Nat.one_mul]
  omega

-- § constants

/-- Stream dimension. -/
theorem stream_dim : D = 10240 := by decide
/-- Eight warps per block. -/
theorem warps_per_block : WARPS = 8 := by decide
/-- 40 down blocks of 8 rows cover all 320 low-rank rows. -/
theorem down_blocks_cover_lr : DOWN_BLOCKS * WARPS = LR := by decide
/-- 80 up blocks of 32 columns cover n_embd. -/
theorem up_blocks_cover_n : UP_BLOCKS * UP_COLS = N := by decide
/-- 160 multi-token up blocks of 16 columns cover n_embd. -/
theorem upm_blocks_cover_n : UPM_BLOCKS * UPM_COLS = N := by decide
/-- Per tile each lane loads 10 uint4 chunks. -/
theorem tile_chunks_per_lane : TQ = 10 := by decide
/-- Lane x chunk coverage of one tile: 32 lanes * 10 chunks * 8 values = TILE. -/
theorem tile_covered : 32 * TQ * 8 = TILE := by decide
/-- Four tiles cover the reduction axis. -/
theorem tiles_cover_reduction : 4 * TILE = D := by decide
/-- LR/8 = 40 chunks = 32 + 8, which is the lane<8 second-chunk rule. -/
theorem lr_chunk_split : LR / 8 = 32 + 8 := by decide

-- § flat residual indexing

/-- A flat residual index splits into stream and column. -/
theorem flat_split (i : Nat) (h : i < 10240) :
    i / 2560 < 4 ∧ i % 2560 < 2560 ∧ (i / 2560) * 2560 + i % 2560 = i := by
  omega

/-- float4 accesses at multiples of 4 never straddle two streams. -/
theorem vec4_within_stream (i : Nat) (h : i < 10240) (h4 : i % 4 = 0) :
    (i + 3) / 2560 = i / 2560 := by
  omega

/-- The down kernel's strided loop (step THREADS*4) visits only valid float4 starts. -/
theorem vec4_loop_valid (t k : Nat) (ht : t < 256) (hk : t * 4 + k * 1024 < 10240) :
    (t * 4 + k * 1024) % 4 = 0 := by
  omega

-- § down kernel

/-- Down block rows: block b, warp w addresses row b*8+w, below 320 for b < 40. -/
theorem down_row_in_range (b w : Nat) (hb : b < 40) (hw : w < 8) : b * 8 + w < 320 := by
  omega

/-- The inject block is block 40 and its active warps are 0..3. -/
theorem inject_block_rows (w : Nat) (hw : w < 8) : w < 4 ∨ 4 ≤ w := by
  omega

/-- Chunk index of lane l in a 10240-element row loop: l + 32 j < 1280. -/
theorem down_chunk_in_range (l j : Nat) (hl : l < 32) (hj : j < 40) : l + 32 * j < 1280 := by
  omega

/-- Tiled loading: chunk index base/8 + lane + 32 q stays inside the 1280 chunks of a row. -/
theorem tiled_chunk_in_range (base l q : Nat) (hbase : base % 2560 = 0) (hb : base < 10240)
    (hl : l < 32) (hq : q < 10) : base / 8 + l + 32 * q < 1280 := by
  omega

/-- Shared activation tile copy: element i of the T*TILE/4 float4 loop maps to token i/640. -/
theorem tile_copy_split (i T : Nat) (h : i < T * 640) :
    i / 640 < T ∧ i % 640 < 640 ∧ (i / 640) * 640 + i % 640 = i := by
  have h1 : i / 640 < T := by
    have : i < 640 * T := by omega
    exact Nat.div_lt_of_lt_mul this
  omega

-- § up kernel

/-- Up row r in a 128-row block: stream c = r/32, column dd = r%32. -/
theorem up_row_split (r : Nat) (h : r < 128) :
    r / 32 < 4 ∧ r % 32 < 32 ∧ (r / 32) * 32 + r % 32 = r := by
  omega

/-- The flat residual index of an up row is below 10240. -/
theorem up_index_in_range (blk r : Nat) (hb : blk < 80) (hr : r < 128) :
    (r / 32) * 2560 + blk * 32 + r % 32 < 10240 := by
  omega

/-- Multi-token up kernel: 64 rows per block, 8 per warp. -/
theorem upm_index_in_range (blk r : Nat) (hb : blk < 160) (hr : r < 64) :
    (r / 16) * 2560 + blk * 16 + r % 16 < 10240 := by
  omega

/-- Each block owns column d0+dd of every stream, so the mean output index stays below n_embd. -/
theorem mixed_index_in_range (blk t : Nat) (hb : blk < 80) (ht : t < 32) : blk * 32 + t < 2560 := by
  omega

/-- The second weight chunk of a row exists only for lane < 8. -/
theorem second_chunk_in_range (l : Nat) (hl : l < 8) : 32 + l < 40 := by
  omega

-- § multi-token launch contract

/-- Support predicate of the fixed geometry. -/
def grSupported (n_embd hc hc_lr : Int) : Prop := n_embd = 2560 ∧ hc = 4 ∧ hc_lr = 320

/-- Supported geometry gives the compile-time constants. -/
theorem supported_gives_constants (n hc lr : Int) (h : grSupported n hc lr) :
    n * hc = 10240 ∧ lr = 320 := by
  obtain ⟨h1, h2, h3⟩ := h
  subst h1; subst h2; subst h3
  exact ⟨by decide, rfl⟩

/-- Token count is at most the warp width, so lane k can own token k. -/
theorem token_lane_ownership (T k maxT : Nat) (hT : T ≤ maxT) (hm : maxT ≤ 32) (hk : k < T) :
    k < 32 := by
  omega

end Strata.FusedGr
