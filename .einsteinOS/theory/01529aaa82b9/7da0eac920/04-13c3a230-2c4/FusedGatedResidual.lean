/-!
# MODULE: fused_gated_residual
Fused hyper-connection / gated-residual read for the fixed geometry n_embd = 2560, hc = 4 streams,
hc_lr = 320: a per-stream RMS-norm, a down-projection (D = hc*n_embd -> 320) with SiLU, an up-projection with
a sigmoid gate, and the mean over streams. A multi-token variant reads each weight matrix once for T tokens.

## Source semantics
* D = N * HC = 10240; the down kernel uses 40 blocks of 8 warps (one row per warp = 320 rows) plus one inject block.
* The up kernel assigns 32 columns x 4 streams = 128 rows per block over 80 blocks (single token) or 16 columns
  over 160 blocks (multi token).
* float4 accesses never straddle two streams because N is a multiple of 4.
* Lanes stride the reduction axis in uint4 (8 x bf16) chunks: 1280 chunks over 32 lanes = 40 per lane.
* The multi-token kernel stages the activations in tiles of 2560 floats; lane k writes token k, needing T ≤ 32.

## Form template identifiers
N → n_embd ; HC → stream_count ; LR → low_rank ; `gr_down_kernel` → down_project_silu ;
`gr_up_kernel` → up_project_gate ; `gr_*_multi_kernel` → multi_token_variants ; kFusedGrMaxT → max_tokens.
-/

namespace Strata.FusedGr

-- § geometry constants

/-- n_embd. -/
def N : Nat := 2560
/-- Residual streams. -/
def HC : Nat := 4
/-- Down-projected width. -/
def LR : Nat := 320
/-- Flattened stream dimension. -/
def D : Nat := N * HC

/-- D = 10240. -/
theorem D_value : D = 10240 := rfl

/-- 8 warps per 256-thread block. -/
theorem warps_per_block : 256 / 32 = 8 := rfl

/-- 40 down blocks of 8 rows cover the 320 low-rank rows. -/
theorem down_blocks_cover : 40 * 8 = LR := rfl

/-- 80 up blocks of 32 columns cover n_embd. -/
theorem up_blocks_cover : 80 * 32 = N := rfl

/-- 160 multi-token up blocks of 16 columns cover n_embd. -/
theorem upm_blocks_cover : 160 * 16 = N := rfl

/-- 32 columns times 4 streams give 128 rows per up block. -/
theorem up_rows_per_block : 32 * 4 = 128 := rfl

/-- 16 rows per warp in the single-token up kernel (8 warps). -/
theorem up_rows_per_warp : 128 / 8 = 16 := rfl

/-- 8 rows per warp in the multi-token up kernel. -/
theorem upm_rows_per_warp : (4 * 16) / 8 = 8 := rfl

-- § chunking

/-- D / 8 = 1280 chunks. -/
theorem down_chunks : D / 8 = 1280 := by decide

/-- 1280 chunks over 32 lanes = 40 each. -/
theorem down_chunks_per_lane : 1280 / 32 = 40 := rfl

/-- LR / 8 = 40 chunks; lanes 0..31 take one each and lanes 0..7 take a second (32 + lane). -/
theorem up_chunks : LR / 8 = 40 ∧ 40 - 32 = 8 := by decide

/-- The second chunk index 32 + lane stays below 40 exactly for lane < 8. -/
theorem up_second_chunk (lane : Nat) (h : lane < 32) : 32 + lane < 40 ↔ lane < 8 := by omega

/-- Tile of 2560 floats: TQ = 10 chunks per lane. -/
theorem tile_chunks : 2560 / 8 / 32 = 10 := rfl

/-- 4 tiles of 2560 cover D. -/
theorem tiles_cover : 4 * 2560 = D := rfl

/-- Per-lane tile chunk indexing `lane + 32 q` tiles 0..319 exactly. -/
theorem tile_chunk_cover (lane q : Nat) (hl : lane < 32) (hq : q < 10) : lane + 32 * q < 320 := by omega

/-- Tile chunk index uniquely determines (q, lane). -/
theorem tile_chunk_unique (l q l' q' : Nat) (h1 : l < 32) (h2 : l' < 32)
    (h : l + 32 * q = l' + 32 * q') : l = l' ∧ q = q' := by omega

-- § stream indexing

/-- Float4 loads at multiples of 4 never straddle a stream boundary at N = 2560. -/
theorem float4_same_stream (i : Nat) (h : i % 4 = 0) : i / 2560 = (i + 3) / 2560 := by omega

/-- Flat index splits into stream c and offset d. -/
theorem stream_split (i : Nat) : i = (i / 2560) * 2560 + i % 2560 := by omega

/-- Stream index is below HC for i < D. -/
theorem stream_lt (i : Nat) (h : i < 10240) : i / 2560 < 4 := by omega

/-- Row index `c*N + d0 + dd` of the up kernel lies below D. -/
theorem up_row_lt (c d0 dd : Nat) (hc : c < 4) (hd : d0 + dd < 2560) : c * 2560 + d0 + dd < 10240 := by omega

/-- The up kernel's block `bx` and in-block column `dd` determine d = bx*32+dd < N. -/
theorem up_col_lt (bx dd : Nat) (hb : bx < 80) (hd : dd < 32) : bx * 32 + dd < 2560 := by omega

/-- Multi-token variant: bx*16+dd < N. -/
theorem upm_col_lt (bx dd : Nat) (hb : bx < 160) (hd : dd < 16) : bx * 16 + dd < 2560 := by omega

/-- Each (block, dd) pair is a distinct column. -/
theorem up_col_unique (b d b' d' : Nat) (h1 : d < 32) (h2 : d' < 32) (h : b * 32 + d = b' * 32 + d') :
    b = b' ∧ d = d' := by omega

/-- The down kernel's inject block has 8 warps but only the first 4 (HC) are active. -/
theorem inject_rows_bound (warp : Nat) (h : warp < 4) : warp < 4 := h

-- § multi-token

/-- Dynamic shared memory for T tokens is T*TILE*4 bytes and never exceeds the attribute-maximum for T ≤ Tmax. -/
theorem shared_mem_le (T Tmax : Nat) (h : T ≤ Tmax) : T * 2560 * 4 ≤ Tmax * 2560 * 4 := by
  have := Nat.mul_le_mul_right 10240 h
  omega

/-- Lane k writes token k, so every token has a writer when Tmax ≤ 32. -/
theorem lane_writes_token (k T : Nat) (hk : k < T) (hT : T ≤ 32) : k < 32 := by omega

/-- Staging loop trip count: T*(TILE/4) float4 loads divided among 256 threads cover all. -/
theorem stage_cover (i T : Nat) (h : i < T * 640) : i / 640 < T := by omega

/-- Staged element (k, off) reads source `(k*D + base)/4 + off` which is below T*D/4 for base+4*off < D. -/
theorem stage_src_lt (k T base off : Nat) (hk : k < T) (hb : base + 2560 ≤ 10240) (ho : off < 640) :
    (k * 10240 + base) / 4 + off < T * 2560 := by
  have h1 : (k + 1) * 10240 ≤ T * 10240 := Nat.mul_le_mul_right 10240 (Nat.succ_le_of_lt hk)
  have h2 : (k + 1) * 10240 = k * 10240 + 10240 := Nat.succ_mul k 10240
  omega

/-- Supported geometry predicate. -/
def supported (n hc lr : Nat) : Bool := n == 2560 && hc == 4 && lr == 320

/-- The kernels accept exactly the one compiled geometry. -/
theorem supported_iff (n hc lr : Nat) : supported n hc lr = true ↔ n = 2560 ∧ hc = 4 ∧ lr = 320 := by
  unfold supported; simp [and_assoc]

end Strata.FusedGr
