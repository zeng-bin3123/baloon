/-!
# MODULE: prefill_kernels
SOURCES: src/prefill/kernels.cu
CUDA kernels of the batched prompt path: hyper-connection norm/mix/write/broadcast, GDN gates /
conv / recurrence, MoE routing / dequant / SwiGLU / gather / combine, and QSA helpers
(rms rows, rope, split-q, gated attention, KV append).  All kernels address flat buffers by
decomposing a linear thread index into `(token, stream, dim)` style coordinates.

## Source semantics
* With `N = 2560`, `HC = 4`, `D = N*HC`, a flat index `i < T*D` decomposes as
  `t = i / D`, `c = (i % D) / N`, `d = i % N` and recomposes exactly.
* Launch grids are ceiling divisions that cover every element.
* The Strata expert blob is gate/up codes, down codes, gate/up scales, down scales
  (1,382,400 bytes) and every dequant read stays inside its section.
* The router addresses 512 experts as 32 lanes x 16 registers.

## Form template identifiers
* `N`, `HC`, `D`, `LR`, `S`, `HK`, `HV`, `C` -> numerals 2560, 4, 10240, 320, 128, 16, 48, 10240
* `blocks_for`                               -> `blocksFor`
* `blob_dequant_kernel`                      -> `guOff`, `downOff`, `guScaleOff`, `downScaleOff`
-/

namespace Strata.PrefillKernels

-- § hyper-connection indexing

/-- Recomposition of a flat `(t, c, d)` index over `D = 10240`, `N = 2560`. -/
theorem hc_index_recompose (i : Nat) :
    i / 10240 * 10240 + (i % 10240) / 2560 * 2560 + i % 2560 = i := by
  omega

/-- The stream coordinate `c` of `gr_write` is a valid stream (`< 4`). -/
theorem hc_stream_lt (i : Nat) : (i % 10240) / 2560 < 4 := by
  omega

/-- The dimension coordinate is a valid embedding index. -/
theorem hc_dim_lt (i : Nat) : i % 2560 < 2560 := by
  omega

/-- The token coordinate is in range when `i < T*D`. -/
theorem hc_tok_lt (T i : Nat) (h : i < T * 10240) : i / 10240 < T := by
  omega

/-- `gr_broadcast` reads `e[t*N + d]`, which is inside the `T*N` embedding block. -/
theorem broadcast_src_lt (T i : Nat) (h : i < T * 10240) : i / 10240 * 2560 + i % 2560 < T * 2560 := by
  omega

/-- `gr_write` reads the residual and the injection: `inj[t*ld + c]` with `c < 4`. -/
theorem write_inj_lt (i ld : Nat) (hld : 4 ≤ ld) : i / 10240 * ld + (i % 10240) / 2560 < (i / 10240 + 1) * ld := by
  have : (i % 10240) / 2560 < 4 := by omega
  have h2 : (i / 10240 + 1) * ld = i / 10240 * ld + ld := by rw [Nat.succ_mul]
  omega

-- § launch grids

/-- `blocks_for(n, t) = (n + t - 1) / t` blocks. -/
def blocksFor (n t : Nat) : Nat := (n + t - 1) / t

/-- A ceiling-divided launch grid covers every element: `n <= blocks * t`. -/
theorem blocksFor_cover (n t : Nat) (ht : 0 < t) : n ≤ blocksFor n t * t := by
  unfold blocksFor
  have h := (Nat.div_lt_iff_lt_mul ht).mp (Nat.lt_succ_self ((n + t - 1) / t))
  rw [Nat.succ_mul] at h
  omega

/-- The GDN conv grid `C/128` blocks of 128 threads covers all `C = 10240` channels exactly. -/
theorem gdn_conv_grid : 10240 / 128 * 128 = 10240 := by decide

/-- The q/k/v channel layout: `2*HK + HV = 80` heads of 128 equal `C = 10240`. -/
theorem gdn_channels : (2 * 16 + 48) * 128 = 10240 := by decide

/-- The recurrence row groups cover all `S = 128` rows: `RG * RPG = S`. -/
theorem gdn_rows_cover : 4 * (128 / 4) = 128 := by decide

/-- A row-group's rows `rg*RPG + r` stay below `S`. -/
theorem gdn_row_lt (rg r : Nat) (hrg : rg < 4) (hr : r < 32) : rg * 32 + r < 128 := by
  omega

-- § expert blob layout (Strata blob)

/-- Gate/up codes: 1280 rows x 640 bytes. -/
def guCodes : Nat := 1280 * 640
/-- Offset of the down codes. -/
def downOff : Nat := guCodes
/-- Offset of the gate/up scales (after 2560 x 160 B of down codes). -/
def guScaleOff : Nat := downOff + 2560 * 160
/-- Offset of the down scales (after 1280 x 40 f16 gate/up scales). -/
def downScaleOff : Nat := guScaleOff + 1280 * 40 * 2
/-- Total blob size: down scales are 2560 x 10 f16. -/
def blobBytes : Nat := downScaleOff + 2560 * 10 * 2

/-- The blob is 1,382,400 bytes (the "1.38 MB" slot read-back). -/
theorem blobBytes_value : blobBytes = 1382400 := by decide

/-- Section offsets are strictly increasing, so the sections are disjoint. -/
theorem blob_sections_ordered : downOff < guScaleOff ∧ guScaleOff < downScaleOff ∧ downScaleOff < blobBytes := by
  decide

/-- A gate/up scale index `row*40 + (byte*4)/64` stays within the 1280x40 table. -/
theorem gu_scale_idx_lt (row byte : Nat) (hr : row < 1280) (hb : byte < 640) :
    row * 40 + (byte * 4) / 64 < 1280 * 40 := by
  omega

/-- A down scale index `row*10 + (byte*4)/64` stays within the 2560x10 table. -/
theorem down_scale_idx_lt (row byte : Nat) (hr : row < 2560) (hb : byte < 160) :
    row * 10 + (byte * 4) / 64 < 2560 * 10 := by
  omega

/-- The four dequantized outputs of one code byte are inside the gate/up matrix row block. -/
theorem gu_out_idx_lt (row byte k : Nat) (hr : row < 1280) (hb : byte < 640) (hk : k < 4) :
    row * 2560 + byte * 4 + k < 1280 * 2560 := by
  omega

/-- The four dequantized outputs of one code byte are inside the down matrix. -/
theorem down_out_idx_lt (row byte k : Nat) (hr : row < 2560) (hb : byte < 160) (hk : k < 4) :
    row * 640 + byte * 4 + k < 2560 * 640 := by
  omega

/-- SwiGLU interleaved: columns `2k`, `2k+1` of a 1280-wide row with `k < 640` are in range. -/
theorem swiglu_il_idx (k : Nat) (hk : k < 640) : 2 * k + 1 < 1280 := by
  omega

/-- `gather_rows16` moves `width/8` uint4 per row; for width 2560 that is 320. -/
theorem gather_per_row : 2560 / 8 = 320 := by decide

-- § router lane layout

/-- Expert `lane + i*32` is recovered from register `i` of `lane`: lane is `e % 32`. -/
theorem route_lane (lane i : Nat) (hl : lane < 32) : (lane + i * 32) % 32 = lane := by
  omega

/-- Register index recovered by `ex / 32`. -/
theorem route_reg (lane i : Nat) (hl : lane < 32) : (lane + i * 32) / 32 = i := by
  omega

/-- 32 lanes x 16 registers cover all 512 experts (index in range). -/
theorem route_cover (lane i : Nat) (hl : lane < 32) (hi : i < 16) : lane + i * 32 < 512 := by
  omega

/-- Top-k uses `k = 10` ranks writing `ids[t*10 + rank]`, inside the `T*10` array. -/
theorem route_out_idx (t T rank : Nat) (ht : t < T) (hr : rank < 10) : t * 10 + rank < T * 10 := by
  have h : (t + 1) * 10 ≤ T * 10 := Nat.mul_le_mul_right 10 ht
  have h2 : (t + 1) * 10 = t * 10 + 10 := by rw [Nat.succ_mul]
  omega

-- § QSA helper indexing

/-- `split_q` recomposition: `i = t*6144 + h*256 + d` with `h = (i/256)%24`, `d = i%256`. -/
theorem split_q_recompose (i : Nat) :
    i / 6144 * 6144 + ((i / 256) % 24) * 256 + i % 256 = i := by
  omega

/-- The gate half of a full-q row sits at offset `h*512 + 256 + d` below `24*512`. -/
theorem qfull_gate_idx_lt (h d : Nat) (hh : h < 24) (hd : d < 256) : h * 512 + 256 + d < 24 * 512 := by
  omega

/-- KV append: a thread's channel `g*64 + tid` is below the 256 head dim. -/
theorem kv_channel_lt (g tid : Nat) (hg : g < 4) (ht : tid < 64) : g * 64 + tid < 256 := by
  omega

/-- KV append quant-scale slot `row*4 + g` is exactly one slot per 64-channel group. -/
theorem kv_scale_slot (row g : Nat) (hg : g < 4) : row * 4 + g < (row + 1) * 4 := by
  omega

/-- Rope pairs `(pair, pair+32)` are distinct halves of a 64-dim rotary block. -/
theorem rope_pair_in_block (pair : Nat) (h : pair < 32) : pair < pair + 32 ∧ pair + 32 < 64 := by
  omega

/-- Rope theta scale exponent `-2/64` corresponds to 32 frequency pairs in 64 dims. -/
theorem rope_pairs_count : 64 / 2 = 32 := by decide

end Strata.PrefillKernels
