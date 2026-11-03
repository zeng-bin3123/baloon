/-!
# MODULE: prefill_kernels
Index, layout and clamping arithmetic of the prefill CUDA kernels: hyper-connection read/write, GDN conv and
recurrence, MoE routing/dequantization/combine, and QSA helpers (RMS, RoPE, KV append with INT8 quantization).

## Source semantics
* Launch grids cover their element count: `blocks = (n + 255) / 256` blocks of 256 threads.
* Residual streams are `[T][4][2560]` (D = 10240); thread `i` decodes to `(t, c, d)` with `c < 4`, `d < 2560`.
* A Strata expert blob is `1280*640` gate/up code bytes, `2560*160` down code bytes, `1280*40` and `2560*10`
  fp16 scales: 1,382,400 bytes in total. One code byte holds 4 ternary weights sharing a 64-weight scale group.
* Routing: lane `l` holds experts `l + 32*i`, `i < 16`, covering 512 experts exactly once.
* GDN recurrence: 4 row groups of 32 rows tile the 128-row state; q/k/v slices of the 10240-wide conv output do not
  overlap.
* KV append: grid z in 0..7 splits into group `z/2 < 4` and K/V flag; INT8 values are clamped to [-127, 127].

## Form template identifiers
`blocks_for` -> `blocksFor`; D, N, HC -> literals 10240, 2560, 4; expert blob offsets -> `O_D_CODES`, `O_GU_SC`,
`O_D_SC`; clamp -> `clampQ`.
-/

namespace Strata.Prefill.Kernels

-- § launch geometry

/-- Number of 256-thread blocks for `n` elements. -/
def blocksFor (n : Nat) : Nat := (n + 255) / 256

/-- The launch covers all `n` elements. -/
theorem blocks_cover (n : Nat) : n ≤ blocksFor n * 256 := by
  unfold blocksFor
  omega

/-- The launch overshoots by less than one block. -/
theorem blocks_tight (n : Nat) : blocksFor n * 256 < n + 256 := by
  unfold blocksFor
  omega

/-- Route kernel: 8 warps per block cover all `T` tokens. -/
theorem route_blocks_cover (T : Nat) : T ≤ (T + 7) / 8 * 8 := by
  omega

-- § hyper-connection

/-- Decoding a thread index `i < T*10240` into `(t, c, d)` is exact, with `c < 4` and `d < 2560` and `t < T`. -/
theorem gr_write_decomp (i T : Nat) (h : i < T * 10240) :
    i / 10240 < T ∧ i % 10240 / 2560 < 4 ∧ i % 2560 < 2560 ∧
      i / 10240 * 10240 + i % 10240 / 2560 * 2560 + i % 2560 = i := by
  omega

/-- `gr_mix` reads `t*10240 + c*2560 + d` for `c < 4`, which is inside `[T][10240]`. -/
theorem gr_mix_read_in_range (i T c : Nat) (hi : i < T * 2560) (hc : c < 4) :
    i / 2560 * 10240 + c * 2560 + i % 2560 < T * 10240 := by
  omega

/-- A `gr_norm` row `t*4 + c` has channel `c = row % 4`, and its elements stay in the 10240-wide norm weight. -/
theorem gr_norm_channel (row d : Nat) (hd : d < 2560) : row % 4 * 2560 + d < 10240 := by
  omega

-- § GDN

/-- The conv kernel's 80 blocks of 128 threads cover exactly the 10240 channels. -/
theorem gdn_conv_grid : 10240 / 128 * 128 = 10240 := by decide

/-- Recurrence: row group `rg < 4` and local row `r < 32` tile the 128 state rows with a unique decomposition. -/
theorem rec_rows_tile (rg r : Nat) (hg : rg < 4) (hr : r < 32) :
    rg * 32 + r < 128 ∧ (rg * 32 + r) / 32 = rg ∧ (rg * 32 + r) % 32 = r := by
  omega

/-- The key slice offset `16*128 + qh*128 + tid` stays in the second quarter of the conv output. -/
theorem rec_key_in_range (head tid : Nat) (hh : head < 48) (ht : tid < 128) :
    16 * 128 + head % 16 * 128 + tid < 2 * 16 * 128 := by
  omega

/-- The value slice `2*16*128 + head*128 + col` stays inside the 10240 channels. -/
theorem rec_value_in_range (head col : Nat) (hh : head < 48) (hc : col < 128) :
    2 * 16 * 128 + head * 128 + col < 10240 := by
  omega

/-- The L2-norm kernel over 32 q/k heads touches only the first 4096 channels. -/
theorem l2_in_range (head tid : Nat) (hh : head < 32) (ht : tid < 128) : head * 128 + tid < 4096 := by
  omega

/-- `gdn_gates` decodes `i < T*48` into `t < T` and `h < 48`. -/
theorem gates_decomp (i T : Nat) (h : i < T * 48) : i / 48 < T ∧ i % 48 < 48 := by
  omega

-- § MoE

/-- Lane/expert decomposition in the routing kernel. -/
theorem lane_expert (lane i : Nat) (hl : lane < 32) (hi : i < 16) :
    lane + i * 32 < 512 ∧ (lane + i * 32) / 32 = i ∧ (lane + i * 32) % 32 = lane := by
  omega

/-- Every expert id below 512 is held by exactly one (lane, i). -/
theorem expert_held (e : Nat) (he : e < 512) : e % 32 < 32 ∧ e / 32 < 16 ∧ e % 32 + e / 32 * 32 = e := by
  omega

/-- Expert blob layout: sections are contiguous and total 1,382,400 bytes (the planner's `expert_blob`). -/
theorem blob_layout :
    1280 * 640 = 819200 ∧ 819200 + 2560 * 160 = 1228800 ∧ 1228800 + 1280 * 40 * 2 = 1331200 ∧
      1331200 + 2560 * 10 * 2 = 1382400 := by decide

/-- Gate/up scale index `row*40 + byte*4/64` is within the `1280*40` scales. -/
theorem gu_scale_idx (row byte : Nat) (hr : row < 1280) (hb : byte < 640) :
    row * 40 + byte * 4 / 64 < 1280 * 40 := by
  omega

/-- Down scale index `row*10 + byte*4/64` is within the `2560*10` scales. -/
theorem d_scale_idx (row byte : Nat) (hr : row < 2560) (hb : byte < 160) :
    row * 10 + byte * 4 / 64 < 2560 * 10 := by
  omega

/-- Gate/up outputs `row*2560 + byte*4 + k` are within `1280*2560` FP16 elements. -/
theorem gu_out_idx (row byte k : Nat) (hr : row < 1280) (hb : byte < 640) (hk : k < 4) :
    row * 2560 + byte * 4 + k < 1280 * 2560 := by
  omega

/-- Down outputs `row*640 + byte*4 + k` are within `2560*640` FP16 elements. -/
theorem d_out_idx (row byte k : Nat) (hr : row < 2560) (hb : byte < 160) (hk : k < 4) :
    row * 640 + byte * 4 + k < 2560 * 640 := by
  omega

/-- A thread index in the down range decodes to a valid `(row, byte)`. -/
theorem d_thread_decomp (i : Nat) (hlo : 1280 * 640 ≤ i) (hhi : i < 1280 * 640 + 2560 * 160) :
    (i - 1280 * 640) / 160 < 2560 ∧ (i - 1280 * 640) % 160 < 160 := by
  omega

/-- A 2-bit code field is below 4. -/
theorem code_field_lt (c k : Nat) : (c >>> (2 * k)) % 4 < 4 := Nat.mod_lt _ (by decide)

/-- Interleaved SwiGLU reads `2k` and `2k+1` below 1280 and row `r < n`. -/
theorem swiglu_il_range (i n : Nat) (h : i < n * 640) :
    i / 640 < n ∧ i / 640 * 1280 + 2 * (i % 640) + 1 < n * 1280 := by
  omega

/-- Gathering rows moves `width/8` 16-byte vectors per row; for width 2560 that is 320. -/
theorem gather_vectors : 2560 / 8 = 320 ∧ 8 * 2 = 16 := by decide

-- § QSA helpers

/-- `split_q` reads the first 256 of each 512-wide head slot. -/
theorem split_q_range (h d : Nat) (hh : h < 24) (hd : d < 256) : h * 512 + d < 24 * 512 := by
  omega

/-- `gate_attn` reads the gate half `h*512 + 256 + d`, also inside the slot. -/
theorem gate_attn_range (h d : Nat) (hh : h < 24) (hd : d < 256) : h * 512 + 256 + d < 24 * 512 := by
  omega

/-- RoPE rotates pairs `(pair, pair + 32)` inside a 64-wide rotary prefix. -/
theorem rope_pair_range (pair : Nat) (h : pair < 32) : pair + 32 < 64 := by
  omega

/-- Head packing: K (2 heads of 256), Q (24 heads of 256), indexer Q (4 heads of 128) fit their row strides. -/
theorem head_strides : 2 * 256 = 512 ∧ 24 * 256 = 6144 ∧ 4 * 128 = 512 := by decide

/-- KV append grid: `z < 8` gives group `z/2 < 4`, and the written element `g*64 + tid` is below 256. -/
theorem kv_grid (z tid : Nat) (hz : z < 8) (ht : tid < 64) : z / 2 < 4 ∧ z / 2 * 64 + tid < 256 := by
  omega

/-- The paged row `page*2 + kvh` is inside `2 * pages` for `kvh < 2`. -/
theorem kv_page_row (page pages kvh : Nat) (hp : page < pages) (hk : kvh < 2) : page * 2 + kvh < pages * 2 := by
  omega

/-- INT8 clamp used by `kv_append`. -/
def clampQ (q : Int) : Int := if q < -127 then -127 else if q > 127 then 127 else q

/-- The clamped value is a legal INT8 code (never -128). -/
theorem clampQ_range (q : Int) : -127 ≤ clampQ q ∧ clampQ q ≤ 127 := by
  unfold clampQ
  split_ifs <;> omega

/-- In range, clamping is the identity. -/
theorem clampQ_id (q : Int) (h1 : -127 ≤ q) (h2 : q ≤ 127) : clampQ q = q := by
  unfold clampQ
  split_ifs <;> omega

end Strata.Prefill.Kernels
