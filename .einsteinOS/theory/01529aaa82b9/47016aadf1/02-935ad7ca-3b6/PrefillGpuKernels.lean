/-!
# MODULE: prefill_gpu_kernels
Index arithmetic, layout and numeric-range contracts of the CUDA kernels of the batched prefill path
(hyper-connection, gated-delta-net, MoE dequant/route/combine, QSA helpers, KV append).

## Source semantics
Every kernel is a flat grid over a row-major buffer; correctness rests on (a) the grid covering the buffer,
(b) every decomposed index staying inside its buffer, (c) the packed expert blob layout summing to one
expert's byte size, (d) bf16 rounding producing a 16-bit pattern and being exact on bf16 values, and
(e) the int8 KV quantizer clamping to [-127,127].

## Form template identifiers
- `blocksFor`      <- blocks_for (ceil-divide launch size)
- `bfBits`         <- bf (round-to-nearest-even f32 -> bf16 bits)
- `code2`/`dequantVal` <- 2-bit ternary-ish code decode in blob_dequant_kernel
- `O_D_CODES`, `O_GU_SC`, `O_D_SC`, `BLOB_BYTES` <- Strata expert blob offsets
- `clamp127`       <- int8 clamp in kv_append_kernel
-/

namespace Strata.PrefillKernels

-- § constants

def N : Nat := 2560
def HC : Nat := 4
def D : Nat := N * HC
def LR : Nat := 320
def S : Nat := 128
def HK : Nat := 16
def HV : Nat := 48
def C : Nat := 10240
def ZV : Nat := HV * S

/-- The four hyper-connection streams are `4 * 2560 = 10240` wide. -/
theorem hc_width : D = 10240 := by decide

/-- The GDN conv channels split into q (16 heads), k (16 heads) and v (48 heads) of width 128. -/
theorem qkv_channels : C = 2 * HK * S + HV * S := by decide

/-- The value part of the GDN projection is `48 * 128 = 6144` wide. -/
theorem zv_value : ZV = 6144 := by decide

/-- The full q projection is 24 heads of `2 * 256` (query and gate). -/
theorem q_full_width : 24 * 512 = 12288 := by decide

-- § launch geometry

/-- Ceil-divide used to size 1-D grids. -/
def blocksFor (n t : Nat) : Nat := (n + t - 1) / t

/-- A grid of `blocksFor n t` blocks of `t` threads covers all `n` elements. -/
theorem blocksFor_covers (n t : Nat) (ht : 0 < t) : n ≤ blocksFor n t * t := by
  unfold blocksFor
  have h1 := Nat.div_add_mod (n + t - 1) t
  have h2 := Nat.mod_lt (n + t - 1) ht
  rw [Nat.mul_comm]
  omega

/-- ... and never launches a whole spare block. -/
theorem blocksFor_tight (n t : Nat) (ht : 0 < t) : blocksFor n t * t < n + t := by
  unfold blocksFor
  have h1 := Nat.div_add_mod (n + t - 1) t
  have h2 := Nat.mod_lt (n + t - 1) ht
  rw [Nat.mul_comm]
  omega

/-- The router launches `(T+7)/8` blocks of 8 warps: every token gets a warp. -/
theorem route_grid_covers (T : Nat) : T ≤ ((T + 7) / 8) * 8 := by omega

/-- `gdn_conv` launches `C/128` blocks of 128 threads: exactly one thread per channel. -/
theorem gdn_conv_grid_exact : C / 128 * 128 = C := by decide

/-- `gdn_l2` runs one block of `S` threads per q/k head: `2*HK` blocks cover the q/k channels. -/
theorem gdn_l2_covers_qk : 2 * HK * S = 4096 := by decide

/-- A block-wide sum over at most 1024 threads uses at most 32 warp partials (the `sh[32]` array). -/
theorem block_sum_warps_le (b : Nat) (hb : b ≤ 1024) : (b + 31) / 32 ≤ 32 := by omega

/-- Lane/warp decomposition of a thread id. -/
theorem lane_warp_decompose (tid : Nat) : tid = (tid / 32) * 32 + tid % 32 := by omega

/-- The GDN recurrence splits the 128 state rows over 4 row groups of 32. -/
theorem rec_rows_partition (row : Nat) (hr : row < 128) :
    ∃ rg r, rg < 4 ∧ r < 32 ∧ row = rg * 32 + r :=
  ⟨row / 32, row % 32, by omega, by omega, by omega⟩

/-- The state index `((row*HV + head)*S + col)` stays inside the `S*HV*S` state. -/
theorem gdn_state_index_lt (row head col : Nat) (hr : row < 128) (hh : head < 48) (hc : col < 128) :
    (row * 48 + head) * 128 + col < 128 * 48 * 128 := by omega

/-- RoPE rotates pairs `(p, p+32)` and the 32 pair ids cover all 64 rotated dims. -/
theorem rope_pairs_cover (i : Nat) (hi : i < 64) : i < 32 ∨ (32 ≤ i ∧ i - 32 < 32) := by omega

-- § hyper-connection index maps

/-- Decomposing `t*D + c*N + d` recovers token, stream and channel. -/
theorem hc_decompose (t c d : Nat) (hc : c < 4) (hd : d < 2560) :
    (t * 10240 + c * 2560 + d) / 10240 = t ∧
    ((t * 10240 + c * 2560 + d) % 10240) / 2560 = c ∧
    (t * 10240 + c * 2560 + d) % 2560 = d := by omega

/-- `gr_mix` reads stay inside the `T*D` stream buffer. -/
theorem mix_read_in_range (t T c d : Nat) (ht : t < T) (hc : c < 4) (hd : d < 2560) :
    t * 10240 + c * 2560 + d < T * 10240 := by omega

/-- `moe_combine` reads the 10 routed slots of token `t` inside the `T*10` tables. -/
theorem combine_slot_in_range (t T k : Nat) (ht : t < T) (hk : k < 10) : t * 10 + k < T * 10 := by omega

/-- A flat index `r*640 + k` decomposes back into row and column (swiglu kernels). -/
theorem swiglu_flat_decompose (r k : Nat) (hk : k < 640) :
    (r * 640 + k) / 640 = r ∧ (r * 640 + k) % 640 = k := by omega

/-- The interleaved gate/up reads `2k` and `2k+1` stay inside the `n*1280` buffer. -/
theorem swiglu_il_index (r k n : Nat) (hr : r < n) (hk : k < 640) :
    r * 1280 + 2 * k + 1 < n * 1280 := by omega

/-- Generic "row-major cell is inside the buffer" bound. -/
theorem row_lt (q M ps r : Nat) (hq : q < M) (hr : r < ps) : q * ps + r < M * ps := by
  have h1 : (q + 1) * ps ≤ M * ps := Nat.mul_le_mul_right ps hq
  have h2 : (q + 1) * ps = q * ps + ps := by rw [Nat.add_mul, Nat.one_mul]
  omega

/-- `gather_rows16` copies `uint4` units; a row/unit pair stays inside the destination. -/
theorem gather_index_lt (r j n per : Nat) (hr : r < n) (hj : j < per) : r * per + j < n * per :=
  row_lt r n per j hr hj

-- § expert blob layout

def O_D_CODES : Nat := 1280 * 640
def O_GU_SC : Nat := O_D_CODES + 2560 * 160
def O_D_SC : Nat := O_GU_SC + 1280 * 40 * 2
def BLOB_BYTES : Nat := O_D_SC + 2560 * 10 * 2

/-- One expert blob (codes + f16 scales) is exactly 1,382,400 bytes, the planner's `expert_blob`. -/
theorem blob_bytes_value : BLOB_BYTES = 1382400 := by decide

/-- The four regions of the blob are laid out in increasing, non-overlapping order. -/
theorem blob_regions_ordered : O_D_CODES < O_GU_SC ∧ O_GU_SC < O_D_SC ∧ O_D_SC < BLOB_BYTES := by decide

/-- The 2-bit code of weight `k` inside a code byte. -/
def code2 (c k : Nat) : Nat := (c >>> (2 * k)) &&& 3

/-- A 2-bit code is at most 3. -/
theorem code2_le (c k : Nat) : code2 c k ≤ 3 := Nat.and_le_right

/-- The decoded multiplier `code - 1`, before the block scale. -/
def dequantVal (c k : Nat) : Int := (code2 c k : Int) - 1

/-- Decoded multipliers lie in `[-1, 2]`. -/
theorem dequantVal_range (c k : Nat) : -1 ≤ dequantVal c k ∧ dequantVal c k ≤ 2 := by
  have := code2_le c k
  unfold dequantVal
  omega

/-- Gate/up output index `row*2560 + byte*4 + k` stays inside the `1280 x 2560` matrix. -/
theorem gu_index_lt (row byte k : Nat) (hr : row < 1280) (hb : byte < 640) (hk : k < 4) :
    row * 2560 + byte * 4 + k < 1280 * 2560 := by omega

/-- Down output index stays inside the `2560 x 640` matrix. -/
theorem down_index_lt (row byte k : Nat) (hr : row < 2560) (hb : byte < 160) (hk : k < 4) :
    row * 640 + byte * 4 + k < 2560 * 640 := by omega

/-- Gate/up scale index (one scale per 64 weights, 40 per row) stays in range. -/
theorem gu_scale_index_lt (row byte : Nat) (hr : row < 1280) (hb : byte < 640) :
    row * 40 + (byte * 4) / 64 < 1280 * 40 := by omega

/-- Down scale index (10 per row) stays in range. -/
theorem down_scale_index_lt (row byte : Nat) (hr : row < 2560) (hb : byte < 160) :
    row * 10 + (byte * 4) / 64 < 2560 * 10 := by omega

/-- Distinct (byte, k) pairs write distinct output weights. -/
theorem code_slot_injective (b1 k1 b2 k2 : Nat) (h1 : k1 < 4) (h2 : k2 < 4)
    (h : b1 * 4 + k1 = b2 * 4 + k2) : b1 = b2 ∧ k1 = k2 := by omega

-- § numeric formats

/-- bf16 rounding of an f32 bit pattern (round-to-nearest-even, uint32 wrap), as bits. -/
def bfBits (u : Nat) : Nat := ((u + 0x7fff + ((u >>> 16) &&& 1)) % 4294967296) >>> 16

/-- The bf16 result always fits in 16 bits. -/
theorem bfBits_lt (u : Nat) : bfBits u < 65536 := by
  unfold bfBits
  have := Nat.mod_lt (u + 0x7fff + ((u >>> 16) &&& 1)) (by decide : 0 < 4294967296)
  simp only [Nat.shiftRight_eq_div_pow, Nat.reducePow] at this ⊢
  omega

/-- Rounding a value that is already bf16 (low 16 bits zero) is the identity. -/
theorem bfBits_exact (h : Nat) (hh : h < 65536) : bfBits (h * 65536) = h := by
  have hb : ((h * 65536) >>> 16) &&& 1 ≤ 1 := Nat.and_le_right
  unfold bfBits
  simp only [Nat.shiftRight_eq_div_pow, Nat.reducePow] at hb ⊢
  omega

/-- The int8 clamp of the KV quantizer. -/
def clamp127 (q : Int) : Int := if q < -127 then -127 else if q > 127 then 127 else q

/-- Quantized KV values lie in `[-127, 127]`. -/
theorem clamp127_range (q : Int) : -127 ≤ clamp127 q ∧ clamp127 q ≤ 127 := by
  unfold clamp127
  repeat' split
  all_goals omega

/-- Clamping leaves in-range values unchanged. -/
theorem clamp127_id (q : Int) (h1 : -127 ≤ q) (h2 : q ≤ 127) : clamp127 q = q := by
  unfold clamp127
  repeat' split
  all_goals omega

/-- KV pool row `(page*2 + kvh)*page_size + pos % page_size` stays inside the pool. -/
theorem kv_row_lt (page kvh nPages ps r : Nat) (hp : page < nPages) (hk : kvh < 2) (hr : r < ps) :
    (page * 2 + kvh) * ps + r < (nPages * 2) * ps :=
  row_lt (page * 2 + kvh) (nPages * 2) ps r (by omega) hr

end Strata.PrefillKernels
