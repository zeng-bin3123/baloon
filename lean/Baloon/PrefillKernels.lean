-- Baloon.PrefillKernels: invariants of the batched-prompt CUDA kernels.
-- Independent implementation proving the same mathematical facts as the
-- theory layer with our own names and proof terms.

namespace Baloon.PrefillKernels

-- ── § hyper-connection geometry (N=2560, HC=4, D=N*HC=10240) ─────────────────

/-- Recompose flat index i from token, stream, dim coordinates. -/
theorem hcIndexRecompose (i : Nat) :
    i / 10240 * 10240 + (i % 10240) / 2560 * 2560 + i % 2560 = i := by omega

/-- Stream coordinate c = (i % D) / N is a valid stream (< HC = 4). -/
theorem hcStreamLt (i : Nat) : (i % 10240) / 2560 < 4 := by omega

/-- Dimension coordinate d = i % N is a valid embedding index (< N = 2560). -/
theorem hcDimLt (i : Nat) : i % 2560 < 2560 := by omega

/-- Token coordinate t = i / D is in range when i < T*D. -/
theorem hcTokLt (T i : Nat) (h : i < T * 10240) : i / 10240 < T := by omega

/-- gr_broadcast source index t*N + d is inside the T*N embedding block. -/
theorem broadcastSrcLt (T i : Nat) (h : i < T * 10240) :
    i / 10240 * 2560 + i % 2560 < T * 2560 := by omega

/-- gr_write injection index t*ld + c stays inside (t+1)*ld when ld ≥ 4. -/
theorem writeInjLt (i ld : Nat) (hld : 4 ≤ ld) :
    i / 10240 * ld + (i % 10240) / 2560 < (i / 10240 + 1) * ld := by
  have : (i % 10240) / 2560 < 4 := by omega
  have h2 : (i / 10240 + 1) * ld = i / 10240 * ld + ld := by rw [Nat.succ_mul]
  omega

-- ── § launch grids (blocksFor = ceiling division) ────────────────────────────

/-- `blocks_for(n, t)`: smallest number of t-sized blocks covering n elements. -/
def blocksFor (n t : Nat) : Nat := (n + (t - 1)) / t

/-- A ceiling-divided grid covers every element: n ≤ blocksFor(n,t) * t. -/
theorem blocksFor_cover (n t : Nat) (ht : 0 < t) : n ≤ blocksFor n t * t := by
  unfold blocksFor
  have h := (Nat.div_lt_iff_lt_mul ht).mp (Nat.lt_succ_self ((n + (t - 1)) / t))
  rw [Nat.succ_mul] at h; omega

/-- GDN conv grid: C/128 blocks of 128 threads covers all C = 10240 channels. -/
theorem gdnConvGrid : 10240 / 128 * 128 = 10240 := by decide

/-- QKV channel layout: (2*HK + HV) * S = (2*16 + 48) * 128 = 10240. -/
theorem gdnChannels : (2 * 16 + 48) * 128 = 10240 := by decide

/-- RG row-groups cover all S = 128 rows: 4 * (128/4) = 128. -/
theorem gdnRowsCover : 4 * (128 / 4) = 128 := by decide

/-- A row-group's rows rg*RPG + r stay below S = 128. -/
theorem gdnRowLt (rg r : Nat) (hrg : rg < 4) (hr : r < 32) : rg * 32 + r < 128 := by
  omega

-- ── § expert blob layout (blob_dequant_kernel) ───────────────────────────────

/-- Gate/up code section: 1280 rows × 640 bytes. -/
def guCodes : Nat := (1280 * 640)
/-- Down codes section offset. -/
def downOff : Nat := (guCodes)
/-- Gate/up scales offset: 819200 + 409600 bytes of gate/up and down codes. -/
def guScaleOff : Nat := 1228800
/-- Down scales offset: after 1280 × 40 f16 gate/up scales. -/
def downScaleOff : Nat := 1331200
/-- Total blob size: 1,382,400 bytes. -/
def blobBytes : Nat := 1382400

/-- The blob is exactly 1,382,400 bytes. -/
theorem blobBytes_value : blobBytes = 1382400 := rfl

/-- Section offsets are strictly increasing (sections are disjoint). -/
theorem blobSectionsOrdered :
    downOff < guScaleOff ∧ guScaleOff < downScaleOff ∧ downScaleOff < blobBytes := by
  decide

/-- Gate/up scale index row*40 + (byte*4)/64 stays within the 1280×40 table. -/
theorem guScaleIdxLt (row byte : Nat) (hr : row < 1280) (hb : byte < 640) :
    row * 40 + (byte * 4) / 64 < 1280 * 40 := by omega

/-- Down scale index row*10 + (byte*4)/64 stays within the 2560×10 table. -/
theorem downScaleIdxLt (row byte : Nat) (hr : row < 2560) (hb : byte < 160) :
    row * 10 + (byte * 4) / 64 < 2560 * 10 := by omega

/-- Four dequantized outputs of one gate/up code byte are inside the 1280×2560 matrix. -/
theorem guOutIdxLt (row byte k : Nat) (hr : row < 1280) (hb : byte < 640) (hk : k < 4) :
    row * 2560 + byte * 4 + k < 1280 * 2560 := by omega

/-- Four dequantized outputs of one down code byte are inside the 2560×640 matrix. -/
theorem downOutIdxLt (row byte k : Nat) (hr : row < 2560) (hb : byte < 160) (hk : k < 4) :
    row * 640 + byte * 4 + k < 2560 * 640 := by omega

-- ── § SwiGLU and gather ───────────────────────────────────────────────────────

/-- SwiGLU interleaved: columns 2k and 2k+1 of a 1280-wide row (k < 640) are in range. -/
theorem swigluIlIdx (k : Nat) (hk : k < 640) : 2 * k + 1 < 1280 := by omega

/-- gather_rows16 moves width/8 uint4 per row; for width=2560 that is 320. -/
theorem gatherPerRow : 2560 / 8 = 320 := by decide

-- ── § router lane layout (route_kernel, 32 lanes × 16 regs = 512 experts) ────

/-- Expert lane+i*32 is recovered as (lane+i*32) % 32 = lane. -/
theorem routeLane (lane i : Nat) (hl : lane < 32) : (lane + i * 32) % 32 = lane := by omega

/-- Register index recovered by (lane+i*32)/32 = i. -/
theorem routeReg (lane i : Nat) (hl : lane < 32) : (lane + i * 32) / 32 = i := by omega

/-- 32 lanes × 16 registers cover all 512 experts. -/
theorem routeCover (lane i : Nat) (hl : lane < 32) (hi : i < 16) :
    lane + i * 32 < 512 := by omega

/-- Top-k output index t*10 + rank is inside the T*10 output array. -/
theorem routeOutIdx (t T rank : Nat) (ht : t < T) (hr : rank < 10) :
    t * 10 + rank < T * 10 := by
  have h : (t + 1) * 10 ≤ T * 10 := Nat.mul_le_mul_right 10 ht
  have h2 : (t + 1) * 10 = t * 10 + 10 := by rw [Nat.succ_mul]
  omega

-- ── § QSA helper indexing (split_q, gate_attn, kv_append, rope) ──────────────

/-- split_q recomposition: i = t*6144 + h*256 + d. -/
theorem splitQRecompose (i : Nat) :
    i / 6144 * 6144 + ((i / 256) % 24) * 256 + i % 256 = i := by omega

/-- Gate half of a full-q row at h*512 + 256 + d is inside 24*512. -/
theorem qfullGateIdxLt (h d : Nat) (hh : h < 24) (hd : d < 256) :
    h * 512 + 256 + d < 24 * 512 := by omega

/-- kv_append channel g*64 + tid is below 256 head dims. -/
theorem kvChannelLt (g tid : Nat) (hg : g < 4) (ht : tid < 64) :
    g * 64 + tid < 256 := by omega

/-- KV quant-scale slot row*4 + g is exactly one slot per 64-channel group. -/
theorem kvScaleSlot (row g : Nat) (hg : g < 4) : row * 4 + g < (row + 1) * 4 := by omega

/-- RoPE pairs (pair, pair+32) are distinct halves of a 64-dim rotary block. -/
theorem ropePairInBlock (pair : Nat) (h : pair < 32) :
    pair < pair + 32 ∧ pair + 32 < 64 := by omega

/-- RoPE theta scale exponent: 32 frequency pairs in 64 dims. -/
theorem ropePairsCount : 64 / 2 = 32 := by decide

-- ── § test_index_invariants check lines (test_prefill_kernels.cu) ─────────────

namespace TestInvariants

/-- 32 lanes × 16 registers cover all 512 experts exactly. -/
theorem kExperts_cover : 32 * 16 = 512 := by decide

/-- blocksFor(10, 4) * 4 ≥ 10 (ceiling-division covers 10 elements in blocks of 4). -/
theorem blocksFor_10_4 : blocksFor 10 4 * 4 ≥ 10 := by decide

/-- blocksFor(8, 4) * 4 = 8 (exact division). -/
theorem blocksFor_8_4 : blocksFor 8 4 * 4 = 8 := by decide

/-- blocksFor(1, 256) = 1 (one block needed for one element). -/
theorem blocksFor_1_256 : blocksFor 1 256 = 1 := by decide

/-- RG * (S / RG) = 128 (row groups cover all S rows). -/
theorem rg_covers_s : 4 * (128 / 4) = 128 := by decide

end TestInvariants

end Baloon.PrefillKernels
