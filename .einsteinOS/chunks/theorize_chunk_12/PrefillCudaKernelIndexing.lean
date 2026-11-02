/-!
# MODULE: prefill_cuda_kernel_indexing
CUDA kernels of the chunked prefill path: hyper-connection norm/mix/write, GDN gates/conv/recurrence,
MoE routing, expert-blob dequantization, SwiGLU, row gathering, MoE combine, QSA helpers and KV append.
This module theorizes the index arithmetic and value ranges those kernels rely on.

## Source semantics
* One thread per flat output index `i`; `(t, d) = (i / N, i % N)` must tile `[0, T*N)` exactly.
* Model dimensions are fixed: `N = 2560`, `HC = 4`, `D = N*HC`, `C = 2*HK*S + HV*S`.
* The expert blob is a packed layout: gate/up codes, down codes, gate/up f16 scales, down f16 scales.
  Each code byte carries four 2-bit codes mapped to `{-1,0,1,2}` and multiplied by a block scale (block = 64 weights).
* KV append addresses `(page*2 + kvh)*page_size + pos % page_size`, and quantized values are clamped to `[-127,127]`.

## Form template identifiers
* `N` → hidden width; `HC` → number of residual streams; `D` → flat residual width per token
* `blobDCodes/blobGuScales/blobDScales` → byte offsets in the expert blob
* `clamp127` → int8 saturation; `kvRow` → pooled KV row address
-/

namespace Strata.Prefill.Kernels

-- § model constants

/-- Hidden width. -/
def N : Nat := 2560
/-- Number of hyper-connection streams. -/
def HC : Nat := 4
/-- Flat residual width per token. -/
def D : Nat := N * HC
/-- Hyper-connection low-rank width. -/
def LR : Nat := 320
/-- GDN head state size. -/
def S : Nat := 128
/-- Number of GDN q/k heads. -/
def HK : Nat := 16
/-- Number of GDN value heads. -/
def HV : Nat := 48
/-- Conv channels: q, k and v. -/
def C : Nat := 10240

/-- The residual width is four hidden widths. -/
theorem D_eq : D = 10240 := by decide

/-- The conv channel count splits into q, k and v parts. -/
theorem C_decomp : C = 2 * HK * S + HV * S := by decide

/-- `gr_silu` covers `T*LR` entries and `LR` is a 32nd of `D`. -/
theorem LR_eq : LR = D / 32 := by decide

/-- A GDN value head row group: 4 row groups of 32 rows cover the state size. -/
theorem rows_per_group : S / 4 = 32 ∧ 4 * (S / 4) = S := by decide

-- § flat index decomposition

/-- Splitting a flat index into `(i / n, i % n)` stays in range and reassembles. -/
theorem index_split (i T n : Nat) (hn : 0 < n) (h : i < T * n) :
    i / n < T ∧ i % n < n ∧ (i / n) * n + i % n = i := by
  refine ⟨(Nat.div_lt_iff_lt_mul hn).2 h, Nat.mod_lt _ hn, ?_⟩
  rw [Nat.mul_comm (i / n) n]
  exact Nat.div_add_mod i n

/-- In `gr_write`, the channel index `(i % D) / N` is below `HC`. -/
theorem channel_lt (i : Nat) : (i % D) / N < HC := by
  have h : i % D < D := Nat.mod_lt _ (by decide)
  have hD : D = 10240 := by decide
  have hN : N = 2560 := by decide
  have hH : HC = 4 := by decide
  rw [hN, hH, hD]
  rw [hD] at h
  omega

/-- In `gr_broadcast`, `i % N` is a valid hidden column even though `i` runs to `T*D`. -/
theorem broadcast_col_lt (i : Nat) : i % N < N := Nat.mod_lt _ (by decide)

-- § expert blob layout

/-- Byte offset of the down codes. -/
def blobDCodes : Nat := 1280 * 640
/-- Byte offset of the gate/up scales. -/
def blobGuScales : Nat := blobDCodes + 2560 * 160
/-- Byte offset of the down scales. -/
def blobDScales : Nat := blobGuScales + 1280 * 40 * 2
/-- Total blob size. -/
def blobTotal : Nat := blobDScales + 2560 * 10 * 2

/-- The concrete offsets used by `blob_dequant_kernel`. -/
theorem blob_offsets :
    blobDCodes = 819200 ∧ blobGuScales = 1228800 ∧ blobDScales = 1331200 ∧ blobTotal = 1382400 := by
  decide

/-- Four codes per byte times 640 bytes is one gate/up row of `N` weights. -/
theorem gu_row_width : 640 * 4 = N := by decide

/-- Four codes per byte times 160 bytes is one down row of 640 weights. -/
theorem down_row_width : 160 * 4 = 640 := by decide

/-- The gate/up scale index of a code byte is below 40 per row. -/
theorem gu_scale_idx_lt (byte : Nat) (h : byte < 640) : (byte * 4) / 64 < 40 := by omega

/-- The down scale index of a code byte is below 10 per row. -/
theorem d_scale_idx_lt (byte : Nat) (h : byte < 160) : (byte * 4) / 64 < 10 := by omega

/-- Gate/up output positions stay inside the `1280 x 2560` dequantized matrix. -/
theorem gu_out_lt (row byte k : Nat) (hr : row < 1280) (hb : byte < 640) (hk : k < 4) :
    row * 2560 + byte * 4 + k < 1280 * 2560 := by omega

/-- Down output positions stay inside the `2560 x 640` dequantized matrix. -/
theorem d_out_lt (row byte k : Nat) (hr : row < 2560) (hb : byte < 160) (hk : k < 4) :
    row * 640 + byte * 4 + k < 2560 * 640 := by omega

/-- A 2-bit code is mapped to `code - 1`. -/
def code2 (c k : Nat) : Int := (((c >>> (2 * k)) &&& 3 : Nat) : Int) - 1

/-- Decoded codes lie in `{-1,0,1,2}`. -/
theorem code2_range (c k : Nat) : -1 ≤ code2 c k ∧ code2 c k ≤ 2 := by
  have h : (c >>> (2 * k)) &&& 3 ≤ 3 := Nat.and_le_right
  unfold code2
  omega

-- § routing and combine addressing

/-- `route_kernel` writes `ids[t*10+rank]` strictly inside `[0, T*10)`. -/
theorem route_slot_lt (t rank T : Nat) (ht : t < T) (hr : rank < 10) : t * 10 + rank < T * 10 := by
  omega

/-- `moe_combine` reads `Dm[slot*N + d]` inside the `rows x N` buffer when `slot < rows`. -/
theorem combine_read_lt (slot d rows : Nat) (hs : slot < rows) (hd : d < N) :
    slot * N + d < rows * N := by
  have h1 : (slot + 1) * N ≤ rows * N := Nat.mul_le_mul_right N hs
  have h2 : (slot + 1) * N = slot * N + N := by rw [Nat.add_mul, Nat.one_mul]
  omega

/-- `gather_rows16` handles eight bf16 per `uint4`, so widths divisible by 8 give integral groups. -/
theorem gather_groups (width : Nat) (h : width % 8 = 0) : (width / 8) * 8 = width := by omega

/-- The SwiGLU interleaved layout reads `2k` and `2k+1` below 1280 for `k < 640`. -/
theorem swiglu_il_pair_lt (k : Nat) (hk : k < 640) : 2 * k + 1 < 1280 := by omega

-- § KV append

/-- Saturating int8 clamp. -/
def clamp127 (q : Int) : Int := if q < -127 then -127 else if q > 127 then 127 else q

/-- The clamp output is a valid symmetric int8 code. -/
theorem clamp127_range (q : Int) : -127 ≤ clamp127 q ∧ clamp127 q ≤ 127 := by
  unfold clamp127
  split_ifs <;> omega

/-- Clamp is the identity on in-range values. -/
theorem clamp127_id (q : Int) (h1 : -127 ≤ q) (h2 : q ≤ 127) : clamp127 q = q := by
  unfold clamp127
  split_ifs <;> omega

/-- The pooled KV row address. -/
def kvRow (page kvh ps r : Nat) : Nat := (page * 2 + kvh) * ps + r

/-- A KV row address is inside a pool of `P` pages (two kv heads per page). -/
theorem kvRow_lt (P page kvh ps r : Nat) (hp : page < P) (hk : kvh < 2) (hr : r < ps) :
    kvRow page kvh ps r < P * 2 * ps := by
  unfold kvRow
  have h1 : page * 2 + kvh + 1 ≤ P * 2 := by omega
  have h2 := Nat.mul_le_mul_right ps h1
  have h3 : (page * 2 + kvh) * ps + r < (page * 2 + kvh) * ps + ps := Nat.add_lt_add_left hr _
  have h4 : (page * 2 + kvh + 1) * ps = (page * 2 + kvh) * ps + ps := Nat.succ_mul _ _
  calc (page * 2 + kvh) * ps + r < (page * 2 + kvh) * ps + ps := h3
    _ = (page * 2 + kvh + 1) * ps := h4.symm
    _ ≤ P * 2 * ps := h2

/-- `kv_append` launches 8 groups of 64 values per head: `z>>1` ranges over 4 groups covering 256. -/
theorem kv_group_cover (z t : Nat) (hz : z < 8) (ht : t < 64) : (z / 2) * 64 + t < 256 := by omega

-- § GDN state layout

/-- Conv history size of one GDN layer: three past inputs per channel. -/
def convFloats : Nat := C * 3
/-- State matrix floats: `S * HV * S`. -/
def stateFloats : Nat := S * HV * S

/-- Floats per GDN layer for the concrete model. -/
theorem gdn_floats_value : stateFloats + convFloats = 786432 + 30720 := by decide

end Strata.Prefill.Kernels
