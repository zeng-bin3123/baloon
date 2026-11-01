/-!
# MODULE: iq_quant_dot_kernels
Block geometry, row layout and call-enumeration contracts of the IQ-family quantised matrix-vector kernels
(`mmvq_kernel`, grouped `native_gu_kernel` / `native_down_kernel`), the q8_1 activation quantiser and the dequantisers.

## Source semantics
* Each format `TY` is described by `(qk, ipb, step)`: values per block, dot calls per block, iqs stride.  A warp
  enumerates calls `k = lane, lane+32, ...` below `nb * ipb`, with `kbx = k / ipb` and `iqs = step * (k % ipb)`.
* Every dot call reads the q8_1 activation block(s) `x + kbx * (qk/32) + (iqs/2 or iqs)`; this index must stay inside
  the `qk/32` q8_1 blocks that cover one quant block.
* A row of `n` values occupies `n / qk * sizeof(block)` bytes.  The grouped expert blob is
  `[gate rows | up rows | down rows]` with `up_off = n_ff * gu_row` and `down_off = 2 * up_off`.
* Sign handling uses `__vsub4(x ^ m, m)` with `m = 0xFF` per negated byte, which negates modulo 256.
* The Q2_0 dequantiser maps a 2-bit code to `code - 1`.

## Form template identifiers
* `qk` / `ipb` / `step`       : per-format block width, dot calls per block, iqs stride
* `rowBytes`                  : bytes per weight row
* `alignUp256`                : scratch region rounding
* `up_off` / `down_off`       : expert blob offsets
-/

namespace Strata.IqKernels

-- § call enumeration

/-- Lane-strided enumeration: each call index `k` belongs to exactly one lane and round. -/
theorem call_lane_unique (k : Nat) : k % 32 < 32 ∧ k % 32 + 32 * (k / 32) = k := by omega

/-- Block and part decomposition `k = kbx * ipb + part` with `part < ipb`. -/
theorem call_block_part (k ipb : Nat) (hi : 0 < ipb) :
    k / ipb * ipb + k % ipb = k ∧ k % ipb < ipb := by
  refine ⟨?_, Nat.mod_lt _ hi⟩
  have := Nat.div_add_mod k ipb
  rw [Nat.mul_comm] at this
  exact this

/-- Call block index stays below `nb` when `k < nb * ipb`. -/
theorem call_block_lt (k ipb nb : Nat) (hk : k < nb * ipb) (hi : 0 < ipb) : k / ipb < nb := by
  exact Nat.div_lt_of_lt_mul (by rw [Nat.mul_comm]; exact hk)

-- § per-format q8_1 index ranges (qk = 256, ipb = 8, step = 2: iq2_xxs, iq2_xs, iq3_xxs, iq3_s, iq2_s)

/-- `iqs = 2 * part` with `part < 8` gives `iqs / 2 < 8`: the q8_1 index stays inside the 8 blocks of a 256-block. -/
theorem q8_index_step2_qk256 (part : Nat) (h : part < 8) : (2 * part) / 2 < 256 / 32 := by omega

/-- iq1_m (qk = 256, ipb = 8, step = 1): `bq8_1[iqs]` stays inside the 8 q8_1 blocks. -/
theorem q8_index_iq1_m (part : Nat) (h : part < 8) : 1 * part < 256 / 32 := by omega

/-- iq4_nl (qk = 32, ipb = 2, step = 2): ints `iqs + l + 4` of the 8-int q8_1 block for `l < 2`, `iqs ∈ {0, 2}`. -/
theorem iq4_nl_int_index (part l : Nat) (hp : part < 2) (hl : l < 2) : 2 * part + l + 4 < 8 := by omega

/-- iq4_nl low-nibble ints `iqs + l` stay inside the first 4 ints of the block. -/
theorem iq4_nl_low_index (part l : Nat) (hp : part < 2) (hl : l < 2) : 2 * part + l < 4 := by omega

/-- q2_0 (qk = 64, ipb = 2, step = 1): `bq8_1 + iqs` stays inside the 2 q8_1 blocks of a 64-block. -/
theorem q2_0_q8_index (part : Nat) (h : part < 2) : 1 * part < 64 / 32 := by omega

/-- q2_0 packed index `iqs*4 + j` with `j < 4` stays inside the 8 `int16` of the 16-byte `qs`. -/
theorem q2_0_qs_index (part j : Nat) (hp : part < 2) (hj : j < 4) : part * 4 + j < 8 := by omega

/-- A 64-value q2_0 block packs 2 bits per value into 16 bytes. -/
theorem q2_0_qs_bytes : 64 * 2 / 8 = 16 := by decide

-- § row layout

/-- Block sizes in bytes (from `ggml-common.h`) for the supported types. -/
def blockBytes : Nat → Nat
  | 16 => 66   -- iq2_xxs
  | 17 => 74   -- iq2_xs
  | 18 => 98   -- iq3_xxs
  | 20 => 18   -- iq4_nl
  | 21 => 110  -- iq3_s
  | 22 => 82   -- iq2_s
  | 29 => 56   -- iq1_m
  | 23 => 136  -- iq4_xs
  | 11 => 110  -- q3_K
  | 42 => 18   -- q2_0
  | _ => 0

/-- Values per block of each supported type. -/
def valuesPerBlock : Nat → Nat
  | 20 => 32
  | 42 => 64
  | _ => 256

/-- Bytes in a row of `n` values. -/
def rowBytes (t n : Nat) : Nat := n / valuesPerBlock t * blockBytes t

/-- The production gate/up row size for a 2560-wide model at iq2_xxs: 10 blocks of 66 bytes. -/
theorem rowBytes_iq2_xxs_2560 : rowBytes 16 2560 = 660 := by decide

/-- Unsupported types have zero-sized rows. -/
theorem rowBytes_unsupported (n : Nat) : rowBytes 99 n = n / 256 * 0 := by
  unfold rowBytes blockBytes; rfl

-- § expert blob layout

/-- Offset of the up matrix inside an expert blob. -/
def upOff (n_ff gu_row : Nat) : Nat := n_ff * gu_row
/-- Offset of the down matrix. -/
def downOff (n_ff gu_row : Nat) : Nat := 2 * upOff n_ff gu_row
/-- Total blob size. -/
def blobBytes (n_ff gu_row n_embd d_row : Nat) : Nat := downOff n_ff gu_row + n_embd * d_row

/-- The down region starts where the up region ends. -/
theorem down_after_up (n_ff gu_row : Nat) : downOff n_ff gu_row = upOff n_ff gu_row + n_ff * gu_row := by
  unfold downOff upOff; omega

/-- Gate row `r < n_ff` ends before the up region begins. -/
theorem gate_row_in_region (n_ff gu_row r : Nat) (hr : r < n_ff) : (r + 1) * gu_row ≤ upOff n_ff gu_row := by
  unfold upOff
  exact Nat.mul_le_mul_right gu_row hr

/-- Up row `r < n_ff` ends before the down region begins. -/
theorem up_row_in_region (n_ff gu_row r : Nat) (hr : r < n_ff) :
    upOff n_ff gu_row + (r + 1) * gu_row ≤ downOff n_ff gu_row := by
  have h : (r + 1) * gu_row ≤ n_ff * gu_row := Nat.mul_le_mul_right gu_row hr
  unfold downOff upOff
  omega

/-- The blob has room for the whole down matrix. -/
theorem blob_holds_down (n_ff gu_row n_embd d_row : Nat) :
    downOff n_ff gu_row + n_embd * d_row ≤ blobBytes n_ff gu_row n_embd d_row := by
  unfold blobBytes; omega

/-- Gate/up rows: `row < 2*n_ff` splits into gate (`< n_ff`) or up (`r = row - n_ff`). -/
theorem gu_row_split (n_ff row : Nat) (h : row < 2 * n_ff) (hu : n_ff ≤ row) : row - n_ff < n_ff := by omega

/-- Grouped gate/up grid covers all `2*n_ff` rows with 8 warps per block. -/
theorem gu_grid_cover (n_ff row : Nat) (h : row < 2 * n_ff) : row / 8 < (2 * n_ff + 7) / 8 := by omega

/-- Grouped down grid covers all `n_embd` rows with 8 warps per block. -/
theorem down_grid_cover (n_embd r : Nat) (h : r < n_embd) : r / 8 < (n_embd + 7) / 8 := by omega

/-- mmvq grid covers all `n_out` rows with 4 warps per block. -/
theorem mmvq_grid_cover (n_out r : Nat) (h : r < n_out) : r / 4 < (n_out + 3) / 4 := by omega

-- § scratch alignment

/-- Round up to a multiple of 256 bytes. -/
def alignUp256 (n : Nat) : Nat := (n + 255) / 256 * 256

/-- Rounding never shrinks. -/
theorem alignUp256_ge (n : Nat) : n ≤ alignUp256 n := by unfold alignUp256; omega

/-- Rounded sizes are 256-byte aligned. -/
theorem alignUp256_aligned (n : Nat) : alignUp256 n % 256 = 0 := by unfold alignUp256; omega

/-- Each scratch region starts at a multiple of 256 when all preceding sizes are rounded. -/
theorem scratch_offset_aligned (a b : Nat) : (alignUp256 a + alignUp256 b) % 256 = 0 := by
  unfold alignUp256; omega

-- § q8_1 quantiser geometry

/-- Quantiser: element `i` goes to block `i / 32`, slot `i % 32`. -/
theorem q8_1_slot (i : Nat) : i / 32 * 32 + i % 32 = i ∧ i % 32 < 32 := by omega

/-- The quantiser needs `n` to be a whole number of warps for the full-mask shuffles. -/
theorem q8_1_blocks_cover (n i : Nat) (h : i < n) : i / 256 < (n + 255) / 256 := by omega

-- § sign trick and codebook offsets

/-- Per-byte negation by xor with `0xFF` then subtracting `0xFF` (`+1` mod 256): `x ^ 0xFF = 255 - x`, so the
byte becomes `256 - x`, whose sum with `x` is `0` modulo 256. -/
theorem byte_negate_via_xor (x : Nat) (h : x < 256) : (x + (255 - x + 1)) % 256 = 0 := by omega

/-- Q2_0 codes map to `code - 1`, i.e. into `[-1, 2]`. -/
theorem q2_0_code_value (code : Nat) (h : code < 4) : -1 ≤ (code : Int) - 1 ∧ (code : Int) - 1 ≤ 2 := by omega

/-- Q2_0 superblock: 4 blocks of 64 values, 8 threads per block writing 8 values each = 256 values. -/
theorem q2_0_superblock : 4 * 8 * 8 = 256 ∧ 32 * 8 = 256 := by decide

/-- Dequant thread map `tid → (il, ib) = (tid / 8, tid % 8)` writes `32*ib + 8*il + j` with `j < 8`, inside 256. -/
theorem dq_thread_map_lt (tid j : Nat) (ht : tid < 32) (hj : j < 8) : 32 * (tid % 8) + 8 * (tid / 8) + j < 256 := by
  omega

end Strata.IqKernels
