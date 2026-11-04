/-!
# MODULE: iq_kernels_vecdot_layout
IQ-family (and Q3_K / Q2_0) matrix-vector kernels over Q8_1 activations, dequantizers, the Q8_1 quantizer and
the grouped native-expert launcher. This module captures the index arithmetic: per-format iteration traits,
sign unpacking parity, row byte sizes, the gate/up row split and the expert blob layout/scratch carving.

## Source semantics
* A row of `n` values is `n / qk` blocks; the warp walks `k = lane, lane+32, ...` over `nb * ipb` dot calls with
  `kbx = k / ipb`, `iqs = step * (k % ipb)`.
* `unpack_ksigns v = (v ^ (parity(v) << 7)) * 0x01010101`: the 8-bit sign byte has even parity.
* Gate/up rows `0 .. 2*n_ff - 1`: rows `>= n_ff` are up rows with row index `row - n_ff`.
* Expert blob: `[gate | up | down]`, `up_off = n_ff * gu_row`, `down_off = 2 * up_off`.
* Scratch regions are rounded up to 256 bytes.

## Form template identifiers
- `Fmt<TY>` (`qk`, `ipb`, `step`) → `format_traits`
- `row_dot`                       → `warp_row_dot`
- `native_gu_kernel`              → `grouped_gate_up_kernel`
- `native_down_kernel`            → `grouped_down_kernel`
- `native_expert_layout`          → `expert_blob_layout`
-/

namespace Strata.IqKernels

-- § sign unpacking

/-- Population count of the low 8 bits. -/
def popc8 (v : Nat) : Nat := (List.range 8).foldl (fun a i => a + ((v >>> i) &&& 1)) 0

/-- `unpack_ksigns` sign byte for a 7-bit key. -/
def signByte (v : Nat) : Nat := v ^^^ ((popc8 v % 2) <<< 7)

/-- Every 7-bit key expands to an 8-bit sign byte of even parity (the 8th sign is implied). -/
theorem signByte_even_parity : ∀ v : Nat, v < 128 → popc8 (signByte v) % 2 = 0 := by
  decide

/-- The sign byte stays a byte. -/
theorem signByte_lt : ∀ v : Nat, v < 128 → signByte v < 256 := by
  decide

-- § iteration traits

/-- Dot calls of a `(qk, ipb, step)` format over a warp-strided loop: `kbx`, `iqs` of call `k`. -/
def callBlock (ipb k : Nat) : Nat := k / ipb
/-- The `iqs` of call `k`. -/
def callIqs (ipb step k : Nat) : Nat := step * (k % ipb)

/-- Calls stay inside the row: `kbx < nb` for `k < nb * ipb`. -/
theorem call_block_lt (ipb nb k : Nat) (hi : 0 < ipb) (hk : k < nb * ipb) : callBlock ipb k < nb := by
  unfold callBlock
  exact (Nat.div_lt_iff_lt_mul hi).mpr hk

/-- `iqs` stays below `step * ipb`. -/
theorem call_iqs_lt (ipb step k : Nat) (hi : 0 < ipb) (hs : 0 < step) :
    callIqs ipb step k < step * ipb := by
  unfold callIqs
  exact Nat.mul_lt_mul_of_pos_left (Nat.mod_lt _ hi) hs

/-- Every call index decomposes as block times `ipb` plus part. -/
theorem call_decompose (ipb k : Nat) : k = callBlock ipb k * ipb + k % ipb := by
  unfold callBlock
  rw [Nat.mul_comm]
  exact (Nat.div_add_mod k ipb).symm

/-- Lane-strided traversal: call `k` is handled by lane `k % 32` at iteration `k / 32`. -/
theorem lane_stride_cover (k : Nat) : k = k % 32 + 32 * (k / 32) := by omega

/-- Format traits table `(type, qk, ipb, step)` transcribed from `Fmt<TY>`. -/
def fmtTraits : List (Nat × Nat × Nat × Nat) :=
  [(16, 256, 8, 2), (17, 256, 8, 2), (18, 256, 8, 2), (20, 32, 2, 2),
   (21, 256, 8, 2), (22, 256, 8, 2), (29, 256, 8, 1), (42, 64, 2, 1)]

/-- The `iqs` of a call times the vector width never exceeds the block's 32-value chunks: `step * ipb` is
`qk / 32 * 8`-bounded, here checked per format as `step * ipb <= 16`. -/
theorem fmt_iqs_bound : ∀ x ∈ fmtTraits, x.2.2.2 * x.2.2.1 ≤ 16 := by decide

/-- Block sizes `qk` are multiples of 32 so Q8_1 chunks (32 values) tile every block exactly. -/
theorem fmt_qk_multiple_of_32 : ∀ x ∈ fmtTraits, x.2.1 % 32 = 0 := by decide

-- § row bytes

/-- `iq_row_bytes` for a format with `qk` values and `bytes` per block. -/
def rowBytes (qk bytes n : Nat) : Nat := (n / qk) * bytes

/-- For a whole number of blocks the row size is exactly `blocks * bytes`. -/
theorem rowBytes_blocks (qk bytes a : Nat) (hq : 0 < qk) : rowBytes qk bytes (a * qk) = a * bytes := by
  unfold rowBytes
  rw [Nat.mul_div_cancel _ hq]

/-- Row sizes are additive over whole blocks. -/
theorem rowBytes_add (qk bytes a b : Nat) (hq : 0 < qk) :
    rowBytes qk bytes ((a + b) * qk) = rowBytes qk bytes (a * qk) + rowBytes qk bytes (b * qk) := by
  rw [rowBytes_blocks _ _ _ hq, rowBytes_blocks _ _ _ hq, rowBytes_blocks _ _ _ hq, Nat.add_mul]

/-- Supported type ids. -/
def isIq (t : Nat) : Bool :=
  t == 16 || t == 17 || t == 18 || t == 20 || t == 21 || t == 22 || t == 23 || t == 29 || t == 42 || t == 11

/-- The dispatchable (MMVQ) formats are a subset of the supported ones. -/
theorem mmvq_formats_supported : ∀ x ∈ fmtTraits, isIq x.1 = true := by decide

-- § Q8_1 quantizer indexing

/-- Element `i` lives in block `i / 32` at slot `i % 32`, and the pair recomposes `i`. -/
theorem q8_1_index (i : Nat) : i = 32 * (i / 32) + i % 32 := by omega

/-- Columns of `n_in = 32 q` elements are block-aligned: column `c` element `j < n_in` is in block
`c * q + j / 32` of the flat activation image. -/
theorem q8_1_column_block (q c j : Nat) :
    (32 * (c * q) + j) / 32 = c * q + j / 32 := by omega

-- § grouped native experts

/-- Gate/up row split: `row < 2 n_ff`; up rows have in-range row index. -/
theorem gu_row_split (n_ff row : Nat) (h : row < 2 * n_ff) :
    (if row ≥ n_ff then row - n_ff else row) < n_ff := by
  split <;> omega

/-- The gate/up grid covers all `2 * n_ff` rows with `GU_ROWS = 8` warps per block. -/
theorem gu_grid_covers (n_ff row : Nat) (h : row < 2 * n_ff) :
    row / 8 < (2 * n_ff + 8 - 1) / 8 := by omega

/-- Expert blob regions: `[0, up_off)` gate, `[up_off, down_off)` up, then down; sizes add to `bytes`. -/
theorem blob_layout (n_ff gu_row n_embd d_row : Nat) :
    let up_off := n_ff * gu_row
    let down_off := 2 * up_off
    let bytes := down_off + n_embd * d_row
    up_off ≤ down_off ∧ down_off ≤ bytes ∧ down_off = 2 * (n_ff * gu_row) := by
  intro up_off down_off bytes
  refine ⟨?_, ?_, rfl⟩ <;> simp only [up_off, down_off, bytes] <;> omega

/-- The up matrix is exactly `n_ff` rows after the gate matrix. -/
theorem up_row_offset (n_ff gu_row r : Nat) :
    n_ff * gu_row + r * gu_row = (n_ff + r) * gu_row := by
  rw [Nat.add_mul]

/-- Scratch regions are rounded up to 256 bytes: never smaller, always aligned. -/
theorem align256_ok (f : Nat) : f ≤ (f + 255) / 256 * 256 ∧ (f + 255) / 256 * 256 % 256 = 0 := by
  omega

/-- `native_expert_scratch_bytes`: gate, up, h each padded, then the Q8_1 image; offsets are 256-aligned. -/
theorem scratch_offsets_aligned (f g : Nat) :
    let fa := (f + 255) / 256 * 256
    let ga := (g + 255) / 256 * 256
    (3 * fa) % 256 = 0 ∧ fa ≤ 2 * fa ∧ 3 * fa + ga ≥ 3 * f + g := by
  intro fa ga
  simp only [fa, ga]
  omega

/-- Down rows: `r = blockIdx.x * 8 + warp`; covering `n_embd` rows needs `(n_embd + 7) / 8` blocks. -/
theorem down_grid_covers (n_embd r : Nat) (h : r < n_embd) : r / 8 < (n_embd + 7) / 8 := by omega

/-- SwiGLU entry kernel guards `i < n`; grid of 256-thread blocks covers `n`. -/
theorem elementwise_grid_covers (n i : Nat) (h : i < n) : i / 256 < (n + 255) / 256 := by omega

end Strata.IqKernels
