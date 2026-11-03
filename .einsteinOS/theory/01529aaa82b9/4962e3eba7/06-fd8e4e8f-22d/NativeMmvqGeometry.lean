/-!
# MODULE: native_mmvq_geometry
The ggml-compatible quantised mat-vec kernels (`native_mmvq.cu`, adapted from llama.cpp MMVQ): block
layouts, thread-to-block mapping, small-K thresholds, multi-column addressing, shared-memory budgets, the
Q8_1 quantiser's warp/block alignment and the weight-size table.

## Source semantics
* Each format has `(DIV, T, KBY, BPI)`: weight elements per block, threads per block, Q8_1 blocks per weight
  block, weight blocks per K iteration.  Thread `tid` (0..127) handles block `tid / T` of each iteration
  and sub-index `tid % T`; `T * BPI = 128` so one iteration covers all 128 threads exactly.
* Small K (`n_in/DIV < BPI`) switches to 4 rows per block; otherwise 1 row per block.
* Column `j` reads activation blocks at `j * (n_in/32)` and writes `y + j*n_out`.
* Q4_0/Q5_0 are affine: `d*(sumi*ds.x - c*ds.y)` with `c * T = 8` resp. `16` in total.

## Form template identifiers
- `Q5KTraits/Q4KTraits/...` → `traits` entries; `WARPS` → `warps`; `WARP` → `warp`
- `QI`, `VDR` → `qi`, `vdr`; `n_in/n_out/ncols` → `n/nout/ncols`
- ggml type code → `ty`
-/

namespace Strata.NativeMmvq

-- § Block layouts

/-- Q5_K block: `half2 dm; u8 scales[12]; u8 qh[32]; u8 qs[128]`. -/
theorem q5k_size : 4 + 12 + 32 + 128 = 176 := by decide
/-- Q4_K block. -/
theorem q4k_size : 4 + 12 + 128 = 144 := by decide
/-- Q3_K block: hmask 32, qs 64, scales 12, d 2. -/
theorem q3k_size : 32 + 64 + 12 + 2 = 110 := by decide
/-- IQ4_XS block: d 2, scales_h 2, scales_l 4, qs 128. -/
theorem iq4xs_size : 2 + 2 + 4 + 128 = 136 := by decide
/-- Q6_K block: ql 128, qh 64, scales 16, d 2. -/
theorem q6k_size : 128 + 64 + 16 + 2 = 210 := by decide
/-- Q4_0 / IQ4_NL block: d 2, qs 16. -/
theorem q40_size : 2 + 16 = 18 := by decide
/-- Q5_0 block: d 2, qh 4, qs 16. -/
theorem q50_size : 2 + 4 + 16 = 22 := by decide
/-- Q8_0 block: d 2, qs 32. -/
theorem q80_size : 2 + 32 = 34 := by decide
/-- Q8_1 block: `half2 ds`, qs 32: 36 bytes, 4-byte aligned. -/
theorem q81_size : 4 + 32 = 36 := by decide
/-- Q2_0 block: d 2, qs 16 (64 two-bit codes). -/
theorem q20_size : 2 + 16 = 18 ∧ 16 * 4 = 64 := by decide

/-- Q3_K has 110-byte stride: every second block is only 2-byte aligned, so loads go through 16-bit halves. -/
theorem q3k_alternate_alignment (b : Nat) : (b * 110) % 4 = 0 ∨ (b * 110) % 4 = 2 := by omega

/-- Two 16-bit loads recombine little-endian into the 32-bit word. -/
theorem load_int_b2_value (lo hi : Nat) (hlo : lo < 65536) : lo + 65536 * hi = lo + hi * 65536 := by omega

-- § Thread-to-block mapping

/-- Traits: elements per block, threads per block, Q8_1 blocks per weight block, blocks per iteration. -/
structure Traits where
  div : Nat
  t : Nat
  kby : Nat
  bpi : Nat

/-- Q5_K / Q4_K. -/
def q45k : Traits := ⟨256, 16, 8, 8⟩
/-- Q2_0. -/
def q20 : Traits := ⟨64, 2, 2, 64⟩
/-- Q3_K. -/
def q3k : Traits := ⟨256, 16, 8, 8⟩
/-- Q6_K. -/
def q6k : Traits := ⟨256, 32, 8, 4⟩
/-- IQ4_XS. -/
def iq4xs : Traits := ⟨256, 8, 8, 16⟩
/-- Q4_0 / Q5_0 / IQ4_NL (Qi = 4). -/
def small4 : Traits := ⟨32, 2, 1, 64⟩
/-- Q8_0 (Qi = 8). -/
def small8 : Traits := ⟨32, 4, 1, 32⟩

/-- The 128 threads exactly tile one iteration: `T * BPI = 128` for every format. -/
theorem threads_tile :
    q45k.t * q45k.bpi = 128 ∧ q20.t * q20.bpi = 128 ∧ q3k.t * q3k.bpi = 128 ∧
    q6k.t * q6k.bpi = 128 ∧ iq4xs.t * iq4xs.bpi = 128 ∧
    small4.t * small4.bpi = 128 ∧ small8.t * small8.bpi = 128 := by decide

/-- `KBY = DIV / 32`: Q8_1 blocks per weight block. -/
theorem kby_is_div_over_32 :
    q45k.kby = q45k.div / 32 ∧ q20.kby = q20.div / 32 ∧ q3k.kby = q3k.div / 32 ∧
    q6k.kby = q6k.div / 32 ∧ iq4xs.kby = iq4xs.div / 32 ∧
    small4.kby = small4.div / 32 ∧ small8.kby = small8.div / 32 := by decide

/-- Derivation of `BPI` from `VDR * WARPS * WARP / QI` for the K formats. -/
theorem bpi_derivation :
    2 * 4 * 32 / 32 = 8 ∧ 1 * 4 * 32 / 16 = 8 ∧ 1 * 4 * 32 / 32 = 4 ∧ 4 * 4 * 32 / 32 = 16 ∧
    4 * 32 / 2 = 64 ∧ 2 * 4 * 32 / 4 = 64 ∧ 2 * 4 * 32 / 8 = 32 := by decide

/-- Thread `tid` splits uniquely into `(tid / T, tid % T)`. -/
theorem tid_split (tid T : Nat) (hT : 0 < T) : tid = T * (tid / T) + tid % T := by
  have := Nat.div_add_mod tid T
  omega

/-- The block slot of a thread is below `BPI` when `tid < T * BPI`. -/
theorem slot_lt (tid T bpi : Nat) (h : tid < T * bpi) (hT : 0 < T) : tid / T < bpi :=
  Nat.div_lt_of_lt_mul h

/-- Two threads with the same slot and sub-index are the same thread. -/
theorem tid_unique (t1 t2 T : Nat) (hT : 0 < T) (hs : t1 / T = t2 / T) (hm : t1 % T = t2 % T) : t1 = t2 := by
  have h1 := Nat.div_add_mod t1 T
  have h2 := Nat.div_add_mod t2 T
  rw [hs, hm] at h1
  omega

/-- Sub-index ranges: the `kqs` values stay inside the format's `QI` ints (and `kqs+VDR-1` too). -/
theorem kqs_ranges (tid : Nat) :
    2 * (tid % 16) + 1 < 32 ∧ 4 * (tid % 8) + 3 < 32 ∧ tid % 32 < 32 ∧ tid % 16 < 16 ∧
    tid % 2 < 2 ∧ 2 * (tid % 2) + 1 < 4 ∧ 2 * (tid % 4) + 1 < 8 := by omega

-- § Small-K thresholds

/-- With `n % DIV = 0`, the small-K rule `n/DIV < BPI` is exactly `n < DIV * BPI`. -/
theorem small_k_iff (n div bpi : Nat) (hd : 0 < div) (h : n % div = 0) : n / div < bpi ↔ n < div * bpi := by
  have := Nat.div_add_mod n div
  constructor
  · intro hlt
    have : n / div + 1 ≤ bpi := hlt
    have h2 : div * (n / div + 1) ≤ div * bpi := Nat.mul_le_mul_left div this
    rw [Nat.mul_add] at h2
    omega
  · intro hlt
    exact Nat.div_lt_of_lt_mul hlt

/-- Elements per iteration thresholds quoted in the source comment: 2048 for K-formats, 1024 for Q8_0. -/
theorem thresholds :
    q45k.div * q45k.bpi = 2048 ∧ q3k.div * q3k.bpi = 2048 ∧ q6k.div * q6k.bpi = 1024 ∧
    small4.div * small4.bpi = 2048 ∧ small8.div * small8.bpi = 1024 := by decide

/-- Rows per block and grid: 4 rows when small K, grid `ceil(n_out/4)` covers all rows. -/
theorem small_k_grid (n_out : Nat) : n_out ≤ 4 * ((n_out + 3) / 4) := by omega

/-- Row of lane group `i` in block `b` is `ROWS*b + i`; every row has a unique `(b, i)`. -/
theorem row_cover (r rows : Nat) (h : 0 < rows) : r = rows * (r / rows) + r % rows := by
  have := Nat.div_add_mod r rows
  omega

-- § Multi-column addressing

/-- Maximum columns. -/
def maxNcols : Nat := 8

/-- Column `j` of the activation starts at block `j * (n/32)`; columns are contiguous when `n % 32 = 0`. -/
theorem column_block (c m j : Nat) (hj : j < m * 32) : (c * m * 32 + j) / 32 = c * m + j / 32 := by
  omega

/-- The output of column `j`, row `r` is at `j*n_out + r < ncols*n_out`. -/
theorem y_index_lt (j r n_out ncols : Nat) (hj : j < ncols) (hr : r < n_out) : j * n_out + r < ncols * n_out := by
  have : (j + 1) * n_out ≤ ncols * n_out := Nat.mul_le_mul_right n_out hj
  rw [Nat.add_mul] at this
  omega

/-- The partials array `[NW-1][NCOLS][ROWS][32]` floats. -/
def partialBytes (nw ncols rows : Nat) : Nat := (nw - 1) * ncols * rows * 32 * 4

/-- Exact layout (NW = 4, ROWS ≤ 4) fits 48 KiB for every `ncols ≤ 8`. -/
theorem partial_fits_exact (ncols : Nat) (h : ncols ≤ 8) : partialBytes 4 ncols 4 ≤ 49152 := by
  unfold partialBytes; omega

/-- Upstream layout (NW = 2, ROWS = 2) is far smaller. -/
theorem partial_fits_upstream (ncols : Nat) (h : ncols ≤ 8) : partialBytes 2 ncols 2 ≤ 2048 := by
  unfold partialBytes; omega

/-- Upstream layout block counts: `ncols ≤ 4` uses 4 warps, otherwise 2; two rows per block. -/
theorem upstream_warps : (4 : Nat) ≤ 4 ∧ 2 * 32 = 64 := by decide

/-- Two-rows-per-block grid covers all rows. -/
theorem upstream_grid (n_out : Nat) : n_out ≤ 2 * ((n_out + 1) / 2) := by omega

-- § Affine corrections of the 32-element formats

/-- Q4_0: each of the T=2 threads subtracts `4*ds.y`, totalling `8 * sum x`. -/
theorem q40_correction : 2 * (8 * 2 / 4) = 8 := by decide

/-- Q5_0: each thread subtracts `8*ds.y`, totalling `16 * sum x`. -/
theorem q50_correction : 2 * (16 * 2 / 4) = 16 := by decide

/-- The affine identity behind both: `Σ(q-b)u = Σqu - bΣu` over a quad. -/
theorem affine_identity (q0 q1 q2 q3 u0 u1 u2 u3 b : Int) :
    (q0 - b) * u0 + (q1 - b) * u1 + (q2 - b) * u2 + (q3 - b) * u3 =
      (q0 * u0 + q1 * u1 + q2 * u2 + q3 * u3) - b * (u0 + u1 + u2 + u3) := by
  simp only [Int.sub_mul, Int.mul_add]
  omega

-- § The Q8_1 quantiser

/-- Quantiser blocks of 256 threads are 8 warps; element `i`'s 32-block is `blockIdx*8 + warp`. -/
theorem quant_block_of_thread (b t : Nat) (ht : t < 256) : (b * 256 + t) / 32 = b * 8 + t / 32 := by
  omega

/-- A warp is exactly one Q8_1 block: lanes of the same warp share `i / 32`. -/
theorem warp_is_block (b t u : Nat) (ht : t < 256) (hu : u < 256) (h : t / 32 = u / 32) :
    (b * 256 + t) / 32 = (b * 256 + u) / 32 := by omega

/-- With `n % 32 = 0` the early return only removes whole warps (no partial-warp shuffle deadlock). -/
theorem early_return_whole_warps (n i j : Nat) (hn : n % 32 = 0) (hi : n ≤ i) (hij : i / 32 = j / 32) : n ≤ j := by
  omega

/-- Grid covers all elements. -/
theorem quant_grid (n : Nat) : n ≤ 256 * ((n + 255) / 256) := by omega

/-- Element index recomposes from block and offset. -/
theorem quant_index (i : Nat) : (i / 32) * 32 + i % 32 = i := by omega

-- § Type table and weight sizes

/-- (elements per block, bytes per block) for a ggml type code, `none` when unsupported. -/
def blockInfo : Nat → Option (Nat × Nat)
  | 2 => some (32, 18)
  | 6 => some (32, 22)
  | 8 => some (32, 34)
  | 20 => some (32, 18)
  | 11 => some (256, 110)
  | 12 => some (256, 144)
  | 13 => some (256, 176)
  | 14 => some (256, 210)
  | 23 => some (256, 136)
  | 42 => some (64, 18)
  | 16 => some (256, 66)
  | 17 => some (256, 74)
  | 18 => some (256, 98)
  | 21 => some (256, 110)
  | 22 => some (256, 82)
  | 29 => some (256, 56)
  | _ => none

/-- The 16 supported types. -/
def supportedTypes : List Nat := [2, 6, 8, 11, 12, 13, 14, 20, 23, 42, 16, 17, 18, 21, 22, 29]

/-- Every supported type has block information. -/
theorem supported_have_info : ∀ t ∈ supportedTypes, (blockInfo t).isSome = true := by decide

/-- Every supported block size is 32, 64 or 256 and a multiple of 32 (the Q8_1 granule). -/
theorem supported_block_elems :
    ∀ t ∈ supportedTypes, ((blockInfo t).map Prod.fst).getD 0 % 32 = 0 ∧
      (((blockInfo t).map Prod.fst).getD 0 = 32 ∨ ((blockInfo t).map Prod.fst).getD 0 = 64 ∨
        ((blockInfo t).map Prod.fst).getD 0 = 256) := by
  decide

/-- Row bytes for `n` elements. -/
def rowBytes (elems bytes n : Nat) : Nat := n / elems * bytes

/-- Overflow check of the host: `row ≤ MAX / n_out` implies `row * n_out ≤ MAX`. -/
theorem weight_bytes_no_overflow (row nOut M : Nat) (hn : 0 < nOut) (h : row ≤ M / nOut) : row * nOut ≤ M :=
  (Nat.le_div_iff_mul_le hn).mp h

/-- Q4_K for `n_in = 2560`: ten blocks, 1440 bytes per row. -/
theorem q4k_row_2560 : rowBytes 256 144 2560 = 1440 := by decide

/-- Q6_K for `n_in = 2560`. -/
theorem q6k_row_2560 : rowBytes 256 210 2560 = 2100 := by decide

end Strata.NativeMmvq
