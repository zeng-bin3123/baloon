/-!
# MODULE: native_quantized_mat_vec
SOURCES: src/kernels/cuda/native_mmvq.cu
Integer model of the pinned llama.cpp MMVQ path: block geometry of every supported
GGML weight format, the Q8_1 activation contract, the DP4A integer dot products and the
thread-to-block iteration map of the 1-column and 2..8-column kernels.

## Source semantics
* Every weight format is a fixed-size block (`static_assert` sizes) covering 32, 64 or 256
  elements; `native_mmvq_weight_bytes` is `(n_in / elems) * bytes * n_out`, rejecting shapes
  that are not whole blocks.
* A dot product accumulates DP4A integer products exactly (integer, order-free) and only
  then applies float scales; affine formats (Q4_0, Q5_0, Q2_0, Q4_K/Q5_K mins) fold their
  offset into the activation sum carried by Q8_1 (`ds.y`).
* A K-loop thread handles block `tid / T + i * BPI`; every block of a row is visited exactly
  once.  Small-K (`blocks_per_row < BPI`) switches to 4 rows per CUDA block.
* Multi-column: column `j` reads activations at `j * (n_in/32)` and writes `y[j*n_out+row]`;
  the outputs of distinct (column,row) pairs never collide.

## Form template identifiers
Q5KBlock/Q4KBlock/... → `blockSpec`; `__dp4a` → `dp4a`; `q5_q8_dot`/`small_q8_dot` → `dot`;
`BLOCKS_PER_ITER` → `bpi`; `validate_shape` → `validShape`; `native_q8_1_bytes` → `q81Bytes`;
`iq4nl_values` → `iq4nl`; `__byte_perm(0x020100ff,…)` → `q2Decode`; `aux` scale unpack → `k4Scale`.
-/

namespace NativeQuantizedMatVec

set_option maxRecDepth 8000

-- § integer dot products

/-- Exact integer dot product of two code lists (DP4A semantics, no rounding). -/
def dot : List Int → List Int → Int
  | a :: as, b :: bs => a * b + dot as bs
  | _, _ => 0

/-- Sum of a list of integers. -/
def lsum : List Int → Int
  | [] => 0
  | a :: as => a + lsum as

/-- `__dp4a(a, b, c)`: four signed byte products added to an accumulator. -/
def dp4a (a b : List Int) (c : Int) : Int := c + dot a b

/-- Chained DP4A calls are exact integer sums, so the call order cannot change the result. -/
theorem dp4a_chain (a b a' b' : List Int) (c : Int) :
    dp4a a b (dp4a a' b' c) = dp4a a' b' (dp4a a b c) := by
  unfold dp4a; omega

/-- A DP4A against `0x01010101` (all-ones) is the plain sum of the other operand. -/
theorem dot_ones (x : List Int) : dot (x.map fun _ => 1) x = lsum x := by
  induction x with
  | nil => rfl
  | cons a as ih =>
    simp only [List.map, dot, lsum]
    rw [ih, Int.one_mul]

/-- Affine offset folding: `Σ (q - k) x = Σ q x - k Σ x`.  This is why Q4_0/Q5_0/Q2_0 can
    subtract a multiple of the Q8_1 block sum instead of re-centering every code. -/
theorem dot_offset (k : Int) : ∀ (c x : List Int), c.length = x.length →
    dot (c.map (· - k)) x = dot c x - k * lsum x
  | [], [], _ => by simp [dot, lsum]
  | a :: as, b :: bs, h => by
    have ih := dot_offset k as bs (by simpa using h)
    simp only [List.map, dot, lsum]
    rw [ih, Int.sub_mul, Int.mul_add]
    omega
  | [], _ :: _, h => by simp at h
  | _ :: _, [], h => by simp at h

-- § block geometry

/-- (elements per block, bytes per block) for each GGML type the native path decodes itself. -/
def blockSpec : Nat → Option (Nat × Nat)
  | 2 => some (32, 18)    -- Q4_0  : d + 16 nibble bytes
  | 6 => some (32, 22)    -- Q5_0  : d + qh[4] + qs[16]
  | 8 => some (32, 34)    -- Q8_0  : d + 32 int8
  | 20 => some (32, 18)   -- IQ4_NL
  | 11 => some (256, 110) -- Q3_K
  | 12 => some (256, 144) -- Q4_K
  | 13 => some (256, 176) -- Q5_K
  | 14 => some (256, 210) -- Q6_K
  | 23 => some (256, 136) -- IQ4_XS
  | 42 => some (64, 18)   -- Q2_0
  | _ => none

/-- The byte sizes are exactly the sums of the struct fields asserted by `static_assert`. -/
theorem block_field_sums :
    (4 + 12 + 32 + 128 = 176) ∧ (4 + 12 + 128 = 144) ∧ (128 + 64 + 16 + 2 = 210) ∧
    (32 + 64 + 12 + 2 = 110) ∧ (2 + 2 + 4 + 128 = 136) ∧ (2 + 16 = 18) ∧
    (2 + 4 + 16 = 22) ∧ (2 + 32 = 34) ∧ (4 + 32 = 36) := by decide

/-- Density check: every format stores at least 2 bits and at most 9 bits per element
    (`8 * bytes` lies between `2 * elems` and `9 * elems`). -/
def specDensityOk (t : Nat) : Bool :=
  match blockSpec t with
  | some (e, b) => 2 * e ≤ 8 * b && 8 * b ≤ 9 * e
  | none => true

/-- Checked over every type id below 64. -/
theorem bits_per_element_bounds : ∀ t, t < 64 → specDensityOk t = true := by decide

/-- The 16 types accepted by `native_mmvq_supported`. -/
def supported (t : Nat) : Bool :=
  t == 2 || t == 6 || t == 8 || t == 11 || t == 12 || t == 13 || t == 14 || t == 20 ||
  t == 23 || t == 42 || t == 16 || t == 17 || t == 18 || t == 21 || t == 22 || t == 29

/-- The IQ types delegated to `iq_mmvq`. -/
def delegatedIQ (t : Nat) : Bool :=
  t == 16 || t == 17 || t == 18 || t == 21 || t == 22 || t == 29

/-- A type is supported exactly when it has a native block spec or is delegated to `iq_mmvq`;
    the dispatcher `native_mmvq` therefore never reaches its `default:` throw for a supported type. -/
theorem supported_iff : ∀ t, t < 64 →
    (supported t = true ↔ ((blockSpec t).isSome = true ∨ delegatedIQ t = true)) := by decide

/-- `validate_shape`: positive, whole blocks, and 1..8 columns. -/
def validShape (nIn ncols elems : Nat) : Prop :=
  0 < nIn ∧ nIn % elems = 0 ∧ 1 ≤ ncols ∧ ncols ≤ 8

/-- A valid shape has at least one whole block per row. -/
theorem validShape_has_block {nIn ncols elems : Nat} (he : 0 < elems)
    (h : validShape nIn ncols elems) : 1 ≤ nIn / elems := by
  obtain ⟨hpos, hdiv, _, _⟩ := h
  have hle : elems ≤ nIn := Nat.le_of_dvd hpos (Nat.dvd_of_mod_eq_zero hdiv)
  exact (Nat.le_div_iff_mul_le he).mpr (by simpa using hle)

/-- Weight byte count of `native_mmvq_weight_bytes` (before the overflow check). -/
def weightBytes (elems bytes nIn nOut : Nat) : Nat := (nIn / elems) * bytes * nOut

/-- Weight bytes are exactly the number of blocks times the block size. -/
theorem weightBytes_blocks (elems bytes k nOut : Nat) (he : 0 < elems) :
    weightBytes elems bytes (k * elems) nOut = k * nOut * bytes := by
  unfold weightBytes
  rw [Nat.mul_div_cancel _ he, Nat.mul_right_comm]

/-- `native_q8_1_bytes`: one 36-byte Q8_1 block per 32 activations per column. -/
def q81Bytes (nIn ncols : Nat) : Nat := ncols * (nIn / 32) * 36

/-- Quantizing `ncols` contiguous columns as one vector of `ncols * nIn` values keeps every
    32-element block inside one column when `32 ∣ nIn`: the flat block count equals the
    per-column count times `ncols`. -/
theorem q81_flat_blocks (k ncols : Nat) : (ncols * (k * 32)) / 32 = ncols * ((k * 32) / 32) := by
  rw [← Nat.mul_assoc, Nat.mul_div_cancel _ (by decide), Nat.mul_div_cancel _ (by decide)]

-- § iteration geometry

/-- Blocks per K iteration: `VDR * WARPS * WARP / QI` (4 warps of 32). -/
def bpi (vdr qi : Nat) : Nat := vdr * 4 * 32 / qi

/-- Pinned per-format blocks per iteration: Q5_K/Q4_K 8, Q2_0 64, Q3_K 8, IQ4_XS 16,
    Q6_K 4, Q4_0/Q5_0/IQ4_NL 64, Q8_0 32. -/
theorem bpi_table :
    bpi 2 32 = 8 ∧ bpi 1 2 = 64 ∧ bpi 1 16 = 8 ∧ bpi 4 32 = 16 ∧ bpi 1 32 = 4 ∧
    bpi 2 4 = 64 ∧ bpi 2 8 = 32 := by decide

/-- The small-K switch `n_in / 32 < BPI` is exactly the 2048-element (QI=4) and
    1024-element (QI=8) thresholds quoted in the source. -/
theorem smallK_thresholds (nIn : Nat) :
    (nIn / 32 < bpi 2 4 ↔ nIn < 2048) ∧ (nIn / 32 < bpi 2 8 ↔ nIn < 1024) := by
  simp only [bpi]; constructor <;> omega

/-- For Q5_K/Q4_K the small-K switch fires below 2048 inputs (8 super-blocks of 256). -/
theorem smallK_k_formats (nIn : Nat) : nIn / 256 < bpi 2 32 ↔ nIn < 2048 := by
  simp only [bpi]; omega

/-- Thread map of the K formats: `tid ↦ (tid / 16, 2 * (tid % 16))` is injective on the
    128 threads and lands in 8 blocks × even quant offsets `< 32`. -/
theorem kmap_injective (t1 t2 : Nat) (_h1 : t1 < 128) (_h2 : t2 < 128)
    (h : t1 / 16 = t2 / 16 ∧ 2 * (t1 % 16) = 2 * (t2 % 16)) : t1 = t2 := by omega

/-- Range of the K-format thread map. -/
theorem kmap_range (t : Nat) (h : t < 128) : t / 16 < 8 ∧ 2 * (t % 16) < 32 := by omega

/-- Strided coverage: block `b` is visited by thread-group `b % BPI` at iteration `b / BPI`,
    so the loop `kbx = start; kbx < n; kbx += BPI` visits every block exactly once. -/
theorem strided_cover (b p : Nat) : b % p + p * (b / p) = b := Nat.mod_add_div b p

/-- Two (start, iteration) pairs reaching the same block are the same pair. -/
theorem strided_unique (p s1 s2 i1 i2 : Nat) (h1 : s1 < p) (h2 : s2 < p)
    (h : s1 + p * i1 = s2 + p * i2) : s1 = s2 ∧ i1 = i2 := by
  have hp : 0 < p := by omega
  have e1 : (s1 + p * i1) % p = s1 := by rw [Nat.add_mul_mod_self_left, Nat.mod_eq_of_lt h1]
  have e2 : (s2 + p * i2) % p = s2 := by rw [Nat.add_mul_mod_self_left, Nat.mod_eq_of_lt h2]
  have hs : s1 = s2 := by rw [← e1, ← e2, h]
  subst hs
  refine ⟨rfl, ?_⟩
  have : p * i1 = p * i2 := by omega
  exact Nat.eq_of_mul_eq_mul_left hp this

/-- Multi-column outputs `y[j * n_out + row]` never collide for rows `< n_out`. -/
theorem output_index_injective {n j1 j2 r1 r2 : Nat} (h1 : r1 < n) (h2 : r2 < n)
    (h : j1 * n + r1 = j2 * n + r2) : j1 = j2 ∧ r1 = r2 := by
  have h' : r1 + n * j1 = r2 + n * j2 := by
    rw [Nat.mul_comm n j1, Nat.mul_comm n j2]; omega
  obtain ⟨hr, hj⟩ := strided_unique n r1 r2 j1 j2 h1 h2 h'
  exact ⟨hj, hr⟩

/-- Small-K launches `⌈n_out/4⌉` CUDA blocks of 4 rows: every row has a block. -/
theorem smallK_grid_covers (nOut r : Nat) (h : r < nOut) : r / 4 < (nOut + 3) / 4 := by omega

-- § code decoding

/-- Byte `i` of a little-endian 32-bit word. -/
def byteOf (x i : Nat) : Nat := (x >>> (8 * i)) % 256

/-- Reinterpret a byte as `int8_t`. -/
def toInt8 (b : Nat) : Int := if b ≥ 128 then (b : Int) - 256 else b

/-- `__byte_perm(0x020100ff, 0x020100ff, q)` maps a Q2_0 code `c ∈ 0..3` to `c - 1`:
    the selector constant *is* the codebook `{-1, 0, 1, 2}`. -/
theorem q2Decode : ∀ c, c < 4 → toInt8 (byteOf 0x020100ff c) = (c : Int) - 1 := by decide

/-- The pinned nonlinear IQ4 codebook. -/
def iq4nl : List Int := [-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113]

/-- The IQ4 codebook is strictly increasing and every entry fits in `int8_t` (no DP4A overflow). -/
theorem iq4nl_sorted_int8 :
    iq4nl.length = 16 ∧ (∀ i, i < 15 → iq4nl[i]! < iq4nl[i+1]!) ∧
    (∀ i, i < 16 → (-127 : Int) ≤ iq4nl[i]! ∧ iq4nl[i]! ≤ (127 : Int)) := by decide

/-- K-quant 6-bit scale/min unpack (`aux`): low groups take `s & 0x3f`, high groups take the
    nibble of `s4` plus the two top bits of `s0` moved to bits 4..5. -/
def k4Scale (j s0 s4 : Nat) : Nat :=
  if j < 2 then s0 % 64 else s4 % 16 + (s0 / 64) * 16

/-- Every unpacked K-quant scale or min is a 6-bit value. -/
theorem k4Scale_lt (j s0 s4 : Nat) (h0 : s0 < 256) : k4Scale j s0 s4 < 64 := by
  unfold k4Scale; split <;> omega

/-- The bit expression `(s0 & 0xc0) >> 2` equals the arithmetic `(s0 / 64) * 16` on bytes. -/
theorem k4Scale_bits : ∀ s, s < 256 → (s &&& 0xc0) >>> 2 = (s / 64) * 16 := by decide

/-- The high-group mask `uint32_t(-int32_t(j >= 2))` is all-zeros or all-ones. -/
theorem hi_mask (b : Bool) :
    ((-(if b then (1 : Int) else 0)) % 2^32) = (if b then 0xFFFFFFFF else 0) := by
  cases b <;> decide

/-- IQ4_XS sub-block scale `ls = low4 | high2 << 4` is 6-bit, so `ls - 32 ∈ [-32, 31]`. -/
theorem iq4xs_scale_range (lo hi : Nat) :
    (-32 : Int) ≤ ((lo % 16 + (hi % 4) * 16 : Nat) : Int) - 32 ∧
    ((lo % 16 + (hi % 4) * 16 : Nat) : Int) - 32 ≤ 31 := by omega

/-- Q3_K scale `(low4 | high2 << 4) - 32` and Q6_K code `(low4 | high2 << 4) - 32` share the
    range `[-32, 31]`: both are signed 6-bit values. -/
theorem q3_q6_signed6 (lo hi : Nat) (hl : lo < 16) (hh : hi < 4) :
    (-32 : Int) ≤ ((lo + hi * 16 : Nat) : Int) - 32 ∧ ((lo + hi * 16 : Nat) : Int) - 32 ≤ 31 := by
  omega

-- § affine corrections

/-- Threads per 32-element block for the small formats is `Qi / VDR`: Q4_0/Q5_0/IQ4_NL use 2,
    Q8_0 uses 4. -/
theorem small_threads_per_block : 4 / 2 = 2 ∧ 8 / 2 = 4 := by decide

/-- Each Q4_0 call covers half a block and subtracts `4 · ds.y`; the two calls of a block
    together subtract `8 · Σx`, the zero point of the 4-bit code (`q - 8`).  Q5_0 likewise
    totals `16 · Σx`. -/
theorem small_affine_totals : 2 * 4 = 8 ∧ 2 * 8 = 16 := by decide

/-- Q4_0 whole-block identity: with codes centered at 8, the integer dot equals the raw code
    dot minus `8 · Σx`, which is what the summed per-call corrections reproduce. -/
theorem q4_0_block (c x : List Int) (h : c.length = x.length) :
    dot (c.map (· - 8)) x = dot c x - 8 * lsum x := dot_offset 8 c x h

end NativeQuantizedMatVec
