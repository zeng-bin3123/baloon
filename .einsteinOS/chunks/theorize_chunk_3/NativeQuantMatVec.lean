/-!
# MODULE: native_quant_mat_vec
Block layouts, thread-to-block mappings, grid rules and launch contracts of the native quantized
matrix-vector kernels (Q4_0, Q5_0, Q8_0, IQ4_NL, Q3_K, Q4_K, Q5_K, Q6_K, IQ4_XS, Q2_0) for 1 to 8 activation
columns, plus the q8_1 activation quantiser.

## Source semantics
- Activations are quantised to q8_1 blocks of 32 (36 bytes: half2 scale/sum + 32 int8).
- A kernel block has 4 warps x 32 lanes = 128 threads; thread `tid` handles weight block `tid / T` (+ k*BPI) and
  intra-block index `kqs(tid)`, where `T * BPI = 128` for every format.
- "Small K" (fewer weight blocks per row than BPI) uses 4 rows per block, otherwise 1 row per block; the
  non-exact multi-column layout uses 2 rows per block.
- Weight bytes = `(n_in / block_elems) * block_bytes * n_out`, with an overflow guard.

## Form template identifiers
- QK -> superblock_values, QI/VDR -> int_slots/vector_dot_ratio, BPI -> blocks_per_iter, T -> threads_per_block
- Q81Block -> act_block, native_mmvq -> quant_matvec, ncols -> n_columns
-/

namespace Strata.NativeMmvq

-- § helpers

/-- Row-major index bound. -/
theorem idx_lt {a b A B : Nat} (ha : a < A) (hb : b < B) : a * B + b < A * B := by
  have h1 : (a + 1) * B ≤ A * B := Nat.mul_le_mul_right B (Nat.succ_le_of_lt ha)
  have h2 : (a + 1) * B = a * B + B := by rw [Nat.add_mul, Nat.one_mul]
  omega

-- § struct sizes

/-- Q5_K block: half2 + 12 scale bytes + 32 high bits + 128 low nibbles. -/
theorem q5k_size : 4 + 12 + 32 + 128 = 176 := by decide
/-- Q4_K block: half2 + 12 + 128. -/
theorem q4k_size : 4 + 12 + 128 = 144 := by decide
/-- Q3_K block: 32 hmask + 64 qs + 12 scales + 2. -/
theorem q3k_size : 32 + 64 + 12 + 2 = 110 := by decide
/-- Q6_K block: 128 ql + 64 qh + 16 scales + 2. -/
theorem q6k_size : 128 + 64 + 16 + 2 = 210 := by decide
/-- IQ4_XS block: 2 + 2 + 4 + 128. -/
theorem iq4xs_size : 2 + 2 + 4 + 128 = 136 := by decide
/-- Q4_0 / IQ4_NL / Q2_0 block: half + 16 bytes. -/
theorem q40_size : 2 + 16 = 18 := by decide
/-- Q5_0 block: half + 4 high-bit bytes + 16. -/
theorem q50_size : 2 + 4 + 16 = 22 := by decide
/-- Q8_0 block: half + 32. -/
theorem q80_size : 2 + 32 = 34 := by decide
/-- q8_1 block: half2 + 32. -/
theorem q81_size : 4 + 32 = 36 := by decide

/-- Q4_K / Q5_K qs offsets: Q5_K qs begins after dm(4) + scales(12) + qh(32) = 48. -/
theorem q5k_qs_offset : 4 + 12 + 32 = 48 := by decide

/-- Q2_0 packs 64 two-bit codes into 16 bytes. -/
theorem q20_density : 16 * 4 = 64 := by decide

/-- Q5_0 high-bit masks are single bits 4, 12, 20 and 28. -/
theorem q50_high_masks :
    (0x10 : Nat) = 1 <<< 4 ∧ (0x1000 : Nat) = 1 <<< 12 ∧ (0x100000 : Nat) = 1 <<< 20 ∧
      (0x10000000 : Nat) = 1 <<< 28 :=
  ⟨rfl, rfl, rfl, rfl⟩

-- § thread to block mapping

/-- Threads per block: 4 warps of 32. -/
theorem block_threads : 4 * 32 = 128 := by decide

/-- Format traits (T lanes per weight block, BPI blocks per iteration) as (T, BPI) pairs. -/
def traitPairs : List (Nat × Nat) :=
  [(16, 8), (16, 8), (2, 64), (16, 8), (32, 4), (8, 16), (2, 64), (2, 64), (4, 32), (2, 64)]

/-- Every format covers all 128 threads: T * BPI = 128. -/
theorem trait_pairs_cover : ∀ p ∈ traitPairs, p.1 * p.2 = 128 := by decide

/-- Thread index splits into (weight block lane group, lane within block). -/
theorem thread_split (tid T : Nat) : T * (tid / T) + tid % T = tid :=
  Nat.div_add_mod tid T

/-- The block group index of a thread below `T*BPI` is below BPI. -/
theorem thread_group_lt (tid T BPI : Nat) (h : tid < T * BPI) : tid / T < BPI :=
  Nat.div_lt_of_lt_mul h

/-- The lane within a weight block is below T. -/
theorem thread_lane_lt (tid T : Nat) (hT : 0 < T) : tid % T < T := Nat.mod_lt _ hT

/-- Distinct threads handle distinct (group, lane) pairs when T lanes share a block. -/
theorem thread_pair_inj (tid tid' T : Nat) (hT : 0 < T) (h1 : tid / T = tid' / T)
    (h2 : tid % T = tid' % T) : tid = tid' := by
  have e := Nat.div_add_mod tid T
  have e' := Nat.div_add_mod tid' T
  rw [h1, h2] at e
  omega

/-- Small-K threshold values: BPI formulas VDR*4*32/QI. -/
theorem bpi_values :
    2 * 4 * 32 / 32 = 8 ∧ 4 * 32 / 16 = 8 ∧ 4 * 32 / 2 = 64 ∧ 4 * 4 * 32 / 32 = 16 ∧
      4 * 32 / 32 = 4 := by
  decide

/-- 32-element formats: BPI = 2*128/Qi gives 64 for Qi = 4 and 32 for Qi = 8. -/
theorem small_bpi : 2 * 4 * 32 / 4 = 64 ∧ 2 * 4 * 32 / 8 = 32 := by decide

/-- Small-K element thresholds: 64 blocks of 32 = 2048 and 32 blocks = 1024. -/
theorem small_k_thresholds : 64 * 32 = 2048 ∧ 32 * 32 = 1024 := by decide

-- § rows per block and grids

/-- Four rows per block cover all rows. -/
theorem small_k_grid_covers (n_out : Nat) : n_out ≤ (n_out + 3) / 4 * 4 := by
  omega

/-- Two rows per block (upstream multi-column layout) cover all rows. -/
theorem upstream_grid_covers (n_out : Nat) : n_out ≤ (n_out + 1) / 2 * 2 := by
  omega

/-- Row of (block, i) with ROWS = 4 is below the padded row count. -/
theorem row_in_padded (b i n_out : Nat) (hb : b < (n_out + 3) / 4) (hi : i < 4) :
    4 * b + i < (n_out + 3) / 4 * 4 := by
  omega

/-- Row guards: padded rows above n_out are skipped (`row0 + i < n_out`). -/
theorem guarded_rows (row0 i n_out : Nat) (h : row0 + i < n_out) : row0 + i < n_out := h

/-- Weight block address `row*blocks_per_row + kbx` stays inside the matrix. -/
theorem weight_block_lt (row kbx n_out bpr : Nat) (hr : row < n_out) (hk : kbx < bpr) :
    row * bpr + kbx < n_out * bpr :=
  idx_lt hr hk

/-- Activation block address of column j: `j*(n_in/32) + kby` stays inside `ncols*(n_in/32)`. -/
theorem act_block_lt (j kby ncols stride : Nat) (hj : j < ncols) (hk : kby < stride) :
    j * stride + kby < ncols * stride :=
  idx_lt hj hk

/-- Output address `j*n_out + row` stays inside `ncols*n_out`. -/
theorem out_addr_lt (j row ncols n_out : Nat) (hj : j < ncols) (hr : row < n_out) :
    j * n_out + row < ncols * n_out :=
  idx_lt hj hr

-- § shared-memory partials

/-- Cross-warp partial buffer of the exact layout (3 extra warps, up to 8 columns, 4 rows): bytes. -/
theorem partial_bytes_exact (ncols : Nat) (h : ncols ≤ 8) : 3 * ncols * 4 * 32 * 4 ≤ 49152 := by
  omega

/-- Single-column partial buffer is 3*4*32 floats. -/
theorem partial_bytes_single : 3 * 4 * 32 * 4 = 1536 := by decide

/-- The upstream layout with 5..8 columns, 2 warps, 2 rows needs far less. -/
theorem partial_bytes_upstream : 1 * 8 * 2 * 32 * 4 = 2048 := by decide

-- § q8_1 quantiser

/-- Bytes of `ncols` quantised columns: ncols * (n_in/32) * 36. -/
def q81Bytes (n_in ncols : Nat) : Nat := ncols * (n_in / 32) * 36

/-- Quantised bytes add over columns. -/
theorem q81_bytes_add (n_in a b : Nat) : q81Bytes n_in (a + b) = q81Bytes n_in a + q81Bytes n_in b := by
  unfold q81Bytes
  simp only [Nat.add_mul]

/-- Columns quantise as one long vector: every 32-block stays inside one column. -/
theorem q8_block_in_column (c m k : Nat) (hk : k < 32 * m) :
    (c * (32 * m) + k) / 32 = c * m + k / 32 := by
  have e : c * (32 * m) = 32 * (c * m) := Nat.mul_left_comm c 32 m
  omega

/-- The column of a block is bounded: block index below `(c+1)*m`. -/
theorem q8_block_bound (c m k : Nat) (hk : k < 32 * m) : c * m + k / 32 < (c + 1) * m := by
  have e : (c + 1) * m = c * m + m := by rw [Nat.add_mul, Nat.one_mul]
  omega

/-- Quantiser: element i is block i/32, position i%32. -/
theorem quant_split (i : Nat) : (i / 32) * 32 + i % 32 = i := by
  omega

/-- Only whole warps return early because n_in is a multiple of 32. -/
theorem whole_warps_return (i n : Nat) (hn : n % 32 = 0) (h : n ≤ i) : (i / 32) * 32 ≥ n := by
  omega

/-- Quantiser grid covers all elements with 256 threads per block. -/
theorem quant_grid_covers (n : Nat) : n ≤ (n + 255) / 256 * 256 := by
  omega

-- § weight byte accounting

/-- Total weight bytes. -/
def weightBytes (n_in block_elems block_bytes n_out : Nat) : Nat :=
  (n_in / block_elems) * block_bytes * n_out

/-- Weight bytes split additively over output rows. -/
theorem weight_bytes_add (n_in e b o1 o2 : Nat) :
    weightBytes n_in e b (o1 + o2) = weightBytes n_in e b o1 + weightBytes n_in e b o2 := by
  unfold weightBytes
  simp only [Nat.mul_add]

/-- The overflow guard `row_bytes <= MAX / n_out` implies `row_bytes * n_out <= MAX`. -/
theorem overflow_guard (rb n_out M : Nat) (hn : 0 < n_out) (h : rb ≤ M / n_out) : rb * n_out ≤ M :=
  (Nat.le_div_iff_mul_le hn).mp h

/-- Divisible shapes give exactly `n_in/elems` whole blocks per row. -/
theorem whole_blocks (n_in elems : Nat) (he : 0 < elems) (h : n_in % elems = 0) :
    (n_in / elems) * elems = n_in := by
  have := Nat.div_add_mod n_in elems
  rw [Nat.mul_comm] at this
  omega

-- § dispatch table

/-- Type codes supported by `native_mmvq_supported`. -/
def supportedTypes : List Nat := [2, 6, 8, 11, 12, 13, 14, 20, 23, 42, 16, 17, 18, 21, 22, 29]

/-- Sixteen distinct supported types. -/
theorem supported_nodup : supportedTypes.Nodup ∧ supportedTypes.length = 16 := by decide

/-- Types routed to the IQ kernel set. -/
def iqRoute : List Nat := [16, 17, 18, 21, 22, 29]

/-- Types routed to the native kernels. -/
def nativeRoute : List Nat := [2, 6, 8, 20, 11, 12, 13, 14, 23, 42]

/-- Every supported type has exactly one route. -/
theorem every_type_routed : ∀ t ∈ supportedTypes, (t ∈ iqRoute ∨ t ∈ nativeRoute) ∧
    ¬ (t ∈ iqRoute ∧ t ∈ nativeRoute) := by
  decide

/-- Block geometry table (type, elements, bytes) used by `native_mmvq_weight_bytes`. -/
def geometry : List (Nat × Nat × Nat) :=
  [(2, 32, 18), (6, 32, 22), (8, 32, 34), (20, 32, 18), (11, 256, 110), (12, 256, 144),
   (13, 256, 176), (14, 256, 210), (23, 256, 136), (42, 64, 18)]

/-- Every native-route type has a geometry row. -/
theorem native_types_have_geometry : ∀ t ∈ nativeRoute, ∃ g ∈ geometry, g.1 = t := by
  decide

/-- Geometry rows agree with the struct sizes: Q3_K..Q6_K and IQ4_XS byte counts. -/
theorem geometry_matches_structs :
    (256, 110) ∈ geometry.map (fun g => (g.2.1, g.2.2)) ∧ (256, 144) ∈ geometry.map (fun g => (g.2.1, g.2.2)) ∧
    (256, 176) ∈ geometry.map (fun g => (g.2.1, g.2.2)) ∧ (256, 210) ∈ geometry.map (fun g => (g.2.1, g.2.2)) ∧
    (256, 136) ∈ geometry.map (fun g => (g.2.1, g.2.2)) := by
  decide

-- § launch contract

/-- Shape validation: positive n_in divisible by the block element count, 1..8 columns. -/
def shapeOk (n_in ncols elems : Nat) : Prop := 0 < n_in ∧ n_in % elems = 0 ∧ 1 ≤ ncols ∧ ncols ≤ 8

/-- Valid shapes give an integral number of q8_1 blocks per column when elems is a multiple of 32. -/
theorem shape_ok_q8_blocks (n_in ncols elems : Nat) (h : shapeOk n_in ncols elems)
    (he : elems % 32 = 0) (hpos : 0 < elems) : n_in % 32 = 0 := by
  obtain ⟨_, h2, _, _⟩ := h
  have : ∃ m, elems = 32 * m := ⟨elems / 32, by omega⟩
  obtain ⟨m, hm⟩ := this
  subst hm
  have : n_in % (32 * m) = 0 := h2
  have h3 : 32 ∣ n_in := by
    have : 32 * m ∣ n_in := Nat.dvd_of_mod_eq_zero this
    exact Nat.dvd_trans ⟨m, rfl⟩ this
  exact Nat.mod_eq_zero_of_dvd h3

/-- Pointer validation: 4-byte alignment of an address. -/
def aligned4 (addr : Nat) : Prop := addr % 4 = 0

/-- Offsets that are multiples of 4 preserve alignment. -/
theorem aligned_offset (addr off : Nat) (h : aligned4 addr) (ho : off % 4 = 0) : aligned4 (addr + off) := by
  unfold aligned4 at *
  omega

/-- Multi-column kernels are used only for 2..8 columns. -/
theorem multi_range (ncols : Nat) (h1 : 2 ≤ ncols) (h2 : ncols ≤ 8) : 1 < ncols ∧ ncols ≤ 8 := by
  omega

end Strata.NativeMmvq
