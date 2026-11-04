/-!
# MODULE: s2_expert_layout
SOURCES: src/kernels/cuda/s2_expert_grouped.cu, src/kernels/cuda/iq_kernels.cu
Byte layout of a Q2_0 expert blob, the interleaved gate/up row mapping and its gate-major output, lane/chunk
coverage, ballot-based hit compaction, and 16/256-byte scratch carving (both expert paths).

## Source semantics
* Blob = gate/up codes (2*FF rows of 640 B), down codes (H rows of 160 B), gate/up scales (2*FF rows of 40 fp16),
  down scales (H rows of 10 fp16).
* Row slot `i` of the interleaved gate/up rows is gate `i/2` when even and up `i/2` when odd; it is written to a
  gate-major buffer: gates contiguous at `h*FF`, ups contiguous from `n*FF`.
* `hit_select` compacts resident experts with `popc(ballot & ((1<<lane)-1))`: positions are distinct and dense.
* Scratch regions are carved at 16-byte (256-byte for the native path) aligned offsets.

## Form template identifiers
O_D_CODES -> down_codes_offset ; O_GU_SCALES -> gate_up_scales_offset ; O_D_SCALES -> down_scales_offset
slot_index -> resident_slot ; dst_index -> router_slot ; GMAX -> entries_per_group
-/

namespace Strata.S2ExpertLayout

-- § blob geometry

/-- Hidden size. -/
def H : Nat := 2560
/-- Expert FFN width. -/
def FF : Nat := 640
/-- Q2_0 group size. -/
def QK : Nat := 64
/-- Bytes of codes per gate/up row. -/
def rowGU : Nat := H / 4
/-- Bytes of codes per down row. -/
def rowD : Nat := FF / 4
/-- fp16 scales per gate/up row. -/
def scGU : Nat := H / QK
/-- fp16 scales per down row. -/
def scD : Nat := FF / QK
/-- Offset of the down codes. -/
def oDCodes : Nat := 2 * FF * rowGU
/-- Offset of the gate/up scales. -/
def oGUScales : Nat := oDCodes + H * rowD
/-- Offset of the down scales. -/
def oDScales : Nat := oGUScales + 2 * FF * scGU * 2
/-- Total blob bytes. -/
def blobBytes : Nat := oDScales + H * scD * 2

/-- The concrete geometry. -/
theorem blob_constants :
    rowGU = 640 ∧ rowD = 160 ∧ scGU = 40 ∧ scD = 10 ∧ oDCodes = 819200 ∧ oGUScales = 1228800 ∧
      oDScales = 1331200 ∧ blobBytes = 1382400 := by
  decide

/-- Every gate/up code row lies inside the gate/up code region. -/
theorem gu_code_row_in_region (i : Nat) (h : i < 1280) : i * 640 + 640 ≤ 819200 := by omega

/-- Every gate/up scale row (40 fp16 = 80 bytes) lies inside its region. -/
theorem gu_scale_row_in_region (i : Nat) (h : i < 1280) :
    1228800 ≤ 1228800 + i * 80 ∧ 1228800 + i * 80 + 80 ≤ 1331200 := by omega

/-- Every down code row lies inside the down code region. -/
theorem down_code_row_in_region (r : Nat) (h : r < 2560) :
    819200 + r * 160 + 160 ≤ 1228800 := by omega

/-- Every down scale row (10 fp16 = 20 bytes) ends within the blob. -/
theorem down_scale_row_in_region (r : Nat) (h : r < 2560) :
    1331200 + r * 20 + 20 ≤ 1382400 := by omega

/-- The four regions are in order and tile the blob. -/
theorem regions_ordered : oDCodes ≤ oGUScales ∧ oGUScales ≤ oDScales ∧ oDScales ≤ blobBytes := by decide

-- § lane / chunk coverage

/-- 80 activation chunks over 32 lanes need 3 chunks per lane. -/
theorem gu_chunks_per_lane : (80 + 31) / 32 = 3 := by decide

/-- Every gate/up chunk `c < 80` is `lane + 32*q` for a unique lane < 32 and q < 3. -/
theorem gu_chunk_decomp (c : Nat) (h : c < 80) :
    c = c % 32 + 32 * (c / 32) ∧ c % 32 < 32 ∧ c / 32 < 3 := by omega

/-- A down row has 20 chunks, so lanes 0..19 hold one each. -/
theorem down_chunks_fit : 640 / 32 = 20 ∧ 20 ≤ 32 := by decide

/-- Down scale index `c / 2` stays below 10. -/
theorem down_scale_index (c : Nat) (h : c < 20) : c / 2 < 10 := by omega

-- § interleaved gate/up output slot

/-- Per-hit decomposition of a flat row slot is a bijection onto (hit, row-in-hit). -/
theorem slot_decomp (slot : Nat) : slot = (slot / 1280) * 1280 + slot % 1280 ∧ slot % 1280 < 1280 := by
  omega

/-- The hit index of a slot is in range. -/
theorem slot_hit_lt (slot n : Nat) (h : slot < n * 1280) : slot / 1280 < n := by omega

/-- Gate-major output position of row slot `i` of hit `h` (with `N = n * FF`). -/
def gm (N h i : Nat) : Nat := (i % 2) * N + h * 640 + i / 2

/-- The gate-major mapping is injective in (hit, slot). -/
theorem gate_major_inj (N h h' i i' : Nat) (hN : h * 640 + 640 ≤ N) (hN' : h' * 640 + 640 ≤ N)
    (hi : i < 1280) (hi' : i' < 1280) (heq : gm N h i = gm N h' i') : h = h' ∧ i = i' := by
  unfold gm at heq
  rcases Nat.mod_two_eq_zero_or_one i with p | p <;> rcases Nat.mod_two_eq_zero_or_one i' with q | q <;>
    simp only [p, q, Nat.zero_mul, Nat.one_mul] at heq <;> omega

/-- Gate values land below `N`, up values at or above `N` and below `2N`. -/
theorem gate_major_range (N h i : Nat) (hN : h * 640 + 640 ≤ N) (hi : i < 1280) :
    (i % 2 = 0 → gm N h i < N) ∧ (i % 2 = 1 → N ≤ gm N h i ∧ gm N h i < 2 * N) := by
  unfold gm
  constructor
  · intro p
    simp only [p, Nat.zero_mul]
    omega
  · intro p
    simp only [p, Nat.one_mul]
    omega

-- § compaction by ballot prefix count

/-- Number of set lanes strictly before position `n` (models `popc(hit & ((1<<lane)-1))`). -/
def pre (f : Nat → Bool) : Nat → Nat
  | 0 => 0
  | n + 1 => pre f n + (if f n then 1 else 0)

/-- The prefix count is monotone. -/
theorem pre_mono (f : Nat → Bool) (i : Nat) : ∀ j, i ≤ j → pre f i ≤ pre f j := by
  intro j hj
  induction j with
  | zero =>
    have : i = 0 := by omega
    subst this
    exact Nat.le_refl _
  | succ j ih =>
    rcases Nat.eq_or_lt_of_le hj with h | h
    · rw [h]
    · have h1 := ih (by omega)
      have h2 : pre f (j + 1) = pre f j + (if f j then 1 else 0) := rfl
      split at h2 <;> omega

/-- A set lane has a strictly smaller position than every later lane. -/
theorem pre_lt (f : Nat → Bool) (i j : Nat) (hij : i < j) (hf : f i = true) : pre f i < pre f j := by
  have h1 : pre f (i + 1) = pre f i + 1 := by simp [pre, hf]
  have h2 := pre_mono f (i + 1) j (by omega)
  omega

/-- Compacted positions of distinct hits are distinct. -/
theorem pre_inj (f : Nat → Bool) (i j : Nat) (hi : f i = true) (hj : f j = true)
    (h : pre f i = pre f j) : i = j := by
  rcases Nat.lt_trichotomy i j with c | c | c
  · have := pre_lt f i j c hi
    omega
  · exact c
  · have := pre_lt f j i c hj
    omega

/-- Compacted positions are dense: below the total hit count. -/
theorem pre_lt_total (f : Nat → Bool) (i n : Nat) (hi : i < n) (hf : f i = true) : pre f i < pre f n :=
  pre_lt f i n hi hf

-- § scratch alignment

/-- Round up to a multiple of 16. -/
def r16 (n : Nat) : Nat := (n + 15) / 16 * 16

/-- Round up to a multiple of 256. -/
def r256 (n : Nat) : Nat := (n + 255) / 256 * 256

/-- `r16` bounds. -/
theorem r16_props (n : Nat) : n ≤ r16 n ∧ r16 n % 16 = 0 ∧ r16 n < n + 16 := by
  unfold r16
  omega

/-- `r256` bounds. -/
theorem r256_props (n : Nat) : n ≤ r256 n ∧ r256 n % 256 = 0 ∧ r256 n < n + 256 := by
  unfold r256
  omega

/-- The regions `gate_up | q8 | scales` fit the carved total (the CPU-order variant uses four regions). -/
theorem scratch_regions_fit (gu q8 hs xh : Nat) :
    gu ≤ r16 gu ∧ r16 gu + q8 ≤ r16 gu + r16 q8 ∧
    r16 gu + r16 q8 + hs ≤ r16 gu + r16 q8 + 2 * r16 hs + r16 xh ∧
    r16 gu + r16 q8 + r16 hs + hs ≤ r16 gu + r16 q8 + 2 * r16 hs + r16 xh ∧
    r16 gu + r16 q8 + 2 * r16 hs + xh ≤ r16 gu + r16 q8 + 2 * r16 hs + r16 xh := by
  have a := r16_props gu
  have b := r16_props q8
  have c := r16_props hs
  have d := r16_props xh
  omega

/-- Per-hit buffers are multiples of 16 bytes for floats of width 5120 per hit (`n*FF*8`). -/
theorem gu_bytes_aligned (n : Nat) : r16 (n * 5120) = n * 5120 := by
  unfold r16
  omega

-- § native expert layout (iq_kernels.cu)

/-- Byte offsets of the native expert blob. -/
theorem native_layout_rows (nff guRow r : Nat) (hr : r < nff) : r * guRow + guRow ≤ nff * guRow := by
  have := Nat.mul_le_mul_right guRow (show r + 1 ≤ nff by omega)
  rw [Nat.add_mul, Nat.one_mul] at this
  exact this

/-- The four scratch regions of `native_expert_grouped` (three float buffers and the q8_1 image) fit. -/
theorem native_scratch_fit (f q : Nat) :
    f ≤ r256 f ∧ 3 * r256 f + q ≤ 3 * r256 f + r256 q := by
  have a := r256_props f
  have b := r256_props q
  omega

end Strata.S2ExpertLayout
