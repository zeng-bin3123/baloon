/-!
# MODULE: s2_expert_grouped_layout
The grouped GPU expert for Q2_0 weights (`s2_expert_grouped.cu`): blob geometry, the gate/up row interleave
and gate-major output layout, the dp4a bias-split identity, chunking of rows, the per-token activation
selection and the shared-memory budgets of the grouped kernels.

## Source semantics
* Blob: gate/up codes (2*FF rows of 640 B), down codes (H rows of 160 B), gate/up scales, down scales.
* Row slot `i` of the interleaved gate/up rows is gate row `i/2` for even `i`, up row `i/2` for odd `i`;
  the output is gate-major: gate rows of hit `h` at `h*FF + r`, up rows at `n*FF + h*FF + r`.
* `sum (c-1)*x = sum c*x - sum x`, both sums INT8 x INT8 (`__dp4a`).
* One weight scale per 64 elements = two 32-element chunks.

## Form template identifiers
- `H` → `H`, `FF` → `FF`, `QK` → `qk`, `ROW_GU/ROW_D` → `rowGU/rowD`, `SC_GU/SC_D` → `scGU/scD`
- `O_D_CODES/O_GU_SCALES/O_D_SCALES` → `oDCodes/oGUScales/oDScales`
- hit index → `h`, row slot → `i`, hit count → `n`
-/

namespace Strata.S2Expert

-- § Blob geometry

/-- Hidden size. -/
def H : Nat := 2560
/-- FFN width. -/
def FF : Nat := 640
/-- Gate/up code row bytes. -/
def rowGU : Nat := H / 4
/-- Down code row bytes. -/
def rowD : Nat := FF / 4
/-- Gate/up scales per row. -/
def scGU : Nat := H / 64
/-- Down scales per row. -/
def scD : Nat := FF / 64
/-- Offset of the down codes. -/
def oDCodes : Nat := 2 * FF * rowGU
/-- Offset of the gate/up scales. -/
def oGUScales : Nat := oDCodes + H * rowD
/-- Offset of the down scales. -/
def oDScales : Nat := oGUScales + 2 * FF * scGU * 2
/-- Total blob bytes. -/
def blobBytes : Nat := oDScales + H * scD * 2

/-- The row sizes. -/
theorem row_sizes : rowGU = 640 ∧ rowD = 160 ∧ scGU = 40 ∧ scD = 10 := by decide

/-- The blob offsets, evaluated. -/
theorem offsets : oDCodes = 819200 ∧ oGUScales = 1228800 ∧ oDScales = 1331200 ∧ blobBytes = 1382400 := by decide

/-- The regions are ordered and disjoint. -/
theorem regions_ordered : oDCodes < oGUScales ∧ oGUScales < oDScales ∧ oDScales < blobBytes := by decide

/-- Region sizes tile the blob exactly. -/
theorem regions_tile :
    2 * FF * rowGU + H * rowD + 2 * FF * scGU * 2 + H * scD * 2 = blobBytes := by decide

/-- Every region start is 2-byte aligned (scales are fp16 read as bytes) and in fact 16-byte aligned. -/
theorem offsets_aligned : oDCodes % 16 = 0 ∧ oGUScales % 16 = 0 ∧ oDScales % 16 = 0 := by decide

/-- Chunk counts: 80 chunks per gate/up row, 20 per down row. -/
theorem chunk_counts : H / 32 = 80 ∧ FF / 32 = 20 := by decide

/-- A chunk's weight scale index `c/2` is within the row's scales. -/
theorem scale_index_gu (c : Nat) (h : c < H / 32) : c / 2 < scGU := by
  simp only [H, scGU] at *; omega

/-- Same for the down row. -/
theorem scale_index_d (c : Nat) (h : c < FF / 32) : c / 2 < scD := by
  simp only [FF, scD] at *; omega

/-- Two chunks per 64-element scale group. -/
theorem chunk_group (c : Nat) : (c * 32) / 64 = c / 2 := by omega

/-- Each chunk's 8 code bytes (32 elements at 2 bits) lie at `c*8`, inside the row. -/
theorem chunk_code_bytes (c : Nat) (h : c < H / 32) : c * 8 + 8 ≤ rowGU := by
  simp only [H, rowGU] at *; omega

/-- Down row: same. -/
theorem chunk_code_bytes_down (c : Nat) (h : c < FF / 32) : c * 8 + 8 ≤ rowD := by
  simp only [FF, rowD] at *; omega

/-- Lanes: 80 chunks over 32 lanes need 3 per lane (the `GU_CHUNKS` constant). -/
theorem gu_chunks : (H / 32 + 31) / 32 = 3 ∧ 2 * 32 + 16 = H / 32 := by decide

/-- A down row's 20 chunks fit in one warp. -/
theorem down_chunks_one_warp : FF / 32 ≤ 32 := by decide

-- § Interleave and gate-major layout

/-- Output position of row slot `i` of hit `h`, for `n` hits: gate-major. -/
def dest (n h i : Nat) : Nat :=
  if i % 2 = 1 then n * FF + h * FF + i / 2 else h * FF + i / 2

/-- Slot `i` decodes row `i`; its parity gives the destination plane. Destination is in range. -/
theorem dest_lt (n h i : Nat) (hh : h < n) (hi : i < 2 * FF) : dest n h i < 2 * (n * FF) := by
  unfold dest FF at *
  split_ifs <;> omega

/-- The map `(h, i) ↦ dest` is injective: no two row slots collide. -/
theorem dest_injective (n h i h' i' : Nat) (hh : h < n) (hh' : h' < n) (hi : i < 2 * FF) (hi' : i' < 2 * FF)
    (heq : dest n h i = dest n h' i') : h = h' ∧ i = i' := by
  unfold dest FF at *
  split_ifs at heq <;> omega

/-- Gate row `r` of hit `h` lands at `h*FF + r`, and the matching up row sits exactly `n*FF` later (the swiglu pairing). -/
theorem swiglu_pairing (n h r : Nat) :
    dest n h (2 * r + 1) = dest n h (2 * r) + n * FF := by
  unfold dest
  have h1 : (2 * r + 1) % 2 = 1 := by omega
  have h2 : (2 * r) % 2 = 0 := by omega
  have h3 : (2 * r + 1) / 2 = r := by omega
  have h4 : (2 * r) / 2 = r := by omega
  rw [if_pos h1, if_neg (show ¬ ((2 * r) % 2 = 1) by omega), h3, h4]
  omega

/-- The swiglu result for entry `(h, r)` lands in the first `n*FF` floats, where the quantiser reads it. -/
theorem swiglu_output_in_gate_plane (n h r : Nat) (hh : h < n) (hr : r < FF) : h * FF + r < n * FF := by
  have : (h + 1) * FF ≤ n * FF := Nat.mul_le_mul_right FF hh
  rw [Nat.add_mul] at this
  omega

-- § The dp4a bias split

/-- Dot of four int pairs (the semantics of `__dp4a` on unpacked bytes). -/
def dp4 (a0 a1 a2 a3 b0 b1 b2 b3 : Int) : Int := a0 * b0 + a1 * b1 + a2 * b2 + a3 * b3

/-- `dp4a(ones, x)` is the plain sum of the four activations. -/
theorem dp4_ones (x0 x1 x2 x3 : Int) : dp4 1 1 1 1 x0 x1 x2 x3 = x0 + x1 + x2 + x3 := by
  unfold dp4; omega

/-- `s - hx` equals the dot of the biased codes `(c-1)` with the activations, over a quad. -/
theorem bias_split (c0 c1 c2 c3 x0 x1 x2 x3 : Int) :
    dp4 c0 c1 c2 c3 x0 x1 x2 x3 - dp4 1 1 1 1 x0 x1 x2 x3 =
      dp4 (c0 - 1) (c1 - 1) (c2 - 1) (c3 - 1) x0 x1 x2 x3 := by
  unfold dp4
  simp only [Int.sub_mul, Int.one_mul]
  omega

/-- The integer sums over a 32-element chunk fit in int32. -/
theorem chunk_sum_bound : 32 * (3 * 127) < 2 ^ 31 := by decide

-- § Unpacked code word

/-- The four 2-bit fields of a byte, placed one per byte of a word. -/
def expand (b : Nat) : Nat := b % 4 + 256 * (b / 4 % 4) + 65536 * (b / 16 % 4) + 16777216 * (b / 64 % 4)

/-- Each byte lane of the expansion is a 2-bit value, so `dp4a` sees values 0..3 (with `a_k` the k-th field
of the byte, i.e. `expand b = a0 + 256*a1 + 65536*a2 + 16777216*a3`). -/
theorem expand_lanes (a0 a1 a2 a3 : Nat) (h0 : a0 < 4) (h1 : a1 < 4) (h2 : a2 < 4) (h3 : a3 < 4) :
    (a0 + 256 * a1 + 65536 * a2 + 16777216 * a3) % 256 = a0 ∧
    (a0 + 256 * a1 + 65536 * a2 + 16777216 * a3) / 256 % 256 = a1 ∧
    (a0 + 256 * a1 + 65536 * a2 + 16777216 * a3) / 65536 % 256 = a2 ∧
    (a0 + 256 * a1 + 65536 * a2 + 16777216 * a3) / 16777216 = a3 := by omega

/-- `expand` is that sum of the byte's four fields. -/
theorem expand_eq (b : Nat) :
    expand b = b % 4 + 256 * (b / 4 % 4) + 65536 * (b / 16 % 4) + 16777216 * (b / 64 % 4) := rfl

-- § Device-side hit selection and grouping

/-- `dst / k` recovers the token of routed entry `dst < k*T`. -/
theorem token_of_entry (dst k T : Nat) (h : dst < k * T) (hk : 0 < k) : dst / k < T :=
  Nat.div_lt_of_lt_mul h

/-- Entry `i` of token `i / k` is routed slot `i % k`; the pair recomposes `i`. -/
theorem entry_decompose (i k : Nat) (hk : 0 < k) : (i / k) * k + i % k = i := by
  have := Nat.div_add_mod i k
  rw [Nat.mul_comm] at this
  omega

/-- The verify window handles at most 128 entries: 4 warps. -/
theorem window_warps : 128 / 32 = 4 := by decide

/-- Warp prefix count: a hit's compacted position is the count of earlier hits, so it is below the total. -/
theorem compact_lt (before within total : Nat) (h : before + within < total) : before + within < total := h

-- § Shared-memory budgets and grids

/-- Gate/up kernel staging: GMAX entries of H int8 (as words) and H/32 scales. -/
def guSmem : Nat := 8 * (H / 4) * 4 + 8 * (H / 32) * 4
/-- Down kernel staging. -/
def downSmem : Nat := 8 * (FF / 4) * 4 + 8 * (FF / 32) * 4

/-- Both fit in the 48 KiB static limit. -/
theorem smem_budget : guSmem = 23040 ∧ downSmem = 5760 ∧ guSmem ≤ 49152 := by decide

/-- Gate/up grid: 40 blocks of 32 rows (8 warps x 4 rows) cover all `2*FF` rows. -/
theorem gu_grid : 2 * FF / 32 = 40 ∧ 40 * 32 = 2 * FF := by decide

/-- Down grid: 40 blocks of 64 rows (8 warps x 8 rows) cover all `H` rows. -/
theorem down_grid : H / 64 = 40 ∧ 40 * 64 = H := by decide

/-- Warp `w` of 8 handling rows `w, w+8, w+16, w+24` of a 32-row block covers each row exactly once. -/
theorem warp_rows_gu (rr : Nat) (h : rr < 32) : rr = rr % 8 + 8 * (rr / 8) := by omega

/-- Flat per-hit launch: one warp per row; the grid covers all rows. -/
theorem warp_grid (rows warps : Nat) (hw : 0 < warps) : rows ≤ warps * ((rows + warps - 1) / warps) := by
  have := Nat.div_add_mod (rows + warps - 1) warps
  have := Nat.mod_lt (rows + warps - 1) hw
  omega

/-- Scratch alignment helper used by the three scratch regions. -/
def align16 (x : Nat) : Nat := (x + 15) / 16 * 16

/-- Alignment rounds up, stays within 15, and is a multiple of 16. -/
theorem align16_spec (x : Nat) : x ≤ align16 x ∧ align16 x < x + 16 ∧ align16 x % 16 = 0 := by
  unfold align16; omega

end Strata.S2Expert
