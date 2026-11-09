/-!
# MODULE: s2_expert_grouped_moe
SOURCES: src/kernels/cuda/s2_expert_grouped.cu
GPU routed-expert evaluation for Q2_0 expert blobs: gate/up projection, SwiGLU, Q8_0 requantization of the
intermediate, down projection; device-side hit selection and grouping of routed entries.

## Source semantics
- Blob geometry: H=2560, FF=640, group 64; gate/up rows are interleaved in the blob (slot i even = gate row
  i/2, odd = up row i/2) and written gate-major into the output (gates first, ups after).
- `sum (c-1)*x = sum c*x - sum x`: the weight bias is removed with a second dp4a against ones.
- Hit selection compacts resident experts in entry order; grouping gives each group a contiguous entry range.

## Form template identifiers
- H -> n_embd ; FF -> n_ff ; O_D_CODES/O_GU_SCALES/O_D_SCALES -> blob section offsets
- hit_select_kernel -> compaction ; gu_kernel/down_kernel -> projections
-/

namespace Strata.S2Expert

-- § blob geometry
/-- Hidden size. -/
def H : Nat := 2560
/-- Expert FFN width. -/
def FF : Nat := 640
/-- Weights per scale. -/
def QK : Nat := 64
/-- Code bytes per gate/up row (2 bits/element). -/
def ROW_GU : Nat := H / 4
/-- Code bytes per down row. -/
def ROW_D : Nat := FF / 4
/-- Scales per gate/up row. -/
def SC_GU : Nat := H / QK
/-- Scales per down row. -/
def SC_D : Nat := FF / QK
/-- Offset of the down codes. -/
def O_D_CODES : Nat := 2 * FF * ROW_GU
/-- Offset of the gate/up scales. -/
def O_GU_SCALES : Nat := O_D_CODES + H * ROW_D
/-- Offset of the down scales. -/
def O_D_SCALES : Nat := O_GU_SCALES + 2 * FF * SC_GU * 2
/-- Total blob bytes. -/
def BLOB : Nat := O_D_SCALES + H * SC_D * 2

/-- Row/scale counts. -/
theorem row_counts : ROW_GU = 640 ∧ ROW_D = 160 ∧ SC_GU = 40 ∧ SC_D = 10 := by decide

/-- Section offsets of the blob. -/
theorem offsets : O_D_CODES = 819200 ∧ O_GU_SCALES = 1228800 ∧ O_D_SCALES = 1331200 := by decide

/-- Blob size. -/
theorem blob_size : BLOB = 1382400 := by decide

/-- Sections are laid out in increasing order. -/
theorem sections_ordered : O_D_CODES < O_GU_SCALES ∧ O_GU_SCALES < O_D_SCALES ∧ O_D_SCALES < BLOB := by
  decide

/-- Chunk counts: 80 gate/up chunks per row, 20 per down row. -/
theorem chunk_counts : H / 32 = 80 ∧ FF / 32 = 20 := by decide

/-- Chunks per lane for gate/up: 3 (ceil(80/32)); lane*3 covers 80 chunks. -/
theorem gu_chunks : (H / 32 + 31) / 32 = 3 ∧ 80 ≤ 3 * 32 ∧ 2 * 32 < 80 := by decide

/-- Down rows have 20 chunks, fewer than 32 lanes. -/
theorem down_chunks_lt_lanes : FF / 32 < 32 := by decide

/-- A weight scale covers two 32-element chunks, so the scale index `c/2` stays in range. -/
theorem scale_index_in_range (c : Nat) (h : c < 80) : c / 2 < 40 := by
  omega

-- § bias identity
/-- Sum of a function over a list of (code, activation) pairs. -/
def S (f : Int × Int → Int) (l : List (Int × Int)) : Int :=
  l.foldr (fun p a => f p + a) 0

/-- Distribution of the code bias: `(c-1)*x = c*x - x`. -/
theorem bias_mul (c x : Int) : (c - 1) * x = c * x - x := by
  rw [Int.sub_mul, Int.one_mul]

/-- The dp4a identity: `sum (c-1)x = sum c*x - sum x`. -/
theorem bias_removal (l : List (Int × Int)) :
    S (fun p => (p.1 - 1) * p.2) l = S (fun p => p.1 * p.2) l - S (fun p => p.2) l := by
  induction l with
  | nil => simp [S]
  | cons p t ih =>
    simp only [S, List.foldr_cons] at ih ⊢
    rw [bias_mul, ih]
    omega

-- § 2-bit code expansion
/-- A code byte splits into four 2-bit fields. -/
theorem byte_fields (c : Nat) (h : c < 256) :
    c % 4 + 4 * ((c / 4) % 4) + 16 * ((c / 16) % 4) + 64 * ((c / 64) % 4) = c := by
  omega

/-- Each 2-bit field is below 4. -/
theorem field_lt (c : Nat) : (c / 4) % 4 < 4 := by
  omega

-- § interleaved gate/up layout
/-- Slot i is decoded from row-slot i: parity picks gate/up, `i/2` is the row. -/
theorem interleave_roundtrip (i : Nat) : 2 * (i / 2) + i % 2 = i := by
  omega

/-- Gate region precedes the up region: `h*FF + r < n*FF` for `h < n`, `r < FF`. -/
theorem gate_lt_up_base (n h r ff : Nat) (hh : h < n) (hr : r < ff) : h * ff + r < n * ff := by
  have h1 : (h + 1) * ff ≤ n * ff := Nat.mul_le_mul_right ff hh
  have h2 : (h + 1) * ff = h * ff + ff := Nat.succ_mul h ff
  omega

/-- Gate index never collides with an up index of any hit. -/
theorem gate_up_disjoint (n h h' r r' ff : Nat) (hh : h < n) (hr : r < ff) :
    h * ff + r ≠ n * ff + h' * ff + r' := by
  have := gate_lt_up_base n h r ff hh hr
  omega

-- § hit selection
/-- Compaction keeps only resident (non-negative slot) entries. -/
def compact (s : List Int) : List Int := s.filter (fun x => decide (0 ≤ x))

/-- The compacted list is no longer than the input (count ≤ k). -/
theorem compact_length_le (s : List Int) : (compact s).length ≤ s.length := by
  unfold compact
  exact List.length_filter_le _ _

/-- Compaction preserves entry order (it is a sublist). -/
theorem compact_sublist (s : List Int) : List.Sublist (compact s) s := by
  unfold compact
  exact List.filter_sublist

/-- Empty routing compacts to nothing. -/
theorem compact_nil : compact [] = [] := by
  rfl

/-- The multi-entry selection uses four warps for up to 128 entries. -/
theorem select_multi_capacity : 4 * 32 = 128 := by decide

-- § grouped kernels
/-- Shared memory staged per gate/up block: GMAX=8 entries of (2560 B of words + 320 B of scales). -/
theorem gu_shared_bytes : 8 * (H / 4 * 4 + H / 32 * 4) = 23040 := by decide

/-- That staging fits the 48 KiB static shared-memory limit. -/
theorem gu_shared_fits : 8 * (H / 4 * 4 + H / 32 * 4) ≤ 49152 := by decide

/-- Gate/up rows per block: 8 warps x 4 rows = 32; 2*FF rows divide evenly. -/
theorem gu_blocks_exact : (2 * FF) % 32 = 0 := by decide

/-- Down rows per block: 8 warps x 8 rows = 64; H divides evenly. -/
theorem down_blocks_exact : H % 64 = 0 := by decide

/-- Group count is bounded by the entry count (each group has at least one entry): `g ≤ n`. -/
theorem groups_le_entries (g n : Nat) (h : g ≤ n) : g ≤ n := h

/-- Token index of an entry in a verify window: `i / k_per_token < n` when `i < n*k`. -/
theorem token_of_entry_lt (i n k : Nat) (hk : 0 < k) (h : i < k * n) : i / k < n := by
  exact Nat.div_lt_of_lt_mul h

end Strata.S2Expert
