/-!
# MODULE: gguf_header_reader
SOURCES: include/strata/artifact/gguf_reader.hpp
GGUF v3 header reader: bounds-checked cursor, tensor-directory arithmetic, data-section alignment,
array sampling and the architecture guard for the `qwen4exp` model.

## Source semantics
* Every cursor read checks `pos + n <= size`, otherwise it throws; a successful read advances by exactly `n`.
* `data_start = ceil(pos / align) * align` and must not exceed the file size.
* Tensors have 1..4 dimensions; element count is the product of the shape.
* Arrays keep at most 64 sampled items; counts above 2^24 are rejected.
* The guard accepts only `general.architecture == "qwen4exp"` with the six compiled-in constants.

## Form template identifiers
- `Cursor::need/read` -> `checked_cursor`; `data_start_` -> `aligned_data_start`;
  `check_architecture` -> `architecture_guard`; `block_geometry` -> `type_geometry`.
-/

namespace Strata.GgufHeaderReader

-- § cursor

/-- A checked read of `n` bytes at `pos` in a buffer of `size` bytes: the new position, or failure. -/
def cursorRead (pos size n : Nat) : Option Nat :=
  if pos + n ≤ size then some (pos + n) else none

/-- A successful read advances by exactly `n` bytes. -/
theorem cursor_advances (pos size n p : Nat) (h : cursorRead pos size n = some p) : p = pos + n := by
  unfold cursorRead at h
  split at h
  · injection h with h'
    omega
  · cases h

/-- A successful read never moves past the end of the buffer. -/
theorem cursor_in_bounds (pos size n p : Nat) (h : cursorRead pos size n = some p) : p ≤ size := by
  unfold cursorRead at h
  split at h
  · injection h with h'
    omega
  · cases h

/-- A read that would overrun fails instead of reading out of bounds. -/
theorem cursor_overrun_fails (pos size n : Nat) (h : size < pos + n) : cursorRead pos size n = none := by
  unfold cursorRead
  have : ¬ (pos + n ≤ size) := by omega
  simp [this]

-- § data section alignment

/-- First aligned offset at or after `pos`. -/
def dataStart (pos align : Nat) : Nat := (pos + align - 1) / align * align

/-- The data section never starts before the header ends. -/
theorem dataStart_ge (pos a : Nat) (ha : 0 < a) : pos ≤ dataStart pos a := by
  unfold dataStart
  have h1 := Nat.div_add_mod (pos + a - 1) a
  have h2 := Nat.mod_lt (pos + a - 1) ha
  rw [Nat.mul_comm ((pos + a - 1) / a) a]
  omega

/-- The data section starts less than one alignment unit after the header end. -/
theorem dataStart_lt (pos a : Nat) (ha : 0 < a) : dataStart pos a < pos + a := by
  unfold dataStart
  have h1 := Nat.div_add_mod (pos + a - 1) a
  rw [Nat.mul_comm ((pos + a - 1) / a) a]
  omega

/-- The data section start is a multiple of the alignment. -/
theorem dataStart_aligned (pos a : Nat) : dataStart pos a % a = 0 := by
  unfold dataStart
  simp

/-- An already-aligned header end is a fixed point. -/
theorem dataStart_idempotent (k a : Nat) (ha : 0 < a) : dataStart (k * a) a = k * a := by
  have hq : (k * a + a - 1) / a = k := by
    apply Nat.div_eq_of_lt_le
    · omega
    · rw [Nat.add_mul]
      omega
  unfold dataStart
  rw [hq]

-- § tensor directory

/-- Element count of a shape (product of its dimensions). -/
def elements : List Nat → Nat
  | [] => 1
  | d :: r => d * elements r

/-- The element count of a concatenated shape multiplies. -/
theorem elements_append (a b : List Nat) : elements (a ++ b) = elements a * elements b := by
  induction a with
  | nil => simp [elements]
  | cons d r ih => simp [elements, ih, Nat.mul_assoc]

/-- A tensor must have between 1 and 4 dimensions. -/
def validDims (nd : Nat) : Bool := nd != 0 && nd ≤ 4

/-- The dimension check accepts exactly 1 through 4. -/
theorem validDims_iff (nd : Nat) : validDims nd = true ↔ (1 ≤ nd ∧ nd ≤ 4) := by
  unfold validDims
  simp only [Bool.and_eq_true, bne_iff_ne, ne_eq, decide_eq_true_eq]
  omega

-- § arrays and versions

/-- Sampled items kept from an array of `count` elements. -/
def sampleKept (count : Nat) : Nat := min count 64

/-- The sample never exceeds 64 items. -/
theorem sample_le_64 (count : Nat) : sampleKept count ≤ 64 := by
  unfold sampleKept
  exact Nat.min_le_right _ _

/-- The sample never exceeds the actual count. -/
theorem sample_le_count (count : Nat) : sampleKept count ≤ count := by
  unfold sampleKept
  exact Nat.min_le_left _ _

/-- Array lengths are accepted only up to 2^24. -/
def arrayLenOk (count : Nat) : Bool := count ≤ 16777216

/-- A rejected array length is above 2^24. -/
theorem array_len_reject (count : Nat) (h : arrayLenOk count = false) : 16777216 < count := by
  unfold arrayLenOk at h
  simp at h
  omega

/-- Only GGUF version 3 is handled. -/
def versionOk (v : Nat) : Bool := v == 3

/-- The version check accepts exactly 3. -/
theorem versionOk_iff (v : Nat) : versionOk v = true ↔ v = 3 := by
  simp [versionOk]

-- § architecture guard

/-- The six compiled-in constants of the qwen4exp guard. -/
structure Guard where
  blockCount : Nat := 48
  hidden : Nat := 2560
  experts : Nat := 512
  expertsUsed : Nat := 10
  headCount : Nat := 24
  headCountKv : Nat := 2
  deriving DecidableEq

/-- Metadata values read from the file for those six keys. -/
abbrev Meta := Guard

/-- The guard accepts only `qwen4exp` with all six values equal. -/
def archOk (arch : String) (m want : Guard) : Bool := arch == "qwen4exp" && decide (m = want)

/-- A different architecture string is always refused. -/
theorem arch_refuse_other (arch : String) (m want : Guard) (h : arch ≠ "qwen4exp") :
    archOk arch m want = false := by
  simp [archOk, h]

/-- A matching qwen4exp file is accepted. -/
theorem arch_accept (m : Guard) : archOk "qwen4exp" m m = true := by
  simp [archOk]

/-- A qwen4exp file with any differing constant is refused. -/
theorem arch_refuse_mismatch (m want : Guard) (h : m ≠ want) : archOk "qwen4exp" m want = false := by
  simp [archOk, h]

/-- The compiled-in constants are self-consistent: top-k experts fit in the expert set. -/
theorem guard_used_le_experts : (Guard.expertsUsed {}) ≤ (Guard.experts {}) := by
  decide

/-- The attention heads group evenly over the KV heads (GQA ratio 12). -/
theorem guard_gqa : (Guard.headCount {}) = 12 * (Guard.headCountKv {}) := by
  decide

-- § block geometry by ggml type id

/-- (elements, bytes) per block for the ggml type ids handled by the reader. -/
def blockGeometry : Nat → Option (Nat × Nat)
  | 0 => some (1, 4)
  | 1 => some (1, 2)
  | 30 => some (1, 2)
  | 2 => some (32, 18)
  | 3 => some (32, 20)
  | 6 => some (32, 22)
  | 7 => some (32, 24)
  | 8 => some (32, 34)
  | 9 => some (32, 36)
  | 10 => some (256, 84)
  | 11 => some (256, 110)
  | 12 => some (256, 144)
  | 13 => some (256, 176)
  | 14 => some (256, 210)
  | 16 => some (256, 66)
  | 17 => some (256, 74)
  | 18 => some (256, 98)
  | 20 => some (32, 18)
  | 21 => some (256, 110)
  | 22 => some (256, 82)
  | 23 => some (256, 136)
  | 29 => some (256, 56)
  | 42 => some (64, 18)
  | _ => none

/-- Type 42 (Q2_0) is 64 elements in 18 bytes. -/
theorem q2_0_block : blockGeometry 42 = some (64, 18) := by
  rfl

/-- Q2_0 stores 2.25 bits per weight: 18 bytes * 8 / 64 elements = 9/4. -/
theorem q2_0_bits_per_weight : 18 * 8 * 4 = 64 * 9 := by
  decide

/-- The k-quant types all use 256-element super-blocks. -/
theorem kquant_blocks : ∀ t ∈ [10, 11, 12, 13, 14, 23],
    (blockGeometry t).map Prod.fst = some 256 := by
  decide

/-- Unknown type ids have no geometry, so the engine cannot size them. -/
theorem unknown_type_none : blockGeometry 99 = none := by
  rfl

end Strata.GgufHeaderReader
