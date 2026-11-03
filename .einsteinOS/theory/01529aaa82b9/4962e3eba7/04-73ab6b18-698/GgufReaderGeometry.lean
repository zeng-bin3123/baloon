/-!
# MODULE: gguf_reader_geometry
GGUF v3 reader model: ggml type tables (names and block geometry), bounds-checked cursor,
data-section alignment, array sampling, architecture guard constants.

## Source semantics
`block_geometry` maps a ggml type id to (elements per block, bytes per block). Q2_0 (42) is 64/18.
The Cursor never reads past the mapped size. The data section starts at the header end rounded up to
`general.alignment` (default 32).

## Form template identifiers
blockGeom -> block_geometry ; rowBytes -> row_bytes ; dataStart -> data_start ; step -> cursor_need
-/

set_option maxRecDepth 100000

namespace Strata.Gguf

/-- Block geometry per ggml type id, transcribed from `block_geometry`. -/
def blockGeom : Nat → Option (Nat × Nat)
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

/-- Type ids for which `ggml_type_name` returns something other than "?". -/
def namedTypes : List Nat :=
  [0, 1, 2, 3, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 30, 34, 35, 39, 40, 41, 42]

/-- Geometry entries are always positive, so no division by zero can occur downstream. -/
def geomOk (t : Nat) : Bool :=
  match blockGeom t with
  | some (e, b) => decide (0 < e) && decide (0 < b)
  | none => true

/-- All geometry entries for type ids below 64 are positive. -/
theorem geom_positive : ∀ t < 64, geomOk t = true := by decide

/-- Q2_0 is 64 elements in 18 bytes. -/
theorem q2_0_geom : blockGeom 42 = some (64, 18) := rfl

/-- Named types without geometry: the reader names them but cannot size them. -/
theorem named_without_geom :
    namedTypes.filter (fun t => (blockGeom t).isNone) = [15, 19, 34, 35, 39, 40, 41] := by decide

/-- IQ1_M (29) has geometry but its name is "?": count_type("IQ1_M") can never match. -/
theorem iq1m_unnamed : (blockGeom 29).isSome = true ∧ (namedTypes.contains 29) = false := by decide

/-- Legacy-block types share geometry where ggml does: Q4_0 and IQ4_NL are both 32/18. -/
theorem q4_0_iq4nl_same : blockGeom 2 = blockGeom 20 := rfl

/-- Q3_K and IQ3_S are both 256/110. -/
theorem q3k_iq3s_same : blockGeom 11 = blockGeom 21 := rfl

/-- Row size in bytes: `n / elems * bytes`. -/
def rowBytes (t n : Nat) : Option Nat :=
  match blockGeom t with
  | some (e, b) => some (n / e * b)
  | none => none

/-- A 160-wide IQ4_NL row (the PLE table) is 90 bytes. -/
theorem ple_row_bytes : rowBytes 20 160 = some 90 := by decide

/-- A 2560-wide Q2_0 row is 720 bytes. -/
theorem q2_row_bytes : rowBytes 42 2560 = some 720 := by decide

/-- Bits per weight (x100): Q2_0 is 2.25, Q4_0 is 4.5. -/
theorem bpw : 18 * 8 * 100 / 64 = 225 ∧ 18 * 8 * 100 / 32 = 450 := ⟨rfl, rfl⟩

-- § Cursor

/-- One bounds-checked advance. -/
def step (pos size n : Nat) : Option Nat :=
  if pos + n ≤ size then some (pos + n) else none

/-- A successful read never leaves the file and advances exactly n. -/
theorem step_ok {pos size n p : Nat} (h : step pos size n = some p) : p = pos + n ∧ p ≤ size := by
  unfold step at h
  split at h
  · cases h; omega
  · cases h

/-- A read past the end is always refused. -/
theorem step_fail {pos size n : Nat} (h : size < pos + n) : step pos size n = none := by
  unfold step
  rw [if_neg (by omega)]

/-- Sequential reads preserve the invariant pos <= size. -/
theorem step_mono {pos size n p : Nat} (h : step pos size n = some p) : pos ≤ p := by
  have := step_ok h; omega

-- § Data section alignment

/-- Start of the data section: header end rounded up to the alignment. -/
def dataStart (pos a : Nat) : Nat := (pos + a - 1) / a * a

/-- The data section never starts before the header ends. -/
theorem ds_ge (pos a : Nat) (ha : 0 < a) : pos ≤ dataStart pos a := by
  unfold dataStart
  have h1 := Nat.div_add_mod (pos + a - 1) a
  have h2 := Nat.mod_lt (pos + a - 1) ha
  rw [Nat.mul_comm ((pos + a - 1) / a) a]
  omega

/-- The rounding adds less than one alignment unit. -/
theorem ds_lt (pos a : Nat) (ha : 0 < a) : dataStart pos a < pos + a := by
  unfold dataStart
  have h1 := Nat.div_add_mod (pos + a - 1) a
  have h2 := Nat.mod_lt (pos + a - 1) ha
  rw [Nat.mul_comm ((pos + a - 1) / a) a]
  omega

/-- The data section is aligned. -/
theorem ds_aligned (pos a : Nat) : dataStart pos a % a = 0 := by
  unfold dataStart
  exact Nat.mul_mod_left _ _

/-- The PLE shard header of 192 bytes is already 32-aligned. -/
theorem header192 : dataStart 192 32 = 192 := by decide

-- § Array sampling and guard constants

/-- Kept sample size of an array metadata value. -/
def keptItems (count : Nat) : Nat := min count 64

/-- The sample is bounded by 64 and by the real count. -/
theorem kept_bounds (c : Nat) : keptItems c ≤ 64 ∧ keptItems c ≤ c := by
  unfold keptItems; omega

/-- The qwen4exp guard constants are mutually consistent: 24 heads over 2 KV heads, group of 12. -/
theorem guard_consts : 24 % 2 = 0 ∧ 24 / 2 = 12 ∧ 10 ≤ 512 ∧ 2560 = 16 * 160 := by decide

end Strata.Gguf
