/-!
# MODULE: gguf_reader
GGUF v3 reader model: block geometry per ggml type id, bounds-checked cursor, data-section alignment, architecture guard.

## Source semantics
* `block_geometry t` gives (elements per block, bytes per block); Q2_0 (id 42) is 64/18.
* The cursor refuses any read that would pass the end of the header region.
* `data_start = ceil(pos / align) * align`, and must not pass EOF.
* The architecture guard accepts only `qwen4exp` with all six compiled-in constants matching.

## Form template identifiers
* block_geometry -> `block_geometry`
* Cursor::need / read -> `cursor_read`
* data_start_ -> `data_start`
* check_architecture -> `arch_guard`
-/

namespace Strata.Gguf

-- § Block geometry

/-- (elements, bytes) per block for each supported ggml type id. -/
def blockGeometry (t : Nat) : Option (Nat × Nat) :=
  match t with
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

/-- Q2_0 is 64 elements in 18 bytes. -/
theorem q2_0_geometry : blockGeometry 42 = some (64, 18) := by decide

/-- IQ4_NL and Q4_0 share geometry. -/
theorem iq4nl_q4_0_same : blockGeometry 20 = blockGeometry 2 := by decide

/-- IQ3_S and Q3_K share geometry. -/
theorem iq3s_q3k_same : blockGeometry 21 = blockGeometry 11 := by decide

/-- Every registered geometry has positive elements and bytes. -/
theorem geometry_positive :
    (List.range 64).all (fun t =>
      match blockGeometry t with
      | some (e, b) => decide (0 < e) && decide (0 < b)
      | none => true) = true := by
  decide

/-- All K-quant ids 10..14 have 256-element super-blocks. -/
theorem kquant_super_blocks :
    ([10, 11, 12, 13, 14].all (fun t =>
      match blockGeometry t with
      | some (e, _) => e == 256
      | none => false)) = true := by
  decide

/-- Bytes of a row of `n_in` elements of a type. -/
def rowBytes (t n_in : Nat) : Option Nat :=
  match blockGeometry t with
  | some (e, b) => some (n_in / e * b)
  | none => none

/-- A 160-element IQ4_NL row is 5 blocks of 18 bytes = 90 bytes. -/
theorem ple_row_bytes : rowBytes 20 160 = some 90 := by decide

/-- Type 29 has geometry but no name in `ggml_type_name`, which falls through to "?". -/
theorem iq1m_has_geometry : (blockGeometry 29).isSome = true := by decide

-- § Cursor

/-- Cursor state: position and region size. -/
structure Cur where
  pos : Nat
  size : Nat

/-- A bounds-checked read of `n` bytes. -/
def Cur.read (c : Cur) (n : Nat) : Option Cur :=
  if c.pos + n ≤ c.size then some { c with pos := c.pos + n } else none

/-- A successful read advances by exactly `n` and stays within the region. -/
theorem read_some (c c' : Cur) (n : Nat) (h : c.read n = some c') :
    c'.pos = c.pos + n ∧ c'.pos ≤ c'.size ∧ c'.size = c.size := by
  unfold Cur.read at h
  by_cases hb : c.pos + n ≤ c.size
  · simp only [hb, if_true, Option.some.injEq] at h
    subst h
    exact ⟨rfl, hb, rfl⟩
  · simp [hb] at h

/-- A read that would pass the end fails instead of overrunning. -/
theorem read_none (c : Cur) (n : Nat) (h : c.size < c.pos + n) : c.read n = none := by
  unfold Cur.read
  have hb : ¬ (c.pos + n ≤ c.size) := by omega
  simp [hb]

/-- The cursor never moves backwards. -/
theorem read_monotone (c c' : Cur) (n : Nat) (h : c.read n = some c') : c.pos ≤ c'.pos := by
  have := read_some c c' n h
  omega

-- § Magic and alignment

/-- The GGUF magic 0x46554747 is the little-endian reading of the bytes 'G','G','U','F'. -/
theorem gguf_magic : 0x47 + 0x47 * 256 + 0x55 * 65536 + 0x46 * 16777216 = 0x46554747 := by decide

/-- Alignment rounding of the data-section start. -/
def dataStart (pos align : Nat) : Nat := (pos + align - 1) / align * align

/-- The data start is a multiple of the alignment. -/
theorem dataStart_dvd (pos align : Nat) : align ∣ dataStart pos align := by
  unfold dataStart
  exact Nat.dvd_mul_left align _

/-- The data start is not before the end of the header. -/
theorem dataStart_ge (pos align : Nat) (ha : 0 < align) : pos ≤ dataStart pos align := by
  unfold dataStart
  have h := Nat.div_add_mod (pos + align - 1) align
  have h2 := Nat.mod_lt (pos + align - 1) ha
  rw [Nat.mul_comm align ((pos + align - 1) / align)] at h
  omega

/-- The padding is strictly less than one alignment unit. -/
theorem dataStart_lt (pos align : Nat) (ha : 0 < align) : dataStart pos align < pos + align := by
  unfold dataStart
  have h := Nat.div_mul_le_self (pos + align - 1) align
  omega

/-- The default alignment 32 leaves an already-aligned header unchanged. -/
theorem dataStart_aligned (k : Nat) : dataStart (32 * k) 32 = 32 * k := by
  unfold dataStart
  omega

-- § Architecture guard

/-- The compiled-in architecture constants. -/
structure Guard where
  blockCount : Nat := 48
  hidden : Nat := 2560
  experts : Nat := 512
  expertsUsed : Nat := 10
  headCount : Nat := 24
  headCountKv : Nat := 2

/-- The guard accepts only `qwen4exp` with every (found, wanted) pair equal. -/
def archOk (arch : String) (pairs : List (Nat × Nat)) : Bool :=
  arch == "qwen4exp" && pairs.all (fun p => p.1 == p.2)

/-- Any other architecture is refused. -/
theorem archOk_wrong_arch (arch : String) (pairs : List (Nat × Nat)) (h : arch ≠ "qwen4exp") :
    archOk arch pairs = false := by
  unfold archOk
  simp [h]

/-- The default guard has a whole GQA ratio (24 query heads over 2 kv heads). -/
theorem guard_gqa_ratio : (24 : Nat) % 2 = 0 ∧ 24 / 2 = 12 := by decide

/-- Top-k experts fit in the expert count. -/
theorem guard_experts_fit : (10 : Nat) ≤ 512 := by decide

end Strata.Gguf
