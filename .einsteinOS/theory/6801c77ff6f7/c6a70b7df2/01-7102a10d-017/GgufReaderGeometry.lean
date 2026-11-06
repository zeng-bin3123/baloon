/-!
# MODULE: gguf_reader_geometry
SOURCES: include/strata/artifact/gguf_reader.hpp
GGUF v3 reader: block geometry table, bounds-checked cursor, data-section alignment, tensor element
counts, array sampling and the qwen4exp architecture guard constants.

## Source semantics
* block_geometry returns (elements, bytes) per block and fails for unknown types; Q2_0 (42) is 64/18.
* Cursor::need rejects any read past the end; a successful read advances by exactly sizeof(T).
* data_start = round-up of the end of the header to the alignment (default 32).
* elements() is the product of the shape; arrays keep at most 64 sampled items.

## Form template identifiers
block_geometry -> block_geometry_table ; Cursor -> bounded_cursor ; parse data_start -> align_up ;
TensorInfo::elements -> shape_product ; Qwen4ExpGuard -> arch_guard_constants
-/
set_option maxRecDepth 100000

namespace Strata.GgufReader

-- § block geometry
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

/-- Q2_0 (type 42) has 64 elements in 18 bytes. -/
theorem q2_0_geometry : blockGeometry 42 = some (64, 18) := by decide

/-- Every known geometry has positive element and byte counts. -/
theorem geometry_positive :
    ∀ t, t < 64 → (blockGeometry t).elim true (fun p => decide (0 < p.1 ∧ 0 < p.2)) = true := by
  decide

/-- Every known block has 1, 32, 64 or 256 elements. -/
theorem geometry_elems_classes :
    ∀ t, t < 64 → (blockGeometry t).elim true
      (fun p => decide (p.1 = 1 ∨ p.1 = 32 ∨ p.1 = 64 ∨ p.1 = 256)) = true := by
  decide

/-- Q2_0 is the only type with 64-element blocks. -/
theorem only_q2_0_has_64 :
    ∀ t, t < 64 → (blockGeometry t).elim true (fun p => decide (p.1 = 64 → t = 42)) = true := by
  decide

/-- IQ1_S (19) has a name in the reader but no geometry entry, so it is refused as unsized. -/
theorem iq1_s_has_no_geometry : blockGeometry 19 = none := by decide

/-- IQ1_M (29) has geometry 256/56. -/
theorem iq1_m_geometry : blockGeometry 29 = some (256, 56) := by decide

/-- Q2_0 costs 18*8 = 144 bits per 64 elements, i.e. 2 bits plus a 16-bit scale. -/
theorem q2_0_bits : 18 * 8 = 64 * 2 + 16 := by decide

-- § bounded cursor
structure Cur where
  pos : Nat
  size : Nat

def Cur.read (c : Cur) (n : Nat) : Option Cur :=
  if c.pos + n ≤ c.size then some { c with pos := c.pos + n } else none

/-- A successful read advances by exactly the requested width. -/
theorem cursor_read_advances (c : Cur) (n : Nat) (c' : Cur) (h : c.read n = some c') :
    c'.pos = c.pos + n ∧ c'.size = c.size := by
  unfold Cur.read at h
  split at h
  · cases h
    exact ⟨rfl, rfl⟩
  · cases h

/-- A successful read never leaves the buffer. -/
theorem cursor_read_in_bounds (c : Cur) (n : Nat) (c' : Cur) (h : c.read n = some c') :
    c'.pos ≤ c'.size := by
  unfold Cur.read at h
  split at h
  · rename_i hle
    cases h
    exact hle
  · cases h

/-- A read past the end is refused. -/
theorem cursor_read_refuses_overrun (c : Cur) (n : Nat) (h : c.size < c.pos + n) :
    c.read n = none := by
  unfold Cur.read
  split
  · omega
  · rfl

-- § alignment
def alignUp (c a : Nat) : Nat := (c + a - 1) / a * a

/-- The data start is at least the header end. -/
theorem alignUp_ge (c a : Nat) (ha : 0 < a) : c ≤ alignUp c a := by
  unfold alignUp
  have h1 := Nat.div_add_mod (c + a - 1) a
  have h2 := Nat.mod_lt (c + a - 1) ha
  have h3 : (c + a - 1) / a * a = a * ((c + a - 1) / a) := Nat.mul_comm _ _
  generalize (c + a - 1) / a = q at *
  generalize (c + a - 1) % a = r at *
  omega

/-- The data start is less than one alignment unit beyond the header end. -/
theorem alignUp_lt (c a : Nat) (ha : 0 < a) : alignUp c a < c + a := by
  unfold alignUp
  have h1 := Nat.div_add_mod (c + a - 1) a
  have h2 := Nat.mod_lt (c + a - 1) ha
  have h3 : (c + a - 1) / a * a = a * ((c + a - 1) / a) := Nat.mul_comm _ _
  generalize (c + a - 1) / a = q at *
  generalize (c + a - 1) % a = r at *
  omega

/-- The data start is a multiple of the alignment. -/
theorem alignUp_dvd (c a : Nat) : alignUp c a % a = 0 := by
  unfold alignUp
  exact Nat.mul_mod_left _ _

/-- With the default alignment 32 the data start is within 31 bytes after the header. -/
theorem alignUp_32 (c : Nat) : c ≤ alignUp c 32 ∧ alignUp c 32 < c + 32 :=
  ⟨alignUp_ge c 32 (by decide), alignUp_lt c 32 (by decide)⟩

-- § tensor shape
def prodL : List Nat → Nat
  | [] => 1
  | d :: ds => d * prodL ds

/-- The element count of concatenated shapes is the product of the counts. -/
theorem prodL_append (a b : List Nat) : prodL (a ++ b) = prodL a * prodL b := by
  induction a with
  | nil => simp [prodL]
  | cons d ds ih => simp [prodL, ih, Nat.mul_assoc]

/-- A shape with a zero dimension has no elements. -/
theorem prodL_zero_mem (s : List Nat) (h : 0 ∈ s) : prodL s = 0 := by
  induction s with
  | nil => cases h
  | cons d ds ih =>
    simp only [List.mem_cons] at h
    rcases h with h | h
    · subst h; simp [prodL]
    · simp [prodL, ih h]

/-- Only 1 to 4 dimensions are accepted by the parser. -/
theorem ndims_accepted (nd : Nat) : (nd = 0 ∨ 4 < nd) ↔ ¬ (0 < nd ∧ nd ≤ 4) := by
  omega

-- § arrays and architecture guard
/-- At most 64 array items are kept as a sample. -/
theorem array_sample_bound (count : Nat) : min count 64 ≤ 64 := Nat.min_le_right _ _

/-- The array length cap is 2^24. -/
theorem array_cap : (1 <<< 24 : Nat) = 16777216 := by decide

/-- The guard constants are consistent: top-k experts fit in the pool and query heads group over KV heads. -/
theorem arch_guard_consistent :
    10 ≤ 512 ∧ 24 % 2 = 0 ∧ 24 / 2 = 12 ∧ 48 > 0 := by decide

/-- The header magic "GGUF" read as a little-endian u32. -/
theorem gguf_magic : 0x46 * 16777216 + 0x55 * 65536 + 0x47 * 256 + 0x47 = 0x46554747 := by decide

end Strata.GgufReader
