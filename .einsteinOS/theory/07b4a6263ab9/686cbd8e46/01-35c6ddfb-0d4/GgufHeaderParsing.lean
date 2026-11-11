/-!
# MODULE: gguf_header_parsing
SOURCES: include/strata/artifact/gguf_reader.hpp
GGUF v3 header reader: bounds-checked cursor, metadata value decoding, tensor directory, aligned
data-section start, the ggml block-geometry table and the qwen4exp architecture guard.

## Source semantics
* `Cursor::need(n)` fails iff `pos + n > size`; every successful read keeps `pos <= size`, so the
  reader cannot run past the mapping; a sequence of reads advances `pos` by the sum of their sizes.
* Arrays longer than 2^24 are refused; only `min(count, 64)` items are kept.  Nesting depth > 2 fails.
* Signed integer metadata is stored sign-extended into `uint64` (two's complement mod 2^64) and is
  recoverable.
* `data_start = ceil(pos / align) * align`: aligned, at or after the header end, less than one
  alignment unit past it.
* `block_geometry` gives (elements, bytes); Q2_0 = (64, 18).  Every listed quantized type has bytes
  equal to its header fields plus packed payload.
* `check_architecture` returns "" iff the architecture string is `qwen4exp` and all six required
  keys are present and equal the compiled-in constants; otherwise it reports the FIRST failure.

## Form template identifiers
Cursor -> Cursor, need/read -> Cursor.read, read_value ARRAY -> keptItems, data_start -> alignUp,
block_geometry -> blockGeometry, check_architecture -> checkReqs, Qwen4ExpGuard -> guardDefaults
-/

namespace Strata.Gguf

-- § bounds-checked cursor

/-- Cursor state: current position and mapped size. -/
structure Cursor where
  pos : Nat
  size : Nat

/-- `need(n)` then advance by `n`; `none` is the thrown "unexpected end of file". -/
def Cursor.read (c : Cursor) (n : Nat) : Option Cursor :=
  if c.pos + n > c.size then none else some { c with pos := c.pos + n }

/-- A successful read keeps the cursor inside the mapping and advances by exactly `n`. -/
theorem read_in_bounds (c c' : Cursor) (n : Nat) (h : c.read n = some c') :
    c'.pos ≤ c'.size ∧ c'.pos = c.pos + n ∧ c'.size = c.size := by
  unfold Cursor.read at h
  split at h
  · cases h
  · cases h; simp; omega

/-- A read fails exactly when it would run past the end. -/
theorem read_fails_iff (c : Cursor) (n : Nat) : c.read n = none ↔ c.pos + n > c.size := by
  unfold Cursor.read; split <;> simp_all

/-- Run a sequence of reads, stopping at the first failure. -/
def readAll : Cursor → List Nat → Option Cursor
  | c, [] => some c
  | c, n :: ns => match c.read n with
    | none => none
    | some c' => readAll c' ns

/-- A successful sequence of reads advances by the total size and never leaves the mapping. -/
theorem readAll_spec (ns : List Nat) (c c' : Cursor) (h : readAll c ns = some c') :
    c'.pos = c.pos + ns.foldr (· + ·) 0 ∧ c'.size = c.size ∧ c.pos ≤ c'.pos := by
  induction ns generalizing c with
  | nil => simp [readAll] at h; subst h; simp
  | cons n ns ih =>
    simp only [readAll] at h
    split at h
    · cases h
    · rename_i c1 hc1
      obtain ⟨_, hp, hs⟩ := read_in_bounds c c1 n hc1
      obtain ⟨hp', hs', _⟩ := ih c1 h
      simp only [List.foldr]
      refine ⟨?_, ?_, ?_⟩ <;> omega

/-- A GGUF string is a u64 length then that many bytes: it consumes `8 + len` bytes. -/
theorem string_read_size (c : Cursor) (len : Nat) (c' : Cursor) (h : readAll c [8, len] = some c') :
    c'.pos = c.pos + 8 + len := by
  have := (readAll_spec [8, len] c c' h).1
  simp at this; omega

-- § metadata values

/-- Items kept for an array of `count` elements (a sample of at most 64). -/
def keptItems (count : Nat) : Nat := min count 64

/-- Arrays are capped: at most 64 items kept, and all are kept for short arrays. -/
theorem keptItems_spec (count : Nat) :
    keptItems count ≤ 64 ∧ keptItems count ≤ count ∧ (count ≤ 64 → keptItems count = count) := by
  unfold keptItems; omega

/-- Array length guard: an accepted count is at most 2^24. -/
def arrayLenOk (count : Nat) : Bool := decide (count ≤ 2 ^ 24)

/-- The guard accepts exactly the counts up to 16,777,216. -/
theorem arrayLenOk_iff (count : Nat) : arrayLenOk count = true ↔ count ≤ 16777216 := by
  unfold arrayLenOk; simp

/-- Sign extension into `uint64`: `(uint64)(int64)x`. -/
def signExtend64 (x : Int) : Nat := (x % 18446744073709551616).toNat

/-- Reading the stored word back as `int64` recovers the original signed value. -/
def asInt64 (u : Nat) : Int := if u < 9223372036854775808 then u else (u : Int) - 18446744073709551616

/-- Signed metadata (I8/I16/I32/I64) survives the uint64 storage exactly. -/
theorem signExtend_roundtrip (x : Int) (h1 : -9223372036854775808 ≤ x) (h2 : x < 9223372036854775808) :
    asInt64 (signExtend64 x) = x := by
  unfold asInt64 signExtend64
  by_cases hx : 0 ≤ x
  · have : x % 18446744073709551616 = x := Int.emod_eq_of_lt hx (by omega)
    rw [this]; split <;> omega
  · have : x % 18446744073709551616 = x + 18446744073709551616 := by omega
    rw [this]; split <;> omega

/-- Booleans are normalised: any non-zero byte reads as 1. -/
def readBool (b : Nat) : Nat := if b ≠ 0 then 1 else 0

/-- A decoded bool is 0 or 1. -/
theorem readBool_binary (b : Nat) : readBool b = 0 ∨ readBool b = 1 := by
  unfold readBool; split <;> simp

-- § data section alignment

/-- `(pos + align - 1) / align * align`. -/
def alignUp (pos align : Nat) : Nat := (pos + align - 1) / align * align

/-- The data section starts on an alignment boundary, at or after the header, within one unit. -/
theorem alignUp_spec (pos align : Nat) (ha : 0 < align) :
    alignUp pos align % align = 0 ∧ pos ≤ alignUp pos align ∧ alignUp pos align < pos + align := by
  unfold alignUp
  have hdm := Nat.div_add_mod (pos + align - 1) align
  have hlt := Nat.mod_lt (pos + align - 1) ha
  refine ⟨Nat.mul_mod_left _ _, ?_, ?_⟩
  · rw [Nat.mul_comm]
    generalize align * ((pos + align - 1) / align) = m at *
    omega
  · rw [Nat.mul_comm]
    generalize align * ((pos + align - 1) / align) = m at *
    omega

/-- Default alignment 32 is used when `general.alignment` is absent or zero. -/
def effectiveAlign (mv : Option Nat) : Nat :=
  match mv with
  | some a => if a ≠ 0 then a else 32
  | none => 32

/-- The effective alignment is never zero, so `alignUp_spec` always applies. -/
theorem effectiveAlign_pos (m : Option Nat) : 0 < effectiveAlign m := by
  unfold effectiveAlign
  split
  · split <;> omega
  · decide

-- § block geometry

/-- `block_geometry`: (elements per block, bytes per block), `none` for unsupported types. -/
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

/-- Q2_0 (type 42) is 64 elements in 18 bytes: an fp16 scale plus 64 two-bit codes. -/
theorem q2_0_geometry : blockGeometry 42 = some (64, 18) ∧ 2 + 64 * 2 / 8 = 18 := by decide

/-- The table agrees with the field layouts the dequantizers read
(Q4_0, Q5_0, Q8_0, IQ4_NL, Q3_K, Q4_K, Q5_K, Q6_K, IQ4_XS). -/
theorem geometry_matches_dequant_layouts :
    blockGeometry 2 = some (32, 2 + 16) ∧ blockGeometry 6 = some (32, 2 + 4 + 16) ∧
    blockGeometry 8 = some (32, 2 + 32) ∧ blockGeometry 20 = some (32, 2 + 16) ∧
    blockGeometry 11 = some (256, 32 + 64 + 12 + 2) ∧ blockGeometry 12 = some (256, 2 + 2 + 12 + 128) ∧
    blockGeometry 13 = some (256, 2 + 2 + 12 + 32 + 128) ∧ blockGeometry 14 = some (256, 128 + 64 + 16 + 2) ∧
    blockGeometry 23 = some (256, 2 + 2 + 4 + 128) := by decide

/-- The table agrees with the i-quant block sizes the AVX-512 kernels stride by. -/
theorem geometry_matches_iq_kernels :
    blockGeometry 16 = some (256, 66) ∧ blockGeometry 17 = some (256, 74) ∧
    blockGeometry 18 = some (256, 98) ∧ blockGeometry 21 = some (256, 110) ∧
    blockGeometry 22 = some (256, 82) := by decide

/-- Bytes of a tensor row of `n` elements (a whole number of blocks). -/
def rowBytes (elems bytes n : Nat) : Nat := n / elems * bytes

/-- A Q2_0 row of the hidden size 2560 is 40 blocks = 720 bytes. -/
theorem q2_0_row_2560 : rowBytes 64 18 2560 = 720 := by decide

-- § architecture guard

/-- The compiled-in qwen4exp constants. -/
def guardDefaults : List (String × Nat) :=
  [("qwen4exp.block_count", 48), ("qwen4exp.embedding_length", 2560), ("qwen4exp.expert_count", 512),
   ("qwen4exp.expert_used_count", 10), ("qwen4exp.attention.head_count", 24),
   ("qwen4exp.attention.head_count_kv", 2)]

/-- Check each required key in order; `none` is success, `some key` names the first failure. -/
def checkReqs (lookup : String → Option Nat) : List (String × Nat) → Option String
  | [] => none
  | (k, w) :: rest =>
    match lookup k with
    | none => some k
    | some v => if v = w then checkReqs lookup rest else some k

/-- The guard passes iff every required key is present with exactly the expected value. -/
theorem checkReqs_none_iff (lookup : String → Option Nat) (reqs : List (String × Nat)) :
    checkReqs lookup reqs = none ↔ ∀ r ∈ reqs, lookup r.1 = some r.2 := by
  induction reqs with
  | nil => simp [checkReqs]
  | cons r rest ih =>
    obtain ⟨k, w⟩ := r
    simp only [checkReqs, List.mem_cons, forall_eq_or_imp]
    split
    · rename_i hk; simp [hk]
    · rename_i v hk
      split
      · rename_i hv; subst hv; rw [ih]; simp [hk]
      · rename_i hv; simp [hk, hv]

/-- When the guard fails, it names a key that is genuinely missing or mismatched. -/
theorem checkReqs_failure_sound (lookup : String → Option Nat) (reqs : List (String × Nat)) (k : String)
    (h : checkReqs lookup reqs = some k) : ∃ w, (k, w) ∈ reqs ∧ lookup k ≠ some w := by
  induction reqs with
  | nil => simp [checkReqs] at h
  | cons r rest ih =>
    obtain ⟨k0, w0⟩ := r
    simp only [checkReqs] at h
    split at h
    · rename_i hk; cases h; exact ⟨w0, by simp, by simp [hk]⟩
    · rename_i v hk
      split at h
      · obtain ⟨w, hw, hne⟩ := ih h; exact ⟨w, List.mem_cons_of_mem _ hw, hne⟩
      · rename_i hv; cases h; exact ⟨w0, by simp, by simp [hk]; exact hv⟩

/-- The full guard: architecture string first, then the six keys. -/
def checkArchitecture (arch : Option String) (lookup : String → Option Nat) : Bool :=
  match arch with
  | some "qwen4exp" => (checkReqs lookup guardDefaults).isNone
  | _ => false

/-- The guard accepts only `qwen4exp` with hidden size 2560, 512 experts and 10 used experts (and
the other compiled constants). -/
theorem checkArchitecture_sound (arch : Option String) (lookup : String → Option Nat)
    (h : checkArchitecture arch lookup = true) :
    arch = some "qwen4exp" ∧ lookup "qwen4exp.embedding_length" = some 2560 ∧
    lookup "qwen4exp.expert_count" = some 512 ∧ lookup "qwen4exp.expert_used_count" = some 10 := by
  unfold checkArchitecture at h
  split at h
  · have hn : checkReqs lookup guardDefaults = none := by
      cases hc : checkReqs lookup guardDefaults <;> simp_all
    have hall := (checkReqs_none_iff lookup guardDefaults).1 hn
    refine ⟨rfl, hall ("qwen4exp.embedding_length", 2560) (by simp [guardDefaults]),
      hall ("qwen4exp.expert_count", 512) (by simp [guardDefaults]),
      hall ("qwen4exp.expert_used_count", 10) (by simp [guardDefaults])⟩
  · cases h

end Strata.Gguf
