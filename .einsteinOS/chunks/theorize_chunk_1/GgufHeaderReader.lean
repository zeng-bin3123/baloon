/-!
# MODULE: gguf_header_reader
Model of the GGUF v3 header reader: ggml type-id naming, block geometry (elements / bytes per block),
tensor element counts, the bounds-checked header cursor, data-section alignment and the
architecture guard.

## Source semantics
* `block_geometry t` returns `(elems, bytes)` for supported type ids and fails otherwise. Q2_0 (id 42)
  is 64 elements / 18 bytes.
* `Cursor.read` either advances `pos` by exactly the size read or fails; it never reads past `size`.
* `data_start = ceil(pos / align) * align`, and must not exceed the file size.
* The architecture guard returns the empty string iff every required key is present and equal.
* Array metadata is bounded: count ≤ 2^24 and nesting depth ≤ 2.

## Form template identifiers
* `blockGeometry`      ← `strata::block_geometry`
* `alignUp`            ← `data_start_` computation in `GgufFile::parse`
* `Cursor`             ← `strata::Cursor`
* `tensorElements`     ← `TensorInfo::elements`
* `archGuard`          ← `strata::check_architecture`
-/

namespace Strata.Gguf

-- § block geometry

/-- Block geometry `(elements per block, bytes per block)` for a ggml type id; `none` if unsupported. -/
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

/-- Q2_0 is 64 elements in an 18-byte block (fp16 scale plus 16 code bytes). -/
theorem q2_0_geometry : blockGeometry 42 = some (64, 18) := by
  rfl

/-- The Q2_0 block payload layout: 2 bytes of scale plus 16 bytes of codes, 4 codes per byte. -/
theorem q2_0_layout : 2 + 16 = 18 ∧ 16 * 4 = 64 := by
  exact ⟨rfl, rfl⟩

/-- Unsupported type ids are rejected. -/
theorem blockGeometry_unknown : blockGeometry 4 = none ∧ blockGeometry 5 = none ∧
    blockGeometry 100 = none := by
  exact ⟨rfl, rfl, rfl⟩

/-- Every supported type has a positive element count per block. -/
theorem blockGeometry_elems_pos : ∀ t, t < 64 → ∀ e b, blockGeometry t = some (e, b) → 0 < e := by
  intro t ht e b h
  interval_cases t <;> simp [blockGeometry] at h <;> omega

/-- Every supported type has a positive byte count per block. -/
theorem blockGeometry_bytes_pos : ∀ t, t < 64 → ∀ e b, blockGeometry t = some (e, b) → 0 < b := by
  intro t ht e b h
  interval_cases t <;> simp [blockGeometry] at h <;> omega

/-- Q4_0 and IQ4_NL share a geometry. -/
theorem iq4_nl_geometry_eq_q4_0 : blockGeometry 20 = blockGeometry 2 := by
  rfl

/-- Q3_K and IQ3_S share a geometry. -/
theorem q3_k_geometry_eq_iq3_s : blockGeometry 11 = blockGeometry 21 := by
  rfl

/-- All K-quant super-blocks hold 256 elements. -/
theorem kquant_super_block_256 :
    (blockGeometry 10).map Prod.fst = some 256 ∧ (blockGeometry 11).map Prod.fst = some 256 ∧
    (blockGeometry 12).map Prod.fst = some 256 ∧ (blockGeometry 13).map Prod.fst = some 256 ∧
    (blockGeometry 14).map Prod.fst = some 256 := by
  exact ⟨rfl, rfl, rfl, rfl, rfl⟩

/-- Bytes occupied by `n` elements of type `t` (only meaningful when `n` is a block multiple). -/
def rowBytes (t n : Nat) : Option Nat :=
  match blockGeometry t with
  | some (e, b) => some (n / e * b)
  | none => none

/-- A 4096-element Q2_0 row is 64 blocks = 1152 bytes. -/
theorem q2_0_row_4096 : rowBytes 42 4096 = some 1152 := by
  rfl

/-- Row bytes for F32 is 4 per element. -/
theorem f32_row_bytes (n : Nat) : rowBytes 0 n = some (n * 4) := by
  simp [rowBytes, blockGeometry]

/-- Row bytes for F16 is 2 per element. -/
theorem f16_row_bytes (n : Nat) : rowBytes 1 n = some (n * 2) := by
  simp [rowBytes, blockGeometry]

-- § tensor info

/-- Element count of a tensor shape: product of dims. -/
def tensorElements (shape : List Nat) : Nat := shape.foldl (· * ·) 1

/-- A 2-D shape multiplies out. -/
theorem tensorElements_two (a b : Nat) : tensorElements [a, b] = a * b := by
  simp [tensorElements]

/-- A zero dimension yields zero elements. -/
theorem tensorElements_zero_dim_example : tensorElements [3, 0, 5] = 0 := by
  rfl

/-- Helper: folding multiplication from accumulator `a`. -/
theorem foldl_mul_acc (l : List Nat) (a : Nat) :
    l.foldl (· * ·) a = a * l.foldl (· * ·) 1 := by
  induction l generalizing a with
  | nil => simp
  | cons x xs ih =>
    simp only [List.foldl_cons]
    rw [ih (a * x), ih (1 * x)]
    simp [Nat.mul_assoc]

/-- Element count is multiplicative over concatenated shapes. -/
theorem tensorElements_append (l₁ l₂ : List Nat) :
    tensorElements (l₁ ++ l₂) = tensorElements l₁ * tensorElements l₂ := by
  unfold tensorElements
  rw [List.foldl_append, foldl_mul_acc l₂]

-- § alignment / data start

/-- Round `pos` up to a multiple of `align`, as in `GgufFile::parse`. -/
def alignUp (pos align : Nat) : Nat := (pos + align - 1) / align * align

/-- The aligned start is a multiple of the alignment. -/
theorem alignUp_dvd (pos align : Nat) : align ∣ alignUp pos align := by
  unfold alignUp
  exact Nat.dvd_mul_left align _

/-- The aligned start is never before `pos`. -/
theorem alignUp_ge (pos align : Nat) (h : 0 < align) : pos ≤ alignUp pos align := by
  unfold alignUp
  have h1 := Nat.div_add_mod (pos + align - 1) align
  have h2 := Nat.mod_lt (pos + align - 1) h
  have h3 : align * ((pos + align - 1) / align) = (pos + align - 1) / align * align :=
    Nat.mul_comm _ _
  omega

/-- The aligned start overshoots by less than one alignment unit. -/
theorem alignUp_lt (pos align : Nat) (h : 0 < align) : alignUp pos align < pos + align := by
  unfold alignUp
  have h1 := Nat.div_add_mod (pos + align - 1) align
  have h3 : align * ((pos + align - 1) / align) = (pos + align - 1) / align * align :=
    Nat.mul_comm _ _
  omega

/-- Aligning an already aligned position is the identity. -/
theorem alignUp_of_aligned (k align : Nat) (h : 0 < align) : alignUp (k * align) align = k * align := by
  unfold alignUp
  have : (k * align + align - 1) / align = k := by
    have e : k * align + align - 1 = (align - 1) + align * k := by
      have := Nat.mul_comm k align
      omega
    rw [e, Nat.add_mul_div_left _ _ h, Nat.div_eq_of_lt (by omega)]
    simp
  rw [this]

/-- Default alignment 32 on position 100 gives 128. -/
theorem alignUp_example : alignUp 100 32 = 128 := by
  decide

-- § bounds-checked cursor

/-- Cursor state: position and total size. -/
structure Cursor where
  pos : Nat
  size : Nat

/-- Reading `n` bytes: fails (`none`) if it would pass the end, else advances by exactly `n`. -/
def Cursor.read (c : Cursor) (n : Nat) : Option Cursor :=
  if c.pos + n ≤ c.size then some { c with pos := c.pos + n } else none

/-- A successful read advances by exactly `n`. -/
theorem Cursor.read_pos (c c' : Cursor) (n : Nat) (h : c.read n = some c') :
    c'.pos = c.pos + n := by
  unfold Cursor.read at h
  split at h
  · cases h; rfl
  · cases h

/-- A successful read never exceeds the size. -/
theorem Cursor.read_in_bounds (c c' : Cursor) (n : Nat) (h : c.read n = some c') :
    c'.pos ≤ c'.size := by
  unfold Cursor.read at h
  split at h
  · rename_i hle
    cases h
    exact hle
  · cases h

/-- A successful read preserves the size. -/
theorem Cursor.read_size (c c' : Cursor) (n : Nat) (h : c.read n = some c') :
    c'.size = c.size := by
  unfold Cursor.read at h
  split at h
  · cases h; rfl
  · cases h

/-- A read fails exactly when it would overrun. -/
theorem Cursor.read_none_iff (c : Cursor) (n : Nat) : c.read n = none ↔ c.size < c.pos + n := by
  unfold Cursor.read
  split
  · rename_i h
    constructor
    · intro h'; cases h'
    · intro h'; omega
  · rename_i h
    constructor
    · intro _; omega
    · intro _; rfl

/-- Two reads compose into one read of the summed length. -/
theorem Cursor.read_read (c c₁ c₂ : Cursor) (n m : Nat)
    (h₁ : c.read n = some c₁) (h₂ : c₁.read m = some c₂) : c.read (n + m) = some c₂ := by
  have p1 := Cursor.read_pos c c₁ n h₁
  have s1 := Cursor.read_size c c₁ n h₁
  have p2 := Cursor.read_pos c₁ c₂ m h₂
  have s2 := Cursor.read_size c₁ c₂ m h₂
  have b2 := Cursor.read_in_bounds c₁ c₂ m h₂
  unfold Cursor.read
  have hle : c.pos + (n + m) ≤ c.size := by omega
  rw [if_pos hle]
  cases c₂
  simp only at p2 s2 ⊢
  congr 1
  simp only [Cursor.mk.injEq]
  omega

/-- Reading zero bytes from an in-bounds cursor succeeds and is a no-op. -/
theorem Cursor.read_zero (c : Cursor) (h : c.pos ≤ c.size) : c.read 0 = some c := by
  unfold Cursor.read
  rw [if_pos (by omega)]
  rfl

-- § array bounds

/-- Maximum array length accepted by the reader (2^24). -/
def maxArrayLen : Nat := 1 <<< 24

/-- The array length cap is 16777216. -/
theorem maxArrayLen_val : maxArrayLen = 16777216 := by
  decide

/-- Nesting is rejected beyond depth 2. -/
def depthOk (d : Nat) : Bool := d ≤ 2

/-- Depth 0..2 are accepted, 3 is rejected. -/
theorem depthOk_cases : depthOk 0 = true ∧ depthOk 2 = true ∧ depthOk 3 = false := by
  decide

-- § architecture guard

/-- Guard constants for the target model. -/
structure Guard where
  blockCount : Nat := 48
  hidden : Nat := 2560
  experts : Nat := 512
  expertsUsed : Nat := 10
  headCount : Nat := 24
  headCountKv : Nat := 2

/-- Required metadata (key, expected value) pairs. -/
def Guard.reqs (g : Guard) : List (String × Nat) :=
  [("qwen4exp.block_count", g.blockCount), ("qwen4exp.embedding_length", g.hidden),
   ("qwen4exp.expert_count", g.experts), ("qwen4exp.expert_used_count", g.expertsUsed),
   ("qwen4exp.attention.head_count", g.headCount),
   ("qwen4exp.attention.head_count_kv", g.headCountKv)]

/-- Guard result: `""` when OK, a non-empty message otherwise. `md` maps keys to integer values. -/
def archGuard (g : Guard) (arch : Option String) (md : String → Option Nat) : String :=
  match arch with
  | none => "missing general.architecture"
  | some a =>
    if a ≠ "qwen4exp" then "wrong architecture"
    else
      match g.reqs.find? (fun r => md r.1 ≠ some r.2) with
      | none => ""
      | some _ => "metadata mismatch"

/-- The guard has exactly six requirements. -/
theorem guard_reqs_length (g : Guard) : g.reqs.length = 6 := by
  rfl

/-- Missing architecture is refused. -/
theorem archGuard_missing (g : Guard) (md : String → Option Nat) :
    archGuard g none md ≠ "" := by
  simp [archGuard]

/-- A wrong architecture string is refused. -/
theorem archGuard_wrong (g : Guard) (md : String → Option Nat) (a : String) (h : a ≠ "qwen4exp") :
    archGuard g (some a) md ≠ "" := by
  simp [archGuard, h]

/-- If every required key matches, the guard accepts. -/
theorem archGuard_ok (g : Guard) (md : String → Option Nat)
    (h : ∀ r ∈ g.reqs, md r.1 = some r.2) : archGuard g (some "qwen4exp") md = "" := by
  have hf : g.reqs.find? (fun r => md r.1 ≠ some r.2) = none := by
    rw [List.find?_eq_none]
    intro r hr
    simp [h r hr]
  unfold archGuard
  have hc : ¬ ("qwen4exp" ≠ "qwen4exp") := by simp
  dsimp only
  rw [if_neg hc]
  simp only [hf]

/-- Accepting implies every required key matched (soundness of the guard). -/
theorem archGuard_sound (g : Guard) (md : String → Option Nat)
    (h : archGuard g (some "qwen4exp") md = "") : ∀ r ∈ g.reqs, md r.1 = some r.2 := by
  intro r hr
  by_contra hne
  have hf : g.reqs.find? (fun r => md r.1 ≠ some r.2) ≠ none := by
    intro hnone
    rw [List.find?_eq_none] at hnone
    have := hnone r hr
    simp at this
    exact hne this
  unfold archGuard at h
  simp only [ne_eq, not_true_eq_false, if_false] at h
  cases hh : g.reqs.find? (fun r => md r.1 ≠ some r.2) with
  | none => exact hf hh
  | some _ => rw [hh] at h; simp at h

end Strata.Gguf
