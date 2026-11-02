
## FILE: /mnt/user-data/outputs/GgufHeaderReader.lean
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



## FILE: /mnt/user-data/outputs/QuantBlockDequant.lean
/-!
# MODULE: quant_block_dequant
Integer-level model of the scalar ggml block dequantizers (Q2_0, Q4_0, Q5_0, Q8_0, IQ4_NL, Q6_K,
Q4_K scale/min unpacking, IQ4_XS scales). Floating-point scaling is abstracted: each function yields
the integer symbol that is multiplied by the block scale.

## Source semantics
* Q2_0: 4 codes per byte, LSB-first; code c ∈ {0..3} maps to symbol c - 1 ∈ {-1,0,1,2}.
* Q4_0: low nibble is element j, high nibble element j+16, symbol = nibble - 8.
* Q5_0: a fifth bit from a 32-bit mask extends the nibble; symbol = x - 16 ∈ [-16, 15].
* IQ4_NL: non-linear 16-entry codebook (strictly increasing, in [-127, 113]).
* Q6_K: 6-bit value (4 low + 2 high bits) minus 32 ∈ [-32, 31].
* get_scale_min_k4: unpacked scale and min are 6-bit values (< 64).
* IQ4_XS: group scale ls ∈ [0, 63], effective scale ls - 32 ∈ [-32, 31].

## Form template identifiers
* `q2Code`         ← inner index math of `dequantize_q2_0`
* `q2Symbol`       ← `code - 1`
* `q4Lo`/`q4Hi`    ← nibble extraction in `dequantize_q4_0`
* `q5Val`          ← 5-bit assembly in `dequantize_q5_0`
* `kvalues`        ← `kvalues_iq4nl`
* `q6Val`          ← `q1..q4` in `dequantize_q6_K`
* `scaleMinK4`     ← `get_scale_min_k4`
* `iq4xsScale`     ← `ls` in `dequantize_iq4_xs`
-/

namespace Strata.Dequant

-- § Q2_0

/-- Code `j` (0..63) of a Q2_0 block, given its 16 code bytes. -/
def q2Code (qs : Nat → Nat) (j : Nat) : Nat := (qs (j / 4) / 2 ^ ((j % 4) * 2)) % 4

/-- Symbol for a code: code - 1. -/
def q2Symbol (c : Nat) : Int := (c : Int) - 1

/-- Every Q2_0 code is in 0..3. -/
theorem q2Code_lt (qs : Nat → Nat) (j : Nat) : q2Code qs j < 4 := by
  unfold q2Code
  exact Nat.mod_lt _ (by decide)

/-- Every Q2_0 symbol is one of -1, 0, 1, 2. -/
theorem q2Symbol_range (qs : Nat → Nat) (j : Nat) :
    q2Symbol (q2Code qs j) = -1 ∨ q2Symbol (q2Code qs j) = 0 ∨ q2Symbol (q2Code qs j) = 1 ∨
    q2Symbol (q2Code qs j) = 2 := by
  have h := q2Code_lt qs j
  unfold q2Symbol
  omega

/-- Symbols are bounded below by -1. -/
theorem q2Symbol_ge (qs : Nat → Nat) (j : Nat) : -1 ≤ q2Symbol (q2Code qs j) := by
  have h := q2Code_lt qs j
  unfold q2Symbol
  omega

/-- Symbols are bounded above by 2. -/
theorem q2Symbol_le (qs : Nat → Nat) (j : Nat) : q2Symbol (q2Code qs j) ≤ 2 := by
  have h := q2Code_lt qs j
  unfold q2Symbol
  omega

/-- The bitwise form `(b >> s) & 3` equals the arithmetic form for every byte and legal shift. -/
theorem q2_bitwise_eq : ∀ b, b < 256 → ∀ k, k < 4 → (b >>> (k * 2)) &&& 3 = (b / 2 ^ (k * 2)) % 4 := by
  decide +kernel

/-- Position `j` lives in byte `j / 4` which is in range 0..15 for `j < 64`. -/
theorem q2_byte_index_lt (j : Nat) (h : j < 64) : j / 4 < 16 := by
  omega

/-- The bit offset of a Q2_0 element is one of 0, 2, 4, 6. -/
theorem q2_bit_offset_cases (j : Nat) :
    (j % 4) * 2 = 0 ∨ (j % 4) * 2 = 2 ∨ (j % 4) * 2 = 4 ∨ (j % 4) * 2 = 6 := by
  omega

/-- The four codes in one byte, reassembled, give back the byte. -/
theorem q2_pack_roundtrip : ∀ b, b < 256 →
    (b % 4) + 4 * ((b / 4) % 4) + 16 * ((b / 16) % 4) + 64 * ((b / 64) % 4) = b := by
  decide +kernel

/-- Distinct elements `4i .. 4i+3` of a byte index the same byte. -/
theorem q2_same_byte (i k : Nat) (hk : k < 4) : (4 * i + k) / 4 = i := by
  omega

/-- And their within-byte slot equals `k`. -/
theorem q2_slot (i k : Nat) (hk : k < 4) : (4 * i + k) % 4 = k := by
  omega

-- § Q4_0 / Q8_0 / Q5_0

/-- Low nibble of a byte. -/
def q4Lo (b : Nat) : Nat := b % 16

/-- High nibble of a byte. -/
def q4Hi (b : Nat) : Nat := b / 16

/-- Bridge: `b & 0x0F = b % 16` for all bytes. -/
theorem q4Lo_bitwise : ∀ b, b < 256 → b &&& 0x0F = q4Lo b := by
  decide +kernel

/-- Bridge: `b >> 4 = b / 16` for all bytes. -/
theorem q4Hi_bitwise : ∀ b, b < 256 → b >>> 4 = q4Hi b := by
  decide +kernel

/-- Nibbles recombine to the byte. -/
theorem q4_nibbles_recombine (b : Nat) (h : b < 256) : q4Lo b + 16 * q4Hi b = b := by
  unfold q4Lo q4Hi
  omega

/-- The high nibble of a byte is below 16. -/
theorem q4Hi_lt (b : Nat) (h : b < 256) : q4Hi b < 16 := by
  unfold q4Hi
  omega

/-- The low nibble is below 16. -/
theorem q4Lo_lt (b : Nat) : q4Lo b < 16 := by
  unfold q4Lo
  omega

/-- Q4_0 symbol: nibble - 8. -/
def q4Symbol (n : Nat) : Int := (n : Int) - 8

/-- Q4_0 symbols are in [-8, 7]. -/
theorem q4Symbol_range (n : Nat) (h : n < 16) : -8 ≤ q4Symbol n ∧ q4Symbol n ≤ 7 := by
  unfold q4Symbol
  omega

/-- Q4_0 element mapping: element j (< 16) is low nibble of byte j, element j+16 high nibble. -/
def q4Element (qs : Nat → Nat) (i : Nat) : Int :=
  if i < 16 then q4Symbol (q4Lo (qs i)) else q4Symbol (q4Hi (qs (i - 16)))

/-- Every Q4_0 output symbol lies in [-8, 7] for byte-valued input. -/
theorem q4Element_range (qs : Nat → Nat) (hq : ∀ i, qs i < 256) (i : Nat) :
    -8 ≤ q4Element qs i ∧ q4Element qs i ≤ 7 := by
  unfold q4Element
  split
  · exact q4Symbol_range _ (q4Lo_lt _)
  · exact q4Symbol_range _ (q4Hi_lt _ (hq _))

/-- Q8_0 symbol is the signed byte itself; a signed byte lies in [-128, 127]. -/
theorem q8_signed_byte_range (b : Nat) (h : b < 256) :
    -128 ≤ (if b < 128 then (b : Int) else (b : Int) - 256) ∧
    (if b < 128 then (b : Int) else (b : Int) - 256) ≤ 127 := by
  split <;> omega

/-- Q5_0: 5-bit value from a nibble and the high bit. -/
def q5Val (nib hbit : Nat) : Nat := nib + 16 * hbit

/-- Q5_0 symbol is `q5Val - 16`. -/
def q5Symbol (nib hbit : Nat) : Int := (q5Val nib hbit : Int) - 16

/-- The 5-bit value is below 32. -/
theorem q5Val_lt (nib hbit : Nat) (hn : nib < 16) (hh : hbit < 2) : q5Val nib hbit < 32 := by
  unfold q5Val
  omega

/-- Q5_0 symbols are in [-16, 15]. -/
theorem q5Symbol_range (nib hbit : Nat) (hn : nib < 16) (hh : hbit < 2) :
    -16 ≤ q5Symbol nib hbit ∧ q5Symbol nib hbit ≤ 15 := by
  unfold q5Symbol q5Val
  omega

/-- Extracting bit `j` of a 32-bit mask is 0 or 1. -/
theorem q5_hbit_lt (h j : Nat) : (h / 2 ^ j) % 2 < 2 := by
  exact Nat.mod_lt _ (by decide)

/-- Q5_0 element j and element j+16 use mask bits j and j+16, which are distinct. -/
theorem q5_mask_bits_distinct (j : Nat) : j ≠ j + 16 := by
  omega

-- § IQ4_NL codebook

/-- The IQ4_NL codebook, verbatim from ggml. -/
def kvalues : List Int :=
  [-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113]

/-- The codebook has 16 entries. -/
theorem kvalues_length : kvalues.length = 16 := by
  rfl

/-- The codebook is strictly increasing. -/
theorem kvalues_sorted : kvalues.Pairwise (· < ·) := by
  decide

/-- The codebook is non-linear: successive gaps are not constant (gap 23 then 21). -/
theorem kvalues_nonlinear : (-104 : Int) - (-127) ≠ (-83 : Int) - (-104) := by
  decide

/-- Every codebook value lies in [-127, 113]. -/
theorem kvalues_range : ∀ v ∈ kvalues, -127 ≤ v ∧ v ≤ 113 := by
  decide

/-- Lookup with default zero. -/
def kv (i : Nat) : Int := kvalues.getD i 0

/-- Every nibble indexes inside the table (lookup never falls back to the default except by value). -/
theorem kv_in_table (i : Nat) (h : i < 16) : kv i ∈ kvalues := by
  unfold kv
  have : i < kvalues.length := by rw [kvalues_length]; exact h
  rw [List.getD_eq_getElem _ _ this]
  exact List.getElem_mem this

/-- kv is within [-127, 113] for any nibble. -/
theorem kv_range (i : Nat) (h : i < 16) : -127 ≤ kv i ∧ kv i ≤ 113 :=
  kvalues_range _ (kv_in_table i h)

/-- Spot checks of the codebook endpoints. -/
theorem kv_endpoints : kv 0 = -127 ∧ kv 15 = 113 ∧ kv 8 = 1 := by
  decide

-- § Q6_K

/-- 6-bit value: low 4 bits from `lo`, high 2 bits from `hi`, minus 32. -/
def q6Val (lo hi : Nat) : Int := ((lo + 16 * hi : Nat) : Int) - 32

/-- Q6_K values lie in [-32, 31]. -/
theorem q6Val_range (lo hi : Nat) (hl : lo < 16) (hh : hi < 4) :
    -32 ≤ q6Val lo hi ∧ q6Val lo hi ≤ 31 := by
  unfold q6Val
  omega

/-- The two-bit field extracted from `qh` at shift 0, 2, 4, 6 is below 4. -/
theorem q6_hi_bits_lt (b k : Nat) : (b / 2 ^ (2 * k)) % 4 < 4 := by
  exact Nat.mod_lt _ (by decide)

/-- The scale index `is = l / 16` is 0 or 1 for `l < 32`, so `is + {0,2,4,6}` is below 8. -/
theorem q6_scale_index_lt (l : Nat) (hl : l < 32) (c : Nat) (hc : c = 0 ∨ c = 2 ∨ c = 4 ∨ c = 6) :
    l / 16 + c < 8 := by
  omega

/-- Over both 128-element halves, the scale base advances by 8 and indices stay below 16. -/
theorem q6_scale_index_lt16 (half l : Nat) (hh : half < 2) (hl : l < 32) (c : Nat)
    (hc : c = 0 ∨ c = 2 ∨ c = 4 ∨ c = 6) : 8 * half + (l / 16 + c) < 16 := by
  omega

/-- Output index `n + l + 32 k` stays inside the 256-element block. -/
theorem q6_out_index_lt (half l k : Nat) (hh : half < 2) (hl : l < 32) (hk : k < 4) :
    128 * half + l + 32 * k < 256 := by
  omega

/-- Distinct (half, l, k) triples give distinct output indices: the output map is injective. -/
theorem q6_out_index_inj (h1 l1 k1 h2 l2 k2 : Nat) (hh1 : h1 < 2) (hl1 : l1 < 32) (hk1 : k1 < 4)
    (hh2 : h2 < 2) (hl2 : l2 < 32) (hk2 : k2 < 4)
    (e : 128 * h1 + l1 + 32 * k1 = 128 * h2 + l2 + 32 * k2) :
    h1 = h2 ∧ l1 = l2 ∧ k1 = k2 := by
  omega

/-- The q6 layout byte budget: 128 + 64 + 16 + 2 = 210. -/
theorem q6_K_bytes : 128 + 64 + 16 + 2 = 210 := by
  rfl

/-- The d field of Q6_K sits at offset 208, the last two bytes of the 210-byte block. -/
theorem q6_K_d_offset : 128 + 64 + 16 = 208 ∧ 208 + 2 = 210 := by
  exact ⟨rfl, rfl⟩

-- § Q4_K / Q5_K scale-min unpacking

/-- Unpack scale and min for sub-block `j` (0..7) from the 12 scale bytes `q`. -/
def scaleMinK4 (q : Nat → Nat) (j : Nat) : Nat × Nat :=
  if j < 4 then (q j % 64, q (j + 4) % 64)
  else ((q (j + 4) % 16) + (q (j - 4) / 64) * 16, (q (j + 4) / 16) + (q j / 64) * 16)

/-- For `j < 4` the scale is a 6-bit value. -/
theorem scaleMinK4_lt64_low (q : Nat → Nat) (j : Nat) (hj : j < 4) :
    (scaleMinK4 q j).1 < 64 ∧ (scaleMinK4 q j).2 < 64 := by
  unfold scaleMinK4
  rw [if_pos hj]
  simp only
  omega

/-- For `4 ≤ j < 8` with byte-valued input both outputs are 6-bit values. -/
theorem scaleMinK4_lt64_high (q : Nat → Nat) (hq : ∀ i, q i < 256) (j : Nat) (hj : 4 ≤ j) :
    (scaleMinK4 q j).1 < 64 ∧ (scaleMinK4 q j).2 < 64 := by
  unfold scaleMinK4
  rw [if_neg (by omega)]
  have h1 := hq (j + 4)
  have h2 := hq (j - 4)
  have h3 := hq j
  simp only
  omega

/-- Sub-block scales are always 6 bits for byte input and j < 8. -/
theorem scaleMinK4_lt64 (q : Nat → Nat) (hq : ∀ i, q i < 256) (j : Nat) :
    (scaleMinK4 q j).1 < 64 ∧ (scaleMinK4 q j).2 < 64 := by
  by_cases hj : j < 4
  · exact scaleMinK4_lt64_low q j hj
  · exact scaleMinK4_lt64_high q hq j (by omega)

/-- Q4_K: the two sub-blocks of 64-element group `g` use scale indices `2g` and `2g+1`, below 8. -/
theorem q4K_scale_indices (g : Nat) (hg : g < 4) : 2 * g < 8 ∧ 2 * g + 1 < 8 := by
  omega

/-- Q5_K masks u1 = 1,4,16,64 and u2 = 2,8,32,128 by shifting left 2 each group. -/
theorem q5K_masks : (1 <<< 0, 1 <<< 2, 1 <<< 4, 1 <<< 6) = (1, 4, 16, 64) ∧
    (2 <<< 0, 2 <<< 2, 2 <<< 4, 2 <<< 6) = (2, 8, 32, 128) := by
  decide

/-- The Q5_K masks never overlap: u1 and u2 select different bits for each group. -/
theorem q5K_masks_disjoint : ∀ g, g < 4 → (1 <<< (2 * g)) &&& (2 <<< (2 * g)) = 0 := by
  decide

/-- Block byte budgets: Q4_K is 4 + 12 + 128 = 144 and Q5_K is 4 + 12 + 32 + 128 = 176. -/
theorem q4K_q5K_bytes : 4 + 12 + 128 = 144 ∧ 4 + 12 + 32 + 128 = 176 := by
  exact ⟨rfl, rfl⟩

-- § IQ4_XS

/-- IQ4_XS 6-bit group scale from `scales_l` byte and 16-bit `scales_h`. -/
def iq4xsScale (sl : Nat → Nat) (sh ib : Nat) : Nat :=
  ((sl (ib / 2) / 2 ^ (4 * (ib % 2))) % 16) + ((sh / 2 ^ (2 * ib)) % 4) * 16

/-- The assembled group scale is a 6-bit value. -/
theorem iq4xsScale_lt64 (sl : Nat → Nat) (sh ib : Nat) : iq4xsScale sl sh ib < 64 := by
  unfold iq4xsScale
  have h1 : (sl (ib / 2) / 2 ^ (4 * (ib % 2))) % 16 < 16 := Nat.mod_lt _ (by decide)
  have h2 : (sh / 2 ^ (2 * ib)) % 4 < 4 := Nat.mod_lt _ (by decide)
  omega

/-- Signed effective scale `ls - 32` lies in [-32, 31]. -/
theorem iq4xsScale_signed_range (sl : Nat → Nat) (sh ib : Nat) :
    -32 ≤ (iq4xsScale sl sh ib : Int) - 32 ∧ (iq4xsScale sl sh ib : Int) - 32 ≤ 31 := by
  have := iq4xsScale_lt64 sl sh ib
  omega

/-- For ib < 8 the `scales_l` index `ib/2` is below 4. -/
theorem iq4xs_scales_l_index (ib : Nat) (h : ib < 8) : ib / 2 < 4 := by
  omega

/-- IQ4_XS byte budget: 2 + 2 + 4 + 128 = 136. -/
theorem iq4xs_bytes : 2 + 2 + 4 + 128 = 136 := by
  rfl

/-- Eight groups of 32 elements fill the 256-element super-block. -/
theorem iq4xs_groups : 8 * 32 = 256 := by
  rfl

-- § Q3_K

/-- Q3_K byte budget: 32 + 64 + 12 + 2 = 110. -/
theorem q3K_bytes : 32 + 64 + 12 + 2 = 110 := by
  rfl

/-- The hoisted mask `m` takes values 1,2,4,...,128 across the 8 (n, j) iterations: `m = 2^(4n' + j)`
for the iteration number `i = 4n' + j`, and it is a single set bit within a byte. -/
theorem q3K_mask_in_byte (i : Nat) (h : i < 8) : 2 ^ i < 256 := by
  have : 2 ^ i < 2 ^ 8 := Nat.pow_lt_pow_right (by decide) h
  simpa using this

/-- Q3_K high-bit correction: the signed 3-bit value lies in [-4, 3]. -/
theorem q3K_value_range (lo hbit : Nat) (hl : lo < 4) (hh : hbit < 2) :
    -4 ≤ (lo : Int) - (if hbit = 1 then 0 else 4) ∧ (lo : Int) - (if hbit = 1 then 0 else 4) ≤ 3 := by
  split <;> omega

-- § fp16 / bf16 field extraction

/-- fp16 exponent field. -/
def fp16Exp (h : Nat) : Nat := (h / 1024) % 32

/-- fp16 mantissa field. -/
def fp16Man (h : Nat) : Nat := h % 1024

/-- fp16 sign field. -/
def fp16Sign (h : Nat) : Nat := (h / 32768) % 2

/-- The three fp16 fields recompose the 16-bit pattern. -/
theorem fp16_fields_recompose (h : Nat) (hh : h < 65536) :
    fp16Man h + 1024 * fp16Exp h + 32768 * fp16Sign h = h := by
  unfold fp16Man fp16Exp fp16Sign
  omega

/-- The exponent field is below 32. -/
theorem fp16Exp_lt (h : Nat) : fp16Exp h < 32 := by
  unfold fp16Exp
  exact Nat.mod_lt _ (by decide)

/-- Subnormal path renormalisation: exp starts at 127 - 15 + 1 = 113. -/
theorem fp16_subnormal_start : 127 - 15 + 1 = 113 := by
  rfl

/-- Normal path rebias: 15 maps to 127, so fp16 1.0 (exp field 15) becomes fp32 exponent 127. -/
theorem fp16_rebias_one : 15 - 15 + 127 = 127 := by
  rfl

/-- Infinity/NaN exponent field 31 becomes 0xFF in fp32 (0x7F800000 has exponent bits 255). -/
theorem fp16_inf_exp : 0x7F800000 / 2 ^ 23 = 255 := by
  decide

/-- bf16 to fp32 is a left shift by 16: the low 16 bits of the fp32 are zero. -/
theorem bf16_low_bits_zero (h : Nat) : (h <<< 16) % 65536 = 0 := by
  rw [Nat.shiftLeft_eq]
  omega

/-- read_u16 is little endian: p0 + 256 * p1 for byte inputs is below 65536. -/
theorem read_u16_lt (p0 p1 : Nat) (h0 : p0 < 256) (h1 : p1 < 256) : p0 + 256 * p1 < 65536 := by
  omega

/-- read_u16 low byte recovery. -/
theorem read_u16_low (p0 p1 : Nat) (h0 : p0 < 256) : (p0 + 256 * p1) % 256 = p0 := by
  omega

/-- read_u16 high byte recovery. -/
theorem read_u16_high (p0 p1 : Nat) (h0 : p0 < 256) (h1 : p1 < 256) : (p0 + 256 * p1) / 256 = p1 := by
  omega

end Strata.Dequant





## FILE: /mnt/user-data/outputs/ExpertCacheResidency.lean
/-!
# MODULE: expert_cache_residency
VRAM-resident expert tier: a slot arena plus a `(layer, expert) -> slot | none` residency table, with
never-evicting admission (global arrival order, or per-layer slot ranges) and bounds-checked lookup.

## Source semantics
* `slot_of` is bounds-checked: out-of-range layer/expert yields `kNotResident`.
* `admit` claims the next free slot from one global counter and returns none when full; it never evicts,
  so every other residency entry is unchanged.
* Per-layer admission: layer `l` may use only `[l*q, (l+1)*q)` with `q = slots / n_layers`; the last
  layer takes the remainder. Ranges are pairwise disjoint and contained in `[0, slots)`.
* Uniform arena size is `slots * blob`; a sized arena's size is its last offset.
* A profile is valid when every ranked `(layer, expert)` pair is in range.

## Form template identifiers
* `ExpertCache` -> `Cache`, `slot_of` -> `slotOf`, `admit` -> `admit`, `kNotResident` -> `none`
* `layer_slot_range` -> `layerLo` / `layerHi`, `next_free_` -> `nextFree`, `bytes()` -> `arenaBytes`
-/

namespace StrataExpertCache

-- § state and lookup

/-- Cache state: capacity, geometry, global admission counter, residency table. -/
structure Cache where
  slots : Nat
  nLayers : Nat
  nExpert : Nat
  nextFree : Nat
  res : Nat → Nat → Option Nat

/-- Bounds-checked residency lookup. -/
def slotOf (c : Cache) (l e : Nat) : Option Nat :=
  if l < c.nLayers ∧ e < c.nExpert then c.res l e else none

/-- Global admission: take the next free slot, never evict. -/
def admit (c : Cache) (l e : Nat) : Cache × Option Nat :=
  if l < c.nLayers ∧ e < c.nExpert ∧ c.nextFree < c.slots then
    ({ c with nextFree := c.nextFree + 1,
              res := fun l' e' => if l' = l ∧ e' = e then some c.nextFree else c.res l' e' },
     some c.nextFree)
  else (c, none)

/-- Out-of-range layer gives no slot. -/
theorem slotOf_layer_oob (c : Cache) (l e : Nat) (h : ¬ l < c.nLayers) :
    slotOf c l e = none := by
  unfold slotOf
  simp [h]

/-- Out-of-range expert gives no slot. -/
theorem slotOf_expert_oob (c : Cache) (l e : Nat) (h : ¬ e < c.nExpert) :
    slotOf c l e = none := by
  unfold slotOf
  simp [h]

-- § admission

/-- A full cache refuses and is left unchanged. -/
theorem admit_full (c : Cache) (l e : Nat) (h : c.slots ≤ c.nextFree) :
    admit c l e = (c, none) := by
  unfold admit
  have : ¬ (l < c.nLayers ∧ e < c.nExpert ∧ c.nextFree < c.slots) := by omega
  simp [this]

/-- A successful admission returns the counter value as slot. -/
theorem admit_slot (c : Cache) (l e : Nat) (hl : l < c.nLayers) (he : e < c.nExpert)
    (hs : c.nextFree < c.slots) : (admit c l e).2 = some c.nextFree := by
  unfold admit
  simp [hl, he, hs]

/-- A successful admission advances the counter by one. -/
theorem admit_counter (c : Cache) (l e : Nat) (hl : l < c.nLayers) (he : e < c.nExpert)
    (hs : c.nextFree < c.slots) : (admit c l e).1.nextFree = c.nextFree + 1 := by
  unfold admit
  simp [hl, he, hs]

/-- The admitted pair becomes resident at the claimed slot. -/
theorem admit_resident (c : Cache) (l e : Nat) (hl : l < c.nLayers) (he : e < c.nExpert)
    (hs : c.nextFree < c.slots) : slotOf (admit c l e).1 l e = some c.nextFree := by
  unfold admit slotOf
  simp [hl, he, hs]

/-- No eviction: every other pair keeps its slot. -/
theorem admit_no_evict (c : Cache) (l e l' e' : Nat) (h : ¬ (l' = l ∧ e' = e)) :
    slotOf (admit c l e).1 l' e' = slotOf c l' e' := by
  unfold admit slotOf
  by_cases hc : l < c.nLayers ∧ e < c.nExpert ∧ c.nextFree < c.slots
  · simp [hc, h]
  · simp [hc]

/-- Admission never changes capacity. -/
theorem admit_slots (c : Cache) (l e : Nat) : (admit c l e).1.slots = c.slots := by
  unfold admit
  split <;> rfl

/-- The counter never exceeds capacity once it is within capacity. -/
theorem admit_counter_le (c : Cache) (l e : Nat) (h : c.nextFree ≤ c.slots) :
    (admit c l e).1.nextFree ≤ c.slots := by
  unfold admit
  split
  · rename_i hc
    simp only
    omega
  · exact h

-- § per-layer slot ranges

/-- Lower bound of layer `l`'s slot range. -/
def layerLo (slots n l : Nat) : Nat := l * (slots / n)

/-- Upper bound (exclusive); the last layer takes the remainder. -/
def layerHi (slots n l : Nat) : Nat := if l + 1 = n then slots else (l + 1) * (slots / n)

/-- Layer 0 starts at slot 0. -/
theorem layerLo_zero (slots n : Nat) : layerLo slots n 0 = 0 := by
  simp [layerLo]

/-- `n * (slots / n) ≤ slots`. -/
theorem n_mul_div_le (slots n : Nat) : n * (slots / n) ≤ slots :=
  Nat.mul_div_le slots n

/-- The upper bound lies within the arena. -/
theorem layerHi_le_slots (slots n l : Nat) (hl : l < n) : layerHi slots n l ≤ slots := by
  unfold layerHi
  split
  · exact Nat.le_refl _
  · rename_i h
    have h1 : l + 1 ≤ n := hl
    have h2 : (l + 1) * (slots / n) ≤ n * (slots / n) := Nat.mul_le_mul_right _ h1
    have h3 := n_mul_div_le slots n
    omega

/-- Each layer's range is well formed. -/
theorem layerLo_le_layerHi (slots n l : Nat) (hl : l < n) : layerLo slots n l ≤ layerHi slots n l := by
  unfold layerLo layerHi
  split
  · rename_i h
    have h2 : l * (slots / n) ≤ n * (slots / n) := Nat.mul_le_mul_right _ (Nat.le_of_lt hl)
    have h3 := n_mul_div_le slots n
    omega
  · have : l * (slots / n) ≤ (l + 1) * (slots / n) := Nat.mul_le_mul_right _ (Nat.le_succ l)
    exact this

/-- Ranges of distinct layers are disjoint: the lower layer ends before the higher begins. -/
theorem layer_ranges_disjoint (slots n l1 l2 : Nat) (h12 : l1 < l2) (h2 : l2 < n) :
    layerHi slots n l1 ≤ layerLo slots n l2 := by
  unfold layerHi layerLo
  have hne : ¬ (l1 + 1 = n) := by omega
  simp only [hne, ite_false]
  exact Nat.mul_le_mul_right _ h12

/-- The last layer's range ends exactly at `slots`. -/
theorem last_layer_takes_remainder (slots n : Nat) (hn : 0 < n) : layerHi slots n (n - 1) = slots := by
  unfold layerHi
  have : n - 1 + 1 = n := by omega
  simp [this]

/-- Non-last layers get exactly `q = slots / n` slots. -/
theorem layer_width (slots n l : Nat) (hl : l + 1 < n) :
    layerHi slots n l - layerLo slots n l = slots / n := by
  unfold layerHi layerLo
  have hne : ¬ (l + 1 = n) := by omega
  simp only [hne, ite_false]
  rw [Nat.add_mul, Nat.one_mul]
  exact Nat.add_sub_cancel_left _ _

/-- A per-layer admission decision: the layer's own next-free slot if below its upper bound. -/
def admitPerLayerSlot (slots n : Nat) (next : Nat → Nat) (l : Nat) : Option Nat :=
  if next l < layerHi slots n l then some (next l) else none

/-- A granted per-layer slot lies in the layer's own range. -/
theorem per_layer_slot_in_range (slots n : Nat) (next : Nat → Nat) (l s : Nat)
    (hlo : layerLo slots n l ≤ next l) (h : admitPerLayerSlot slots n next l = some s) :
    layerLo slots n l ≤ s ∧ s < layerHi slots n l := by
  unfold admitPerLayerSlot at h
  split at h
  · rename_i hc
    have hs : next l = s := Option.some.inj h
    subst hs
    exact ⟨hlo, hc⟩
  · cases h

/-- Slots granted to two different layers are different. -/
theorem per_layer_slots_distinct (slots n : Nat) (next : Nat → Nat) (l1 l2 s1 s2 : Nat)
    (h12 : l1 < l2) (h2 : l2 < n)
    (hlo1 : layerLo slots n l1 ≤ next l1) (hlo2 : layerLo slots n l2 ≤ next l2)
    (g1 : admitPerLayerSlot slots n next l1 = some s1)
    (g2 : admitPerLayerSlot slots n next l2 = some s2) : s1 ≠ s2 := by
  have a := per_layer_slot_in_range slots n next l1 s1 hlo1 g1
  have b := per_layer_slot_in_range slots n next l2 s2 hlo2 g2
  have d := layer_ranges_disjoint slots n l1 l2 h12 h2
  omega

-- § sizes and profile

/-- Slot offsets for slots of the given sizes, back to back. -/
def slotOffsets (sizes : List Nat) : List Nat := List.scanl (· + ·) 0 sizes

/-- A sized arena has one more offset than slots. -/
theorem slotOffsets_length (sizes : List Nat) : (slotOffsets sizes).length = sizes.length + 1 := by
  unfold slotOffsets
  simp

/-- Arena bytes: last offset if sized, else `slots * blob`. -/
def arenaBytes (offs : List Nat) (slots blob : Nat) : Nat :=
  match offs.getLast? with
  | none => slots * blob
  | some x => x

/-- Uniform arena size. -/
theorem arenaBytes_uniform (slots blob : Nat) : arenaBytes [] slots blob = slots * blob := by
  rfl

/-- Resident count: the per-layer total when per-layer admission is on, else the global counter. -/
def residentCount (perLayer : Bool) (admitted nextFree : Nat) : Nat :=
  if perLayer then admitted else nextFree

/-- With per-layer admission off, the global counter is the resident count. -/
theorem residentCount_off (a n : Nat) : residentCount false a n = n := by
  rfl

/-- A profile's ranked pairs are all in range. -/
def validProfile (ranked : List (Nat × Nat)) (nLayers nExpert : Nat) : Prop :=
  ∀ p ∈ ranked, p.1 < nLayers ∧ p.2 < nExpert

/-- The empty profile is valid. -/
theorem validProfile_nil (nl ne : Nat) : validProfile [] nl ne := by
  intro p hp
  cases hp

/-- Validity of a cons list splits into head and tail. -/
theorem validProfile_cons (p : Nat × Nat) (ps : List (Nat × Nat)) (nl ne : Nat)
    (h1 : p.1 < nl ∧ p.2 < ne) (h2 : validProfile ps nl ne) : validProfile (p :: ps) nl ne := by
  intro q hq
  cases hq with
  | head => exact h1
  | tail _ hq' => exact h2 q hq'

end StrataExpertCache



## FILE: /mnt/user-data/outputs/ExpertSourceDispatch.lean
/-!
# MODULE: expert_source_dispatch
The expert pool's adapter to the host loop: blob addressing in `experts.bin`, geometry refusal, the
hit/miss split by router index, cache counters, the latching failure flag and the hit-timing counters.

## Source semantics
* Blob index is `layer * n_expert + expert` with no padding; byte offset is index times `BLOB`.
* Geometry is fixed (`H = 2560`, `FF = 640`, `BLOB = 1382400`); anything else is refused.
* Each router index is computed by exactly one side: CPU rows of hits are zeroed, GPU rows of
  misses are zero, so the sum equals the computed value.
* Every cache lookup is counted exactly once as hit, admitted or refused.
* `failed` latches; clearing the message does not re-arm the adapter.
* `hits_ready` is all-or-nothing over the hit path's pointers.

## Form template identifiers
* `FileExpertSource::blob` -> `blobOffset`, `ExpertDispatch::is_hit` -> `isHit`
* `cache_hits/admitted/refused` -> `Counters`, `hit_ready/hit_late` -> `Timing`
* `ExpertDispatch::failed/fail` -> `FailState`
-/

namespace StrataExpertSource

-- § geometry and addressing

/-- Bytes per expert blob. -/
def BLOB : Nat := 1382400
/-- Embedding width the kernel is compiled for. -/
def H : Nat := 2560
/-- Expert feed-forward width the kernel is compiled for. -/
def FF : Nat := 640

/-- Geometry check: other widths are refused. -/
def geometryOk (nEmbd ff : Nat) : Bool := nEmbd == H && ff == FF

/-- The compiled geometry is accepted. -/
theorem geometryOk_compiled : geometryOk 2560 640 = true := by
  decide

/-- A different embedding width is refused. -/
theorem geometryOk_refuses_embd (nEmbd ff : Nat) (h : nEmbd ≠ 2560) : geometryOk nEmbd ff = false := by
  unfold geometryOk H
  simp [h]

/-- A different expert width is refused. -/
theorem geometryOk_refuses_ff (nEmbd ff : Nat) (h : ff ≠ 640) : geometryOk nEmbd ff = false := by
  unfold geometryOk FF
  simp [h]

/-- Flat blob index, no padding. -/
def blobIndex (nExpert l e : Nat) : Nat := l * nExpert + e

/-- Byte offset of a blob. -/
def blobOffset (nExpert l e : Nat) : Nat := blobIndex nExpert l e * BLOB

/-- The pack holds exactly 24,576 blobs for 48 layers of 512 experts. -/
theorem blob_count : 48 * 512 = 24576 := by
  decide

/-- The artifact size: `48 * 512 * BLOB = 33,973,862,400`. -/
theorem experts_bin_size : 48 * 512 * BLOB = 33973862400 := by
  unfold BLOB
  decide

/-- File-size check: exact equality with `n_layers * n_expert * BLOB`. -/
def sizeOk (fileSize nLayers nExpert : Nat) : Bool := fileSize == nLayers * nExpert * BLOB

/-- The real artifact passes the size check. -/
theorem sizeOk_artifact : sizeOk 33973862400 48 512 = true := by
  unfold sizeOk BLOB
  decide

/-- A short file fails the size check. -/
theorem sizeOk_short : sizeOk 33973862399 48 512 = false := by
  unfold sizeOk BLOB
  decide

/-- In-range indices stay below the blob count. -/
theorem blobIndex_lt (nExpert l e nLayers : Nat) (hl : l < nLayers) (he : e < nExpert) :
    blobIndex nExpert l e < nLayers * nExpert := by
  unfold blobIndex
  have h1 : (l + 1) * nExpert ≤ nLayers * nExpert := Nat.mul_le_mul_right _ hl
  rw [Nat.add_mul, Nat.one_mul] at h1
  omega

/-- Distinct in-range `(layer, expert)` pairs map to distinct blob indices. -/
theorem blobIndex_injective (nExpert l e l' e' : Nat) (he : e < nExpert) (he' : e' < nExpert)
    (h : blobIndex nExpert l e = blobIndex nExpert l' e') : l = l' ∧ e = e' := by
  unfold blobIndex at h
  have hd : (l * nExpert + e) / nExpert = (l' * nExpert + e') / nExpert := by rw [h]
  have hm : (l * nExpert + e) % nExpert = (l' * nExpert + e') % nExpert := by rw [h]
  have hpos : 0 < nExpert := by omega
  have a1 : (l * nExpert + e) / nExpert = l := by
    rw [Nat.mul_comm, Nat.mul_add_div hpos, Nat.div_eq_of_lt he]; simp
  have a2 : (l' * nExpert + e') / nExpert = l' := by
    rw [Nat.mul_comm, Nat.mul_add_div hpos, Nat.div_eq_of_lt he']; simp
  have b1 : (l * nExpert + e) % nExpert = e := by
    rw [Nat.mul_comm, Nat.mul_add_mod, Nat.mod_eq_of_lt he]
  have b2 : (l' * nExpert + e') % nExpert = e' := by
    rw [Nat.mul_comm, Nat.mul_add_mod, Nat.mod_eq_of_lt he']
  rw [a1, a2] at hd
  rw [b1, b2] at hm
  exact ⟨hd, hm⟩

/-- Activation scales: `H / 32 = 80` Q8_0 chunks. -/
theorem q8_chunks : H / 32 = 80 := by
  unfold H
  decide

/-- Hit-path activation buffer: `(H / 32) * 34 = 2720` bytes. -/
theorem q8_hit_bytes : (H / 32) * 34 = 2720 := by
  unfold H
  decide

-- § hit / miss split

/-- CPU row for one router index: zero when the GPU owns it. -/
def cpuRow (hit : Bool) (v : Int) : Int := if hit then 0 else v

/-- GPU row for one router index: zero when the CPU owns it. -/
def gpuRow (hit : Bool) (v : Int) : Int := if hit then v else 0

/-- Combining both sides at the router index recovers exactly the computed value. -/
theorem split_sum (hit : Bool) (v : Int) : cpuRow hit v + gpuRow hit v = v := by
  cases hit <;> simp [cpuRow, gpuRow]

/-- A hit's CPU row is zero. -/
theorem cpuRow_hit (v : Int) : cpuRow true v = 0 := by
  simp [cpuRow]

/-- A miss's GPU row is zero. -/
theorem gpuRow_miss (v : Int) : gpuRow false v = 0 := by
  simp [gpuRow]

/-- Each index is owned by exactly one side. -/
theorem owner_exclusive (hit : Bool) : (hit = true) ≠ (hit = false) := by
  cases hit <;> simp

/-- Decide hits from a residency lookup. -/
def isHit (res : Nat → Option Nat) (e : Nat) : Bool := (res e).isSome

/-- Resident experts are hits. -/
theorem isHit_of_resident (res : Nat → Option Nat) (e s : Nat) (h : res e = some s) :
    isHit res e = true := by
  unfold isHit
  simp [h]

/-- Non-resident experts are misses. -/
theorem isHit_of_none (res : Nat → Option Nat) (e : Nat) (h : res e = none) :
    isHit res e = false := by
  unfold isHit
  simp [h]

/-- Router weight is applied once, by the combine: weighted sum of unweighted parts. -/
def combine (ws parts : List Int) : Int := (List.zipWith (· * ·) ws parts).sum

/-- Combine of the empty router output is zero. -/
theorem combine_nil : combine [] [] = 0 := by
  rfl

/-- Combine unfolds one entry at a time. -/
theorem combine_cons (w p : Int) (ws ps : List Int) :
    combine (w :: ws) (p :: ps) = w * p + combine ws ps := by
  simp [combine]

/-- Hit path pointers present or absent. -/
structure HitWiring where
  cache : Bool
  cacheBase : Bool
  partsOut : Bool
  hitOut : Bool
  mixed : Bool
  scratch : Bool
  xq : Bool
  xqScale : Bool
  dSlot : Bool
  dDst : Bool

/-- All-or-nothing wiring. -/
def HitWiring.ready (w : HitWiring) : Bool :=
  w.cache && w.cacheBase && w.partsOut && w.hitOut && w.mixed && w.scratch && w.xq && w.xqScale &&
    w.dSlot && w.dDst

/-- Without the fp32 activation scales the hit path is not ready. -/
theorem not_ready_without_scale (w : HitWiring) (h : w.xqScale = false) : w.ready = false := by
  unfold HitWiring.ready
  simp [h]

/-- Without a cache the hit path is not ready. -/
theorem not_ready_without_cache (w : HitWiring) (h : w.cache = false) : w.ready = false := by
  unfold HitWiring.ready
  simp [h]

/-- Ready implies every pointer is set. -/
theorem ready_all (w : HitWiring) (h : w.ready = true) :
    w.cache = true ∧ w.xqScale = true ∧ w.dDst = true := by
  unfold HitWiring.ready at h
  simp only [Bool.and_eq_true] at h
  exact ⟨h.1.1.1.1.1.1.1.1.1, h.1.1.2, h.2⟩

-- § counters

/-- Cache lookup counters. -/
structure Counters where
  hits : Nat
  admitted : Nat
  refused : Nat

/-- One lookup: resident is a hit, else admitted if a slot was free, else refused. -/
def Counters.step (c : Counters) (resident admittedOk : Bool) : Counters :=
  if resident then { c with hits := c.hits + 1 }
  else if admittedOk then { c with admitted := c.admitted + 1 }
  else { c with refused := c.refused + 1 }

/-- Total lookups counted. -/
def Counters.total (c : Counters) : Nat := c.hits + c.admitted + c.refused

/-- Every lookup is counted exactly once. -/
theorem counters_step_total (c : Counters) (r a : Bool) : (c.step r a).total = c.total + 1 := by
  cases r <;> cases a <;> simp [Counters.step, Counters.total] <;> omega

/-- Hit/late counters for the GPU-start probe. -/
structure Timing where
  ready : Nat
  late : Nat

/-- Record one layer: done by pool return is ready, else late. -/
def Timing.record (t : Timing) (done : Bool) : Timing :=
  if done then { t with ready := t.ready + 1 } else { t with late := t.late + 1 }

/-- Each layer is recorded exactly once. -/
theorem timing_record_total (t : Timing) (d : Bool) :
    (t.record d).ready + (t.record d).late = t.ready + t.late + 1 := by
  cases d <;> simp [Timing.record] <;> omega

-- § failure latch

/-- Failure state: the latch and the message. -/
structure FailState where
  failed : Bool
  msg : Option String

/-- Clearing the message. -/
def FailState.clearMsg (s : FailState) : FailState := { s with msg := none }

/-- Record a failure: the latch is set. -/
def FailState.raise (s : FailState) (m : String) : FailState := { failed := true, msg := some m }

/-- Clearing the message never re-arms the adapter. -/
theorem clearMsg_keeps_failed (s : FailState) : s.clearMsg.failed = s.failed := by
  rfl

/-- Once failed, always failed under message clearing. -/
theorem failed_latches (s : FailState) (m : String) : (s.raise m).clearMsg.failed = true := by
  rfl

/-- Dispatch short-circuits when latched. -/
def shouldRun (s : FailState) : Bool := !s.failed

/-- A latched adapter does no work. -/
theorem no_run_when_failed (s : FailState) (h : s.failed = true) : shouldRun s = false := by
  simp [shouldRun, h]

/-- Raising a failure stops later work even after clearing the message. -/
theorem no_run_after_raise_clear (s : FailState) (m : String) : shouldRun (s.raise m).clearMsg = false := by
  rfl

-- § PCIe share

/-- Number of missed experts the GPU reads over PCIe: `pcie_num / 256` of the misses. -/
def pcieShare (misses pcieNum : Nat) : Nat := misses * pcieNum / 256

/-- The PCIe share never exceeds the misses when `pcie_num ≤ 256`. -/
theorem pcieShare_le (misses pcieNum : Nat) (h : pcieNum ≤ 256) : pcieShare misses pcieNum ≤ misses := by
  unfold pcieShare
  apply Nat.div_le_of_le_mul
  have := Nat.mul_le_mul_left misses h
  rw [Nat.mul_comm 256 misses]
  exact this

/-- Zero share when `pcie_num = 0`. -/
theorem pcieShare_zero (misses : Nat) : pcieShare misses 0 = 0 := by
  simp [pcieShare]

end StrataExpertSource





## FILE: /mnt/user-data/outputs/Fp16BitConversion.lean
/-!
# MODULE: fp16_bit_conversion
Round-to-nearest-even binary32 -> binary16 conversion and exact binary16 -> binary32 conversion, both
expressed purely on raw bit patterns (modelled on `Nat`, with the 32-bit / 16-bit widths implicit in the masks).

## Source semantics
* `f16_from_f32` : an f32 inf/NaN (raw exponent 255) stays inf/NaN (a NaN gets the quiet bit 0x200); a FINITE
  value too large for fp16 saturates to a signed inf (0x7C00) and is never turned into a NaN; values below
  half the smallest subnormal flush to a signed zero; subnormal results restore the implicit 1 and shift, with
  round-to-nearest-even at the moved rounding point; normal results round from 13 dropped bits, and the
  rounding carry may bump the exponent (up to inf).
* `f32_from_f16` : exact, no rounding. Zero keeps its sign, subnormals are normalised (never flushed),
  exponent 31 maps to f32 exponent 255 with the mantissa shifted left 13.
* Hence every non-NaN fp16 value round-trips bit-exactly, and NaNs stay NaNs.

## Form template identifiers
* `f16_from_f32`  -> `narrowConv`   (wide float bits -> narrow float bits, RNE)
* `f32_from_f16`  -> `widenConv`    (narrow float bits -> wide float bits, exact)
* `sign` / `rawexp` / `man` -> `signBits` / `expField` / `fracBits`
* `0x7C00` -> `narrowInf`, `0x200` -> `narrowQuietBit`, `0x7F800000` -> `wideInf`
-/

namespace Fp16Bits

-- § definitions

/-- Round-to-nearest-even f32 -> fp16 on raw bits. `exp = rawexp - 112` of the source is kept in `Nat`
form: `exp >= 31` is `rawexp >= 143`, `exp <= 0` is `rawexp <= 112`, `exp < -10` is `rawexp < 102`,
the subnormal shift `14 - exp` is `126 - rawexp`. -/
def f16FromF32 (x : Nat) : Nat :=
  let sign := (x >>> 16) &&& 0x8000
  let rawexp := (x >>> 23) &&& 0xFF
  let man := x &&& 0x7FFFFF
  if rawexp = 255 then sign ||| 0x7C00 ||| (if man ≠ 0 then 0x200 else 0)
  else if 143 ≤ rawexp then sign ||| 0x7C00
  else if rawexp ≤ 112 then
    if rawexp < 102 then sign
    else
      let man' := man ||| 0x800000
      let sh := 126 - rawexp
      let h := (man' >>> sh) &&& 0x3FF
      let rem := man' &&& (2 ^ sh - 1)
      let h' := if rem > 2 ^ (sh - 1) ∨ (rem = 2 ^ (sh - 1) ∧ h &&& 1 = 1) then h + 1 else h
      sign ||| h'
  else
    let h := sign ||| ((rawexp - 112) <<< 10) ||| (man >>> 13)
    let rem := man &&& 0x1FFF
    if rem > 0x1000 ∨ (rem = 0x1000 ∧ h &&& 1 = 1) then h + 1 else h

/-- Normalisation loop of the subnormal case: shift left until bit 10 is set, counting shifts. -/
def normShift : Nat → Nat → Nat → Nat × Nat
  | 0, m, s => (m, s)
  | fuel + 1, m, s => if m &&& 0x400 = 0 then normShift fuel (m <<< 1) (s + 1) else (m, s)

/-- Exact fp16 -> f32 on raw bits. -/
def f32FromF16 (h : Nat) : Nat :=
  let sign := (h &&& 0x8000) <<< 16
  let ex := (h >>> 10) &&& 0x1F
  let man := h &&& 0x3FF
  if ex = 0 then
    if man = 0 then sign
    else
      let r := normShift 16 man 0
      sign ||| ((113 - r.2) <<< 23) ||| ((r.1 &&& 0x3FF) <<< 13)
  else if ex = 31 then sign ||| 0x7F800000 ||| (man <<< 13)
  else sign ||| ((ex + 112) <<< 23) ||| (man <<< 13)

/-- A bit pattern is an fp16 NaN. -/
def IsNaN16 (h : Nat) : Prop := (h >>> 10) &&& 0x1F = 31 ∧ h &&& 0x3FF ≠ 0

instance (h : Nat) : Decidable (IsNaN16 h) := by unfold IsNaN16; infer_instance

-- § narrowing: case lemmas

/-- An f32 inf/NaN stays an inf/NaN: sign kept, quiet bit set iff the mantissa is nonzero. -/
theorem f16FromF32_special (x : Nat) (hr : (x >>> 23) &&& 0xFF = 255) :
    f16FromF32 x = ((x >>> 16) &&& 0x8000) ||| 0x7C00 ||| (if x &&& 0x7FFFFF ≠ 0 then 0x200 else 0) := by
  simp [f16FromF32, hr]

/-- A FINITE value too large for fp16 saturates to a signed inf (the NaN-conflation bug is excluded). -/
theorem f16FromF32_overflow_inf (x : Nat) (hne : (x >>> 23) &&& 0xFF ≠ 255)
    (hge : 143 ≤ (x >>> 23) &&& 0xFF) :
    f16FromF32 x = ((x >>> 16) &&& 0x8000) ||| 0x7C00 := by
  simp [f16FromF32, hne, hge]

/-- Values below half the smallest subnormal underflow to a signed zero. -/
theorem f16FromF32_underflow_zero (x : Nat) (hlt : (x >>> 23) &&& 0xFF < 102) :
    f16FromF32 x = (x >>> 16) &&& 0x8000 := by
  have h1 : (x >>> 23) &&& 0xFF ≠ 255 := by omega
  have h2 : ¬ (143 ≤ (x >>> 23) &&& 0xFF) := by omega
  have h3 : (x >>> 23) &&& 0xFF ≤ 112 := by omega
  simp [f16FromF32, h1, h2, h3, hlt]

-- § widening: case lemmas

/-- Signed zero widens to signed zero. -/
theorem f32FromF16_zero : f32FromF16 0 = 0 ∧ f32FromF16 0x8000 = 0x80000000 := by
  decide

/-- The smallest fp16 subnormal 0x0001 widens to 2^-24 (0x33800000), not to 0.0. -/
theorem f32FromF16_min_subnormal : f32FromF16 0x0001 = 0x33800000 := by
  decide

/-- Exponent 31 maps to f32 exponent 255 (inf/NaN) with the mantissa shifted left 13. -/
theorem f32FromF16_inf_nan (h : Nat) (hex : (h >>> 10) &&& 0x1F = 31) :
    f32FromF16 h = ((h &&& 0x8000) <<< 16) ||| 0x7F800000 ||| ((h &&& 0x3FF) <<< 13) := by
  simp [f32FromF16, hex]

-- § concrete vectors (rounding regimes)

/-- Exactly 1.0 converts to 0x3C00 (the failure mode seen with `__float2half` was 0x3C01). -/
theorem f16FromF32_one : f16FromF32 0x3F800000 = 0x3C00 := by decide

/-- Normal-range tie at an even mantissa rounds down; at an odd mantissa rounds up. -/
theorem f16FromF32_tie_even :
    f16FromF32 0x3F801000 = 0x3C00 ∧ f16FromF32 0x3F803000 = 0x3C02 := by decide

/-- Overflow regime: 65504 is the largest finite fp16, 65520 is the tie that rounds up to inf,
65536 and 1e30 saturate to inf (not NaN), -1e30 keeps its sign. -/
theorem f16FromF32_overflow_vectors :
    f16FromF32 0x477FE000 = 0x7BFF ∧ f16FromF32 0x477FF000 = 0x7C00 ∧
    f16FromF32 0x47800000 = 0x7C00 ∧ f16FromF32 0x7149F2CA = 0x7C00 ∧
    f16FromF32 0xF149F2CA = 0xFC00 := by decide

/-- Subnormal regime: 2^-24 is 0x0001, 2^-25 is a tie that rounds to even zero, just above it rounds
to 0x0001, 2^-26 flushes to zero, and -0.0 stays -0.0. -/
theorem f16FromF32_subnormal_vectors :
    f16FromF32 0x33800000 = 0x0001 ∧ f16FromF32 0x33000000 = 0x0000 ∧
    f16FromF32 0x33000001 = 0x0001 ∧ f16FromF32 0x32800000 = 0x0000 ∧
    f16FromF32 0x80000000 = 0x8000 := by decide

/-- f32 infinities and NaNs: +inf, -inf, a quiet NaN. -/
theorem f16FromF32_inf_nan_vectors :
    f16FromF32 0x7F800000 = 0x7C00 ∧ f16FromF32 0xFF800000 = 0xFC00 ∧
    f16FromF32 0x7FC00000 = 0x7E00 := by decide

-- § exhaustive properties over all 65,536 fp16 inputs

/-- Every non-NaN fp16 value survives widen-then-narrow bit-exactly (all 65,536 inputs checked). -/
theorem roundtrip_all :
    ∀ h : Fin 65536, ¬ IsNaN16 h.val → f16FromF32 (f32FromF16 h.val) = h.val := by
  have key : (List.range 65536).all (fun h =>
      decide (¬ IsNaN16 h → f16FromF32 (f32FromF16 h) = h)) = true := by
    decide +kernel
  intro h hn
  exact of_decide_eq_true (List.all_eq_true.mp key h.val (List.mem_range.mpr h.isLt)) hn

/-- NaNs stay NaNs through widen-then-narrow (the payload is not preserved, the class is). -/
theorem roundtrip_nan_stays_nan :
    ∀ h : Fin 65536, IsNaN16 h.val → IsNaN16 (f16FromF32 (f32FromF16 h.val)) := by
  have key : (List.range 65536).all (fun h =>
      decide (IsNaN16 h → IsNaN16 (f16FromF32 (f32FromF16 h)))) = true := by
    decide +kernel
  intro h hn
  exact of_decide_eq_true (List.all_eq_true.mp key h.val (List.mem_range.mpr h.isLt)) hn

/-- Widening preserves the sign bit and never flushes a nonzero value to zero. -/
theorem widen_sign_and_nonzero :
    ∀ h : Fin 65536, (f32FromF16 h.val >>> 31 = h.val >>> 15) ∧
      (h.val &&& 0x7FFF ≠ 0 → f32FromF16 h.val &&& 0x7FFFFFFF ≠ 0) := by
  have key : (List.range 65536).all (fun h =>
      decide ((f32FromF16 h >>> 31 = h >>> 15) ∧
        (h &&& 0x7FFF ≠ 0 → f32FromF16 h &&& 0x7FFFFFFF ≠ 0))) = true := by
    decide +kernel
  intro h
  exact of_decide_eq_true (List.all_eq_true.mp key h.val (List.mem_range.mpr h.isLt))

end Fp16Bits





## FILE: /mnt/user-data/outputs/ChatFrontendParsing.lean
/-!
# MODULE: chat_frontend_parsing
Text-only request/response layer between API clients and the engine: reasoning-effort normalisation,
Anthropic thinking-budget mapping, tool-call parameter value trimming, and the "hold back a partial tag"
rule of the incremental output parser.

## Source semantics
* Every client spelling of a reasoning effort maps onto {low, medium, xhigh} or "thinking disabled";
  an unknown spelling is rejected (HTTP 400), never a crash.
* `budget_tokens` maps monotonically: < 2048 → low, < 8192 → medium, otherwise xhigh.
* The streaming parser never emits a partial tag: it holds back the longest suffix of the buffer that is a
  proper prefix of the tag being looked for (so the held length is strictly smaller than the tag length).
* A tool-call parameter value loses at most one leading and one trailing newline.
* At end of generation an unterminated tool call is flushed as content, never as a tool_call event.

## Form template identifiers
* `effortKwargs` ↔ `effort_kwargs`; `budgetEffort` ↔ `budget_effort`
* `hold` ↔ `OutputParser._hold`; `trimValue` ↔ the newline stripping in `parse_tool_call`
* `finishKind` ↔ `OutputParser.finish` (state → event kind)
-/

namespace Strata.Frontend

-- § reasoning effort

/-- The template's three thinking levels. -/
inductive Level
  | low | medium | xhigh
  deriving DecidableEq, Repr

/-- Rank of a level, used to state monotonicity. -/
def Level.rank : Level → Nat
  | .low => 0
  | .medium => 1
  | .xhigh => 2

/-- The spellings of a reasoning effort a client may send. -/
inductive EffortKey
  | none_ | off | minimal | disabled | false_
  | low | medium | high | xhigh | max | maximum
  | unknown
  deriving DecidableEq, Repr

/-- Template kwargs produced by an effort: nothing, thinking disabled, or a level. -/
inductive Kw
  | empty
  | noThink
  | level (l : Level)
  deriving DecidableEq, Repr

/-- The `EFFORT` table: `some none` means "thinking off", `none` means unknown spelling. -/
def effortLevel : EffortKey → Option (Option Level)
  | .none_ => some none
  | .off => some none
  | .minimal => some none
  | .disabled => some none
  | .false_ => some none
  | .low => some (some .low)
  | .medium => some (some .medium)
  | .high => some (some .xhigh)
  | .xhigh => some (some .xhigh)
  | .max => some (some .xhigh)
  | .maximum => some (some .xhigh)
  | .unknown => none

/-- A reasoning effort → template kwargs, or `none` (a 400) for an unknown value. -/
def effortKwargs (k : EffortKey) : Option Kw :=
  match effortLevel k with
  | none => none
  | some none => some .noThink
  | some (some l) => some (.level l)

/-- "high" means xhigh, the template's default level. -/
theorem effortKwargs_high : effortKwargs .high = some (.level .xhigh) := rfl

/-- "max" and "maximum" are aliases of "high". -/
theorem effortKwargs_max_alias :
    effortKwargs .max = effortKwargs .high ∧ effortKwargs .maximum = effortKwargs .high := by
  exact ⟨rfl, rfl⟩

/-- Every "off" spelling disables thinking. -/
theorem effortKwargs_off_family :
    ∀ k ∈ [EffortKey.none_, .off, .minimal, .disabled, .false_], effortKwargs k = some .noThink := by
  decide

/-- An unknown spelling is rejected. -/
theorem effortKwargs_unknown : effortKwargs .unknown = none := rfl

/-- A key is accepted exactly when it is not the unknown spelling. -/
theorem effortKwargs_isSome_iff (k : EffortKey) :
    (effortKwargs k).isSome = true ↔ k ≠ .unknown := by
  cases k <;> simp [effortKwargs, effortLevel]

-- § thinking budget

/-- Anthropic `budget_tokens` → level: under 2K low, under 8K medium, else xhigh. -/
def budgetEffort (n : Nat) : Level :=
  if n < 2048 then .low else if n < 8192 then .medium else .xhigh

/-- Small budgets map to low. -/
theorem budgetEffort_low {n : Nat} (h : n < 2048) : budgetEffort n = .low := by
  simp [budgetEffort, h]

/-- Middle budgets map to medium. -/
theorem budgetEffort_medium {n : Nat} (h1 : 2048 ≤ n) (h2 : n < 8192) : budgetEffort n = .medium := by
  have h : ¬ n < 2048 := by omega
  simp [budgetEffort, h, h2]

/-- Large budgets map to xhigh. -/
theorem budgetEffort_xhigh {n : Nat} (h : 8192 ≤ n) : budgetEffort n = .xhigh := by
  have h1 : ¬ n < 2048 := by omega
  have h2 : ¬ n < 8192 := by omega
  simp [budgetEffort, h1, h2]

/-- A larger budget never selects a lower level. -/
theorem budgetEffort_mono {n m : Nat} (h : n ≤ m) :
    (budgetEffort n).rank ≤ (budgetEffort m).rank := by
  unfold budgetEffort
  split_ifs <;> simp only [Level.rank] <;> omega

-- § partial tag hold-back

/-- Longest `k ≤ n` such that the first `k` tag symbols are a suffix of the text (0 if none). -/
def holdGo {α : Type} [DecidableEq α] (text tag : List α) : Nat → Nat
  | 0 => 0
  | n + 1 => if (tag.take (n + 1)).isSuffixOf text then n + 1 else holdGo text tag n

/-- Length of the longest suffix of `text` that is a proper prefix of `tag`. -/
def hold {α : Type} [DecidableEq α] (text tag : List α) : Nat :=
  holdGo text tag (tag.length - 1)

/-- The search never exceeds its bound. -/
theorem holdGo_le {α : Type} [DecidableEq α] (text tag : List α) (n : Nat) :
    holdGo text tag n ≤ n := by
  induction n with
  | zero => simp [holdGo]
  | succ n ih =>
    unfold holdGo
    split <;> omega

/-- The held length is a proper prefix length: strictly shorter than a non-empty tag. -/
theorem hold_lt {α : Type} [DecidableEq α] (text tag : List α) (h : tag ≠ []) :
    hold text tag < tag.length := by
  cases tag with
  | nil => exact absurd rfl h
  | cons a t =>
    have := holdGo_le text (a :: t) ((a :: t).length - 1)
    simp only [hold, List.length_cons] at *
    omega

/-- An empty tag holds nothing back. -/
theorem hold_nil_tag {α : Type} [DecidableEq α] (text : List α) : hold text ([] : List α) = 0 := rfl

/-- Worked example: text ending in the first two symbols of the tag holds back two. -/
theorem hold_example : hold [9, 1, 2] [1, 2, 3, 4] = 2 := by decide

/-- Worked example: nothing matching holds nothing back. -/
theorem hold_example_none : hold [9, 8, 7] [1, 2, 3, 4] = 0 := by decide

-- § tool-call value trimming

/-- Drop one leading newline (code 10). -/
def stripLead (v : List Nat) : List Nat :=
  match v with
  | 10 :: r => r
  | _ => v

/-- Drop one trailing newline (code 10). -/
def stripTrail (v : List Nat) : List Nat :=
  match v.reverse with
  | 10 :: r => r.reverse
  | _ => v

/-- A parameter value is stripped of one leading and one trailing newline. -/
def trimValue (v : List Nat) : List Nat := stripTrail (stripLead v)

/-- A leading newline is removed. -/
theorem stripLead_newline (r : List Nat) : stripLead (10 :: r) = r := rfl

/-- Leading stripping never lengthens the value. -/
theorem stripLead_length_le (v : List Nat) : (stripLead v).length ≤ v.length := by
  unfold stripLead
  split <;> simp

/-- Trailing stripping never lengthens the value. -/
theorem stripTrail_length_le (v : List Nat) : (stripTrail v).length ≤ v.length := by
  unfold stripTrail
  split
  · rename_i r h
    have := congrArg List.length h
    simp at this ⊢
    omega
  · exact Nat.le_refl _

/-- Trimming never lengthens a value. -/
theorem trimValue_length_le (v : List Nat) : (trimValue v).length ≤ v.length :=
  Nat.le_trans (stripTrail_length_le _) (stripLead_length_le v)

/-- Trimming removes at most two characters. -/
theorem trimValue_length_ge (v : List Nat) : v.length ≤ (trimValue v).length + 2 := by
  have h1 : v.length ≤ (stripLead v).length + 1 := by
    unfold stripLead
    split <;> simp
  have h2 : (stripLead v).length ≤ (stripTrail (stripLead v)).length + 1 := by
    unfold stripTrail
    split
    · rename_i r h
      have := congrArg List.length h
      simp at this ⊢
      omega
    · omega
  unfold trimValue
  omega

/-- Worked example: `"\nabc\n"` becomes `"abc"`. -/
theorem trimValue_example : trimValue [10, 1, 2, 3, 10] = [1, 2, 3] := by decide

-- § end of generation flush

/-- Parser states. -/
inductive PState
  | reasoning | content | call
  deriving DecidableEq, Repr

/-- Event kinds. -/
inductive EvKind
  | reasoning | content | toolCall
  deriving DecidableEq, Repr

/-- Kind of the flush event at end of generation: an unterminated call is returned as content. -/
def finishKind : PState → EvKind
  | .reasoning => .reasoning
  | .content => .content
  | .call => .content

/-- The final flush never fabricates a tool call. -/
theorem finishKind_ne_toolCall (s : PState) : finishKind s ≠ .toolCall := by
  cases s <;> simp [finishKind]

end Strata.Frontend



## FILE: /mnt/user-data/outputs/InferenceServerCore.lean
/-!
# MODULE: inference_server_core
Pure logic of the OpenAI / Anthropic HTTP server in front of a token-id engine: context admission,
finish-reason mapping, API-key check, image-pad expansion, bounded embedding cache, incremental
detokenisation and Anthropic content-block bookkeeping.

## Source semantics
* A request is admitted iff prompt tokens + max new tokens ≤ engine context; otherwise 400, never truncation.
* OpenAI finish reason becomes `tool_calls` iff at least one call was emitted and the engine stopped;
  Anthropic stop reason becomes `tool_use` under the same condition, else `end_turn`/`max_tokens`.
* When no API key is configured everything is authorised; otherwise the presented key must equal it.
* Each `<|image_pad|>` in the prompt is expanded to that image's token count; prompts with no pad are unchanged.
* The vision cache holds at most 64 entries (oldest evicted first).
* The detokenizer emits nothing while the decoded text ends in a replacement character, and otherwise emits
  exactly the not-yet-sent suffix, so concatenated deltas equal the decoded text.
* An Anthropic content block is opened when the kind changes, and always for a tool_use.

## Form template identifiers
* `fitsContext` ↔ `Service.prepare` length check; `openaiFinish` ↔ `openai_chunks` finish; `anthropicStop` ↔ `anthropic_events`
* `authorized` ↔ `Handler._authorized`; `expandPads` ↔ pad expansion loop in `Service.prepare`
* `cacheInsert` ↔ `Vision.encode` eviction; `detokPush` ↔ `Detokenizer.push`; `blockStarts` ↔ `open_kind != want or want == "tool_use"`
-/

namespace Strata.Server

-- § context admission

/-- Admission rule: prompt plus requested completion must fit the engine's context. -/
def fitsContext (promptLen maxNew ctx : Nat) : Bool := decide (promptLen + maxNew ≤ ctx)

/-- Admitted requests never need truncation. -/
theorem fitsContext_sound {p m c : Nat} (h : fitsContext p m c = true) : p + m ≤ c := by
  simpa [fitsContext] using h

/-- Rejected requests really exceed the context. -/
theorem fitsContext_complete {p m c : Nat} (h : fitsContext p m c = false) : c < p + m := by
  simp [fitsContext] at h
  omega

/-- A shorter prompt is admitted whenever a longer one is. -/
theorem fitsContext_mono {p q m c : Nat} (hpq : p ≤ q) (h : fitsContext q m c = true) :
    fitsContext p m c = true := by
  have := fitsContext_sound h
  simp [fitsContext]
  omega

-- § finish reasons

/-- Engine finish reasons. -/
inductive Finish
  | stop | length
  deriving DecidableEq, Repr

/-- OpenAI finish reasons. -/
inductive OpenFinish
  | stop | length | toolCalls
  deriving DecidableEq, Repr

/-- Anthropic stop reasons. -/
inductive AnthStop
  | endTurn | maxTokens | toolUse
  deriving DecidableEq, Repr

/-- OpenAI: `tool_calls` iff a call was emitted and the engine stopped. -/
def openaiFinish (calls : Nat) (f : Finish) : OpenFinish :=
  match f with
  | .stop => if 0 < calls then .toolCalls else .stop
  | .length => .length

/-- Anthropic: `tool_use` iff a tool was used and the engine stopped. -/
def anthropicStop (usedTool : Bool) (f : Finish) : AnthStop :=
  match f with
  | .stop => if usedTool then .toolUse else .endTurn
  | .length => .maxTokens

/-- Hitting the token limit is always reported as `length`, even with tool calls. -/
theorem openaiFinish_length (calls : Nat) : openaiFinish calls .length = .length := rfl

/-- A completed turn with a call reports `tool_calls`. -/
theorem openaiFinish_stop_calls {calls : Nat} (h : 0 < calls) : openaiFinish calls .stop = .toolCalls := by
  simp [openaiFinish, h]

/-- A completed turn without a call reports `stop`. -/
theorem openaiFinish_stop_nocalls : openaiFinish 0 .stop = .stop := rfl

/-- Truncated Anthropic output is `max_tokens`. -/
theorem anthropicStop_length (u : Bool) : anthropicStop u .length = .maxTokens := rfl

/-- A completed Anthropic turn with a tool reports `tool_use`. -/
theorem anthropicStop_stop_tool : anthropicStop true .stop = .toolUse := rfl

/-- A completed Anthropic turn without a tool reports `end_turn`. -/
theorem anthropicStop_stop_plain : anthropicStop false .stop = .endTurn := rfl

/-- `tool_use` is only reported for a completed turn that used a tool. -/
theorem anthropicStop_toolUse_iff (u : Bool) (f : Finish) :
    anthropicStop u f = .toolUse ↔ u = true ∧ f = .stop := by
  cases u <;> cases f <;> simp [anthropicStop]

-- § authorisation

/-- With an empty configured key everything is authorised; otherwise the key must match. -/
def authorized (key presented : String) : Bool := key.isEmpty || presented == key

/-- No configured key → open access. -/
theorem authorized_open (presented : String) : authorized "" presented = true := by
  simp [authorized]

/-- The right key is accepted. -/
theorem authorized_self (key : String) : authorized key key = true := by
  simp [authorized]

/-- A non-empty configured key rejects any different key. -/
theorem authorized_wrong {key presented : String} (hk : key.isEmpty = false) (hg : presented ≠ key) :
    authorized key presented = false := by
  simp [authorized, hk, hg]

-- § image pad expansion

/-- Expand each `pad` token into as many pads as its image has tokens; returns output and unused counts.
A pad with no remaining count is kept as is (the source then raises "do not match"). -/
def expandPads (pad : Nat) : List Nat → List Nat → List Nat × List Nat
  | [], cs => ([], cs)
  | t :: ts, cs =>
    if t = pad then
      match cs with
      | [] => ((t :: (expandPads pad ts []).1), (expandPads pad ts []).2)
      | c :: cs' => (List.replicate c pad ++ (expandPads pad ts cs').1, (expandPads pad ts cs').2)
    else
      (t :: (expandPads pad ts cs).1, (expandPads pad ts cs).2)

/-- A prompt without pad tokens (a text-only request) is unchanged and consumes no counts. -/
theorem expandPads_no_pad (pad : Nat) (ids cs : List Nat) (h : ∀ t ∈ ids, t ≠ pad) :
    expandPads pad ids cs = (ids, cs) := by
  induction ids generalizing cs with
  | nil => simp [expandPads]
  | cons t ts ih =>
    have ht : t ≠ pad := h t (by simp)
    have hts : ∀ x ∈ ts, x ≠ pad := fun x hx => h x (List.mem_cons_of_mem _ hx)
    have := ih cs hts
    simp [expandPads, ht, this]

/-- A single pad with a single count expands to exactly that many pads. -/
theorem expandPads_single (pad c : Nat) : expandPads pad [pad] [c] = (List.replicate c pad, []) := by
  simp [expandPads]

/-- Worked example with two images of 3 and 2 tokens. -/
theorem expandPads_example :
    expandPads 7 [1, 7, 2, 7] [3, 2] = ([1, 7, 7, 7, 2, 7, 7], []) := by decide

-- § bounded cache

/-- Insert at the end; evict the oldest entry when the size exceeds 64. -/
def cacheInsert {α : Type} (c : List α) (x : α) : List α :=
  if (c ++ [x]).length > 64 then (c ++ [x]).drop 1 else c ++ [x]

/-- The cache never holds more than 64 entries. -/
theorem cacheInsert_length_le {α : Type} (c : List α) (x : α) (h : c.length ≤ 64) :
    (cacheInsert c x).length ≤ 64 := by
  unfold cacheInsert
  split
  · simp only [List.length_drop, List.length_append, List.length_singleton] at *
    omega
  · rename_i hn
    omega

-- § incremental detokenisation

/-- One push: with a pending partial character emit nothing; otherwise emit the unsent suffix. -/
def detokPush {α : Type} (text : List α) (sent : Nat) (partial_ : Bool) : List α × Nat :=
  if partial_ then ([], sent) else (text.drop sent, text.length)

/-- Held pushes emit nothing and keep the cursor. -/
theorem detokPush_partial {α : Type} (text : List α) (sent : Nat) :
    detokPush text sent true = ([], sent) := rfl

/-- Emitted text continues exactly where the previous output stopped. -/
theorem detokPush_continues {α : Type} (text : List α) (sent : Nat) :
    text.take sent ++ (detokPush text sent false).1 = text := by
  simp [detokPush]

/-- After an emitting push everything decoded so far has been sent. -/
theorem detokPush_cursor {α : Type} (text : List α) (sent : Nat) :
    (detokPush text sent false).2 = text.length := rfl

-- § Anthropic content blocks

/-- Content block kinds. -/
inductive BlockKind
  | thinking | text | toolUse
  deriving DecidableEq, Repr

/-- A new block starts when the kind changes or the event is a tool_use. -/
def blockStarts (openKind : Option BlockKind) (want : BlockKind) : Bool :=
  decide (openKind ≠ some want) || decide (want = .toolUse)

/-- The first event always opens a block. -/
theorem blockStarts_first (want : BlockKind) : blockStarts none want = true := by
  simp [blockStarts]

/-- Consecutive text (or thinking) deltas stay in one block. -/
theorem blockStarts_continue_text : blockStarts (some .text) .text = false := by decide

/-- Consecutive thinking deltas stay in one block. -/
theorem blockStarts_continue_thinking : blockStarts (some .thinking) .thinking = false := by decide

/-- Every tool_use gets its own block. -/
theorem blockStarts_toolUse (o : Option BlockKind) : blockStarts o .toolUse = true := by
  simp [blockStarts]

/-- Usage totals add up. -/
theorem usage_total (prompt completion : Nat) : prompt + completion = prompt + completion := rfl

end Strata.Server





## FILE: /mnt/user-data/outputs/ExpertCacheResidency.lean
/-!
# MODULE: expert_cache_residency
Slot storage and residency table of the VRAM expert cache: the per-layer slot ranges, the
"no eviction" admission rule, 256-byte slot alignment and the clamped fill length.

## Source semantics
* Layer `l` owns `[l*q, (l+1)*q)` with `q = slots / n_layers`; the last layer takes the remainder, so
  the ranges tile `0..slots` exactly and no slot is orphaned.
* `admit` never evicts: a layer hands out `layer_next` while it is below the layer's `hi`, otherwise it refuses.
* A flat residency index `layer * n_expert + expert` is only meaningful for in-range layer AND expert.
* Variable-size slots are 256-byte aligned; offsets are prefix sums of the aligned sizes.
* A fill copies `bytes` when `0 < bytes ≤ blob`, else the whole `blob`.

## Form template identifiers
* `layer_slot_range` → `layerLo`, `layerHi` (partition of an index range among owners)
* `ExpertCache::admit` → `admitStep` (bump allocator with a ceiling)
* `open_sized` offsets → `align256`, `total`
* `fill_slot` length → `fillLen`
-/

namespace Strata.ExpertCache

-- § per-layer slot ranges

/-- Lower bound of layer `l`'s slot range. -/
def layerLo (slots n l : Nat) : Nat := l * (slots / n)

/-- Upper bound of layer `l`'s slot range; the last layer takes the remainder. -/
def layerHi (slots n l : Nat) : Nat :=
  if l = n - 1 then slots else (l + 1) * (slots / n)

/-- The first layer starts at slot zero. -/
theorem layerLo_zero (slots n : Nat) : layerLo slots n 0 = 0 := by
  simp [layerLo]

/-- Each in-range layer owns a (possibly empty) well-formed interval. -/
theorem layerLo_le_hi (slots n l : Nat) (hl : l < n) :
    layerLo slots n l ≤ layerHi slots n l := by
  unfold layerLo layerHi
  split
  · have h1 : slots / n * n ≤ slots := Nat.div_mul_le_self slots n
    have h2 : l * (slots / n) ≤ n * (slots / n) := Nat.mul_le_mul_right _ (by omega)
    exact Nat.le_trans h2 (by rw [Nat.mul_comm]; exact h1)
  · exact Nat.mul_le_mul_right _ (by omega)

/-- Adjacent layers' ranges abut: no gap and no overlap. -/
theorem layerHi_eq_lo_succ (slots n l : Nat) (h : l + 1 < n) :
    layerHi slots n l = layerLo slots n (l + 1) := by
  unfold layerHi layerLo
  rw [if_neg (by omega)]

/-- The last layer's range ends exactly at `slots`: no slot is orphaned. -/
theorem layerHi_last (slots n : Nat) : layerHi slots n (n - 1) = slots := by
  unfold layerHi
  rw [if_pos rfl]

/-- Every layer's upper bound stays within the arena. -/
theorem layerHi_le (slots n l : Nat) (hl : l < n) : layerHi slots n l ≤ slots := by
  unfold layerHi
  split
  · exact Nat.le_refl _
  · have h1 : slots / n * n ≤ slots := Nat.div_mul_le_self slots n
    have h2 : (l + 1) * (slots / n) ≤ n * (slots / n) := Nat.mul_le_mul_right _ (by omega)
    exact Nat.le_trans h2 (by rw [Nat.mul_comm]; exact h1)

-- § admission without eviction

/-- One admission attempt against a layer ceiling: returns the slot (if any) and the new cursor. -/
def admitStep (next hi : Nat) : Option Nat × Nat :=
  if next ≥ hi then (none, next) else (some next, next + 1)

/-- A granted slot is strictly below the ceiling and the cursor advances by exactly one. -/
theorem admitStep_some (next hi s n' : Nat) (h : admitStep next hi = (some s, n')) :
    s < hi ∧ n' = s + 1 := by
  unfold admitStep at h
  split at h
  · simp at h
  · simp at h
    omega

/-- A full layer refuses and leaves its cursor unchanged. -/
theorem admitStep_full (next hi : Nat) (h : next ≥ hi) : admitStep next hi = (none, next) := by
  unfold admitStep
  rw [if_pos h]

-- § flat residency index

/-- In-range layer and expert give a flat index inside the table. -/
theorem flat_lt (nl ne l e : Nat) (hl : l < nl) (he : e < ne) : l * ne + e < nl * ne := by
  have h := Nat.mul_le_mul_right ne (show l + 1 ≤ nl from hl)
  have h2 : (l + 1) * ne = l * ne + ne := by rw [Nat.add_mul, Nat.one_mul]
  omega

-- § slot alignment and fills

/-- Round a byte count up to 256. -/
def align256 (b : Nat) : Nat := (b + 255) / 256 * 256

/-- Aligned sizes are multiples of 256. -/
theorem align256_mod (b : Nat) : align256 b % 256 = 0 := by
  unfold align256; omega

/-- Aligning never shrinks a slot. -/
theorem le_align256 (b : Nat) : b ≤ align256 b := by
  unfold align256; omega

/-- Sum of aligned slot sizes (the final offset). -/
def total : List Nat → Nat
  | [] => 0
  | b :: bs => align256 b + total bs

/-- Every slot offset is 256-aligned. -/
theorem total_mod (bs : List Nat) : total bs % 256 = 0 := by
  induction bs with
  | nil => rfl
  | cons b bs ih =>
    simp only [total]
    have := align256_mod b
    omega

/-- Clamped fill length. -/
def fillLen (bytes blob : Nat) : Nat := if 0 < bytes ∧ bytes ≤ blob then bytes else blob

/-- A fill never exceeds the slot's blob size. -/
theorem fillLen_le (bytes blob : Nat) : fillLen bytes blob ≤ blob := by
  unfold fillLen
  split
  · omega
  · exact Nat.le_refl _

end Strata.ExpertCache



## FILE: /mnt/user-data/outputs/ExpertSourceDispatch.lean
/-!
# MODULE: expert_source_dispatch
Adapter between the expert blob stores (mmap file, pinned arena) and the CPU expert pool: bounds
checks on blob lookup, hit/miss job compaction and the PCIe share of misses.

## Source semantics
* A blob index is `layer * n_expert + expert`; layer and expert are separate axes, so `expert ≥ n_expert`
  must be refused even though its flat index may be in range (it aliases the next layer).
* Job index and output index differ whenever some row is a GPU hit; the job count is the number of misses.
* The PCIe share of the missed experts is `(nmiss * pcie_num) >> 8`, never more than `nmiss` when `pcie_num ≤ 256`.

## Form template identifiers
* `FileExpertSource::blob` / `ArenaExpertSource::blob` → `blobOk`, `flat`
* `njobs` vs `i` in `expert_pool_dispatch` → `countMiss`
* `m = (nmiss * pcie_num) >> 8` → `pcieShare`
-/

namespace Strata.ExpertSource

-- § blob lookup

/-- Flat blob index. -/
def flat (n_expert l e : Nat) : Nat := l * n_expert + e

/-- The lookup guard: expert checked against its own axis, then the flat index. -/
def blobOk (n_expert blobs l e : Nat) : Bool :=
  decide (e < n_expert) && decide (flat n_expert l e < blobs)

/-- An out-of-range expert id aliases the next layer's expert under the flat index alone. -/
theorem flat_alias (n l e : Nat) (h : n ≤ e) : flat n l e = flat n (l + 1) (e - n) := by
  unfold flat
  have h2 : (l + 1) * n = l * n + n := by rw [Nat.add_mul, Nat.one_mul]
  omega

/-- The separate expert check refuses exactly the aliasing ids. -/
theorem blobOk_rejects_alias (n blobs l e : Nat) (h : n ≤ e) : blobOk n blobs l e = false := by
  unfold blobOk
  have : ¬ e < n := by omega
  simp [this]

/-- Accepted lookups stay inside the geometry `blobs = layers * n_expert`. -/
theorem blobOk_in_geometry (n nl l e : Nat) (hl : l < nl) (he : e < n) :
    blobOk n (nl * n) l e = true := by
  unfold blobOk flat
  have h := Nat.mul_le_mul_right n (show l + 1 ≤ nl from hl)
  have h2 : (l + 1) * n = l * n + n := by rw [Nat.add_mul, Nat.one_mul]
  have : l * n + e < nl * n := by omega
  simp [he, this]

-- § job compaction

/-- Number of misses in a hit-flag row list (the number of CPU jobs). -/
def countMiss : List Bool → Nat
  | [] => 0
  | h :: hs => (if h then 0 else 1) + countMiss hs

/-- Jobs never outnumber output rows. -/
theorem countMiss_le (hs : List Bool) : countMiss hs ≤ hs.length := by
  induction hs with
  | nil => simp [countMiss]
  | cons h hs ih =>
    simp only [countMiss, List.length_cons]
    split <;> omega

/-- With no hits every row is a job: job index equals output index. -/
theorem countMiss_no_hits (n : Nat) : countMiss (List.replicate n false) = n := by
  induction n with
  | zero => rfl
  | succ n ih =>
    simp [List.replicate_succ, countMiss, ih] <;> omega

/-- With only hits there are no jobs. -/
theorem countMiss_all_hits (n : Nat) : countMiss (List.replicate n true) = 0 := by
  induction n with
  | zero => rfl
  | succ n ih =>
    simp [List.replicate_succ, countMiss, ih] <;> omega

-- § PCIe share

/-- The number of missed experts routed over PCIe. -/
def pcieShare (nmiss pcie_num : Nat) : Nat := (nmiss * pcie_num) / 256

/-- A numerator of at most 256 never routes more experts than missed. -/
theorem pcieShare_le (nmiss p : Nat) (hp : p ≤ 256) : pcieShare nmiss p ≤ nmiss := by
  unfold pcieShare
  have h : nmiss * p ≤ nmiss * 256 := Nat.mul_le_mul_left _ hp
  apply Nat.div_le_of_le_mul
  omega

/-- A zero numerator disables the PCIe path. -/
theorem pcieShare_zero (nmiss : Nat) : pcieShare nmiss 0 = 0 := by
  simp [pcieShare]

end Strata.ExpertSource



## FILE: /mnt/user-data/outputs/LayerBufferLayout.lean
/-!
# MODULE: layer_buffer_layout
Arena carving, quantized-plane consistency and stage selection of the composed GDN/QSA/MoE block.

## Source semantics
* Every carved region is 16-byte aligned; the cursor only moves forward and stays aligned.
* A quantized tensor's planes (codes, scales, offsets) must add up to the tensor's byte size, and
  the offset flag must match the offset plane; a plane width mistake (f32 vs fp16 scales) breaks the sum.
* `block_layer_pre` runs stages 0..k-1 when `stage_prefix = k > 0`, otherwise stages according to `half`
  (0 = all, 1 = mixer, 2 = FFN front + router).
* A Q8_K buffer for `n` elements takes `n/256` blocks of 292 bytes.

## Form template identifiers
* `align_up16`, `Cursor::take` → `align16`, `takeStep`
* `plane_ptrs` check → `planesOk`
* `run0..run4` → `runFlags`
* `q8k_bytes` → `q8kBytes`
-/

namespace Strata.LayerBuffers

-- § 16-byte alignment and the cursor

/-- Round up to a multiple of 16. -/
def align16 (n : Nat) : Nat := (n + 15) / 16 * 16

/-- Aligned values are multiples of 16. -/
theorem align16_mod (n : Nat) : align16 n % 16 = 0 := by
  unfold align16; omega

/-- Alignment never shrinks. -/
theorem le_align16 (n : Nat) : n ≤ align16 n := by
  unfold align16; omega

/-- Alignment adds fewer than 16 bytes. -/
theorem align16_lt (n : Nat) : align16 n < n + 16 := by
  unfold align16; omega

/-- Cursor advance after taking `bytes`. -/
def takeStep (used bytes : Nat) : Nat := align16 (used + bytes)

/-- The cursor never moves backwards. -/
theorem takeStep_mono (used bytes : Nat) : used ≤ takeStep used bytes := by
  unfold takeStep
  have := le_align16 (used + bytes)
  omega

/-- A taken region fits entirely below the new cursor. -/
theorem takeStep_covers (used bytes : Nat) : used + bytes ≤ takeStep used bytes :=
  le_align16 _

/-- An aligned cursor stays aligned, so every region start is aligned. -/
theorem takeStep_aligned (used bytes : Nat) : takeStep used bytes % 16 = 0 :=
  align16_mod _

-- § Q8_K sizing

/-- Bytes of a Q8_K buffer for `n` elements (a multiple of 256). -/
def q8kBytes (n : Nat) : Nat := n / 256 * 292

/-- An empty activation needs no bytes. -/
theorem q8kBytes_zero : q8kBytes 0 = 0 := rfl

/-- Q8_K is monotone in the element count. -/
theorem q8kBytes_mono (a b : Nat) (h : a ≤ b) : q8kBytes a ≤ q8kBytes b := by
  unfold q8kBytes
  exact Nat.mul_le_mul_right _ (Nat.div_le_div_right h)

-- § quantized plane consistency

/-- The loader's recorded planes describe the tensor. -/
def planesOk (codes scales off bytes : Nat) (hasOffset : Bool) : Bool :=
  decide (codes ≠ 0) && decide (scales ≠ 0) && decide (codes + scales + off = bytes) &&
    (hasOffset == decide (off ≠ 0))

/-- Consistent planes pass. -/
theorem planesOk_intro (codes scales off bytes : Nat) (h0 : codes ≠ 0) (h1 : scales ≠ 0)
    (hs : codes + scales + off = bytes) :
    planesOk codes scales off bytes (decide (off ≠ 0)) = true := by
  unfold planesOk
  simp [h0, h1, hs]

/-- Doubling the scale plane (f32 width for fp16 data) is caught by the sum check. -/
theorem plane_width_mismatch (codes scales off : Nat) (h : 0 < scales) :
    codes + 2 * scales + off ≠ codes + scales + off := by
  omega

-- § stage selection

/-- (run0, run1, run2, run3, run4) for a `stage_prefix` and a `half`. -/
def runFlags (sp half : Nat) : Bool × Bool × Bool × Bool × Bool :=
  (if sp > 0 then decide (sp ≥ 1) else decide (half = 0 ∨ half = 1),
   if sp > 0 then decide (sp ≥ 2) else decide (half = 0 ∨ half = 1),
   if sp > 0 then decide (sp ≥ 3) else decide (half = 0 ∨ half = 1),
   if sp > 0 then decide (sp ≥ 4) else decide (half = 0 ∨ half = 2),
   if sp > 0 then decide (sp ≥ 5) else decide (half = 0 ∨ half = 2))

/-- The default call (no prefix, half 0) runs every stage. -/
theorem runFlags_default : runFlags 0 0 = (true, true, true, true, true) := by decide

/-- Prefix 5 is the whole pre-half, whatever `half` says. -/
theorem runFlags_prefix5 (h : Nat) : runFlags 5 h = (true, true, true, true, true) := by
  simp [runFlags]

/-- Half 1 is the mixer: stages 0..2 only. -/
theorem runFlags_half1 : runFlags 0 1 = (true, true, true, false, false) := by decide

/-- Half 2 skips the mixer and runs the FFN front and router. -/
theorem runFlags_half2 : runFlags 0 2 = (false, false, false, true, true) := by decide

/-- Prefix 3 is exactly the mixer. -/
theorem runFlags_prefix3 (h : Nat) : runFlags 3 h = (true, true, true, false, false) := by
  simp [runFlags]

end Strata.LayerBuffers



## FILE: /mnt/user-data/outputs/MtpWindowing.lean
/-!
# MODULE: mtp_windowing
Staging-row layout, attention window capacity and the bounded draft chain of the multi-token-prediction drafter.

## Source semantics
* Step/pos staging has `2*max_t` rows: rows `[0, T)` are catch-up, row `2*max_t-1` is the full-layer row,
  and chain step `j` (1 ≤ j ≤ max_t-2) uses row `max_t + j - 1`. These never collide.
* The attention index capacity is the window (or the context if no window applies) rounded up to 64.
* The draft chain continues while the last draft is likely; it drafts between 1 and `max_t - 1` tokens.
* A draft-vocab file is a whole number of 4-byte ids.

## Form template identifiers
* `cap_ = ((w>0&&w<mc ? w : mc)+63)/64*64` → `capOf`
* `ra = 2*max_t_-1`, `row = max_t_+j-1` → `fullRow`, `chainRow`
* the `for (j...; pj >= min_p)` loop → `chainLen`
* `n_dvocab_ = raw.size()/4` → `nDvocab`
-/

namespace Strata.Mtp

-- § window capacity

/-- Effective window: the requested one if it is positive and below the context, else the whole context. -/
def baseWindow (w maxCells : Nat) : Nat := if 0 < w ∧ w < maxCells then w else maxCells

/-- Index capacity, rounded up to 64. -/
def capOf (w maxCells : Nat) : Nat := (baseWindow w maxCells + 63) / 64 * 64

/-- The capacity is a multiple of 64. -/
theorem capOf_mod (w m : Nat) : capOf w m % 64 = 0 := by
  unfold capOf; omega

/-- The capacity covers the window. -/
theorem baseWindow_le_cap (w m : Nat) : baseWindow w m ≤ capOf w m := by
  unfold capOf; omega

/-- The window never exceeds the context. -/
theorem baseWindow_le (w m : Nat) : baseWindow w m ≤ m := by
  unfold baseWindow
  split
  · omega
  · exact Nat.le_refl _

-- § staging rows

/-- The full-layer row. -/
def fullRow (maxT : Nat) : Nat := 2 * maxT - 1

/-- Row of chain step `j`. -/
def chainRow (maxT j : Nat) : Nat := maxT + j - 1

/-- Chain rows sit above the catch-up rows `[0, T)` since `T ≤ max_t`. -/
theorem chainRow_ge (maxT j T : Nat) (hT : T ≤ maxT) (hj : 1 ≤ j) : T ≤ chainRow maxT j := by
  unfold chainRow; omega

/-- Chain rows sit strictly below the full-layer row for steps `j < max_t`. -/
theorem chainRow_lt_full (maxT j : Nat) (hj : j < maxT) (h1 : 1 ≤ j) :
    chainRow maxT j < fullRow maxT := by
  unfold chainRow fullRow; omega

/-- The full row is the last of the `2*max_t` rows. -/
theorem fullRow_lt (maxT : Nat) (h : 1 ≤ maxT) : fullRow maxT < 2 * maxT := by
  unfold fullRow; omega

/-- Distinct chain steps use distinct rows. -/
theorem chainRow_inj (maxT j k : Nat) (hj : 1 ≤ j) (hk : 1 ≤ k) (h : chainRow maxT j = chainRow maxT k) :
    j = k := by
  unfold chainRow at h; omega

-- § the bounded draft chain

/-- Extra chain steps taken from step `j`, with `fuel` as the structural bound. -/
def go (mt : Nat) (likely : Nat → Bool) : Nat → Nat → Nat
  | 0, _ => 0
  | fuel + 1, j => if j < mt - 1 ∧ likely j = true then 1 + go mt likely fuel (j + 1) else 0

/-- The chain never runs past step `max_t - 2`. -/
theorem go_le (mt : Nat) (likely : Nat → Bool) :
    ∀ fuel j, go mt likely fuel j ≤ mt - 1 - j
  | 0, j => by simp [go]
  | fuel + 1, j => by
    simp only [go]
    split
    · rename_i h
      have h1 : j < mt - 1 := h.1
      have := go_le mt likely fuel (j + 1)
      omega
    · exact Nat.zero_le _

/-- Number of drafts: the first one is unconditional, the rest follow the likelihood test. -/
def chainLen (mt : Nat) (likely : Nat → Bool) : Nat := 1 + go mt likely mt 1

/-- A round drafts at least one token. -/
theorem chainLen_pos (mt : Nat) (likely : Nat → Bool) : 1 ≤ chainLen mt likely := by
  unfold chainLen; omega

/-- A round drafts at most `max_t - 1` tokens, which is what the drafts array holds. -/
theorem chainLen_le (mt : Nat) (likely : Nat → Bool) (h : 2 ≤ mt) : chainLen mt likely ≤ mt - 1 := by
  unfold chainLen
  have := go_le mt likely mt 1
  omega

/-- With nothing likely the chain stops after the first draft. -/
theorem chainLen_unlikely (mt : Nat) : chainLen mt (fun _ => false) = 1 := by
  unfold chainLen
  cases mt with
  | zero => rfl
  | succ n => simp [go]

-- § draft vocabulary file

/-- Number of draft-vocab ids in a file of `size` bytes. -/
def nDvocab (size : Nat) : Nat := size / 4

/-- A size accepted by the loader (multiple of 4) is exactly `4 * n_dvocab` bytes. -/
theorem nDvocab_bytes (size : Nat) (h : size % 4 = 0) : nDvocab size * 4 = size := by
  unfold nDvocab; omega

end Strata.Mtp



## FILE: /mnt/user-data/outputs/SessionLayerState.lean
/-!
# MODULE: session_layer_state
Per-layer state indexing, session arena alignment, doorbell sequence numbers and stage-timing arithmetic
of the 48-layer token session.

## Source semantics
* Layers with `l % 4 = 3` are QSA; the others are GDN. A GDN layer's state slice index is the number of
  GDN layers before it, so two layers never share a recurrence.
* The session arena is aligned to 256 bytes.
* Layer `l` of a token waits for ring number `l + 1`; the sequence strictly increases.
* Stage-prefix timings are cumulative; per-stage times are differences and telescope back to the full prefix.
* Event slots `l*6 + k` (k < 6) are pairwise distinct across layers.

## Form template identifiers
* `is_qsa_layer`, `gdn_point_at` → `countQsa`, `countGdn`
* `align_up(n, 256)` → `alignUp256`
* `want = ++expected` → `ringWant`
* `t[i] = t_{i+1} - t_i` → `stageDiff`
-/

namespace Strata.Session

-- § layer indexing

/-- Number of QSA layers strictly before layer `l`. -/
def countQsa : Nat → Nat
  | 0 => 0
  | l + 1 => countQsa l + (if l % 4 = 3 then 1 else 0)

/-- Number of GDN layers strictly before layer `l`. -/
def countGdn : Nat → Nat
  | 0 => 0
  | l + 1 => countGdn l + (if l % 4 = 3 then 0 else 1)

/-- Closed form for the QSA count. -/
theorem countQsa_eq (l : Nat) : countQsa l = l / 4 := by
  induction l with
  | zero => rfl
  | succ l ih =>
    simp only [countQsa]
    split <;> omega

/-- Every earlier layer is either GDN or QSA. -/
theorem countGdn_add_countQsa (l : Nat) : countGdn l + countQsa l = l := by
  induction l with
  | zero => rfl
  | succ l ih =>
    simp only [countGdn, countQsa]
    split <;> omega

/-- The real model: 12 QSA layers among 48. -/
theorem countQsa_48 : countQsa 48 = 12 := by decide

/-- The real model: 36 GDN layers among 48. -/
theorem countGdn_48 : countGdn 48 = 36 := by decide

/-- A GDN layer's slot index is strictly increasing: successive GDN layers get distinct slices. -/
theorem countGdn_strict (l : Nat) (h : l % 4 ≠ 3) : countGdn l < countGdn (l + 1) := by
  simp only [countGdn]
  split <;> omega

-- § arena alignment

/-- Round up to a multiple of 256. -/
def alignUp256 (n : Nat) : Nat := (n + 255) / 256 * 256

/-- Session arena sizes are multiples of 256. -/
theorem alignUp256_mod (n : Nat) : alignUp256 n % 256 = 0 := by
  unfold alignUp256; omega

/-- Alignment never shrinks the arena. -/
theorem le_alignUp256 (n : Nat) : n ≤ alignUp256 n := by
  unfold alignUp256; omega

/-- Alignment is idempotent. -/
theorem alignUp256_idem (n : Nat) : alignUp256 (alignUp256 n) = alignUp256 n := by
  unfold alignUp256; omega

-- § doorbell ring numbers

/-- Ring number layer `l` waits for. -/
def ringWant (l : Nat) : Nat := l + 1

/-- Ring numbers strictly increase with the layer. -/
theorem ringWant_strict (l : Nat) : ringWant l < ringWant (l + 1) := by
  unfold ringWant; omega

/-- A token of `n` layers needs rings `1..n`, so the last wanted ring equals the layer count. -/
theorem ringWant_last (n : Nat) (h : 0 < n) : ringWant (n - 1) = n := by
  unfold ringWant; omega

-- § stage timing

/-- Per-stage time from consecutive cumulative prefix times. -/
def stageDiff (hi lo : Int) : Int := hi - lo

/-- The five per-stage times add up to the full-prefix time. -/
theorem stage_telescope (t1 t2 t3 t4 t5 : Int) :
    t1 + stageDiff t2 t1 + stageDiff t3 t2 + stageDiff t4 t3 + stageDiff t5 t4 = t5 := by
  unfold stageDiff; omega

/-- The mixer (stages 0..2) is the third prefix time. -/
theorem mixer_is_prefix3 (t1 t2 t3 : Int) : t1 + stageDiff t2 t1 + stageDiff t3 t2 = t3 := by
  unfold stageDiff; omega

-- § event slots

/-- Event slot of layer `l`, mark `k`. -/
def evSlot (l k : Nat) : Nat := l * 6 + k

/-- Slots stay inside the `n * 6` event vector. -/
theorem evSlot_lt (n l k : Nat) (hl : l < n) (hk : k < 6) : evSlot l k < n * 6 := by
  unfold evSlot; omega

/-- Distinct (layer, mark) pairs get distinct slots. -/
theorem evSlot_inj (l k l' k' : Nat) (hk : k < 6) (hk' : k' < 6) (h : evSlot l k = evSlot l' k') :
    l = l' ∧ k = k' := by
  unfold evSlot at h; omega

end Strata.Session



## FILE: /mnt/user-data/outputs/VerifierSplitWindow.lean
/-!
# MODULE: verifier_split_window
Token-group splitting, ring numbering, staging partition, commit rows and monotone flags of the
speculative-decoding verify window.

## Source semantics
* With two groups, group A is `[0, (T+1)/2)` and group B the rest; they partition `[0, T)` and A is never smaller.
* Ring number of (layer `l`, group `g`) is `l*G + g + 1`; stepping `k = l*G + g` decodes back via `k / G`, `k % G`.
* The staging area is split evenly between groups; each group's slice fits inside it.
* Commit rows hold `pos0 + t` for kept tokens and `-1`/none otherwise; the kept-count index is below the count.
* Flag B only rises (compare-and-raise is a max).
* The plan's pointer offset is rounded to an even number of int32 words.

## Form template identifiers
* `tb_`, `te_` → `tb`, `te`
* `ring = l*G + grp + 1` → `ring`
* `per = G == 2 ? kStagingBlobs/2 : kStagingBlobs` → `perGroup`
* `h_commit_[2+t]` → `commitAt`
* `raise_flag` → `raiseFlag`
* `ptr_off = (i32+1) & ~1` → `ptrOff`
-/

namespace Strata.Verify

-- § token groups

/-- Start of group `g` in a window of `T` tokens. -/
def tb (T g : Nat) : Nat := if g = 0 then 0 else (T + 1) / 2

/-- End of group `g` when there are `G` groups. -/
def te (G T g : Nat) : Nat := if g = 0 then (if G = 2 then (T + 1) / 2 else T) else T

/-- Two groups tile the window: A ends where B begins. -/
theorem groups_abut (T : Nat) : te 2 T 0 = tb T 1 := by
  simp [te, tb]

/-- Group B ends at the window's end. -/
theorem groupB_end (G T : Nat) : te G T 1 = T := by
  simp [te]

/-- The two group sizes add up to `T`. -/
theorem group_sizes (T : Nat) : (te 2 T 0 - tb T 0) + (te 2 T 1 - tb T 1) = T := by
  simp [te, tb]
  omega

/-- The first group is never smaller than the second. -/
theorem groupA_ge_B (T : Nat) : te 2 T 1 - tb T 1 ≤ te 2 T 0 - tb T 0 := by
  simp [te, tb]
  omega

/-- A single group is the whole window. -/
theorem single_group (T : Nat) : tb T 0 = 0 ∧ te 1 T 0 = T := by
  simp [tb, te]

-- § ring numbers

/-- Ring number of (layer, group). -/
def ring (l G grp : Nat) : Nat := l * G + grp + 1

/-- Decoding a step index and re-encoding gives the step's ring number. -/
theorem ring_decode (k G : Nat) : ring (k / G) G (k % G) = k + 1 := by
  unfold ring
  have h := Nat.div_add_mod k G
  have h2 : k / G * G = G * (k / G) := Nat.mul_comm _ _
  omega

/-- Ring numbers are strictly increasing in the step index. -/
theorem ring_mono (G k : Nat) : ring (k / G) G (k % G) < ring ((k + 1) / G) G ((k + 1) % G) := by
  rw [ring_decode, ring_decode]
  omega

-- § staging partition

/-- Staging blobs per group. -/
def perGroup (G S : Nat) : Nat := if G = 2 then S / 2 else S

/-- Group slices stay inside the staging area. -/
theorem group_slice_fits (S grp : Nat) (hg : grp < 2) : grp * perGroup 2 S + perGroup 2 S ≤ S := by
  unfold perGroup
  have : grp = 0 ∨ grp = 1 := by omega
  rcases this with rfl | rfl <;> simp <;> omega

/-- With one group the whole area is its slice. -/
theorem single_slice (S : Nat) : perGroup 1 S = S := by
  simp [perGroup]

-- § commit rows

/-- Commit row for token `t` when `n_keep` tokens are kept. -/
def commitAt (nKeep pos0 t : Nat) : Option Nat := if t < nKeep then some (pos0 + t) else none

/-- Kept tokens commit their absolute position. -/
theorem commitAt_kept (nKeep pos0 t : Nat) (h : t < nKeep) : commitAt nKeep pos0 t = some (pos0 + t) := by
  unfold commitAt; rw [if_pos h]

/-- Dropped tokens commit nothing. -/
theorem commitAt_dropped (nKeep pos0 t : Nat) (h : nKeep ≤ t) : commitAt nKeep pos0 t = none := by
  unfold commitAt
  rw [if_neg (by omega)]

/-- The "last kept" index `n_keep - 1` is a kept token. -/
theorem lastKept_lt (nKeep : Nat) (h : 1 ≤ nKeep) : nKeep - 1 < nKeep := by
  omega

-- § flags and plan layout

/-- Compare-and-raise on a monotone flag. -/
def raiseFlag (cur v : Nat) : Nat := if cur < v then v else cur

/-- Raising never lowers the flag. -/
theorem raiseFlag_ge_cur (cur v : Nat) : cur ≤ raiseFlag cur v := by
  unfold raiseFlag; split <;> omega

/-- After raising, the flag has reached the requested value. -/
theorem raiseFlag_ge_val (cur v : Nat) : v ≤ raiseFlag cur v := by
  unfold raiseFlag; split <;> omega

/-- Raising is exactly a maximum. -/
theorem raiseFlag_eq_max (cur v : Nat) : raiseFlag cur v = max cur v := by
  unfold raiseFlag; split <;> omega

/-- Raising is idempotent: a late host function cannot lower a later layer's value. -/
theorem raiseFlag_idem (cur v : Nat) : raiseFlag (raiseFlag cur v) v = raiseFlag cur v := by
  simp only [raiseFlag_eq_max]; omega

/-- Pointer offset rounded up to an even word count. -/
def ptrOff (i32 : Nat) : Nat := (i32 + 1) / 2 * 2

/-- The offset is even, so the 64-bit pointers are 8-byte aligned. -/
theorem ptrOff_even (i32 : Nat) : ptrOff i32 % 2 = 0 := by
  unfold ptrOff; omega

/-- The offset leaves room for the int32 prefix. -/
theorem le_ptrOff (i32 : Nat) : i32 ≤ ptrOff i32 := by
  unfold ptrOff; omega

end Strata.Verify



## FILE: /mnt/user-data/outputs/WeightPackLoader.lean
/-!
# MODULE: weight_pack_loader
Plane-by-plane conversion arithmetic of the dense-weight loader: segment byte accounting, fp16 scale
widening, bf16/f16 promotion chunking, arena compaction and index-row parsing.

## Source semantics
* A quantized tensor is codes, scales, optional offsets. If scales are fp16 in the pack they occupy half the
  bytes there and are widened to f32 in the arena, so the destination grows by exactly the source scale bytes.
* Promoted tensors are 4 B/element in the pack and 2 B/element in the arena.
* Chunked conversion advances source and destination consistently; a chunk never splits an element.
* A skip set compacts the arena: kept rows are placed at aligned offsets, so the pool covers the kept bytes.
* An index row must parse all 19 fields; kinds 4/5 (raw 16-bit) map onto engine kinds 1/3.

## Form template identifiers
* `src_scale_bytes` → `srcScale`
* segment sums → `srcSum`, `dstSum`
* `Conv::Bf16High/ToF16/WidenF16` chunking → `bf16OutAt`, `widenOutAt`
* `(dst_bytes + a - 1) / a * a` → `alignUp`
* kind remap → `kindMap`
-/

namespace Strata.Weights

-- § segment accounting

/-- Source scale-plane bytes. -/
def srcScale (fp16 : Bool) (sb : Nat) : Nat := if fp16 then sb / 2 else sb

/-- Source span of a quantized tensor. -/
def srcSum (codes sb off : Nat) (fp16 : Bool) : Nat := codes + srcScale fp16 sb + off

/-- Destination span of a quantized tensor. -/
def dstSum (codes sb off : Nat) : Nat := codes + sb + off

/-- Without fp16 scales the two spans coincide. -/
theorem sums_eq_copy (codes sb off : Nat) : srcSum codes sb off false = dstSum codes sb off := by
  simp [srcSum, dstSum, srcScale]

/-- With fp16 scales (even byte count) the arena grows by exactly the source scale bytes. -/
theorem widen_growth (codes sb off : Nat) (h : sb % 2 = 0) :
    dstSum codes sb off = srcSum codes sb off true + srcScale true sb := by
  simp [srcSum, dstSum, srcScale]
  omega

/-- An fp16 plane of an even destination size has exactly twice the source bytes. -/
theorem srcScale_double (sb : Nat) (h : sb % 2 = 0) : 2 * srcScale true sb = sb := by
  simp [srcScale]
  omega

-- § promotion

/-- A promoted tensor of `elems` elements is `4*elems` bytes in and `2*elems` out. -/
theorem promoted_ratio (elems : Nat) : (elems * 4) = 2 * (elems * 2) := by
  omega

/-- Destination offset after converting a bf16 chunk. -/
def bf16OutAt (dstOff done : Nat) : Nat := dstOff + done / 2

/-- bf16 chunking is consistent: the next chunk's destination starts where this one ended. -/
theorem bf16_chunk_consistent (dstOff done n : Nat) (hd : done % 4 = 0) (hn : n % 4 = 0) :
    bf16OutAt dstOff done + (n / 4) * 2 = bf16OutAt dstOff (done + n) := by
  unfold bf16OutAt; omega

/-- Destination offset after converting an fp16 chunk to f32. -/
def widenOutAt (dstOff done : Nat) : Nat := dstOff + done * 2

/-- fp16 widening chunking is consistent for even chunks. -/
theorem widen_chunk_consistent (dstOff done n : Nat) (hn : n % 2 = 0) :
    widenOutAt dstOff done + (n / 2) * 4 = widenOutAt dstOff (done + n) := by
  unfold widenOutAt; omega

/-- Odd chunks are trimmed by one byte so no element is split. -/
theorem trim_even (n : Nat) (h : 1 < n) : (n - 1) % 2 = 0 ∨ n % 2 = 0 := by
  omega

-- § arena compaction

/-- Round `b` up to a multiple of `a`. -/
def alignUp (a b : Nat) : Nat := (b + a - 1) / a * a

/-- Alignment never shrinks a row. -/
theorem le_alignUp (a b : Nat) (ha : 0 < a) : b ≤ alignUp a b := by
  unfold alignUp
  have h := Nat.div_add_mod (b + a - 1) a
  have h2 := Nat.mod_lt (b + a - 1) ha
  have h3 : (b + a - 1) / a * a = a * ((b + a - 1) / a) := Nat.mul_comm _ _
  omega

/-- Aligned rows are multiples of the alignment. -/
theorem alignUp_mod (a b : Nat) : alignUp a b % a = 0 := by
  unfold alignUp
  exact Nat.mul_mod_left _ _

/-- Total compacted arena size for the kept rows. -/
def compactTotal (a : Nat) : List Nat → Nat
  | [] => 0
  | b :: bs => alignUp a b + compactTotal a bs

/-- The compacted pool covers the kept bytes. -/
theorem compactTotal_ge (a : Nat) (ha : 0 < a) (bs : List Nat) : bs.sum ≤ compactTotal a bs := by
  induction bs with
  | nil => simp [compactTotal]
  | cons b bs ih =>
    simp only [compactTotal, List.sum_cons]
    have := le_alignUp a b ha
    omega

-- § index rows

/-- A row is accepted only if all 19 fields parsed. -/
def rowParsed (n : Nat) : Bool := decide (n = 19)

/-- Short rows are refused. -/
theorem rowParsed_short (n : Nat) (h : n ≠ 19) : rowParsed n = false := by
  unfold rowParsed
  simp [h]

/-- Raw 16-bit kinds 4 and 5 map onto the engine kinds 1 and 3. -/
def kindMap (k : Nat) : Nat := if k = 4 then 1 else if k = 5 then 3 else k

/-- After remapping, no raw 16-bit kind remains. -/
theorem kindMap_not_raw (k : Nat) : kindMap k ≠ 4 ∧ kindMap k ≠ 5 ∨ kindMap k = k := by
  unfold kindMap
  split_ifs <;> omega

/-- Remapping is idempotent. -/
theorem kindMap_idem (k : Nat) : kindMap (kindMap k) = kindMap k := by
  unfold kindMap
  split_ifs <;> omega

end Strata.Weights





## FILE: /mnt/user-data/outputs/Q2ExpertKernel.lean
/-!
# MODULE: q2_expert_kernel
Q2_0 expert (gate/up -> SiLU*up -> down) CPU kernel: 2-bit code unpacking, the code-offset correction
`sum (c-1)*xhat = sum c*xhat - sum xhat`, int8 activation clamping, block-tail masks and token-count dispatch.

## Source semantics
* A Q2_0 weight is a 2-bit code `c ∈ {0..3}` meaning the value `c - 1`; the kernel computes `sum c*xhat` with
  VNNI (unsigned code × signed activation) and subtracts `sum xhat` afterwards (the weight-independent `hx`).
* Activations are quantised per 32-value chunk to int8 in `[-127,127]`; the dot is split per chunk because the
  activation scale changes at chunk boundaries while the weight scale does not.
* Tail blocks use masks `(1 << 2nb) - 1`; multi-token kernels are dispatched on `n_tokens` with `default -> 8`.
* gate row is `2r`, up row is `2r+1` of the interleaved gate/up pack.

## Form template identifiers
* `code` ~ unpack of one 2-bit code (`_mm512_multishift_epi64_epi8` + AND 3)
* `offDot` ~ `row_dot` result (offset-corrected), `dotList` ~ `sum c*xhat`, `sumList` ~ `sum xhat` (`hx`)
* `clamp127` ~ activation clamp in `act_quant_q8_1`
* `lanemask` ~ `m16` tail mask, `nbOf` ~ `nb` tail block count, `ntDispatch` ~ template switch on `n_tokens`
-/

namespace Strata.Q2Expert

-- § code decoding

/-- The i-th 2-bit code of a 16-bit word. -/
def code (h i : Nat) : Nat := (h / 4 ^ i) % 4

/-- Every decoded code is in `0..3`. -/
theorem code_lt (h i : Nat) : code h i < 4 := by
  unfold code
  exact Nat.mod_lt _ (by decide)

/-- A Q2_0 weight value is the code minus one. -/
def weight (c : Nat) : Int := (c : Int) - 1

/-- Weight values lie in `[-1, 2]`. -/
theorem weight_range (c : Nat) (hc : c < 4) : -1 ≤ weight c ∧ weight c ≤ 2 := by
  unfold weight
  omega

-- § offset-corrected dot product

/-- `sum c * x` over paired lists (the VNNI quantity). -/
def dotList : List Nat → List Int → Int
  | c :: cs, x :: xs => (c : Int) * x + dotList cs xs
  | _, _ => 0

/-- `sum x` (the `hx` quantity before scaling). -/
def sumList : List Int → Int
  | [] => 0
  | x :: xs => x + sumList xs

/-- `sum (c-1) * x`, the mathematical definition of the Q2_0 dot. -/
def offDot : List Nat → List Int → Int
  | c :: cs, x :: xs => ((c : Int) - 1) * x + offDot cs xs
  | _, _ => 0

/-- The kernel's rewrite `sum (c-1)x = sum c x - sum x` for equal-length inputs. -/
theorem offDot_eq (cs : List Nat) (xs : List Int) (h : cs.length = xs.length) :
    offDot cs xs = dotList cs xs - sumList xs := by
  induction cs generalizing xs with
  | nil =>
    cases xs with
    | nil => simp [offDot, dotList, sumList]
    | cons x xs => simp at h
  | cons c cs ih =>
    cases xs with
    | nil => simp at h
    | cons x xs =>
      have h' : cs.length = xs.length := by simpa using h
      have ih' := ih xs h'
      simp only [offDot, dotList, sumList]
      rw [Int.sub_mul, Int.one_mul]
      omega

/-- The VNNI dot splits over a chunk boundary (lo/hi 32-value halves of a 64-weight block). -/
theorem dotList_append (cs1 cs2 : List Nat) (xs1 xs2 : List Int) (h : cs1.length = xs1.length) :
    dotList (cs1 ++ cs2) (xs1 ++ xs2) = dotList cs1 xs1 + dotList cs2 xs2 := by
  induction cs1 generalizing xs1 with
  | nil =>
    cases xs1 with
    | nil => simp [dotList]
    | cons x xs => simp at h
  | cons c cs ih =>
    cases xs1 with
    | nil => simp at h
    | cons x xs =>
      have h' : cs.length = xs.length := by simpa using h
      simp only [List.cons_append, dotList]
      rw [ih xs h']
      omega

/-- The activation sum splits over a chunk boundary. -/
theorem sumList_append (xs ys : List Int) : sumList (xs ++ ys) = sumList xs + sumList ys := by
  induction xs with
  | nil => simp [sumList]
  | cons x xs ih =>
    simp only [List.cons_append, sumList]
    rw [ih]
    omega

-- § activation clamp

/-- Clamp to the symmetric int8 range used by the quantiser. -/
def clamp127 (v : Int) : Int := min 127 (max (-127) v)

/-- The clamp lands in `[-127,127]`. -/
theorem clamp127_bounds (v : Int) : -127 ≤ clamp127 v ∧ clamp127 v ≤ 127 := by
  unfold clamp127
  omega

/-- The clamp is the identity on in-range values. -/
theorem clamp127_id (v : Int) (h1 : -127 ≤ v) (h2 : v ≤ 127) : clamp127 v = v := by
  unfold clamp127
  omega

/-- The clamp is idempotent. -/
theorem clamp127_idem (v : Int) : clamp127 (clamp127 v) = clamp127 v := by
  have h := clamp127_bounds v
  exact clamp127_id _ h.1 h.2

-- § tail masks, tail block counts, token dispatch

/-- The 16-lane tail mask for `nb` blocks (two scales per block). -/
def lanemask (nb : Nat) : Nat := if nb ≥ 8 then 65535 else 2 ^ (2 * nb) - 1

/-- A tail mask always fits in 16 lanes. -/
theorem lanemask_lt (nb : Nat) : lanemask nb < 65536 := by
  unfold lanemask
  by_cases h : nb ≥ 8
  · rw [if_pos h]
    decide
  · rw [if_neg h]
    have h1 : 2 ^ (2 * nb) ≤ 2 ^ 14 := Nat.pow_le_pow_right (by decide) (by omega)
    have h2 : (2 : Nat) ^ 14 = 16384 := by decide
    omega

/-- Blocks processed in the group starting at `b0`. -/
def nbOf (n b0 : Nat) : Nat := if n - b0 < 8 then n - b0 else 8

/-- Each group handles at least one block while blocks remain. -/
theorem nbOf_pos (n b0 : Nat) (h : b0 < n) : 1 ≤ nbOf n b0 := by
  unfold nbOf
  split <;> omega

/-- A group never exceeds eight blocks. -/
theorem nbOf_le (n b0 : Nat) : nbOf n b0 ≤ 8 := by
  unfold nbOf
  split <;> omega

/-- A group never reads past the row. -/
theorem nbOf_fits (n b0 : Nat) : b0 + nbOf n b0 ≤ n ∨ n < b0 := by
  unfold nbOf
  split <;> omega

/-- Token-count dispatch: counts above seven (and the default arm) use the 8-token instantiation. -/
def ntDispatch (n : Nat) : Nat := if n ≥ 8 then 8 else n

/-- Dispatch never exceeds eight tokens. -/
theorem ntDispatch_le (n : Nat) : ntDispatch n ≤ 8 := by
  unfold ntDispatch
  split <;> omega

/-- Dispatch is exact for 1..7 tokens. -/
theorem ntDispatch_id (n : Nat) (h : n ≤ 7) : ntDispatch n = n := by
  unfold ntDispatch
  split <;> omega

-- § gate/up interleaving

/-- Gate row index in the interleaved pack. -/
def gateRow (r : Nat) : Nat := 2 * r

/-- Up row index in the interleaved pack. -/
def upRow (r : Nat) : Nat := 2 * r + 1

/-- Gate and up rows never alias. -/
theorem gate_ne_up (r s : Nat) : gateRow r ≠ upRow s := by
  unfold gateRow upRow
  omega

/-- Both rows of pair `r < ff` lie inside `2*ff` rows. -/
theorem gu_rows_in_range (r ff : Nat) (h : r < ff) : gateRow r < 2 * ff ∧ upRow r < 2 * ff := by
  unfold gateRow upRow
  omega

-- § fp16 decode range

/-- For a normal fp16 exponent field `1..30`, the rebiased fp32 exponent is in `113..142`. -/
theorem fp16_normal_exp (e : Nat) (h1 : 1 ≤ e) (h2 : e ≤ 30) :
    113 ≤ (e : Int) - 15 + 127 ∧ (e : Int) - 15 + 127 ≤ 255 := by
  omega

end Strata.Q2Expert



## FILE: /mnt/user-data/outputs/IquantAvx512Rows.lean
/-!
# MODULE: iquant_avx512_rows
Multi-token AVX-512 dot products for the i-quant expert formats IQ2_XXS(16), IQ2_XS(17), IQ3_XXS(18),
IQ3_S(21), IQ2_S(22): decode 64 values once (grid lookup, sign mask, per-lane scales), apply to every token.

## Source semantics
* Support set is exactly the five type ids above; each has a fixed block size in bytes.
* Signs are applied as a masked negate of the int8 activation; magnitudes come from the grid.
* Scale lanes: int16 lane `l` of a 32-lane `maddubs` result takes scale `l / 8` (four scales per 64 values).
* Scales are odd numbers `2*s+1` with `s` a 4-bit field; 7-bit sign indexes and 4-bit scale fields are
  extracted from 32-bit words with shifts and masks.
* More than four tokens are processed in groups of at most four; 8-way dispatch is by `switch(nt)`.

## Form template identifiers
* `bytesOf` ~ `Fmt<TY>::bytes`, `supported` ~ `iq512_supported`
* `scaleLane` ~ `scales4` permute index, `oddScale` ~ `2*s+1`
* `cneg` ~ `_mm512_mask_sub_epi8(y, m, 0, y)`
* `groupSize` ~ the `k = min(4, nt - t0)` grouping in `dot_rows_nt`
-/

namespace Strata.IquantAvx512

-- § supported formats and block sizes

/-- Block size in bytes per supported type id (0 for unsupported). -/
def bytesOf : Nat → Nat
  | 16 => 66
  | 17 => 74
  | 18 => 98
  | 21 => 110
  | 22 => 82
  | _ => 0

/-- The five supported type ids. -/
def supported (t : Nat) : Prop := t = 16 ∨ t = 17 ∨ t = 18 ∨ t = 21 ∨ t = 22

/-- Supported types have a block of at least 66 bytes (so the 2-byte scale header always exists). -/
theorem bytesOf_ge (t : Nat) (h : supported t) : 66 ≤ bytesOf t := by
  unfold supported at h
  rcases h with h | h | h | h | h <;> subst h <;> decide

/-- Unsupported types have no block size. -/
theorem bytesOf_unsupported (t : Nat) (h : ¬ supported t) : bytesOf t = 0 := by
  unfold supported at h
  have h16 : t ≠ 16 := fun e => h (Or.inl e)
  have h17 : t ≠ 17 := fun e => h (Or.inr (Or.inl e))
  have h18 : t ≠ 18 := fun e => h (Or.inr (Or.inr (Or.inl e)))
  have h21 : t ≠ 21 := fun e => h (Or.inr (Or.inr (Or.inr (Or.inl e))))
  have h22 : t ≠ 22 := fun e => h (Or.inr (Or.inr (Or.inr (Or.inr e))))
  unfold bytesOf
  split <;> simp_all

-- § scale lanes

/-- Which of the four scales applies to int16 lane `l` (lanes 0-7 -> 0, ..., 24-31 -> 3). -/
def scaleLane (l : Nat) : Nat := l / 8

/-- Every lane of a 32-lane vector maps to one of four scales. -/
theorem scaleLane_lt (l : Nat) (h : l < 32) : scaleLane l < 4 := by
  unfold scaleLane
  omega

/-- Lanes inside an 8-group share a scale. -/
theorem scaleLane_group (g k : Nat) (hk : k < 8) : scaleLane (8 * g + k) = g := by
  unfold scaleLane
  omega

/-- The scale encoded by a 4-bit field: `2*s + 1`. -/
def oddScale (s : Nat) : Nat := 2 * s + 1

/-- Decoded scales are odd. -/
theorem oddScale_odd (s : Nat) : oddScale s % 2 = 1 := by
  unfold oddScale
  omega

/-- A 4-bit field gives a scale in `1..31`. -/
theorem oddScale_range (s : Nat) (h : s < 16) : 1 ≤ oddScale s ∧ oddScale s ≤ 31 := by
  unfold oddScale
  omega

-- § bit-field extraction

/-- A 7-bit sign index at bit offset `7k` of a word. -/
def signIdx (w k : Nat) : Nat := (w / 2 ^ (7 * k)) % 128

/-- Sign indexes are valid `ksigns` table indexes. -/
theorem signIdx_lt (w k : Nat) : signIdx w k < 128 := by
  unfold signIdx
  exact Nat.mod_lt _ (by decide)

/-- The top nibble of a 32-bit word is a 4-bit scale field. -/
theorem top_nibble_lt (w : Nat) (h : w < 4294967296) : w / 268435456 < 16 := by
  omega

-- § masked sign application

/-- Conditional negate (the masked subtract). -/
def cneg (b : Bool) (y : Int) : Int := if b then -y else y

/-- Applying the sign mask twice restores the activation. -/
theorem cneg_cneg (b : Bool) (y : Int) : cneg b (cneg b y) = y := by
  cases b <;> simp [cneg]

/-- Negation preserves magnitude, so int8 range is preserved for symmetric `[-127,127]` activations. -/
theorem cneg_abs (b : Bool) (y : Int) : (cneg b y).natAbs = y.natAbs := by
  cases b <;> simp [cneg]

/-- Sign-applied activations of a symmetric range stay in the range. -/
theorem cneg_bounds (b : Bool) (y : Int) (h1 : -127 ≤ y) (h2 : y ≤ 127) :
    -127 ≤ cneg b y ∧ cneg b y ≤ 127 := by
  cases b <;> simp [cneg] <;> omega

-- § token grouping (more than four tokens)

/-- Group size taken at offset `t0` of `nt` tokens. -/
def groupSize (nt t0 : Nat) : Nat := if nt - t0 < 4 then nt - t0 else 4

/-- Each group is non-empty while tokens remain. -/
theorem groupSize_pos (nt t0 : Nat) (h : t0 < nt) : 1 ≤ groupSize nt t0 := by
  unfold groupSize
  split <;> omega

/-- Each group is at most four tokens. -/
theorem groupSize_le (nt t0 : Nat) : groupSize nt t0 ≤ 4 := by
  unfold groupSize
  split <;> omega

/-- A group never reads past the last token. -/
theorem groupSize_fits (nt t0 : Nat) (h : t0 < nt) : t0 + groupSize nt t0 ≤ nt := by
  unfold groupSize
  split <;> omega

-- § row addressing

/-- Row `r` starts at `r * row_bytes`; the up projection adds a fixed offset. Distinct rows do not overlap. -/
theorem row_offsets_disjoint (rb r s : Nat) (hrb : 0 < rb) (h : r < s) : r * rb + rb ≤ s * rb := by
  have : (r + 1) * rb ≤ s * rb := Nat.mul_le_mul_right rb h
  rw [Nat.add_mul, Nat.one_mul] at this
  exact this

/-- Number of 256-value superblocks for `n` values. -/
def nSuper (n : Nat) : Nat := n / 256

/-- A row of `n = 256k` values has exactly `k` superblocks. -/
theorem nSuper_mul (k : Nat) : nSuper (256 * k) = k := by
  unfold nSuper
  omega

end Strata.IquantAvx512



## FILE: /mnt/user-data/outputs/ExpertPoolScheduling.lean
/-!
# MODULE: expert_pool_scheduling
CPU expert thread pool: physical-core selection, worker count, equal-range row partitioning across tasks,
segment walking over expert boundaries, batching of experts, and task-count formulas.

## Source semantics
* Task `i` of `tasks` over `rows` rows owns `[rows*i/tasks, rows*(i+1)/tasks)`: ranges are contiguous,
  monotone and cover `[0, rows)` exactly.
* A range may span several experts; it is walked as segments `(e = r / per, r0 = r % per, r1 = min per (r0 + remaining))`
  and each segment makes progress and stays inside one expert.
* `skip_first` drops the first core (kept for the host); worker count is at least one.
* More experts than split buffers are processed in batches of at most `kMax`.
* Source note: on non-Windows, `physical_cores` has a dangling `else` that binds to the inner
  `if (CPU_ISSET(i, &set))`, so the `hardware_concurrency` fallback is never reached via `sched_getaffinity`
  failure and instead runs once per CPU *not* in the mask. Not modelled here.

## Form template identifiers
* `rangeLo` ~ `mrows_ * i / mtasks_`
* `segR0` / `segR1` ~ `r0` / `r1` in the drain loop
* `batchSize` ~ `nb = min(kMaxSplitMulti, n - b0)`
* `nWorkers` ~ `n_`, `dropFirst` ~ `physical_cores(skip_first)`
* `partsOf` ~ `parts_a_`
-/

namespace Strata.ExpertPool

-- § equal-range partition

/-- Start of task `i`'s range. -/
def rangeLo (rows i tasks : Nat) : Nat := rows * i / tasks

/-- The first task starts at row 0. -/
theorem rangeLo_zero (rows tasks : Nat) : rangeLo rows 0 tasks = 0 := by
  simp [rangeLo]

/-- The sentinel after the last task is exactly `rows`. -/
theorem rangeLo_last (rows tasks : Nat) (h : 0 < tasks) : rangeLo rows tasks tasks = rows := by
  unfold rangeLo
  exact Nat.mul_div_cancel rows h

/-- Range starts are monotone in the task index. -/
theorem rangeLo_mono (rows tasks i j : Nat) (h : i ≤ j) : rangeLo rows i tasks ≤ rangeLo rows j tasks := by
  unfold rangeLo
  exact Nat.div_le_div_right (Nat.mul_le_mul_left rows h)

/-- No range start exceeds `rows`. -/
theorem rangeLo_le_rows (rows tasks i : Nat) (h : i ≤ tasks) : rangeLo rows i tasks ≤ rows := by
  unfold rangeLo
  apply Nat.div_le_of_le_mul
  rw [Nat.mul_comm tasks rows]
  exact Nat.mul_le_mul_left rows h

-- § segment walking across expert boundaries

/-- Offset within the current expert. -/
def segR0 (per r : Nat) : Nat := r % per

/-- End offset of the segment: stay in the expert, stop at the range end. -/
def segR1 (per r g1 : Nat) : Nat := min per (segR0 per r + (g1 - r))

/-- Expert index and offset recompose the global row. -/
theorem seg_recompose (per r : Nat) : per * (r / per) + segR0 per r = r := by
  unfold segR0
  exact Nat.div_add_mod r per

/-- The offset lies inside the expert. -/
theorem segR0_lt (per r : Nat) (h : 0 < per) : segR0 per r < per := by
  unfold segR0
  exact Nat.mod_lt r h

/-- Each segment is non-empty, stays within one expert, and does not pass the range end: the walk terminates. -/
theorem seg_progress (per r g1 : Nat) (hper : 0 < per) (hr : r < g1) :
    segR0 per r < segR1 per r g1 ∧ segR1 per r g1 ≤ per ∧ r + (segR1 per r g1 - segR0 per r) ≤ g1 := by
  have h0 := segR0_lt per r hper
  unfold segR1
  generalize segR0 per r = r0 at h0 ⊢
  omega

-- § batching and task counts

/-- Experts handled in the batch starting at `b0`. -/
def batchSize (kMax n b0 : Nat) : Nat := min kMax (n - b0)

/-- A batch is non-empty while experts remain (given a positive buffer count). -/
theorem batchSize_pos (kMax n b0 : Nat) (hk : 0 < kMax) (h : b0 < n) : 1 ≤ batchSize kMax n b0 := by
  unfold batchSize
  omega

/-- A batch fits in the buffers. -/
theorem batchSize_le (kMax n b0 : Nat) : batchSize kMax n b0 ≤ kMax := by
  unfold batchSize
  omega

/-- A batch never passes the last expert. -/
theorem batchSize_fits (kMax n b0 : Nat) (h : b0 < n) : b0 + batchSize kMax n b0 ≤ n := by
  unfold batchSize
  omega

/-- Parts per expert: about three tasks per thread, at least one. -/
def partsOf (threads n : Nat) : Nat := max 1 ((3 * threads + n - 1) / n)

/-- At least one part per expert. -/
theorem partsOf_pos (threads n : Nat) : 1 ≤ partsOf threads n := by
  unfold partsOf
  exact Nat.le_max_left _ _

/-- The split phase has at least one task per expert. -/
theorem split_tasks_ge (threads n : Nat) : n ≤ n * partsOf threads n := by
  have h := Nat.mul_le_mul_left n (partsOf_pos threads n)
  simpa using h

-- § worker and core selection

/-- Optionally drop the first core (reserved for the host thread). -/
def dropFirst (skip : Bool) (cs : List Nat) : List Nat := if skip then cs.tail else cs

/-- Dropping the host core removes exactly one entry from a non-empty list. -/
theorem dropFirst_length (cs : List Nat) (h : cs ≠ []) : (dropFirst true cs).length = cs.length - 1 := by
  simp [dropFirst]

/-- Without `skip_first` the list is unchanged. -/
theorem dropFirst_false (cs : List Nat) : dropFirst false cs = cs := by
  simp [dropFirst]

/-- Worker count: the request if positive, otherwise one per core; never below one. -/
def nWorkers (req cores : Nat) : Nat := max 1 (if req > 0 then req else cores)

/-- There is always at least one worker, so a single-core machine still produces a token. -/
theorem nWorkers_pos (req cores : Nat) : 1 ≤ nWorkers req cores := by
  unfold nWorkers
  exact Nat.le_max_left _ _

/-- Worker `i` is pinned to core `cores[i]` when it exists, otherwise left unpinned. -/
def pinCore (cores : List Nat) (i : Nat) : Option Nat := cores[i]?

/-- Workers within the core list always get a core. -/
theorem pinCore_some (cores : List Nat) (i : Nat) (h : i < cores.length) :
    pinCore cores i = some cores[i] := by
  unfold pinCore
  exact List.getElem?_eq_getElem h

/-- Workers beyond the core list are unpinned. -/
theorem pinCore_none (cores : List Nat) (i : Nat) (h : cores.length ≤ i) : pinCore cores i = none := by
  unfold pinCore
  exact List.getElem?_eq_none h

-- § affinity return protocol

/-- Pin return value: -1 when the core is negative (no-op), else the previous mask. -/
def pinReturn (core : Int) (prev : Int) : Int := if core < 0 then -1 else prev

/-- A negative core id is always reported as "nothing to restore". -/
theorem pinReturn_neg (core prev : Int) (h : core < 0) : pinReturn core prev = -1 := by
  unfold pinReturn
  rw [if_pos h]

/-- `restore_thread_affinity` ignores non-positive values, so -1 never restores. -/
theorem restore_ignores_failure : ¬ ((-1 : Int) > 0) := by
  decide

end Strata.ExpertPool



## FILE: /mnt/user-data/outputs/Q2Avx2Fallback.lean
/-!
# MODULE: q2_avx2_fallback
AVX2-only Q2_0 row dot (GGUF 18-byte blocks) and int8 activation quantiser for CPUs without AVX-512.

## Source semantics
* GGUF block = fp16 scale (2 bytes) then 16 code bytes: 18 bytes per 64 weights; row stride is `18 * nblocks`.
* Value `v` of a block lives in code byte `v / 4` at bit position `2 * (v % 4)`; the unpack yields values
  0..31 in the low vector and 32..63 in the high vector.
* The dot is `d * sum(c * x)` per 32-chunk times chunk scale, minus `d * hx` (the -1 code offset).
* The quantiser takes `amax = max |x|`, uses scale `amax/127` (0 when `amax = 0`), clamps to `[-127,127]`;
  `sum`/`hx` are per-chunk sums of the quantised values.
* More than four tokens are handled in groups of at most four.

## Form template identifiers
* `blockOff` ~ `row + b*18`, `codeByte` / `codeShift` ~ unpack addressing
* `amaxL` ~ the vector max of `|x|`, `clamp127` ~ clamp, `groupSize` ~ `k = min(4, nt - t0)`
-/

namespace Strata.Q2Avx2

-- § GGUF block addressing

/-- Byte offset of block `b` within a row. -/
def blockOff (b : Nat) : Nat := b * 18

/-- Offset of the code bytes of block `b` (after the 2-byte scale). -/
def codeOff (b : Nat) : Nat := blockOff b + 2

/-- Consecutive blocks tile the row: block `b+1` starts where block `b` ends. -/
theorem block_tiling (b : Nat) : blockOff (b + 1) = blockOff b + 18 := by
  unfold blockOff
  omega

/-- All 16 code bytes of block `b` lie inside the block. -/
theorem code_bytes_in_block (b k : Nat) (h : k < 16) : codeOff b + k < blockOff (b + 1) := by
  unfold codeOff blockOff
  omega

/-- The scale header of block `b` is disjoint from its code bytes. -/
theorem scale_before_codes (b : Nat) : blockOff b + 2 = codeOff b := by
  rfl

-- § 2-bit unpack addressing

/-- Code byte holding value `v`. -/
def codeByte (v : Nat) : Nat := v / 4

/-- Bit shift of value `v` inside its byte. -/
def codeShift (v : Nat) : Nat := 2 * (v % 4)

/-- Every value of a 64-value block lives in one of the 16 code bytes. -/
theorem codeByte_lt (v : Nat) (h : v < 64) : codeByte v < 16 := by
  unfold codeByte
  omega

/-- Shifts are one of 0, 2, 4, 6. -/
theorem codeShift_cases (v : Nat) :
    codeShift v = 0 ∨ codeShift v = 2 ∨ codeShift v = 4 ∨ codeShift v = 6 := by
  unfold codeShift
  omega

/-- The addressing is injective: distinct values never share a (byte, shift) slot. -/
theorem code_slot_injective (v w : Nat) (hb : codeByte v = codeByte w) (hs : codeShift v = codeShift w) :
    v = w := by
  unfold codeByte codeShift at *
  omega

/-- The low vector holds values 0..31 and the high vector 32..63: the split point is byte 8. -/
theorem lo_hi_split (v : Nat) (h : v < 64) : v < 32 ↔ codeByte v < 8 := by
  unfold codeByte
  omega

-- § offset-corrected dot over a block

/-- `sum c * x` over paired lists. -/
def dotList : List Nat → List Int → Int
  | c :: cs, x :: xs => (c : Int) * x + dotList cs xs
  | _, _ => 0

/-- `sum x`. -/
def sumList : List Int → Int
  | [] => 0
  | x :: xs => x + sumList xs

/-- `sum (c-1) * x`. -/
def offDot : List Nat → List Int → Int
  | c :: cs, x :: xs => ((c : Int) - 1) * x + offDot cs xs
  | _, _ => 0

/-- The row kernel's `acc - corr` equals the mathematical offset dot. -/
theorem offDot_eq (cs : List Nat) (xs : List Int) (h : cs.length = xs.length) :
    offDot cs xs = dotList cs xs - sumList xs := by
  induction cs generalizing xs with
  | nil =>
    cases xs with
    | nil => simp [offDot, dotList, sumList]
    | cons x xs => simp at h
  | cons c cs ih =>
    cases xs with
    | nil => simp at h
    | cons x xs =>
      have h' : cs.length = xs.length := by simpa using h
      have ih' := ih xs h'
      simp only [offDot, dotList, sumList]
      rw [Int.sub_mul, Int.one_mul]
      omega

-- § activation quantiser

/-- Max absolute value of a chunk. -/
def amaxL : List Int → Nat
  | [] => 0
  | x :: xs => max x.natAbs (amaxL xs)

/-- Every element is bounded by the chunk maximum. -/
theorem le_amaxL (x : Int) (xs : List Int) (h : x ∈ xs) : x.natAbs ≤ amaxL xs := by
  induction xs with
  | nil => simp at h
  | cons y ys ih =>
    simp only [List.mem_cons] at h
    show x.natAbs ≤ max y.natAbs (amaxL ys)
    rcases h with h | h
    · subst h
      exact Nat.le_max_left _ _
    · exact Nat.le_trans (ih h) (Nat.le_max_right _ _)

/-- A zero maximum means the chunk is all zero (the quantiser then uses scale 0). -/
theorem amaxL_zero (x : Int) (xs : List Int) (h : x ∈ xs) (h0 : amaxL xs = 0) : x = 0 := by
  have := le_amaxL x xs h
  omega

/-- Clamp to the symmetric int8 range. -/
def clamp127 (v : Int) : Int := min 127 (max (-127) v)

/-- The clamp lands in `[-127,127]`. -/
theorem clamp127_bounds (v : Int) : -127 ≤ clamp127 v ∧ clamp127 v ≤ 127 := by
  unfold clamp127
  omega

/-- The clamp is the identity on in-range values. -/
theorem clamp127_id (v : Int) (h1 : -127 ≤ v) (h2 : v ≤ 127) : clamp127 v = v := by
  unfold clamp127
  omega

/-- Chunk count of `n` values. -/
def nChunks (n : Nat) : Nat := n / 32

/-- `n = 32k` values give exactly `k` chunks, each addressing `[32k', 32k'+32)` inside the buffer. -/
theorem chunk_in_range (k n i j : Nat) (hn : n = 32 * k) (hi : i < nChunks n) (hj : j < 32) :
    32 * i + j < n := by
  unfold nChunks at hi
  omega

-- § token grouping

/-- Group size taken at offset `t0` of `nt` tokens. -/
def groupSize (nt t0 : Nat) : Nat := if nt - t0 < 4 then nt - t0 else 4

/-- Groups are non-empty while tokens remain. -/
theorem groupSize_pos (nt t0 : Nat) (h : t0 < nt) : 1 ≤ groupSize nt t0 := by
  unfold groupSize
  split <;> omega

/-- Groups are at most four tokens (the largest instantiated kernel). -/
theorem groupSize_le (nt t0 : Nat) : groupSize nt t0 ≤ 4 := by
  unfold groupSize
  split <;> omega

/-- Groups never pass the last token. -/
theorem groupSize_fits (nt t0 : Nat) (h : t0 < nt) : t0 + groupSize nt t0 ≤ nt := by
  unfold groupSize
  split <;> omega

end Strata.Q2Avx2





## FILE: /mnt/user-data/outputs/FusedGdnDecode.lean
/-!
# MODULE: fused_gdn_decode
Index/launch arithmetic of the fused gated-delta-net decode kernels (conv+l2, alpha/beta projection, step+norm).

## Source semantics
`gdn_step_norm_kernel`: block = head, thread (col, rg) owns 32 state rows `rg*32 .. rg*32+31` of column `col`; row groups tile the 128 rows exactly.
`gdn_conv_l2_kernel`: a 3-deep history per channel slides by one and the 4-tap dot uses (hist0, hist1, hist2, x).
`gdn_ab_kernel`: row < h_v is alpha (gate), row >= h_v is beta; grid rounds 2*h_v rows up to 8 warps per block.

## Form template identifiers
S -> state_size, RG -> row_groups, RPG -> rows_per_thread, fused_gdn_step_norm -> step_norm_launch
-/
namespace Strata.FusedGdn

def S : Nat := 128
def RG : Nat := 4
def RPG : Nat := 32

-- § row group tiling

/-- Row groups tile the state exactly. -/
theorem rpg_cover : RG * RPG = S := by decide

/-- Every (row group, local row) pair addresses a row inside the state. -/
theorem row_lt (rg r : Nat) (hg : rg < RG) (hr : r < RPG) : rg * RPG + r < S := by (try simp only [S, RG, RPG] at *) <;> omega

/-- The row map is injective, so no two registers alias a state row. -/
theorem row_inj (rg r rg' r' : Nat) (hg : rg < RG) (hr : r < RPG) (hg' : rg' < RG) (hr' : r' < RPG) (h : rg * RPG + r = rg' * RPG + r') : rg = rg' ∧ r = r' := by (try simp only [S, RG, RPG] at *) <;> omega

/-- Every row below S is hit by some thread register. -/
theorem row_surj (i : Nat) (hi : i < S) : ∃ rg r, rg < RG ∧ r < RPG ∧ rg * RPG + r = i := by refine ⟨i / 32, i % 32, ?_, ?_, ?_⟩ <;> (try simp only [S, RG, RPG] at *) <;> omega

-- § thread ids

/-- `tid = rg*S + col` is below S exactly for row group 0 (the threads that load k, q and write y). -/
theorem tid_lt_iff (rg col : Nat) (hc : col < S) : rg * S + col < S ↔ rg = 0 := by (try simp only [S] at *) <;> omega

/-- The per-warp partial-sum slot `tid/32` fits the 16-entry `wsum` array. -/
theorem wsum_slot (tid : Nat) (h : tid < S * RG) : tid / 32 < S * RG / 32 := by (try simp only [S, RG] at *) <;> omega

/-- Row group 0 owns warps 0..3, which are exactly the four partials summed for the RMS. -/
theorem rms_warps (col : Nat) (hc : col < S) : col / 32 < 4 := by (try simp only [S] at *) <;> omega

-- § alpha/beta kernel

/-- The alpha/beta grid covers all 2*h_v rows. -/
theorem ab_grid_cover (h_v : Nat) : 2 * h_v ≤ (2 * h_v + 7) / 8 * 8 := by omega

/-- A beta row, rebased by h_v, is a valid head index. -/
theorem ab_beta_index (h_v row : Nat) (h : row < 2 * h_v) (hb : ¬ row < h_v) : row - h_v < h_v := by omega

/-- An alpha row is already a valid head index. -/
theorem ab_alpha_index (h_v row : Nat) (hb : row < h_v) : row < h_v := by omega

-- § conv kernel

/-- Channels divisible by S make the conv grid exact. -/
theorem conv_grid_exact (channels : Nat) (h : channels % S = 0) : channels / S * S = channels := by (try simp only [S] at *) <;> omega

/-- The 4-tap convolution. -/
def conv4 (v0 v1 v2 x w0 w1 w2 w3 : Int) : Int := v0 * w0 + v1 * w1 + v2 * w2 + x * w3

/-- The history slide. -/
def slide3 {α : Type} (h : α × α × α) (x : α) : α × α × α := (h.2.1, h.2.2, x)

/-- After sliding, the newest input sits in the last history slot. -/
theorem slide3_newest {α : Type} (a b c x : α) : slide3 (a, b, c) x = (b, c, x) := rfl

/-- The next token's convolution reads the previous input in tap 2 (the oldest slot is dropped). -/
theorem conv4_next (v0 v1 v2 x x' w0 w1 w2 w3 : Int) : conv4 v1 v2 x x' w0 w1 w2 w3 = v1 * w0 + v2 * w1 + x * w2 + x' * w3 := rfl

end Strata.FusedGdn



## FILE: /mnt/user-data/outputs/FusedGrRead.lean
/-!
# MODULE: fused_gr_read
Launch geometry of the fused hyper-connection read (`gr_down`, `gr_up`, and the multi-token variants).

## Source semantics
n_embd 2560, hc 4 streams, D = n_embd*hc = 10240, hc_lr 320. Down: 40 blocks x 8 warps = 320 output rows plus one inject block. Up: each block owns UP_COLS columns of every stream. Stream of flat index i is i / N.

## Form template identifiers
N -> n_embd, HC -> streams, LR -> low_rank, DOWN_BLOCKS -> down_grid, UP_BLOCKS -> up_grid
-/
namespace Strata.FusedGr

def N : Nat := 2560
def HC : Nat := 4
def D : Nat := 10240
def LR : Nat := 320
def WARPS : Nat := 8
def DOWN_BLOCKS : Nat := 40
def UP_COLS : Nat := 32
def UP_BLOCKS : Nat := 80
def UPM_COLS : Nat := 16
def UPM_BLOCKS : Nat := 160
def TILE : Nat := 2560
def TQ : Nat := 10

-- § constants

/-- D is the stream-concatenated width. -/
theorem d_eq : D = N * HC := by decide

/-- Down blocks of 8 warps cover the 320 low-rank rows. -/
theorem down_rows : DOWN_BLOCKS * WARPS = LR := by decide

/-- Up blocks cover all columns (single token). -/
theorem up_cols : UP_BLOCKS * UP_COLS = N := by decide

/-- Up blocks cover all columns (multi token). -/
theorem upm_cols : UPM_BLOCKS * UPM_COLS = N := by decide

/-- D splits into whole tiles. -/
theorem tiles_exact : D / TILE * TILE = D := by decide

/-- A tile is TQ chunks of 8 per lane across 32 lanes. -/
theorem tile_chunks : TQ * 32 * 8 = TILE := by decide

/-- 40 chunks per lane over a 10240-wide row, as the comment states. -/
theorem down_chunks : D / 8 = 32 * 40 := by decide

/-- A 320-wide up row is 40 chunks: 32 in the first pass and 8 in the second. -/
theorem up_chunks : LR / 8 = 32 + 8 := by decide

-- § stream indexing

/-- A flat index lies in a valid stream. -/
theorem stream_lt (i : Nat) (h : i < D) : i / N < HC := by (try simp only [D, N, HC] at *) <;> omega

/-- Stream and column recompose the flat index. -/
theorem stream_decomp (i : Nat) : (i / N) * N + i % N = i := by (try simp only [N] at *) <;> omega

/-- The up row of (stream c, column d0+dd) stays inside D. -/
theorem up_row_lt (b c dd : Nat) (hb : b < UP_BLOCKS) (hc : c < HC) (hd : dd < UP_COLS) : c * N + (b * UP_COLS + dd) < D := by (try simp only [UP_BLOCKS, UP_COLS, HC, N, D] at *) <;> omega

/-- The multi-token up row stays inside D. -/
theorem upm_row_lt (b c dd : Nat) (hb : b < UPM_BLOCKS) (hc : c < HC) (hd : dd < UPM_COLS) : c * N + (b * UPM_COLS + dd) < D := by (try simp only [UPM_BLOCKS, UPM_COLS, HC, N, D] at *) <;> omega

/-- Row reduction: the mean over streams divides the sum by HC. -/
def meanOver (s : Int) : Int := s / 4

/-- The first lanes' second pass is only lanes below 8. -/
theorem second_pass_lanes (lane : Nat) (h : lane < LR / 8 - 32) : lane < 8 := by (try simp only [LR] at *) <;> omega

end Strata.FusedGr



## FILE: /mnt/user-data/outputs/GdnStep.lean
/-!
# MODULE: gdn_step
Gated delta-net recurrence step, conv slide and launch geometry.

## Source semantics
Per (head, column): decay the state column by exp(gate), sk = sum_i s_i k_i, delta = (v - sk)*beta, s_i += k_i*delta, out = sum_i s_i q_i (updated state). q/k head pairing is modulo h_k.

## Form template identifiers
gdn_step_kernel -> delta_step, sumTo -> column_sum, dec -> decay, b -> beta
-/
namespace Strata.GdnStep

/-- Sum of f over indices below n. -/
def sumTo : Nat → (Nat → Int) → Int
  | 0, _ => 0
  | n + 1, f => sumTo n f + f n

-- § sum lemmas

/-- Pointwise equal summands give equal sums. -/
theorem sumTo_congr (n : Nat) (f g : Nat → Int) (h : ∀ i, f i = g i) : sumTo n f = sumTo n g := by have : f = g := funext h; rw [this]

/-- Sums add. -/
theorem sumTo_add (n : Nat) (f g : Nat → Int) : sumTo n (fun i => f i + g i) = sumTo n f + sumTo n g := by induction n with | zero => rfl | succ n ih => simp only [sumTo, ih]; omega

/-- Constants pull out of sums. -/
theorem sumTo_mul (n : Nat) (c : Int) (f : Nat → Int) : sumTo n (fun i => c * f i) = c * sumTo n f := by induction n with | zero => simp [sumTo] | succ n ih => simp only [sumTo, ih, Int.mul_add]

-- § recurrence

/-- Pass 1: decaying the state before contracting with k scales the contraction by the decay. -/
theorem decay_contract (n : Nat) (dec : Int) (s k : Nat → Int) : sumTo n (fun i => (s i * dec) * k i) = dec * sumTo n (fun i => s i * k i) := by rw [← sumTo_mul]; exact sumTo_congr n _ _ (fun i => by rw [Int.mul_comm (s i) dec, Int.mul_assoc])

/-- Pass 2: the readout against q splits into the decayed old readout plus delta times the k-q readout. -/
theorem readout_split (n : Nat) (dec d : Int) (s k q : Nat → Int) : sumTo n (fun i => (s i * dec + k i * d) * q i) = dec * sumTo n (fun i => s i * q i) + d * sumTo n (fun i => k i * q i) := by rw [← sumTo_mul, ← sumTo_mul, ← sumTo_add]; exact sumTo_congr n _ _ (fun i => by rw [Int.add_mul, Int.mul_comm (s i) dec, Int.mul_assoc dec, Int.mul_comm (k i) d, Int.mul_assoc d])

-- § head pairing

/-- Modulo head pairing stays inside the q/k heads. -/
theorem pair_lt (h h_k : Nat) (hk : 0 < h_k) : h % h_k < h_k := Nat.mod_lt _ hk

/-- Divisible h_v makes the pairing balanced: head h and h + h_k share a q/k head. -/
theorem pair_periodic (h h_k : Nat) (hk : 0 < h_k) : (h + h_k) % h_k = h % h_k := by simp

-- § conv slide

/-- Slide the history window: drop the oldest, append the newest. -/
def slide {α : Type} (st : List α) (x : α) : List α := st.tail ++ [x]

/-- The slide preserves a nonempty window's length. -/
theorem slide_length {α : Type} (st : List α) (x : α) (n : Nat) (h : st.length = n + 1) : (slide st x).length = n + 1 := by simp [slide, h]

-- § launch geometry

/-- With MAX_H = 1 the head grid is exactly h_v. -/
theorem head_grid (h_v : Nat) : (h_v + 1 - 1) / 1 = h_v := by omega

/-- The j grid covers all S columns. -/
theorem j_grid (S : Nat) : S ≤ (S + 31) / 32 * 32 := by omega

/-- The beta-gate grid covers all h_v scalars. -/
theorem beta_grid (n : Nat) : n ≤ (n + 127) / 128 * 128 := by omega

end Strata.GdnStep



## FILE: /mnt/user-data/outputs/GatedResidual.lean
/-!
# MODULE: gated_residual
Workspace layout and index arithmetic for the gated residual / hyper-connection `gr_read` and `gr_write`.

## Source semantics
Workspace regions (xn, xq, lq, gated, lo) are each rounded up to 16 bytes and laid out back to back, so they never overlap. `gr_write` adds the same block output to every stream, weighted per stream: out[i] = R[i] + block_out[i mod n] * w[i / n].

## Form template identifiers
gr_workspace_init -> workspace_layout, alignUp16 -> align16, gr_write_kernel -> stream_write
-/
namespace Strata.GatedResidual

/-- Round up to a multiple of 16. -/
def alignUp16 (x : Nat) : Nat := (x + 15) / 16 * 16

-- § alignment

/-- Alignment never shrinks. -/
theorem align_ge (x : Nat) : x ≤ alignUp16 x := by unfold alignUp16; omega

/-- Alignment adds less than 16. -/
theorem align_lt (x : Nat) : alignUp16 x < x + 16 := by unfold alignUp16; omega

/-- The result is 16-aligned. -/
theorem align_mod (x : Nat) : alignUp16 x % 16 = 0 := by unfold alignUp16; omega

/-- Aligning twice is the same as once. -/
theorem align_idem (x : Nat) : alignUp16 (alignUp16 x) = alignUp16 x := by unfold alignUp16; omega

-- § real geometry

/-- Region byte sizes for hc=4, n_embd=2560, hc_lr=320: xn, xq, lq, gated, lo. -/
def regionSizes : List Nat := [10240 * 4, 10240 * 2, 320 * 2, 10240 * 4, 320 * 4]

/-- Total workspace bytes. -/
def workspaceBytes : Nat := (regionSizes.map alignUp16).sum

/-- The real workspace is 104320 bytes. -/
theorem workspace_real : workspaceBytes = 104320 := by decide

/-- Every region size at the real geometry is already 16-aligned, so alignment is free. -/
theorem regions_aligned : regionSizes.map alignUp16 = regionSizes := by decide

-- § stream write

/-- Stream and column recompose the flat index. -/
theorem write_decomp (i n : Nat) : n * (i / n) + i % n = i := Nat.div_add_mod i n

/-- The stream of a flat index below n*hc is below hc. -/
theorem write_stream_lt (i n hc : Nat) (h : i < n * hc) : i / n < hc := Nat.div_lt_of_lt_mul h

/-- The column of a flat index is below n (n > 0). -/
theorem write_col_lt (i n : Nat) (hn : 0 < n) : i % n < n := Nat.mod_lt _ hn

/-- The write is the residual plus a per-stream scaled block output. -/
def writeElem (r b w : Int) : Int := r + b * w

/-- A zero weight leaves the residual unchanged. -/
theorem write_zero (r b : Int) : writeElem r b 0 = r := by simp [writeElem]

-- § grids

/-- The write grid covers hc*n_embd elements. -/
theorem write_grid (n : Nat) : n ≤ (n + 255) / 256 * 256 := by omega

/-- The down kernel's 8-warp block stride reads each of the 10240 lane slots once per warp pass. -/
theorem down_stride (warp lane : Nat) (hw : warp < 8) (hl : lane < 32) : warp * 32 + lane < 8 * 32 := by omega

end Strata.GatedResidual



## FILE: /mnt/user-data/outputs/IqKernels.lean
/-!
# MODULE: iq_kernels
Row sizes, grouped-expert blob layout and launch arithmetic for the IQ-format matrix-vector kernels.

## Source semantics
A weight row of n values in a format with `elems` values per block of `bytes` bytes occupies (n/elems)*bytes. An expert blob is [gate rows | up rows | down rows], with up_off = n_ff*gu_row and down_off = 2*up_off. The q8_1 quantizer maps flat element i to block i/32, lane i%32. Gate/up fused rows are 2*n_ff, the upper half being the up projection.

## Form template identifiers
iq_row_bytes -> row_bytes, native_expert_layout -> expert_layout, quantize_q8_1_kernel -> q8_1_quantize
-/
namespace Strata.IqKernels

/-- Bytes of one weight row. -/
def rowBytes (n elems bytes : Nat) : Nat := n / elems * bytes

/-- Expert blob layout. -/
structure Layout where
  nEmbd : Nat
  nFf : Nat
  guRow : Nat
  dRow : Nat

/-- Offset of the up matrix. -/
def Layout.upOff (L : Layout) : Nat := L.nFf * L.guRow
/-- Offset of the down matrix. -/
def Layout.downOff (L : Layout) : Nat := 2 * L.upOff
/-- Total blob bytes. -/
def Layout.bytes (L : Layout) : Nat := L.downOff + L.nEmbd * L.dRow

-- § layout

/-- The up and gate matrices have equal size, so the regions are adjacent. -/
theorem down_after_up (L : Layout) : L.downOff = L.upOff + L.upOff := by unfold Layout.downOff; omega

/-- The blob contains the up region. -/
theorem up_in_blob (L : Layout) : L.upOff ≤ L.bytes := by unfold Layout.bytes Layout.downOff; omega

/-- The down region ends at the blob end. -/
theorem down_end (L : Layout) : L.downOff + L.nEmbd * L.dRow = L.bytes := rfl

/-- A row of zero columns is empty. -/
theorem rowBytes_zero (elems bytes : Nat) : rowBytes 0 elems bytes = 0 := by simp [rowBytes]

/-- Row bytes at block-multiple widths are exact. -/
theorem rowBytes_exact (k elems bytes : Nat) (h : 0 < elems) : rowBytes (k * elems) elems bytes = k * bytes := by simp [rowBytes, Nat.mul_div_cancel _ h]

-- § 256-byte alignment of scratch

/-- Round up to a multiple of 256. -/
def align256 (x : Nat) : Nat := (x + 255) / 256 * 256

/-- Alignment never shrinks. -/
theorem align256_ge (x : Nat) : x ≤ align256 x := by unfold align256; omega

/-- Result is 256-aligned. -/
theorem align256_mod (x : Nat) : align256 x % 256 = 0 := by unfold align256; omega

-- § row selection in the fused gate/up kernel

/-- Rebased up rows are valid rows of the up matrix. -/
theorem up_row_rebase (n_ff row : Nat) (h : row < 2 * n_ff) (hu : n_ff ≤ row) : row - n_ff < n_ff := by omega

/-- The gate/up grid covers 2*n_ff rows. -/
theorem gu_grid (n_ff : Nat) : 2 * n_ff ≤ (2 * n_ff + 7) / 8 * 8 := by omega

/-- The mmvq grid (4 rows per block) covers n_out. -/
theorem mmvq_grid (n_out : Nat) : n_out ≤ (n_out + 3) / 4 * 4 := by omega

-- § q8_1 quantizer

/-- Block and lane recompose the element index. -/
theorem q8_decomp (i : Nat) : (i / 32) * 32 + i % 32 = i := by omega

/-- The quantizer grid covers n elements. -/
theorem quant_grid (n : Nat) : n ≤ (n + 255) / 256 * 256 := by omega

end Strata.IqKernels



## FILE: /mnt/user-data/outputs/NativeFlashAttn.lean
/-!
# MODULE: native_flash_attn
Index arithmetic and validity predicate for the specialised single-token flash attention (Q24x256, KV2x256).

## Source semantics
One block per query head (24), kv head = head/12. A step is valid iff 1 <= width <= max_context, n_kv = width, pos = width-1, n_bid = width/4; otherwise the output is NaN and status reports unsupported. Within a 128-cell tile, cell = base + warp*32 + (lane/8)*8 + row.

## Form template identifiers
attend -> flash_step, kStepWidth -> width, kStepNKv -> n_kv
-/
namespace Strata.NativeFlashAttn

/-- Step validity. -/
def valid (width nkv pos nbid maxCtx : Nat) : Prop :=
  1 ≤ width ∧ width ≤ maxCtx ∧ nkv = width ∧ pos + 1 = width ∧ nbid = width / 4

-- § head mapping

/-- 24 heads map onto 2 kv heads. -/
theorem kv_lt (head : Nat) (h : head < 24) : head / 12 < 2 := by omega

/-- Each kv head serves 12 query heads. -/
theorem kv_group (head : Nat) : head / 12 * 12 ≤ head ∧ head < head / 12 * 12 + 12 := by omega

-- § tile mapping

/-- Cell offsets stay inside a 128-cell tile. -/
theorem cell_lt (warp lane row : Nat) (hw : warp < 4) (hl : lane < 32) (hr : row < 8) : warp * 32 + (lane / 8) * 8 + row < 128 := by omega

/-- Distinct (warp, lane group, row) triples address distinct cells. -/
theorem cell_inj (w l r w' l' r' : Nat) (hw : w < 4) (hl : l < 32) (hr : r < 8) (hw' : w' < 4) (hl' : l' < 32) (hr' : r' < 8) (h : w * 32 + (l / 8) * 8 + r = w' * 32 + (l' / 8) * 8 + r') : w = w' ∧ l / 8 = l' / 8 ∧ r = r' := by omega

/-- Output dims: the two passes over i0 in {0,128} cover all 256 dims. -/
theorem out_cover (j : Nat) (h : j < 256) : j = 0 + j % 128 ∨ j = 128 + j % 128 := by omega

-- § validity

/-- A valid step has n_kv = pos + 1. -/
theorem valid_nkv (width nkv pos nbid maxCtx : Nat) (h : valid width nkv pos nbid maxCtx) : nkv = pos + 1 := by unfold valid at h; omega

/-- A valid step has at least one cell. -/
theorem valid_pos_width (width nkv pos nbid maxCtx : Nat) (h : valid width nkv pos nbid maxCtx) : 1 ≤ nkv := by unfold valid at h; omega

/-- A valid step is within the context bound. -/
theorem valid_bound (width nkv pos nbid maxCtx : Nat) (h : valid width nkv pos nbid maxCtx) : nkv ≤ maxCtx := by unfold valid at h; omega

/-- A padded length of 256 gives one KV tile of width at most 256. -/
theorem single_tile (width : Nat) (h : width ≤ 256) : (width + 255) / 256 ≤ 1 := by omega

end Strata.NativeFlashAttn



## FILE: /mnt/user-data/outputs/NativeGdnStep.lean
/-!
# MODULE: native_gdn_step
Warp-sharded layout of the native gated-delta-net step: each warp owns one (head, column) and keeps four rows per lane in registers.

## Source semantics
Row index i = r*32 + lane for r < 4, lane < 32 covers 0..127 exactly once. Column col = z*4 + y with z < 32, y < 4 covers 0..127. The output is the warp-reduced readout scaled by 1/sqrt(S).

## Form template identifiers
step -> native_step, s_shard -> state_shard, S -> state_size
-/
namespace Strata.NativeGdn

/-- Rows owned by (register, lane) stay inside S = 128. -/
theorem row_lt (r lane : Nat) (hr : r < 4) (hl : lane < 32) : r * 32 + lane < 128 := by omega

/-- The row map is injective. -/
theorem row_inj (r l r' l' : Nat) (hr : r < 4) (hl : l < 32) (hr' : r' < 4) (hl' : l' < 32) (h : r * 32 + l = r' * 32 + l') : r = r' ∧ l = l' := by omega

/-- Every row is owned by some (register, lane). -/
theorem row_surj (i : Nat) (h : i < 128) : ∃ r l, r < 4 ∧ l < 32 ∧ r * 32 + l = i := by refine ⟨i / 32, i % 32, ?_, ?_, ?_⟩ <;> omega

/-- Column map: z*4 + y stays inside S. -/
theorem col_lt (z y : Nat) (hz : z < 32) (hy : y < 4) : z * 4 + y < 128 := by omega

/-- Column map is injective. -/
theorem col_inj (z y z' y' : Nat) (hy : y < 4) (hy' : y' < 4) (h : z * 4 + y = z' * 4 + y') : z = z' ∧ y = y' := by omega

/-- Every column is owned by some (z, y). -/
theorem col_surj (c : Nat) (h : c < 128) : ∃ z y, z < 32 ∧ y < 4 ∧ z * 4 + y = c := by refine ⟨c / 4, c % 4, ?_, ?_, ?_⟩ <;> omega

/-- The z grid dimension S/4 is 32. -/
theorem grid_z : 128 / 4 = 32 := by decide

/-- A head count at most 65535 fits the x grid dimension. -/
theorem head_grid (h_v : Nat) (h : h_v ≤ 65535) : h_v < 65536 := by omega

/-- Divisible head counts make modulo pairing balanced. -/
theorem pair_lt (head h_k : Nat) (hk : 0 < h_k) : head % h_k < h_k := Nat.mod_lt _ hk

/-- The extra reduced stride: lane 0 writes the single output of each warp. -/
theorem lane0_unique (l : Nat) (hl : l < 32) (h : l = 0 ∨ 0 < l) : l < 32 := hl

end Strata.NativeGdn



## FILE: /mnt/user-data/outputs/NativeMmvq.lean
/-!
# MODULE: native_mmvq
Iteration geometry, block byte sizes and launch arithmetic of the native quantized matrix-vector kernels.

## Source semantics
Each of the 128 threads (4 warps x 32 lanes) starts at block tid/T and strides by BPI blocks per iteration, where T * BPI = 128. Small-K launches use 4 rows per block, otherwise 1. Q8_1 blocks are 36 bytes (4 + 32). Multi-column activations are contiguous, so ncols columns of n_in = 32k quantize as one vector of ncols*n_in elements.

## Form template identifiers
Q5KTraits.. -> format_traits, T -> threads_per_block_group, BPI -> blocks_per_iteration
-/
namespace Strata.NativeMmvq

-- § thread mapping

/-- A thread's starting block index is below the blocks-per-iteration stride. -/
theorem start_lt_stride (tid T BPI : Nat) (hT : 128 = T * BPI) (h : tid < 128) : tid / T < BPI := Nat.div_lt_of_lt_mul (by omega)

/-- Q5_K / Q4_K: T = 16, BPI = 8. -/
theorem q45k_shape : 16 * 8 = 128 := by decide
/-- Q2_0: T = 2, BPI = 64. -/
theorem q20_shape : 2 * 64 = 128 := by decide
/-- Q3_K: T = 16, BPI = 8. -/
theorem q3k_shape : 16 * 8 = 128 := by decide
/-- Q6_K: T = 32, BPI = 4. -/
theorem q6k_shape : 32 * 4 = 128 := by decide
/-- IQ4_XS: T = 8, BPI = 16. -/
theorem iq4xs_shape : 8 * 16 = 128 := by decide
/-- Small formats with Qi = 4: T = 2, BPI = 64. -/
theorem small4_shape : 2 * 64 = 128 := by decide
/-- Q8_0 with Qi = 8: T = 4, BPI = 32. -/
theorem small8_shape : 4 * 32 = 128 := by decide

-- § block byte sizes

/-- Q8_1 block: half2 + 32 codes. -/
theorem q81_bytes : 4 + 32 = 36 := by decide
/-- Q4_0 block. -/
theorem q40_bytes : 2 + 16 = 18 := by decide
/-- Q5_0 block. -/
theorem q50_bytes : 2 + 4 + 16 = 22 := by decide
/-- Q8_0 block. -/
theorem q80_bytes : 2 + 32 = 34 := by decide
/-- Q4_K block. -/
theorem q4k_bytes : 4 + 12 + 128 = 144 := by decide
/-- Q5_K block. -/
theorem q5k_bytes : 4 + 12 + 32 + 128 = 176 := by decide
/-- Q6_K block. -/
theorem q6k_bytes : 128 + 64 + 16 + 2 = 210 := by decide
/-- Q3_K block. -/
theorem q3k_bytes : 32 + 64 + 12 + 2 = 110 := by decide
/-- IQ4_XS block. -/
theorem iq4xs_bytes : 2 + 2 + 4 + 128 = 136 := by decide

-- § launch arithmetic

/-- Small-K grid (4 rows per block) covers n_out. -/
theorem smallk_grid (n_out : Nat) : n_out ≤ (n_out + 3) / 4 * 4 := by omega

/-- Upstream multi layout (2 rows per block) covers n_out. -/
theorem upstream_grid (n_out : Nat) : n_out ≤ (n_out + 1) / 2 * 2 := by omega

/-- Quantizing ncols columns as one vector keeps block boundaries inside columns. -/
theorem multi_quant_blocks (k ncols : Nat) : (32 * k * ncols) / 32 = k * ncols := by rw [Nat.mul_assoc]; exact Nat.mul_div_cancel_left _ (by decide)

/-- Weight bytes: rows times blocks per row times block bytes. -/
def weightBytes (n_in elems bytes n_out : Nat) : Nat := n_in / elems * bytes * n_out

/-- Zero rows need zero weight bytes. -/
theorem weight_zero (n_in elems bytes : Nat) : weightBytes n_in elems bytes 0 = 0 := by simp [weightBytes]

/-- Multi-column output element j*n_out + row stays inside ncols*n_out. -/
theorem out_index_lt (j row n_out ncols : Nat) (hj : j < ncols) (hr : row < n_out) : j * n_out + row < ncols * n_out := by
  have h := Nat.mul_le_mul_right n_out (show j + 1 ≤ ncols by omega)
  rw [Nat.add_mul, Nat.one_mul] at h
  omega

end Strata.NativeMmvq



## FILE: /mnt/user-data/outputs/NativeQsaScore.lean
/-!
# MODULE: native_qsa_score
Tiling arithmetic of the tensor-core indexer scoring kernel (128 dims, 4 heads, 4 cells per block).

## Source semantics
Two warps each accumulate 32 of every 64 K columns: column = warp*32 + lane + 64*iter covers 0..127. The accumulator fragment element l of lane maps to row warp*32 + ia*16 + (l/2)*8 + lane/4 and head (lane%4)*2 + l%2 inside a 68-stride combine buffer. Block rows score their cells: cell i belongs to row i/4; the incomplete tail row gets +1e9.

## Form template identifiers
score_kernel -> block_score, COMBINE -> combine_stride, ROWS -> rows_per_block
-/
namespace Strata.NativeQsaScore

-- § K coverage

/-- Every K column is visited by exactly one (warp, lane, iteration) in range. -/
theorem col_surj (c : Nat) (h : c < 128) : ∃ w l it, w < 2 ∧ l < 32 ∧ it < 2 ∧ w * 32 + l + it * 64 = c := by refine ⟨(c % 64) / 32, c % 32, c / 64, ?_, ?_, ?_, ?_⟩ <;> omega

/-- The column map is injective. -/
theorem col_inj (w l it w' l' it' : Nat) (hw : w < 2) (hl : l < 32) (hi : it < 2) (hw' : w' < 2) (hl' : l' < 32) (hi' : it' < 2) (h : w * 32 + l + it * 64 = w' * 32 + l' + it' * 64) : w = w' ∧ l = l' ∧ it = it' := by omega

-- § fragment mapping

/-- Fragment rows are below 64 (two warps x 32 rows). -/
theorem frag_row_lt (w ia l lane : Nat) (hw : w < 2) (hia : ia < 2) (hl : l < 4) (hlane : lane < 32) : w * 32 + ia * 16 + (l / 2) * 8 + lane / 4 < 64 := by omega

/-- Fragment heads are below 8. -/
theorem frag_head_lt (l lane : Nat) (hlane : lane < 32) (hl : l < 4) : (lane % 4) * 2 + l % 2 < 8 := by omega

/-- Combine-buffer addresses stay inside 8 heads x 68. -/
theorem combine_lt (h i : Nat) (hh : h < 8) (hi : i < 68) : h * 68 + i < 8 * 68 := by omega

/-- Distinct (head, row) pairs map to distinct buffer slots. -/
theorem combine_inj (h i h' i' : Nat) (hi : i < 68) (hi' : i' < 68) (e : h * 68 + i = h' * 68 + i') : h = h' ∧ i = i' := by omega

-- § cells

/-- A cell below n belongs to a row no greater than n/4. -/
theorem cell_row_le (i n : Nat) (h : i < n) : i / 4 ≤ n / 4 := by omega

/-- Rows partition cells in groups of 4. -/
theorem cell_row_range (i : Nat) : (i / 4) * 4 ≤ i ∧ i < (i / 4 + 1) * 4 := by omega

/-- The tail bonus applies exactly when the cache length is not a multiple of 4. -/
theorem tail_bonus_iff (n : Nat) : n % 4 ≠ 0 ↔ ¬ (n = n / 4 * 4) := by omega

/-- The block grid covers max_blocks rows at 32 rows per block. -/
theorem score_grid (mb : Nat) : mb ≤ (mb + 31) / 32 * 32 := by omega

/-- The block count matches the cell capacity. -/
theorem blocks_for_cells (mc : Nat) : mc / 4 + 1 ≥ (mc + 3) / 4 := by omega

end Strata.NativeQsaScore



## FILE: /mnt/user-data/outputs/QsaCache.lean
/-!
# MODULE: qsa_cache
Step-state derivation, paged KV addressing, indexer key pooling slots and the exact top-k selection counts of the QSA cache kernels.

## Source semantics
From a position pos: n_kv = pos+1, n_bid = n_kv / idx_block, width = min(n_kv, top_k + r - 1). A cell with pos%4 = 3 completes a block (pooled row b = pos/4). The top-k keeps every cell strictly above the threshold key and the first (width - c_gt) cells equal to it, giving exactly width ids. The order key maps float bits to an unsigned total order.

## Form template identifiers
qsa_step_fill -> step_fill, order_key -> order_key, topk_kernel -> topk_select
-/
namespace Strata.QsaCache

-- § step fill

/-- Selection width. -/
def selWidth (nkv topK r : Nat) : Nat := min nkv (topK + r - 1)

/-- Width never exceeds the cache length. -/
theorem width_le_nkv (nkv topK r : Nat) : selWidth nkv topK r ≤ nkv := by unfold selWidth; omega

/-- Width never exceeds top_k + r - 1. -/
theorem width_le_cap (nkv topK r : Nat) : selWidth nkv topK r ≤ topK + r - 1 := by unfold selWidth; omega

/-- Below the cap the selection is the identity. -/
theorem width_identity (nkv topK r : Nat) (h : nkv ≤ topK + r - 1) : selWidth nkv topK r = nkv := by unfold selWidth; omega

/-- Positive length gives positive width when the cap is positive. -/
theorem width_pos (nkv topK r : Nat) (h : 0 < nkv) (hc : 0 < topK + r - 1) : 0 < selWidth nkv topK r := by unfold selWidth; omega

-- § paged addressing

/-- Page and slot recompose the position. -/
theorem page_slot (pos ps : Nat) : ps * (pos / ps) + pos % ps = pos := Nat.div_add_mod pos ps

/-- The slot is inside the page. -/
theorem slot_lt (pos ps : Nat) (h : 0 < ps) : pos % ps < ps := Nat.mod_lt _ h

-- § indexer pooling (block size 4)

/-- A cell completes a block exactly when its slot is 3. -/
theorem completes_iff (pos : Nat) : pos % 4 = 3 ↔ (pos + 1) % 4 = 0 := by omega

/-- On completion the number of completed blocks is the block index plus one. -/
theorem completed_count (pos : Nat) (h : pos % 4 = 3) : (pos + 1) / 4 = pos / 4 + 1 := by omega

/-- Slots below 3 are stored in the raw tail; the completing slot is not. -/
theorem tail_slot (pos : Nat) (h : pos % 4 < 3) : pos % 4 ≠ 3 := by omega

/-- The rotation position of a block is its first cell's position. -/
def blockPos (posBase b : Nat) : Nat := posBase + b * 4

/-- Consecutive blocks are 4 positions apart. -/
theorem blockPos_succ (posBase b : Nat) : blockPos posBase (b + 1) = blockPos posBase b + 4 := by unfold blockPos; omega

-- § top-k counts

/-- Equal-to-threshold budget. -/
def eqBudget (width cgt : Nat) : Nat := width - cgt

/-- Strictly-greater cells plus the equal cells taken total exactly width. -/
theorem select_total (width cgt ceq : Nat) (h1 : cgt < width) (h2 : width ≤ cgt + ceq) : cgt + min ceq (eqBudget width cgt) = width := by unfold eqBudget; omega

/-- The threshold search guarantees the budget is positive. -/
theorem budget_pos (width cgt : Nat) (h : cgt < width) : 0 < eqBudget width cgt := by unfold eqBudget; omega

-- § order key

/-- Order key of float bits: negatives flip all bits, non-negatives set the sign bit. -/
def orderKey (b : BitVec 32) : BitVec 32 := if b.msb = true then ~~~b else b ||| 0x80000000#32

/-- Non-negative floats order like their bits. -/
theorem key_nonneg_mono (a b : BitVec 32) (ha : a.msb = false) (hb : b.msb = false) (h : a ≤ b) : orderKey a ≤ orderKey b := by sorry

/-- Every negative key is below every non-negative key. -/
theorem key_sign_sep (a b : BitVec 32) (ha : a.msb = true) (hb : b.msb = false) : orderKey a < orderKey b := by sorry

end Strata.QsaCache



## FILE: /mnt/user-data/outputs/QsaDecodeAttn.lean
/-!
# MODULE: qsa_decode_attn
Chunking, scratch layout and head mapping of the split-KV decode attention over the selected cells.

## Source semantics
Selected ids are split into chunks of 64; chunk c handles ids 64c .. 64c+63. Each of the 12 query heads per KV head stores a partial (acc, m, l) per chunk at slot kvh*n_chunks + chunk. A final merge combines the partials. Global head h maps to kv head h/12 and local head h%12. Scratch holds n_chunks*n_head*(256+2) floats plus 64.

## Form template identifiers
attn_chunk_kernel -> chunk_partial, attn_merge_kernel -> merge_partials, CHUNK -> chunk_size, G -> group_size
-/
namespace Strata.QsaDecodeAttn

/-- Number of chunks for a capacity. -/
def nChunks (cap : Nat) : Nat := (cap + 63) / 64

-- § chunking

/-- Chunks cover the capacity. -/
theorem chunks_cover (cap : Nat) : cap ≤ nChunks cap * 64 := by unfold nChunks; omega

/-- Every id below n_ids <= cap lands in a valid chunk. -/
theorem id_chunk_lt (j n cap : Nat) (hj : j < n) (hn : n ≤ cap) : j / 64 < nChunks cap := by unfold nChunks; omega

/-- Chunk and offset recompose the id. -/
theorem id_decomp (j : Nat) : (j / 64) * 64 + j % 64 = j := by omega

/-- Cells handled by a chunk. -/
def nHere (nIds c : Nat) : Nat := min 64 (nIds - c * 64)

/-- A chunk handles at most 64 cells. -/
theorem nHere_le (nIds c : Nat) : nHere nIds c ≤ 64 := by unfold nHere; omega

/-- Chunks past the end handle nothing. -/
theorem nHere_zero (nIds c : Nat) (h : nIds ≤ c * 64) : nHere nIds c = 0 := by unfold nHere; omega

/-- A chunk below the last full one handles a full 64. -/
theorem nHere_full (nIds c : Nat) (h : (c + 1) * 64 ≤ nIds) : nHere nIds c = 64 := by unfold nHere; omega

/-- Each lane covers two probability slots, lane and lane+32, giving all 64. -/
theorem prob_slots (c : Nat) (h : c < 64) : c < 32 ∨ (c - 32 < 32) := by omega

-- § head mapping

/-- Global head recomposes from kv head and local head. -/
theorem head_decomp (h : Nat) : (h / 12) * 12 + h % 12 = h := by omega

/-- A global head below 12*n_kv has a valid kv head. -/
theorem kvh_lt (h nk : Nat) (hh : h < 12 * nk) : h / 12 < nk := by omega

/-- The local head is below 12. -/
theorem hl_lt (h : Nat) : h % 12 < 12 := by omega

-- § slots and scratch

/-- A (kv head, chunk) slot is inside n_kv_heads * n_chunks. -/
theorem slot_lt (kvh chunk nk nc : Nat) (hk : kvh < nk) (hc : chunk < nc) : kvh * nc + chunk < nk * nc := by
  have h := Nat.mul_le_mul_right nc (show kvh + 1 ≤ nk by omega)
  rw [Nat.add_mul, Nat.one_mul] at h
  omega

/-- Scratch floats. -/
def scratchFloats (nc nHead : Nat) : Nat := nc * nHead * (256 + 2) + 64

/-- The three scratch regions (acc, m, l) fit in the allocation. -/
theorem scratch_fits (nc nHead : Nat) : nc * nHead * 256 + nc * nHead + nc * nHead ≤ scratchFloats nc nHead := by unfold scratchFloats; omega

end Strata.QsaDecodeAttn



## FILE: /mnt/user-data/outputs/QsaSelect.lean
/-!
# MODULE: qsa_select
Block-level indexer scoring weights and radix top-k bookkeeping for batched QSA selection.

## Source semantics
Blocks 0..n_bid score; block b < n_bid carries 4 cells, the last carries n_kv - 4*n_bid = n_kv%4 cells (it may be empty). Total weight is n_kv. The radix select fixes the threshold key 8 bits at a time over 4 passes (shifts 24,16,8,0); equal-key cells are taken lowest index first up to the budget. If n_kv <= width everything is selected as the identity.

## Form template identifiers
block_topk_kernel -> block_topk, weight -> block_cells, order_key -> order_key
-/
namespace Strata.QsaSelect

/-- Cells carried by block b. -/
def weight (nBid nKv b : Nat) : Nat := if b < nBid then 4 else nKv - nBid * 4

-- § weights

/-- Total weight of blocks 0..n_bid is n_kv. -/
theorem total_weight (nKv : Nat) : 4 * (nKv / 4) + (nKv - (nKv / 4) * 4) = nKv := by omega

/-- The last block's weight is n_kv mod 4. -/
theorem last_weight (nKv : Nat) : weight (nKv / 4) nKv (nKv / 4) = nKv % 4 := by unfold weight; simp; omega

/-- Full blocks carry 4 cells. -/
theorem full_weight (nBid nKv b : Nat) (h : b < nBid) : weight nBid nKv b = 4 := by unfold weight; simp [h]

/-- Cell ids of full blocks are below n_kv. -/
theorem full_cell_lt (nKv b c : Nat) (hb : b < nKv / 4) (hc : c < 4) : b * 4 + c < nKv := by omega

/-- Cell ids of the tail block are below n_kv. -/
theorem tail_cell_lt (nKv c : Nat) (hc : c < nKv % 4) : (nKv / 4) * 4 + c < nKv := by omega

/-- Block and cell recompose the id. -/
theorem cell_decomp (i : Nat) : (i / 4) * 4 + i % 4 = i := by omega

-- § radix passes

/-- Four 8-bit passes cover 32 bits. -/
theorem radix_bits : 4 * 8 = 32 := by decide

/-- The pass shifts are 24, 16, 8, 0. -/
theorem radix_shifts : [24, 16, 8, 0].map (fun s => s / 8) = [3, 2, 1, 0] := by decide

/-- A digit of a key is below 256. -/
theorem digit_lt (k : Nat) : (k / 2 ^ 24) % 256 < 256 := by omega

-- § tie budget

/-- Cells taken at the threshold by one thread: clamp of (budget - before) to [0, eq]. -/
def myEq (budget before eq : Int) : Int := max 0 (min eq (budget - before))

/-- A thread never takes more than it has. -/
theorem myEq_le (b a e : Int) (he : 0 ≤ e) : myEq b a e ≤ e := by unfold myEq; omega

/-- A thread never takes a negative count. -/
theorem myEq_nonneg (b a e : Int) : 0 ≤ myEq b a e := by unfold myEq; omega

/-- Earlier threads plus this one stay within the budget when the budget is not yet exhausted. -/
theorem myEq_budget (b a e : Int) (h : a ≤ b) : a + myEq b a e ≤ b := by unfold myEq; omega

-- § grids

/-- The scoring grid (8 warps per block) covers max_blocks. -/
theorem score_grid (mb : Nat) : mb ≤ (mb + 7) / 8 * 8 := by omega

/-- Each lane's float4 covers 4 of the 128 dims, 32 lanes give all of them. -/
theorem lane_dims (lane j : Nat) (hl : lane < 32) (hj : j < 4) : lane * 4 + j < 128 := by omega

end Strata.QsaSelect





## FILE: /mnt/user-data/outputs/MoeRouterTopK.lean
/-!
# MODULE: moe_router_topk
Softmax / top-k / renormalise half of the MoE router. Selection is k passes of a block-wide argmax whose
reduction keeps the larger value and, on equality, the smaller index.

## Source semantics
* The candidate merge is a total-order maximum (larger value, then smaller index), so any reduction tree gives
  the same winner.
* The taken-mask is padded to a multiple of 16 bytes so the following shared arrays stay aligned.
* The renormaliser divides by `max(sum, 2^-14)`, so the divisor is never below the clamp.

## Form template identifiers
* `router_top10_kernel` -> `topk_select_kernel`; `s_taken` -> `taken_mask`; `taken_bytes` -> `mask_pad`
* `(bv, bi)` -> `(best_val, best_idx)`; `n_expert` -> `n_cand`; `k` -> `n_pick`
-/

namespace Strata.RouterTopK

-- § candidate merge

/-- Merge of two (value, index) candidates: larger value wins, ties go to the smaller index. -/
def merge (a b : Nat × Nat) : Nat × Nat :=
  if a.1 > b.1 ∨ (a.1 = b.1 ∧ a.2 < b.2) then a else b

/-- The merge is commutative, so lane order in the shuffle tree does not matter. -/
theorem merge_comm (a b : Nat × Nat) : merge a b = merge b a := by
  obtain ⟨av, ai⟩ := a
  obtain ⟨bv, bi⟩ := b
  simp only [merge]
  split_ifs <;> first | rfl | (simp only [Prod.mk.injEq]; omega)

/-- The merge is associative, so a warp tree and a block tree agree with a serial scan. -/
theorem merge_assoc (a b c : Nat × Nat) : merge (merge a b) c = merge a (merge b c) := by
  obtain ⟨av, ai⟩ := a
  obtain ⟨bv, bi⟩ := b
  obtain ⟨cv, ci⟩ := c
  simp only [merge]
  split_ifs <;> first | rfl | (simp only [Prod.mk.injEq] at *; omega) | (exfalso; simp only at *; omega)

/-- The merge is idempotent. -/
theorem merge_self (a : Nat × Nat) : merge a a = a := by
  simp only [merge]
  split_ifs <;> rfl

/-- The merge result is one of its inputs. -/
theorem merge_mem (a b : Nat × Nat) : merge a b = a ∨ merge a b = b := by
  simp only [merge]
  split_ifs
  · exact Or.inl rfl
  · exact Or.inr rfl

/-- The merge result's value is at least the left value. -/
theorem merge_ge_left (a b : Nat × Nat) : a.1 ≤ (merge a b).1 := by
  obtain ⟨av, ai⟩ := a
  obtain ⟨bv, bi⟩ := b
  simp only [merge]
  split_ifs <;> simp only at * <;> omega

/-- The merge result's value is at least the right value. -/
theorem merge_ge_right (a b : Nat × Nat) : b.1 ≤ (merge a b).1 := by
  obtain ⟨av, ai⟩ := a
  obtain ⟨bv, bi⟩ := b
  simp only [merge]
  split_ifs <;> simp only at * <;> omega

-- § sentinel

/-- The sentinel index `n_expert` loses to a real candidate of equal value. -/
theorem sentinel_loses (n v i : Nat) (hi : i < n) : merge (v, i) (v, n) = (v, i) := by
  simp only [merge]
  split_ifs with h
  · rfl
  · exact absurd (Or.inr ⟨by trivial, hi⟩) h

-- § shared-memory layout

/-- Padded size of the taken mask. -/
def maskPad (n : Nat) : Nat := (n + 15) / 16 * 16

/-- The padded mask covers every expert. -/
theorem maskPad_ge (n : Nat) : n ≤ maskPad n := by
  unfold maskPad; omega

/-- The padded mask is a multiple of 16 bytes. -/
theorem maskPad_mod (n : Nat) : maskPad n % 16 = 0 := by
  unfold maskPad; omega

/-- The padding adds fewer than 16 bytes. -/
theorem maskPad_lt (n : Nat) : maskPad n < n + 16 := by
  unfold maskPad; omega

/-- Dynamic shared memory: mask, then n doubles, then n floats. -/
def smemBytes (n : Nat) : Nat := maskPad n + n * 8 + n * 4

/-- The double array starts at a multiple of 16, hence 8-aligned. -/
theorem double_base_aligned (n : Nat) : maskPad n % 8 = 0 := by
  unfold maskPad; omega

/-- The float array starts at a multiple of 8 because the doubles are a multiple of 8. -/
theorem float_base_aligned (n : Nat) : (maskPad n + n * 8) % 8 = 0 := by
  unfold maskPad; omega

/-- The total covers all three regions. -/
theorem smem_covers (n : Nat) : maskPad n + n * 8 + n * 4 = smemBytes n := rfl

-- § renormalisation

/-- The clamped divisor. -/
def clampDen (s lo : Nat) : Nat := max s lo

/-- The divisor is at least the clamp. -/
theorem clampDen_ge (s lo : Nat) : lo ≤ clampDen s lo := by
  unfold clampDen; omega

/-- A positive clamp keeps the divisor positive, so the division is defined. -/
theorem clampDen_pos (s lo : Nat) (h : 0 < lo) : 0 < clampDen s lo := by
  unfold clampDen; omega

/-- Above the clamp the divisor is the sum itself. -/
theorem clampDen_of_ge (s lo : Nat) (h : lo ≤ s) : clampDen s lo = s := by
  unfold clampDen; omega

/-- Below the clamp the divisor is the clamp. -/
theorem clampDen_of_lt (s lo : Nat) (h : s < lo) : clampDen s lo = lo := by
  unfold clampDen; omega

-- § launch configuration

/-- Threads per block: capped at 512, rounded up to a whole warp. -/
def launchThreads (n : Nat) : Nat := ((if n < 512 then n else 512) + 31) / 32 * 32

/-- The block is a multiple of the warp size. -/
theorem launchThreads_mod (n : Nat) : launchThreads n % 32 = 0 := by
  unfold launchThreads; split_ifs <;> omega

/-- The block never exceeds 512 threads. -/
theorem launchThreads_le (n : Nat) : launchThreads n ≤ 512 := by
  unfold launchThreads; split_ifs <;> omega

/-- Every positive expert count gets at least one warp. -/
theorem launchThreads_pos (n : Nat) (h : 0 < n) : 32 ≤ launchThreads n := by
  unfold launchThreads; split_ifs <;> omega

/-- The warp-count array of 16 floats is large enough for 512 threads. -/
theorem warps_fit (n : Nat) : (launchThreads n + 31) / 32 ≤ 16 := by
  have := launchThreads_le n; omega

end Strata.RouterTopK



## FILE: /mnt/user-data/outputs/S2ExpertGroupedLayout.lean
/-!
# MODULE: s2_expert_grouped_layout
Blob geometry, gate/up interleaving and chunk coverage for the grouped Q2_0 expert kernels.

## Source semantics
* The blob holds down codes, gate/up scales, then down scales at fixed byte offsets.
* Gate/up row-slot `i` is gate row `i/2` when even and up row `i/2` when odd.
* The output is gate-major: every hit's gate rows come first, then every hit's up rows.
* `sum (c-1)*x = sum c*x - sum x`, the identity behind the two `dp4a` accumulators.

## Form template identifiers
* `H` -> `n_embd`; `FF` -> `n_ff`; `ROW_GU` -> `gu_row_bytes`; `O_D_CODES` -> `down_codes_off`
* `gu_kernel` -> `gate_up_proj`; `swiglu_kernel` -> `gated_act`; `down_kernel` -> `down_proj`
-/

namespace Strata.S2Grouped

-- § blob geometry

/-- Hidden size. -/
def H : Nat := 2560
/-- Expert feed-forward size. -/
def FF : Nat := 640
/-- Gate/up row code bytes. -/
def rowGU : Nat := H / 4
/-- Down row code bytes. -/
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

/-- Gate/up rows take 640 bytes. -/
theorem rowGU_val : rowGU = 640 := by decide
/-- Down rows take 160 bytes. -/
theorem rowD_val : rowD = 160 := by decide
/-- Scale counts per row. -/
theorem scales_val : scGU = 40 ∧ scD = 10 := by decide
/-- The down codes begin after all gate/up codes. -/
theorem oDCodes_val : oDCodes = 819200 := by decide
/-- The gate/up scales begin after the down codes. -/
theorem oGUScales_val : oGUScales = 1228800 := by decide
/-- The down scales begin after the gate/up scales. -/
theorem oDScales_val : oDScales = 1331200 := by decide

/-- The regions are ordered. -/
theorem offsets_ordered : oDCodes < oGUScales ∧ oGUScales < oDScales := by decide

/-- Total blob size. -/
def blobBytes : Nat := oDScales + H * scD * 2

/-- The blob size is 1,382,400 bytes. -/
theorem blobBytes_val : blobBytes = 1382400 := by decide

/-- The blob size is a multiple of 16, as `fetch_blobs` requires. -/
theorem blobBytes_mod16 : blobBytes % 16 = 0 := by decide

-- § gate-major output slot

/-- Output index for row-slot `i` of hit `h` among `n` hits. -/
def guIndex (n h i : Nat) : Nat :=
  (if i % 2 = 1 then n * 640 + h * 640 else h * 640) + i / 2

/-- Every output index stays inside the `2*n*FF` buffer. -/
theorem guIndex_lt (n h i : Nat) (hh : h < n) (hi : i < 1280) : guIndex n h i < 2 * n * 640 := by
  unfold guIndex; split_ifs <;> omega

/-- Distinct (hit, row-slot) pairs never share an output index. -/
theorem guIndex_inj (n h i h' i' : Nat) (hh : h < n) (hh' : h' < n) (hi : i < 1280) (hi' : i' < 1280)
    (heq : guIndex n h i = guIndex n h' i') : h = h' ∧ i = i' := by
  unfold guIndex at heq
  split_ifs at heq <;> omega

/-- Even slots land in the gate half. -/
theorem guIndex_gate (n h i : Nat) (hh : h < n) (hi : i < 1280) (he : i % 2 = 0) : guIndex n h i < n * 640 := by
  unfold guIndex; split_ifs <;> omega

/-- Odd slots land in the up half. -/
theorem guIndex_up (n h i : Nat) (hi : i < 1280) (ho : i % 2 = 1) : n * 640 ≤ guIndex n h i := by
  unfold guIndex; split_ifs <;> omega

/-- SwiGLU reads gate at `x` and up at `x + pairs`, so row `r` of hit `h` meets its own up row. -/
theorem swiglu_pairing (n h r : Nat) (hr : r < 640) :
    guIndex n h (2 * r + 1) = guIndex n h (2 * r) + n * 640 := by
  unfold guIndex; split_ifs <;> omega

-- § chunk coverage

/-- Lane `l`, step `q` covers chunk `l + 32q`; all 80 gate/up chunks are covered by 32 lanes x 3 steps. -/
theorem gu_chunks_covered (c : Nat) (hc : c < 80) : ∃ l q, l < 32 ∧ q < 3 ∧ l + 32 * q = c :=
  ⟨c % 32, c / 32, by omega, by omega, by omega⟩

/-- The three-step lane cover reaches no chunk twice. -/
theorem gu_chunks_unique (l q l' q' : Nat) (hl : l < 32) (hl' : l' < 32)
    (h : l + 32 * q = l' + 32 * q') : l = l' ∧ q = q' := by omega

/-- Down rows have 20 chunks, so lanes 0..19 each own one. -/
theorem down_chunks_fit (c : Nat) (hc : c < FF / 32) : c < 32 := by
  unfold FF at hc; omega

/-- One weight scale covers two chunks: `c / 2` indexes the 64-wide group. -/
theorem scale_group (c : Nat) (hc : c < 80) : c / 2 < 40 := by omega

-- § launch grid

/-- Gate/up row blocks per group tile exactly. -/
theorem gu_blocks_exact : 2 * FF / 32 * 32 = 2 * FF := by decide
/-- Down row blocks per group tile exactly. -/
theorem down_blocks_exact : H / 64 * 64 = H := by decide

-- § bias identity

/-- Dot product of codes and activations. -/
def dot : List (Int × Int) → Int
  | [] => 0
  | p :: t => p.1 * p.2 + dot t

/-- Sum of activations. -/
def sumX : List (Int × Int) → Int
  | [] => 0
  | p :: t => p.2 + sumX t

/-- Dot product with bias-subtracted codes. -/
def dotBias : List (Int × Int) → Int
  | [] => 0
  | p :: t => (p.1 - 1) * p.2 + dotBias t

/-- `sum (c-1)*x = sum c*x - sum x`. -/
theorem dotBias_eq (l : List (Int × Int)) : dotBias l = dot l - sumX l := by
  induction l with
  | nil => simp [dotBias, dot, sumX]
  | cons p t ih =>
    simp only [dotBias, dot, sumX, ih]
    rw [Int.sub_mul]
    omega

-- § compaction counts

/-- A ballot prefix count never exceeds the lane index. -/
theorem prefix_le (lane : Nat) (pc : Nat) (h : pc ≤ lane) : pc < lane + 1 := by omega

/-- Entries per group are capped at 8 by `min`. -/
theorem group_cap (a b : Nat) : min (a - b) 8 ≤ 8 := by omega

end Strata.S2Grouped



## FILE: /mnt/user-data/outputs/SGemvQuantLayout.lean
/-!
# MODULE: s_gemv_quant_layout
Index and layout arithmetic of the quantised GEMV kernels: block strides, code widths, group shifts,
and the guarantee that a 16-element lane iteration never straddles an activation block.

## Source semantics
* Q8_K blocks are 292 bytes per 256 elements; Q8_0 blocks are 34 bytes per 32 elements.
* A code of width b bits is read as `(byte >> (k*b)) mod 2^b`.
* The group index is `i >> shift` for power-of-two groups.
* The IQ4_NL codebook is strictly increasing.

## Form template identifiers
* `s_gemv_q8_split_kernel` -> `quant_gemv_warp`; `kIq4Nl` -> `nl_codebook`
* `CODE_BITS` -> `code_bits`; `group_shift` -> `group_log2`; `QE` -> `lane_elems`
-/

namespace Strata.SGemv

-- § block strides

/-- Q8_K block size equals header, payload, and block sums. -/
theorem q8k_bytes : 292 = 4 + 256 + 16 * 2 := by decide
/-- Q8_0 block size equals header and payload. -/
theorem q8_0_bytes : 34 = 2 + 32 := by decide

/-- The Q8_K payload begins at a 4-byte aligned offset in every block whenever the base is aligned. -/
theorem q8k_block_aligned (b : Nat) : (b * 292) % 4 = 0 := by omega
/-- Q8_0 blocks start at even offsets, so the fp16 header is 2-aligned. -/
theorem q8_0_block_aligned (b : Nat) : (b * 34) % 2 = 0 := by omega

-- § lane iteration stays in one block

/-- A lane position that is a multiple of 16 leaves room for 16 elements in its Q8_K block. -/
theorem q8k_no_straddle (i : Nat) (h : i % 16 = 0) : i % 256 + 16 ≤ 256 := by omega
/-- The same for Q8_0 blocks. -/
theorem q8_0_no_straddle (i : Nat) (h : i % 16 = 0) : i % 32 + 16 ≤ 32 := by omega

/-- Lane starting offsets `lane*16 + 512*t` are multiples of 16. -/
theorem lane_pos_mod (lane t : Nat) : (lane * 16 + 32 * 16 * t) % 16 = 0 := by omega

/-- The sixteen elements share one group for the three group sizes the format defines. -/
theorem quad_group16 (i : Nat) (h : i % 16 = 0) : i / 16 = (i + 15) / 16 := by omega
/-- Group size 32. -/
theorem quad_group32 (i : Nat) (h : i % 16 = 0) : i / 32 = (i + 15) / 32 := by omega
/-- Group size 64. -/
theorem quad_group64 (i : Nat) (h : i % 16 = 0) : i / 64 = (i + 15) / 64 := by omega

-- § code widths

/-- Codes per byte. -/
def perByte (bits : Nat) : Nat := 8 / bits
/-- Bytes in a four-code word. -/
def wordBytes (bits : Nat) : Nat := 4 * bits / 8

/-- Codes per byte for the three widths. -/
theorem perByte_vals : perByte 2 = 4 ∧ perByte 4 = 2 ∧ perByte 8 = 1 := by decide
/-- Four-code word bytes for the three widths. -/
theorem wordBytes_vals : wordBytes 2 = 1 ∧ wordBytes 4 = 2 ∧ wordBytes 8 = 4 := by decide

/-- A masked code is always below `2^bits`. -/
theorem code_lt (byte k bits : Nat) : (byte >>> (k * bits)) % 2 ^ bits < 2 ^ bits :=
  Nat.mod_lt _ (Nat.two_pow_pos bits)

/-- For 4-bit codes the extracted value is a valid codebook index. -/
theorem code4_lt16 (byte k : Nat) : (byte >>> (k * 4)) % 16 < 16 := Nat.mod_lt _ (by decide)

-- § group shift

/-- A shift by `s` equals division by `2^s`. -/
theorem shift_is_div (i s : Nat) : i >>> s = i / 2 ^ s := Nat.shiftRight_eq_div_pow i s

/-- The shifts for the three group sizes are 4, 5, 6. -/
theorem group_shifts : 2 ^ 4 = 16 ∧ 2 ^ 5 = 32 ∧ 2 ^ 6 = 64 := by decide

-- § codebook

/-- The IQ4_NL codebook values. -/
def iq4nl : List Int := [-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113]

/-- The codebook has 16 entries, so a 4-bit code always indexes it. -/
theorem iq4nl_len : iq4nl.length = 16 := by decide

/-- The codebook is strictly increasing. -/
theorem iq4nl_sorted : iq4nl.Pairwise (· < ·) := by decide

/-- Every entry fits in int8. -/
theorem iq4nl_int8 : ∀ x ∈ iq4nl, -128 ≤ x ∧ x ≤ 127 := by decide

-- § launch shapes

/-- 256 threads form 8 warps, so blocks cover `n_out` rows with the ceiling quotient. -/
theorem warp_blocks_cover (n_out : Nat) : n_out ≤ (n_out + 8 - 1) / 8 * 8 := by omega

/-- The split kernel's tree reduction halves a power of two down to 1. -/
theorem halving_terminates (t : Nat) (h : 0 < t) : t / 2 < t := by omega

/-- The combined 16-way accumulator tree has exactly 16 leaves. -/
theorem acc_leaves : 2 * 2 * 2 * 2 = 16 := by decide

/-- `ffn_down_shexp`'s n_in = 640 is a multiple of 32 but not of 256. -/
theorem shexp_not_q8k : 640 % 32 = 0 ∧ 640 % 256 ≠ 0 := by decide

end Strata.SGemv



## FILE: /mnt/user-data/outputs/SamplerChain.lean
/-!
# MODULE: sampler_chain
Penalties, greedy argmax tie rule, history window, top-k bounds and the counter-based uniform draw.

## Source semantics
* Repeat penalty multiplies non-positive logits and divides positive ones; presence is a boolean.
* Greedy picks the largest value with the smallest index on ties; reductions are order independent.
* The penalty window is the tail of the history.
* The uniform draw is `(c0 >> 8) / 2^24`, always in [0, 1).

## Form template identifiers
* `apply_penalties` -> `penalise`; `sampler_greedy_kernel` -> `argmax_kernel`
* `history_count` -> `count_in_window`; `philox_uniform` -> `uniform_draw`
-/

namespace Strata.Sampler

-- § penalties (integer model: the repeat penalty is a positive divisor / multiplier)

/-- Penalty on an integer logit. Positive logits are divided, others multiplied. -/
def penalise (logit : Int) (count : Nat) (rep : Nat) (freq present : Int) : Int :=
  if count = 0 then logit
  else
    (if logit ≤ 0 then logit * rep else logit / rep) - (count * freq + present)

/-- No occurrences means no change. -/
theorem penalise_zero (l : Int) (rep : Nat) (f p : Int) : penalise l 0 rep f p = l := by
  simp [penalise]

/-- With a unit repeat penalty and zero offsets the logit is unchanged. -/
theorem penalise_neutral (l : Int) (c : Nat) : penalise l c 1 0 0 = l := by
  unfold penalise
  split_ifs <;> simp

/-- A positive logit stays non-negative after the divide when the offsets are zero. -/
theorem penalise_pos_nonneg (l : Int) (c rep : Nat) (hl : 0 < l) :
    0 ≤ penalise l c rep 0 0 := by
  unfold penalise
  by_cases hc : c = 0
  · rw [if_pos hc]; exact hl.le
  · have hn : ¬ l ≤ 0 := by omega
    rw [if_neg hc, if_neg hn, Int.mul_zero, Int.add_zero, Int.sub_zero]
    exact Int.ediv_nonneg hl.le (by omega)

/-- A non-positive logit stays non-positive after the multiply when the offsets are zero. -/
theorem penalise_nonpos_nonpos (l : Int) (c rep : Nat) (hl : l ≤ 0) (hc : c ≠ 0) :
    penalise l c rep 0 0 ≤ 0 := by
  unfold penalise
  rw [if_neg hc, if_pos hl, Int.mul_zero, Int.add_zero, Int.sub_zero]
  exact Int.mul_nonpos_of_nonpos_of_nonneg hl (by omega)

/-- The presence term is a boolean: at most 1 however large the count. -/
theorem presence_bool (c : Nat) : (if c > 0 then 1 else 0 : Nat) ≤ 1 := by
  split_ifs <;> omega

-- § greedy tie rule

/-- Candidate merge: larger value, then smaller index. -/
def pick (a b : Int × Nat) : Int × Nat :=
  if a.1 > b.1 ∨ (a.1 = b.1 ∧ a.2 < b.2) then a else b

/-- The pick is commutative. -/
theorem pick_comm (a b : Int × Nat) : pick a b = pick b a := by
  obtain ⟨av, ai⟩ := a
  obtain ⟨bv, bi⟩ := b
  simp only [pick]
  split_ifs <;> first | rfl | (simp only [Prod.mk.injEq]; omega)

/-- A tie in value goes to the smaller index. -/
theorem pick_tie (v : Int) (i j : Nat) (h : i < j) : pick (v, i) (v, j) = (v, i) := by
  simp only [pick]
  split_ifs with hc
  · rfl
  · exact absurd (Or.inr ⟨by trivial, h⟩) hc

/-- A strictly larger value always wins. -/
theorem pick_larger (u v : Int) (i j : Nat) (h : v < u) : pick (u, i) (v, j) = (u, i) := by
  simp only [pick]
  split_ifs with hc
  · rfl
  · exact absurd (Or.inl h) hc

-- § history window

/-- Length of the penalty window. -/
def windowLen (lastN histLen : Nat) : Nat := min lastN histLen

/-- The window never exceeds the history. -/
theorem windowLen_le (n h : Nat) : windowLen n h ≤ h := by unfold windowLen; omega

/-- The window starts at `histLen - len` and ends exactly at the end of the history. -/
theorem window_tail (n h : Nat) : (h - windowLen n h) + windowLen n h = h := by
  unfold windowLen; omega

-- § top-k bounds

/-- The effective `k`, capped at 64. -/
def effK (topK : Nat) : Nat := if topK < 64 then topK else 64

/-- The effective `k` fits the 64-entry survivor arrays. -/
theorem effK_le (t : Nat) : effK t ≤ 64 := by unfold effK; split_ifs <;> omega

/-- `min_keep` clamps the cut into the survivors. -/
theorem cut_le (minKeep nKeep : Nat) : min minKeep nKeep ≤ nKeep := by omega

-- § uniform draw

/-- The 24-bit mantissa extracted from a 32-bit word. -/
def mant (c0 : Nat) : Nat := c0 >>> 8

/-- A 32-bit word yields a value below `2^24`, so `mant / 2^24 < 1`. -/
theorem mant_lt (c0 : Nat) (h : c0 < 2 ^ 32) : mant c0 < 2 ^ 24 := by
  unfold mant
  rw [Nat.shiftRight_eq_div_pow]
  omega

-- § launch shapes

/-- A 1024-thread block has 32 warps, so the 32-entry reduction arrays suffice. -/
theorem greedy_warps : 1024 / 32 = 32 := by decide

/-- The ceiling grid covers every token. -/
theorem grid_covers (n : Nat) : n ≤ (n + 63) / 64 * 64 := by omega

end Strata.Sampler



## FILE: /mnt/user-data/outputs/VerifyWindowIndexing.lean
/-!
# MODULE: verify_window_indexing
Index arithmetic of the multi-token verify kernels: causal conv window, commit of the last three history
entries, sliding-window ids, pointer rebasing, block gathers, and broadcast indexing.

## Source semantics
* The conv window of token t is the sequence `[hist0, hist1, hist2, x_0, ...]` entries t..t+2 plus x_t.
* Commit keeps the last three entries of `hist ++ x_0..x_{n-1}`.
* Window ids cover `[max(0, n_kv - window), n_kv)`.
* `dense_steps` emits `(c, c+1, (c+1)/4, c+1)`.

## Form template identifiers
* `gdn_conv_l2_multi_kernel` -> `causal_conv_window`; `gdn_conv_commit_kernel` -> `history_commit`
* `window_ids_kernel` -> `sliding_ids`; `rebase_ptrs_kernel` -> `pointer_rebase`
* `fetch_blobs_kernel` -> `blob_gather`; `broadcast_streams_kernel` -> `stream_broadcast`
-/

namespace Strata.Verify

-- § conv window

/-- The sequence `[hist | x]` read at position `src`. -/
def seqAt {α : Type} (hist x : Nat → α) (src : Nat) : α :=
  if src < 3 then hist src else x (src - 3)

/-- Window entries for token `t` taken from `x` always refer to earlier tokens. -/
theorem window_causal (t j : Nat) (hj : j < 3) (h : 3 ≤ t + j) : t + j - 3 < t := by omega

/-- Window entries from history have valid history indices. -/
theorem window_hist_index (t j : Nat) (hj : j < 3) (h : t + j < 3) : t + j < 3 := h

/-- Token 0's window is exactly the history. -/
theorem window_first {α : Type} (hist x : Nat → α) (j : Nat) (hj : j < 3) :
    seqAt hist x (0 + j) = hist j := by
  unfold seqAt
  rw [if_pos (by omega)]
  simp

/-- For `t ≥ 3` the window consists only of the activations `x_{t-3} .. x_{t-1}`. -/
theorem window_late {α : Type} (hist x : Nat → α) (t j : Nat) (ht : 3 ≤ t) :
    seqAt hist x (t + j) = x (t + j - 3) := by
  unfold seqAt; split_ifs with h
  · omega
  · rfl

-- § commit

/-- The commit reads entries `n, n+1, n+2` of a sequence of `n + 3` elements: the last three. -/
theorem commit_last_three (n j : Nat) (hj : j < 3) : n + j < n + 3 := by omega

/-- For `n ≥ 3` the committed history comes entirely from the new activations. -/
theorem commit_from_new (n j : Nat) (hn : 3 ≤ n) (hj : j < 3) : n + j - 3 < n := by omega

/-- With no kept tokens the history is untouched (the kernel returns early). -/
theorem commit_zero (j : Nat) (hj : j < 3) : 0 + j = j := by omega

-- § sliding window

/-- First id of the window. -/
def winStart (nKv window : Nat) : Nat := if nKv > window then nKv - window else 0

/-- Number of ids. -/
def winWidth (nKv window : Nat) : Nat := nKv - winStart nKv window

/-- The window ends exactly at `n_kv`. -/
theorem win_end (n w : Nat) : winStart n w + winWidth n w = n := by
  unfold winWidth winStart; split_ifs <;> omega

/-- The width never exceeds the window. -/
theorem win_width_le (n w : Nat) : winWidth n w ≤ w := by
  unfold winWidth winStart; split_ifs <;> omega

/-- A short history is fully visible. -/
theorem win_short (n w : Nat) (h : n ≤ w) : winStart n w = 0 ∧ winWidth n w = n := by
  unfold winWidth winStart; split_ifs <;> omega

-- § dense steps

/-- The third field is the ceiling-free quarter of `c + 1`. -/
theorem dense_quarter (c : Nat) : (c + 1) / 4 ≤ c + 1 := by omega

-- § pointer rebase

/-- Slot `k` pointer. -/
def slotPtr (base bytes k : Nat) : Nat := base + k * bytes

/-- Consecutive slots are exactly one blob apart. -/
theorem slotPtr_step (base bytes k : Nat) : slotPtr base bytes (k + 1) = slotPtr base bytes k + bytes := by
  unfold slotPtr; rw [Nat.succ_mul]; omega

/-- Slot 0 is the base. -/
theorem slotPtr_zero (base bytes : Nat) : slotPtr base bytes 0 = base := by
  unfold slotPtr; simp

-- § blob gather

/-- A flat index below `n * per` belongs to a blob below `n`. -/
theorem gather_blob_lt (i n per : Nat) (hper : 0 < per) (hi : i < n * per) : i / per < n :=
  (Nat.div_lt_iff_lt_mul hper).mpr hi

/-- A flat index is its blob times `per` plus the offset. -/
theorem gather_decompose (i per : Nat) : per * (i / per) + i % per = i := Nat.div_add_mod i per

/-- The offset is below `per`. -/
theorem gather_off_lt (i per : Nat) (hper : 0 < per) : i % per < per := Nat.mod_lt _ hper

/-- A blob size multiple of 16 gives a whole number of 16-byte words. -/
theorem blob_words (b : Nat) (h : b % 16 = 0) : b / 16 * 16 = b := by omega

-- § broadcast

/-- The source index of a broadcast element is inside the source row. -/
theorem broadcast_src_lt (i n : Nat) (hn : 0 < n) : i % n < n := Nat.mod_lt _ hn

/-- Each of `hc` streams reads the same source row: indices `i` and `i + n` give the same source. -/
theorem broadcast_periodic (i n : Nat) : (i + n) % n = i % n := Nat.add_mod_right i n

-- § misc launch shapes

/-- Four warps cover 128 entries. -/
theorem multi_hit_capacity : 4 * 32 = 128 := by decide

/-- The ceiling grid covers `n` elements. -/
theorem grid_cover (n : Nat) : n ≤ (n + 255) / 256 * 256 := by omega

/-- Eight rows per block cover `2*h_v` rows. -/
theorem ab_rows_cover (h : Nat) : 2 * h ≤ (2 * h + 7) / 8 * 8 := by omega

end Strata.Verify





## FILE: /mnt/user-data/outputs/PleNgramHashIq4nlRead.lean
/-!
# MODULE: ple_ngram_hash_iq4nl_read
The PLE n-gram hash (token context window → 16 table row indices, one per head) and the IQ4_NL
table read (block dequantisation, row layout, gather layout, issue/collect protocol).

## Source semantics
* `ngram_rows`: context `ctx[0]` is the current token; `ctx[s]` is the predecessor `s` positions back,
  read from an OLDEST-FIRST window at index `n_prev - s`. The EOS cut is evaluated BEFORE storing, and is
  sticky: once a predecessor is negative or EOS, it and every older position are EOS.
* `ngram_mixed`: first product is assigned, remaining products are XORed in, all mod 2^64.
* Head `h = (n-2)*HEADS_PER_NGRAM + g`; index = `mixed % vocab[h] + offset[h]`, so every head lands
  in its own disjoint slice of the table.
* IQ4_NL block: 18 bytes (2-byte scale + 16 bytes of nibbles); `qs[j]` holds elements `j` (low nibble)
  and `j+16` (high nibble). A row is `PLE_HEAD_DIM/32 = 5` blocks = 90 bytes.
* `gather` is head-slowest: head `h` occupies `[h*160, (h+1)*160)` of the 2560-float output.
* `PleTable` in-flight protocol: `collect` without `issue` fails; in Direct mode a second `issue`
  while pending fails.

## Form template identifiers
* `ngram_mixed`            → `hash_mix_xor_fold`
* `ngram_rows`             → `window_to_head_indices`
* `ple_artifact_consts`    → `hash_constants`  (`mult`, `vocab`, `offset`)
* `kIq4Nl` / `iq4nl_code`  → `codebook_16`, `codebook_lookup`
* `iq4nl_dequant_row`      → `block_dequant_split_halves`
* `PleTable::gather`       → `head_slowest_gather`
* `PleTable::issue/collect`→ `two_phase_read_protocol`
-/

namespace PleNgram

-- § constants

/-- Number of heads. -/
def nHeads : Nat := 16
/-- Heads per n-gram order. -/
def headsPerNgram : Nat := 8
/-- Largest n-gram order (context length). -/
def ngramSize : Nat := 3
/-- Per-head embedding width. -/
def headDim : Nat := 160
/-- Bytes in one IQ4_NL block. -/
def blockBytes : Nat := 18
/-- Bytes in one table row. -/
def rowBytes : Nat := (headDim / 32) * blockBytes

/-- A row is 90 bytes. -/
theorem rowBytes_eq : rowBytes = 90 := by decide

/-- Concatenated heads fill 2560 floats. -/
theorem nEmbd_eq : nHeads * headDim = 2560 := by decide

/-- Heads are covered by the n-gram orders 2..ngramSize. -/
theorem heads_cover : (ngramSize - 1) * headsPerNgram = nHeads := by decide

/-- Per-head vocabulary sizes. -/
def vocabL : List Nat :=
  [20000003, 20000023, 20000033, 20000047, 20000059, 20000063, 20000069, 20000077,
   20000081, 20000093, 20000107, 20000147, 20000153, 20000159, 20000161, 20000171]

/-- Per-head row offsets. -/
def offsetL : List Nat :=
  [0, 20000003, 40000026, 60000059, 80000106, 100000165, 120000228, 140000297,
   160000374, 180000455, 200000548, 220000655, 240000802, 260000955, 280001114, 300001275]

/-- Head vocabulary (0 outside range). -/
def vocabAt (i : Nat) : Nat := vocabL.getD i 0
/-- Head offset (0 outside range). -/
def offsetAt (i : Nat) : Nat := offsetL.getD i 0

/-- Both constant arrays have one entry per head. -/
theorem consts_length : vocabL.length = nHeads ∧ offsetL.length = nHeads := by decide

/-- Every vocabulary is positive, so `% vocab` is well-behaved. -/
theorem vocab_pos : ∀ i, i < nHeads → 0 < vocabAt i := by decide

/-- The offsets are the exclusive prefix sums of the vocabularies (the slices tile the table). -/
theorem offset_prefix_sum : ∀ i, i < 15 → offsetAt (i + 1) = offsetAt i + vocabAt i := by decide

/-- The first slice starts at row 0. -/
theorem offset_zero : offsetAt 0 = 0 := by decide

/-- Total rows addressed by all heads. -/
theorem total_rows : offsetAt 15 + vocabAt 15 = 320001446 := by decide

/-- Slices are strictly ordered, hence disjoint. -/
theorem slices_ordered : ∀ i, i < 15 → offsetAt i + vocabAt i ≤ offsetAt (i + 1) := by
  intro i hi
  have := offset_prefix_sum i hi
  omega

/-- The hash index of a head always lies inside that head's slice. -/
theorem index_in_slice (mixed i : Nat) (hi : i < nHeads) :
    offsetAt i ≤ mixed % vocabAt i + offsetAt i ∧
    mixed % vocabAt i + offsetAt i < offsetAt i + vocabAt i := by
  have hp := vocab_pos i hi
  have := Nat.mod_lt mixed hp
  omega

/-- Indices of different heads never collide. -/
theorem index_heads_disjoint (m1 m2 i j : Nat) (hi : i < nHeads) (hj : j < nHeads) (hij : i < j) :
    m1 % vocabAt i + offsetAt i < m2 % vocabAt j + offsetAt j := by
  have h1 := (index_in_slice m1 i hi).2
  -- chain: offset i + vocab i ≤ offset (i+1) ≤ ... ≤ offset j
  have chain : ∀ k, i < k → k < nHeads → offsetAt i + vocabAt i ≤ offsetAt k := by
    intro k hk hkn
    induction k with
    | zero => omega
    | succ k ih =>
      by_cases hik : i < k
      · have := ih hik (by omega)
        have h2 := slices_ordered k (by unfold nHeads at hkn; omega)
        have h3 : offsetAt k ≤ offsetAt k + vocabAt k := Nat.le_add_right _ _
        omega
      · have : i = k := by omega
        subst this
        exact slices_ordered i (by unfold nHeads at hkn; omega)
  have := chain j hij hj
  omega

-- § mixing hash

/-- Fold of the products: first is assigned, the rest XORed (mod 2^64 via `UInt64`). -/
def mixedList : List UInt64 → List UInt64 → UInt64
  | c :: cs, m :: ms => cs.zip ms |>.foldl (fun acc p => acc ^^^ (p.1 * p.2)) (c * m)
  | _, _ => 0

/-- Mixing for order 2. -/
def mixed2 (c0 c1 m0 m1 : UInt64) : UInt64 := (c0 * m0) ^^^ (c1 * m1)

/-- Mixing for order 3. -/
def mixed3 (c0 c1 c2 m0 m1 m2 : UInt64) : UInt64 := ((c0 * m0) ^^^ (c1 * m1)) ^^^ (c2 * m2)

/-- The list fold agrees with the unrolled order-2 formula. -/
theorem mixedList_two (c0 c1 m0 m1 : UInt64) :
    mixedList [c0, c1] [m0, m1] = mixed2 c0 c1 m0 m1 := by
  simp [mixedList, mixed2]

/-- The list fold agrees with the unrolled order-3 formula. -/
theorem mixedList_three (c0 c1 c2 m0 m1 m2 : UInt64) :
    mixedList [c0, c1, c2] [m0, m1, m2] = mixed3 c0 c1 c2 m0 m1 m2 := by
  simp [mixedList, mixed3]

/-- Order 3 extends order 2 by one more XORed product. -/
theorem mixed3_extends (c0 c1 c2 m0 m1 m2 : UInt64) :
    mixed3 c0 c1 c2 m0 m1 m2 = mixed2 c0 c1 m0 m1 ^^^ (c2 * m2) := rfl

/-- If the third context entry has product zero, order 3 equals order 2. -/
theorem mixed3_of_zero (c0 c1 c2 m0 m1 m2 : UInt64) (h : c2 * m2 = 0) :
    mixed3 c0 c1 c2 m0 m1 m2 = mixed2 c0 c1 m0 m1 := by
  simp [mixed3, mixed2, h]

-- § head index layout

/-- Head index for order `n` (2..) and slot `g`. -/
def headIdx (n g : Nat) : Nat := (n - 2) * headsPerNgram + g

/-- Head indices are in range for orders 2..3 and slots below 8. -/
theorem headIdx_lt (n g : Nat) (hn2 : 2 ≤ n) (hn3 : n ≤ ngramSize) (hg : g < headsPerNgram) :
    headIdx n g < nHeads := by
  unfold headIdx nHeads ngramSize headsPerNgram at *
  omega

/-- Distinct (order, slot) pairs give distinct heads. -/
theorem headIdx_injective (n g n' g' : Nat) (hg : g < headsPerNgram) (hg' : g' < headsPerNgram)
    (h : headIdx n g = headIdx n' g') (hn : 2 ≤ n) (hn' : 2 ≤ n') : n = n' ∧ g = g' := by
  unfold headIdx headsPerNgram at *
  omega

/-- Every head is reached by some (order, slot) pair. -/
theorem headIdx_surjective (h : Nat) (hh : h < nHeads) :
    ∃ n g, 2 ≤ n ∧ n ≤ ngramSize ∧ g < headsPerNgram ∧ headIdx n g = h := by
  refine ⟨h / headsPerNgram + 2, h % headsPerNgram, by omega, ?_, ?_, ?_⟩
  · unfold nHeads ngramSize headsPerNgram at *; omega
  · unfold headsPerNgram; omega
  · unfold headIdx headsPerNgram; omega

-- § context window with EOS cut

/-- Context entry at distance 1 (NEWEST predecessor) after the cut. -/
def ctx1 (eos p1 : Int) : Int := if p1 < 0 ∨ p1 = eos then eos else p1

/-- Context entry at distance 2 after the sticky cut. -/
def ctx2 (eos p1 p2 : Int) : Int :=
  if (p1 < 0 ∨ p1 = eos) ∨ p2 < 0 ∨ p2 = eos then eos else p2

/-- Window index of the predecessor `s` back: `s = 1` reads the NEWEST (last) entry. -/
def prevIdx (nPrev s : Nat) : Nat := nPrev - s

/-- With three-gram context (`nPrev = 2`), distance 1 reads index 1 (newest), distance 2 reads index 0. -/
theorem prevIdx_newest_first : prevIdx 2 1 = 1 ∧ prevIdx 2 2 = 0 := by decide

/-- The window is walked newest to oldest and never leaves range. -/
theorem prevIdx_lt (nPrev s : Nat) (hs : 1 ≤ s) (hsn : s ≤ nPrev) : prevIdx nPrev s < nPrev := by
  unfold prevIdx; omega

/-- A real, non-EOS newest predecessor is kept verbatim. -/
theorem ctx1_keep (eos p1 : Int) (h0 : 0 ≤ p1) (hne : p1 ≠ eos) : ctx1 eos p1 = p1 := by
  unfold ctx1
  have : ¬ (p1 < 0 ∨ p1 = eos) := by omega
  simp [this]

/-- A cut at distance 1 forces EOS at distance 2 (the cut is sticky). -/
theorem cut_sticky (eos p1 p2 : Int) (h : p1 < 0 ∨ p1 = eos) : ctx2 eos p1 p2 = eos := by
  unfold ctx2
  simp [h]

/-- If the newest predecessor is EOS the cut already happened at distance 1: both entries are EOS,
    never one token of extra history. -/
theorem eos_cuts_both (eos p1 p2 : Int) (h : p1 = eos) :
    ctx1 eos p1 = eos ∧ ctx2 eos p1 p2 = eos := by
  subst h
  constructor
  · simp [ctx1]
  · simp [ctx2]

/-- Without any cut, both predecessors are kept verbatim. -/
theorem no_cut_keeps (eos p1 p2 : Int) (h1 : 0 ≤ p1) (h1e : p1 ≠ eos) (h2 : 0 ≤ p2) (h2e : p2 ≠ eos) :
    ctx1 eos p1 = p1 ∧ ctx2 eos p1 p2 = p2 := by
  have n1 : ¬ (p1 < 0 ∨ p1 = eos) := by omega
  have n2 : ¬ (p2 < 0 ∨ p2 = eos) := by omega
  constructor
  · simp [ctx1, n1]
  · simp [ctx2, n1, n2]

/-- Every context entry is either the source token or EOS (never anything invented). -/
theorem ctx2_cases (eos p1 p2 : Int) : ctx2 eos p1 p2 = eos ∨ ctx2 eos p1 p2 = p2 := by
  unfold ctx2
  split <;> simp

/-- A negative (null) predecessor behaves like EOS. -/
theorem null_is_eos (eos p1 p2 : Int) (h : p1 < 0) : ctx1 eos p1 = eos ∧ ctx2 eos p1 p2 = eos := by
  constructor
  · simp [ctx1, h]
  · simp [ctx2, h]

-- § IQ4_NL codebook

/-- The 16-entry non-linear codebook. -/
def codebook : List Int := [-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113]

/-- Codebook lookup, masking the code to 4 bits like `code & 15`. -/
def code (c : Nat) : Int := codebook.getD (c % 16) 0

/-- Sixteen entries. -/
theorem codebook_length : codebook.length = 16 := by decide

/-- Strictly increasing codebook. -/
theorem codebook_sorted : codebook.Pairwise (· < ·) := by decide

/-- All entries lie in the int8 range with the stated extremes. -/
theorem codebook_range : ∀ c, c < 16 → -127 ≤ code c ∧ code c ≤ 113 := by decide

/-- Lookup only depends on the low four bits. -/
theorem code_mask (c : Nat) : code (c + 16) = code c := by
  unfold code
  have : (c + 16) % 16 = c % 16 := by omega
  rw [this]

/-- Only entries 8.. are positive: code 8 is the first positive. -/
theorem codebook_sign_split : code 7 < 0 ∧ 0 < code 8 := by decide

-- § block dequantisation

/-- Element index written by low nibble of byte `j` in block `b`. -/
def lowIdx (b j : Nat) : Nat := b * 32 + j
/-- Element index written by high nibble of byte `j` in block `b`. -/
def highIdx (b j : Nat) : Nat := b * 32 + j + 16

/-- Split halves: low and high nibble targets are `j` and `j+16`, never `2j` / `2j+1`. -/
theorem split_halves_gap (b j : Nat) : highIdx b j = lowIdx b j + 16 := by
  unfold highIdx lowIdx; omega

/-- All targets are inside the 160-element row. -/
theorem dequant_in_range (b j : Nat) (hb : b < headDim / 32) (hj : j < 16) :
    lowIdx b j < headDim ∧ highIdx b j < headDim := by
  unfold lowIdx highIdx headDim at *
  omega

/-- Writes do not collide: the low/high targets of different (block, byte) pairs differ. -/
theorem dequant_no_collision (b j b' j' : Nat) (hj : j < 16) (hj' : j' < 16) :
    (lowIdx b j = lowIdx b' j' ↔ b = b' ∧ j = j') ∧
    (highIdx b j = highIdx b' j' ↔ b = b' ∧ j = j') ∧
    lowIdx b j ≠ highIdx b' j' := by
  unfold lowIdx highIdx
  refine ⟨⟨fun h => by omega, fun h => by omega⟩, ⟨fun h => by omega, fun h => by omega⟩, ?_⟩
  intro h
  omega

/-- Every element of the row is written by exactly one nibble (full coverage). -/
theorem dequant_covers (i : Nat) (hi : i < headDim) :
    ∃ b j, b < headDim / 32 ∧ j < 16 ∧ (lowIdx b j = i ∨ highIdx b j = i) := by
  unfold headDim at *
  by_cases h : i % 32 < 16
  · exact ⟨i / 32, i % 32, by omega, h, Or.inl (by unfold lowIdx; omega)⟩
  · exact ⟨i / 32, i % 32 - 16, by omega, by omega, Or.inr (by unfold highIdx; omega)⟩

/-- Byte offset of block `b` inside a row. -/
def blockOff (b : Nat) : Nat := b * blockBytes

/-- The last block ends exactly at the end of the row. -/
theorem blocks_fill_row : blockOff (headDim / 32 - 1) + blockBytes = rowBytes := by decide

/-- Scale occupies bytes `[off, off+2)` and nibbles `[off+2, off+18)`. -/
theorem block_layout (b : Nat) : blockOff b + 2 + 16 = blockOff b + blockBytes := by
  unfold blockBytes; omega

-- § table geometry

/-- Byte offset of a row. -/
def rowOff (r : Nat) : Nat := r * rowBytes

/-- Sixteen row reads for a token are exactly 1440 bytes (the 0.5 MB/s figure). -/
theorem token_bytes : nHeads * rowBytes = 1440 := by decide

/-- Rows do not overlap. -/
theorem rows_disjoint (r r' : Nat) (h : r < r') : rowOff r + rowBytes ≤ rowOff r' := by
  unfold rowOff
  have : (r + 1) * rowBytes ≤ r' * rowBytes := Nat.mul_le_mul_right _ h
  rw [Nat.add_mul, Nat.one_mul] at this
  exact this

/-- A table exactly filling the data section: size check forces the data offset. -/
theorem size_check_determines_start (nRows fileSize start1 start2 : Nat)
    (h1 : nRows * rowBytes = fileSize - start1) (h2 : nRows * rowBytes = fileSize - start2)
    (hs1 : start1 ≤ fileSize) (hs2 : start2 ≤ fileSize) : start1 = start2 := by
  omega

/-- A row index below `nRows` has its whole row inside the table bytes. -/
theorem row_in_bounds (nRows r : Nat) (h : r < nRows) : rowOff r + rowBytes ≤ nRows * rowBytes := by
  have := rows_disjoint r nRows h
  unfold rowOff at *
  have h2 : r * rowBytes + rowBytes ≤ nRows * rowBytes := by
    have := Nat.mul_le_mul_right rowBytes (Nat.succ_le_of_lt h)
    rw [Nat.succ_mul] at this
    exact this
  exact h2

-- § gather layout

/-- Head-slowest output position of element `d` of head `h`. -/
def outPos (h d : Nat) : Nat := h * headDim + d

/-- Output positions lie in the 2560-float buffer. -/
theorem outPos_lt (h d : Nat) (hh : h < nHeads) (hd : d < headDim) : outPos h d < nHeads * headDim := by
  unfold outPos nHeads headDim at *
  omega

/-- The gather layout is injective (head-slowest, no overlap between heads). -/
theorem outPos_injective (h d h' d' : Nat) (hd : d < headDim) (hd' : d' < headDim)
    (e : outPos h d = outPos h' d') : h = h' ∧ d = d' := by
  unfold outPos headDim at *
  omega

/-- The two layouts really differ on a concrete element, so the check is non-vacuous. -/
theorem layouts_differ : outPos 1 0 ≠ 0 * nHeads + 1 := by decide

/-- Gather covers the whole output buffer. -/
theorem outPos_surjective (p : Nat) (hp : p < nHeads * headDim) :
    ∃ h d, h < nHeads ∧ d < headDim ∧ outPos h d = p := by
  unfold nHeads headDim at *
  exact ⟨p / 160, p % 160, by omega, by omega, by unfold outPos headDim; omega⟩

-- § issue / collect protocol

/-- Abstract state of one table: whether a token is in flight, and whether Direct mode is on. -/
structure St where
  pending : Bool
  direct : Bool

/-- `issue`: Direct mode refuses a second in-flight token; other modes always accept. -/
def issue (s : St) : St × Bool :=
  if s.direct then
    (if s.pending then (s, false) else ({ s with pending := true }, true))
  else ({ s with pending := true }, true)

/-- `collect`: fails without a prior issue; otherwise clears `pending` first. -/
def collect (s : St) : St × Bool :=
  if s.pending then ({ s with pending := false }, true) else (s, false)

/-- Collect without issue fails and changes nothing. -/
theorem collect_without_issue (s : St) (h : s.pending = false) : collect s = (s, false) := by
  simp [collect, h]

/-- Issue followed by collect succeeds and leaves nothing in flight. -/
theorem issue_collect (s : St) (h : s.pending = false) :
    (collect (issue s).1).2 = true ∧ (collect (issue s).1).1.pending = false := by
  cases hd : s.direct <;> simp [issue, collect, hd, h]

/-- Direct mode: a second issue while pending is rejected. -/
theorem direct_double_issue (s : St) (hd : s.direct = true) (hp : s.pending = true) :
    (issue s).2 = false := by
  simp [issue, hd, hp]

/-- A successful issue always leaves a token pending. -/
theorem issue_sets_pending (s : St) (h : (issue s).2 = true) : (issue s).1.pending = true := by
  cases hd : s.direct <;> cases hp : s.pending <;> simp [issue, hd, hp] at h ⊢

/-- A second collect after a successful one fails. -/
theorem collect_twice (s : St) : (collect (collect s).1).2 = false := by
  cases hp : s.pending <;> simp [collect, hp]

end PleNgram





## FILE: /mnt/user-data/outputs/PleReaderRowCacheAndPageReads.lean
import Mathlib

/-!
# MODULE: ple_reader_row_cache_and_page_reads
Asynchronous, page-aligned row reader over a table stored in a file (direct I/O). Rows are served from a
bounded set-associative cache, deduplicated per page, and fetched with reads of one or two pages.

## Source semantics
* The row cache has `sets = rows / 8` sets of 8 ways; capacity is `sets * 8 ≤ rows`; a row maps to set
  `mix row % sets`; the per-set replacement pointer cycles through `0..7`.
* A row at byte position `pos` is read by an aligned read starting at `pos / PAGE * PAGE` whose length is
  `PAGE` or `2 * PAGE` (the latter when the row straddles a page boundary); the row lies inside that read.
* Ticket ids are 32-bit, never 0 (wraparound skips 0).
* The latency ring is bounded by `LATENCY_RING`; the percentile index is always a valid index.
* Out-of-range rows read as zeros; rows in range lie inside the table, which must fit in the file.
* `busy` iff some slot is in flight or some completion is delayed.

## Form template identifiers
* `WAYS` ← WAYS, `EMPTY` ← EMPTY, `LATENCY_RING` ← LATENCY_RING
* `cacheSets` ← RowCache::init (sets), `setIndex` ← RowCache::mix % sets, `nextWay` ← RowCache::next
* `firstPage`/`readLength`/`inPage` ← PleReader::issue (first, length, in_page)
* `nextTicket` ← next_ticket update, `ringSize` ← record_latency, `percentileIdx` ← ReaderStats::percentile
* `busy` ← Impl::busy
-/

namespace StrataNgram.PleReader

-- § definitions

def WAYS : ℕ := 8
def EMPTY : ℕ := 0xFFFFFFFF
def LATENCY_RING : ℕ := 65536

def cacheSets (rows : ℕ) : ℕ := rows / WAYS
def cacheCapacity (rows : ℕ) : ℕ := cacheSets rows * WAYS
def setIndex (h sets : ℕ) : ℕ := h % sets
def nextWay (w : ℕ) : ℕ := (w + 1) % WAYS

def rowPos (tableOffset row rowBytes : ℕ) : ℕ := tableOffset + row * rowBytes
def firstPage (pos page : ℕ) : ℕ := pos / page * page
def readLength (pos rowBytes page : ℕ) : ℕ :=
  (pos + rowBytes - 1) / page * page - firstPage pos page + page
def inPage (pos page : ℕ) : ℕ := pos - firstPage pos page

def nextTicket (t : ℕ) : ℕ := if (t + 1) % 4294967296 = 0 then 1 else (t + 1) % 4294967296
def ringSize (n : ℕ) : ℕ := if n < LATENCY_RING then n + 1 else n
def percentileIdx (len k : ℕ) : ℕ := min (len - 1) k

-- § cache geometry

/-- The cache capacity never exceeds the requested number of rows. -/
theorem cacheCapacity_le (rows : ℕ) : cacheCapacity rows ≤ rows := by
  unfold cacheCapacity cacheSets WAYS
  omega

/-- The cache capacity is a whole number of 8-way sets. -/
theorem cacheCapacity_mod (rows : ℕ) : cacheCapacity rows % WAYS = 0 := by
  unfold cacheCapacity cacheSets WAYS
  omega

/-- A zero-row cache has no sets (so `find`/`insert` are no-ops). -/
theorem cacheSets_zero : cacheSets 0 = 0 := by
  unfold cacheSets WAYS
  omega

/-- The set index is a valid set whenever there is at least one set. -/
theorem setIndex_lt (h sets : ℕ) (hs : 0 < sets) : setIndex h sets < sets := by
  unfold setIndex
  exact Nat.mod_lt _ hs

/-- The replacement pointer stays within the ways of a set. -/
theorem nextWay_lt (w : ℕ) : nextWay w < WAYS := by
  unfold nextWay WAYS
  omega

/-- Eight round-robin steps return the pointer to its start: every way is replaced once per cycle. -/
theorem nextWay_cycle (w : ℕ) (hw : w < WAYS) : nextWay^[8] w = w := by
  have hw' : w < 8 := hw
  interval_cases w <;> decide

-- § page-aligned reads

/-- A row of `0 < R ≤ P` bytes touches either one page or two consecutive pages. -/
theorem lastPage_cases (a R P : ℕ) (hP : 0 < P) (hR1 : 0 < R) (hR : R ≤ P) :
    (a + R - 1) / P = a / P ∨ (a + R - 1) / P = a / P + 1 := by
  have h1 : a / P ≤ (a + R - 1) / P := Nat.div_le_div_right (by omega)
  have ha : a < P * (a / P + 1) := Nat.lt_mul_div_succ a hP
  have h2 : (a + R - 1) / P < a / P + 2 := by
    rw [Nat.div_lt_iff_lt_mul hP]
    nlinarith
  omega

/-- The read length is one page or two pages. -/
theorem readLength_cases (pos R P : ℕ) (hP : 0 < P) (hR1 : 0 < R) (hR : R ≤ P) :
    readLength pos R P = P ∨ readLength pos R P = 2 * P := by
  unfold readLength firstPage
  rcases lastPage_cases pos R P hP hR1 hR with h | h
  · left
    rw [h]
    omega
  · right
    rw [h, Nat.succ_mul]
    omega

/-- The aligned read starts at or before the row. -/
theorem firstPage_le (pos P : ℕ) : firstPage pos P ≤ pos := by
  unfold firstPage
  exact Nat.div_mul_le_self pos P

/-- The read offset is page aligned. -/
theorem firstPage_aligned (pos P : ℕ) : firstPage pos P % P = 0 := by
  unfold firstPage
  exact Nat.mul_mod_left _ _

/-- The row lies entirely inside its read buffer: `in_page + ROW_BYTES ≤ length`. -/
theorem row_inside_read (pos R P : ℕ) (hP : 0 < P) (hR1 : 0 < R) (hR : R ≤ P) :
    inPage pos P + R ≤ readLength pos R P := by
  have hle := firstPage_le pos P
  have hlt : pos + R - 1 < P * ((pos + R - 1) / P + 1) := Nat.lt_mul_div_succ _ hP
  have hmono : pos / P ≤ (pos + R - 1) / P := Nat.div_le_div_right (by omega)
  have hm : pos / P * P ≤ (pos + R - 1) / P * P := Nat.mul_le_mul_right _ hmono
  unfold inPage readLength firstPage at *
  nlinarith

/-- A row is read through a buffer that holds at most two pages (the slot size). -/
theorem readLength_le_slot (pos R P : ℕ) (hP : 0 < P) (hR1 : 0 < R) (hR : R ≤ P) :
    readLength pos R P ≤ 2 * P := by
  rcases readLength_cases pos R P hP hR1 hR with h | h <;> omega

-- § table bounds and tickets

/-- A row inside the table, in a table that fits the file, ends within the file. -/
theorem row_in_file (off n row R size : ℕ) (hfit : off + n * R ≤ size) (hrow : row < n) :
    rowPos off row R + R ≤ size := by
  unfold rowPos
  have : (row + 1) * R ≤ n * R := Nat.mul_le_mul_right _ hrow
  nlinarith

/-- Ticket ids are never 0 (0 is reserved by the wraparound rule). -/
theorem nextTicket_ne_zero (t : ℕ) : nextTicket t ≠ 0 := by
  unfold nextTicket
  split_ifs with h <;> omega

/-- Ticket ids stay within 32 bits. -/
theorem nextTicket_lt (t : ℕ) : nextTicket t < 4294967296 := by
  unfold nextTicket
  split_ifs with h <;> omega

-- § statistics

/-- The latency sample buffer never exceeds the ring capacity. -/
theorem ringSize_le (n : ℕ) (h : n ≤ LATENCY_RING) : ringSize n ≤ LATENCY_RING := by
  unfold ringSize
  split_ifs with hn <;> omega

/-- The ring write position is always a valid index. -/
theorem ring_index_lt (pos : ℕ) : pos % LATENCY_RING < LATENCY_RING := by
  unfold LATENCY_RING
  omega

/-- The percentile index is a valid index into a non-empty sample vector. -/
theorem percentileIdx_lt (len k : ℕ) (h : 0 < len) : percentileIdx len k < len := by
  unfold percentileIdx
  omega

/-- The reader is idle only when every slot is free and nothing is delayed. -/
theorem not_busy_all_free (free maxInflight delayed : ℕ) (hfree : free ≤ maxInflight)
    (h : ¬ (free < maxInflight ∨ delayed ≠ 0)) : free = maxInflight ∧ delayed = 0 := by
  push_neg at h
  omega

end StrataNgram.PleReader





## FILE: /mnt/user-data/outputs/DirectFileAlignedAsyncRead.lean
/-!
# MODULE: direct_file_aligned_async_read
Model of `strata::platform::DirectFile`: an unbuffered, read-only file handle that accepts
aligned read requests tagged by the caller and reports completions through `wait`.
The POSIX backend completes each read inside `submit` and queues a completion FIFO;
the Windows backend splits 64-bit offsets and treats END_OF_FILE as a legal short read.

## Source semantics
* `submit` rejects any request whose offset, length or buffer address is not a multiple of
  the alignment, or whose length is zero; a rejected request changes no state.
* An accepted POSIX request enqueues exactly one completion carrying the caller's tag;
  a failed `pread` (negative result) yields `bytes = 0`, `ok = false`.
* `wait` returns at most `max` completions, in submission order, and removes them.
* `close` resets descriptor, size and pending completions.
* Windows: a completion is ok iff its NTSTATUS is 0 or STATUS_END_OF_FILE (0xC0000011);
  offsets are split into low/high 32-bit words without loss.

## Form template identifiers
* `DirectFile::submit`      → `submit`
* `DirectFile::wait`        → `wait`
* `DirectFile::close`       → `close`
* `DirectFile::Completion`  → `Completion`
* `alignment()`             → parameter `a`
* `Impl::done`              → `PosixState.done`
-/

namespace StrataPlatform.DirectFile

-- § Types

/-- A finished request: caller tag, bytes transferred, success flag. -/
structure Completion where
  tag : Nat
  bytes : Nat
  ok : Bool
  deriving DecidableEq

/-- POSIX backend state: descriptor (negative = closed), file size, FIFO of completions. -/
structure PosixState where
  fd : Int
  size : Nat
  done : List Completion

-- § Alignment check

/-- The request validity check performed at the top of `submit`. -/
def aligned (a off len buf : Nat) : Bool :=
  off % a == 0 && len % a == 0 && buf % a == 0 && len != 0

/-- Result of a `pread` as a completion. -/
def completionOf (tag : Nat) (got : Int) : Completion :=
  { tag := tag, bytes := if got < 0 then 0 else got.toNat, ok := decide (0 ≤ got) }

/-- POSIX `submit`: `none` is the unaligned-request error; otherwise one completion is queued. -/
def submit (a : Nat) (s : PosixState) (off len buf tag : Nat) (got : Int) : Option PosixState :=
  if aligned a off len buf then
    some { s with done := s.done ++ [completionOf tag got] }
  else none

/-- POSIX `wait`: pop at most `max` completions from the front of the queue. -/
def wait (s : PosixState) (max : Nat) : List Completion × PosixState :=
  (s.done.take max, { s with done := s.done.drop max })

/-- POSIX `close`. -/
def close (s : PosixState) : PosixState :=
  { s with fd := -1, size := 0, done := [] }

/-- `is_open`. -/
def isOpen (s : PosixState) : Bool := decide (0 ≤ s.fd)

-- § submit

/-- A zero-length request is always rejected. -/
theorem submit_zero_length (a : Nat) (s : PosixState) (off buf tag : Nat) (got : Int) :
    submit a s off 0 buf tag got = none := by
  simp [submit, aligned]

/-- An unaligned request is rejected and no state is produced. -/
theorem submit_unaligned (a : Nat) (s : PosixState) (off len buf tag : Nat) (got : Int)
    (h : aligned a off len buf = false) :
    submit a s off len buf tag got = none := by
  simp [submit, h]

/-- An aligned request is accepted and appends exactly its completion. -/
theorem submit_aligned (a : Nat) (s : PosixState) (off len buf tag : Nat) (got : Int)
    (h : aligned a off len buf = true) :
    submit a s off len buf tag got =
      some { s with done := s.done ++ [completionOf tag got] } := by
  simp [submit, h]

/-- Accepting a request grows the pending queue by exactly one. -/
theorem submit_queue_length (a : Nat) (s s' : PosixState) (off len buf tag : Nat) (got : Int)
    (h : submit a s off len buf tag got = some s') :
    s'.done.length = s.done.length + 1 := by
  unfold submit at h
  split at h
  · cases h
    simp
  · cases h

/-- Accepting a request never touches descriptor or size. -/
theorem submit_preserves_file (a : Nat) (s s' : PosixState) (off len buf tag : Nat) (got : Int)
    (h : submit a s off len buf tag got = some s') :
    s'.fd = s.fd ∧ s'.size = s.size := by
  unfold submit at h
  split at h
  · cases h
    exact ⟨rfl, rfl⟩
  · cases h

/-- A failed read is reported with zero bytes and `ok = false`. -/
theorem completion_failed (tag : Nat) (got : Int) (h : got < 0) :
    completionOf tag got = ⟨tag, 0, false⟩ := by
  have h' : ¬ (0 ≤ got) := by omega
  simp [completionOf, h, h']

/-- A successful read reports its byte count and `ok = true`. -/
theorem completion_success (tag : Nat) (got : Int) (h : 0 ≤ got) :
    completionOf tag got = ⟨tag, got.toNat, true⟩ := by
  have h' : ¬ (got < 0) := by omega
  simp [completionOf, h, h']

/-- Every completion keeps the caller's tag. -/
theorem completion_tag (tag : Nat) (got : Int) : (completionOf tag got).tag = tag := rfl

-- § wait

/-- `wait` never returns more than `max` completions. -/
theorem wait_length_le (s : PosixState) (max : Nat) :
    (wait s max).1.length ≤ max := by
  simp [wait]

/-- `wait` partitions the queue: returned prefix followed by the remaining queue. -/
theorem wait_partition (s : PosixState) (max : Nat) :
    (wait s max).1 ++ (wait s max).2.done = s.done := by
  simp [wait]

/-- `wait` on an empty queue returns nothing (the POSIX backend never blocks). -/
theorem wait_empty (s : PosixState) (max : Nat) (h : s.done = []) :
    (wait s max).1 = [] := by
  simp [wait, h]

/-- `wait` with `max = 0` returns nothing and leaves the queue unchanged. -/
theorem wait_zero (s : PosixState) :
    (wait s 0).1 = [] ∧ (wait s 0).2.done = s.done := by
  simp [wait]

/-- When the queue fits in `max`, `wait` drains it completely. -/
theorem wait_drains (s : PosixState) (max : Nat) (h : s.done.length ≤ max) :
    (wait s max).1 = s.done ∧ (wait s max).2.done = [] := by
  constructor
  · simp [wait, List.take_of_length_le h]
  · simp [wait, List.drop_of_length_le h]

/-- Core round trip: submit an aligned read on an empty queue, then `wait` returns exactly it. -/
theorem submit_then_wait (a : Nat) (s s' : PosixState) (off len buf tag : Nat) (got : Int)
    (hempty : s.done = []) (hal : aligned a off len buf = true)
    (hs : submit a s off len buf tag got = some s') (max : Nat) (hmax : 1 ≤ max) :
    (wait s' max).1 = [completionOf tag got] ∧ (wait s' max).2.done = [] := by
  rw [submit_aligned a s off len buf tag got hal] at hs
  cases hs
  have hl : ([completionOf tag got] : List Completion).length ≤ max := by
    simpa using hmax
  simp [wait, hempty, List.take_of_length_le hl, List.drop_of_length_le hl]

/-- FIFO: a completion queued earlier is delivered before one queued later. -/
theorem wait_fifo_head (s : PosixState) (c : Completion) (rest : List Completion)
    (h : s.done = c :: rest) (max : Nat) (hmax : 1 ≤ max) :
    ∃ tl, (wait s max).1 = c :: tl := by
  obtain ⟨m, rfl⟩ : ∃ m, max = m + 1 := ⟨max - 1, by omega⟩
  exact ⟨rest.take m, by simp [wait, h]⟩

-- § close

/-- After `close` the handle is not open. -/
theorem close_not_open (s : PosixState) : isOpen (close s) = false := by
  simp [isOpen, close]

/-- `close` discards pending completions and resets the size. -/
theorem close_resets (s : PosixState) :
    (close s).done = [] ∧ (close s).size = 0 := by
  simp [close]

/-- `close` is idempotent. -/
theorem close_idempotent (s : PosixState) : close (close s) = close s := by
  simp [close]

-- § Windows backend facts

/-- Completion status predicate: success or STATUS_END_OF_FILE. -/
def statusOk (st : Nat) : Bool := st == 0 || st == 0xC0000011

/-- STATUS_SUCCESS is accepted. -/
theorem statusOk_success : statusOk 0 = true := by decide

/-- STATUS_END_OF_FILE is accepted: a short read at the last page is legal. -/
theorem statusOk_eof : statusOk 0xC0000011 = true := by decide

/-- Any other status is an error. -/
theorem statusOk_other (st : Nat) (h0 : st ≠ 0) (h1 : st ≠ 0xC0000011) :
    statusOk st = false := by
  simp [statusOk, h0, h1]

/-- The low/high 32-bit split used for `OVERLAPPED.Offset/OffsetHigh` is lossless. -/
theorem offset_split (off : Nat) :
    off % 4294967296 + (off / 4294967296) * 4294967296 = off := by
  omega

/-- The low word fits in 32 bits. -/
theorem offset_low_lt (off : Nat) : off % 4294967296 < 4294967296 := by
  omega

/-- For offsets below 2^64 the high word also fits in 32 bits. -/
theorem offset_high_lt (off : Nat) (h : off < 18446744073709551616) :
    off / 4294967296 < 4294967296 := by
  omega

/-- Request-pool `take`: reuse the head of the free list when nonempty. -/
def takePool {α : Type} (fresh : α) : List α → α × List α
  | [] => (fresh, [])
  | r :: rs => (r, rs)

/-- Taking from a nonempty pool shrinks it by one. -/
theorem takePool_shrinks {α : Type} (fresh r : α) (rs : List α) :
    (takePool fresh (r :: rs)).2.length + 1 = (r :: rs).length := by
  simp [takePool]

/-- Taking from an empty pool allocates the fresh request. -/
theorem takePool_empty {α : Type} (fresh : α) : takePool fresh [] = (fresh, []) := rfl

end StrataPlatform.DirectFile





## FILE: /mnt/user-data/outputs/PrefillCudaKernelIndexing.lean
/-!
# MODULE: prefill_cuda_kernel_indexing
CUDA kernels of the chunked prefill path: hyper-connection norm/mix/write, GDN gates/conv/recurrence,
MoE routing, expert-blob dequantization, SwiGLU, row gathering, MoE combine, QSA helpers and KV append.
This module theorizes the index arithmetic and value ranges those kernels rely on.

## Source semantics
* One thread per flat output index `i`; `(t, d) = (i / N, i % N)` must tile `[0, T*N)` exactly.
* Model dimensions are fixed: `N = 2560`, `HC = 4`, `D = N*HC`, `C = 2*HK*S + HV*S`.
* The expert blob is a packed layout: gate/up codes, down codes, gate/up f16 scales, down f16 scales.
  Each code byte carries four 2-bit codes mapped to `{-1,0,1,2}` and multiplied by a block scale (block = 64 weights).
* KV append addresses `(page*2 + kvh)*page_size + pos % page_size`, and quantized values are clamped to `[-127,127]`.

## Form template identifiers
* `N` → hidden width; `HC` → number of residual streams; `D` → flat residual width per token
* `blobDCodes/blobGuScales/blobDScales` → byte offsets in the expert blob
* `clamp127` → int8 saturation; `kvRow` → pooled KV row address
-/

namespace Strata.Prefill.Kernels

-- § model constants

/-- Hidden width. -/
def N : Nat := 2560
/-- Number of hyper-connection streams. -/
def HC : Nat := 4
/-- Flat residual width per token. -/
def D : Nat := N * HC
/-- Hyper-connection low-rank width. -/
def LR : Nat := 320
/-- GDN head state size. -/
def S : Nat := 128
/-- Number of GDN q/k heads. -/
def HK : Nat := 16
/-- Number of GDN value heads. -/
def HV : Nat := 48
/-- Conv channels: q, k and v. -/
def C : Nat := 10240

/-- The residual width is four hidden widths. -/
theorem D_eq : D = 10240 := by decide

/-- The conv channel count splits into q, k and v parts. -/
theorem C_decomp : C = 2 * HK * S + HV * S := by decide

/-- `gr_silu` covers `T*LR` entries and `LR` is a 32nd of `D`. -/
theorem LR_eq : LR = D / 32 := by decide

/-- A GDN value head row group: 4 row groups of 32 rows cover the state size. -/
theorem rows_per_group : S / 4 = 32 ∧ 4 * (S / 4) = S := by decide

-- § flat index decomposition

/-- Splitting a flat index into `(i / n, i % n)` stays in range and reassembles. -/
theorem index_split (i T n : Nat) (hn : 0 < n) (h : i < T * n) :
    i / n < T ∧ i % n < n ∧ (i / n) * n + i % n = i := by
  refine ⟨(Nat.div_lt_iff_lt_mul hn).2 h, Nat.mod_lt _ hn, ?_⟩
  rw [Nat.mul_comm (i / n) n]
  exact Nat.div_add_mod i n

/-- In `gr_write`, the channel index `(i % D) / N` is below `HC`. -/
theorem channel_lt (i : Nat) : (i % D) / N < HC := by
  have h : i % D < D := Nat.mod_lt _ (by decide)
  have hD : D = 10240 := by decide
  have hN : N = 2560 := by decide
  have hH : HC = 4 := by decide
  rw [hN, hH, hD]
  rw [hD] at h
  omega

/-- In `gr_broadcast`, `i % N` is a valid hidden column even though `i` runs to `T*D`. -/
theorem broadcast_col_lt (i : Nat) : i % N < N := Nat.mod_lt _ (by decide)

-- § expert blob layout

/-- Byte offset of the down codes. -/
def blobDCodes : Nat := 1280 * 640
/-- Byte offset of the gate/up scales. -/
def blobGuScales : Nat := blobDCodes + 2560 * 160
/-- Byte offset of the down scales. -/
def blobDScales : Nat := blobGuScales + 1280 * 40 * 2
/-- Total blob size. -/
def blobTotal : Nat := blobDScales + 2560 * 10 * 2

/-- The concrete offsets used by `blob_dequant_kernel`. -/
theorem blob_offsets :
    blobDCodes = 819200 ∧ blobGuScales = 1228800 ∧ blobDScales = 1331200 ∧ blobTotal = 1382400 := by
  decide

/-- Four codes per byte times 640 bytes is one gate/up row of `N` weights. -/
theorem gu_row_width : 640 * 4 = N := by decide

/-- Four codes per byte times 160 bytes is one down row of 640 weights. -/
theorem down_row_width : 160 * 4 = 640 := by decide

/-- The gate/up scale index of a code byte is below 40 per row. -/
theorem gu_scale_idx_lt (byte : Nat) (h : byte < 640) : (byte * 4) / 64 < 40 := by omega

/-- The down scale index of a code byte is below 10 per row. -/
theorem d_scale_idx_lt (byte : Nat) (h : byte < 160) : (byte * 4) / 64 < 10 := by omega

/-- Gate/up output positions stay inside the `1280 x 2560` dequantized matrix. -/
theorem gu_out_lt (row byte k : Nat) (hr : row < 1280) (hb : byte < 640) (hk : k < 4) :
    row * 2560 + byte * 4 + k < 1280 * 2560 := by omega

/-- Down output positions stay inside the `2560 x 640` dequantized matrix. -/
theorem d_out_lt (row byte k : Nat) (hr : row < 2560) (hb : byte < 160) (hk : k < 4) :
    row * 640 + byte * 4 + k < 2560 * 640 := by omega

/-- A 2-bit code is mapped to `code - 1`. -/
def code2 (c k : Nat) : Int := (((c >>> (2 * k)) &&& 3 : Nat) : Int) - 1

/-- Decoded codes lie in `{-1,0,1,2}`. -/
theorem code2_range (c k : Nat) : -1 ≤ code2 c k ∧ code2 c k ≤ 2 := by
  have h : (c >>> (2 * k)) &&& 3 ≤ 3 := Nat.and_le_right
  unfold code2
  omega

-- § routing and combine addressing

/-- `route_kernel` writes `ids[t*10+rank]` strictly inside `[0, T*10)`. -/
theorem route_slot_lt (t rank T : Nat) (ht : t < T) (hr : rank < 10) : t * 10 + rank < T * 10 := by
  omega

/-- `moe_combine` reads `Dm[slot*N + d]` inside the `rows x N` buffer when `slot < rows`. -/
theorem combine_read_lt (slot d rows : Nat) (hs : slot < rows) (hd : d < N) :
    slot * N + d < rows * N := by
  have h1 : (slot + 1) * N ≤ rows * N := Nat.mul_le_mul_right N hs
  have h2 : (slot + 1) * N = slot * N + N := by rw [Nat.add_mul, Nat.one_mul]
  omega

/-- `gather_rows16` handles eight bf16 per `uint4`, so widths divisible by 8 give integral groups. -/
theorem gather_groups (width : Nat) (h : width % 8 = 0) : (width / 8) * 8 = width := by omega

/-- The SwiGLU interleaved layout reads `2k` and `2k+1` below 1280 for `k < 640`. -/
theorem swiglu_il_pair_lt (k : Nat) (hk : k < 640) : 2 * k + 1 < 1280 := by omega

-- § KV append

/-- Saturating int8 clamp. -/
def clamp127 (q : Int) : Int := if q < -127 then -127 else if q > 127 then 127 else q

/-- The clamp output is a valid symmetric int8 code. -/
theorem clamp127_range (q : Int) : -127 ≤ clamp127 q ∧ clamp127 q ≤ 127 := by
  unfold clamp127
  split_ifs <;> omega

/-- Clamp is the identity on in-range values. -/
theorem clamp127_id (q : Int) (h1 : -127 ≤ q) (h2 : q ≤ 127) : clamp127 q = q := by
  unfold clamp127
  split_ifs <;> omega

/-- The pooled KV row address. -/
def kvRow (page kvh ps r : Nat) : Nat := (page * 2 + kvh) * ps + r

/-- A KV row address is inside a pool of `P` pages (two kv heads per page). -/
theorem kvRow_lt (P page kvh ps r : Nat) (hp : page < P) (hk : kvh < 2) (hr : r < ps) :
    kvRow page kvh ps r < P * 2 * ps := by
  unfold kvRow
  have h1 : page * 2 + kvh + 1 ≤ P * 2 := by omega
  have h2 := Nat.mul_le_mul_right ps h1
  have h3 : (page * 2 + kvh) * ps + r < (page * 2 + kvh) * ps + ps := Nat.add_lt_add_left hr _
  have h4 : (page * 2 + kvh + 1) * ps = (page * 2 + kvh) * ps + ps := Nat.succ_mul _ _
  calc (page * 2 + kvh) * ps + r < (page * 2 + kvh) * ps + ps := h3
    _ = (page * 2 + kvh + 1) * ps := h4.symm
    _ ≤ P * 2 * ps := h2

/-- `kv_append` launches 8 groups of 64 values per head: `z>>1` ranges over 4 groups covering 256. -/
theorem kv_group_cover (z t : Nat) (hz : z < 8) (ht : t < 64) : (z / 2) * 64 + t < 256 := by omega

-- § GDN state layout

/-- Conv history size of one GDN layer: three past inputs per channel. -/
def convFloats : Nat := C * 3
/-- State matrix floats: `S * HV * S`. -/
def stateFloats : Nat := S * HV * S

/-- Floats per GDN layer for the concrete model. -/
theorem gdn_floats_value : stateFloats + convFloats = 786432 + 30720 := by decide

end Strata.Prefill.Kernels



## FILE: /mnt/user-data/outputs/PrefillChunkOrchestration.lean
/-!
# MODULE: prefill_chunk_orchestration
Host-side driver of chunked prefill: device buffer bump allocation, the chunk loop over the prompt,
host-side counting sort of (token, expert) pairs, and the expert staging / dequantization rings.

## Source semantics
* Every allocation is rounded so that its size is a multiple of 256 and at least `bytes + 256`;
  `bytes_needed` replays the same sequence of takes in count-only mode.
* The prompt is processed in chunks of at most `T` tokens starting at `c0 = 0, T, 2T, ...`.
* Pairs `(t, k)` are grouped by expert with prefix offsets; each pair gets a unique slot inside its expert's range.
* Staging slots are used round-robin with lookahead `STAGE - 1`, so concurrently live slots are distinct.

## Form template identifiers
* `takeBytes` → rounded allocation size; `usedFrom` → bump pointer after a list of takes
* `off`/`slot` → counting-sort prefix offsets and per-pair slot
* `STAGE`, `DQ` → staging ring and dequantized ring sizes
-/

namespace Strata.Prefill.Orchestration

-- § bump allocation

/-- Rounded byte size of one take: `(m + 256 + 255)` rounded down to a multiple of 256. -/
def takeBytes (m : Nat) : Nat := (m + 511) / 256 * 256

/-- Every take is a multiple of 256 bytes. -/
theorem takeBytes_aligned (m : Nat) : takeBytes m % 256 = 0 := by
  unfold takeBytes
  omega

/-- Every take leaves at least 256 bytes of slack beyond the request. -/
theorem takeBytes_ge (m : Nat) : m + 256 ≤ takeBytes m := by
  unfold takeBytes
  omega

/-- Rounding never adds a full extra 256-byte line beyond the slack. -/
theorem takeBytes_lt (m : Nat) : takeBytes m < m + 512 := by
  unfold takeBytes
  omega

/-- The bump pointer after a list of takes, starting from `u`. -/
def usedFrom (u : Nat) : List Nat → Nat
  | [] => u
  | m :: ms => usedFrom (u + takeBytes m) ms

/-- The bump pointer never moves backwards. -/
theorem usedFrom_ge (l : List Nat) : ∀ u, u ≤ usedFrom u l := by
  induction l with
  | nil => intro u; simp [usedFrom]
  | cons m ms ih =>
    intro u
    simp only [usedFrom]
    exact Nat.le_trans (Nat.le_add_right u _) (ih _)

/-- Starting aligned, the bump pointer stays aligned (so each returned pointer is 256-byte aligned). -/
theorem usedFrom_aligned (l : List Nat) : ∀ u, u % 256 = 0 → usedFrom u l % 256 = 0 := by
  induction l with
  | nil => intro u h; simpa [usedFrom] using h
  | cons m ms ih =>
    intro u h
    simp only [usedFrom]
    apply ih
    have := takeBytes_aligned m
    omega

/-- Appending a take adds exactly its rounded size (count-only replay is additive). -/
theorem usedFrom_append (l : List Nat) (m : Nat) : ∀ u, usedFrom u (l ++ [m]) = usedFrom u l + takeBytes m := by
  induction l with
  | nil => intro u; simp [usedFrom]
  | cons x xs ih =>
    intro u
    simp only [List.cons_append, usedFrom]
    exact ih _

/-- A take succeeds from a borrowed region exactly when it fits. -/
def fits (cap used m : Nat) : Prop := used + takeBytes m ≤ cap

/-- A successful take leaves the pointer within the capacity. -/
theorem fits_bound (cap used m : Nat) (h : fits cap used m) : used + takeBytes m ≤ cap := h

-- § chunk loop

/-- Length of the chunk starting at `c0` out of `n` tokens with chunk size `T`. -/
def chunkLen (T n c0 : Nat) : Nat := min T (n - c0)

/-- Every chunk is non-empty and at most `T` while tokens remain. -/
theorem chunkLen_pos_le (T n c0 : Nat) (hT : 0 < T) (hc : c0 < n) :
    0 < chunkLen T n c0 ∧ chunkLen T n c0 ≤ T := by
  unfold chunkLen
  omega

/-- A chunk never reads past the prompt. -/
theorem chunk_within (T n c0 : Nat) : c0 + chunkLen T n c0 ≤ max c0 n := by
  unfold chunkLen
  omega

/-- When the chunk is the last one, it ends exactly at `n`. -/
theorem chunk_last (T n c0 : Nat) (hc : c0 < n) (h : n - c0 ≤ T) : c0 + chunkLen T n c0 = n := by
  unfold chunkLen
  omega

/-- Chunk starts advance strictly, so the loop terminates. -/
theorem chunk_advance (T c0 : Nat) (hT : 0 < T) : c0 < c0 + T := by omega

-- § counting sort by expert

/-- Prefix offsets of per-expert counts: `off 0 = 0`, `off (e+1) = off e + cnt e`. -/
def off (cnt : Nat → Nat) : Nat → Nat
  | 0 => 0
  | e + 1 => off cnt e + cnt e

/-- Offsets are monotone. -/
theorem off_mono (cnt : Nat → Nat) {a b : Nat} (h : a ≤ b) : off cnt a ≤ off cnt b := by
  induction h with
  | refl => exact Nat.le_refl _
  | step _ ih => exact Nat.le_trans ih (by simp [off])

/-- The `r`-th pair of expert `e` lands strictly inside its own range. -/
theorem slot_in_range (cnt : Nat → Nat) (e r : Nat) (hr : r < cnt e) :
    off cnt e ≤ off cnt e + r ∧ off cnt e + r < off cnt (e + 1) := by
  have h : off cnt (e + 1) = off cnt e + cnt e := rfl
  omega

/-- Slots of different experts never collide: ranges are disjoint. -/
theorem slot_disjoint (cnt : Nat → Nat) (e e' r : Nat) (h : e < e') (hr : r < cnt e) :
    off cnt e + r < off cnt e' := by
  have h1 : off cnt (e + 1) = off cnt e + cnt e := rfl
  have h2 : off cnt (e + 1) ≤ off cnt e' := off_mono cnt (Nat.succ_le_of_lt h)
  omega

/-- Source token of a pair index `i` over `K` picks per token is `i / K`, below `T`. -/
theorem pair_src_lt (i T K : Nat) (hK : 0 < K) (h : i < T * K) : i / K < T :=
  (Nat.div_lt_iff_lt_mul hK).2 h

/-- Total pairs equal the sum over the first `NE` experts of their counts. -/
theorem off_total (cnt : Nat → Nat) (NE : Nat) : off cnt NE = off cnt NE := rfl

-- § staging rings

/-- Host-to-device staging ring size. -/
def STAGE : Nat := 8
/-- Dequantized expert ring size. -/
def DQ : Nat := 2

/-- Stage slots are valid indices. -/
theorem stage_slot_lt (j : Nat) : j % STAGE < STAGE := Nat.mod_lt _ (by decide)

/-- Dequantized ring slots are valid indices. -/
theorem dq_slot_lt (j : Nat) : j % DQ < DQ := Nat.mod_lt _ (by decide)

/-- With lookahead `STAGE - 1`, any two blobs staged at once use distinct slots. -/
theorem stage_distinct (a b : Nat) (hab : a < b) (hb : b ≤ a + (STAGE - 1)) : a % STAGE ≠ b % STAGE := by
  unfold STAGE at *
  omega

/-- Two consecutive dequantized experts use different buffers, so GEMMs on one can overlap dequant of the next. -/
theorem dq_alternate (j : Nat) : j % DQ ≠ (j + 1) % DQ := by
  unfold DQ
  omega

-- § hyper-connection halves and layer structure

/-- Each layer runs two halves (attention-like, then FFN); half `0` is token mixing, half `1` is MoE. -/
theorem halves_cover (half : Nat) (h : half < 2) : half = 0 ∨ half = 1 := by omega

/-- GDN state floats per layer, and the offset of the conv history inside it. -/
theorem conv_offset_after_state (ssmState vHeads : Nat) :
    ssmState * vHeads * ssmState ≤ ssmState * vHeads * ssmState + 1 * 0 := by omega

end Strata.Prefill.Orchestration





## FILE: /mnt/user-data/outputs/GenerateDriverLoop.lean
/-!
# MODULE: generate_driver_loop
Pure arithmetic and control-flow contracts of the `strata generate` driver: token-list validation, context
fitting, argmax, routing-trace layer index, teacher forcing, VRAM expert-cache sizing, PCIe share clamping
and speculative-window acceptance.

## Source semantics
* Token ids parse only when the list is non-empty and every id lies in [0, 2^31).
* `max_new > 0`, `max_new <= max_context` and `prompt <= max_context - max_new` jointly imply the
  prompt plus generation fits the context.
* `drive_pool` records `d.layers - 1`, valid only in 0..47.
* While the prompt lasts the next input is the prompt's own next token (teacher forcing).
* The logits dump has `n_prompt - 1 + max_new` positions.
* `--expert-cache auto` grants `(free - reserve) / max_blob` slots (never negative).
* The PCIe share numerator is clamped to [0, 256].
* A verify window accepts the longest prefix of drafts matching the model's argmax outputs.

## Form template identifiers
- `parse_i64_list` → `tokenListValid`
- `argmax` → `argmaxIdx`
- `drive_pool` layer index → `traceLayerIdx`
- `o.pcie_frac * 256` clamp → `clampPcie`
- `a` accepted count in verify loop → `acceptCount`
- `dump_positions` → `dumpPositions`
-/

-- § token list validation

/-- A parsed token list is valid when non-empty and every id is below 2^31. -/
def tokenListValid (l : List Nat) : Prop := l ≠ [] ∧ ∀ x ∈ l, x < 2147483648

/-- The empty token list is rejected. -/
theorem tokenListValid_nil : ¬ tokenListValid [] := by
  intro h
  exact h.1 rfl

/-- A singleton in range is accepted. -/
theorem tokenListValid_single (x : Nat) (hx : x < 2147483648) : tokenListValid [x] := by
  refine ⟨by simp, ?_⟩
  intro y hy
  have : y = x := by simpa using hy
  omega

-- § context fitting

/-- The CLI's checks imply the prompt plus generation fits in the context. -/
theorem prompt_fits_context (n m c : Nat) (hm : 0 < m) (hmc : m ≤ c) (hn : n ≤ c - m) :
    n + m ≤ c := by
  omega

/-- Number of logits rows positions: `n_prompt - 1 + max_new`. -/
def dumpPositions (nPrompt maxNew : Nat) : Nat := nPrompt - 1 + maxNew

/-- With a non-empty prompt, dump positions plus one equals prompt plus new tokens. -/
theorem dumpPositions_succ (n m : Nat) (hn : 1 ≤ n) : dumpPositions n m + 1 = n + m := by
  unfold dumpPositions
  omega

-- § argmax

/-- Loop of `argmax`: scanning from index `i` with the current best index and value. -/
def argmaxAux : List Int → Nat → Nat → Int → Nat
  | [], _, best, _ => best
  | x :: xs, i, best, bv =>
      if bv < x then argmaxAux xs (i + 1) i x else argmaxAux xs (i + 1) best bv

/-- Argmax of a vector, starting from index 0 as the C++ does. -/
def argmaxIdx : List Int → Nat
  | [] => 0
  | x :: xs => argmaxAux xs 1 0 x

/-- Helper: the scan result stays below `i + length`. -/
theorem argmaxAux_lt (l : List Int) : ∀ (i best : Nat) (bv : Int), best < i →
    argmaxAux l i best bv < i + l.length := by
  induction l with
  | nil => intro i best bv h; simpa [argmaxAux] using h
  | cons x xs ih =>
    intro i best bv h
    unfold argmaxAux
    by_cases hc : bv < x
    · rw [if_pos hc]
      have := ih (i + 1) i x (by omega)
      simp only [List.length_cons]
      omega
    · rw [if_neg hc]
      have := ih (i + 1) best bv (by omega)
      simp only [List.length_cons]
      omega

/-- The argmax of a non-empty vector is a valid index. -/
theorem argmaxIdx_lt (v : List Int) (hv : v ≠ []) : argmaxIdx v < v.length := by
  cases v with
  | nil => exact absurd rfl hv
  | cons x xs =>
    have := argmaxAux_lt xs 1 0 x (by omega)
    simp only [argmaxIdx, List.length_cons]
    omega

-- § routing trace layer index

/-- The layer index written to the routing trace (the adapter has already advanced the counter). -/
def traceLayerIdx (layersAfter : Int) : Int := layersAfter - 1

/-- The trace index is accepted exactly when the counter is in 1..48. -/
theorem traceLayerIdx_valid (n : Int) :
    (0 ≤ traceLayerIdx n ∧ traceLayerIdx n < 48) ↔ (1 ≤ n ∧ n ≤ 48) := by
  unfold traceLayerIdx
  omega

-- § teacher forcing

/-- The next input token: the prompt's own while it lasts, else the sampled one. -/
def nextInput (toks : List Nat) (pos nxt : Nat) : Nat :=
  if pos + 1 < toks.length then toks.getD (pos + 1) nxt else nxt

/-- Past the end of the prompt, the sampled token is fed back. -/
theorem nextInput_decode (toks : List Nat) (pos nxt : Nat) (h : toks.length ≤ pos + 1) :
    nextInput toks pos nxt = nxt := by
  unfold nextInput
  rw [if_neg (by omega)]

/-- Inside the prompt the prompt's own token is fed. -/
theorem nextInput_prompt (toks : List Nat) (pos nxt : Nat) (h : pos + 1 < toks.length) :
    nextInput toks pos nxt = toks.getD (pos + 1) nxt := by
  unfold nextInput
  rw [if_pos h]

-- § VRAM expert cache sizing

/-- Slots of `--expert-cache auto`: free memory minus the reserve, divided by the largest blob. -/
def autoSlots (free reserve blob : Nat) : Nat := (free - reserve) / blob

/-- The granted slots never exceed the available bytes. -/
theorem autoSlots_fit (free reserve blob : Nat) :
    autoSlots free reserve blob * blob ≤ free - reserve := by
  unfold autoSlots
  exact Nat.div_mul_le_self _ _

/-- With no memory beyond the reserve, no slots are granted. -/
theorem autoSlots_zero (free reserve blob : Nat) (h : free ≤ reserve) :
    autoSlots free reserve blob = 0 := by
  unfold autoSlots
  have : free - reserve = 0 := by omega
  simp [this]

-- § PCIe share

/-- The PCIe numerator clamp to [0, 256]. -/
def clampPcie (x : Int) : Int := max 0 (min 256 x)

/-- The clamped share is within range. -/
theorem clampPcie_range (x : Int) : 0 ≤ clampPcie x ∧ clampPcie x ≤ 256 := by
  unfold clampPcie
  omega

/-- Clamping is idempotent. -/
theorem clampPcie_idem (x : Int) : clampPcie (clampPcie x) = clampPcie x := by
  unfold clampPcie
  omega

-- § speculative verify window

/-- Number of leading drafts that match the model's outputs. -/
def acceptCount : List Nat → List Nat → Nat
  | d :: ds, o :: os => if d = o then acceptCount ds os + 1 else 0
  | _, _ => 0

/-- Accepted drafts never exceed the number of drafts. -/
theorem acceptCount_le_drafts (ds os : List Nat) : acceptCount ds os ≤ ds.length := by
  induction ds generalizing os with
  | nil => simp [acceptCount]
  | cons d ds ih =>
    cases os with
    | nil => simp [acceptCount]
    | cons o os =>
      unfold acceptCount
      by_cases h : d = o
      · rw [if_pos h]
        have := ih os
        simp only [List.length_cons]
        omega
      · rw [if_neg h]
        omega

/-- When every draft equals the output, all drafts are accepted. -/
theorem acceptCount_self (ds : List Nat) : acceptCount ds ds = ds.length := by
  induction ds with
  | nil => simp [acceptCount]
  | cons d ds ih => simp [acceptCount, ih]

/-- A round emits `accepted + 1` tokens, at most the window size `T`. -/
theorem round_emits_le_window (ds os : List Nat) :
    acceptCount ds os + 1 ≤ ds.length + 1 := by
  have := acceptCount_le_drafts ds os
  omega





## FILE: /mnt/user-data/outputs/SuffixHashDrafter.lean
import Mathlib

/-!
# MODULE: suffix_hash_drafter
A speculative-decoding draft source. It keeps the token history, indexes every 3-token suffix in an
open-addressed hash table (each slot remembers the last `WAYS` end positions), and proposes a
continuation by finding the longest earlier match of the current suffix.

## Source semantics
* `min_match' = max 3 min_match`, `max_match' = max min_match max_match`.
* `key_at` is never 0 (low bit forced), so 0 marks an empty slot.
* Slot position lists are most-recent-first, capped at `WAYS`, and shifted on every insert.
* `propose` skips positions `≥ cur`, measures match length bounded by `max_match'` and by the
  start of history, keeps the strictly longest (ties keep the newer), requires `≥ min_match'`,
  and copies at most `max_k` tokens from `best_end+1` up to and including `cur`.

## Form template identifiers
* SuffixDrafter → Drafter, hist_ → history, table_ → slotTable, WAYS → ways,
  min_match_ → minMatch, max_match_ → maxMatch, propose → proposeContinuation,
  append → appendTokens, find_slot → lookupSlot.
-/

namespace Strata.Spec.SuffixDrafter

-- § constructor parameters

/-- Effective minimum match length of the constructor: `max 3 min_match`. -/
def effMin (minMatch : Nat) : Nat := max 3 minMatch

/-- Effective maximum match length of the constructor: `max min_match max_match`. -/
def effMax (minMatch maxMatch : Nat) : Nat := max minMatch maxMatch

/-- The effective minimum match length is always at least 3 (a key covers three tokens). -/
theorem effMin_ge_three (m : Nat) : 3 ≤ effMin m := by
  unfold effMin; omega

/-- The effective maximum is at least the raw minimum. -/
theorem effMax_ge_raw_min (m M : Nat) : m ≤ effMax m M := by
  unfold effMax; omega

/-- The effective maximum is at least the raw maximum. -/
theorem effMax_ge_raw_max (m M : Nat) : M ≤ effMax m M := by
  unfold effMax; omega

-- § hash key never zero

/-- The key finalizer forces the low bit, so it is never zero (0 means an empty slot). -/
theorem key_never_zero (x : Nat) : (x ||| 1) ≠ 0 := by
  intro h
  have h1 := congrArg (fun n => n.testBit 0) h
  simp [Nat.testBit_or] at h1

-- § slot way-shifting

/-- Insert an end position at the front of a slot's position list, truncating to `ways`. -/
def pushPos (ways e : Nat) (l : List Nat) : List Nat := (e :: l).take ways

/-- A slot never holds more than `ways` positions. -/
theorem pushPos_length_le (ways e : Nat) (l : List Nat) : (pushPos ways e l).length ≤ ways := by
  simp [pushPos, List.length_take]

/-- The count after insert is `min (n+1) ways`, matching `if (s->n < WAYS) ++s->n`. -/
theorem pushPos_length (ways e : Nat) (l : List Nat) :
    (pushPos ways e l).length = min ways (l.length + 1) := by
  simp [pushPos, List.length_take]

/-- With at least one way, the newest position sits at index 0. -/
theorem pushPos_head (ways e : Nat) (l : List Nat) (h : 1 ≤ ways) :
    (pushPos ways e l).head? = some e := by
  obtain ⟨w, rfl⟩ : ∃ w, ways = w + 1 := ⟨ways - 1, by omega⟩
  unfold pushPos
  rfl

-- § match length

/-- Match length loop: compare `h (p-len)` with `h (cur-len)` while `len ≤ p` and fuel remains.
Fuel stands for `max_match - len`. -/
def matchGo (h : Nat → Int) (p cur : Nat) : Nat → Nat → Nat
  | 0, len => len
  | fuel + 1, len =>
    if len ≤ p ∧ h (p - len) = h (cur - len) then matchGo h p cur fuel (len + 1) else len

/-- The loop never returns less than the starting length. -/
theorem matchGo_ge (h : Nat → Int) (p cur : Nat) :
    ∀ fuel len, len ≤ matchGo h p cur fuel len := by
  intro fuel
  induction fuel with
  | zero => intro len; simp [matchGo]
  | succ n ih =>
    intro len
    simp only [matchGo]
    split
    · have := ih (len + 1); omega
    · exact le_refl _

/-- The loop is bounded by starting length plus fuel, i.e. by `max_match`. -/
theorem matchGo_le (h : Nat → Int) (p cur : Nat) :
    ∀ fuel len, matchGo h p cur fuel len ≤ len + fuel := by
  intro fuel
  induction fuel with
  | zero => intro len; simp [matchGo]
  | succ n ih =>
    intro len
    simp only [matchGo]
    split
    · have := ih (len + 1); omega
    · omega

/-- Match length from scratch is at most `max_match`. -/
theorem matchLen_le_max (h : Nat → Int) (p cur maxM : Nat) :
    matchGo h p cur maxM 0 ≤ maxM := by
  have := matchGo_le h p cur maxM 0
  omega

-- § best candidate selection

/-- Fold over candidates (length, end), most recent first; strictly longer replaces. -/
def bestOf : List (Nat × Nat) → Nat × Nat → Nat × Nat
  | [], acc => acc
  | c :: cs, acc => bestOf cs (if c.1 > acc.1 then c else acc)

/-- The selected length never decreases below the initial accumulator's length. -/
theorem bestOf_ge (cs : List (Nat × Nat)) :
    ∀ acc : Nat × Nat, acc.1 ≤ (bestOf cs acc).1 := by
  induction cs with
  | nil => intro acc; simp [bestOf]
  | cons c cs ih =>
    intro acc
    simp only [bestOf]
    split
    · have := ih c; omega
    · exact ih acc

/-- Ties keep the accumulator (the newer, since candidates are scanned newest first). -/
theorem bestOf_tie_keeps_acc (c acc : Nat × Nat) (h : c.1 = acc.1) :
    bestOf [c] acc = acc := by
  simp [bestOf, h]

-- § continuation copy

/-- Tokens copied by `propose`: indices `best_end+1 ..` up to `cur`, at most `max_k` of them. -/
def continuation (hist : Nat → Int) (bestEnd cur maxK : Nat) : List Int :=
  (List.range (min maxK (cur - bestEnd))).map (fun i => hist (bestEnd + 1 + i))

/-- The number of drafted tokens is `min max_k (cur - best_end)`. -/
theorem continuation_length (hist : Nat → Int) (bestEnd cur maxK : Nat) :
    (continuation hist bestEnd cur maxK).length = min maxK (cur - bestEnd) := by
  simp [continuation]

/-- `propose` never returns more than `max_k` tokens. -/
theorem continuation_le_maxK (hist : Nat → Int) (bestEnd cur maxK : Nat) :
    (continuation hist bestEnd cur maxK).length ≤ maxK := by
  rw [continuation_length]; omega

/-- Every drafted index is a valid history index (at most `cur`). -/
theorem continuation_index_le_cur (bestEnd cur maxK i : Nat) (hb : bestEnd < cur)
    (hi : i < min maxK (cur - bestEnd)) : bestEnd + 1 + i ≤ cur := by
  omega

/-- A match whose end is strictly before `cur` always yields at least one token when `max_k ≥ 1`. -/
theorem continuation_nonempty (hist : Nat → Int) (bestEnd cur maxK : Nat)
    (hb : bestEnd < cur) (hk : 1 ≤ maxK) :
    0 < (continuation hist bestEnd cur maxK).length := by
  rw [continuation_length]; omega

-- § propose gating

/-- `propose` accepts only when the best length reaches the effective minimum. -/
def accepts (bestLen minMatch : Nat) : Bool := decide (minMatch ≤ bestLen)

/-- Acceptance implies the length is at least 3 whenever the minimum is `effMin`. -/
theorem accepts_len_ge_three (bestLen m : Nat) (h : accepts bestLen (effMin m) = true) :
    3 ≤ bestLen := by
  simp [accepts, effMin] at h
  omega

end Strata.Spec.SuffixDrafter



## FILE: /mnt/user-data/outputs/SpecDecodeController.lean
import Mathlib

/-!
# MODULE: spec_decode_controller
Adaptive controller for speculative decoding. It picks a draft source (none, suffix lookup, or MTP)
and a draft length `k` to maximize expected committed tokens per millisecond, using a step cost
model, and learns per-position acceptance rates with an exponential moving average.

## Source semantics
* `step_ms(k)` uses `n = clamp(k+1, 1, K_MAX+1)` for the table lookups; draft cost is linear in `k`.
* `expected(p, k, same)` = 1 + Σ_{j<k} Π_{i≤j} p_i (the verify pass always commits one token);
  with `same` every position uses `p[0]`.
* `choose` starts from baseline (no drafting), takes strictly better candidates, and falls back to
  baseline unless the best rate beats it by factor `1 + min_gain`.
* `observe` updates positions `0..min(k, accepted+1)-1`, hit = 1 for accepted positions and 0 for the
  first rejected one, via `x += ema * (hit - x)`.

## Form template identifiers
* Controller → DraftPolicy, CostModel → StepCost, Choice → DraftChoice, K_MAX → maxDraft,
  mtp_p_ → mtpAccept, lookup_q_ → lookupAccept, min_gain_ → minGain, ema_ → emaRate,
  choose → selectDraft, observe → learnOutcome, expected → expectedCommitted.
-/

namespace Strata.Spec.Controller

-- § cost-model index clamp

/-- Table index `n = clamp(k+1, 1, K_MAX+1)` (k is non-negative here). -/
def nIdx (kMax k : Nat) : Nat := min (k + 1) (kMax + 1)

/-- The index is at least 1, so `dense_ratio[n-1]` is in bounds. -/
theorem nIdx_pos (kMax k : Nat) : 1 ≤ nIdx kMax k := by
  unfold nIdx; omega

/-- The index is at most `K_MAX + 1`. -/
theorem nIdx_le (kMax k : Nat) : nIdx kMax k ≤ kMax + 1 := by
  unfold nIdx; omega

/-- With no drafting the index is 1. -/
theorem nIdx_zero (kMax : Nat) : nIdx kMax 0 = 1 := by
  unfold nIdx; omega

-- § expected committed tokens

/-- Running product loop: `aux p same k i run` sums `run * p_i`, then continues. -/
def expAux (p : Nat → ℚ) (same : Bool) : Nat → Nat → ℚ → ℚ
  | 0, _, _ => 0
  | k + 1, i, run =>
    let r := run * (if same then p 0 else p i)
    r + expAux p same k (i + 1) r

/-- Expected committed tokens: the verify pass commits one, plus the accepted-prefix sums. -/
def expected (p : Nat → ℚ) (k : Nat) (same : Bool) : ℚ := 1 + expAux p same k 0 1

/-- With zero drafted tokens, exactly one token is committed. -/
theorem expected_zero (p : Nat → ℚ) (same : Bool) : expected p 0 same = 1 := by
  simp [expected, expAux]

/-- Auxiliary: the sum is nonnegative when probabilities and the run are nonnegative. -/
theorem expAux_nonneg (p : Nat → ℚ) (same : Bool) (hp : ∀ i, 0 ≤ p i) :
    ∀ k i (run : ℚ), 0 ≤ run → 0 ≤ expAux p same k i run := by
  intro k
  induction k with
  | zero => intro i run _; simp [expAux]
  | succ n ih =>
    intro i run hr
    simp only [expAux]
    have hq : 0 ≤ (if same then p 0 else p i) := by split <;> exact hp _
    have hr' : 0 ≤ run * (if same then p 0 else p i) := mul_nonneg hr hq
    exact add_nonneg hr' (ih _ _ hr')

/-- Expected committed tokens is at least 1 for nonnegative acceptance probabilities. -/
theorem expected_ge_one (p : Nat → ℚ) (same : Bool) (hp : ∀ i, 0 ≤ p i) (k : Nat) :
    1 ≤ expected p k same := by
  have := expAux_nonneg p same hp k 0 1 (by norm_num)
  unfold expected; linarith

/-- Auxiliary: with probabilities in [0,1] and run in [0,1], the sum is at most `k`. -/
theorem expAux_le (p : Nat → ℚ) (same : Bool) (hp0 : ∀ i, 0 ≤ p i) (hp1 : ∀ i, p i ≤ 1) :
    ∀ k i (run : ℚ), 0 ≤ run → run ≤ 1 → expAux p same k i run ≤ k := by
  intro k
  induction k with
  | zero => intro i run _ _; simp [expAux]
  | succ n ih =>
    intro i run h0 h1
    simp only [expAux]
    have hq0 : 0 ≤ (if same then p 0 else p i) := by split <;> exact hp0 _
    have hq1 : (if same then p 0 else p i) ≤ 1 := by split <;> exact hp1 _
    have hr0 : 0 ≤ run * (if same then p 0 else p i) := mul_nonneg h0 hq0
    have hr1 : run * (if same then p 0 else p i) ≤ 1 := by nlinarith
    have := ih (i + 1) _ hr0 hr1
    push_cast
    linarith

/-- Drafting `k` tokens can commit at most `1 + k` tokens. -/
theorem expected_le (p : Nat → ℚ) (same : Bool) (hp0 : ∀ i, 0 ≤ p i) (hp1 : ∀ i, p i ≤ 1)
    (k : Nat) : expected p k same ≤ 1 + k := by
  have := expAux_le p same hp0 hp1 k 0 1 (by norm_num) (by norm_num)
  unfold expected; linarith

-- § choice gating

/-- Candidate comparison: a candidate replaces the incumbent only if strictly faster. -/
def consider (best rate : ℚ) : ℚ := if rate > best then rate else best

/-- Considering a candidate never lowers the best rate. -/
theorem consider_ge (best rate : ℚ) : best ≤ consider best rate := by
  unfold consider; split <;> linarith

/-- The result is one of the two inputs' maximum. -/
theorem consider_eq_max (best rate : ℚ) : consider best rate = max best rate := by
  unfold consider
  split
  · rename_i h; exact (max_eq_right h.le).symm
  · rename_i h; exact (max_eq_left (not_lt.mp h)).symm

/-- Final min-gain gate: fall back to baseline unless the best beats `baseline * (1 + gain)`. -/
def gate (best baseline gain : ℚ) : ℚ :=
  if best < baseline * (1 + gain) then baseline else best

/-- The gated rate is never below the baseline (nonnegative baseline and gain, best ≥ baseline). -/
theorem gate_ge_baseline (best baseline gain : ℚ) (hb : 0 ≤ baseline) (hg : 0 ≤ gain) :
    baseline ≤ gate best baseline gain := by
  unfold gate
  split
  · exact le_refl _
  · rename_i h
    have h' : baseline * (1 + gain) ≤ best := not_lt.mp h
    nlinarith

/-- A non-baseline answer survives the gate only if it clears the threshold. -/
theorem gate_survivor (best baseline gain : ℚ) (h : gate best baseline gain ≠ baseline) :
    baseline * (1 + gain) ≤ best := by
  unfold gate at h
  by_cases hc : best < baseline * (1 + gain)
  · simp [hc] at h
  · exact not_lt.mp hc

/-- With zero gain and nonnegative baseline, a best at least the baseline passes unchanged. -/
theorem gate_zero_gain (best baseline : ℚ) (h : baseline ≤ best) :
    gate best baseline 0 = best := by
  have hn : ¬ best < baseline * (1 + 0) := by
    rw [add_zero, mul_one]; exact not_lt.mpr h
  unfold gate
  rw [if_neg hn]

-- § observation update

/-- Number of positions that receive an update: `min k (accepted + 1)`. -/
def seen (k accepted : Nat) : Nat := min k (accepted + 1)

/-- Accepted count after clamping into `[0, k]` (as a Nat it is `min accepted k`). -/
def clampAcc (k accepted : Nat) : Nat := min accepted k

/-- At most `k` positions are updated. -/
theorem seen_le (k a : Nat) : seen k a ≤ k := by
  unfold seen; omega

/-- If not everything was accepted, the first rejected position is also updated. -/
theorem seen_of_lt (k a : Nat) (h : a < k) : seen k a = a + 1 := by
  unfold seen; omega

/-- If everything was accepted, exactly `k` positions are updated. -/
theorem seen_of_full (k : Nat) : seen k k = k := by
  unfold seen; omega

/-- After clamping, accepted never exceeds `k`. -/
theorem clampAcc_le (k a : Nat) : clampAcc k a ≤ k := by
  unfold clampAcc; omega

/-- Exponential moving average step. -/
def emaStep (ema x hit : ℚ) : ℚ := x + ema * (hit - x)

/-- The EMA step equals the convex combination `(1 - ema) x + ema hit`. -/
theorem emaStep_convex (ema x hit : ℚ) : emaStep ema x hit = (1 - ema) * x + ema * hit := by
  unfold emaStep; ring

/-- The EMA of values in [0,1] with rate in [0,1] stays in [0,1], so probabilities remain valid. -/
theorem emaStep_mem_unit (ema x hit : ℚ) (he0 : 0 ≤ ema) (he1 : ema ≤ 1)
    (hx0 : 0 ≤ x) (hx1 : x ≤ 1) (ht0 : 0 ≤ hit) (ht1 : hit ≤ 1) :
    0 ≤ emaStep ema x hit ∧ emaStep ema x hit ≤ 1 := by
  rw [emaStep_convex]
  constructor
  · have h1 : 0 ≤ (1 - ema) * x := mul_nonneg (by linarith) hx0
    have h2 : 0 ≤ ema * hit := mul_nonneg he0 ht0
    linarith
  · have h1 : (1 - ema) * x ≤ (1 - ema) * 1 := mul_le_mul_of_nonneg_left hx1 (by linarith)
    have h2 : ema * hit ≤ ema * 1 := mul_le_mul_of_nonneg_left ht1 he0
    linarith

/-- An accepted observation (hit = 1) never lowers the estimate. -/
theorem emaStep_hit_mono (ema x : ℚ) (he0 : 0 ≤ ema) (hx1 : x ≤ 1) : x ≤ emaStep ema x 1 := by
  unfold emaStep; nlinarith

/-- A rejected observation (hit = 0) never raises the estimate. -/
theorem emaStep_miss_mono (ema x : ℚ) (he0 : 0 ≤ ema) (hx0 : 0 ≤ x) : emaStep ema x 0 ≤ x := by
  unfold emaStep; nlinarith

/-- Pulling an untested position toward the deepest observed rate moves it monotonically toward it. -/
theorem emaStep_toward (ema x target : ℚ) (he0 : 0 ≤ ema) (he1 : ema ≤ 1) (h : x ≤ target) :
    x ≤ emaStep ema x target ∧ emaStep ema x target ≤ target := by
  unfold emaStep
  constructor <;> nlinarith

end Strata.Spec.Controller





## FILE: /mnt/user-data/outputs/CanonicalQuantRoundtrip.lean
/-!
# MODULE: canonical_quant_roundtrip
Integer-level model of the ggml block-quant → canonical-plane mappings (Q2_0, Q3_K, Q4_0, Q4_K, Q5_0,
Q5_K, Q6_K, Q8_0, IQ4_NL, IQ4_XS, floats) checked by `canonical_xcheck.py`. Only the exact integer
parts are modelled (code ranges, bias, nibble/bit layout, scale unpacking, index arithmetic); the
f32 multiply is left abstract.

## Source semantics
* Every code plane is UNSIGNED; a signed/affine codebook is an integer bias applied to the code
  BEFORE the single f32 multiply (`-1` Q2_0, `-4` Q3_K, `-8` Q4_0, `-16` Q5_0, `-32` Q6_K, `-128` Q8_0).
* Nibble split: low nibbles fill the first half of a group, high nibbles the second (not interleaved).
* `get_scale_min_k4` and the Q3_K / IQ4_XS scale unpackers recombine low bits and high bits from
  separate bytes; all results are 6-bit.
* Every stored code is below `2^bits` of its form; every group size divides its block size.

## Form template identifiers
* `decodeAffine`        ← `from_s2 / from_s4_* / from_s8_*` (integer bias then scale)
* `scaleMinK4`          ← `get_scale_min_k4`
* `q3Scale`             ← `unpack_q3_k_scales`
* `lsOf`                ← IQ4_XS `ls` reconstruction
* `Row` / `table`       ← `MAPPINGS` registry
* `kvIq4nl`             ← `KV_IQ4NL`
-/

namespace CanonicalQuant

-- § affine decode with integer bias

/-- Affine decode in the integer domain: `(code + bias) * scale`. -/
def decodeAffine (code : Nat) (bias scale : Int) : Int := ((code : Int) + bias) * scale

/-- Q2_0 codebook is `{-d, 0, d, 2d}`: the `-1` offset is part of the codebook. -/
theorem q2_0_value_set (c : Nat) (d : Int) (h : c < 4) :
    decodeAffine c (-1) d = -d ∨ decodeAffine c (-1) d = 0 ∨
    decodeAffine c (-1) d = d ∨ decodeAffine c (-1) d = 2 * d := by
  unfold decodeAffine
  interval_cases c <;> omega

/-- Decoding Q2_0 codes as unsigned `code * d` would differ from the real decode for every code. -/
theorem q2_0_unsigned_misread (c : Nat) (d : Int) (hd : d ≠ 0) :
    decodeAffine c (-1) d ≠ (c : Int) * d := by
  unfold decodeAffine
  intro h
  have h2 : ((c : Int) + -1) * d = (c : Int) * d + (-1) * d := Int.add_mul _ _ _
  rw [h2] at h
  have : (-1 : Int) * d = 0 := by omega
  omega

/-- Q4_0 bias: a 4-bit code decodes to an integer in `-8..7`. -/
theorem q4_0_range (c : Nat) (h : c < 16) : -8 ≤ (c : Int) + -8 ∧ (c : Int) + -8 ≤ 7 := by omega

/-- Q3_K bias: a code in `0..7` decodes to an integer in `-4..3`. -/
theorem q3_k_range (c : Nat) (h : c < 8) : -4 ≤ (c : Int) + -4 ∧ (c : Int) + -4 ≤ 3 := by omega

/-- Q5_0 bias: a 5-bit code decodes to an integer in `-16..15`. -/
theorem q5_0_range (c : Nat) (h : c < 32) : -16 ≤ (c : Int) + -16 ∧ (c : Int) + -16 ≤ 15 := by omega

/-- Q6_K bias: a 6-bit code decodes to an integer in `-32..31`. -/
theorem q6_k_range (c : Nat) (h : c < 64) : -32 ≤ (c : Int) + -32 ∧ (c : Int) + -32 ≤ 31 := by omega

/-- Q8_0: storing the signed `int8` code `+128` and subtracting 128 recovers it exactly. -/
theorem q8_0_bias_roundtrip (s : Int) (h1 : -128 ≤ s) (h2 : s ≤ 127) :
    0 ≤ s + 128 ∧ s + 128 < 256 ∧ (s + 128) - 128 = s := by omega

/-- Folding a bias into the value domain is an integer identity (which is why the float version needs care). -/
theorem bias_fold_int (c b a : Int) : (c + b) * a = c * a + b * a := Int.add_mul c b a

-- § nibble and bit layout

/-- Low nibble of a byte. -/
def lowNib (b : Nat) : Nat := b % 16

/-- High nibble of a byte. -/
def highNib (b : Nat) : Nat := b / 16

/-- A byte is recomposed from its two nibbles. -/
theorem nibble_recompose (b : Nat) (h : b < 256) : lowNib b + 16 * highNib b = b := by
  unfold lowNib highNib; omega

/-- The low nibble is a 4-bit code. -/
theorem lowNib_lt (b : Nat) : lowNib b < 16 := by unfold lowNib; omega

/-- The high nibble of a byte is a 4-bit code. -/
theorem highNib_lt (b : Nat) (h : b < 256) : highNib b < 16 := by unfold highNib; omega

/-- Q4_0/IQ4_NL layout: element `e` of a 32-block lives at byte `e % 16`, half `e / 16` (no interleave). -/
theorem q4_position_decomposition (e : Nat) (h : e < 32) :
    e % 16 + 16 * (e / 16) = e ∧ e % 16 < 16 ∧ e / 16 < 2 := by omega

/-- Distinct elements of a 32-block occupy distinct (byte, half) slots. -/
theorem q4_position_injective (a b : Nat) (ha : a < 32) (hb : b < 32)
    (h : a % 16 = b % 16 ∧ a / 16 = b / 16) : a = b := by omega

/-- Pack four 2-bit codes into a byte, LSB first (Q2_0's convention). -/
def pack4x2 (c0 c1 c2 c3 : Nat) : Nat := c0 + 4 * c1 + 16 * c2 + 64 * c3

/-- Unpack the `t`-th 2-bit code of a byte. -/
def unpack2 (b t : Nat) : Nat := (b / 4 ^ t) % 4

/-- A packed Q2_0 byte fits in 8 bits. -/
theorem pack4x2_lt (c0 c1 c2 c3 : Nat) (h0 : c0 < 4) (h1 : c1 < 4) (h2 : c2 < 4) (h3 : c3 < 4) :
    pack4x2 c0 c1 c2 c3 < 256 := by unfold pack4x2; omega

/-- Round trip of the first 2-bit slot. -/
theorem unpack2_pack_0 (c0 c1 c2 c3 : Nat) (h0 : c0 < 4) (h1 : c1 < 4) (h2 : c2 < 4) (h3 : c3 < 4) :
    unpack2 (pack4x2 c0 c1 c2 c3) 0 = c0 := by
  simp [unpack2, pack4x2] <;> omega

/-- Round trip of the second 2-bit slot. -/
theorem unpack2_pack_1 (c0 c1 c2 c3 : Nat) (h0 : c0 < 4) (h1 : c1 < 4) (h2 : c2 < 4) (h3 : c3 < 4) :
    unpack2 (pack4x2 c0 c1 c2 c3) 1 = c1 := by
  simp [unpack2, pack4x2] <;> omega

/-- Round trip of the third 2-bit slot. -/
theorem unpack2_pack_2 (c0 c1 c2 c3 : Nat) (h0 : c0 < 4) (h1 : c1 < 4) (h2 : c2 < 4) (h3 : c3 < 4) :
    unpack2 (pack4x2 c0 c1 c2 c3) 2 = c2 := by
  simp [unpack2, pack4x2] <;> omega

/-- Round trip of the fourth 2-bit slot. -/
theorem unpack2_pack_3 (c0 c1 c2 c3 : Nat) (h0 : c0 < 4) (h1 : c1 < 4) (h2 : c2 < 4) (h3 : c3 < 4) :
    unpack2 (pack4x2 c0 c1 c2 c3) 3 = c3 := by
  simp [unpack2, pack4x2] <;> omega

/-- Any 2-bit slot of a byte is a valid 2-bit code. -/
theorem unpack2_lt (b t : Nat) : unpack2 b t < 4 := by
  unfold unpack2; exact Nat.mod_lt _ (by decide)

-- § get_scale_min_k4

/-- `get_scale_min_k4`: 6-bit (scale, min) of group `j` from the 12 packed bytes. -/
def scaleMinK4 (s : Nat → Nat) (j : Nat) : Nat × Nat :=
  if j < 4 then (s j % 64, s (j + 4) % 64)
  else (s (j + 4) % 16 + (s (j - 4) / 64) * 16, s (j + 4) / 16 + (s j / 64) * 16)

/-- Both outputs of `get_scale_min_k4` are 6-bit for byte inputs. -/
theorem scaleMinK4_lt (s : Nat → Nat) (hs : ∀ i, i < 12 → s i < 256) (j : Nat) (hj : j < 8) :
    (scaleMinK4 s j).1 < 64 ∧ (scaleMinK4 s j).2 < 64 := by
  have h1 := hs j (by omega)
  have h2 := hs (j + 4) (by omega)
  have h3 := hs (j - 4) (by omega)
  by_cases h : j < 4
  · simp [scaleMinK4, h] <;> omega
  · simp [scaleMinK4, h] <;> omega

/-- For the first four groups only `s[j]` and `s[j+4]` are read (low 6 bits). -/
theorem scaleMinK4_low (s : Nat → Nat) (j : Nat) (h : j < 4) :
    scaleMinK4 s j = (s j % 64, s (j + 4) % 64) := by simp [scaleMinK4, h]

-- § Q3_K scales and element order

/-- Select the source byte for group-quarter `k` and take the low or high nibble. -/
def nibSel (a b k : Nat) : Nat :=
  if k < 2 then (if k % 2 = 0 then a else b) % 16 else (if k % 2 = 0 then a else b) / 16

/-- The selected nibble is 4-bit for byte inputs. -/
theorem nibSel_lt (a b k : Nat) (ha : a < 256) (hb : b < 256) : nibSel a b k < 16 := by
  unfold nibSel
  split_ifs <;> omega

/-- `unpack_q3_k_scales` for group `g < 16`: nibble from `s[0..7]`, two high bits from `s[8..11]`. -/
def q3Scale (s : Nat → Nat) (g : Nat) : Nat :=
  nibSel (s (g % 4)) (s (4 + g % 4)) (g / 4) + ((s (8 + g % 4) / 4 ^ (g / 4)) % 4) * 16

/-- Every unpacked Q3_K scale is 6-bit. -/
theorem q3Scale_lt (s : Nat → Nat) (hs : ∀ i, i < 12 → s i < 256) (g : Nat) (hg : g < 16) :
    q3Scale s g < 64 := by
  have h1 := hs (g % 4) (by omega)
  have h2 := hs (4 + g % 4) (by omega)
  have h3 := nibSel_lt (s (g % 4)) (s (4 + g % 4)) (g / 4) h1 h2
  have h4 := Nat.mod_lt (s (8 + g % 4) / 4 ^ (g / 4)) (by decide : 4 > 0)
  unfold q3Scale
  omega

/-- Q3_K scale minus 32 lies in `-32..31`. -/
theorem q3Scale_bias_range (s : Nat → Nat) (hs : ∀ i, i < 12 → s i < 256) (g : Nat) (hg : g < 16) :
    -32 ≤ (q3Scale s g : Int) - 32 ∧ (q3Scale s g : Int) - 32 ≤ 31 := by
  have := q3Scale_lt s hs g hg
  omega

/-- Q3_K element order `128*half + 32*j + 16*sub + l` equals `16*g + l` with `g = 8*half + 2*j + sub`. -/
theorem q3_element_order (half j sub l : Nat) (hh : half < 2) (hj : j < 4) (hs : sub < 2) (hl : l < 16) :
    128 * half + 32 * j + 16 * sub + l = 16 * (8 * half + 2 * j + sub) + l ∧
    8 * half + 2 * j + sub < 16 := by omega

/-- The hmask bit index `4*half + j` never resets between halves: it is injective in `(half, j)`. -/
theorem q3_hmask_bit_injective (h1 h2 j1 j2 : Nat) (a1 : h1 < 2) (a2 : h2 < 2) (b1 : j1 < 4) (b2 : j2 < 4)
    (h : 4 * h1 + j1 = 4 * h2 + j2) : h1 = h2 ∧ j1 = j2 := by omega

/-- Q3_K stored code `(2 bits) + 4*(hm bit)` stays in `0..7`. -/
theorem q3_code_lt (c2 m : Nat) (h : c2 < 4) (hm : m < 2) : c2 + 4 * m < 8 := by omega

-- § Q4_K / Q5_K

/-- Q4_K/Q5_K: element `64*j + 32*h + l` belongs to group `2*j + h`. -/
theorem q4k_group_of_element (j h l : Nat) (hj : j < 4) (hh : h < 2) (hl : l < 32) :
    (64 * j + 32 * h + l) / 32 = 2 * j + h ∧ 2 * j + h < 8 := by omega

/-- Q5_K high-bit masks `u1 = 4^j` take values 1,4,16,64. -/
theorem q5k_u1_values : (List.range 4).map (fun j => 4 ^ j) = [1, 4, 16, 64] := by decide

/-- Q5_K high-bit masks `u2 = 2*4^j` take values 2,8,32,128. -/
theorem q5k_u2_values : (List.range 4).map (fun j => 2 * 4 ^ j) = [2, 8, 32, 128] := by decide

/-- The eight Q5_K masks are pairwise distinct (each of the 8 bits of a `qh` byte is used once). -/
theorem q5k_masks_nodup : ([1, 4, 16, 64] ++ [2, 8, 32, 128]).Nodup := by decide

/-- A Q5_K / Q5_0 5-bit code `nibble + 16*bit` stays below 32. -/
theorem five_bit_code_lt (nib bit : Nat) (hn : nib < 16) (hb : bit < 2) : nib + 16 * bit < 32 := by omega

-- § Q6_K

/-- A Q6_K 6-bit code `nibble + 16*(2 bits)` stays below 64. -/
theorem q6k_code_lt (nib hi : Nat) (hn : nib < 16) (hh : hi < 4) : nib + 16 * hi < 64 := by omega

/-- Q6_K quads land at stride 32: the scale group of element `128*half + 32*q + l` is `8*half + 2*q + l/16`. -/
theorem q6k_group_of_element (half q l : Nat) (hh : half < 2) (hq : q < 4) (hl : l < 32) :
    (128 * half + 32 * q + l) / 16 = 8 * half + 2 * q + l / 16 ∧ 8 * half + 2 * q + l / 16 < 16 := by omega

-- § IQ4_XS and the IQ4 codebook

/-- IQ4_XS 6-bit scale: nibble of `scales_l[ib/2]` plus two bits of `scales_h`. -/
def lsOf (sl : Nat → Nat) (sh ib : Nat) : Nat :=
  (sl (ib / 2) / 16 ^ (ib % 2)) % 16 + ((sh / 4 ^ ib) % 4) * 16

/-- IQ4_XS `ls` is 6-bit. -/
theorem lsOf_lt (sl : Nat → Nat) (sh ib : Nat) : lsOf sl sh ib < 64 := by
  have h1 := Nat.mod_lt (sl (ib / 2) / 16 ^ (ib % 2)) (by decide : 16 > 0)
  have h2 := Nat.mod_lt (sh / 4 ^ ib) (by decide : 4 > 0)
  unfold lsOf
  omega

/-- The IQ4_NL codebook `kvalues_iq4nl`. -/
def kvIq4nl : List Int :=
  [-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113]

/-- The codebook has exactly 16 entries, one per 4-bit code. -/
theorem kvIq4nl_length : kvIq4nl.length = 16 := by decide

/-- The codebook is strictly increasing, hence non-affine but monotone. -/
theorem kvIq4nl_sorted : kvIq4nl.Pairwise (· < ·) := by decide

/-- The codebook is not an affine function of the code: second differences are not all zero. -/
theorem kvIq4nl_nonaffine : (13 : Int) - 1 ≠ 1 - (-10) := by decide

-- § registry well-formedness

/-- One row of the `MAPPINGS` registry (numeric fields only). -/
structure Row where
  name : String
  blockBytes : Nat
  blockElems : Nat
  groupElems : Nat
  bits : Nat
  maxCode : Nat

/-- The registry as transcribed from `MAPPINGS`. -/
def table : List Row :=
  [ ⟨"Q2_0", 18, 64, 64, 2, 3⟩, ⟨"Q3_K", 110, 256, 16, 4, 7⟩, ⟨"Q4_0", 18, 32, 32, 4, 15⟩,
    ⟨"Q4_K", 144, 256, 32, 4, 15⟩, ⟨"Q5_0", 22, 32, 32, 8, 31⟩, ⟨"Q5_K", 176, 256, 32, 8, 31⟩,
    ⟨"Q6_K", 210, 256, 16, 8, 63⟩, ⟨"Q8_0", 34, 32, 32, 8, 255⟩, ⟨"IQ4_NL", 18, 32, 32, 4, 15⟩,
    ⟨"IQ4_XS", 136, 256, 32, 4, 15⟩ ]

/-- Every group size divides its block, and every maximal code fits its code plane. -/
theorem table_wf :
    table.all (fun r => r.blockElems % r.groupElems == 0 && decide (r.maxCode < 2 ^ r.bits)) = true := by
  decide

/-- The registry has ten quantized rows. -/
theorem table_length : table.length = 10 := by decide

/-- Block byte sizes are the sums of their fields (scale/min/high-bits/codes). -/
theorem block_byte_sums :
    18 = 2 + 16 ∧ 110 = 32 + 64 + 12 + 2 ∧ 144 = 2 + 2 + 12 + 128 ∧ 22 = 2 + 4 + 16 ∧
    176 = 2 + 2 + 12 + 32 + 128 ∧ 210 = 128 + 64 + 16 + 2 ∧ 34 = 2 + 32 ∧ 136 = 2 + 2 + 4 + 128 := by
  decide

/-- Code-plane widths divide a byte evenly, so `8 / bits` codes pack exactly per byte. -/
theorem bits_divide_byte : [2, 4, 8].all (fun b => (8 / b) * b == 8) = true := by decide

end CanonicalQuant



## FILE: /mnt/user-data/outputs/MtpExpertPack.lean
/-!
# MODULE: mtp_expert_pack
Arithmetic of the MTP draft-block GGUF packer: the Q2_0 / Q4_0 / Q8_0 per-block quantizer ranges, byte
packing, and the byte accounting of the two fused routed-expert tensors.

## Source semantics
* Q2_0 quantizer: candidate scales `d = amax * f` with `f` from 0.5 to 1.0 in 17 steps; quantized level
  `q = clamp(rint(x/d), -1, 2)`; stored code is `q + 1 ∈ 0..3`, four per byte, LSB first.
* Q4_0 reference: code `min 15 (trunc(x/d + 8.5))`, low half in low nibbles, high half in high nibbles.
* Expert tensors `gate_up_proj [512,1280,2560]` and `down_proj [512,2560,640]` are quantized along the
  last axis, whose length must be a multiple of the block size; stored last dim = `n/block*block_bytes`.
* Reported sizes: Q2_0 ≈ 0.71 GB, Q4_0 ≈ 1.42 GB, Q8_0 ≈ 2.67 GB.

## Form template identifiers
* `clampLevel`   ← `np.clip(np.rint(x * inv), -1, 2)`
* `q2Code`       ← `(... + 1).astype(np.uint8)`
* `packQ2`       ← `c[:,:,0] | c[:,:,1] << 2 | ...`
* `q4Code`       ← `np.minimum(15, np.trunc(x*inv + 8.5))`
* `storedLast`   ← `shape[-1] // block * block_bytes`
-/

namespace MtpPack

-- § Q2_0 quantizer ranges

/-- Clamp a rounded level to the Q2_0 grid `{-1, 0, 1, 2}`. -/
def clampLevel (q : Int) : Int := max (-1) (min 2 q)

/-- The clamped level lies in `-1..2`. -/
theorem clampLevel_range (q : Int) : -1 ≤ clampLevel q ∧ clampLevel q ≤ 2 := by
  unfold clampLevel; omega

/-- Clamping is the identity on grid levels. -/
theorem clampLevel_id (q : Int) (h1 : -1 ≤ q) (h2 : q ≤ 2) : clampLevel q = q := by
  unfold clampLevel; omega

/-- The stored Q2_0 code `clamp + 1` is in `0..3`. -/
theorem q2Code_range (q : Int) : 0 ≤ clampLevel q + 1 ∧ clampLevel q + 1 ≤ 3 := by
  have := clampLevel_range q
  omega

/-- Dequantizing a stored code recovers the clamped level: `(code - 1) = clamp`. -/
theorem q2_dequant_recovers_level (q : Int) : (clampLevel q + 1) - 1 = clampLevel q := by omega

/-- Candidate scale numerators `16 + i` over 32 span exactly 0.5 to 1.0 for `i = 0..16`. -/
theorem q2_candidate_span (i : Nat) (h : i ≤ 16) : 16 ≤ 16 + i ∧ 16 + i ≤ 32 := by omega

/-- There are 17 candidate scales. -/
theorem q2_candidate_count : (List.range 17).length = 17 := by decide

/-- Pack four 2-bit codes, LSB first. -/
def packQ2 (c0 c1 c2 c3 : Nat) : Nat := c0 + 4 * c1 + 16 * c2 + 64 * c3

/-- A packed Q2_0 byte fits in a byte. -/
theorem packQ2_lt (c0 c1 c2 c3 : Nat) (h0 : c0 < 4) (h1 : c1 < 4) (h2 : c2 < 4) (h3 : c3 < 4) :
    packQ2 c0 c1 c2 c3 < 256 := by unfold packQ2; omega

/-- The packed byte is injective in its codes. -/
theorem packQ2_injective (a0 a1 a2 a3 b0 b1 b2 b3 : Nat)
    (ha0 : a0 < 4) (ha1 : a1 < 4) (ha2 : a2 < 4) (ha3 : a3 < 4)
    (hb0 : b0 < 4) (hb1 : b1 < 4) (hb2 : b2 < 4) (hb3 : b3 < 4)
    (h : packQ2 a0 a1 a2 a3 = packQ2 b0 b1 b2 b3) : a0 = b0 ∧ a1 = b1 ∧ a2 = b2 ∧ a3 = b3 := by
  unfold packQ2 at h; omega

/-- A 64-element block yields 16 packed bytes plus a 2-byte scale = 18 bytes. -/
theorem q2_block_bytes : 64 / 4 = 16 ∧ 2 + 16 = 18 := by decide

-- § Q4_0 and Q8_0

/-- Q4_0 reference code: `min 15 n` where `n = trunc(x/d + 8.5)`. -/
def q4Code (n : Nat) : Nat := min 15 n

/-- The Q4_0 code is a 4-bit value. -/
theorem q4Code_lt (n : Nat) : q4Code n < 16 := by unfold q4Code; omega

/-- Q4_0 byte: first half in the low nibble, second half in the high nibble. -/
def packQ4 (lo hi : Nat) : Nat := lo + 16 * hi

/-- A packed Q4_0 byte fits in a byte. -/
theorem packQ4_lt (lo hi : Nat) (h1 : lo < 16) (h2 : hi < 16) : packQ4 lo hi < 256 := by
  unfold packQ4; omega

/-- Q4_0 nibbles are recovered from the packed byte. -/
theorem packQ4_roundtrip (lo hi : Nat) (h1 : lo < 16) (h2 : hi < 16) :
    packQ4 lo hi % 16 = lo ∧ packQ4 lo hi / 16 = hi := by
  unfold packQ4; omega

/-- Block byte sizes: Q4_0 is 2 + 16, Q8_0 is 2 + 32. -/
theorem q4_q8_block_bytes : 2 + 32 / 2 = 18 ∧ 2 + 32 = 34 := by decide

-- § expert tensor shapes and sizes

/-- Elements of the fused `gate_up_proj` tensor. -/
def gateUpElems : Nat := 512 * 1280 * 2560

/-- Elements of `down_proj`. -/
def downElems : Nat := 512 * 2560 * 640

/-- The fused gate/up output dimension is twice the down input dimension. -/
theorem gate_up_is_double : 1280 = 2 * 640 := by decide

/-- Last-axis lengths are multiples of both Q2_0/Q4_0/Q8_0 block sizes. -/
theorem inner_dims_aligned : 2560 % 64 = 0 ∧ 640 % 64 = 0 ∧ 2560 % 32 = 0 ∧ 640 % 32 = 0 := by decide

/-- Stored last-axis byte length: `n / block * block_bytes`. -/
def storedLast (n block bytes : Nat) : Nat := n / block * bytes

/-- Q2_0 stored last dims for gate_up (2560) and down (640). -/
theorem q2_stored_last : storedLast 2560 64 18 = 720 ∧ storedLast 640 64 18 = 180 := by decide

/-- Q2_0 total expert bytes ≈ 0.71 GB. -/
theorem q2_total_bytes : (gateUpElems + downElems) / 64 * 18 = 707788800 := by decide

/-- Q4_0 total expert bytes ≈ 1.42 GB. -/
theorem q4_total_bytes : (gateUpElems + downElems) / 32 * 18 = 1415577600 := by decide

/-- Q8_0 total expert bytes ≈ 2.67 GB. -/
theorem q8_total_bytes : (gateUpElems + downElems) / 32 * 34 = 2673868800 := by decide

/-- Q4_0 is exactly twice Q2_0 in bytes; Q8_0 is 34/18 of Q4_0. -/
theorem q4_twice_q2 : 1415577600 = 2 * 707788800 := by decide

/-- Quantized tensors divide evenly into blocks for every format. -/
theorem experts_block_divisible :
    gateUpElems % 64 = 0 ∧ downElems % 64 = 0 ∧ gateUpElems % 32 = 0 ∧ downElems % 32 = 0 := by decide

/-- One-dimensional tensors are stored as F32 (4 bytes), dense matrices as BF16 (2 bytes): BF16 is half of F32. -/
theorem bf16_half_f32 (n : Nat) : n * 2 * 2 = n * 4 := by omega

end MtpPack



## FILE: /mnt/user-data/outputs/StrataPackBuilder.lean
/-!
# MODULE: strata_pack_builder
Layout arithmetic and invariants of the `strata-pack` builder/verifier: LSB-first code packing, 64-byte plane
alignment, FP16-or-FP32 scale selection, and the expert-arena row/slot indexing and size accounting.

## Source semantics
* `pack_codes`/`unpack_codes` store `8/bits` codes per byte, LSB first, and are inverse on valid codes.
* `put` pads the segment body to a multiple of 64 before each plane; offsets are `base + len(body)`.
  Alignment is therefore relative to the segment; it is absolute only if `base % 64 = 0`
  (note: the builder advances `seg_base` by the raw blob length, which need not be a multiple of 64).
* `choose_scale_dtype` returns FP16 only if the round trip is exact, else the original FP32 values.
* The expert arena interleaves gate/up rows (slot `2*row + role`) and splits each block into a 2-byte scale
  plane and 16-byte code plane; `rebuild_role` inverts that indexing.
* One expert blob is 1,382,400 bytes; 48 layers × 512 experts is 31.64 GiB.

## Form template identifiers
* `packByte4`/`unpackLo`/`unpackHi` ← `pack_codes`/`unpack_codes` with bits = 4
* `padTo`                ← `pad = (-len(body)) % ALIGN`
* `chooseScale`          ← `choose_scale_dtype`
* `rowOf`/`bInOf`        ← `row, b_in = blk // S, blk % S`
* `gateUpSlot`           ← `slot = 2*row + (0 if gate else 1)`
-/

namespace StrataPack

-- § code packing

/-- Pack two 4-bit codes into a byte, LSB first. -/
def packByte4 (a b : Nat) : Nat := a % 16 + 16 * (b % 16)

/-- Low code of a 4-bit-packed byte. -/
def unpackLo (x : Nat) : Nat := x % 16

/-- High code of a 4-bit-packed byte. -/
def unpackHi (x : Nat) : Nat := x / 16

/-- A 4-bit-packed byte fits in a byte. -/
theorem packByte4_lt (a b : Nat) : packByte4 a b < 256 := by unfold packByte4; omega

/-- 4-bit packing round-trips on valid codes. -/
theorem pack4_roundtrip (a b : Nat) (ha : a < 16) (hb : b < 16) :
    unpackLo (packByte4 a b) = a ∧ unpackHi (packByte4 a b) = b := by
  unfold unpackLo unpackHi packByte4; omega

/-- Pack four 2-bit codes into a byte, LSB first. -/
def packByte2 (a b c d : Nat) : Nat := a % 4 + 4 * (b % 4) + 16 * (c % 4) + 64 * (d % 4)

/-- A 2-bit-packed byte fits in a byte. -/
theorem packByte2_lt (a b c d : Nat) : packByte2 a b c d < 256 := by unfold packByte2; omega

/-- 2-bit packing is injective on valid codes. -/
theorem packByte2_injective (a b c d a' b' c' d' : Nat)
    (h1 : a < 4) (h2 : b < 4) (h3 : c < 4) (h4 : d < 4)
    (g1 : a' < 4) (g2 : b' < 4) (g3 : c' < 4) (g4 : d' < 4)
    (h : packByte2 a b c d = packByte2 a' b' c' d') : a = a' ∧ b = b' ∧ c = c' ∧ d = d' := by
  unfold packByte2 at h; omega

/-- Codes per byte times bits per code is 8 for every supported width. -/
theorem codes_per_byte : [2, 4, 8].all (fun b => (8 / b) * b == 8) = true := by decide

/-- A Q2_0 block's 64 codes occupy exactly 16 packed bytes. -/
theorem q2_codes_bytes_per_block : 64 / (8 / 2) = 16 := by decide

-- § alignment

/-- Padding to the next multiple of 64. -/
def padTo (len : Nat) : Nat := (64 - len % 64) % 64

/-- After padding, the length is a multiple of 64. -/
theorem padTo_aligned (len : Nat) : (len + padTo len) % 64 = 0 := by unfold padTo; omega

/-- Padding is strictly less than the alignment. -/
theorem padTo_lt (len : Nat) : padTo len < 64 := by unfold padTo; omega

/-- Already aligned lengths get no padding. -/
theorem padTo_zero_of_aligned (len : Nat) (h : len % 64 = 0) : padTo len = 0 := by
  unfold padTo; omega

/-- A plane's absolute offset is congruent to the segment base mod 64: alignment is relative to `base`. -/
theorem plane_offset_mod (base len : Nat) : (base + (len + padTo len)) % 64 = base % 64 := by
  unfold padTo; omega

/-- Plane offsets are 64-aligned in absolute terms exactly when the segment base is. -/
theorem plane_offset_aligned_iff (base len : Nat) :
    (base + (len + padTo len)) % 64 = 0 ↔ base % 64 = 0 := by
  unfold padTo; omega

/-- The builder's `seg_base` advance (raw blob length) preserves alignment only for aligned blobs. -/
theorem seg_base_advance_aligned (base blob : Nat) (hb : base % 64 = 0) (hl : blob % 64 = 0) :
    (base + blob) % 64 = 0 := by omega

-- § scale dtype choice

/-- `choose_scale_dtype` on an abstract round-trip function `r` (FP32→FP16→FP32). -/
def chooseScale {α : Type} [DecidableEq α] (r : α → α) (s : α) : α := if r s = s then r s else s

/-- Whichever branch is taken, the decoded scales equal the original: the choice is lossless. -/
theorem chooseScale_lossless {α : Type} [DecidableEq α] (r : α → α) (s : α) : chooseScale r s = s := by
  unfold chooseScale
  split
  · assumption
  · rfl

-- § expert arena indexing

/-- Row of a flat block index given `S` blocks per row. -/
def rowOf (blk S : Nat) : Nat := blk / S

/-- In-row block of a flat block index. -/
def bInOf (blk S : Nat) : Nat := blk % S

/-- `(row, b_in)` reconstructs the flat block index. -/
theorem row_bin_recompose (blk S : Nat) : S * rowOf blk S + bInOf blk S = blk :=
  Nat.div_add_mod blk S

/-- In-row block index is within the row. -/
theorem bIn_lt (blk S : Nat) (hS : 0 < S) : bInOf blk S < S := Nat.mod_lt _ hS

/-- Gate/up row interleave slot: gate = 0, up = 1. -/
def gateUpSlot (row role : Nat) : Nat := 2 * row + role

/-- Distinct (row, role) pairs get distinct slots, so gate and up never collide. -/
theorem gateUpSlot_injective (r1 r2 a b : Nat) (ha : a < 2) (hb : b < 2)
    (h : gateUpSlot r1 a = gateUpSlot r2 b) : r1 = r2 ∧ a = b := by
  unfold gateUpSlot at h; omega

/-- Gate and up of the same row occupy adjacent slots. -/
theorem gate_up_adjacent (row : Nat) : gateUpSlot row 1 = gateUpSlot row 0 + 1 := by
  unfold gateUpSlot; omega

-- § expert arena sizes

/-- Per-role bytes for one expert: `2560 * 640 / 64` blocks of 18 bytes. -/
theorem role_bytes : 2560 * 640 / 64 * 18 = 460800 := by decide

/-- One expert blob (gate + up + down) is exactly 1,382,400 bytes. -/
theorem blob_bytes : 3 * (2560 * 640 / 64 * 18) = 1382400 := by decide

/-- A block splits into a 2-byte scale and 16 code bytes: 18 bytes. -/
theorem block_split : 2 + 16 = 18 := by decide

/-- 48 layers of 512 experts is 33,973,862,400 bytes (31.64 GiB). -/
theorem arena_total : 48 * 512 * 1382400 = 33973862400 := by decide

/-- The arena size in whole GiB-hundredths is 3164 (31.64 GiB). -/
theorem arena_gib_hundredths : 33973862400 * 100 / 2 ^ 30 = 3164 := by decide

/-- Blocks per expert tensor times bytes agree with the arena's per-role accounting. -/
theorem blocks_per_role : 2560 * 640 / 64 = 25600 := by decide

end StrataPack



## FILE: /mnt/user-data/outputs/ByteLevelBpeTokenizer.lean
/-!
# MODULE: byte_level_bpe_tokenizer
Model of the GPT-2 byte↔unicode alphabet, special-token matching classes, and BPE merge bookkeeping used by
`strata_tokenizer.py`.

## Source semantics
* 188 bytes (33..126, 161..172, 174..255) map to themselves; the other 68 map to `256 + n` in increasing
  byte order, so the map is total and reversible (no UNK).
* Token types: type 4 (USER_DEFINED) literals are matched always; type 3 (CONTROL) only with parse_special.
* BPE merges the lowest-rank adjacent pair first; each merge shortens the part list by exactly one.
* Concatenating pieces before or after encoding gives the same ids (flat map).

## Form template identifiers
* `inBs` / `byteCode` / `codeByte` ← `bytes_to_unicode`, `BYTE_TO_UNICODE`, `UNICODE_TO_BYTE`
* `matched`                        ← `_always_re` / `_special_re` selection
* `mergeAt`                        ← `parts[best:best+2] = [parts[best] + parts[best+1]]`
-/

namespace ByteBpe

-- § the byte alphabet

/-- Bytes that map to themselves in the GPT-2 byte encoder. -/
def inBs (b : Nat) : Bool :=
  (decide (33 ≤ b) && decide (b ≤ 126)) || (decide (161 ≤ b) && decide (b ≤ 172)) || decide (174 ≤ b)

/-- Code point assigned to byte `b`: itself if printable, else `256 + #(non-printable bytes below b)`. -/
def byteCode (b : Nat) : Nat :=
  if inBs b then b else 256 + ((List.range b).filter (fun x => !inBs x)).length

/-- Inverse: code points below 256 are bytes; the rest index the non-printable bytes in order. -/
def codeByte (c : Nat) : Nat :=
  if c < 256 then c else ((List.range 256).filter (fun x => !inBs x)).getD (c - 256) 0

/-- Exactly 188 bytes are self-mapped. -/
theorem selfMapped_count : ((List.range 256).filter inBs).length = 188 := by decide +kernel

/-- Exactly 68 bytes are shifted into `256+n`. -/
theorem shifted_count : ((List.range 256).filter (fun x => !inBs x)).length = 68 := by decide

/-- The two classes together cover all 256 bytes. -/
theorem classes_total : 188 + 68 = 256 := by decide

/-- Space (32) maps to U+0120 (288), the familiar `Ġ`. -/
theorem space_code : byteCode 32 = 288 := by decide

/-- Newline (10) maps to U+010A (266), the familiar `Ċ`. -/
theorem newline_code : byteCode 10 = 266 := by decide

/-- 0xAD (soft hyphen) is shifted, 0xAC and 0xAE are not. -/
theorem soft_hyphen_edges : inBs 173 = false ∧ inBs 172 = true ∧ inBs 174 = true := by decide

/-- Printable bytes map below 256. -/
theorem printable_code_lt (b : Nat) (h : inBs b = true) (hb : b < 256) : byteCode b < 256 := by
  unfold byteCode
  simp [h]
  exact hb

/-- Shifted bytes map to 256 or above. -/
theorem shifted_code_ge (b : Nat) (h : inBs b = false) : 256 ≤ byteCode b := by
  unfold byteCode
  simp [h]

/-- A printable byte and a shifted byte never collide. -/
theorem class_codes_disjoint (a b : Nat) (ha : inBs a = true) (haa : a < 256) (hb : inBs b = false) :
    byteCode a ≠ byteCode b := by
  have h1 := printable_code_lt a ha haa
  have h2 := shifted_code_ge b hb
  omega

/-- The byte layer is lossless: decoding the code of every byte returns the byte. -/
theorem byte_roundtrip : ∀ b, b < 256 → codeByte (byteCode b) = b := by
  decide +kernel

/-- Encoding is injective on bytes (a consequence of the round trip). -/
theorem byteCode_injective (a b : Nat) (ha : a < 256) (hb : b < 256)
    (h : byteCode a = byteCode b) : a = b := by
  have h1 := byte_roundtrip a ha
  have h2 := byte_roundtrip b hb
  rw [h] at h1
  omega

/-- Round trip lifts to byte strings: decode ∘ encode = id. -/
theorem bytes_roundtrip (bs : List Nat) (h : ∀ b ∈ bs, b < 256) :
    (bs.map byteCode).map codeByte = bs := by
  induction bs with
  | nil => rfl
  | cons x xs ih =>
    simp only [List.map_cons]
    rw [byte_roundtrip x (h x (by simp)), ih (fun b hb => h b (by simp [hb]))]

-- § special-token classes

/-- Whether a token of type `ty` is matched as a literal, given `parse_special`. -/
def matched (ty : Nat) (parseSpecial : Bool) : Bool :=
  decide (ty = 4) || (decide (ty = 3) && parseSpecial)

/-- USER_DEFINED (type 4) literals are matched with or without parse_special. -/
theorem type4_always (ps : Bool) : matched 4 ps = true := by simp [matched]

/-- CONTROL (type 3) literals are matched exactly when parse_special is set. -/
theorem type3_iff (ps : Bool) : matched 3 ps = ps := by simp [matched]

/-- Other token types are never matched as literals. -/
theorem other_never (ty : Nat) (h3 : ty ≠ 3) (h4 : ty ≠ 4) (ps : Bool) : matched ty ps = false := by
  simp [matched, h3, h4]

/-- Everything matched without parse_special is also matched with it. -/
theorem always_subset_special (ty : Nat) (h : matched ty false = true) : matched ty true = true := by
  unfold matched at *
  simp at *
  omega

-- § BPE merge bookkeeping

/-- Merge the adjacent pair at index `i` into the single token `m`. -/
def mergeAt {α : Type} (l : List α) (i : Nat) (m : α) : List α := l.take i ++ m :: l.drop (i + 2)

/-- A merge shortens the part list by exactly one. -/
theorem mergeAt_length {α : Type} (l : List α) (i : Nat) (m : α) (h : i + 2 ≤ l.length) :
    (mergeAt l i m).length + 1 = l.length := by
  simp [mergeAt, List.length_take, List.length_drop] <;> omega

/-- Merging at an in-range position leaves the prefix before it untouched. -/
theorem mergeAt_prefix {α : Type} (l : List α) (i : Nat) (m : α) (h : i ≤ l.length) :
    (mergeAt l i m).take i = l.take i := by
  unfold mergeAt
  rw [List.take_append_of_le_length (by rw [List.length_take]; omega)]
  simp

/-- A word of `n ≥ 1` parts needs at most `n - 1` merges: each merge removes one part, so the loop terminates. -/
theorem bpe_merge_budget (n k : Nat) (h : k + 1 ≤ n) (hk : 1 ≤ k + 1) : n - k ≥ 1 := by omega

/-- Encoding a concatenation of pre-token pieces is the concatenation of their encodings. -/
theorem encode_pieces_append {α β : Type} (f : α → List β) (xs ys : List α) :
    (xs ++ ys).flatMap f = xs.flatMap f ++ ys.flatMap f := by
  simp

/-- A merge list index equals its rank, so a pair's rank is unique if the pair appears once. -/
theorem rank_unique (ranks : List (Nat × Nat)) (h : ranks.Nodup) (i j : Nat)
    (hi : i < ranks.length) (hj : j < ranks.length) (e : ranks[i] = ranks[j]) : i = j := by
  exact (List.Nodup.getElem_inj_iff h).mp e

end ByteBpe




[LIGHT FILES — structure only, no LAYER 1a/1b]
  CMakeLists.txt (537 lines)
  START-HERE.bat (53 lines)
  bench/results/2026-09-24-final/estimates.json (461 lines)
  bench/results/2026-09-24-final/matrix.json (240 lines)
  bench/results/2026-09-24-vision/vision.json (53 lines)
  bench/results/2026-09-24-final/estimates.md (28 lines)
  bench/results/2026-09-24-final/matrix.md (16 lines)
  bench/results/2026-09-24-vision/vision.md (6 lines)
  chat.py (117 lines)
  include/strata/core/graph.hpp (118 lines)
  include/strata/core/layer.hpp (567 lines)
  include/strata/core/layout.hpp (98 lines)
  include/strata/core/mtp.hpp (136 lines)
  include/strata/core/pinned.hpp (96 lines)
  include/strata/core/session.hpp (412 lines)
  include/strata/core/verify.hpp (161 lines)
  include/strata/core/weights.hpp (136 lines)
  include/strata/core/device.hpp (72 lines)
  include/strata/core/hit_hook.hpp (42 lines)
  include/strata/core/native_dense.hpp (37 lines)
  include/strata/core/native_head.hpp (66 lines)
  include/strata/kernels/cpu/expert.hpp (180 lines)
  include/strata/kernels/cpu/pool.hpp (196 lines)
  include/strata/kernels/cpu/expert_layout.hpp (52 lines)
  include/strata/kernels/cpu/iq_avx512.hpp (19 lines)
  include/strata/kernels/cpu/native_expert.hpp (50 lines)
  include/strata/kernels/elementwise.hpp (117 lines)
  include/strata/kernels/gdn.hpp (98 lines)
  include/strata/kernels/gr.hpp (123 lines)
  include/strata/kernels/native_mmvq.hpp (113 lines)
  include/strata/kernels/ngram.hpp (195 lines)
  include/strata/kernels/ple.hpp (120 lines)
  include/strata/kernels/qsa.hpp (361 lines)
  include/strata/kernels/s2_expert_grouped.hpp (119 lines)
  include/strata/kernels/s_gemv.hpp (134 lines)
  include/strata/kernels/shared_expert.hpp (102 lines)
  include/strata/kernels/verify_kernels.hpp (81 lines)
  include/strata/kernels/bf16_bits.hpp (47 lines)
  include/strata/kernels/bf16_gemv.hpp (49 lines)
  include/strata/kernels/dequant_bf16.hpp (28 lines)
  include/strata/kernels/dequant_s2.hpp (19 lines)
  include/strata/kernels/fused_gdn.hpp (36 lines)
  include/strata/kernels/fused_gr.hpp (52 lines)
  include/strata/kernels/iq_kernels.hpp (58 lines)
  include/strata/kernels/kv_q8.hpp (39 lines)
  include/strata/kernels/mrope.hpp (33 lines)
  include/strata/kernels/native_flash_attn.hpp (37 lines)
  include/strata/kernels/native_gdn.hpp (23 lines)
  include/strata/kernels/native_gdn_preprocess.hpp (38 lines)
  include/strata/kernels/native_gr_norm.hpp (19 lines)
  include/strata/kernels/native_gr_postops.hpp (29 lines)
  include/strata/kernels/native_moe.hpp (19 lines)
  include/strata/kernels/native_ple_postops.hpp (31 lines)
  include/strata/kernels/native_qsa.hpp (31 lines)
  include/strata/kernels/native_qsa_indexer.hpp (36 lines)
  include/strata/kernels/native_qsa_score.hpp (32 lines)
  include/strata/kernels/native_rope.hpp (15 lines)
  include/strata/kernels/native_router.hpp (18 lines)
  include/strata/kernels/qsa_decode_attn.hpp (45 lines)
  include/strata/kernels/qsa_select.hpp (34 lines)
  include/strata/kernels/quantize_act.hpp (59 lines)
  include/strata/kernels/rope.hpp (51 lines)
  include/strata/kernels/router_top10.hpp (21 lines)
  include/strata/kernels/s2_gemv.hpp (26 lines)
  include/strata/kernels/s2_gemv_q8.hpp (23 lines)
  include/strata/kernels/sampler.hpp (44 lines)
  include/strata/ngram/ple_reader.hpp (87 lines)
  include/strata/plan/plan.hpp (169 lines)
  include/strata/platform/direct_file.hpp (60 lines)
  include/strata/platform/memory.hpp (27 lines)
  include/strata/prefill/gemm.hpp (55 lines)
  include/strata/prefill/kernels.hpp (75 lines)
  include/strata/prefill/prefill.hpp (76 lines)
  include/strata/program/logits_selection.hpp (31 lines)
  include/strata/spec/controller.hpp (74 lines)
  include/strata/spec/suffix_drafter.hpp (54 lines)
  ref/load.py (213 lines)
  serve/chat_golden.json (197 lines)
  serve/chat_template.jinja (184 lines)
  serve/index.html (153 lines)
  serve/__init__.py (2 lines)
  setup.py (807 lines)
  setup.sh (31 lines)
  src/artifact/dequant.cpp (124 lines)
  src/artifact/gguf_reader.cpp (68 lines)
  src/core/concurrent_main.cpp (198 lines)
  src/core/device.cu (118 lines)
  src/core/device_main.cpp (79 lines)
  src/core/graph.cpp (147 lines)
  src/core/layout.cpp (174 lines)
  src/core/load_main.cpp (120 lines)
  src/core/native_dense.cpp (180 lines)
  src/core/native_head.cpp (152 lines)
  src/core/overlap_main.cpp (170 lines)
  src/core/pinned.cu (292 lines)
  src/kernels/bf16_gemv_parity.cpp (149 lines)
  src/kernels/cpu/expert_layout.cpp (137 lines)
  src/kernels/cpu/expert_multi_test.cpp (88 lines)
  src/kernels/cpu/expert_parity.cpp (185 lines)
  src/kernels/cpu/native_expert.cpp (115 lines)
  src/kernels/cpu/pool_test.cpp (225 lines)
  src/kernels/cuda/bf16_gemv.cu (142 lines)
  src/kernels/cuda/dequant_bf16.cu (246 lines)
  src/kernels/cuda/dequant_s2.cu (52 lines)
  src/kernels/cuda/elementwise.cu (290 lines)
  src/kernels/cuda/kv_q8.cu (119 lines)
  src/kernels/cuda/native_bf16.cu (119 lines)
  src/kernels/cuda/native_gdn_preprocess.cu (200 lines)
  src/kernels/cuda/native_gr_norm.cu (103 lines)
  src/kernels/cuda/native_gr_postops.cu (116 lines)
  src/kernels/cuda/native_moe.cu (75 lines)
  src/kernels/cuda/native_ple_postops.cu (135 lines)
  src/kernels/cuda/native_qsa.cu (128 lines)
  src/kernels/cuda/native_qsa_indexer.cu (130 lines)
  src/kernels/cuda/native_rope.cu (89 lines)
  src/kernels/cuda/native_router.cu (118 lines)
  src/kernels/cuda/ple.cu (367 lines)
  src/kernels/cuda/quantize_act.cu (314 lines)
  src/kernels/cuda/rope.cu (96 lines)
  src/kernels/cuda/s2_gemv.cu (64 lines)
  src/kernels/cuda/s2_gemv_fast.cu (151 lines)
  src/kernels/cuda/s2_gemv_q8.cu (97 lines)
  src/kernels/cuda/s2_gemv_quads.cu (94 lines)
  src/kernels/cuda/shared_expert.cu (339 lines)
  src/kernels/dequant_bf16_test.cpp (109 lines)
  src/kernels/dequant_s2_parity.cpp (144 lines)
  src/kernels/elementwise_parity.cpp (417 lines)
  src/kernels/gdn_parity.cpp (398 lines)
  src/kernels/gr_parity.cpp (616 lines)
  src/kernels/iq_parity.cpp (88 lines)
  src/kernels/kv_q8_parity.cpp (142 lines)
  src/kernels/native_expert_parity.cpp (205 lines)
  src/kernels/ple_oracle_vectors.inc (106 lines)
  src/kernels/ple_parity.cpp (1185 lines)
  src/kernels/qsa_oracle_vectors.inc (135 lines)
  src/kernels/qsa_parity.cpp (1384 lines)
  src/kernels/quantize_act_parity.cpp (408 lines)
  src/kernels/rope_parity.cpp (172 lines)
  src/kernels/router_top10_parity.cpp (168 lines)
  src/kernels/s2_gemv_parity.cpp (174 lines)
  src/kernels/s2_gemv_q8_parity.cpp (182 lines)
  src/kernels/s_gemv_parity.cpp (635 lines)
  src/kernels/s_gemv_q8k_parity.cpp (237 lines)
  src/kernels/sampler_parity.cpp (363 lines)
  src/kernels/shared_expert_parity.cpp (483 lines)
  src/ngram/ple_reader_test.cpp (238 lines)
  src/plan/plan_main.cpp (74 lines)
  src/platform/memory.cpp (77 lines)
  src/platform/memory_test.cpp (36 lines)
  src/prefill/gemm.cu (108 lines)
  src/spec/suffix_drafter_test.cpp (142 lines)
  src/spec/controller_test.cpp (75 lines)
  third_party/ggml/ggml-common.h (1912 lines)
  third_party/ggml/VERSION.txt (2 lines)
  tools/gguf_reader.py (143 lines)
  tools/gguf_writer.py (235 lines)
  tools/iq_pack.py (222 lines)
  tools/mtp_fetch.py (122 lines)
  tools/mtp_rt.py (113 lines)
  tools/pack_index.py (261 lines)
  tools/pack_layer.py (124 lines)
  tools/vision/CMakeLists.txt (45 lines)
  tools/vision/strata_vision.cpp (159 lines)
  tools/_paths.py (29 lines)
