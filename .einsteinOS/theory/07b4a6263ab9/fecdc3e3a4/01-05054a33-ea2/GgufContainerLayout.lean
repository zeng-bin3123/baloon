/-!
# MODULE: gguf_container_layout
SOURCES: include/strata/artifact/gguf_reader.hpp, tools/gguf_reader.py, tools/gguf_writer.py
The GGUF v3 container as Strata reads and writes it: the magic and fixed header, the metadata value
type table, block geometry per ggml type, data-section alignment, the writer's aligned tensor
offsets, and the `qwen4exp` architecture guard.

## Source semantics
* The file begins with magic `0x46554747` ("GGUF" little-endian), `u32 version == 3`,
  `u64 n_tensors`, `u64 n_kv`: a 24-byte fixed prefix.
* Metadata value types 0..12 have fixed scalar sizes (u8..f64); 8 is string, 9 is array.
* `block_geometry(t)` gives (elements, bytes) per block; a tensor's byte size is
  `elements / block_elems * block_bytes` when divisible (`expected_bytes`).
* The data section starts at `align_up(pos, alignment)` with default alignment 32, and every
  writer offset is padded up to the alignment.
* `check_architecture` refuses unless `general.architecture == "qwen4exp"` and six hyperparameters
  equal the compiled-in constants (48, 2560, 512, 10, 24, 2).

## Form template identifiers
`GGUF_MAGIC` → `ggufMagic`, `(pos + a - 1) / a * a` → `alignUp`, `block_geometry` →
`blockGeometry`, `expected_bytes` → `expectedBytes`, writer offset loop → `writerOffsets`,
`Qwen4ExpGuard` → `Guard`, `check_architecture` → `checkArch`.
-/

namespace Strata.Gguf

-- § magic and header

/-- The GGUF magic number. -/
def ggufMagic : Nat := 0x46554747

/-- The magic is the ASCII bytes `G`,`G`,`U`,`F` read as a little-endian u32. -/
theorem magic_is_ascii :
    ggufMagic = 'G'.toNat + 'G'.toNat * 256 + 'U'.toNat * 65536 + 'F'.toNat * 16777216 := by decide

/-- The fixed header prefix (magic, version, tensor count, kv count) is 24 bytes; the Python reader's
`"<IQQ"` after the magic consumes the remaining 20. -/
theorem header_prefix_bytes : 4 + 4 + 8 + 8 = 24 ∧ 4 + 8 + 8 = 20 := by decide

-- § metadata value types

/-- Scalar byte size of GGUF metadata type `t`, `none` for string/array/unknown. -/
def metaSize : Nat → Option Nat
  | 0 => some 1 | 1 => some 1 | 2 => some 2 | 3 => some 2 | 4 => some 4 | 5 => some 4
  | 6 => some 4 | 7 => some 1 | 10 => some 8 | 11 => some 8 | 12 => some 8
  | _ => none

/-- Exactly the eleven scalar types have a fixed size; string (8) and array (9) do not. -/
theorem metaSize_scalar_ids :
    (List.range 13).filter (fun t => (metaSize t).isSome) = [0, 1, 2, 3, 4, 5, 6, 7, 10, 11, 12] := by
  decide

/-- Every scalar is at most 8 bytes, the size of a `u64`/`i64`/`f64`. -/
theorem metaSize_le_8 : ∀ t, t < 13 → ∀ n, metaSize t = some n → n ≤ 8 := by decide

-- § block geometry

/-- `block_geometry` from the C++ reader: (elements per block, bytes per block). -/
def blockGeometry : Nat → Option (Nat × Nat)
  | 0 => some (1, 4) | 1 => some (1, 2) | 30 => some (1, 2)
  | 2 => some (32, 18) | 3 => some (32, 20) | 6 => some (32, 22) | 7 => some (32, 24)
  | 8 => some (32, 34) | 9 => some (32, 36)
  | 10 => some (256, 84) | 11 => some (256, 110) | 12 => some (256, 144) | 13 => some (256, 176)
  | 14 => some (256, 210) | 16 => some (256, 66) | 17 => some (256, 74) | 18 => some (256, 98)
  | 20 => some (32, 18) | 21 => some (256, 110) | 22 => some (256, 82) | 23 => some (256, 136)
  | 29 => some (256, 56) | 42 => some (64, 18)
  | _ => none

/-- Type 42 (Q2_0) is 64 elements in 18 bytes, the artifact-proven contract. -/
theorem q2_0_geometry : blockGeometry 42 = some (64, 18) := rfl

/-- Every block size the C++ table knows is a power-of-two element count in `{1, 32, 64, 256}`. -/
theorem blockGeometry_elems : ∀ t, t < 43 →
    (blockGeometry t).all (fun p => p.1 == 1 || p.1 == 32 || p.1 == 64 || p.1 == 256) = true := by
  decide

/-- The C++ table agrees with the Python `BLOCK_GEOMETRY` on the types the model uses
(Q2_0, Q3_K, Q4_0, Q4_K, Q5_0, Q5_K, Q6_K, Q8_0, IQ4_NL, IQ4_XS, F32, F16, BF16). -/
theorem blockGeometry_matches_python :
    [blockGeometry 42, blockGeometry 11, blockGeometry 2, blockGeometry 12, blockGeometry 6,
     blockGeometry 13, blockGeometry 14, blockGeometry 8, blockGeometry 20, blockGeometry 23,
     blockGeometry 0, blockGeometry 1, blockGeometry 30] =
    [some (64, 18), some (256, 110), some (32, 18), some (256, 144), some (32, 22),
     some (256, 176), some (256, 210), some (32, 34), some (32, 18), some (256, 136),
     some (1, 4), some (1, 2), some (1, 2)] := by decide

/-- `expected_bytes`: `none` unless the element count is a whole number of blocks. -/
def expectedBytes (elems blockElems blockBytes : Nat) : Option Nat :=
  if elems % blockElems = 0 then some (elems / blockElems * blockBytes) else none

/-- The fused expert tensor `[2560, 640, 512]` of Q2_0 is 512 x 460,800 bytes. -/
theorem expert_tensor_bytes : expectedBytes (2560 * 640 * 512) 64 18 = some (512 * 460800) := by decide

/-- Byte size is additive over whole blocks: splitting a tensor at a block boundary splits its
bytes, which is what makes per-expert slicing a strided copy. -/
theorem expectedBytes_add (a b be bb : Nat) (hbe : 0 < be) :
    (a * be + b * be) / be * bb = a * bb + b * bb := by
  rw [← Nat.add_mul, Nat.mul_div_cancel _ hbe, Nat.add_mul]

-- § alignment

/-- `(n + a - 1) / a * a`: round `n` up to a multiple of `a`. -/
def alignUp (a n : Nat) : Nat := (n + a - 1) / a * a

/-- The aligned position is a multiple of the alignment. -/
theorem alignUp_dvd (a n : Nat) : alignUp a n % a = 0 := by
  unfold alignUp; exact Nat.mul_mod_left _ _

/-- Aligning never moves backwards. -/
theorem le_alignUp {a : Nat} (ha : 0 < a) (n : Nat) : n ≤ alignUp a n := by
  have h := @Nat.lt_div_mul_add (n + a - 1) a ha
  unfold alignUp; omega

/-- Aligning moves forward by less than one alignment unit. -/
theorem alignUp_lt {a : Nat} (ha : 0 < a) (n : Nat) : alignUp a n < n + a := by
  have h := Nat.div_mul_le_self (n + a - 1) a
  unfold alignUp; omega

/-- An already-aligned position is a fixed point. -/
theorem alignUp_of_dvd {a k : Nat} (ha : 0 < a) : alignUp a (k * a) = k * a := by
  have h1 := le_alignUp ha (k * a)
  have h2 := alignUp_lt ha (k * a)
  have h3 := alignUp_dvd a (k * a)
  -- a multiple of `a` in `[k*a, k*a + a)` is `k*a`
  have hm : alignUp a (k * a) = alignUp a (k * a) / a * a := by
    have := Nat.div_add_mod (alignUp a (k * a)) a
    rw [h3, Nat.add_zero] at this
    rw [Nat.mul_comm] at this; exact this.symm
  have hq : alignUp a (k * a) / a = k := by
    have lo : k * a ≤ alignUp a (k * a) / a * a := by rw [← hm]; exact h1
    have hi : alignUp a (k * a) / a * a < (k + 1) * a := by rw [← hm, Nat.add_mul, Nat.one_mul]; exact h2
    have l1 : k ≤ alignUp a (k * a) / a := Nat.le_of_mul_le_mul_right lo ha
    have l2 : alignUp a (k * a) / a < k + 1 := Nat.lt_of_mul_lt_mul_right hi
    omega
  rw [hm, hq]

/-- The default alignment is 32 and the data section of an empty header begins at 32. -/
theorem default_alignment : alignUp 32 24 = 32 := by decide

-- § writer offsets

/-- The writer's tensor offsets: each next offset is the previous end, aligned up. -/
def writerOffsets (a : Nat) : Nat → List Nat → List Nat
  | _, [] => []
  | off, len :: rest => off :: writerOffsets a (alignUp a (off + len)) rest

/-- Starting from an aligned offset, every tensor offset the writer emits is aligned. -/
theorem writerOffsets_aligned (a : Nat) :
    ∀ (lens : List Nat) (off : Nat), off % a = 0 → ∀ x ∈ writerOffsets a off lens, x % a = 0 := by
  intro lens
  induction lens with
  | nil => intro off _ x hx; simp [writerOffsets] at hx
  | cons len rest ih =>
    intro off hoff x hx
    simp only [writerOffsets, List.mem_cons] at hx
    rcases hx with h | h
    · rw [h]; exact hoff
    · exact ih _ (alignUp_dvd a _) x h

/-- Consecutive tensors do not overlap: the next offset is at least the previous end. -/
theorem writer_no_overlap {a : Nat} (ha : 0 < a) (off len : Nat) : off + len ≤ alignUp a (off + len) :=
  le_alignUp ha _

/-- The writer emits one offset per tensor. -/
theorem writerOffsets_length (a : Nat) : ∀ (lens : List Nat) (off : Nat),
    (writerOffsets a off lens).length = lens.length := by
  intro lens
  induction lens with
  | nil => intro off; rfl
  | cons len rest ih => intro off; simp [writerOffsets, ih]

-- § architecture guard

/-- The compiled-in `qwen4exp` hyperparameters. -/
structure Guard where
  blockCount : Nat := 48
  hidden : Nat := 2560
  experts : Nat := 512
  expertsUsed : Nat := 10
  headCount : Nat := 24
  headCountKv : Nat := 2

/-- Metadata view: architecture string and the six numbers (as read; `none` if missing). -/
structure ArchMeta where
  arch : Option String
  values : List (Option Nat)

/-- The required values in the order `check_architecture` tests them. -/
def Guard.required (g : Guard) : List Nat :=
  [g.blockCount, g.hidden, g.experts, g.expertsUsed, g.headCount, g.headCountKv]

/-- `check_architecture`: `true` exactly when the arch is `qwen4exp` and every key is present and
equal to the compiled-in constant. -/
def checkArch (g : Guard) (m : ArchMeta) : Bool :=
  m.arch == some "qwen4exp" && m.values == g.required.map some

/-- A file carrying exactly the compiled-in constants passes the guard. -/
theorem checkArch_accepts_default :
    checkArch {} ⟨some "qwen4exp", [some 48, some 2560, some 512, some 10, some 24, some 2]⟩ = true := by
  decide

/-- Any other architecture string is refused, whatever its hyperparameters. -/
theorem checkArch_rejects_arch (g : Guard) (s : String) (h : s ≠ "qwen4exp") (vs : List (Option Nat)) :
    checkArch g ⟨some s, vs⟩ = false := by
  unfold checkArch
  have : (some s == some "qwen4exp") = false := by
    simp [h]
  simp [this]

/-- A missing key is refused (no silent default). -/
theorem checkArch_rejects_missing :
    checkArch {} ⟨some "qwen4exp", [some 48, none, some 512, some 10, some 24, some 2]⟩ = false := by
  decide

/-- A mismatched expert count is refused. -/
theorem checkArch_rejects_mismatch :
    checkArch {} ⟨some "qwen4exp", [some 48, some 2560, some 256, some 10, some 24, some 2]⟩ = false := by
  decide

-- § reader bounds

/-- The reader accepts tensor ranks 1..4 only (`nd == 0 || nd > 4` throws). -/
def rankOk (nd : Nat) : Bool := !(nd == 0 || nd > 4)

/-- `rankOk` holds exactly for ranks 1 through 4. -/
theorem rankOk_iff (nd : Nat) : rankOk nd = true ↔ 1 ≤ nd ∧ nd ≤ 4 := by
  unfold rankOk
  simp only [Bool.not_eq_true', Bool.or_eq_false_iff, beq_eq_false_iff_ne, decide_eq_false_iff_not]
  omega

end Strata.Gguf
