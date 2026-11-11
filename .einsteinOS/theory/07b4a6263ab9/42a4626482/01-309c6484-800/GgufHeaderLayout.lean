/-!
# MODULE: gguf_header_layout
SOURCES: include/strata/artifact/gguf_reader.hpp, tools/gguf_reader.py, tools/gguf_writer.py
The GGUF v3 header contract shared by the C++ reader, the Python reader and the Python writer:
ggml type ids and block geometry, the magic, data-section alignment, per-tensor byte counts, the
writer's aligned offsets, and the `qwen4exp` architecture guard.

## Source semantics
* magic 0x46554747 is "GGUF" read little-endian; only version 3 is accepted.
* block geometry (elements, bytes) per type; Q2_0 (id 42) is (64, 18). Each byte count equals the
  sum of that block's fields.
* where the C++ table defines a geometry, the Python table agrees with it.
* known table discrepancies: Python names id 41 "Q4_0_8_8" (a duplicate of id 33; C++ says
  "Q1_0"), and the C++ name table has no entry for id 29 (IQ1_M) although its geometry table does.
* data_start = pos rounded up to the alignment (default 32); the writer pads every tensor to the
  alignment, so offsets are aligned and non-overlapping.
* expected_bytes = elements / block_elems * block_bytes when elements is a multiple of block_elems.
* the architecture guard accepts iff general.architecture = "qwen4exp" and six hyperparameters
  equal the compiled constants (48, 2560, 512, 10, 24, 2).

## Form template identifiers
block_geometry → `cppGeometry`; BLOCK_GEOMETRY ∘ GGML_TYPES → `pyGeometry`; ggml_type_name →
`cppTypeName`; GGML_TYPES → `pyTypeName`; data_start rounding / writer padding → `alignUp`;
TensorInfo.expected_bytes → `expectedBytes`; check_architecture → `checkArch`; Qwen4ExpGuard →
`guardReqs`.
-/

namespace Strata.Gguf

-- § type tables

/-- C++ `block_geometry`: (elements per block, bytes per block). -/
def cppGeometry : Nat → Option (Nat × Nat)
  | 0 => some (1, 4) | 1 => some (1, 2) | 30 => some (1, 2)
  | 2 => some (32, 18) | 3 => some (32, 20) | 6 => some (32, 22) | 7 => some (32, 24)
  | 8 => some (32, 34) | 9 => some (32, 36)
  | 10 => some (256, 84) | 11 => some (256, 110) | 12 => some (256, 144)
  | 13 => some (256, 176) | 14 => some (256, 210)
  | 16 => some (256, 66) | 17 => some (256, 74) | 18 => some (256, 98)
  | 20 => some (32, 18) | 21 => some (256, 110) | 22 => some (256, 82) | 23 => some (256, 136)
  | 29 => some (256, 56) | 42 => some (64, 18)
  | _ => none

/-- Python `BLOCK_GEOMETRY` looked up through `GGML_TYPES`, as a function of the type id. -/
def pyGeometry : Nat → Option (Nat × Nat)
  | 0 => some (1, 4) | 1 => some (1, 2) | 30 => some (1, 2) | 28 => some (1, 8)
  | 2 => some (32, 18) | 3 => some (32, 20) | 6 => some (32, 22) | 7 => some (32, 24)
  | 8 => some (32, 34) | 9 => some (32, 36)
  | 10 => some (256, 84) | 11 => some (256, 110) | 12 => some (256, 144)
  | 13 => some (256, 176) | 14 => some (256, 210) | 15 => some (256, 292)
  | 16 => some (256, 66) | 17 => some (256, 74) | 18 => some (256, 98) | 19 => some (256, 50)
  | 20 => some (32, 18) | 21 => some (256, 110) | 22 => some (256, 82) | 23 => some (256, 136)
  | 29 => some (256, 56) | 39 => some (32, 17) | 40 => some (64, 36) | 42 => some (64, 18)
  | _ => none

/-- C++ `ggml_type_name` for the ids where the tables are discussed below. -/
def cppTypeName : Nat → String
  | 29 => "?" | 33 => "?" | 41 => "Q1_0" | 42 => "Q2_0" | _ => "other"

/-- Python `GGML_TYPES` for the same ids. -/
def pyTypeName : Nat → String
  | 29 => "IQ1_M" | 33 => "Q4_0_8_8" | 41 => "Q4_0_8_8" | 42 => "Q2_0" | _ => "other"

-- § geometry facts

/-- Q2_0 is 64 elements in 18 bytes in both readers. -/
theorem q2_0_geometry : cppGeometry 42 = some (64, 18) ∧ pyGeometry 42 = some (64, 18) := by
  decide

/-- Wherever the C++ reader defines a geometry, the Python reference agrees (ids 0..63). -/
theorem cpp_geometry_agrees_with_py :
    (List.range 64).all (fun t => (cppGeometry t).isNone || cppGeometry t == pyGeometry t) = true := by
  decide

/-- Every byte count is the sum of the block's fields, as laid out in the dequantizers. -/
theorem geometry_field_sums :
    cppGeometry 2 = some (32, 2 + 16) ∧            -- Q4_0: d, qs[16]
    cppGeometry 6 = some (32, 2 + 4 + 16) ∧        -- Q5_0: d, qh[4], qs[16]
    cppGeometry 8 = some (32, 2 + 32) ∧            -- Q8_0: d, qs[32]
    cppGeometry 11 = some (256, 32 + 64 + 12 + 2) ∧ -- Q3_K: hmask, qs, scales, d
    cppGeometry 12 = some (256, 2 + 2 + 12 + 128) ∧ -- Q4_K: d, dmin, scales, qs
    cppGeometry 13 = some (256, 2 + 2 + 12 + 32 + 128) ∧ -- Q5_K: d, dmin, scales, qh, qs
    cppGeometry 14 = some (256, 128 + 64 + 16 + 2) ∧ -- Q6_K: ql, qh, scales, d
    cppGeometry 20 = some (32, 2 + 16) ∧           -- IQ4_NL: d, qs[16]
    cppGeometry 23 = some (256, 2 + 2 + 4 + 128) ∧  -- IQ4_XS: d, scales_h, scales_l, qs
    cppGeometry 42 = some (64, 2 + 16) := by        -- Q2_0: d, qs[16]
  decide

/-- Table discrepancy: Python names id 41 like id 33, and C++ names it Q1_0. -/
theorem py_name_41_duplicate : pyTypeName 41 = pyTypeName 33 ∧ cppTypeName 41 ≠ pyTypeName 41 := by
  decide

/-- Table discrepancy: C++ has a geometry for id 29 but no name for it. -/
theorem cpp_iq1m_unnamed : (cppGeometry 29).isSome = true ∧ cppTypeName 29 = "?" := by
  decide

-- § magic

/-- The magic constant is the four bytes 'G','G','U','F' read little-endian. -/
theorem magic_bytes : 0x46554747 = 0x47 + 0x47 * 256 + 0x55 * 65536 + 0x46 * 16777216 := by decide

-- § alignment

/-- Round `n` up to a multiple of `a`, as `(n + a - 1) / a * a`. -/
def alignUp (n a : Nat) : Nat := (n + a - 1) / a * a

/-- The rounded value is a multiple of the alignment. -/
theorem alignUp_dvd (n a : Nat) : (alignUp n a) % a = 0 := by
  unfold alignUp; exact Nat.mul_mod_left _ _

/-- Rounding up never moves backwards. -/
theorem alignUp_ge (n a : Nat) (ha : 0 < a) : n ≤ alignUp n a := by
  unfold alignUp
  have h := Nat.div_add_mod (n + a - 1) a
  have hm := Nat.mod_lt (n + a - 1) ha
  have : (n + a - 1) / a * a = a * ((n + a - 1) / a) := Nat.mul_comm _ _
  omega

/-- Rounding up moves forward by less than one alignment step. -/
theorem alignUp_lt (n a : Nat) (ha : 0 < a) : alignUp n a < n + a := by
  unfold alignUp
  have h := Nat.div_add_mod (n + a - 1) a
  have : (n + a - 1) / a * a = a * ((n + a - 1) / a) := Nat.mul_comm _ _
  omega

/-- The writer's padding `(-len) % align` reaches the same aligned position as `alignUp`. -/
theorem writer_pad_eq (n : Nat) : n + (32 - n % 32) % 32 = alignUp n 32 := by
  unfold alignUp; omega

/-- Writer offsets: the next tensor starts at or after the end of the previous one, aligned. -/
theorem writer_next_offset (off len : Nat) (_h : off % 32 = 0) :
    off + len ≤ alignUp (off + len) 32 ∧ alignUp (off + len) 32 % 32 = 0 :=
  ⟨alignUp_ge _ _ (by decide), alignUp_dvd _ _⟩

-- § tensor byte counts

/-- `TensorInfo.expected_bytes` when the element count is a whole number of blocks. -/
def expectedBytes (elements blockElems blockBytes : Nat) : Nat := elements / blockElems * blockBytes

/-- A Q2_0 expert role tensor of 2560 × 640 × 512 elements occupies 460,800 bytes per expert. -/
theorem q2_expert_role_bytes : expectedBytes (2560 * 640) 64 18 = 460800 := by decide

/-- Byte counts are additive over whole blocks, so per-expert slices tile a role tensor. -/
theorem expectedBytes_mul (k per be bb : Nat) (hbe : 0 < be) :
    expectedBytes (k * (per * be)) be bb = k * expectedBytes (per * be) be bb := by
  unfold expectedBytes
  rw [← Nat.mul_assoc, Nat.mul_div_cancel _ hbe, Nat.mul_div_cancel _ hbe, Nat.mul_assoc]

-- § architecture guard

/-- The six required hyperparameters of `Qwen4ExpGuard`. -/
def guardReqs : List (String × Nat) :=
  [("qwen4exp.block_count", 48), ("qwen4exp.embedding_length", 2560),
   ("qwen4exp.expert_count", 512), ("qwen4exp.expert_used_count", 10),
   ("qwen4exp.attention.head_count", 24), ("qwen4exp.attention.head_count_kv", 2)]

/-- `check_architecture`: `none` means accepted, `some msg` is the refusal (only the first
failing condition is reported, as in the C++). -/
def checkArch (arch : Option String) (meta : String → Option Nat) : Option String :=
  match arch with
  | none => some "missing general.architecture"
  | some a =>
    if a ≠ "qwen4exp" then some "wrong architecture"
    else guardReqs.foldr (fun r acc =>
      match meta r.1 with
      | none => some ("missing " ++ r.1)
      | some v => if v ≠ r.2 then some r.1 else acc) none

/-- The metadata of the target model. -/
def goodMeta (k : String) : Option Nat := (guardReqs.find? (·.1 == k)).map (·.2)

/-- The target model's metadata is accepted. -/
theorem guard_accepts_target : checkArch (some "qwen4exp") goodMeta = none := by decide

/-- A different architecture is refused before any hyperparameter is read. -/
theorem guard_refuses_other_arch (m : String → Option Nat) :
    checkArch (some "llama") m = some "wrong architecture" := by
  simp [checkArch]

/-- A missing architecture key is refused. -/
theorem guard_refuses_missing (m : String → Option Nat) :
    checkArch none m = some "missing general.architecture" := rfl

/-- One wrong hyperparameter (11 experts used instead of 10) is refused. -/
theorem guard_refuses_wrong_value :
    checkArch (some "qwen4exp")
      (fun k => if k = "qwen4exp.expert_used_count" then some 11 else goodMeta k) ≠ none := by
  decide

end Strata.Gguf
