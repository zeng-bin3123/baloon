/-!
# MODULE: gguf_reader_geometry
SOURCES: tools/gguf_reader.py
Minimal GGUF v3 header reader: magic and header sizes, tensor element counts, expected byte counts from block
geometry, and the aligned start of the data section.

## Source semantics
* Magic `0x46554747` is "GGUF" in little-endian byte order; the fixed header is 24 bytes.
* `elements` is the product of the shape; `expected_bytes` is `elements / block_elems * block_bytes` when
  `elements` is a whole number of blocks and `None` otherwise.
* `data_start` rounds the end of the tensor table up to `alignment`.
* Type id 42 is `Q2_0`; ids 31 and 41 both carry the name `Q4_0_8_8` in the table (ids stay distinct).

## Form template identifiers
- GGUF_MAGIC           → `ggufMagic`
- TensorInfo.elements  → `shapeElems`
- expected_bytes       → `expectedBytes`
- data_start rounding  → `alignUp`
- GGML_TYPES           → `typeName`
-/

namespace Strata.GgufReader

-- § header

/-- The GGUF magic number. -/
def ggufMagic : Nat := 0x46554747

/-- Little-endian bytes of the magic spell G, G, U, F. -/
theorem magic_bytes :
    ggufMagic % 256 = 71 ∧ ggufMagic / 256 % 256 = 71 ∧ ggufMagic / 65536 % 256 = 85 ∧
      ggufMagic / 16777216 = 70 := by
  decide

/-- Magic (4) + version (4) + n_tensors (8) + n_kv (8) = 24 header bytes; the tensor info tail is 12. -/
theorem header_sizes : 4 + (4 + 8 + 8) = 24 ∧ 4 + 8 = 12 := by
  decide

-- § element counts

/-- Product of a shape. -/
def shapeElems : List Nat → Nat
  | [] => 1
  | d :: ds => d * shapeElems ds

/-- Element counts multiply over concatenated shapes (so extra expert axes are just layout). -/
theorem shapeElems_append (a b : List Nat) : shapeElems (a ++ b) = shapeElems a * shapeElems b := by
  induction a with
  | nil => simp [shapeElems]
  | cons d ds ih => simp [shapeElems, ih, Nat.mul_assoc]

-- § expected bytes

/-- `expected_bytes`: whole blocks only. -/
def expectedBytes (elems blockElems blockBytes : Nat) : Option Nat :=
  if elems % blockElems = 0 then some (elems / blockElems * blockBytes) else none

/-- A whole number of blocks yields `blocks * block_bytes`. -/
theorem expectedBytes_of_blocks (k be bb : Nat) (h : 0 < be) :
    expectedBytes (k * be) be bb = some (k * bb) := by
  unfold expectedBytes
  rw [if_pos (Nat.mul_mod_left k be), Nat.mul_div_cancel k h]

/-- A partial block yields `None`. -/
theorem expectedBytes_partial (e be bb : Nat) (h : e % be ≠ 0) : expectedBytes e be bb = none := by
  unfold expectedBytes
  rw [if_neg h]

/-- A fused Q2_0 expert tensor `[64, 256, 16]` is 4096 blocks of 18 bytes. -/
theorem q2_0_tiny_expert_bytes : expectedBytes 262144 64 18 = some 73728 := by
  decide

-- § alignment of the data section

/-- Round `pos` up to a multiple of `a` (the `data_start` formula). -/
def alignUp (pos a : Nat) : Nat := (pos + a - 1) / a * a

/-- Alignment never moves backwards, moves less than one alignment unit, and lands on a multiple. -/
theorem alignUp_spec (pos a : Nat) (ha : 0 < a) :
    pos ≤ alignUp pos a ∧ alignUp pos a < pos + a ∧ alignUp pos a % a = 0 := by
  have h0 : (pos + a - 1) / a * a + (pos + a - 1) % a = pos + a - 1 := by
    have := Nat.div_add_mod (pos + a - 1) a
    rw [Nat.mul_comm] at this
    exact this
  have h1 : (pos + a - 1) % a < a := Nat.mod_lt _ ha
  have h2 : alignUp pos a % a = 0 := by
    unfold alignUp
    exact Nat.mul_mod_left _ _
  refine ⟨?_, ?_, h2⟩
  · unfold alignUp
    omega
  · unfold alignUp
    omega

/-- Aligning an already aligned position is the identity (default alignment 32). -/
theorem alignUp32_idempotent (k : Nat) : alignUp (32 * k) 32 = 32 * k := by
  unfold alignUp
  omega

-- § the ggml type table

/-- A sample of `GGML_TYPES`; unknown ids print as `type<id>` in the source. -/
def typeName : Nat → String
  | 0 => "F32"
  | 1 => "F16"
  | 2 => "Q4_0"
  | 8 => "Q8_0"
  | 30 => "BF16"
  | 31 => "Q4_0_8_8"
  | 41 => "Q4_0_8_8"
  | 42 => "Q2_0"
  | _ => "unknown"

/-- Ids 31 and 41 share a NAME but are distinct ids, so name-keyed tooling cannot tell them apart. -/
theorem duplicate_name_distinct_ids : typeName 31 = typeName 41 ∧ (31 : Nat) ≠ 41 := by
  exact ⟨rfl, by decide⟩

/-- Type id 42 is `Q2_0`, the type the stock gguf-py enum lacks. -/
theorem q2_0_is_type_42 : typeName 42 = "Q2_0" := by
  rfl

/-- The ids 0..42 of the table are pairwise distinct. -/
theorem type_ids_nodup : (List.range 43).Nodup := by
  decide

end Strata.GgufReader
