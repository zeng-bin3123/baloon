/-!
# MODULE: gguf_shard_loader
SOURCES: ref/load.py
Demand-paged GGUF loader: shard discovery, per-(layer, expert) lazy byte ranges, row gathering and
the shared dequantization dispatch geometry.

## Source semantics
* `GEOM` maps a type name to (elements per block, bytes per block); every block has a positive element count.
* `expert(name, e)` reads only `per_expert/el*by` bytes starting at `e * (per_expert/el) * by`; experts tile the tensor.
* `rows(name, idx)` reads `row_bytes` at `r * row_bytes`; row `r < ncols` stays inside the tensor.
* BF16 -> FP32 is `bits << 16`, so the low 16 bits are zero.
* Shard discovery refuses to guess: 0 files, 1 file, a shard set, or an ambiguous directory.

## Form template identifiers
- `GEOM` -> `block_geometry_table`; `Gguf.expert` -> `lazy_expert_slice`; `Gguf.rows` -> `lazy_row_gather`;
  `_discover_shards` -> `shard_set_resolver`; `_dequant_flat` -> `dequant_dispatch`.
-/

namespace Strata.GgufShardLoader

-- § block geometry table

/-- The `GEOM` table of ref/load.py as (type name, elements per block, bytes per block). -/
def geomTable : List (String × Nat × Nat) :=
  [("F32", 1, 4), ("F16", 1, 2), ("BF16", 1, 2), ("Q4_0", 32, 18), ("Q5_0", 32, 22),
   ("Q8_0", 32, 34), ("IQ4_NL", 32, 18), ("Q3_K", 256, 110), ("Q4_K", 256, 144),
   ("Q5_K", 256, 176), ("Q6_K", 256, 210), ("IQ4_XS", 256, 136), ("Q2_0", 64, 18)]

/-- Every geometry entry has a positive element count and positive byte count. -/
theorem geom_positive : ∀ e ∈ geomTable, 0 < e.2.1 ∧ 0 < e.2.2 := by
  decide

/-- The table has exactly the 13 encodings of the source. -/
theorem geom_size : geomTable.length = 13 := by
  decide

/-- Q2_0 is 64 elements in 18 bytes (2-byte scale + 16 bytes of 4 codes). -/
theorem q2_0_geometry : ("Q2_0", 64, 18) ∈ geomTable := by
  decide

-- § byte offsets

/-- Bytes of a run of `n` elements in a type with block geometry `(el, by)`. -/
def runBytes (n el by_ : Nat) : Nat := n / el * by_

/-- Start byte of expert `e`: whole experts before it, measured in blocks. -/
def expertStart (e perExpert el by_ : Nat) : Nat := e * (perExpert / el) * by_

/-- Byte length of one expert's run. -/
def expertBytes (perExpert el by_ : Nat) : Nat := perExpert / el * by_

/-- Expert `e + 1` starts exactly where expert `e` ends, so reads never overlap or leave gaps. -/
theorem expert_contiguous (e perExpert el by_ : Nat) :
    expertStart (e + 1) perExpert el by_ = expertStart e perExpert el by_ + expertBytes perExpert el by_ := by
  unfold expertStart expertBytes
  simp [Nat.add_mul]

/-- The start of expert `e` is `e` times the per-expert byte size. -/
theorem expert_start_eq (e perExpert el by_ : Nat) :
    expertStart e perExpert el by_ = e * expertBytes perExpert el by_ := by
  unfold expertStart expertBytes
  rw [Nat.mul_assoc]

/-- A valid expert index reads entirely inside the fused tensor (`e < n_exp`). -/
theorem expert_in_range (e nExp perExpert el by_ : Nat) (h : e < nExp) :
    expertStart e perExpert el by_ + expertBytes perExpert el by_ ≤ nExp * expertBytes perExpert el by_ := by
  have h1 : expertStart e perExpert el by_ + expertBytes perExpert el by_ =
      (e + 1) * expertBytes perExpert el by_ := by
    rw [expert_start_eq, Nat.add_mul, Nat.one_mul]
  rw [h1]
  exact Nat.mul_le_mul_right _ h

/-- Experts tile the tensor: `n` experts of `per` elements occupy `n` times one expert's bytes,
provided the block size divides the per-expert element count (the source raises otherwise). -/
theorem experts_tile (n per el by_ : Nat) (h : el ∣ per) :
    runBytes (n * per) el by_ = n * expertBytes per el by_ := by
  unfold runBytes expertBytes
  rw [Nat.mul_div_assoc n h, Nat.mul_assoc]

/-- Row `r` of a 2-D tensor starts at `r * rowBytes`. -/
def rowStart (r rowBytes : Nat) : Nat := r * rowBytes

/-- `row_bytes` of a row of `width` elements. -/
def rowBytes (width el by_ : Nat) : Nat := width / el * by_

/-- A row index below the column count reads inside the tensor. -/
theorem row_in_range (r ncols rb : Nat) (h : r < ncols) :
    rowStart r rb + rb ≤ ncols * rb := by
  unfold rowStart
  have : (r + 1) * rb ≤ ncols * rb := Nat.mul_le_mul_right _ h
  rw [Nat.add_mul, Nat.one_mul] at this
  exact this

/-- The per-layer embedding table: width 160 in IQ4_NL (32-element, 18-byte blocks) is 90 bytes per row. -/
theorem ple_row_bytes : rowBytes 160 32 18 = 90 := by
  decide

/-- 160 is a multiple of the IQ4_NL block size (the divisibility the source checks). -/
theorem ple_width_divisible : 32 ∣ 160 := by
  decide

-- § BF16 to FP32

/-- BF16 bits widened to f32 bits: shift left by 16. -/
def bf16ToF32Bits (h : Nat) : Nat := h * 65536

/-- Widening a 16-bit pattern stays within 32 bits. -/
theorem bf16_fits32 (h : Nat) (hh : h < 65536) : bf16ToF32Bits h < 4294967296 := by
  unfold bf16ToF32Bits
  omega

/-- The low 16 bits of the widened pattern are zero (the mantissa is just padded). -/
theorem bf16_low_zero (h : Nat) : bf16ToF32Bits h % 65536 = 0 := by
  unfold bf16ToF32Bits
  omega

/-- The high 16 bits are the original pattern. -/
theorem bf16_high_roundtrip (h : Nat) : bf16ToF32Bits h / 65536 = h := by
  unfold bf16ToF32Bits
  omega

-- § shard discovery

/-- Outcome of resolving a directory argument. -/
inductive DirResult where
  | shardSet
  | single
  | noFiles
  | ambiguous
  deriving DecidableEq

/-- Directory resolution: prefer an explicit shard set, accept one unambiguous file, otherwise refuse. -/
def dirDecision (shards found : Nat) : DirResult :=
  if shards ≠ 0 then DirResult.shardSet
  else if found = 1 then DirResult.single
  else if found = 0 then DirResult.noFiles
  else DirResult.ambiguous

/-- An mmproj file next to a real shard set never hijacks the choice: the shard set wins. -/
theorem dir_prefers_shards (shards found : Nat) (h : shards ≠ 0) :
    dirDecision shards found = DirResult.shardSet := by
  simp [dirDecision, h]

/-- With no shards and several files the loader refuses to guess. -/
theorem dir_refuses_ambiguous (found : Nat) (h : 2 ≤ found) :
    dirDecision 0 found = DirResult.ambiguous := by
  have h1 : found ≠ 1 := by omega
  have h2 : found ≠ 0 := by omega
  simp [dirDecision, h1, h2]

/-- A single unambiguous file is accepted. -/
theorem dir_accepts_single : dirDecision 0 1 = DirResult.single := by
  decide

/-- An empty directory is an error, not an empty model. -/
theorem dir_empty : dirDecision 0 0 = DirResult.noFiles := by
  decide

/-- A shard named `-NNNNN-of-MMMMM` is complete only when exactly MMMMM siblings exist. -/
def shardsComplete (found total : Nat) : Bool := found == total

/-- Completeness holds exactly when the sibling count equals the declared total. -/
theorem shards_complete_iff (found total : Nat) :
    shardsComplete found total = true ↔ found = total := by
  simp [shardsComplete]

end Strata.GgufShardLoader
