/-!
# MODULE: gguf_lazy_loader
SOURCES: ref/load.py
Byte addressing of the lazy GGUF reader: per-expert slices of a fused `[in, out, experts]` tensor and
per-row slices of a 2-D tensor, with the block geometry table.

## Source semantics
* `expert(name, e)` reads `per_expert / el * by` bytes at `e * (per_expert / el) * by`; consecutive
  experts are adjacent and the last one ends at the tensor end.
* `rows(name, idx)` reads `row_bytes = width / el * by` bytes at `row * row_bytes`.
* A read is legal only when the element count is a multiple of the block width.
* A Q2_0 routed expert tensor `[2560, 640, 512]` has 460,800 bytes per expert; three of them make the
  1,382,400-byte blob of the CPU kernel.

## Form template identifiers
* GEOM -> block_geometry_table (elements per block, bytes per block)
* Gguf.expert -> expert_slice, Gguf.rows -> row_gather
-/

namespace Strata.GgufLoader

-- § geometry table
/-- (elements per block, bytes per block) per type, as in `GEOM`. -/
def geom : String → Nat × Nat
  | "F32" => (1, 4)
  | "F16" => (1, 2)
  | "BF16" => (1, 2)
  | "Q4_0" => (32, 18)
  | "Q5_0" => (32, 22)
  | "Q8_0" => (32, 34)
  | "IQ4_NL" => (32, 18)
  | "Q3_K" => (256, 110)
  | "Q4_K" => (256, 144)
  | "Q5_K" => (256, 176)
  | "Q6_K" => (256, 210)
  | "IQ4_XS" => (256, 136)
  | "Q2_0" => (64, 18)
  | _ => (1, 0)

/-- Block widths are positive for every listed type, so the divisions below are meaningful. -/
theorem geom_width_pos : ∀ t ∈ ["F32", "F16", "BF16", "Q4_0", "Q5_0", "Q8_0", "IQ4_NL", "Q3_K", "Q4_K",
    "Q5_K", "Q6_K", "IQ4_XS", "Q2_0"], 0 < (geom t).1 := by
  decide

-- § expert slice
/-- Bytes in one expert's slice. -/
def expertBytes (per el by_ : Nat) : Nat := per / el * by_

/-- Start offset of expert `e`. -/
def expertStart (e per el by_ : Nat) : Nat := e * (per / el) * by_

/-- Consecutive experts are adjacent: expert `e` ends exactly where expert `e+1` starts. -/
theorem expert_adjacent (e per el by_ : Nat) :
    expertStart e per el by_ + expertBytes per el by_ = expertStart (e + 1) per el by_ := by
  unfold expertStart expertBytes
  simp only [Nat.add_mul, Nat.one_mul]

/-- The first expert starts at byte 0. -/
theorem expert_first (per el by_ : Nat) : expertStart 0 per el by_ = 0 := by
  simp [expertStart]

/-- The last legal expert ends at `n_exp * bytes`, i.e. the end of the fused tensor. -/
theorem expert_last_end (e nexp per el by_ : Nat) (h : e + 1 = nexp) :
    expertStart e per el by_ + expertBytes per el by_ = nexp * (per / el) * by_ := by
  rw [expert_adjacent, ← h]
  rfl

/-- A routed Q2_0 expert tensor `[2560, 640, 512]`: 460,800 bytes per projection. -/
theorem q2_expert_projection_bytes : expertBytes (2560 * 640) 64 18 = 460800 := by decide

/-- Three projections (gate, up, down) make up the 1,382,400-byte kernel blob. -/
theorem q2_expert_blob_bytes : 3 * expertBytes (2560 * 640) 64 18 = 1382400 := by decide

/-- The whole fused gate tensor holds 512 such experts. -/
theorem q2_fused_tensor_bytes : 512 * expertBytes (2560 * 640) 64 18 = 235929600 := by decide

-- § row gather
/-- Bytes per row of width `w`. -/
def rowBytes (w el by_ : Nat) : Nat := w / el * by_

/-- Row `r` starts at `r * rowBytes` and rows are adjacent. -/
theorem row_adjacent (r rb : Nat) : r * rb + rb = (r + 1) * rb := by
  rw [Nat.add_mul, Nat.one_mul]

/-- A row index below the row count keeps the read inside the tensor. -/
theorem row_in_tensor (r nrows rb : Nat) (h : r < nrows) : r * rb + rb ≤ nrows * rb := by
  have : (r + 1) * rb ≤ nrows * rb := Nat.mul_le_mul_right rb h
  rw [Nat.add_mul, Nat.one_mul] at this
  exact this

/-- The per-token PLE table: 160-wide IQ4_NL rows are 5 blocks of 18 bytes = 90 bytes. -/
theorem ple_row_bytes : rowBytes 160 32 18 = 90 := by decide

/-- 160 is a multiple of the IQ4_NL block width (required by `rows`). -/
theorem ple_width_divisible : 160 % 32 = 0 := by decide

-- § validity of a read
/-- The element count must be a whole number of blocks. -/
def validElements (elements el : Nat) : Bool := el != 0 && elements % el == 0

/-- A valid read covers exactly `elements / el` blocks, with no partial block. -/
theorem valid_exact_blocks (elements el : Nat) (h : validElements elements el = true) :
    elements / el * el = elements := by
  unfold validElements at h
  simp only [Bool.and_eq_true, bne_iff_ne, beq_iff_eq] at h
  exact Nat.div_mul_cancel (Nat.dvd_of_mod_eq_zero h.2)

-- § shard discovery
/-- Shard `i` of `total` must have a sibling for every index: expected count equals found count. -/
def shardsComplete (found total : Nat) : Bool := found == total

/-- Complete shard sets are exactly those with matching counts. -/
theorem shardsComplete_iff (found total : Nat) : shardsComplete found total = true ↔ found = total := by
  simp [shardsComplete]

end Strata.GgufLoader
