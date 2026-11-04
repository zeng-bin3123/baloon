/-!
# MODULE: lazy_gguf_tensor_addressing
`ref/load.py`: demand-paged GGUF access.  Experts and rows are read by byte arithmetic without decoding the
rest of the tensor; the second shard holds only the PLE table.

## Source semantics
* row_bytes = width / block_elems * block_bytes, requiring width % block_elems = 0.
* a row is contiguous: row r starts at r * row_bytes.
* expert e of a fused [in, out, experts] tensor starts at e * (in*out/el) * by, and runs are disjoint.
* tensors are reshaped by reversing the GGUF dims (dim 0 fastest).

## Form template identifiers
GEOM        -> (block_elems, block_bytes) table
row_bytes   -> Gguf.row_bytes
expert      -> Gguf.expert
-/

namespace Strata.LazyGgufTensorAddressing

-- § row addressing

/-- Bytes of one row. -/
def rowBytes (width el by_ : Nat) : Nat := width / el * by_

/-- Start of row `r`. -/
def rowStart (r rb : Nat) : Nat := r * rb

/-- Rows are disjoint contiguous runs. -/
theorem rows_disjoint (r r' rb : Nat) (h : r < r') : rowStart r rb + rb ≤ rowStart r' rb := by
  unfold rowStart
  have : (r + 1) * rb ≤ r' * rb := Nat.mul_le_mul_right rb h
  rw [Nat.add_mul, Nat.one_mul] at this
  exact this

/-- The PLE table row: 160 IQ4_NL elements = 5 blocks of 18 B = 90 B. -/
theorem ple_row_bytes : rowBytes 160 32 18 = 90 := by decide

/-- 320,001,536 rows of 90 B is 28,800,138,240 B (26.82 GiB). -/
theorem ple_table_bytes : 320001536 * 90 = 28800138240 := by decide

/-- Reading 16 rows per token touches 1,440 bytes, a vanishing fraction of the table. -/
theorem ple_token_read : 16 * 90 = 1440 := by decide

-- § expert addressing

/-- Bytes of one expert: `in*out/el*by`. -/
def expertBytes (inF outF el by_ : Nat) : Nat := inF * outF / el * by_

/-- Start of expert `e`. -/
def expertStart (e per : Nat) : Nat := e * per

/-- Expert runs are disjoint. -/
theorem experts_disjoint (e e' per : Nat) (h : e < e') :
    expertStart e per + per ≤ expertStart e' per := by
  unfold expertStart
  have : (e + 1) * per ≤ e' * per := Nat.mul_le_mul_right per h
  rw [Nat.add_mul, Nat.one_mul] at this
  exact this

/-- Q2_0 experts of the real model: 2560 x 640 elements = 460,800 B per expert. -/
theorem q2_0_expert_bytes : expertBytes 2560 640 64 18 = 460800 := by decide

/-- All 512 experts fill the role tensor. -/
theorem role_tensor_bytes : 512 * 460800 = 235929600 := by decide

/-- The per-expert element count is a whole number of Q2_0 blocks. -/
theorem q2_0_whole_blocks : 2560 * 640 % 64 = 0 := by decide

/-- The last expert ends exactly at the tensor end. -/
theorem last_expert_end : expertStart 511 460800 + 460800 = 512 * 460800 := by decide

-- § block geometry table

/-- (type, elements per block, bytes per block). -/
def geom : List (String × Nat × Nat) :=
  [("F32", 1, 4), ("F16", 1, 2), ("BF16", 1, 2), ("Q4_0", 32, 18), ("Q5_0", 32, 22), ("Q8_0", 32, 34),
   ("IQ4_NL", 32, 18), ("Q3_K", 256, 110), ("Q4_K", 256, 144), ("Q5_K", 256, 176), ("Q6_K", 256, 210),
   ("IQ4_XS", 256, 136), ("Q2_0", 64, 18)]

/-- Thirteen types are covered. -/
theorem geom_size : geom.length = 13 := by decide

/-- Every block size is positive, so row arithmetic never divides by zero. -/
theorem geom_positive : ∀ g ∈ geom, 0 < g.2.1 ∧ 0 < g.2.2 := by decide

/-- Block elements are 1, 32, 64 or 256. -/
theorem geom_el_cases : ∀ g ∈ geom, g.2.1 = 1 ∨ g.2.1 = 32 ∨ g.2.1 = 64 ∨ g.2.1 = 256 := by decide

-- § shape reversal

/-- GGUF dims (dim 0 fastest) reversed give the numpy shape. -/
theorem reverse_dims (i o e : Nat) : [i, o, e].reverse = [e, o, i] := rfl

/-- Reversal is an involution. -/
theorem reverse_involutive (l : List Nat) : l.reverse.reverse = l := List.reverse_reverse l

/-- A row index is in range iff it is below the second dim. -/
theorem row_in_range (r rows : Nat) (h : r < rows) (rb : Nat) : r * rb + rb ≤ rows * rb := by
  have : (r + 1) * rb ≤ rows * rb := Nat.mul_le_mul_right rb h
  rw [Nat.add_mul, Nat.one_mul] at this
  exact this

-- § shard discovery

/-- A shard set is complete iff the number of siblings found equals the declared total. -/
def complete (found total : Nat) : Bool := found == total

/-- Two shards of two is complete; one of two is not. -/
theorem shard_completeness : complete 2 2 = true ∧ complete 1 2 = false := by decide

end Strata.LazyGgufTensorAddressing
