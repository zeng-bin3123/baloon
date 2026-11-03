/-!
# MODULE: ple_table_io
The `PleTable` layer of `src/kernels/ngram.cpp`: validation of the table against the file at open, the
one-token-in-flight issue/collect protocol, byte accounting, and the direct-mode offset.

## Source semantics
Open accepts the table only if `rows * 90` exactly fills the file after the GGUF header, which makes the
data offset checkable. Every in-range row then lies inside the file. A table holds at most one token in
flight: `issue` fails while pending, `collect` fails without a prior `issue`, and `gather_batch` refuses to
run while a token is in flight. A token reads 16 rows of 90 bytes (1,440 bytes). Out-of-range rows read as
zeros and are not counted as bytes read.

## Form template identifiers
- `sizeCheck` <- the `need != have` test in PleTable::open
- `issueStep`, `collectStep`, `gatherAllowed` <- `pending` handling in issue / collect / gather_batch
- `bytesAfterRead` <- `bytes_read += PLE_ROW_BYTES` guarded by `row < n_rows`
- `hitPercent` <- the cache-hit percentage in io_report
-/

namespace Strata.PleTableIo

-- § open-time validation

/-- The table must exactly fill the file from the data start. -/
def sizeCheck (rows fileSize dataStart : Nat) : Bool := decide (rows * 90 = fileSize - dataStart)

/-- An accepted table plus the header is exactly the file. -/
theorem sizeCheck_exact (rows fs ds : Nat) (hds : ds ≤ fs) (h : sizeCheck rows fs ds = true) :
    ds + rows * 90 = fs := by
  unfold sizeCheck at h
  have := of_decide_eq_true h
  omega

/-- After an accepted open, every in-range row lies inside the file. -/
theorem row_inside_file (rows fs ds row : Nat) (hds : ds ≤ fs) (h : rows * 90 = fs - ds)
    (hr : row < rows) : ds + row * 90 + 90 ≤ fs := by
  omega

/-- The row width `160/32 * 18` is the 90-byte `PLE_ROW_BYTES`. -/
theorem row_bytes_from_shape : 160 / 32 * 18 = 90 := by decide

/-- Shape check: a 2-D tensor whose fast axis is 160. -/
def shapeOk (ndims fast : Nat) : Bool := decide (ndims = 2 ∧ fast = 160)

/-- A wrongly shaped tensor is rejected. -/
theorem shape_rejects_wrong_width (n : Nat) (h : n ≠ 160) : shapeOk 2 n = false := by
  unfold shapeOk
  simp [h]

/-- In direct mode the reader's table offset is the data start plus the tensor offset; with the tensor
at offset 0 the table ends exactly at the end of the file. -/
theorem direct_table_end (rows fs ds off : Nat) (hds : ds ≤ fs) (h : rows * 90 = fs - ds) (h0 : off = 0) :
    ds + off + rows * 90 = fs := by
  omega

-- § issue / collect protocol

/-- `issue` on a table with a token in flight fails (`none`); otherwise the token becomes pending. -/
def issueStep (pending : Bool) : Option Bool := if pending then none else some true

/-- `collect` needs a pending token and clears it. -/
def collectStep (pending : Bool) : Option Bool := if pending then some false else none

/-- `gather_batch` is refused while a token is in flight. -/
def gatherAllowed (pending : Bool) : Bool := !pending

/-- issue followed by collect returns the table to idle. -/
theorem issue_collect_roundtrip : (issueStep false).bind collectStep = some false := rfl

/-- A second `issue` before `collect` fails: one token in flight per table. -/
theorem issue_twice_fails : issueStep true = none := rfl

/-- `collect` without `issue` fails. -/
theorem collect_without_issue : collectStep false = none := rfl

/-- A batch gather is allowed exactly when the table is idle. -/
theorem gather_allowed_iff (p : Bool) : gatherAllowed p = true ↔ p = false := by
  cases p <;> simp [gatherAllowed]

-- § byte accounting

/-- A token reads 16 rows of 90 bytes. -/
theorem token_bytes : 16 * 90 = 1440 := by decide

/-- A batch of `n` tokens is `n * 16` rows and `n * 1440` bytes. -/
theorem batch_bytes (n : Nat) : n * 16 * 90 = n * 1440 := by omega

/-- Bytes read after one `read_row`: unchanged for an out-of-range row (which reads as zeros). -/
def bytesAfterRead (br row rows : Nat) : Nat := if row < rows then br + 90 else br

/-- Counted bytes never decrease and grow by at most one row. -/
theorem bytesAfterRead_bounds (br row rows : Nat) :
    br ≤ bytesAfterRead br row rows ∧ bytesAfterRead br row rows ≤ br + 90 := by
  unfold bytesAfterRead
  split <;> omega

/-- Cache-hit percentage in the report is at most 100 when hits do not exceed requests. -/
def hitPercent (hits requests : Nat) : Nat := hits * 100 / requests

/-- The percentage never exceeds 100. -/
theorem hitPercent_le (hits requests : Nat) (h : hits ≤ requests) : hitPercent hits requests ≤ 100 := by
  unfold hitPercent
  apply Nat.div_le_of_le_mul
  omega

end Strata.PleTableIo
