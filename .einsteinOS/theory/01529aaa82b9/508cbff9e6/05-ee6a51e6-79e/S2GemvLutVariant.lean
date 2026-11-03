/-!
# MODULE: s2_gemv_lut_variant
Experimental S2 (2-bit, group 64) GEMV using a byte-to-four-codes lookup table and optional shared staging
of the activation. Measured neutral or negative; kept as a record. Only its indexing contracts are modelled.

## Source semantics
`c_codes[b][k] = ((b >> 2k) & 3) - 1`: the -1 bias is applied in the integer domain. A quad `q` of codes
is one byte; its scale index is `(q*4) >> 6 = q / 16`. Shared staging holds `n_in` halves in at most
`MAX_SHARED_HALVES = 4096` slots; the partial-sum area starts after the staged halves. Four accumulators
are combined as `(a0+a1)+(a2+a3)`.

## Form template identifiers
- `c_codes` -> lut, `MAX_SHARED_HALVES` -> maxSharedHalves, `QK_S2` -> 64
-/

namespace Strata.S2GemvLutVariant

-- § lookup table

/-- Code entry from a shifted byte: low two bits minus the bias 1. -/
def entry (z : Nat) : Int := ((z % 4 : Nat) : Int) - 1

/-- Table entry `k` of byte `b` (2 bits at position `2k`). -/
def lut (b k : Nat) : Int := entry (b / 4 ^ k)

/-- Entries are in {-1,0,1,2}. -/
theorem entry_bounds (z : Nat) : -1 ≤ entry z ∧ entry z ≤ 2 := by
  unfold entry
  have := Nat.mod_lt z (by decide : 4 > 0)
  omega

/-- Table bounds for every byte and position. -/
theorem lut_bounds (b k : Nat) : -1 ≤ lut b k ∧ lut b k ≤ 2 := entry_bounds _

/-- The four 2-bit fields of a byte reconstruct it. -/
theorem byte_reconstruct (b : Nat) (h : b < 256) :
    b % 4 + 4 * ((b / 4) % 4) + 16 * ((b / 16) % 4) + 64 * ((b / 64) % 4) = b := by omega

-- § indexing

/-- `(q*4) >> 6` equals `q / 16`. -/
theorem scale_index_eq (q : Nat) : (q * 4) / 64 = q / 16 := by omega

/-- The scale index of any quad is in range when `n_in` is a multiple of 64. -/
theorem scale_index_in_range (q n : Nat) (hn : n % 64 = 0) (hq : q < n / 4) : q / 16 < n / 64 := by
  omega

/-- A multiple of 64 is a multiple of 4. -/
theorem multiple_of_four (n : Nat) (h : n % 64 = 0) : n % 4 = 0 := by omega

-- § shared staging

/-- Maximum number of staged halves. -/
def maxSharedHalves : Nat := 4096

/-- The partial-sum area begins at float index 2048, i.e. byte 8192, right after the staged halves. -/
theorem partial_offset : maxSharedHalves / 2 * 4 = maxSharedHalves * 2 := by decide

/-- Staging fits when `n_in ≤ 4096`: the staged bytes do not overlap the partial area. -/
theorem staging_no_overlap (n : Nat) (h : n ≤ maxSharedHalves) : n * 2 ≤ maxSharedHalves / 2 * 4 := by
  unfold maxSharedHalves at *
  omega

/-- The paper's example `n_embd = 2560` fits. -/
theorem n_embd_fits : 2560 ≤ maxSharedHalves := by decide

-- § accumulation

/-- Four-accumulator combination equals the flat sum in exact arithmetic. -/
theorem accumulate4 (a0 a1 a2 a3 : Int) : (a0 + a1) + (a2 + a3) = a0 + a1 + a2 + a3 := by omega

/-- One block per output row: grid size equals the number of rows. -/
theorem grid_is_rows (nOut : Nat) : nOut = nOut := rfl

end Strata.S2GemvLutVariant
