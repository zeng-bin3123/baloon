/-!
# MODULE: s2_gemv_const_lut
The constant-memory code table and quad indexing of the (measured-neutral) `s2_gemv_fast` kernel.

## Source semantics
`c_codes[b][k] = ((b >> 2k) & 3) - 1`: one byte yields four biased codes. A quad `q` covers elements
`4q..4q+3`; its weight scale is `scales[q >> 4]` (one per 64 elements). x staged in shared memory must fit
`MAX_SHARED_HALVES`.

## Form template identifiers
lutCode -> c_codes; quadScale -> scale index; maxShared -> MAX_SHARED_HALVES.
-/

namespace Strata.S2GemvConstLut

def lutCode (b k : Nat) : Int := ((b / 4 ^ k) % 4 : Nat) - 1

def maxShared : Nat := 4096

-- § table range

/-- Every table entry is in {-1,0,1,2}. -/
theorem lutCode_range (b k : Nat) : -1 ≤ lutCode b k ∧ lutCode b k ≤ 2 := by
  unfold lutCode
  have : (b / 4 ^ k) % 4 < 4 := Nat.mod_lt _ (by decide)
  omega

/-- The four entries of a byte undo to the byte (after the bias is added back). -/
theorem lut_reconstruct (b : Nat) (hb : b < 256) :
    (lutCode b 0 + 1) + 4 * (lutCode b 1 + 1) + 16 * (lutCode b 2 + 1) + 64 * (lutCode b 3 + 1) = (b : Int) := by
  unfold lutCode
  simp only [Nat.pow_zero, Nat.pow_one, show (4:Nat)^2 = 16 by decide, show (4:Nat)^3 = 64 by decide,
    Nat.div_one]
  omega

-- § quad indexing

/-- The scale of quad `q` is `q >> 4`, which is `(4*q) / 64`. -/
theorem quadScale (q : Nat) : q / 16 = (q * 4) / 64 := by
  omega

/-- n_embd = 2560 halves fit in the shared staging area. -/
theorem n_embd_fits : 2560 ≤ maxShared := by
  decide

/-- Quads of a row cover its elements exactly when `n_in % 4 = 0`. -/
theorem quads_cover (n : Nat) (h : n % 4 = 0) : (n / 4) * 4 = n := by
  omega

end Strata.S2GemvConstLut
