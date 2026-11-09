/-!
# MODULE: s2_gemv_fast_negative_result
SOURCES: src/kernels/cuda/s2_gemv_fast.cu
Experimental S2 (2-bit) GEMV with a constant-memory byte->four-code table and optional shared staging of the
activation. Both levers were measured negative/neutral; the file is retained as a record, not as a fast path.

## Source semantics
- Table `c_codes[b][k] = ((b >> 2k) & 3) - 1` for each byte b and field k.
- One block per row; quads (4 elements) per thread; scale index for quad q is `q >> 4`; staging needs
  `n_in <= 4096` halves; partial sums live after the staged halves in dynamic shared memory.

## Form template identifiers
- c_codes -> lutEntry ; MAX_SHARED_HALVES -> maxStage ; QK_S2 -> 64
-/

namespace Strata.S2Fast

/-- Table entry for byte `b`, field `k` (bias -1 applied). -/
def lutEntry (b k : Nat) : Int := ((b / 4 ^ k) % 4 : Nat) - 1

/-- Every table entry is in {-1,0,1,2}. -/
theorem lutEntry_range (b k : Nat) : -1 ≤ lutEntry b k ∧ lutEntry b k ≤ 2 := by
  unfold lutEntry
  have h : (b / 4 ^ k) % 4 < 4 := Nat.mod_lt _ (by decide)
  omega

/-- The four entries of a byte reassemble the byte (biased back by +1). -/
theorem lut_reassemble (b : Nat) (hb : b < 256) :
    (lutEntry b 0 + 1) + 4 * (lutEntry b 1 + 1) + 16 * (lutEntry b 2 + 1) + 64 * (lutEntry b 3 + 1) = b := by
  unfold lutEntry
  simp only [Nat.pow_zero, Nat.pow_one, Nat.div_one]
  omega

/-- Scale index of quad `q`: element `4q` sits in group `(4q)/64 = q/16`. -/
theorem scale_index (q : Nat) : (q * 4) / 64 = q / 16 := by
  omega

/-- If `n_in` is a multiple of 64, then the number of quads is a multiple of 16. -/
theorem quads_multiple_of_16 (n : Nat) (h : n % 64 = 0) : (n / 4) % 16 = 0 := by
  omega

/-- Quad count times four recovers `n_in` when `n_in` is a multiple of 4. -/
theorem quads_cover (n : Nat) (h : n % 4 = 0) : (n / 4) * 4 = n := by
  omega

/-- Staging buffer: 4096 halves is 8 KiB. -/
theorem stage_bytes : 4096 * 2 = 8192 := by decide

/-- The partial-sum array starts at float index `MAX/2`, exactly the byte end of the staged halves. -/
theorem partial_offset_consistent : (4096 / 2) * 4 = 4096 * 2 := by decide

/-- n_embd = 2560 fits the staging limit. -/
theorem n_embd_fits : 2560 ≤ 4096 := by decide

/-- Dynamic shared bytes with staging and 128 threads: 8192 + 512 fits in 48 KiB. -/
theorem smem_with_stage : 4096 * 2 + 128 * 4 ≤ 49152 := by decide

/-- A quad index owned by thread `tid` with stride `tpr` advances within range check `q < n_quads`. -/
theorem quad_strided_cover (q tpr : Nat) (htpr : 0 < tpr) : q % tpr < tpr := by
  exact Nat.mod_lt _ htpr

/-- Tree reduction halves a power-of-two width down to 1 in `log2` steps; 128 -> 1 takes 7 halvings. -/
theorem halvings_128 : 128 / 2 / 2 / 2 / 2 / 2 / 2 / 2 = 1 := by decide

end Strata.S2Fast
