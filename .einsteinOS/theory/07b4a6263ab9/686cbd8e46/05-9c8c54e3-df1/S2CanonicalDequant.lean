/-!
# MODULE: s2_canonical_dequant
SOURCES: src/kernels/cuda/dequant_s2.cu
Dequantization of the canonical S2 plane (Q2_0's form): 2-bit codes packed four per byte LSB-first, one FP32
scale per 64-element group, decoded as `(code - 1) * scale`.  This module models the per-block kernel loop,
proves its byte/shift addressing equals the global plane convention, and pins the decoded value set.

## Source semantics
* Element `i` of the plane is at byte `i/4`, bits `(i%4)*2`; a group/block is 64 elements = 16 code bytes.
* Thread `b` handles block `b`: code byte `b*16 + j/4`, shift `(j%4)*2`, output `b*64 + j`, `j < 64`.
* Decode: the bias −1 is applied to the CODE in the integer domain, then ONE multiply by the scale —
  this is what makes the decode bit-exact against the reference (`docs/pack-format.md` §3.1).
* `n_blocks ≤ 0` launches nothing; grid `⌈n_blocks / 256⌉` × 256.

## Form template identifiers
`dequant_s2_kernel` → `kernelElem`; `QK` → `64`; `CODES_PER_BYTE` → `4`; `code_bias` → `-1`;
plane element accessor → `planeCode`; decoded value → `decode`.
-/

namespace Strata.S2CanonicalDequant

-- § the plane convention

/-- The plane's code for global element `i`: byte `i/4`, bits `(i%4)*2`, two bits wide. -/
def planeCode (byte : Nat → Nat) (i : Nat) : Nat := (byte (i / 4) >>> ((i % 4) * 2)) &&& 3

/-- What the kernel thread for block `b` reads for its element `j`. -/
def kernelCode (byte : Nat → Nat) (b j : Nat) : Nat := (byte (b * 16 + j / 4) >>> ((j % 4) * 2)) &&& 3

/-- Shift-and-mask is division-and-modulo. -/
theorem extract2 (x s : Nat) : (x >>> s) &&& 3 = x / 2 ^ s % 4 := by
  have := Nat.and_two_pow_sub_one_eq_mod (x >>> s) 2
  rw [Nat.shiftRight_eq_div_pow] at this ⊢
  simpa using this

/-- Codes are two bits. -/
theorem planeCode_lt (byte : Nat → Nat) (i : Nat) : planeCode byte i < 4 := by
  unfold planeCode; rw [extract2]; exact Nat.mod_lt _ (by decide)

/-- **Addressing agrees with the plane.**  Block `b`, element `j` reads exactly global element `b*64 + j`
(the identity needs no bound on `j`: 64 is a multiple of the 4 codes per byte). -/
theorem kernel_reads_plane (byte : Nat → Nat) (b j : Nat) :
    kernelCode byte b j = planeCode byte (b * 64 + j) := by
  have h1 : (b * 64 + j) / 4 = b * 16 + j / 4 := by omega
  have h2 : (b * 64 + j) % 4 = j % 4 := by omega
  simp only [kernelCode, planeCode, h1, h2]

/-- Distinct `(block, element)` pairs write distinct outputs `b*64 + j`. -/
theorem out_index_inj (b1 j1 b2 j2 : Nat) (h1 : j1 < 64) (h2 : j2 < 64)
    (h : b1 * 64 + j1 = b2 * 64 + j2) : b1 = b2 ∧ j1 = j2 := by
  omega

/-- A block's 64 elements use exactly its 16 code bytes. -/
theorem block_bytes (b j : Nat) (hj : j < 64) : b * 16 ≤ b * 16 + j / 4 ∧ b * 16 + j / 4 < (b + 1) * 16 := by
  omega

-- § decoding

/-- `(code + code_bias) * scale` with `code_bias = -1`, the bias in the integer domain. -/
def decode (code : Nat) (scale : Int) : Int := ((code : Int) - 1) * scale

/-- The integer multiplier takes only the values `-1, 0, 1, 2`. -/
theorem multiplier_values (code : Nat) (h : code < 4) :
    (code : Int) - 1 = -1 ∨ (code : Int) - 1 = 0 ∨ (code : Int) - 1 = 1 ∨ (code : Int) - 1 = 2 := by
  omega

/-- Code 1 decodes to zero for every scale: the plane can represent an exact zero. -/
theorem decode_one_zero (scale : Int) : decode 1 scale = 0 := by simp [decode]

/-- Decoding is ONE multiply of an integer multiplier by the scale (in the exact model, linear in the scale). -/
theorem decode_linear (code : Nat) (s t : Int) : decode code (s + t) = decode code s + decode code t := by
  unfold decode; rw [Int.mul_add]

-- § launch

/-- Grid size `⌈n_blocks / 256⌉`. -/
def grid (nBlocks : Nat) : Nat := (nBlocks + 255) / 256

/-- Every block has a thread, and the last grid block is not wasted entirely. -/
theorem grid_tight (nBlocks : Nat) (h : 0 < nBlocks) :
    nBlocks ≤ grid nBlocks * 256 ∧ (grid nBlocks - 1) * 256 < nBlocks := by
  unfold grid; omega

end Strata.S2CanonicalDequant
