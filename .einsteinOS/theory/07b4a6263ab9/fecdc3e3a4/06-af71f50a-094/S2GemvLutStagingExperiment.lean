/-!
# MODULE: s2_gemv_lut_staging_experiment
SOURCES: src/kernels/cuda/s2_gemv_fast.cu
The two measured-negative/neutral levers on the S2 GEMV, kept as an experiment: a 256×4 constant-memory
table turning a code byte into four biased code values, and optional staging of `x` in shared memory.
The theory states what makes the experiment a fair A/B: it computes the same values as the quads kernel.

## Source semantics
* `c_codes[b][k] = ((b >> 2k) & 3) − 1` for every byte `b` and slot `k` (host-built once, `ensure_lut`).
* One block per row; quad `q` reads `c[q]`, scale `s[q >> 4]`, activations `4q … 4q+3`.
* Dynamic shared memory: when staging, `MAX_SHARED_HALVES = 4096` halves (8 KiB) for `x`, then the
  `tpr` float partials at float offset `4096/2 = 2048` (byte 8192); without staging, partials at 0.
* Host: `n_in % 4 = 0 ∧ n_in % 64 = 0`; staging requires `n_in ≤ 4096`; `tpr` is not validated.
* Measured result recorded in the source: staging slower, LUT neutral — not adopted.

## Form template identifiers
* `lut` ↔ `c_codes`;  `MAX_SHARED_HALVES`;  `partialByteOffset` ↔ `smem + (STAGE_X ? MAX/2 : 0)`
-/

namespace Strata.S2Fast

/-- `x & 3 = x mod 4`. -/
theorem and3 (x : Nat) : x &&& 3 = x % 4 := by simpa using Nat.and_two_pow_sub_one_eq_mod x 2

-- § the code table

/-- `c_codes[b][k]`. -/
def lut (b k : Nat) : Int := (((b >>> (2 * k)) &&& 3 : Nat) : Int) - 1

/-- The per-element decode of the reference kernels: element `e` of the plane, bias −1. -/
def refDecode (byteAt : Nat → Nat) (e : Nat) : Int := (((byteAt (e / 4) >>> ((e % 4) * 2)) &&& 3 : Nat) : Int) - 1

/-- **The table is the decode**: `lut[c[q]][k]` equals the reference decode of element `4q + k`. -/
theorem lut_eq_decode (byteAt : Nat → Nat) (q k : Nat) (hk : k < 4) :
    lut (byteAt q) k = refDecode byteAt (4 * q + k) := by
  unfold lut refDecode
  have h1 : (4 * q + k) / 4 = q := by omega
  have h2 : (4 * q + k) % 4 * 2 = 2 * k := by omega
  rw [h1, h2]

/-- Table entries are the four biased codes `−1, 0, 1, 2`. -/
theorem lut_range (b k : Nat) : -1 ≤ lut b k ∧ lut b k ≤ 2 := by
  unfold lut; rw [and3]; omega

/-- A few table rows, as the host loop builds them. -/
theorem lut_rows : (lut 0 0, lut 0 3) = (-1, -1) ∧ (lut 255 0, lut 255 3) = (2, 2) ∧
    (lut 0b11100100 0, lut 0b11100100 1, lut 0b11100100 2, lut 0b11100100 3) = (-1, 0, 1, 2) := by decide

/-- The table is 4 KiB of constant memory (256 rows × float4), well under the 64 KiB bank. -/
theorem lut_size : 256 * 4 * 4 = 4096 ∧ 4096 ≤ 65536 := by decide

-- § group and activation addressing

/-- The scale index `q >> 4` is the group of every element of the quad. -/
theorem quad_group (q k : Nat) (hk : k < 4) : (4 * q + k) / 64 = q >>> 4 := by
  rw [Nat.shiftRight_eq_div_pow]; simp; omega

/-- `&sx[q*4]` and `x + q*4` are 8-byte aligned (four halves), so the `uint2`/`__half2` reads are aligned. -/
theorem quad_halves_aligned (q : Nat) : (2 * (4 * q)) % 8 = 0 ∧ (2 * (4 * q + 2)) % 4 = 0 := by omega

-- § shared-memory layout

/-- Halves reserved for staging. -/
def MAX_SHARED_HALVES : Nat := 4096

/-- Byte offset of the partials. -/
def partialByteOffset (stage : Bool) : Nat := 4 * (if stage then MAX_SHARED_HALVES / 2 else 0)

/-- Dynamic shared memory requested by the host. -/
def smemBytes (stage : Bool) (tpr : Nat) : Nat := (if stage then MAX_SHARED_HALVES * 2 else 0) + tpr * 4

/-- **No overlap**: staged halves occupy bytes `[0, 2·n_in) ⊆ [0, 8192)` (by the host's `n_in ≤ 4096`
check) and the partials start at byte 8192 and end at the requested size. -/
theorem staged_layout_disjoint (n_in tpr : Nat) (h : n_in ≤ MAX_SHARED_HALVES) :
    2 * n_in ≤ partialByteOffset true ∧ partialByteOffset true + 4 * tpr = smemBytes true tpr := by
  unfold partialByteOffset smemBytes MAX_SHARED_HALVES at *
  simp; omega

/-- Without staging the partials are the whole allocation. -/
theorem unstaged_layout (tpr : Nat) : partialByteOffset false = 0 ∧ smemBytes false tpr = 4 * tpr := by
  unfold partialByteOffset smemBytes; simp; omega

/-- Up to `tpr = 1024`, both configurations fit the default 48 KiB dynamic limit. -/
theorem smem_fits (tpr : Nat) (h : tpr ≤ 1024) (stage : Bool) : smemBytes stage tpr ≤ 49152 := by
  unfold smemBytes MAX_SHARED_HALVES; cases stage <;> simp <;> omega

/-- `n_embd = 2560` fits the staging limit; the host check refuses rows past 4096 halves. -/
theorem staging_examples : 2560 ≤ MAX_SHARED_HALVES ∧ ¬ (5120 ≤ MAX_SHARED_HALVES) := by decide

-- § host guard

/-- The guard `n_in % 4 = 0 ∧ n_in % 64 = 0` reduces to `% 64`, and it bounds the scale index. -/
theorem guard_bounds_group (n_in q : Nat) (h : n_in % 64 = 0) (hq : q < n_in / 4) :
    q >>> 4 < n_in / 64 ∧ n_in % 4 = 0 := by
  rw [Nat.shiftRight_eq_div_pow]; simp; omega

end Strata.S2Fast
