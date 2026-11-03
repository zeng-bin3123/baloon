/-!
# MODULE: s2_gemv_fast_lut
The experimental S2 GEMV (`s2_gemv_fast.cu`) that stages `x` in shared memory and expands each code byte
through a constant 256x4 table.  Both levers were measured neutral/negative; the kernel is kept as a
negative result.  The theory pins the table, the group index, the shared-memory layout and the strides.

## Source semantics
* `c_codes[b][k] = ((b >> 2k) & 3) - 1`: the four biased codes of byte `b` in element order.
* One byte holds four codes, so a quad index `q` addresses elements `4q..4q+3`; the group (64 elements)
  of quad `q` is `(4q) >> 6 = q >> 4`.
* Staged `x` occupies `MAX_SHARED_HALVES = 4096` halves = 2048 floats; the partials follow it.

## Form template identifiers
- `c_codes` → `lut`; `QK_S2` → `qk`; `MAX_SHARED_HALVES` → `maxHalves`
- `threads_per_row` → `tpr`; `n_in` → `n`
-/

namespace Strata.S2GemvFast

/-- Biased code `k` (0..3) of byte `b` as an integer: `((b / 4^k) % 4) - 1`, `4^k` given as a literal. -/
def lut0 (b : Nat) : Int := ((b % 4 : Nat) : Int) - 1
/-- Code 1 of the byte. -/
def lut1 (b : Nat) : Int := ((b / 4 % 4 : Nat) : Int) - 1
/-- Code 2 of the byte. -/
def lut2 (b : Nat) : Int := ((b / 16 % 4 : Nat) : Int) - 1
/-- Code 3 of the byte. -/
def lut3 (b : Nat) : Int := ((b / 64 % 4 : Nat) : Int) - 1

-- § The table

/-- Every table entry is in {-1, 0, 1, 2}. -/
theorem lut_range (b : Nat) :
    -1 ≤ lut0 b ∧ lut0 b ≤ 2 ∧ -1 ≤ lut1 b ∧ lut1 b ≤ 2 ∧
    -1 ≤ lut2 b ∧ lut2 b ≤ 2 ∧ -1 ≤ lut3 b ∧ lut3 b ≤ 2 := by
  unfold lut0 lut1 lut2 lut3
  omega

/-- The table is a faithful expansion: the four biased codes reassemble the byte. -/
theorem lut_reassembles (b : Nat) (hb : b < 256) :
    (lut0 b + 1) + 4 * (lut1 b + 1) + 16 * (lut2 b + 1) + 64 * (lut3 b + 1) = (b : Int) := by
  unfold lut0 lut1 lut2 lut3
  omega

/-- The table is injective: two bytes with equal expansions are equal. -/
theorem lut_injective (a b : Nat) (ha : a < 256) (hb : b < 256)
    (h0 : lut0 a = lut0 b) (h1 : lut1 a = lut1 b) (h2 : lut2 a = lut2 b) (h3 : lut3 a = lut3 b) : a = b := by
  have := lut_reassembles a ha
  have := lut_reassembles b hb
  omega

/-- The table has 256 rows of four floats: 4096 bytes of constant memory. -/
theorem lut_bytes : 256 * 4 * 4 = 4096 := by decide

-- § Quad addressing

/-- The S2 group index of quad `q` is `q >> 4`: `(4q)/64 = q/16`. -/
theorem quad_group (q : Nat) : (q * 4) / 64 = q / 16 := by omega

/-- Via the shift form used in the kernel. -/
theorem quad_group_shift (q : Nat) : (q * 4) >>> 6 = q >>> 4 := by
  rw [Nat.shiftRight_eq_div_pow, Nat.shiftRight_eq_div_pow]; omega

/-- A quad is entirely inside one 64-element group. -/
theorem quad_inside_group (q : Nat) : (q * 4 + 3) / 64 = (q * 4) / 64 := by omega

/-- `n_in % 64 = 0` gives whole quads and whole groups, with 16 quads per group. -/
theorem shape (n : Nat) (h : n % 64 = 0) : (n / 4) % 16 = 0 ∧ n / 64 * 64 = n ∧ n / 4 = 16 * (n / 64) := by omega

/-- The scale row stride is `n/64` and the code row stride is `n/4`. -/
theorem row_strides (n o : Nat) (h : n % 64 = 0) : o * (n / 4) = o * (16 * (n / 64)) := by
  have : n / 4 = 16 * (n / 64) := by omega
  rw [this]

-- § Shared-memory layout

/-- Staged `x` is 4096 halves = 8192 bytes. -/
theorem staged_bytes : 4096 * 2 = 8192 := by decide

/-- Partials start after `MAX_SHARED_HALVES/2` floats, i.e. exactly after the staged halves. -/
theorem partial_offset : (4096 / 2) * 4 = 4096 * 2 := by decide

/-- The staged and unstaged shared sizes. -/
def smemBytes (stage : Bool) (tpr : Nat) : Nat := (if stage then 4096 * 2 else 0) + tpr * 4

/-- At most 1024 threads per row: shared memory stays within 48 KiB in both modes. -/
theorem smem_fits (stage : Bool) (tpr : Nat) (h : tpr ≤ 1024) : smemBytes stage tpr ≤ 49152 := by
  unfold smemBytes
  split <;> omega

/-- `n_embd = 2560` fits the staging limit with room. -/
theorem n2560_fits : 2560 ≤ 4096 ∧ 2560 % 64 = 0 := by decide

/-- A staged `__half2` read at `q*4` is 4-byte aligned (offset `8q` bytes). -/
theorem half2_aligned (q : Nat) : (q * 4 * 2) % 4 = 0 := by omega

/-- A `uint2` read of `x + 4q` halves is 8-byte aligned. -/
theorem uint2_aligned (q : Nat) : (q * 4 * 2) % 8 = 0 := by omega

-- § Strided loops

/-- Quad `q` is visited by thread `q % tpr` in iteration `q / tpr`. -/
theorem quad_owner (q tpr : Nat) (h : 0 < tpr) : q = (q % tpr) + tpr * (q / tpr) := by
  have := Nat.mod_add_div q tpr
  omega

/-- The owner thread is below `tpr`. -/
theorem quad_owner_lt (q tpr : Nat) (h : 0 < tpr) : q % tpr < tpr := Nat.mod_lt q h

/-- The block-reduction tree halves a power of two down to 1 through `log2` steps: sum of steps equals `tpr - 1`. -/
theorem tree_steps : 512 + 256 + 128 + 64 + 32 + 16 + 8 + 4 + 2 + 1 = 1023 := by decide

end Strata.S2GemvFast
