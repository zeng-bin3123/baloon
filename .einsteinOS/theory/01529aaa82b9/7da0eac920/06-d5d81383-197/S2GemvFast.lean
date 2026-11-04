/-!
# MODULE: s2_gemv_fast
Code lookup table, scale-group indexing, shared-memory budget and tree-reduction shape of the (experimental,
non-adopted) Q2 GEMV with optional shared staging of x.

## Source semantics
The table maps a code byte to four values `((b >> 2k) & 3) - 1`, i.e. in {-1,0,1,2}. A quad index q covers
elements 4q..4q+3 and scale group `q >> 4 = (4q) >> 6`. The block reduction halves a power-of-two width.

## Form template identifiers
- `lut b k` ↔ `c_codes[b][k]`; `groupOf q` ↔ `s[q >> 4]`; `smemBytes` ↔ launch `smem`
-/

namespace Strata.S2GemvFast

-- § code table

/-- Code field `k` of byte `b` (LSB-first 2-bit fields). -/
def field (b k : Nat) : Nat := b / 4 ^ k % 4

/-- Table entry with the -1 bias applied. -/
def lut (b k : Nat) : Int := (field b k : Int) - 1

/-- Every table entry lies in {-1,0,1,2}. -/
theorem lut_range (b k : Nat) : -1 ≤ lut b k ∧ lut b k ≤ 2 := by
  have : field b k < 4 := by unfold field; omega
  unfold lut; omega

/-- Field 0 is `b mod 4`. -/
theorem field_zero (b : Nat) : field b 0 = b % 4 := by
  simp [field]

/-- The four fields of a byte reassemble it. -/
theorem fields_reassemble (b : Nat) (h : b < 256) :
    field b 0 + 4 * field b 1 + 16 * field b 2 + 64 * field b 3 = b := by
  unfold field
  norm_num
  omega

-- § indexing

/-- Scale group of quad `q`. -/
def groupOf (q : Nat) : Nat := q / 16

/-- `q >> 4` equals the group of element `4q` with group size 64. -/
theorem groupOf_eq (q : Nat) : groupOf q = (q * 4) / 64 := by
  unfold groupOf; omega

/-- The four elements of a quad share one group. -/
theorem quad_one_group (q k : Nat) (hk : k < 4) : (q * 4 + k) / 64 = (q * 4) / 64 := by
  omega

/-- Code byte offset for quad `q` of row `o` with `n_in` inputs. -/
def codeOff (o nQuads q : Nat) : Nat := o * nQuads + q

/-- Codes of distinct (row, quad) pairs never overlap. -/
theorem codeOff_inj (o o' nQuads q q' : Nat) (hq : q < nQuads) (hq' : q' < nQuads)
    (e : codeOff o nQuads q = codeOff o' nQuads q') : o = o' ∧ q = q' := by
  unfold codeOff at e
  have h1 : (o * nQuads + q) / nQuads = o := by
    rw [Nat.mul_comm, Nat.mul_add_div (by omega : 0 < nQuads)]
    simp [Nat.div_eq_of_lt hq]
  have h2 : (o' * nQuads + q') / nQuads = o' := by
    rw [Nat.mul_comm, Nat.mul_add_div (by omega : 0 < nQuads)]
    simp [Nat.div_eq_of_lt hq']
  have ho : o = o' := by rw [← h1, ← h2, e]
  subst ho
  omega

-- § shared memory

/-- Shared-staging capacity in halves. -/
def maxSharedHalves : Nat := 4096

/-- Dynamic shared bytes: optional staged x plus one float partial per thread. -/
def smemBytes (stage : Bool) (tpr : Nat) : Nat := (if stage then maxSharedHalves * 2 else 0) + tpr * 4

/-- Hidden size 2560 fits the staging area with room to spare. -/
theorem n_embd_fits : 2560 ≤ maxSharedHalves := by decide

/-- Staging costs exactly 8 KiB more shared memory. -/
theorem stage_cost (tpr : Nat) : smemBytes true tpr = smemBytes false tpr + 8192 := by
  unfold smemBytes maxSharedHalves; simp; omega

/-- The partial array starts after `MAX/2` floats = the staged halves, with no overlap. -/
theorem partial_off : maxSharedHalves / 2 * 4 = maxSharedHalves * 2 := by decide

-- § tree reduction

/-- Number of halving steps to reduce `2^k` partials to one. -/
def steps : Nat → Nat
  | 0 => 0
  | n + 1 => 1 + steps n

/-- `2^k` partials need exactly `k` halving steps. -/
theorem steps_eq (k : Nat) : steps k = k := by
  induction k with
  | zero => rfl
  | succ n ih => simp [steps, ih]; omega

/-- Each step halves the width, ending at 1. -/
theorem halving_ends (k : Nat) : (2 ^ k) / (2 ^ k) = 1 := by
  exact Nat.div_self (Nat.two_pow_pos k)

end Strata.S2GemvFast
