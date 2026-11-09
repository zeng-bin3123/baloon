/-!
# MODULE: qsa_block_select
SOURCES: src/kernels/cuda/qsa_select.cu
Batched block scoring (one warp per pooled block) and a block-granular radix-select top-k that expands the chosen
blocks into cell ids (4 cells per block).

## Source semantics
* A block b < n_bid weighs R = 4 cells; the last block n_bid weighs n_kv - 4 * n_bid (possibly 0).
* If n_kv <= width, the selection is the identity.
* Radix select fixes the 32-bit key 8 bits at a time; the output takes cells above the threshold plus the first
  `eq_budget` equal cells in index order.

## Form template identifiers
- `R` = 4, `weight`, `n_bid`, `eq_budget`, `my_eq`
- `per` : blocks per thread, ceil(nb / 256)
-/

namespace Strata.QsaSelect

-- § block weights

/-- Weight (cell count) of block b. -/
def weight (b nbid nkv : Nat) : Nat := if b < nbid then 4 else nkv - nbid * 4

/-- Weights of blocks 0..n_bid add up to n_kv. -/
theorem total_weight (nkv nbid : Nat) (h : nbid = nkv / 4) : nbid * 4 + weight nbid nbid nkv = nkv := by
  unfold weight; rw [if_neg (Nat.lt_irrefl _)]; omega

/-- The tail block holds fewer than 4 cells. -/
theorem tail_weight_lt (nkv nbid : Nat) (h : nbid = nkv / 4) : weight nbid nbid nkv < 4 := by
  unfold weight; rw [if_neg (Nat.lt_irrefl _)]; omega

/-- Every emitted cell id b * 4 + c is a real cached cell. -/
theorem cell_id_lt (b c nbid nkv : Nat) (h : nbid = nkv / 4) (hb : b ≤ nbid)
    (hc : c < weight b nbid nkv) : b * 4 + c < nkv := by
  unfold weight at hc
  split at hc <;> omega

-- § radix digits

/-- A 32-bit key splits into its four 8-bit digits, the order the select consumes them. -/
theorem key_digits (k : Nat) (hk : k < 4294967296) :
    k = (k / 16777216 % 256) * 16777216 + (k / 65536 % 256) * 65536 + (k / 256 % 256) * 256 + k % 256 := by
  omega

/-- The top digit of a 32-bit key is a valid histogram bin. -/
theorem top_digit_lt (k : Nat) (hk : k < 4294967296) : k / 16777216 < 256 := by
  omega

-- § equal-cell distribution

/-- A thread's share of equal-to-threshold cells. -/
def myEq (budget before eq : Nat) : Nat := min eq (budget - before)

/-- The running equal-cell total never passes the budget when it started below it. -/
theorem myEq_within (budget before eq : Nat) (h : before ≤ budget) : before + myEq budget before eq ≤ budget := by
  unfold myEq; omega

/-- A thread entirely below the budget keeps all its equal cells. -/
theorem myEq_full (budget before eq : Nat) (h : before + eq ≤ budget) : myEq budget before eq = eq := by
  unfold myEq; omega

/-- A thread beyond the budget keeps none. -/
theorem myEq_none (budget before eq : Nat) (h : budget ≤ before) : myEq budget before eq = 0 := by
  unfold myEq; omega

/-- When the bracket gt <= width <= gt + eq holds, gt + myEq selects exactly width cells. -/
theorem select_exact (gt eq width : Nat) (h1 : gt ≤ width) (h2 : width ≤ gt + eq) :
    gt + myEq width gt eq = width := by
  unfold myEq; omega

-- § thread partition of blocks

/-- Blocks per thread. -/
def per (nb : Nat) : Nat := (nb + 255) / 256

/-- A nonempty block range gives a positive per-thread share. -/
theorem per_pos (nb : Nat) (h : 0 < nb) : 0 < per nb := by
  unfold per; omega

/-- 256 threads cover all nb blocks. -/
theorem threads_cover (nb : Nat) : nb ≤ per nb * 256 := by
  unfold per; omega

/-- Every block index has an owning thread below 256. -/
theorem owner_lt (b nb : Nat) (hb : b < nb) : b / per nb < 256 := by
  apply Nat.div_lt_of_lt_mul
  have := threads_cover nb
  omega

/-- The owner's window [t * per, t * per + per) contains the block. -/
theorem owner_window (b p : Nat) (hp : 0 < p) : p * (b / p) ≤ b ∧ b < p * (b / p) + p := by
  have h1 := Nat.div_add_mod b p
  have h2 := Nat.mod_lt b hp
  omega

-- § launch geometry

/-- Eight warps per block cover every block index below max_blocks. -/
theorem score_grid_covers (b maxb : Nat) (h : b < maxb) : b / 8 < (maxb + 7) / 8 := by
  omega

/-- Mixed-radix index bound. -/
theorem idx_lt (a b n m : Nat) (ha : a < n) (hb : b < m) : a * m + b < n * m := by
  have h1 : (a + 1) * m ≤ n * m := Nat.mul_le_mul_right m ha
  have h2 : (a + 1) * m = a * m + m := by rw [Nat.add_mul, Nat.one_mul]
  omega

/-- The score output index qi * max_blocks + b is inside the batched score buffer. -/
theorem score_out_lt (qi b nq maxb : Nat) (hq : qi < nq) (hb : b < maxb) : qi * maxb + b < nq * maxb :=
  idx_lt qi b nq maxb hq hb

end Strata.QsaSelect
