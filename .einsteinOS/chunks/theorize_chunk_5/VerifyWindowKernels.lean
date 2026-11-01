/-!
# MODULE: verify_window_kernels
Index and compaction contracts of the verify-window kernels (`verify_kernels.cu`).

## Source semantics
`ident_hits` makes slot=id, dst=index, count=n. `hit_select*` compact resident entries in entry order
(ballot + popcount prefix). `window_ids` yields a sliding window of width `min(n_kv, window)`.
`dense_steps` emits `[c, c+1, (c+1)/4, c+1]`. `rebase_ptrs` places blob k at `base + k*bytes`.

## Form template identifiers
compact -> ballot compaction; windowStart/windowWidth -> window_ids; rebase -> rebase_ptrs.
-/

namespace Strata.VerifyWindowKernels

-- § ballot compaction

/-- Resident entries compacted in entry order. -/
def compact (slots : List (Option Nat)) : List Nat := slots.filterMap id

theorem compact_length_le (s : List (Option Nat)) : (compact s).length ≤ s.length := by
  unfold compact
  exact List.length_filterMap_le _ _

/-- Compaction distributes over concatenation, which is how the four-warp prefix works. -/
theorem compact_append (a b : List (Option Nat)) : compact (a ++ b) = compact a ++ compact b := by
  unfold compact
  exact List.filterMap_append

theorem compact_nil : compact [] = [] := rfl

theorem compact_none (s : List (Option Nat)) : compact (none :: s) = compact s := by
  simp [compact]

theorem compact_some (n : Nat) (s : List (Option Nat)) : compact (some n :: s) = n :: compact s := by
  simp [compact]

-- § sliding window

def windowStart (nkv window : Nat) : Nat := if nkv > window then nkv - window else 0
def windowWidth (nkv window : Nat) : Nat := nkv - windowStart nkv window

theorem width_le_window (nkv window : Nat) : windowWidth nkv window ≤ window := by
  unfold windowWidth windowStart
  split <;> omega

theorem width_le_kv (nkv window : Nat) : windowWidth nkv window ≤ nkv := by
  unfold windowWidth windowStart
  split <;> omega

theorem start_plus_width (nkv window : Nat) :
    windowStart nkv window + windowWidth nkv window = nkv := by
  unfold windowWidth windowStart
  split <;> omega

-- § dense steps

def denseSteps (c : Nat) : List Nat := [c, c + 1, (c + 1) / 4, c + 1]

theorem denseSteps_length (c : Nat) : (denseSteps c).length = 4 := rfl

theorem denseSteps_kv_eq_width (c : Nat) : (denseSteps c)[1]? = (denseSteps c)[3]? := rfl

-- § pointer rebase

def rebase (base bytes k : Nat) : Nat := base + k * bytes

theorem rebase_strict_mono (base bytes a b : Nat) (hb : 0 < bytes) (h : a < b) :
    rebase base bytes a < rebase base bytes b := by
  unfold rebase
  have := Nat.mul_lt_mul_of_pos_right h hb
  omega

theorem rebase_zero (base bytes : Nat) : rebase base bytes 0 = base := by
  simp [rebase]

-- § conv history commit

/-- Last three of `[hist(3) | x_0 .. x_{n-1}]` when `n ≥ 3` are the last three xs. -/
theorem commit_src_index (n j : Nat) (hn : 3 ≤ n) (hj : j < 3) : ¬ (n + j < 3) := by
  omega

end Strata.VerifyWindowKernels
