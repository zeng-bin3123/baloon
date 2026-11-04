/-!
# MODULE: router_sampler_selection
SOURCES: src/kernels/cuda/router_top10.cu, src/kernels/cuda/sampler.cu
Selection semantics of the MoE router and the sampler: the (value, index) argmax combine used by every shuffle
reduction (ties go to the lowest index), the Philox uniform range, penalty monotonicity, the history window,
top-p cut adjustment and the router's launch/shared-memory geometry.

## Source semantics
* A reduction candidate `b` replaces `a` iff `b.value > a.value` or equal value with smaller index.
* Philox: `(c0 >> 8) * 2^-24` is in [0, 1).
* Repeat penalty multiplies non-positive logits and divides positive ones, so it never raises a logit.
* The history window is the TAIL of the row: start = len - min(last_n, len).
* The router clamps the renormalisation denominator at 2^-14 so it is never zero.

## Form template identifiers
bv/best -> running_best_value/index ; sv/si -> warp_partials ; penalty_repeat -> repeat_penalty
keep_ids/keep_logit -> top_k_survivors ; s_taken -> taken_mask
-/

namespace Strata.RouterSampler

-- § the argmax combine

/-- Combine two (value, index) candidates exactly as the shuffle reduction does. -/
def comb (a b : Int × Nat) : Int × Nat :=
  if b.1 > a.1 ∨ (b.1 = a.1 ∧ b.2 < a.2) then b else a

/-- The combine is commutative, so the reduction order within a warp does not matter. -/
theorem comb_comm (a b : Int × Nat) : comb a b = comb b a := by
  obtain ⟨a1, a2⟩ := a
  obtain ⟨b1, b2⟩ := b
  unfold comb
  split <;> split <;> first | (exfalso; omega) | (apply Prod.ext <;> dsimp only <;> omega)

/-- If the challenger beats the incumbent, it wins. -/
theorem comb_pos (a b : Int × Nat) (h : b.1 > a.1 ∨ (b.1 = a.1 ∧ b.2 < a.2)) : comb a b = b := by
  unfold comb
  exact if_pos h

/-- Otherwise the incumbent stays. -/
theorem comb_neg (a b : Int × Nat) (h : ¬ (b.1 > a.1 ∨ (b.1 = a.1 ∧ b.2 < a.2))) : comb a b = a := by
  unfold comb
  exact if_neg h

/-- The combine is associative, so any reduction tree yields the same winner. -/
theorem comb_assoc (a b c : Int × Nat) : comb (comb a b) c = comb a (comb b c) := by
  by_cases h1 : b.1 > a.1 ∨ (b.1 = a.1 ∧ b.2 < a.2)
  · rw [comb_pos a b h1]
    by_cases h2 : c.1 > b.1 ∨ (c.1 = b.1 ∧ c.2 < b.2)
    · rw [comb_pos b c h2]
      by_cases h3 : c.1 > a.1 ∨ (c.1 = a.1 ∧ c.2 < a.2)
      · rw [comb_pos a c h3]
      · exfalso
        omega
    · rw [comb_neg b c h2, comb_pos a b h1]
  · rw [comb_neg a b h1]
    by_cases h2 : c.1 > b.1 ∨ (c.1 = b.1 ∧ c.2 < b.2)
    · rw [comb_pos b c h2]
    · rw [comb_neg b c h2, comb_neg a b h1]
      by_cases h3 : c.1 > a.1 ∨ (c.1 = a.1 ∧ c.2 < a.2)
      · exfalso
        omega
      · rw [comb_neg a c h3]

/-- On equal values the lower index wins, whichever side it arrives on. -/
theorem comb_tie (v : Int) (i j : Nat) (h : i < j) :
    comb (v, j) (v, i) = (v, i) ∧ comb (v, i) (v, j) = (v, i) := by
  constructor
  · exact comb_pos _ _ (Or.inr ⟨rfl, h⟩)
  · exact comb_neg _ _ (by dsimp only; omega)

/-- The greedy kernel answers 0 when every candidate was `-inf` (index `n_vocab` sentinel). -/
def greedyOut (wi n : Nat) : Nat := if wi < n then wi else 0

/-- The greedy output is always a valid token id. -/
theorem greedyOut_lt (wi n : Nat) (hn : 0 < n) : greedyOut wi n < n := by
  unfold greedyOut
  split <;> omega

-- § Philox uniform

/-- The 24-bit mantissa draw is strictly below 2^24, so the uniform is below 1. -/
theorem philox_lt_one (c0 : Nat) (h : c0 < 4294967296) : c0 / 256 < 16777216 := by omega

-- § penalties (integer model of the sign rule)

/-- The repeat penalty: multiply non-positive logits, divide positive ones. -/
def repeatPen (l r : Int) : Int := if l ≤ 0 then l * r else l / r

/-- A repeat penalty with factor at least one never raises a logit. -/
theorem repeatPen_le (l r : Int) (hr : 1 ≤ r) : repeatPen l r ≤ l := by
  unfold repeatPen
  split
  · rename_i h
    have h0 : 0 ≤ (-l) * (r - 1) := Int.mul_nonneg (by omega) (by omega)
    have h1 : (-l) * (r - 1) = -(l * r) + l := by
      rw [Int.mul_sub, Int.mul_one, Int.neg_mul]
      omega
    omega
  · rename_i h
    exact Int.ediv_le_self r (by omega)

/-- The presence term is a boolean: at most one regardless of the count. -/
theorem presence_bool (count : Nat) : (if count > 0 then 1 else 0 : Nat) ≤ 1 := by
  split <;> omega

/-- The history window is the tail: it starts inside the row and ends at its end. -/
theorem window_tail (plen hist : Nat) :
    let hlen := min plen hist
    hist - hlen + hlen = hist ∧ hist - hlen ≤ hist := by
  intro hlen
  simp only [hlen]
  omega

-- § top-p cut adjustment

/-- `if (cut < min_keep) cut = min(min_keep, n_keep)`. -/
def cutAdj (cut minKeep nKeep : Nat) : Nat :=
  if cut < minKeep then (if minKeep < nKeep then minKeep else nKeep) else cut

/-- The adjusted cut keeps at least one survivor and never exceeds the survivors. -/
theorem cutAdj_bounds (cut minKeep nKeep : Nat) (h1 : 1 ≤ cut) (h2 : cut ≤ nKeep) (h3 : 1 ≤ minKeep) :
    1 ≤ cutAdj cut minKeep nKeep ∧ cutAdj cut minKeep nKeep ≤ nKeep := by
  simp only [cutAdj]
  split_ifs <;> omega

-- § router launch geometry

/-- Threads per token block: `min(n_expert, 512)` rounded up to a whole warp. -/
def thr (n : Nat) : Nat := (min n 512 + 31) / 32 * 32

/-- The thread count is a warp multiple, covers the experts (up to 512), and needs at most 16 warp slots. -/
theorem thr_props (n : Nat) (hn : 0 < n) :
    32 ≤ thr n ∧ thr n ≤ 512 ∧ thr n % 32 = 0 ∧ min n 512 ≤ thr n ∧ (thr n + 31) / 32 ≤ 16 := by
  unfold thr
  omega

/-- Round up to 16 (the `taken` mask padding). -/
def pad16 (n : Nat) : Nat := (n + 15) / 16 * 16

/-- The double array starts 16-aligned and the float array 4-aligned after `8 n` bytes of doubles. -/
theorem smem_alignment (n : Nat) : pad16 n % 16 = 0 ∧ (pad16 n + 8 * n) % 4 = 0 ∧ n ≤ pad16 n := by
  unfold pad16
  omega

/-- Shared memory for the full 512-expert router. -/
theorem smem_512 : pad16 512 + 512 * 8 + 512 * 4 = 6656 := by decide

/-- The clamp keeps the renormalisation denominator positive. -/
theorem clamp_positive (s floor : Nat) (h : 0 < floor) : 0 < max s floor := by omega

/-- The router's selection order: each pass takes the lowest-index maximum, so the first `k` outputs are
the stable descending order (checked here at the level of the combine's tie rule above). -/
theorem pass_takes_lowest_on_tie (v : Int) (i j : Nat) (h : i < j) : (comb (v, i) (v, j)).2 = i := by
  have := (comb_tie v i j h).2
  rw [this]

end Strata.RouterSampler
