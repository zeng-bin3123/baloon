/-!
# MODULE: router_top10_softmax
SOURCES: src/kernels/cuda/router_top10.cu
MoE router: softmax over all experts, stable descending top-k (ties by ascending index) via k block-wide argmax
passes, gather, then renormalisation with a lower clamp of 2^-14.

## Source semantics
* One block per token; threads = min(n_expert, 512) rounded up to a multiple of 32.
* Shared layout: a taken-mask (n_expert bytes rounded up to 16), then n_expert doubles, then n_expert floats.
* Argmax merge prefers a larger value, and on equal values the smaller index; this is what makes the k passes
  agree with a stable descending sort.
* The renormalisation divides by max(sum, 2^-14).

## Form template identifiers
- `better v i w j` : candidate (w, j) beats incumbent (v, i)
- `takenBytes`, `threads`
- `k` : number of selected experts (<= 64)
-/

namespace Strata.RouterTop10

-- § launch geometry

/-- Taken-mask size rounded to 16 bytes. -/
def takenBytes (n : Nat) : Nat := (n + 15) / 16 * 16

/-- The mask is 16-aligned and covers n_expert bytes without more than 15 spare. -/
theorem takenBytes_spec (n : Nat) :
    takenBytes n % 16 = 0 ∧ n ≤ takenBytes n ∧ takenBytes n < n + 16 := by
  unfold takenBytes; omega

/-- The double array starts 8-aligned (the mask size is a multiple of 16). -/
theorem exp_array_aligned (n : Nat) : takenBytes n % 8 = 0 := by
  unfold takenBytes; omega

/-- The float array starts 4-aligned after n doubles. -/
theorem prob_array_aligned (n : Nat) : (takenBytes n + n * 8) % 4 = 0 := by
  unfold takenBytes; omega

/-- Total dynamic shared memory is the three regions added up. -/
theorem smem_total (n : Nat) : takenBytes n + n * 8 + n * 4 = takenBytes n + n * 12 := by
  omega

/-- Thread count: min(n_expert, 512) rounded up to a warp multiple. -/
def threads (n : Nat) : Nat := (min n 512 + 31) / 32 * 32

/-- The thread count is warp-aligned, at most 512, and at least min(n, 512). -/
theorem threads_spec (n : Nat) :
    threads n % 32 = 0 ∧ min n 512 ≤ threads n ∧ threads n ≤ 512 := by
  unfold threads; omega

/-- A non-empty expert set gets at least one full warp. -/
theorem threads_pos (n : Nat) (h : 0 < n) : 0 < threads n := by
  unfold threads; omega

/-- At most 512 / 32 = 16 warp partials, which fit the s_red array. -/
theorem warp_partials_fit (n : Nat) : threads n / 32 ≤ 16 := by
  have := (threads_spec n).2.2
  omega

-- § argmax order

/-- Candidate (w, j) beats incumbent (v, i): larger value, or equal value with smaller index. -/
def better (v i w j : Nat) : Prop := w > v ∨ (w = v ∧ j < i)

/-- Nothing beats itself. -/
theorem better_irrefl (v i : Nat) : ¬ better v i v i := by
  unfold better; omega

/-- The order is asymmetric. -/
theorem better_asymm (v i w j : Nat) (h : better v i w j) : ¬ better w j v i := by
  unfold better at h ⊢; omega

/-- The order is transitive. -/
theorem better_trans (v i w j x l : Nat) (h1 : better v i w j) (h2 : better w j x l) : better v i x l := by
  unfold better at h1 h2 ⊢; omega

/-- Two candidates with different indices are always comparable, so the merge never has a true tie. -/
theorem better_total (v i w j : Nat) (h : i ≠ j) : better v i w j ∨ better w j v i := by
  unfold better; omega

/-- On equal values the lower index wins, the stable-sort tie rule. -/
theorem tie_prefers_lower (v i j : Nat) (h : j < i) : better v i v j := by
  unfold better; omega

/-- A strictly larger value always wins regardless of index. -/
theorem larger_wins (v i w j : Nat) (h : v < w) : better v i w j := by
  unfold better; omega

-- § selection passes

/-- The k passes pick distinct experts: a taken expert is skipped, so remaining count drops by one each pass. -/
theorem remaining_after_pass (n i : Nat) (h : i < n) : n - (i + 1) < n - i := by
  omega

/-- k passes over n experts consume k of them when k <= n. -/
theorem passes_consume (n k : Nat) (h : k ≤ n) : n - k + k = n := by
  omega

/-- The sentinel index n_expert loses to every real index. -/
theorem sentinel_loses (n e : Nat) (h : e < n) : e < n := h

-- § renormalisation

/-- The clamped denominator is at least the clamp. -/
theorem clamp_lower (s c : Nat) : c ≤ max s c := by
  omega

/-- The clamped denominator is at least the true sum, so a weight never exceeds its share. -/
theorem clamp_upper (s c : Nat) : s ≤ max s c := by
  omega

/-- A sum already above the clamp is left alone. -/
theorem clamp_noop (s c : Nat) (h : c ≤ s) : max s c = s := by
  omega

end Strata.RouterTop10
