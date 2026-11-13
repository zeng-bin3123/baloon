/-!
# MODULE: router_top_k_selection
SOURCES: src/kernels/cuda/router_top10.cu
Model of the MoE router's softmax / top-k / renormalise half: the stable descending order
(ties by ascending index), the block-wide argmax combine used by every pass, the ggml lower
clamp of the renormaliser, and the launch / shared-memory geometry.

## Source semantics
* `p = softmax(logits)` over ALL experts; `ids` = stable argsort of `-p`, first `k`;
  `w = p[ids] / max(Σ p[ids], 2⁻¹⁴)`.
* Each of the `k` passes is a block argmax over untaken experts; two candidates combine by
  "larger value, then smaller index", so the reduction tree's shape cannot change the winner.
* `threads = roundup32(min(n_expert, 512))`; shared memory is `align16(n) + 8n + 4n` bytes.
* `k ≤ 64` and `n_expert ≤ 512·64` are refused with an exit, not truncated.

## Form template identifiers
`(ov > bv || (ov == bv && oi < bi))` → `better`/`pick`; `fmax(s, 6.103515625e-05)` → `clamp`;
launch width → `threads`; dynamic shared memory → `smem`.
-/

namespace RouterTopKSelection

-- § the selection order

/-- `x` beats `y`: larger probability, ties broken by smaller expert index. -/
def better (p : Nat → Int) (x y : Nat) : Prop := p x > p y ∨ (p x = p y ∧ x < y)

instance (p : Nat → Int) (x y : Nat) : Decidable (better p x y) := by
  unfold better; infer_instance

/-- `better` is irreflexive. -/
theorem better_irrefl (p : Nat → Int) (x : Nat) : ¬ better p x x := by
  unfold better; omega

/-- `better` is transitive. -/
theorem better_trans (p : Nat → Int) {x y z : Nat} (h1 : better p x y) (h2 : better p y z) :
    better p x z := by
  unfold better at *; omega

/-- `better` is total on distinct experts: the order is a strict total order, i.e. a stable
    descending argsort. -/
theorem better_total (p : Nat → Int) {x y : Nat} (h : x ≠ y) : better p x y ∨ better p y x := by
  unfold better; omega

/-- The reduction step: keep the candidate that `better` prefers. -/
def pick (p : Nat → Int) (a b : Nat) : Nat := if better p b a then b else a

/-- The result is one of the two candidates. -/
theorem pick_mem (p : Nat → Int) (a b : Nat) : pick p a b = a ∨ pick p a b = b := by
  unfold pick; split <;> simp

/-- The result is not beaten by either candidate. -/
theorem pick_max (p : Nat → Int) (a b : Nat) :
    ¬ better p a (pick p a b) ∧ ¬ better p b (pick p a b) := by
  unfold pick; split <;> rename_i h <;> unfold better at * <;> omega

/-- "Not beaten by" chains: if `a` does not beat `r` and `r` does not beat `w`, then `a` does
    not beat `w` (the complement of a strict total order is transitive). -/
theorem notBetter_trans (p : Nat → Int) {a r w : Nat} (h1 : ¬ better p a r)
    (h2 : ¬ better p r w) : ¬ better p a w := by
  unfold better at *; omega

/-- The step is commutative on distinct candidates (the shuffle partner order is irrelevant). -/
theorem pick_comm (p : Nat → Int) (a b : Nat) : pick p a b = pick p b a := by
  unfold pick better
  by_cases h1 : (p b > p a ∨ p b = p a ∧ b < a) <;>
    by_cases h2 : (p a > p b ∨ p a = p b ∧ a < b) <;> simp only [h1, h2, ite_true, ite_false] <;>
    omega

/-- The step is associative, so any reduction tree (warp shuffles, then the warp-0 pass over
    per-warp winners) selects the same expert as the serial ascending scan. -/
theorem pick_assoc (p : Nat → Int) (a b c : Nat) :
    pick p (pick p a b) c = pick p a (pick p b c) := by
  unfold pick
  by_cases h1 : better p b a <;> by_cases h2 : better p c b <;> by_cases h3 : better p c a <;>
    simp only [h1, h2, h3, ite_true, ite_false] <;> unfold better at * <;> omega

/-- Serial left-to-right scan from an initial candidate. -/
def scan (p : Nat → Int) (init : Nat) (l : List Nat) : Nat := l.foldl (pick p) init

/-- The scan's winner is never beaten by any scanned candidate nor by the initial one. -/
theorem scan_max (p : Nat → Int) : ∀ (l : List Nat) (init : Nat),
    ¬ better p init (scan p init l) ∧ ∀ x, x ∈ l → ¬ better p x (scan p init l)
  | [], init => by simp [scan, better_irrefl]
  | y :: ys, init => by
    have ih := scan_max p ys (pick p init y)
    have hm := pick_max p init y
    have e : scan p init (y :: ys) = scan p (pick p init y) ys := rfl
    rw [e]
    refine ⟨notBetter_trans p hm.1 ih.1, ?_⟩
    intro x hx
    simp only [List.mem_cons] at hx
    rcases hx with rfl | hx
    · exact notBetter_trans p hm.2 ih.1
    · exact ih.2 x hx

-- § renormalisation clamp

/-- The literal `6.103515625e-05` is exactly `2⁻¹⁴`. -/
theorem clamp_literal : 6103515625 * 2^14 = 10^14 := by decide

/-- `fmax(s, c)` with `c > 0` is positive, so the renormalising division is always defined,
    and it is the identity clamp whenever the selected mass already exceeds `c`. -/
theorem clamp_pos (s c : Int) (hc : 0 < c) : 0 < max s c ∧ (c ≤ s → max s c = s) := by
  constructor
  · exact Int.lt_of_lt_of_le hc (Int.le_max_right s c)
  · intro h; exact Int.max_eq_left h

-- § launch geometry

/-- Threads per token: `min(n, 512)` rounded up to whole warps. -/
def threads (n : Nat) : Nat := (min n 512 + 31) / 32 * 32

/-- At least one full warp, at most 512 threads, always whole warps. -/
theorem threads_spec (n : Nat) (h : 0 < n) :
    32 ≤ threads n ∧ threads n ≤ 512 ∧ threads n % 32 = 0 := by
  unfold threads; omega

/-- Warp partial slots: at most `512 / 32 = 16`, the size of `s_red` / `s_rid`. -/
theorem warp_slots (n : Nat) : threads n / 32 ≤ 16 := by unfold threads; omega

/-- Dynamic shared memory: taken mask (16-aligned), `n` doubles, `n` floats. -/
def smem (n : Nat) : Nat := (n + 15) / 16 * 16 + 8 * n + 4 * n

/-- The doubles start 16-byte aligned (8-byte alignment needed) and the floats after them
    stay 4-byte aligned. -/
theorem smem_alignment (n : Nat) :
    ((n + 15) / 16 * 16) % 8 = 0 ∧ ((n + 15) / 16 * 16 + 8 * n) % 4 = 0 := by omega

/-- The model's geometry (512 experts) needs 6656 bytes, well inside the 48 KiB default; the
    `n_expert ≤ 512·64` guard admits sizes whose request exceeds 48 KiB, where the failure is
    caught by the post-launch error check rather than the guard. -/
theorem smem_bounds : smem 512 = 6656 ∧ smem 512 ≤ 49152 ∧ 49152 < smem (512 * 64) := by decide

/-- The k-pass argmax does `10 · 512` comparisons against the rank loop's `512²`: 51× less. -/
theorem selection_work : 10 * 512 = 5120 ∧ 512 * 512 = 262144 ∧ 262144 / 5120 = 51 := by decide

/-- Guards of the host wrapper. -/
def accepts (k nExpert : Nat) : Bool := k ≤ 64 && nExpert ≤ 512 * 64

/-- The production shape (10 of 512) is accepted. -/
theorem accepts_production : accepts 10 512 = true := by decide

end RouterTopKSelection
