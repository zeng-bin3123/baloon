/-!
# MODULE: router_top10_selection
The MoE router softmax/top-k/renormalise kernel (`router_top10.cu`): the (value desc, index asc) order that
makes repeated block-wide argmax equal a stable descending argsort, the argmax scan and its tie rule, the
shuffle-tree combine, the shared-memory budget and launch shape, and the clamp constant.

## Source semantics
* `p = softmax(logits)`, `ids = stable argsort(-p)[:k]` with ties broken by ascending index,
  `w = gather(p, ids)`, `w / max(sum w, 2^-14)`.
* The old rank-by-counting loop and the new k-pass argmax agree because a strict `>` scan in ascending
  `e` keeps the lowest index on a tie, and the shuffle combine prefers the lower index on equal values.
* One block per token; `threads = roundup32(min(n_expert, 512))`; dynamic shared memory
  `align16(n) + 8n + 4n` bytes (taken mask, `n` doubles, `n` floats).

## Form template identifiers
- `s_p[e]` → `p e`; `bv/bi` → `(v, i)` pairs; `RT_MAX_THREADS` → `512`
- `n_expert` → `n`; `k` → `k`; `n_tokens` → `nTok`
-/

namespace Strata.Router

-- § The selection order

/-- `a` is strictly better than `b`: larger value, or equal value and smaller index. -/
def better (a b : Nat × Nat) : Prop := a.1 > b.1 ∨ (a.1 = b.1 ∧ a.2 < b.2)

instance (a b : Nat × Nat) : Decidable (better a b) := by
  unfold better; infer_instance

/-- The shuffle-tree combine: keep `mine` unless `other` is strictly better. -/
def combine (mine other : Nat × Nat) : Nat × Nat := if better other mine then other else mine

/-- `better` is irreflexive. -/
theorem better_irrefl (a : Nat × Nat) : ¬ better a a := by
  unfold better; omega

/-- `better` is asymmetric. -/
theorem better_asymm (a b : Nat × Nat) (h : better a b) : ¬ better b a := by
  unfold better at *; omega

/-- `better` is transitive, so the order is a strict order. -/
theorem better_trans (a b c : Nat × Nat) (h1 : better a b) (h2 : better b c) : better a c := by
  unfold better at *; omega

/-- Two distinct pairs are always comparable: the order is total, so argmax is unique. -/
theorem better_total (a b : Nat × Nat) (hne : a ≠ b) : better a b ∨ better b a := by
  by_contra hcon
  apply hne
  unfold better at hcon
  exact Prod.ext (by omega) (by omega)

/-- The combine returns one of its arguments. -/
theorem combine_mem (a b : Nat × Nat) : combine a b = a ∨ combine a b = b := by
  unfold combine
  by_cases h : better b a
  · right; rw [if_pos h]
  · left; rw [if_neg h]

/-- The combine result is not strictly worse than either argument. -/
theorem combine_not_worse (a b : Nat × Nat) : ¬ better a (combine a b) ∧ ¬ better b (combine a b) := by
  unfold combine
  by_cases h : better b a
  · rw [if_pos h]; unfold better at *; omega
  · rw [if_neg h]; unfold better at *; omega

/-- The combine is commutative on distinct pairs, so any warp-tree shape gives the same winner. -/
theorem combine_comm (a b : Nat × Nat) (hne : a ≠ b) : combine a b = combine b a := by
  unfold combine
  by_cases h1 : better b a
  · have h2 : ¬ better a b := better_asymm b a h1
    rw [if_pos h1, if_neg h2]
  · by_cases h2 : better a b
    · rw [if_neg h1, if_pos h2]
    · rcases better_total a b hne with h | h
      · exact absurd h h2
      · exact absurd h h1

/-- A real expert beats the sentinel index `n` at an equal value, so the sentinel never wins. -/
theorem sentinel_loses (v i n : Nat) (h : i < n) : better (v, i) (v, n) := by
  show v > v ∨ (v = v ∧ i < n)
  omega

/-- An empty lane (value `0` standing for `-inf` is below every real probability) loses on value. -/
theorem neg_inf_loses (v i j : Nat) (h : 0 < v) : better (v, i) (0, j) := by
  show v > 0 ∨ (v = 0 ∧ i < j)
  omega

-- § The ascending strict-greater scan

/-- One scan step: replace only on a strictly greater value. -/
def step (p : Nat → Nat) (acc : Nat × Nat) (e : Nat) : Nat × Nat :=
  if p e > acc.1 then (p e, e) else acc

/-- A scan over a list of candidate indices. -/
def scan (p : Nat → Nat) (init : Nat × Nat) (l : List Nat) : Nat × Nat := l.foldl (step p) init

/-- Equal values never replace the incumbent: the earlier (lower) index stays. -/
theorem step_tie (p : Nat → Nat) (acc : Nat × Nat) (e : Nat) (h : p e = acc.1) : step p acc e = acc := by
  unfold step
  rw [if_neg (by omega)]

/-- A strictly greater value replaces the incumbent. -/
theorem step_replace (p : Nat → Nat) (acc : Nat × Nat) (e : Nat) (h : p e > acc.1) :
    step p acc e = (p e, e) := by
  unfold step
  rw [if_pos h]

/-- A step never lowers the running maximum and absorbs the new value. -/
theorem step_ge (p : Nat → Nat) (acc : Nat × Nat) (e : Nat) :
    acc.1 ≤ (step p acc e).1 ∧ p e ≤ (step p acc e).1 := by
  unfold step
  by_cases h : p e > acc.1
  · rw [if_pos h]
    show acc.1 ≤ p e ∧ p e ≤ p e
    omega
  · rw [if_neg h]
    omega

/-- The scan's value is at least the initial value and at least every scanned value. -/
theorem scan_value (p : Nat → Nat) (l : List Nat) :
    ∀ init : Nat × Nat, init.1 ≤ (scan p init l).1 ∧ ∀ x ∈ l, p x ≤ (scan p init l).1 := by
  induction l with
  | nil => intro init; exact ⟨Nat.le_refl _, by simp⟩
  | cons y ys ih =>
    intro init
    have hs := step_ge p init y
    have hih := ih (step p init y)
    have hscan : scan p init (y :: ys) = scan p (step p init y) ys := rfl
    rw [hscan]
    refine ⟨by omega, ?_⟩
    intro x hx
    rcases List.mem_cons.mp hx with rfl | hx'
    · omega
    · exact hih.2 x hx'

-- § Clamp and launch shape

/-- `2^-14 = 6.103515625e-5`: numerator `5^14` over `10^14`, and `5^14 * 2^14 = 10^14`. -/
theorem clamp_constant : 5 ^ 14 * 2 ^ 14 = 10 ^ 14 ∧ 6103515625 = 5 ^ 14 := by decide

/-- Output slot of rank `i` for token `t`: `t*k + i < nTok*k`. -/
theorem out_slot_lt (t i k nTok : Nat) (ht : t < nTok) (hi : i < k) : t * k + i < nTok * k := by
  have : (t + 1) * k ≤ nTok * k := Nat.mul_le_mul_right k ht
  rw [Nat.add_mul] at this
  omega

/-- Threads per block. -/
def threads (n : Nat) : Nat := ((if n < 512 then n else 512) + 31) / 32 * 32

/-- At least one warp, a multiple of 32, at most 512. -/
theorem threads_spec (n : Nat) (h : 0 < n) : 32 ≤ threads n ∧ threads n % 32 = 0 ∧ threads n ≤ 512 := by
  unfold threads
  split_ifs <;> omega

/-- For 512 experts there are 512 threads, 16 warps: the 16-slot reduction scratch is exactly full. -/
theorem threads_512 : threads 512 = 512 ∧ 512 / 32 = 16 := by decide

/-- Warps never exceed the 16 reduction slots. -/
theorem warps_le_16 (n : Nat) (h : 0 < n) : threads n / 32 ≤ 16 := by
  have := threads_spec n h
  omega

/-- The taken mask is 16-aligned. -/
def taken (n : Nat) : Nat := (n + 15) / 16 * 16

/-- Dynamic shared memory: mask, `n` doubles, `n` floats. -/
def smem (n : Nat) : Nat := taken n + n * 8 + n * 4

/-- The doubles start 16-aligned (so 8-aligned) and the floats start 4-aligned. -/
theorem smem_alignment (n : Nat) : taken n % 16 = 0 ∧ (taken n + n * 8) % 4 = 0 := by
  unfold taken; omega

/-- The 512-expert budget. -/
theorem smem_512 : smem 512 = 6656 := by decide

/-- Up to 3500 experts fit in the 48 KiB default dynamic shared memory. -/
theorem smem_fits (n : Nat) (h : n ≤ 3500) : smem n ≤ 49152 := by
  unfold smem taken; omega

/-- **The host guard `n_expert ≤ 512*64 = 32768` does not imply the launch fits**: at 32768 the dynamic
shared request is 425984 bytes, far above 48 KiB. -/
theorem guard_insufficient_for_smem : 32768 ≤ 512 * 64 ∧ 49152 < smem 32768 := by decide

/-- All `k` outputs are written exactly when `k ≤ n` (the host does not check this). -/
theorem outputs_written (k n : Nat) : min k n = k ↔ k ≤ n := by omega

end Strata.Router
