/-!
# MODULE: spec_controller
Speculative-decoding controller: picks the drafter and window size maximizing tokens per millisecond, keeps plain
decoding unless the gain clears a threshold, and learns per-position acceptance with an EMA.

## Source semantics
* The verify pass always commits one token, so expected tokens is `1 + sum of cumulative acceptance products`.
* `step_ms` clamps `n = k + 1` into `[1, K_MAX + 1]` before indexing the 9-entry ratio tables.
* The candidate with the highest rate wins; plain decoding is the baseline.
* If the best rate does not beat `baseline * (1 + min_gain)` the controller returns plain decoding.
* `observe` updates positions `0..min(k, accepted+1)-1`; all indices stay below `K_MAX`.

## Form template identifiers
* `Controller::expected` -> `expAux`/`expected`, `bucket` -> `bucket`, `K_MAX` -> `kMax`
* probabilities are fixed-point per mille (1000 = 1.0)
-/

namespace Strata.SpecController

/-- Largest draft window. -/
def kMax : Nat := 8

-- § table indexing

/-- The clamped table position `n` of a draft count `k`. -/
def clampN (k : Nat) : Nat := max 1 (min (k + 1) (kMax + 1))

/-- The clamped count is in `[1, 9]`. -/
theorem clampN_range (k : Nat) : 1 ≤ clampN k ∧ clampN k ≤ kMax + 1 := by
  unfold clampN kMax
  omega

/-- The table index `n - 1` is below the 9-entry table length, so no out-of-bounds read. -/
theorem clampN_index_lt (k : Nat) : clampN k - 1 < kMax + 1 := by
  have := clampN_range k
  omega

-- § match-length buckets

/-- Lookup acceptance bucket by match length. -/
def bucket (m : Nat) : Nat := if 16 ≤ m then 3 else if 8 ≤ m then 2 else if 5 ≤ m then 1 else 0

/-- There are four buckets. -/
theorem bucket_lt (m : Nat) : bucket m < 4 := by
  unfold bucket
  by_cases h1 : 16 ≤ m <;> by_cases h2 : 8 ≤ m <;> by_cases h3 : 5 ≤ m <;> simp [h1, h2, h3]

/-- Longer matches never fall into a lower bucket. -/
theorem bucket_mono (a b : Nat) (h : a ≤ b) : bucket a ≤ bucket b := by
  unfold bucket
  by_cases h1 : 16 ≤ a <;> by_cases h2 : 8 ≤ a <;> by_cases h3 : 5 ≤ a <;>
    by_cases h4 : 16 ≤ b <;> by_cases h5 : 8 ≤ b <;> by_cases h6 : 5 ≤ b <;>
    simp [h1, h2, h3, h4, h5, h6] <;> omega

-- § expected tokens (fixed-point per mille)

/-- Sum of cumulative acceptance products over `k` positions starting at index `i` with running product `run`. -/
def expAux (p : Nat → Nat) (i run : Nat) : Nat → Nat
  | 0 => 0
  | k + 1 => let r := run * p i / 1000; r + expAux p (i + 1) r k

/-- Expected committed tokens: the always-committed one plus the accepted-draft terms. -/
def expected (p : Nat → Nat) (k : Nat) : Nat := 1000 + expAux p 0 1000 k

/-- Expected tokens is at least one token. -/
theorem expected_ge_one (p : Nat → Nat) (k : Nat) : 1000 ≤ expected p k := by
  unfold expected
  omega

/-- With probabilities at most one, expected tokens never exceed the window `k + 1`. -/
theorem expAux_le (p : Nat → Nat) (hp : ∀ i, p i ≤ 1000) :
    ∀ (k i run : Nat), run ≤ 1000 → expAux p i run k ≤ k * 1000
  | 0, i, run, h => by simp [expAux]
  | k + 1, i, run, h => by
      have h1 : run * p i ≤ 1000 * run := by
        rw [Nat.mul_comm 1000 run]
        exact Nat.mul_le_mul_left run (hp i)
      have h2 : run * p i / 1000 ≤ run := Nat.div_le_of_le_mul h1
      have ih := expAux_le p hp k (i + 1) (run * p i / 1000) (by omega)
      simp only [expAux]
      omega

/-- Expected tokens per step is at most `k + 1`. -/
theorem expected_le (p : Nat → Nat) (hp : ∀ i, p i ≤ 1000) (k : Nat) :
    expected p k ≤ (k + 1) * 1000 := by
  have := expAux_le p hp k 0 1000 (by omega)
  unfold expected
  omega

-- § choosing the best candidate

/-- Best rate: a running maximum starting at the plain-decoding baseline. -/
def best : Nat → List Nat → Nat
  | b, [] => b
  | b, c :: cs => best (max b c) cs

/-- The chosen rate is never below the baseline. -/
theorem best_ge_base : ∀ (cs : List Nat) (b : Nat), b ≤ best b cs
  | [], b => by simp [best]
  | c :: cs, b => by
      have := best_ge_base cs (max b c)
      simp only [best]
      omega

/-- The chosen rate is at least every candidate's rate. -/
theorem best_ge_mem : ∀ (cs : List Nat) (b c : Nat), c ∈ cs → c ≤ best b cs
  | [], b, c, h => by simp at h
  | d :: cs, b, c, h => by
      have h0 := best_ge_base cs (max b d)
      simp only [best]
      rcases List.mem_cons.mp h with h1 | h'
      · omega
      · exact best_ge_mem cs _ c h'

-- § min-gain gate

/-- The threshold gate in per-mille: keep speculation only when `rate * 1000 ≥ base * (1000 + gain)`. -/
def clearsGate (rate base gain : Nat) : Prop := base * (1000 + gain) ≤ rate * 1000

/-- Without a gain threshold, any rate at least the baseline clears the gate. -/
theorem clearsGate_zero (rate base : Nat) (h : base ≤ rate) : clearsGate rate base 0 := by
  unfold clearsGate
  omega

/-- A rate that clears the gate is at least the baseline. -/
theorem clearsGate_ge (rate base gain : Nat) (h : clearsGate rate base gain) : base ≤ rate := by
  unfold clearsGate at h
  have : base * 1000 ≤ base * (1000 + gain) := Nat.mul_le_mul_left base (by omega)
  omega

-- § observe

/-- Positions updated by `observe`: `min k (accepted + 1)` after clamping `accepted` into `[0, k]`. -/
def seen (k accepted : Nat) : Nat := min k (min accepted k + 1)

/-- Updated positions never exceed the window. -/
theorem seen_le_k (k a : Nat) : seen k a ≤ k := by
  unfold seen
  omega

/-- Every updated index is a valid MTP table index when the window is at most `K_MAX`. -/
theorem seen_index_lt (k a i : Nat) (hk : k ≤ kMax) (hi : i < seen k a) : i < kMax := by
  have := seen_le_k k a
  unfold kMax at *
  omega

/-- When all drafts were accepted, the extension loop starts at `k` and reads index `k - 1 ≥ 0`. -/
theorem extend_index_ok (k : Nat) (hk : 0 < k) (hlt : k < kMax) : k - 1 < kMax := by
  unfold kMax at *
  omega

/-- The lookup candidate count is clipped to the window limit. -/
theorem lookup_clip_le (avail : Nat) : min avail kMax ≤ kMax := by
  omega

end Strata.SpecController
