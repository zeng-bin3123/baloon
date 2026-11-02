/-!
# MODULE: spec_draft_controller
Index and bounds contracts of the speculative-decoding controller: cost-table indexing, the argmax-with-threshold
choice, and the acceptance-rate learning step.

## Source semantics
* `step_ms(k)` indexes the ratio tables with `n = clamp(k+1, 1, K_MAX+1)`, entry `n-1`.
* `choose` scans lookup draft lengths `1 .. min(available, K_MAX)` and MTP lengths `1 .. K_MAX`, keeping the
  candidate with the strictly largest tokens/ms, and falls back to "no draft" unless the best beats the baseline
  by `min_gain`.
* `observe(c, accepted)`: `accepted` is clamped to `[0, k]`; positions `0 .. min(k, accepted+1) - 1` are updated
  with `p += ema * (hit - p)`, hit = 1 for accepted positions and 0 for the rejected one.
* After a fully accepted window the untested positions `k .. K_MAX-1` move toward position `k-1`.

## Form template identifiers
CostModel.step_ms -> step_cost ; Controller.choose -> pick_draft ; Controller.observe -> learn_acceptance ;
mtp_p_ -> mtp_accept_rates ; lookup_q_ -> lookup_accept_rates ; K_MAX -> max_draft ; ema_ -> ema_rate
-/

namespace Strata.Spec.Controller

-- § cost-table indexing

/-- `clamp(k+1, 1, K+1)`. -/
def clampN (K k : Nat) : Nat := min (max (k + 1) 1) (K + 1)

/-- The table position `n` is in `[1, K+1]`. -/
theorem clampN_range (K k : Nat) : 1 ≤ clampN K k ∧ clampN K k ≤ K + 1 := by
  unfold clampN
  omega

/-- The table index `n - 1` is inside tables with `K + 1` entries. -/
theorem clampN_index_lt (K k : Nat) : clampN K k - 1 < K + 1 := by
  unfold clampN
  omega

/-- For draft lengths `k ≤ K` the clamp is the identity `k + 1`. -/
theorem clampN_exact (K k : Nat) (h : k ≤ K) : clampN K k = k + 1 := by
  unfold clampN
  omega

-- § candidate selection

/-- A strict-improvement scan never loses to any candidate and never loses to its start value. -/
theorem foldl_max_ge :
    ∀ (cs : List Nat) (init : Nat), init ≤ cs.foldl max init ∧ ∀ x ∈ cs, x ≤ cs.foldl max init
  | [], init => by simp
  | c :: cs, init => by
      have ih := foldl_max_ge cs (max init c)
      simp only [List.foldl_cons]
      refine ⟨Nat.le_trans (Nat.le_max_left _ _) ih.1, ?_⟩
      intro x hx
      rcases List.mem_cons.1 hx with rfl | hx
      · exact Nat.le_trans (Nat.le_max_right _ _) ih.1
      · exact ih.2 x hx

/-- Lookup draft lengths considered never exceed `K_MAX`. -/
theorem lookup_len_le (avail K k : Nat) (h : k ≤ min avail K) : k ≤ K := by omega

/-- Every considered length is at most the number of available lookup tokens. -/
theorem lookup_len_le_avail (avail K k : Nat) (h : k ≤ min avail K) : k ≤ avail := by omega

-- § observe

/-- Accepted count clamped to `[0, k]`. -/
def clampAccepted (a : Int) (k : Nat) : Int := max 0 (min a k)

/-- The clamped acceptance is a valid count. -/
theorem clampAccepted_range (a : Int) (k : Nat) : 0 ≤ clampAccepted a k ∧ clampAccepted a k ≤ k := by
  unfold clampAccepted
  omega

/-- Number of positions updated: the accepted ones plus the first rejected one. -/
def seen (k accepted : Nat) : Nat := min k (accepted + 1)

/-- The update loop never leaves the drafted window. -/
theorem seen_le (k accepted : Nat) : seen k accepted ≤ k := by
  unfold seen
  omega

/-- All accepted positions are updated (as hits). -/
theorem accepted_updated (k accepted : Nat) (h : accepted ≤ k) : accepted ≤ seen k accepted := by
  unfold seen
  omega

/-- A rejection is recorded at position `accepted` exactly when something was drafted beyond it. -/
theorem rejection_recorded (k accepted : Nat) (h : accepted < k) : seen k accepted = accepted + 1 := by
  unfold seen
  omega

/-- If everything was accepted, no rejected position is recorded. -/
theorem all_accepted_seen (k : Nat) : seen k k = k := by
  unfold seen
  omega

/-- The learning indices stay inside the K_MAX-entry rate array. -/
theorem observe_index_lt (K k accepted i : Nat) (hk : k ≤ K) (hi : i < seen k accepted) : i < K := by
  have := seen_le k accepted
  omega

/-- Extension of untested positions after a full window reads position `k-1`, which exists since `k ≥ 1`. -/
theorem extension_source (K k : Nat) (hk : 0 < k) (hKk : k ≤ K) : k - 1 < K := by omega

/-- Untested positions `k .. K-1` are strictly beyond the tested window. -/
theorem extension_range (K k i : Nat) (hk : 0 < k) (hi : k ≤ i) (hiK : i < K) : k - 1 < i := by omega

-- § EMA step (scaled integer model: probabilities are numerators over S)

/-- Scaled EMA: `p' * S = p * (S - a) + h * a` where `a` is the rate and `p, h` are probabilities. -/
def emaScaled (S p a h : Int) : Int := p * (S - a) + h * a

/-- The EMA update is a convex combination: it never leaves the probability range `[0, S]`
(after dividing the scaled value by `S`). -/
theorem emaScaled_range (S p a h : Int) (hp0 : 0 ≤ p) (hpS : p ≤ S) (ha0 : 0 ≤ a) (haS : a ≤ S)
    (hh0 : 0 ≤ h) (hhS : h ≤ S) : 0 ≤ emaScaled S p a h ∧ emaScaled S p a h ≤ S * S := by
  unfold emaScaled
  have hSa : 0 ≤ S - a := by omega
  have h1 : 0 ≤ p * (S - a) := Int.mul_nonneg hp0 hSa
  have h2 : 0 ≤ h * a := Int.mul_nonneg hh0 ha0
  have h3 : p * (S - a) ≤ S * (S - a) := Int.mul_le_mul_of_nonneg_right hpS hSa
  have h4 : h * a ≤ S * a := Int.mul_le_mul_of_nonneg_right hhS ha0
  have h5 : S * (S - a) = S * S - S * a := Int.mul_sub S S a
  omega

/-- A hit (`h = S`) never decreases the rate. -/
theorem ema_hit_monotone (S p a : Int) (hp0 : 0 ≤ p) (hpS : p ≤ S) (ha0 : 0 ≤ a) (haS : a ≤ S) :
    p * S ≤ emaScaled S p a S := by
  unfold emaScaled
  have h1 : p * (S - a) + p * a = p * S := by rw [← Int.mul_add]; congr 1; omega
  have h2 : p * a ≤ S * a := Int.mul_le_mul_of_nonneg_right hpS ha0
  omega

/-- A miss (`h = 0`) never increases the rate. -/
theorem ema_miss_monotone (S p a : Int) (hp0 : 0 ≤ p) (ha0 : 0 ≤ a) (haS : a ≤ S) :
    emaScaled S p a 0 ≤ p * S := by
  unfold emaScaled
  have h1 : p * (S - a) ≤ p * S := by
    have : p * (S - a) + p * a = p * S := by rw [← Int.mul_add]; congr 1; omega
    have h2 : 0 ≤ p * a := Int.mul_nonneg hp0 ha0
    omega
  omega

end Strata.Spec.Controller
