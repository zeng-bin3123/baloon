/-!
# MODULE: gdn_step
Gated delta-net recurrence step, conv slide and launch geometry.

## Source semantics
Per (head, column): decay the state column by exp(gate), sk = sum_i s_i k_i, delta = (v - sk)*beta, s_i += k_i*delta, out = sum_i s_i q_i (updated state). q/k head pairing is modulo h_k.

## Form template identifiers
gdn_step_kernel -> delta_step, sumTo -> column_sum, dec -> decay, b -> beta
-/
namespace Strata.GdnStep

/-- Sum of f over indices below n. -/
def sumTo : Nat → (Nat → Int) → Int
  | 0, _ => 0
  | n + 1, f => sumTo n f + f n

-- § sum lemmas

/-- Pointwise equal summands give equal sums. -/
theorem sumTo_congr (n : Nat) (f g : Nat → Int) (h : ∀ i, f i = g i) : sumTo n f = sumTo n g := by have : f = g := funext h; rw [this]

/-- Sums add. -/
theorem sumTo_add (n : Nat) (f g : Nat → Int) : sumTo n (fun i => f i + g i) = sumTo n f + sumTo n g := by induction n with | zero => rfl | succ n ih => simp only [sumTo, ih]; omega

/-- Constants pull out of sums. -/
theorem sumTo_mul (n : Nat) (c : Int) (f : Nat → Int) : sumTo n (fun i => c * f i) = c * sumTo n f := by induction n with | zero => simp [sumTo] | succ n ih => simp only [sumTo, ih, Int.mul_add]

-- § recurrence

/-- Pass 1: decaying the state before contracting with k scales the contraction by the decay. -/
theorem decay_contract (n : Nat) (dec : Int) (s k : Nat → Int) : sumTo n (fun i => (s i * dec) * k i) = dec * sumTo n (fun i => s i * k i) := by rw [← sumTo_mul]; exact sumTo_congr n _ _ (fun i => by rw [Int.mul_comm (s i) dec, Int.mul_assoc])

/-- Pass 2: the readout against q splits into the decayed old readout plus delta times the k-q readout. -/
theorem readout_split (n : Nat) (dec d : Int) (s k q : Nat → Int) : sumTo n (fun i => (s i * dec + k i * d) * q i) = dec * sumTo n (fun i => s i * q i) + d * sumTo n (fun i => k i * q i) := by rw [← sumTo_mul, ← sumTo_mul, ← sumTo_add]; exact sumTo_congr n _ _ (fun i => by rw [Int.add_mul, Int.mul_comm (s i) dec, Int.mul_assoc dec, Int.mul_comm (k i) d, Int.mul_assoc d])

-- § head pairing

/-- Modulo head pairing stays inside the q/k heads. -/
theorem pair_lt (h h_k : Nat) (hk : 0 < h_k) : h % h_k < h_k := Nat.mod_lt _ hk

/-- Divisible h_v makes the pairing balanced: head h and h + h_k share a q/k head. -/
theorem pair_periodic (h h_k : Nat) (hk : 0 < h_k) : (h + h_k) % h_k = h % h_k := by simp

-- § conv slide

/-- Slide the history window: drop the oldest, append the newest. -/
def slide {α : Type} (st : List α) (x : α) : List α := st.tail ++ [x]

/-- The slide preserves a nonempty window's length. -/
theorem slide_length {α : Type} (st : List α) (x : α) (n : Nat) (h : st.length = n + 1) : (slide st x).length = n + 1 := by simp [slide, h]

-- § launch geometry

/-- With MAX_H = 1 the head grid is exactly h_v. -/
theorem head_grid (h_v : Nat) : (h_v + 1 - 1) / 1 = h_v := by omega

/-- The j grid covers all S columns. -/
theorem j_grid (S : Nat) : S ≤ (S + 31) / 32 * 32 := by omega

/-- The beta-gate grid covers all h_v scalars. -/
theorem beta_grid (n : Nat) : n ≤ (n + 127) / 128 * 128 := by omega

end Strata.GdnStep
