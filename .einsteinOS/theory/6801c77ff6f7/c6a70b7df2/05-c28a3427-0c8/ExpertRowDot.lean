/-!
# MODULE: expert_row_dot
SOURCES: src/kernels/cpu/expert.cpp
Integer-level theory of the Q2_0 expert kernel: the code-offset identity behind the row dot, the int8
activation clamp, bounds that make the VNNI integer sums exact, the exact integer horizontal-sum tree,
the 8-block tail loop of the 512-bit row dot, mask arithmetic, and per-lane scale selection.

## Source semantics
* `sum (c-1) x = sum (c x) - sum x`: the weight-independent correction `hx` is subtracted once.
* Quantized activations lie in [-127,127]; a 32-element chunk sum is bounded by 32*127.
* VNNI integer partial sums are exact (codes 0..3 against int8 never get near int32 range).
* The 512-bit row dot walks the blocks in groups of at most 8 and covers every block exactly once.
* Float evaluation order is NOT modelled here; only the exact integer parts are.

## Form template identifiers
offDot → centered_code_dot, codeDot → raw_code_dot, clamp127 → act_clamp, chunksK → tail_groups,
mask16 → lane_mask, ntDispatch → token_count_dispatch.
-/

namespace Strata.Kernels.Cpu.ExpertRowDot

-- § the code-offset identity

/-- Dot of centered codes `(c-1)` with activations, over matching lists. -/
def offDot : List Int → List Int → Int
  | c :: cs, x :: xs => (c - 1) * x + offDot cs xs
  | _, _ => 0

/-- Dot of raw codes with activations, over matching lists. -/
def codeDot : List Int → List Int → Int
  | c :: cs, x :: xs => c * x + codeDot cs xs
  | _, _ => 0

/-- The Q2_0 identity: the centered dot equals the raw-code dot minus the activation sum. -/
theorem offDot_eq (cs xs : List Int) (h : cs.length = xs.length) :
    offDot cs xs = codeDot cs xs - xs.sum := by
  induction cs generalizing xs with
  | nil =>
    cases xs with
    | nil => simp [offDot, codeDot]
    | cons x xs => simp at h
  | cons c cs ih =>
    cases xs with
    | nil => simp at h
    | cons x xs =>
      have h' : cs.length = xs.length := by simpa using h
      have ih' := ih xs h'
      have hm : (c - 1) * x = c * x - x := by
        rw [Int.sub_mul, Int.one_mul]
      simp only [offDot, codeDot, List.sum_cons]
      rw [hm, ih']
      omega

/-- The same identity at chunk granularity: two chunk sums add to the block sum. -/
theorem two_chunk_split (a0 a1 s0 s1 : Int) :
    (a0 - s0) + (a1 - s1) = (a0 + a1) - (s0 + s1) := by
  omega

-- § the int8 activation clamp and its sums

/-- The clamp applied after rounding: [-127, 127]. -/
def clamp127 (v : Int) : Int := max (-127) (min 127 v)

/-- The clamp lands in range. -/
theorem clamp127_range (v : Int) : -127 ≤ clamp127 v ∧ clamp127 v ≤ 127 := by
  unfold clamp127
  omega

/-- The clamp is the identity on in-range values. -/
theorem clamp127_id (v : Int) (h1 : -127 ≤ v) (h2 : v ≤ 127) : clamp127 v = v := by
  unfold clamp127
  omega

/-- The clamp is idempotent. -/
theorem clamp127_idem (v : Int) : clamp127 (clamp127 v) = clamp127 v := by
  have h := clamp127_range v
  exact clamp127_id _ h.1 h.2

/-- A list of in-range int8 activations has a sum bounded by 127 times its length. -/
theorem sum_bound (l : List Int) (h : ∀ x ∈ l, -127 ≤ x ∧ x ≤ 127) :
    -(127 * (l.length : Int)) ≤ l.sum ∧ l.sum ≤ 127 * (l.length : Int) := by
  induction l with
  | nil => simp
  | cons x xs ih =>
    have hx := h x (by simp)
    have ih' := ih (fun y hy => h y (by simp [hy]))
    simp only [List.sum_cons, List.length_cons]
    omega

-- § exactness of the integer partial sums

/-- A code (0..3) times an int8 activation is bounded by 381 in magnitude. -/
theorem code_mul_bound (c q : Int) (hc0 : 0 ≤ c) (hc1 : c ≤ 3) (hq0 : -127 ≤ q) (hq1 : q ≤ 127) :
    -381 ≤ c * q ∧ c * q ≤ 381 := by
  have hc : c = 0 ∨ c = 1 ∨ c = 2 ∨ c = 3 := by omega
  rcases hc with rfl | rfl | rfl | rfl <;> omega

/-- The exact integer horizontal sum of eight lanes: the `add`/`hadd`/`hadd` tree equals the plain sum. -/
theorem hsum_tree (a b c d e f g h : Int) :
    ((a + e) + (b + f)) + ((c + g) + (d + h)) = a + b + c + d + e + f + g + h := by
  omega

/-- Removing the code offset before float conversion is an exact integer subtraction. -/
theorem oracle_chunk_offset (dot sum : Int) : (dot - sum) + sum = dot := by
  omega

-- § scale selection per lane

/-- A VNNI int32 lane covers four consecutive activations; those four share one 32-chunk, hence one scale. -/
theorem lane_chunk (l e : Nat) (hl : l < 16) (he : e < 4) : (4 * l + e) / 32 = l / 8 := by
  omega

/-- The permute index `base[lane] + 2i` stays inside the 16-lane scale-product vector. -/
theorem permute_index_lt (l i : Nat) (hl : l < 16) (hi : i < 8) : l / 8 + 2 * i < 16 := by
  omega

/-- Block b, half h is activation chunk 2b+h, and chunk c belongs to block c/2. -/
theorem chunk_block (b h : Nat) (hh : h < 2) : (2 * b + h) / 2 = b := by
  omega

/-- The int8 offset of chunk 2b+h equals block offset b*64 plus half offset h*32. -/
theorem chunk_offset (b h : Nat) : (2 * b + h) * 32 = b * 64 + h * 32 := by
  omega

-- § the 8-block tail loop

/-- Group sizes produced by walking `n` items in groups of at most `k` (fuel-bounded). -/
def chunksK (k : Nat) : Nat → Nat → List Nat
  | 0, _ => []
  | f + 1, n => if n = 0 then [] else min k n :: chunksK k f (n - min k n)

/-- With enough fuel the groups cover exactly `n` items. -/
theorem chunksK_sum (k : Nat) (f : Nat) :
    ∀ n : Nat, n ≤ k * f → (chunksK k f n).sum = n := by
  induction f with
  | zero =>
    intro n hn
    have h0 : n = 0 := by simpa using hn
    simp [chunksK, h0]
  | succ f ih =>
    intro n hn
    by_cases h0 : n = 0
    · rw [chunksK, if_pos h0]
      simp [h0]
    · rw [chunksK, if_neg h0, List.sum_cons]
      have e : k * (f + 1) = k * f + k := by rw [Nat.mul_add, Nat.mul_one]
      rw [e] at hn
      have := ih (n - min k n) (by omega)
      rw [this]
      omega

/-- Every group is non-empty (when `k > 0`) and at most `k`. -/
theorem chunksK_bounds (k : Nat) (hk : 0 < k) (f : Nat) :
    ∀ n : Nat, ∀ x ∈ chunksK k f n, 0 < x ∧ x ≤ k := by
  induction f with
  | zero =>
    intro n x hx
    simp [chunksK] at hx
  | succ f ih =>
    intro n x hx
    by_cases h0 : n = 0
    · rw [chunksK, if_pos h0] at hx
      simp at hx
    · rw [chunksK, if_neg h0] at hx
      rcases List.mem_cons.mp hx with h | h
      · rw [h]
        omega
      · exact ih _ x h

/-- Walking `n` blocks in groups of 8 with fuel `n` covers all of them. -/
theorem blocks_cover (n : Nat) : (chunksK 8 n n).sum = n := by
  apply chunksK_sum
  have := Nat.mul_le_mul_right n (show 1 ≤ 8 by decide)
  omega

-- § lane masks

/-- The mask of the valid scale lanes for `nb` blocks: all 16 lanes for a full group, else the low `2*nb` lanes. -/
def mask16 (nb : Nat) : Nat := if 8 ≤ nb then 65535 else 2 ^ (2 * nb) - 1

/-- A full group enables all 16 lanes. -/
theorem mask16_full (nb : Nat) (h : 8 ≤ nb) : mask16 nb = 65535 := by
  unfold mask16
  rw [if_pos h]

/-- The mask always fits in 16 bits. -/
theorem mask16_lt (nb : Nat) (h : nb ≤ 8) : mask16 nb < 65536 := by
  by_cases h8 : 8 ≤ nb
  · rw [mask16_full nb h8]
    decide
  · have h' : nb = 0 ∨ nb = 1 ∨ nb = 2 ∨ nb = 3 ∨ nb = 4 ∨ nb = 5 ∨ nb = 6 ∨ nb = 7 := by omega
    rcases h' with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl <;> decide

/-- A partial group of `nb < 8` blocks enables exactly the low `2*nb` lanes. -/
theorem mask16_partial (nb : Nat) (h : nb < 8) : mask16 nb + 1 = 2 ^ (2 * nb) := by
  have h' : nb = 0 ∨ nb = 1 ∨ nb = 2 ∨ nb = 3 ∨ nb = 4 ∨ nb = 5 ∨ nb = 6 ∨ nb = 7 := by omega
  rcases h' with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl <;> decide

-- § token-count dispatch

/-- The `switch (n_tokens)` template instance chosen: 1..7 exactly, everything else the 8-wide one. -/
def ntDispatch (n : Nat) : Nat := if 1 ≤ n ∧ n ≤ 7 then n else 8

/-- The dispatched template width is between 1 and 8. -/
theorem ntDispatch_range (n : Nat) : 1 ≤ ntDispatch n ∧ ntDispatch n ≤ 8 := by
  unfold ntDispatch
  split <;> omega

/-- For 1..7 tokens the instance width equals the token count (no padding work). -/
theorem ntDispatch_exact (n : Nat) (h1 : 1 ≤ n) (h2 : n ≤ 7) : ntDispatch n = n := by
  unfold ntDispatch
  have hc : 1 ≤ n ∧ n ≤ 7 := by omega
  rw [if_pos hc]

/-- For a verify window (at most 8 tokens) the instance is never narrower than the token count. -/
theorem ntDispatch_ge (n : Nat) (h1 : 1 ≤ n) (h2 : n ≤ 8) : n ≤ ntDispatch n := by
  unfold ntDispatch
  split <;> omega

end Strata.Kernels.Cpu.ExpertRowDot
