/-!
# MODULE: prefill_chunk_driver
Chunked batched prefill: tokens are processed T at a time through all layers; routed (token, k) pairs are
grouped by expert with a counting sort so each expert runs one GEMM over its rows; expert blobs stream through
a staging ring while a small ring holds dequantized experts.

## Source semantics
- The chunk at offset c0 has min(T, n - c0) tokens, so chunk sizes sum to n.
- The counting sort places pair i at slot off[e] + (rank within e) and records its source token i / K.
- Staging uses slot ring index mod STAGE; dequantized experts use j mod DQ.

## Form template identifiers
chunk loop -> chunks ; m.off -> prefixOffsets ; src_host[p] = i / K -> srcToken ;
stage_next -> ringNext
-/

namespace Strata.Prefill

-- § chunking

/-- Chunk sizes for `n` tokens with chunk `T`, using fuel to keep it structural. -/
def chunksAux : Nat → Nat → Nat → List Nat
  | 0, _, _ => []
  | f + 1, T, n => if n = 0 then [] else min T n :: chunksAux f T (n - min T n)

/-- Chunk sizes sum to the token count when there is enough fuel. -/
theorem chunksAux_sum (f T n : Nat) (hT : 0 < T) (hf : n ≤ f) : (chunksAux f T n).sum = n := by
  induction f generalizing n with
  | zero =>
    have : n = 0 := by omega
    subst this
    simp [chunksAux]
  | succ f ih =>
    unfold chunksAux
    by_cases hn : n = 0
    · simp [hn]
    · simp only [hn, if_false, List.sum_cons]
      have h1 : n - min T n ≤ f := by omega
      have := ih (n - min T n) h1
      omega

/-- Every chunk is nonempty and at most T. -/
theorem chunksAux_bounds (f T n : Nat) (hT : 0 < T) : ∀ c ∈ chunksAux f T n, 0 < c ∧ c ≤ T := by
  induction f generalizing n with
  | zero => simp [chunksAux]
  | succ f ih =>
    unfold chunksAux
    by_cases hn : n = 0
    · simp [hn]
    · intro c hc
      simp only [hn, if_false, List.mem_cons] at hc
      rcases hc with h | h
      · subst h; omega
      · exact ih _ c h

-- § counting sort

/-- Source token of (token, k) pair `i`. -/
def srcToken (K i : Nat) : Nat := i / K

/-- A pair index below T*K maps to a token below T. -/
theorem srcToken_lt (K T i : Nat) (h : i < K * T) : srcToken K i < T :=
  Nat.div_lt_of_lt_mul h

/-- A rank within an expert's group stays inside that group's span of the prefix array. -/
theorem slot_in_group (off cnt r : Nat) (h : r < cnt) : off + r < off + cnt := by omega

/-- Prefix offsets of expert counts. -/
def prefixOffsets : List Nat → List Nat
  | [] => [0]
  | c :: cs => 0 :: (prefixOffsets cs).map (· + c)

/-- The prefix array has one more entry than the counts. -/
theorem prefixOffsets_length (cs : List Nat) : (prefixOffsets cs).length = cs.length + 1 := by
  induction cs with
  | nil => simp [prefixOffsets]
  | cons c cs ih => simp [prefixOffsets, ih]

-- § rings

/-- Next ring slot. -/
def ringNext (size i : Nat) : Nat := (i + 1) % size

/-- Ring slots stay in range. -/
theorem ringNext_lt (size i : Nat) (h : 0 < size) : ringNext size i < size := Nat.mod_lt _ h

/-- Dequantized-expert ring index in range for DQ = 2. -/
theorem dq_index_lt (j : Nat) : j % 2 < 2 := Nat.mod_lt _ (by decide)

/-- Lookahead of STAGE-1 never reuses a slot still in flight: distance below STAGE. -/
theorem lookahead_distinct (stage j k : Nat) (hs : 0 < stage) (hj : j < k) (hk : k ≤ j + (stage - 1)) :
    ¬ (stage ∣ k - j) := by
  intro h
  obtain ⟨m, hm⟩ := h
  have : m = 0 ∨ 1 ≤ m := by omega
  rcases this with h0 | h1
  · subst h0; omega
  · have : stage * 1 ≤ stage * m := Nat.mul_le_mul_left _ h1
    omega

end Strata.Prefill
