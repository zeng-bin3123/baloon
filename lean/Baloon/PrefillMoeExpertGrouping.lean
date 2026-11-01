/-!
# MODULE: prefill_moe_expert_grouping
Host-side grouping of a prefill chunk's (token, k) routed pairs by expert, the chunk loop, and staging/dequant ring indices.

## Source semantics
* `cnt[e]` counts pairs routed to expert `e`; `off` is its exclusive prefix sum; `src[slot] = i / K`.
* A chunk of at most `T` tokens is taken at a time: `Tc = min(T, n - c0)`.
* Staging uses `STAGE` slots round-robin; the dequant ring uses `j % DQ`.

## Form template identifiers
* `off` ↔ `m.off`; `chunkLen` ↔ `std::min(m.T, n - c0)`
-/

namespace Strata.Prefill

/-- Exclusive prefix sum of the per-expert counts. -/
def off (cnt : Nat → Nat) : Nat → Nat
  | 0 => 0
  | e + 1 => off cnt e + cnt e

/-- Each expert's group is contiguous. -/
theorem off_succ (cnt : Nat → Nat) (e : Nat) : off cnt (e + 1) = off cnt e + cnt e := rfl

/-- Offsets are monotone. -/
theorem off_mono (cnt : Nat → Nat) {a b : Nat} (h : a ≤ b) : off cnt a ≤ off cnt b := by
  induction b with
  | zero =>
    have : a = 0 := by omega
    subst this
    exact Nat.le_refl _
  | succ b ih =>
    rcases Nat.lt_or_ge a (b + 1) with hlt | hge
    · have := ih (by omega)
      rw [off_succ]
      omega
    · have : a = b + 1 := by omega
      subst this
      exact Nat.le_refl _

/-- The `k`-th pair of expert `e` lands strictly inside that expert's group. -/
theorem slot_in_group (cnt : Nat → Nat) (e k : Nat) (hk : k < cnt e) :
    off cnt e ≤ off cnt e + k ∧ off cnt e + k < off cnt (e + 1) := by
  rw [off_succ]
  omega

/-- Groups of different experts are disjoint. -/
theorem groups_disjoint (cnt : Nat → Nat) {e e' : Nat} (h : e < e') (k k' : Nat) (hk : k < cnt e) :
    off cnt e + k < off cnt e' + k' := by
  have h1 : off cnt (e + 1) ≤ off cnt e' := off_mono cnt h
  rw [off_succ] at h1
  omega

/-- The source token of pair `i` is `i / K`, a valid token when `i < T*K`. -/
theorem src_token_lt (i T K : Nat) (hK : 0 < K) (h : i < T * K) : i / K < T := by
  exact (Nat.div_lt_iff_lt_mul hK).mpr h

/-- Routed ids must be in `[0, NE)`. -/
theorem id_valid (e NE : Nat) (h : e < NE) : ¬ (NE ≤ e) := by omega

/-- Length of the chunk starting at `c0` of an `n`-token request. -/
def chunkLen (T n c0 : Nat) : Nat := min T (n - c0)

/-- A chunk never exceeds the buffer size `T`. -/
theorem chunkLen_le (T n c0 : Nat) : chunkLen T n c0 ≤ T := by
  unfold chunkLen
  omega

/-- A chunk never reads past the request. -/
theorem chunk_in_request (T n c0 : Nat) (h : c0 < n) : c0 + chunkLen T n c0 ≤ n := by
  unfold chunkLen
  omega

/-- A chunk starting inside the request is non-empty. -/
theorem chunk_nonempty (T n c0 : Nat) (hT : 0 < T) (h : c0 < n) : 0 < chunkLen T n c0 := by
  unfold chunkLen
  omega

/-- The loop `c0 += T` makes progress. -/
theorem chunk_progress (T c0 : Nat) (hT : 0 < T) : c0 < c0 + T := by omega

/-- Staging slot after advancing round-robin stays in range. -/
theorem stage_next_lt (s STAGE : Nat) (h : 0 < STAGE) : (s + 1) % STAGE < STAGE := Nat.mod_lt _ h

/-- Dequant ring buffer chosen for expert `j` is in range. -/
theorem dq_index_lt (j DQ : Nat) (h : 0 < DQ) : j % DQ < DQ := Nat.mod_lt _ h

/-- The staging lookahead never wraps onto a slot still needed. -/
theorem lookahead_fits (STAGE j staged : Nat) (hS : 0 < STAGE) (h : staged ≤ j + (STAGE - 1)) :
    staged - j < STAGE ∨ staged ≤ j := by omega

/-- PLE previous-token window shifts by one. -/
def shiftPrev (p : Int × Int) (tok : Int) : Int × Int := (p.2, tok)

/-- After two shifts the window holds exactly the last two tokens. -/
theorem shiftPrev_two (p : Int × Int) (a b : Int) : shiftPrev (shiftPrev p a) b = (a, b) := rfl

/-- `n_embd / 64` blocks cover `n_embd` when it is a multiple of 64. -/
theorem blocks_cover (n_embd : Nat) (h : n_embd % 64 = 0) : n_embd / 64 * 64 = n_embd := by omega

end Strata.Prefill
