/-!
# MODULE: sampler_chain
Deterministic structure of the CUDA sampler (`sampler.cu`): greedy argmax tie rule and its warp/block
reduction, penalty application, penalty history window, top-k/top-p bounds, Philox mantissa range, and launch
geometry.

## Source semantics
Greedy decoding is the global argmax over penalised logits, ties resolved to the lowest index. The reduction
merges two candidates by larger value and, on equal values, smaller index; this order is a commutative,
associative, idempotent selection, so any reduction tree gives the serial answer. The repeat penalty
multiplies non-positive logits and divides positive ones; the presence penalty is a boolean. The history
window is the tail of the row. The uniform draw has 24 mantissa bits.

## Form template identifiers
- `better`/`merge` -> the `ov > bv || (ov == bv && oi < best)` rule
- penalty_repeat/freq/present -> `rep`/`freq`/`present`; penalty_last_n -> `plen`; history_len -> `hl`
- KMAX -> `KMAX`
-/

namespace Strata.Sampler

-- § candidate merge

abbrev Cand := Int × Nat

def merge (a b : Cand) : Cand := if b.1 > a.1 ∨ (b.1 = a.1 ∧ b.2 < a.2) then b else a

/-- Merging is commutative on candidates with distinct indices. -/
theorem merge_comm (a b : Cand) (h : a.2 ≠ b.2) : merge a b = merge b a := by
  obtain ⟨av, ai⟩ := a
  obtain ⟨bv, bi⟩ := b
  simp only [merge] at *
  split_ifs <;> simp_all <;> omega

/-- Merging is associative. -/
theorem merge_assoc (a b c : Cand) : merge (merge a b) c = merge a (merge b c) := by
  obtain ⟨av, ai⟩ := a
  obtain ⟨bv, bi⟩ := b
  obtain ⟨cv, ci⟩ := c
  simp only [merge]
  split_ifs <;> simp_all <;> omega

/-- Merging is idempotent. -/
theorem merge_idem (a : Cand) : merge a a = a := by
  obtain ⟨av, ai⟩ := a
  simp [merge]

/-- The merged value dominates both inputs. -/
theorem merge_value_ge (a b : Cand) : a.1 ≤ (merge a b).1 ∧ b.1 ≤ (merge a b).1 := by
  obtain ⟨av, ai⟩ := a
  obtain ⟨bv, bi⟩ := b
  simp only [merge]
  split_ifs <;> simp_all <;> omega

/-- On equal values the smaller index wins. -/
theorem merge_tie_min (a b : Cand) (h : a.1 = b.1) : (merge a b).2 = min a.2 b.2 := by
  obtain ⟨av, ai⟩ := a
  obtain ⟨bv, bi⟩ := b
  simp only at h
  simp only [merge]
  split_ifs <;> simp_all <;> omega

/-- The merged candidate is one of the inputs. -/
theorem merge_mem (a b : Cand) : merge a b = a ∨ merge a b = b := by
  unfold merge; split_ifs <;> simp

/-- The "no candidate" sentinel (an index equal to n_vocab) never beats a real one when values are equal. -/
theorem sentinel_loses (v : Int) (i n : Nat) (h : i < n) : merge (v, i) (v, n) = (v, i) := by
  simp only [merge]
  split_ifs <;> simp_all <;> omega

-- § strided scan coverage

/-- Thread `tid` scanning `v = tid + k*bd` visits every vocabulary index. -/
theorem scan_cover (bd v : Nat) (hbd : 0 < bd) : ∃ tid k, tid < bd ∧ v = tid + k * bd := by
  refine ⟨v % bd, v / bd, Nat.mod_lt v hbd, ?_⟩
  have h := Nat.mod_add_div v bd
  have h2 := Nat.mul_comm bd (v / bd)
  omega

/-- Distinct (thread, step) pairs visit distinct indices. -/
theorem scan_inj (bd t k t' k' : Nat) (ht : t < bd) (ht' : t' < bd) (hbd : 0 < bd)
    (h : t + k * bd = t' + k' * bd) : t = t' ∧ k = k' := by
  have e1 : (t + k * bd) % bd = t := by
    rw [Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt ht]
  have e2 : (t' + k' * bd) % bd = t' := by
    rw [Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt ht']
  have e3 : (t + k * bd) / bd = k := by
    rw [Nat.add_mul_div_right _ _ hbd, Nat.div_eq_of_lt ht, Nat.zero_add]
  have e4 : (t' + k' * bd) / bd = k' := by
    rw [Nat.add_mul_div_right _ _ hbd, Nat.div_eq_of_lt ht', Nat.zero_add]
  rw [h] at e1 e3
  exact ⟨by omega, by omega⟩

/-- The block of 1024 threads needs 32 warp slots, exactly the size of the shared arrays. -/
theorem greedy_warp_slots (bd : Nat) (h : bd ≤ 1024) : (bd + 31) / 32 ≤ 32 := by omega

/-- The greedy fallback index is always a valid token when the vocabulary is non-empty. -/
theorem greedy_out_valid (wi n : Nat) (hn : 0 < n) : (if wi < n then wi else 0) < n := by
  split_ifs <;> omega

-- § penalties

def penalize (logit count rep freq present : Int) : Int :=
  if count ≤ 0 then logit
  else (if logit ≤ 0 then logit * rep else logit / rep) - (count * freq + (if count > 0 then 1 else 0) * present)

/-- A token absent from the window is not penalised. -/
theorem penalize_absent (logit count rep freq present : Int) (h : count ≤ 0) :
    penalize logit count rep freq present = logit := by
  simp [penalize, h]

/-- The presence term is a boolean: it is `present` once for any positive count. -/
theorem penalize_present_once (logit count rep freq present : Int) (h : 0 < count) :
    penalize logit count rep freq present =
      (if logit ≤ 0 then logit * rep else logit / rep) - (count * freq + present) := by
  have h1 : ¬ count ≤ 0 := by omega
  simp [penalize, h1, h]

/-- Non-positive logits are multiplied and positive ones divided. -/
theorem penalize_branches (logit count rep freq present : Int) (h : 0 < count) :
    (logit ≤ 0 → penalize logit count rep freq present = logit * rep - (count * freq + present)) ∧
    (0 < logit → penalize logit count rep freq present = logit / rep - (count * freq + present)) := by
  have h1 : ¬ count ≤ 0 := by omega
  constructor
  · intro hl; simp [penalize, h1, h, hl]
  · intro hl
    have h2 : ¬ logit ≤ 0 := by omega
    simp [penalize, h1, h, h2]

-- § history window

def histLen (plen hl : Int) : Int := max 0 (min plen hl)

/-- The window length is between 0 and the history length. -/
theorem histLen_bounds (plen hl : Int) (h : 0 ≤ hl) : 0 ≤ histLen plen hl ∧ histLen plen hl ≤ hl := by
  unfold histLen; omega

/-- The tail window starts at a non-negative offset and ends at the history end. -/
theorem hist_tail_offset (plen hl : Int) (h : 0 ≤ hl) :
    0 ≤ hl - histLen plen hl ∧ hl - histLen plen hl + histLen plen hl = hl := by
  unfold histLen; omega

/-- A non-positive penalty window is empty. -/
theorem histLen_nonpos (plen hl : Int) (h : plen ≤ 0) : histLen plen hl = 0 := by
  unfold histLen; omega

-- § top-k / top-p

def KMAX : Nat := 64

def finalCut (cut minKeep nKeep : Nat) : Nat := if cut < minKeep then min minKeep nKeep else cut

/-- The top-p cut never exceeds the number of survivors. -/
theorem finalCut_le (cut minKeep nKeep : Nat) (h : cut ≤ nKeep) : finalCut cut minKeep nKeep ≤ nKeep := by
  unfold finalCut; split_ifs <;> omega

/-- At least one survivor remains when min_keep, the cut and the survivors are positive. -/
theorem finalCut_pos (cut minKeep nKeep : Nat) (h1 : 1 ≤ cut) (h2 : 1 ≤ minKeep) (h3 : 1 ≤ nKeep) :
    1 ≤ finalCut cut minKeep nKeep := by
  unfold finalCut; split_ifs <;> omega

/-- The effective top-k is capped at KMAX. -/
theorem topk_capped (topk : Nat) : (if topk < KMAX then topk else KMAX) ≤ KMAX := by
  unfold KMAX; split_ifs <;> omega

-- § Philox and launch

/-- A 32-bit word shifted right by 8 has at most 24 bits, so the uniform stays strictly below 1. -/
theorem philox_mantissa_lt (c0 : Nat) (h : c0 < 4294967296) : c0 / 256 < 16777216 := by omega

/-- The sampled-path grid covers every token. -/
theorem sampler_grid_cover (n : Nat) : n ≤ (n + 63) / 64 * 64 := by omega

end Strata.Sampler
