/-!
# MODULE: memory_planner
VRAM budget planner: derives KV, recurrent-state and expert-cache sizes from the model geometry and refuses to overcommit.

## Source semantics
* INT8 KV + indexer keys cost `n_qsa * (kv_elems + kv_scales + idx)` bytes per token.
* GDN state per sequence: 36 layers x (recurrence + conv history) fp32.
* `make_plan` throws when `kv + fixed > pool` or when no expert slot fits; otherwise
  `cache = pool - kv - fixed`, `slots = cache / blob`.

## Form template identifiers
* kv_bytes_per_token -> `kv_bytes_per_token`
* state_bytes -> `recurrent_state_bytes`
* make_plan -> `make_plan`
* expert_blob -> `slot_bytes`
-/

namespace Strata.Plan

-- § Geometry arithmetic

/-- INT8 KV bytes per token per QSA layer: K and V elements plus one fp16 scale per group. -/
def kvLayerBytes (nKvHeads headDim group : Nat) : Nat :=
  2 * nKvHeads * headDim + (2 * nKvHeads * headDim / group) * 2

/-- Indexer bytes per token per layer: one shared key head, pooled per block. -/
def idxLayerBytes (keyHeads keyDim block : Nat) : Nat := keyHeads * keyDim / block

/-- KV bytes per token over all QSA layers. -/
def kvBytesPerToken (nQsa nKvHeads headDim group keyHeads keyDim block : Nat) : Nat :=
  nQsa * (kvLayerBytes nKvHeads headDim group + idxLayerBytes keyHeads keyDim block)

/-- The real geometry costs 13,056 bytes per token. -/
theorem real_kv_bytes_per_token : kvBytesPerToken 12 2 256 64 1 128 4 = 13056 := by decide

/-- The corrected indexer term is 32 bytes/token/layer (the 64 and 128 values were the earlier mistakes). -/
theorem indexer_term : idxLayerBytes 1 128 4 = 32 ∧ idxLayerBytes 4 128 4 = 128 := by decide

/-- GDN state per layer: recurrence `S*h_v*S*4` plus conv history `(d_conv-1)*C*4`. -/
def stateLayerBytes (s hv dconv ch : Nat) : Nat := s * hv * s * 4 + (dconv - 1) * ch * 4

/-- Per-layer values from the doc comment: 3,145,728 + 122,880. -/
theorem real_state_layer : stateLayerBytes 128 48 4 10240 = 3145728 + 122880 := by decide

/-- 36 GDN layers give 117,669,888 bytes. -/
theorem real_state_total : 36 * stateLayerBytes 128 48 4 10240 = 117669888 := by decide

/-- That state is worth 85 expert slots of 1,382,400 bytes. -/
theorem state_in_slots : 117669888 / 1382400 = 85 := by decide

/-- GDN layer count is derived: 48 - 12. -/
theorem gdn_layers : 48 - 12 = 36 := by decide

-- § The planner

/-- Core arithmetic of `make_plan`, returning (cache bytes, slots) or none for `DoesNotClose`. -/
def makePlan (kv fixed pool blob : Nat) : Option (Nat × Nat) :=
  if pool < kv + fixed then none
  else if (pool - kv - fixed) / blob = 0 then none
  else some (pool - kv - fixed, (pool - kv - fixed) / blob)

/-- Soundness: a returned plan fits the pool, and its cache and slots are derived as specified. -/
theorem makePlan_sound (kv fixed pool blob c s : Nat) (h : makePlan kv fixed pool blob = some (c, s)) :
    kv + fixed ≤ pool ∧ c = pool - kv - fixed ∧ s = c / blob ∧ 1 ≤ s := by
  unfold makePlan at h
  by_cases h1 : pool < kv + fixed
  · simp [h1] at h
  · by_cases h2 : (pool - kv - fixed) / blob = 0
    · simp [h1, h2] at h
    · simp [h1, h2] at h
      obtain ⟨hc, hs⟩ := h
      refine ⟨by omega, hc.symm, by rw [← hc]; exact hs.symm, by omega⟩

/-- Refusal: an overcommitted request never yields a plan. -/
theorem makePlan_refuses_overcommit (kv fixed pool blob : Nat) (h : pool < kv + fixed) :
    makePlan kv fixed pool blob = none := by
  unfold makePlan
  simp [h]

/-- The slots never use more than the cache (the remainder is unusable, not free). -/
theorem slots_fit_cache (c blob : Nat) : c / blob * blob ≤ c := Nat.div_mul_le_self c blob

/-- Accounting identity: kv + fixed + cache = pool for any accepted plan. -/
theorem plan_closes (kv fixed pool blob c s : Nat) (h : makePlan kv fixed pool blob = some (c, s)) :
    kv + fixed + c = pool := by
  have := makePlan_sound kv fixed pool blob c s h
  omega

/-- KV cost is monotone in max context, so a smaller context never needs more VRAM. -/
theorem kv_monotone_in_context (perToken a b : Nat) (h : a ≤ b) : perToken * a ≤ perToken * b :=
  Nat.mul_le_mul_left perToken h

/-- Raising the fixed costs can only lower the cache. -/
theorem cache_antitone (kv f1 f2 pool : Nat) (h : f1 ≤ f2) : pool - kv - f2 ≤ pool - kv - f1 := by
  omega

end Strata.Plan
