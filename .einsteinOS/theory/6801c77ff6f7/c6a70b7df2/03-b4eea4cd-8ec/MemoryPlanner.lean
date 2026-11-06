/-!
# MODULE: memory_planner
SOURCES: include/strata/plan/plan.hpp
The VRAM memory planner: fixed costs (dense weights, embedding, workspace, recurrent state) plus the KV cache
at `max_context` must fit the pool, and what is left becomes whole expert-cache slots.  It REFUSES instead of
overcommitting.

## Source semantics
* `state_bytes`: 36 GDN layers x (S*h_v*S + (d_conv-1)*C) fp32 = 117,669,888 B.
* `kv_bytes_per_token`: 12 QSA layers x (INT8 K/V plus fp16 scale per 64 plus one pooled 128-wide indexer key per
  4-token block) = 13,056 B per token.
* `make_plan`: throws `DoesNotClose` if `kv + fixed > pool` or if no slot fits; otherwise
  `cache_bytes = pool - kv - fixed`, `cache_slots = cache_bytes / blob`, and the slot remainder is unusable.

## Form template identifiers
Costs <- `Costs`; Plan <- `Plan`; makePlan <- `make_plan` (None = `DoesNotClose`); fixedCost <- `fixed`.
-/

namespace Strata.MemoryPlanner

-- § geometry-derived byte costs

/-- GDN state bytes: layers x (S*h_v*S*4 + (d_conv-1)*C*4). -/
def stateBytes (nGdn s hv dconv c : Nat) : Nat := nGdn * (s * hv * s * 4 + (dconv - 1) * c * 4)

/-- Per-token KV + indexer bytes: layers x (K/V int8 + scales + indexer key per block). -/
def kvBytesPerToken (nQsa nKv headDim group keyHeads keyDim block : Nat) : Nat :=
  nQsa * ((2 * nKv * headDim) + (2 * nKv * headDim / group) * 2 + keyHeads * keyDim / block)

/-- The real recurrent state is 117,669,888 bytes. -/
theorem real_state_bytes : stateBytes 36 128 48 4 10240 = 117669888 := by
  decide

/-- Which is 85 expert-cache slots worth of VRAM (the omission round 198 fixed). -/
theorem real_state_is_85_slots : stateBytes 36 128 48 4 10240 / 1382400 = 85 := by
  decide

/-- The real KV + indexer cost is 13,056 bytes per token. -/
theorem real_kv_per_token : kvBytesPerToken 12 2 256 64 1 128 4 = 13056 := by
  decide

-- § the plan

/-- The fixed costs of `make_plan`. -/
structure Costs where
  expertBlob : Nat
  dense : Nat
  embd : Nat
  mtp : Nat
  workspace : Nat
  state : Nat

/-- A computed plan. -/
structure Plan where
  maxContext : Nat
  kvBytes : Nat
  stateBytes : Nat
  cacheBytes : Nat
  cacheSlots : Nat
  vramBudget : Nat
  vramUsed : Nat

/-- Everything resident that is not the KV cache or the expert cache. -/
def fixedCost (c : Costs) : Nat := c.dense + c.embd + c.mtp + c.workspace + c.state

/-- `make_plan`; `none` stands for the `DoesNotClose` exception. `kv` is `kv_bytes_per_token * max_context`. -/
def makePlan (maxCtx kv pool : Nat) (c : Costs) : Option Plan :=
  if kv + fixedCost c > pool then none
  else if (pool - kv - fixedCost c) / c.expertBlob = 0 then none
  else some
    { maxContext := maxCtx, kvBytes := kv, stateBytes := c.state,
      cacheBytes := pool - kv - fixedCost c,
      cacheSlots := (pool - kv - fixedCost c) / c.expertBlob,
      vramBudget := pool,
      vramUsed := pool - (pool - kv - fixedCost c) % c.expertBlob }

/-- Overcommitted fixed costs are refused, never planned around. -/
theorem refuses_overcommit (maxCtx kv pool : Nat) (c : Costs) (h : kv + fixedCost c > pool) :
    makePlan maxCtx kv pool c = none := by
  unfold makePlan
  rw [if_pos h]

/-- A plan that exists closes: KV + fixed + cache is exactly the pool. -/
theorem plan_closes (maxCtx kv pool : Nat) (c : Costs) (p : Plan)
    (h : makePlan maxCtx kv pool c = some p) : p.kvBytes + fixedCost c + p.cacheBytes = pool := by
  unfold makePlan at h
  split at h
  · cases h
  · rename_i h1
    split at h
    · cases h
    · cases h
      dsimp only
      omega

/-- A plan that exists has at least one slot. -/
theorem plan_has_slot (maxCtx kv pool : Nat) (c : Costs) (p : Plan)
    (h : makePlan maxCtx kv pool c = some p) : p.cacheSlots ≠ 0 := by
  unfold makePlan at h
  split at h
  · cases h
  · split at h
    · cases h
    · rename_i h2
      cases h
      exact h2

/-- Slots are whole: they never use more bytes than the cache has. -/
theorem slots_fit (cacheBytes blob : Nat) : (cacheBytes / blob) * blob ≤ cacheBytes :=
  Nat.div_mul_le_self cacheBytes blob

/-- The plan's budget is the pool it was asked about. -/
theorem plan_budget (maxCtx kv pool : Nat) (c : Costs) (p : Plan)
    (h : makePlan maxCtx kv pool c = some p) : p.vramBudget = pool := by
  unfold makePlan at h
  split at h
  · cases h
  · split at h
    · cases h
    · cases h
      rfl

/-- `vram_used` never exceeds the pool. -/
theorem plan_used_le (maxCtx kv pool : Nat) (c : Costs) (p : Plan)
    (h : makePlan maxCtx kv pool c = some p) : p.vramUsed ≤ pool := by
  unfold makePlan at h
  split at h
  · cases h
  · split at h
    · cases h
    · cases h
      exact Nat.sub_le _ _

/-- Used VRAM is exactly KV + fixed + whole slots: the unusable remainder is the only thing left over. -/
theorem plan_used_eq (maxCtx kv pool : Nat) (c : Costs) (p : Plan)
    (h : makePlan maxCtx kv pool c = some p) :
    p.vramUsed = kv + fixedCost c + c.expertBlob * p.cacheSlots := by
  unfold makePlan at h
  split at h
  · cases h
  · rename_i h1
    split at h
    · cases h
    · cases h
      dsimp only
      have hd := Nat.mod_add_div (pool - kv - fixedCost c) c.expertBlob
      omega

end Strata.MemoryPlanner
