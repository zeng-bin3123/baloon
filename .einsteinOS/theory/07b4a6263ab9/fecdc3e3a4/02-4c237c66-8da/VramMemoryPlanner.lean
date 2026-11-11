/-!
# MODULE: vram_memory_planner
SOURCES: include/strata/plan/plan.hpp
The census + memory planner: derives per-token KV/indexer cost and fixed GDN state cost from the model
geometry, then divides a fixed VRAM pool between the KV cache, fixed costs and expert-cache slots — refusing
(`DoesNotClose`) rather than over-committing.

## Source semantics
* `kv_bytes_per_token = n_qsa_layers * (2*n_kv_heads*head_dim  +  (that / kv_group)*2  +
  indexer_key_heads*indexer_key_dim / qsa_block)`: INT8 K and V, one fp16 scale per 64 elements, and ONE
  shared 128-wide indexer key pooled per 4-token block (32 B/token/layer, not 64 or 128).
* `state_bytes = n_gdn_layers * (S*h_v*S*4 + (d_conv-1)*C*4)`, with `n_gdn_layers = n_layers - n_qsa_layers`;
  fixed and not evictable (117.7 MB = 85 expert slots for the real model).
* `make_plan`: refuse if `kv + fixed > pool`; else `cache = pool - kv - fixed`, `slots = cache / blob`,
  refuse if `slots = 0`; `vram_used = pool - cache % blob` (slot remainder is unusable).
  `DoesNotClose` is modelled as `none`.

## Form template identifiers
Geometry → Geometry, Costs → Costs, Plan → Plan, kv_bytes_per_token → kvBytesPerToken,
state_bytes → stateBytes, n_gdn_layers → Geometry.nGdnLayers, vram_pool_bytes → vramPoolBytes,
make_plan → makePlan, DoesNotClose → `none`, fixed (local) → fixedCost
-/

namespace Strata.Plan

-- § geometry and costs

/-- The model geometry; defaults are the real Qwen3.8-Flash-Next artifact. -/
structure Geometry where
  nLayers : Nat := 48
  nQsaLayers : Nat := 12
  nKvHeads : Nat := 2
  headDim : Nat := 256
  kvGroup : Nat := 64
  qsaBlock : Nat := 4
  indexerQHeads : Nat := 4
  indexerKeyHeads : Nat := 1
  indexerKeyDim : Nat := 128
  ssmStateSize : Nat := 128
  ssmVHeads : Nat := 48
  ssmDConv : Nat := 4
  ssmConvChannels : Nat := 10240

/-- GDN layers are derived, never stored as a third constant. -/
def Geometry.nGdnLayers (g : Geometry) : Nat := g.nLayers - g.nQsaLayers

/-- Byte costs fed to the planner. -/
structure Costs where
  expertBlob : Nat := 1382400
  denseBytes : Nat := 0
  embdBytes : Nat := 0
  mtpBytes : Nat := 0
  workspaceBytes : Nat := 0
  stateBytes : Nat := 0

/-- The indexer's cached-key bytes per token per QSA layer: ONE shared key head. -/
def indexerBytesPerLayer (g : Geometry) : Nat := g.indexerKeyHeads * g.indexerKeyDim / g.qsaBlock

/-- INT8 K/V + fp16 group scales + pooled indexer key, summed over QSA layers. -/
def kvBytesPerToken (g : Geometry) : Nat :=
  let kvElems := 2 * g.nKvHeads * g.headDim
  let kvScales := kvElems / g.kvGroup * 2
  g.nQsaLayers * (kvElems + kvScales + indexerBytesPerLayer g)

/-- GDN recurrence (S, h_v, S) fp32 plus conv history (d_conv-1, C) fp32, over all GDN layers. -/
def stateBytes (g : Geometry) : Nat :=
  g.nGdnLayers * (g.ssmStateSize * g.ssmVHeads * g.ssmStateSize * 4 +
    (g.ssmDConv - 1) * g.ssmConvChannels * 4)

/-- The shared KV + expert-cache VRAM pool (the one number taken from the table). -/
def vramPoolBytes : Nat := 5943000000

/-- The resulting plan. -/
structure Plan where
  maxContext : Nat
  kvBytes : Nat
  stateBytes : Nat
  cacheBytes : Nat
  cacheSlots : Nat
  vramBudget : Nat
  vramUsed : Nat
  dramExperts : Nat

/-- Everything that occupies the pool regardless of context. -/
def fixedCost (c : Costs) : Nat :=
  c.denseBytes + c.embdBytes + c.mtpBytes + c.workspaceBytes + c.stateBytes

/-- `make_plan`; `none` is `DoesNotClose`. -/
def makePlan (ctx : Nat) (g : Geometry) (c : Costs) (pool : Nat := vramPoolBytes) : Option Plan :=
  let kv := kvBytesPerToken g * ctx
  if pool < kv + fixedCost c then none
  else
    let cache := pool - kv - fixedCost c
    let slots := cache / c.expertBlob
    if slots = 0 then none
    else some { maxContext := ctx, kvBytes := kv, stateBytes := c.stateBytes, cacheBytes := cache,
                cacheSlots := slots, vramBudget := pool, vramUsed := pool - cache % c.expertBlob,
                dramExperts := c.denseBytes + c.embdBytes }

-- § derived constants of the real model

/-- The indexer term is 32 B/token/layer (not the 64 or 128 of the two earlier wrong derivations). -/
theorem indexer_bytes_default : indexerBytesPerLayer {} = 32 := by decide

/-- The two wrong derivations, for the record: 1 head × 256 / 4 and 4 heads × 128 / 4. -/
theorem indexer_wrong_derivations : 1 * 256 / 4 = 2 * indexerBytesPerLayer {} ∧
    4 * 128 / 4 = 4 * indexerBytesPerLayer {} := by decide

/-- KV + indexer cost of the real model: 12 × (1024 + 32 + 32) = 13,056 B per token. -/
theorem kv_bytes_per_token_default : kvBytesPerToken {} = 13056 := by decide

/-- 36 of 48 layers are GDN. -/
theorem gdn_layers_default : ({} : Geometry).nGdnLayers = 36 := by decide

/-- GDN state of the real model is 117,669,888 B ≈ 117.7 MB. -/
theorem state_bytes_default : stateBytes {} = 117669888 := by decide

/-- ...which is 85 whole expert-cache slots — the term the planner silently omitted before round 198. -/
theorem state_costs_85_slots : stateBytes {} / ({} : Costs).expertBlob = 85 := by decide

/-- At 32K context with the state counted, the real model gets 3,904 expert slots. -/
theorem plan_32k_slots :
    (makePlan 32768 {} { stateBytes := stateBytes {} }).map Plan.cacheSlots = some 3904 := by decide

/-- At 128K context the same pool leaves 2,976 slots. -/
theorem plan_128k_slots :
    (makePlan 131072 {} { stateBytes := stateBytes {} }).map Plan.cacheSlots = some 2976 := by decide

-- § contracts of makePlan

/-- Unfolding lemma: what a successful plan looks like. -/
theorem makePlan_some {ctx : Nat} {g : Geometry} {c : Costs} {pool : Nat} {p : Plan}
    (h : makePlan ctx g c pool = some p) :
    kvBytesPerToken g * ctx + fixedCost c ≤ pool ∧
    p.kvBytes = kvBytesPerToken g * ctx ∧
    p.cacheBytes = pool - kvBytesPerToken g * ctx - fixedCost c ∧
    p.cacheSlots = p.cacheBytes / c.expertBlob ∧ p.cacheSlots ≠ 0 ∧
    p.vramBudget = pool ∧ p.vramUsed = pool - p.cacheBytes % c.expertBlob ∧
    p.stateBytes = c.stateBytes := by
  unfold makePlan at h
  simp only at h
  split at h
  · exact absurd h (by simp)
  · rename_i hfit
    split at h
    · exact absurd h (by simp)
    · rename_i hslots
      simp only [Option.some.injEq] at h
      subst h
      exact ⟨by omega, rfl, rfl, rfl, hslots, rfl, rfl, rfl⟩

/-- Refusal: if KV plus fixed costs exceed the pool, there is no plan (no silent over-commit). -/
theorem makePlan_refuses_overcommit (ctx : Nat) (g : Geometry) (c : Costs) (pool : Nat)
    (h : pool < kvBytesPerToken g * ctx + fixedCost c) : makePlan ctx g c pool = none := by
  unfold makePlan
  simp only [h, if_true]

/-- The pool closes exactly: KV + fixed + expert cache = pool. -/
theorem plan_closes {ctx : Nat} {g : Geometry} {c : Costs} {pool : Nat} {p : Plan}
    (h : makePlan ctx g c pool = some p) : p.kvBytes + fixedCost c + p.cacheBytes = pool := by
  obtain ⟨hfit, hkv, hcache, -, -, -, -, -⟩ := makePlan_some h
  omega

/-- The fixed GDN state is always inside the plan: it is part of the fixed cost, never evictable. -/
theorem plan_counts_state {ctx : Nat} {g : Geometry} {c : Costs} {pool : Nat} {p : Plan}
    (h : makePlan ctx g c pool = some p) : p.stateBytes + p.kvBytes + p.cacheBytes ≤ pool := by
  have hc := plan_closes h
  obtain ⟨-, -, -, -, -, -, -, hs⟩ := makePlan_some h
  unfold fixedCost at hc
  omega

/-- There is at least one slot and the slots fit inside the cache bytes. -/
theorem plan_slots_fit {ctx : Nat} {g : Geometry} {c : Costs} {pool : Nat} {p : Plan}
    (h : makePlan ctx g c pool = some p) :
    1 ≤ p.cacheSlots ∧ p.cacheSlots * c.expertBlob ≤ p.cacheBytes := by
  obtain ⟨-, -, -, hs, hnz, -, -, -⟩ := makePlan_some h
  refine ⟨by omega, ?_⟩
  rw [hs]
  exact Nat.div_mul_le_self p.cacheBytes c.expertBlob

/-- Reported VRAM use never exceeds the budget. -/
theorem plan_used_le_budget {ctx : Nat} {g : Geometry} {c : Costs} {pool : Nat} {p : Plan}
    (h : makePlan ctx g c pool = some p) : p.vramUsed ≤ p.vramBudget := by
  obtain ⟨-, -, -, -, -, hb, hu, -⟩ := makePlan_some h
  omega

/-- A longer context never yields more expert slots. -/
theorem slots_antitone_in_context {c1 c2 : Nat} {g : Geometry} {c : Costs} {pool : Nat} {p1 p2 : Plan}
    (hle : c1 ≤ c2) (h1 : makePlan c1 g c pool = some p1) (h2 : makePlan c2 g c pool = some p2) :
    p2.cacheSlots ≤ p1.cacheSlots := by
  obtain ⟨-, -, hc1, hs1, -⟩ := makePlan_some h1
  obtain ⟨-, -, hc2, hs2, -⟩ := makePlan_some h2
  have hkv : kvBytesPerToken g * c1 ≤ kvBytesPerToken g * c2 := Nat.mul_le_mul_left _ hle
  have hcache : p2.cacheBytes ≤ p1.cacheBytes := by omega
  rw [hs1, hs2]
  exact Nat.div_le_div_right hcache

/-- A plan exists for a context exactly when KV + fixed fits and at least one slot remains. -/
theorem makePlan_isSome_iff (ctx : Nat) (g : Geometry) (c : Costs) (pool : Nat) :
    (makePlan ctx g c pool).isSome = true ↔
      (kvBytesPerToken g * ctx + fixedCost c ≤ pool ∧
        (pool - kvBytesPerToken g * ctx - fixedCost c) / c.expertBlob ≠ 0) := by
  unfold makePlan
  simp only
  by_cases h1 : pool < kvBytesPerToken g * ctx + fixedCost c
  · simp only [h1, if_true, Option.isSome_none, Bool.false_eq_true, false_iff]
    omega
  · by_cases h2 : (pool - kvBytesPerToken g * ctx - fixedCost c) / c.expertBlob = 0
    · simp only [h1, h2, if_false, if_true, Option.isSome_none, Bool.false_eq_true, false_iff]
      omega
    · simp only [h1, h2, if_false, Option.isSome_some, true_iff]
      exact ⟨by omega, h2⟩

end Strata.Plan
