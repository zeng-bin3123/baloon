/-!
# MODULE: memory_plan
The census/memory planner: derives KV, indexer and recurrent-state byte costs from the model geometry
and divides the remaining VRAM pool into expert-cache slots, refusing plans that do not close.

## Source semantics
* INT8 KV per QSA layer: `2*kv_heads*head_dim` bytes plus one fp16 scale per 64 elements; the indexer caches
  one `key_dim`-wide key per `qsa_block` tokens.  Default geometry gives 13,056 bytes per token.
* GDN state: `(S*h_v*S + (d_conv-1)*C) * 4` bytes per layer over 36 GDN layers = 117,669,888 bytes.
* `make_plan` fails (`DoesNotClose`) when `kv + fixed > pool`; otherwise `cache_bytes = pool - kv - fixed`,
  `cache_slots = cache_bytes / blob`, and the slot remainder is unusable.
* A larger context never yields more cache.

## Form template identifiers
* `kvBytesPerToken` <- `kv_bytes_per_token`
* `stateBytes`      <- `state_bytes`
* `makePlan`        <- `make_plan` (returns `none` for `DoesNotClose`)
* `blob`            <- `Costs::expert_blob` = 1,382,400
-/

namespace Strata.Plan

-- § default geometry

/-- KV elements per layer: K and V for 2 heads of 256. -/
def kvElems : Nat := 2 * 2 * 256
/-- fp16 scale bytes per layer: one per 64 elements, 2 bytes each. -/
def kvScales : Nat := kvElems / 64 * 2
/-- INT8 KV bytes per QSA layer per token. -/
def perLayerKv : Nat := kvElems + kvScales
/-- Indexer key bytes per layer per token: one 128-wide key per 4-token block. -/
def idxLayer : Nat := 1 * 128 / 4
/-- Bytes per token over the 12 QSA layers. -/
def kvBytesPerToken : Nat := 12 * (perLayerKv + idxLayer)

/-- The default geometry costs 13,056 bytes of KV plus indexer per token. -/
theorem kv_bytes_per_token_value : kvBytesPerToken = 13056 := by decide

/-- Number of GDN layers is 36 of 48. -/
theorem gdn_layers_value : 48 - 12 = 36 := by decide

/-- GDN state bytes per layer: recurrence plus convolution history, fp32. -/
def stateLayer : Nat := 128 * 48 * 128 * 4 + (4 - 1) * 10240 * 4

/-- Per-layer state is 3,268,608 bytes. -/
theorem state_layer_value : stateLayer = 3268608 := by decide

/-- Total recurrent state is 117,669,888 bytes (about 85 expert slots). -/
theorem state_bytes_value : 36 * stateLayer = 117669888 := by decide

/-- That state is worth at least 85 expert slots of 1,382,400 bytes. -/
theorem state_is_85_slots : 85 ≤ (36 * stateLayer) / 1382400 := by decide

-- § plan closure

/-- A plan: KV bytes, cache bytes, slots and VRAM used. -/
structure Plan where
  kv : Nat
  cacheBytes : Nat
  slots : Nat
  used : Nat

/-- The planner: `none` when the context does not close or no slot fits. -/
def makePlan (ctx pool fixed blob perTok : Nat) : Option Plan :=
  if perTok * ctx + fixed > pool then none
  else if (pool - perTok * ctx - fixed) / blob = 0 then none
  else some { kv := perTok * ctx, cacheBytes := pool - perTok * ctx - fixed,
              slots := (pool - perTok * ctx - fixed) / blob,
              used := pool - (pool - perTok * ctx - fixed) % blob }

/-- Inversion: a successful plan means the context closed, a slot fits, and the plan has the stated fields. -/
theorem makePlan_some_inv (ctx pool fixed blob perTok : Nat) (p : Plan)
    (h : makePlan ctx pool fixed blob perTok = some p) :
    ¬ (perTok * ctx + fixed > pool) ∧ (pool - perTok * ctx - fixed) / blob ≠ 0 ∧
    p = { kv := perTok * ctx, cacheBytes := pool - perTok * ctx - fixed,
          slots := (pool - perTok * ctx - fixed) / blob,
          used := pool - (pool - perTok * ctx - fixed) % blob } := by
  unfold makePlan at h
  by_cases h1 : perTok * ctx + fixed > pool
  · rw [if_pos h1] at h
    cases h
  · by_cases h2 : (pool - perTok * ctx - fixed) / blob = 0
    · rw [if_neg h1, if_pos h2] at h
      cases h
    · rw [if_neg h1, if_neg h2] at h
      exact ⟨h1, h2, (Option.some.inj h).symm⟩

/-- A successful plan accounts for the whole pool: KV, fixed costs and cache add up to the pool. -/
theorem plan_accounts (ctx pool fixed blob perTok : Nat) (p : Plan)
    (h : makePlan ctx pool fixed blob perTok = some p) : p.kv + fixed + p.cacheBytes = pool := by
  obtain ⟨h1, _, rfl⟩ := makePlan_some_inv ctx pool fixed blob perTok p h
  dsimp only
  omega

/-- The granted slots fit in the cache bytes. -/
theorem plan_slots_fit (ctx pool fixed blob perTok : Nat) (p : Plan)
    (h : makePlan ctx pool fixed blob perTok = some p) : p.slots * blob ≤ p.cacheBytes := by
  obtain ⟨_, _, rfl⟩ := makePlan_some_inv ctx pool fixed blob perTok p h
  exact Nat.div_mul_le_self _ _

/-- A successful plan has at least one slot. -/
theorem plan_has_slot (ctx pool fixed blob perTok : Nat) (p : Plan)
    (h : makePlan ctx pool fixed blob perTok = some p) : 0 < p.slots := by
  obtain ⟨_, h2, rfl⟩ := makePlan_some_inv ctx pool fixed blob perTok p h
  exact Nat.pos_of_ne_zero h2

/-- The VRAM used never exceeds the pool. -/
theorem plan_used_le (ctx pool fixed blob perTok : Nat) (p : Plan)
    (h : makePlan ctx pool fixed blob perTok = some p) : p.used ≤ pool := by
  obtain ⟨_, _, rfl⟩ := makePlan_some_inv ctx pool fixed blob perTok p h
  exact Nat.sub_le _ _

/-- A plan that would overcommit is refused. -/
theorem plan_refuses_overcommit (ctx pool fixed blob perTok : Nat) (h : perTok * ctx + fixed > pool) :
    makePlan ctx pool fixed blob perTok = none := by
  unfold makePlan
  rw [if_pos h]

-- § monotonicity

/-- Cache bytes as a function of the context. -/
def cacheOf (ctx pool fixed perTok : Nat) : Nat := pool - perTok * ctx - fixed

/-- A longer context never leaves more cache bytes. -/
theorem cache_antitone (c1 c2 pool fixed perTok : Nat) (h : c1 ≤ c2) :
    cacheOf c2 pool fixed perTok ≤ cacheOf c1 pool fixed perTok := by
  unfold cacheOf
  have := Nat.mul_le_mul_left perTok h
  omega

/-- Hence a longer context never yields more slots. -/
theorem slots_antitone (c1 c2 pool fixed perTok blob : Nat) (h : c1 ≤ c2) :
    cacheOf c2 pool fixed perTok / blob ≤ cacheOf c1 pool fixed perTok / blob :=
  Nat.div_le_div_right (cache_antitone c1 c2 pool fixed perTok h)

end Strata.Plan
