/-!
# MODULE: vram_memory_planner
Arithmetic planner that splits a fixed VRAM pool between fixed costs, the KV/indexer cache, and
expert-cache slots; it refuses (throws) rather than overcommit.

## Source semantics
* `state_bytes = n_gdn_layers * (S*Hv*S*4 + (d_conv-1)*C*4)`; fixed, non-evictable.
* `kv_bytes_per_token = n_qsa * (kv_elems + 2*(kv_elems/group) + keyHeads*keyDim/block)`.
* `make_plan`: if `kv + fixed > pool` it fails; else `cache_bytes = pool - kv - fixed`,
  `cache_slots = cache_bytes / blob`, and zero slots also fails.

## Form template identifiers
* `pool`        ← `vram_pool_bytes()` / `pool` argument
* `kvBytes`     ← `Plan::kv_bytes`
* `fixedBytes`  ← `dense + embd + mtp + workspace + state`
* `cacheBytes`  ← `Plan::cache_bytes`
* `cacheSlots`  ← `Plan::cache_slots`
* `blob`        ← `Costs::expert_blob`
-/

namespace Strata.VramMemoryPlanner

-- § geometry

structure Geometry where
  nLayers : Nat := 48
  nQsa : Nat := 12
  nKvHeads : Nat := 2
  headDim : Nat := 256
  kvGroup : Nat := 64
  qsaBlock : Nat := 4
  indexerKeyHeads : Nat := 1
  indexerKeyDim : Nat := 128
  ssmStateSize : Nat := 128
  ssmVHeads : Nat := 48
  ssmDConv : Nat := 4
  ssmConvChannels : Nat := 10240

def Geometry.nGdn (g : Geometry) : Nat := g.nLayers - g.nQsa

/-- GDN recurrence + conv history bytes (fp32). -/
def stateBytes (g : Geometry) : Nat :=
  g.nGdn * (g.ssmStateSize * g.ssmVHeads * g.ssmStateSize * 4 + (g.ssmDConv - 1) * g.ssmConvChannels * 4)

/-- INT8 KV + indexer keys per token. -/
def kvBytesPerToken (g : Geometry) : Nat :=
  let kvElems := 2 * g.nKvHeads * g.headDim
  let kvScales := (kvElems / g.kvGroup) * 2
  let perLayer := kvElems + kvScales
  let idxLayer := g.indexerKeyHeads * g.indexerKeyDim / g.qsaBlock
  g.nQsa * (perLayer + idxLayer)

/-- 36 GDN layers on the real artifact. -/
theorem nGdn_default : ({} : Geometry).nGdn = 36 := by decide

/-- The recurrent state is 117,669,888 B, i.e. about 85 expert slots. -/
theorem stateBytes_default : stateBytes {} = 117669888 := by decide

/-- INT8 KV plus indexer keys costs 13,056 B per token on the real artifact. -/
theorem kvBytesPerToken_default : kvBytesPerToken {} = 13056 := by decide

/-- The indexer's cached key is 32 B per token per layer (not 64 or 128). -/
theorem indexer_key_bytes_default : ({} : Geometry).indexerKeyHeads * ({} : Geometry).indexerKeyDim / ({} : Geometry).qsaBlock = 32 := by
  decide

/-- KV bytes grow linearly with context. -/
theorem kv_linear (g : Geometry) (c : Nat) :
    kvBytesPerToken g * c = c * kvBytesPerToken g := Nat.mul_comm _ _

-- § plan

structure Plan where
  kvBytes : Nat
  cacheBytes : Nat
  cacheSlots : Nat
  vramUsed : Nat

/-- `make_plan`; `none` models `DoesNotClose`. -/
def makePlan (kvPerTok ctx fixed pool blob : Nat) : Option Plan :=
  if kvPerTok * ctx + fixed > pool then none
  else
    let cache := pool - kvPerTok * ctx - fixed
    let slots := cache / blob
    if slots = 0 then none
    else some { kvBytes := kvPerTok * ctx, cacheBytes := cache, cacheSlots := slots,
                vramUsed := pool - cache % blob }

/-- A successful plan never overcommits: kv + fixed + cache = pool. -/
theorem plan_closes (kvPerTok ctx fixed pool blob : Nat) (p : Plan)
    (h : makePlan kvPerTok ctx fixed pool blob = some p) :
    p.kvBytes + fixed + p.cacheBytes = pool := by
  unfold makePlan at h
  split_ifs at h with h1 h2
  · cases h
    simp only
    omega

/-- Slots times blob size never exceed the cache bytes. -/
theorem plan_slots_fit (kvPerTok ctx fixed pool blob : Nat) (p : Plan)
    (h : makePlan kvPerTok ctx fixed pool blob = some p) :
    p.cacheSlots * blob ≤ p.cacheBytes := by
  unfold makePlan at h
  split_ifs at h with h1 h2
  · cases h
    simp only
    exact Nat.div_mul_le_self _ _

/-- A successful plan has at least one slot. -/
theorem plan_has_slot (kvPerTok ctx fixed pool blob : Nat) (p : Plan)
    (h : makePlan kvPerTok ctx fixed pool blob = some p) : 0 < p.cacheSlots := by
  unfold makePlan at h
  split_ifs at h with h1 h2
  · cases h
    simp only
    omega

/-- An overcommitted request is refused. -/
theorem plan_refuses_overcommit (kvPerTok ctx fixed pool blob : Nat)
    (h : pool < kvPerTok * ctx + fixed) : makePlan kvPerTok ctx fixed pool blob = none := by
  unfold makePlan
  rw [if_pos h]

end Strata.VramMemoryPlanner
