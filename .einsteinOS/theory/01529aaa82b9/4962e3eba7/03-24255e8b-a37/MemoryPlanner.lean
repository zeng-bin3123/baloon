/-!
# MODULE: memory_planner
Census + VRAM memory planner: derives KV/indexer bytes per token and GDN state bytes from the model geometry,
then divides the remaining VRAM pool into expert-cache slots, refusing (DoesNotClose) rather than overcommitting.

## Source semantics
* `kv_bytes_per_token = n_qsa * (2*kvh*hd + (2*kvh*hd/group)*2 + idx_key_heads*idx_key_dim/block)`.
* `state_bytes = n_gdn * (S*Hv*S*4 + (d_conv-1)*C*4)` (fp32, fixed, not evictable).
* `make_plan` fails iff `kv + fixed > pool` or fewer than one expert slot remains; otherwise
  `kv + fixed + slots*blob <= pool` and `slots >= 1`.

## Form template identifiers
Geometry -> `Geometry`; Costs -> `Costs`; Plan -> `Plan`; kv_bytes_per_token -> `kvBytesPerToken`;
state_bytes -> `stateBytes`; make_plan -> `makePlan`; DoesNotClose -> `none`.
-/

namespace Strata.Plan

-- § geometry and byte costs

/-- Model geometry fields used by the planner (all counts as naturals). -/
structure Geometry where
  nLayers : Nat
  nQsaLayers : Nat
  nKvHeads : Nat
  headDim : Nat
  kvGroup : Nat
  qsaBlock : Nat
  indexerKeyHeads : Nat
  indexerKeyDim : Nat
  ssmStateSize : Nat
  ssmVHeads : Nat
  ssmDConv : Nat
  ssmConvChannels : Nat

/-- The real artifact's geometry as declared in `plan.hpp`. -/
def artifact : Geometry :=
  { nLayers := 48, nQsaLayers := 12, nKvHeads := 2, headDim := 256, kvGroup := 64, qsaBlock := 4,
    indexerKeyHeads := 1, indexerKeyDim := 128, ssmStateSize := 128, ssmVHeads := 48, ssmDConv := 4,
    ssmConvChannels := 10240 }

/-- Number of GDN layers, derived rather than stored. -/
def nGdnLayers (g : Geometry) : Nat := g.nLayers - g.nQsaLayers

/-- GDN recurrence + conv history, bytes per sequence. -/
def stateBytes (g : Geometry) : Nat :=
  nGdnLayers g * (g.ssmStateSize * g.ssmVHeads * g.ssmStateSize * 4 + (g.ssmDConv - 1) * g.ssmConvChannels * 4)

/-- INT8 KV + indexer keys, bytes per token. -/
def kvBytesPerToken (g : Geometry) : Nat :=
  let kvElems := 2 * g.nKvHeads * g.headDim
  let kvScales := kvElems / g.kvGroup * 2
  let perLayer := kvElems + kvScales
  let idxLayer := g.indexerKeyHeads * g.indexerKeyDim / g.qsaBlock
  g.nQsaLayers * (perLayer + idxLayer)

/-- 36 of the 48 layers are GDN. -/
theorem artifact_gdn_layers : nGdnLayers artifact = 36 := by decide

/-- The GDN state is 117,669,888 bytes. -/
theorem artifact_state_bytes : stateBytes artifact = 117669888 := by decide

/-- KV plus indexer keys cost 13,056 bytes per token (32 B/token/layer for the indexer key). -/
theorem artifact_kv_per_token : kvBytesPerToken artifact = 13056 := by decide

/-- The GDN state alone occupies exactly 85 expert cache slots of 1,382,400 B. -/
theorem artifact_state_slots : 117669888 / 1382400 = 85 := by decide

/-- KV memory is monotone in the context length. -/
theorem kv_mono (g : Geometry) (c1 c2 : Nat) (h : c1 ≤ c2) :
    kvBytesPerToken g * c1 ≤ kvBytesPerToken g * c2 :=
  Nat.mul_le_mul_left _ h

-- § the plan

/-- Byte costs that are fixed for the life of a sequence. -/
structure Costs where
  expertBlob : Nat
  dense : Nat
  embd : Nat
  mtp : Nat
  workspace : Nat
  state : Nat

/-- Sum of the fixed (non-KV, non-cache) costs. -/
def fixedCost (c : Costs) : Nat := c.dense + c.embd + c.mtp + c.workspace + c.state

/-- The resulting plan. -/
structure Plan where
  maxContext : Nat
  kvBytes : Nat
  stateBytes : Nat
  cacheBytes : Nat
  cacheSlots : Nat
  vramBudget : Nat
  vramUsed : Nat

/-- `make_plan`: `none` models the `DoesNotClose` exceptions. -/
def makePlan (ctx : Nat) (g : Geometry) (c : Costs) (pool : Nat) : Option Plan :=
  if pool < kvBytesPerToken g * ctx + fixedCost c then none
  else if (pool - kvBytesPerToken g * ctx - fixedCost c) / c.expertBlob = 0 then none
  else
    some { maxContext := ctx, kvBytes := kvBytesPerToken g * ctx, stateBytes := c.state,
           cacheBytes := pool - kvBytesPerToken g * ctx - fixedCost c,
           cacheSlots := (pool - kvBytesPerToken g * ctx - fixedCost c) / c.expertBlob,
           vramBudget := pool,
           vramUsed := pool - (pool - kvBytesPerToken g * ctx - fixedCost c) % c.expertBlob }

/-- Soundness: a returned plan never overcommits the pool, has at least one slot, and `vram_used <= pool`. -/
theorem makePlan_sound (ctx : Nat) (g : Geometry) (c : Costs) (pool : Nat) (p : Plan)
    (h : makePlan ctx g c pool = some p) :
    kvBytesPerToken g * ctx + fixedCost c + p.cacheSlots * c.expertBlob ≤ pool ∧
      0 < p.cacheSlots ∧ p.vramUsed ≤ pool := by
  unfold makePlan at h
  split at h
  · cases h
  · rename_i h1
    split at h
    · cases h
    · rename_i h2
      have hp := Option.some.inj h
      subst hp
      have hd := Nat.div_mul_le_self (pool - kvBytesPerToken g * ctx - fixedCost c) c.expertBlob
      refine ⟨?_, ?_, ?_⟩
      · show kvBytesPerToken g * ctx + fixedCost c +
            (pool - kvBytesPerToken g * ctx - fixedCost c) / c.expertBlob * c.expertBlob ≤ pool
        omega
      · show 0 < (pool - kvBytesPerToken g * ctx - fixedCost c) / c.expertBlob
        exact Nat.pos_of_ne_zero h2
      · show pool - (pool - kvBytesPerToken g * ctx - fixedCost c) % c.expertBlob ≤ pool
        omega

/-- Completeness: if the KV, the fixed costs and one slot fit, a plan is returned. -/
theorem makePlan_isSome (ctx : Nat) (g : Geometry) (c : Costs) (pool : Nat)
    (hb : 0 < c.expertBlob)
    (hfit : kvBytesPerToken g * ctx + fixedCost c + c.expertBlob ≤ pool) :
    (makePlan ctx g c pool).isSome = true := by
  unfold makePlan
  have h1 : ¬ pool < kvBytesPerToken g * ctx + fixedCost c := by omega
  have h2 : (pool - kvBytesPerToken g * ctx - fixedCost c) / c.expertBlob ≠ 0 := by
    have hle : c.expertBlob ≤ pool - kvBytesPerToken g * ctx - fixedCost c := by omega
    exact Nat.pos_iff_ne_zero.mp (Nat.div_pos hle hb)
  simp [h1, h2]

/-- Refusal: if the KV plus fixed costs exceed the pool, no plan exists. -/
theorem makePlan_refuses (ctx : Nat) (g : Geometry) (c : Costs) (pool : Nat)
    (h : pool < kvBytesPerToken g * ctx + fixedCost c) : makePlan ctx g c pool = none := by
  unfold makePlan
  simp [h]

end Strata.Plan
