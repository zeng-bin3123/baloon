/-!
# MODULE: vram_memory_plan
SOURCES: include/strata/plan/plan.hpp
The census and VRAM planner: GDN recurrent-state bytes, INT8 KV plus indexer-key bytes per token,
and `make_plan`, which divides the fixed VRAM pool between KV, fixed costs and expert-cache slots,
refusing (`DoesNotClose`) rather than overcommitting.

## Source semantics
* `state_bytes` = 36 GDN layers x (128*48*128*4 + 3*10240*4) = 117,669,888 B = 85 whole slots.
* `kv_bytes_per_token` = 12 QSA layers x (1024 INT8 + 32 B scales + 32 B indexer key) = 13,056 B;
  the indexer caches ONE shared 128-wide key per 4-token block (32 B/token/layer).
* `make_plan` succeeds only if KV + fixed costs fit the pool and at least one slot remains; then
  KV + fixed + cache = pool, slots x blob <= cache, and the slot remainder is less than one blob.
* More context never yields more cache slots.

## Form template identifiers
`Geometry` → `Geometry`; `Costs` → `Costs`; `state_bytes` → `stateBytes`;
`kv_bytes_per_token` → `kvBytesPerToken`; `make_plan` → `makePlan`; `DoesNotClose` → `Except.error`;
`vram_pool_bytes` → `vramPool`.
-/

namespace Strata.VramMemoryPlan

-- § Geometry and costs

/-- Model geometry, defaults from the real artifact. -/
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

/-- GDN layers are derived, never a third constant. -/
def Geometry.nGdnLayers (g : Geometry) : Nat := g.nLayers - g.nQsaLayers

/-- Byte costs. -/
structure Costs where
  expertBlob : Nat := 1382400
  dense : Nat := 0
  embd : Nat := 0
  mtp : Nat := 0
  workspace : Nat := 0
  state : Nat := 0

/-- The shared VRAM pool, 5,943,000,000 B. -/
def vramPool : Nat := 5943000000

/-- 36 of the 48 layers are GDN. -/
theorem default_gdn_layers : ({} : Geometry).nGdnLayers = 36 := rfl

-- § Recurrent state

/-- `state_bytes(g)`: fp32 recurrence plus conv history, over the GDN layers. -/
def stateBytes (g : Geometry) : Nat :=
  g.nGdnLayers * (g.ssmStateSize * g.ssmVHeads * g.ssmStateSize * 4 +
    (g.ssmDConv - 1) * g.ssmConvChannels * 4)

/-- The recurrence term is 3,145,728 B per layer. -/
theorem recurrence_per_layer : 128 * 48 * 128 * 4 = 3145728 := rfl

/-- The conv-history term is 122,880 B per layer. -/
theorem conv_per_layer : 3 * 10240 * 4 = 122880 := rfl

/-- The missing term was 117,669,888 B. -/
theorem stateBytes_default : stateBytes {} = 117669888 := rfl

/-- ... which is 85 expert cache slots - not a rounding error. -/
theorem state_is_85_slots : stateBytes {} / 1382400 = 85 := rfl

-- § KV and indexer keys

/-- Indexer key bytes per token per layer: one shared key head, pooled per block. -/
def indexerBytes (g : Geometry) : Nat := g.indexerKeyHeads * g.indexerKeyDim / g.qsaBlock

/-- `kv_bytes_per_token(g)`. -/
def kvBytesPerToken (g : Geometry) : Nat :=
  let kvElems := 2 * g.nKvHeads * g.headDim
  let kvScales := kvElems / g.kvGroup * 2
  g.nQsaLayers * (kvElems + kvScales + indexerBytes g)

/-- The correct indexer cost is 32 B/token/layer. -/
theorem indexer_default : indexerBytes {} = 32 := rfl

/-- The two historical errors were 2x and 4x too big. -/
theorem indexer_errors : 1 * 256 / 4 = 2 * 32 ∧ 4 * 128 / 4 = 4 * 32 := ⟨rfl, rfl⟩

/-- 13,056 B of KV and indexer keys per token. -/
theorem kvBytesPerToken_default : kvBytesPerToken {} = 13056 := rfl

-- § The plan

/-- The planner's output. -/
structure Plan where
  maxContext : Nat
  kvBytes : Nat
  stateBytes : Nat
  cacheBytes : Nat
  cacheSlots : Nat
  vramBudget : Nat
  vramUsed : Nat
  dramExperts : Nat

/-- Everything that never leaves the pool. -/
def fixedCost (c : Costs) : Nat := c.dense + c.embd + c.mtp + c.workspace + c.state

/-- `make_plan`; `Except.error` is `DoesNotClose`. -/
def makePlan (ctx : Nat) (g : Geometry) (c : Costs) (pool : Nat := vramPool) : Except String Plan :=
  if pool < kvBytesPerToken g * ctx + fixedCost c then .error "does not close"
  else if (pool - kvBytesPerToken g * ctx - fixedCost c) / c.expertBlob = 0 then .error "no slot"
  else .ok
    { maxContext := ctx
      kvBytes := kvBytesPerToken g * ctx
      stateBytes := c.state
      cacheBytes := pool - kvBytesPerToken g * ctx - fixedCost c
      cacheSlots := (pool - kvBytesPerToken g * ctx - fixedCost c) / c.expertBlob
      vramBudget := pool
      vramUsed := pool - (pool - kvBytesPerToken g * ctx - fixedCost c) % c.expertBlob
      dramExperts := c.dense + c.embd }

/-- Helper: unpack a successful plan into its defining facts. -/
theorem makePlan_ok {ctx : Nat} {g : Geometry} {c : Costs} {pool : Nat} {p : Plan}
    (h : makePlan ctx g c pool = .ok p) :
    kvBytesPerToken g * ctx + fixedCost c ≤ pool ∧
    p.kvBytes = kvBytesPerToken g * ctx ∧
    p.cacheBytes = pool - kvBytesPerToken g * ctx - fixedCost c ∧
    p.cacheSlots = p.cacheBytes / c.expertBlob ∧ p.cacheSlots ≠ 0 ∧
    p.vramBudget = pool ∧ p.vramUsed = pool - p.cacheBytes % c.expertBlob := by
  unfold makePlan at h
  split at h
  · cases h
  · rename_i h1
    split at h
    · cases h
    · rename_i h2
      simp only [Except.ok.injEq] at h
      subst h
      exact ⟨by omega, rfl, rfl, rfl, h2, rfl, rfl⟩

/-- The plan closes: KV + fixed costs + cache is exactly the pool. -/
theorem plan_closes {ctx g c pool p} (h : makePlan ctx g c pool = .ok p) :
    p.kvBytes + fixedCost c + p.cacheBytes = p.vramBudget := by
  obtain ⟨h1, hk, hc, -, -, hb, -⟩ := makePlan_ok h
  rw [hk, hc, hb]; omega

/-- The slots fit in the cache bytes. -/
theorem plan_slots_fit {ctx g c pool p} (h : makePlan ctx g c pool = .ok p) :
    p.cacheSlots * c.expertBlob ≤ p.cacheBytes := by
  obtain ⟨-, -, -, hs, -, -, -⟩ := makePlan_ok h
  rw [hs]; exact Nat.div_mul_le_self _ _

/-- A plan always has at least one slot. -/
theorem plan_has_slot {ctx g c pool p} (h : makePlan ctx g c pool = .ok p) : 1 ≤ p.cacheSlots := by
  obtain ⟨-, -, -, -, hs, -, -⟩ := makePlan_ok h
  omega

/-- The unusable slot remainder is less than one blob, and VRAM used never exceeds the pool. -/
theorem plan_remainder {ctx g c pool p} (h : makePlan ctx g c pool = .ok p) (hb : 0 < c.expertBlob) :
    p.cacheBytes % c.expertBlob < c.expertBlob ∧ p.vramUsed ≤ p.vramBudget := by
  obtain ⟨-, -, -, -, -, hbud, hused⟩ := makePlan_ok h
  exact ⟨Nat.mod_lt _ hb, by rw [hused, hbud]; omega⟩

/-- A request that cannot fit is refused at startup, never planned. -/
theorem plan_refuses {ctx : Nat} {g : Geometry} {c : Costs} {pool : Nat}
    (h : pool < kvBytesPerToken g * ctx + fixedCost c) : ∃ e, makePlan ctx g c pool = .error e :=
  ⟨"does not close", by simp [makePlan, h]⟩

/-- More context never buys more cache slots. -/
theorem slots_antitone {ctx1 ctx2 : Nat} {g : Geometry} {c : Costs} {pool : Nat} {p1 p2 : Plan}
    (hctx : ctx1 ≤ ctx2) (h1 : makePlan ctx1 g c pool = .ok p1) (h2 : makePlan ctx2 g c pool = .ok p2) :
    p2.cacheSlots ≤ p1.cacheSlots := by
  obtain ⟨-, -, hc1, hs1, -, -, -⟩ := makePlan_ok h1
  obtain ⟨-, -, hc2, hs2, -, -, -⟩ := makePlan_ok h2
  have hkv : kvBytesPerToken g * ctx1 ≤ kvBytesPerToken g * ctx2 := Nat.mul_le_mul_left _ hctx
  rw [hs1, hs2]
  apply Nat.div_le_div_right
  rw [hc1, hc2]
  omega

/-- Worked example: at 32K context with no other costs, the default model gets 3,989 slots. -/
theorem example_32k :
    (makePlan 32768 {} {}).toOption.map (·.cacheSlots) = some 3989 := by decide

end Strata.VramMemoryPlan
