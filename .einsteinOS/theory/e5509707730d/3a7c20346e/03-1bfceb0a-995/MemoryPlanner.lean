/-!
# MODULE: memory_planner
SOURCES: include/strata/plan/plan.hpp
Derives, from the model geometry alone, how much VRAM the KV cache, indexer keys and GDN state take, and how
many expert-cache slots remain; refuses (DoesNotClose) rather than returning a plan it cannot honour.

## Source semantics
* GDN state is fixed and not evictable: `n_gdn * (S*h_v*S*4 + (d_conv-1)*C*4)` bytes.
* KV bytes per token: `n_qsa * (INT8 K+V elements + fp16 group scales + indexer key bytes)`, where the indexer
  caches ONE shared key head (`key_heads * key_dim / block`), not one per query head.
* `make_plan`: reject if `kv + fixed > pool`; the cache gets `pool - kv - fixed`; slots are whole blobs; reject
  if there is not even one slot; the slot remainder is unusable and counted as used.

## Form template identifiers
stateBytes -> `state_bytes(Geometry)`
kvBytesPerToken -> `kv_bytes_per_token(Geometry)`
makePlan -> `make_plan` (None = throws `DoesNotClose`)
-/

namespace Strata.Plan

-- § geometry

/-- Model geometry fields used by the byte costs. -/
structure Geometry where
  nLayers : Nat
  nQsa : Nat
  nKvHeads : Nat
  headDim : Nat
  kvGroup : Nat
  qsaBlock : Nat
  idxKeyHeads : Nat
  idxKeyDim : Nat
  ssmState : Nat
  ssmVHeads : Nat
  ssmDConv : Nat
  ssmConvChannels : Nat

/-- The shipped artifact's geometry (the defaults of `plan::Geometry`). -/
def defaultG : Geometry :=
  { nLayers := 48, nQsa := 12, nKvHeads := 2, headDim := 256, kvGroup := 64, qsaBlock := 4,
    idxKeyHeads := 1, idxKeyDim := 128, ssmState := 128, ssmVHeads := 48, ssmDConv := 4,
    ssmConvChannels := 10240 }

/-- Number of GDN layers: derived, not a third constant. -/
def Geometry.nGdn (g : Geometry) : Nat := g.nLayers - g.nQsa

/-- GDN recurrence + conv history bytes per sequence. -/
def stateBytes (g : Geometry) : Nat :=
  g.nGdn * (g.ssmState * g.ssmVHeads * g.ssmState * 4 + (g.ssmDConv - 1) * g.ssmConvChannels * 4)

/-- INT8 KV plus indexer-key bytes per token. -/
def kvBytesPerToken (g : Geometry) : Nat :=
  g.nQsa * ((2 * g.nKvHeads * g.headDim + (2 * g.nKvHeads * g.headDim) / g.kvGroup * 2) +
    g.idxKeyHeads * g.idxKeyDim / g.qsaBlock)

-- § concrete values of the shipped model

/-- 36 of the 48 layers are GDN. -/
theorem default_n_gdn : defaultG.nGdn = 36 := by decide

/-- The GDN state is 117,669,888 bytes. -/
theorem default_state_bytes : stateBytes defaultG = 117669888 := by decide

/-- The GDN state costs 85 whole expert slots (1,382,400 B each), the figure the source quotes. -/
theorem default_state_is_85_slots : stateBytes defaultG / 1382400 = 85 := by decide

/-- KV plus indexer bytes per token: 13,056 (12 layers x (1024 + 32 + 32)). -/
theorem default_kv_per_token : kvBytesPerToken defaultG = 13056 := by decide

-- § planning

/-- The result of a successful plan (the numeric fields of `plan::Plan`). -/
structure Plan where
  kvBytes : Nat
  stateBytes : Nat
  cacheBytes : Nat
  cacheSlots : Nat
  vramUsed : Nat

/-- `make_plan`: `none` models `DoesNotClose`. `fixed` is dense + embedding + MTP + workspace + state. -/
def makePlan (ctx pool blob kvPerTok fixed state : Nat) : Option Plan :=
  if kvPerTok * ctx + fixed > pool then none
  else if (pool - kvPerTok * ctx - fixed) / blob = 0 then none
  else some { kvBytes := kvPerTok * ctx, stateBytes := state,
              cacheBytes := pool - kvPerTok * ctx - fixed,
              cacheSlots := (pool - kvPerTok * ctx - fixed) / blob,
              vramUsed := pool - (pool - kvPerTok * ctx - fixed) % blob }

/-- A plan whose fixed costs plus KV exceed the pool is refused. -/
theorem makePlan_refuses_overflow (ctx pool blob kvPerTok fixed state : Nat)
    (h : kvPerTok * ctx + fixed > pool) : makePlan ctx pool blob kvPerTok fixed state = none := by
  unfold makePlan
  simp [h]

/-- A plan with no room for even one slot is refused. -/
theorem makePlan_refuses_no_slot (ctx pool blob kvPerTok fixed state : Nat)
    (h1 : ¬ kvPerTok * ctx + fixed > pool) (h2 : (pool - kvPerTok * ctx - fixed) / blob = 0) :
    makePlan ctx pool blob kvPerTok fixed state = none := by
  unfold makePlan
  simp [h1, h2]

/-- Every accepted plan exactly splits the pool into KV, fixed costs and cache. -/
theorem makePlan_budget {ctx pool blob kvPerTok fixed state : Nat} {p : Plan}
    (h : makePlan ctx pool blob kvPerTok fixed state = some p) :
    p.kvBytes + fixed + p.cacheBytes = pool := by
  unfold makePlan at h
  by_cases c1 : kvPerTok * ctx + fixed > pool
  · simp [c1] at h
  · by_cases c2 : (pool - kvPerTok * ctx - fixed) / blob = 0
    · simp [c1, c2] at h
    · simp [c1, c2] at h
      subst h
      simp
      omega

/-- Every accepted plan has at least one expert slot. -/
theorem makePlan_slots_pos {ctx pool blob kvPerTok fixed state : Nat} {p : Plan}
    (h : makePlan ctx pool blob kvPerTok fixed state = some p) : 0 < p.cacheSlots := by
  unfold makePlan at h
  by_cases c1 : kvPerTok * ctx + fixed > pool
  · simp [c1] at h
  · by_cases c2 : (pool - kvPerTok * ctx - fixed) / blob = 0
    · simp [c1, c2] at h
    · simp [c1, c2] at h
      subst h
      exact Nat.pos_of_ne_zero c2

/-- The slots never take more bytes than the cache budget. -/
theorem makePlan_slots_fit {ctx pool blob kvPerTok fixed state : Nat} {p : Plan}
    (h : makePlan ctx pool blob kvPerTok fixed state = some p) :
    p.cacheSlots * blob ≤ p.cacheBytes := by
  unfold makePlan at h
  by_cases c1 : kvPerTok * ctx + fixed > pool
  · simp [c1] at h
  · by_cases c2 : (pool - kvPerTok * ctx - fixed) / blob = 0
    · simp [c1, c2] at h
    · simp [c1, c2] at h
      subst h
      simp
      exact Nat.div_mul_le_self _ _

/-- Whole slots plus the unusable remainder account for the entire cache budget. -/
theorem slots_remainder_split (cache blob : Nat) : (cache / blob) * blob + cache % blob = cache := by
  have := Nat.div_add_mod cache blob
  rw [Nat.mul_comm] at this
  exact this

/-- The reported VRAM use never exceeds the pool. -/
theorem makePlan_used_le {ctx pool blob kvPerTok fixed state : Nat} {p : Plan}
    (h : makePlan ctx pool blob kvPerTok fixed state = some p) : p.vramUsed ≤ pool := by
  unfold makePlan at h
  by_cases c1 : kvPerTok * ctx + fixed > pool
  · simp [c1] at h
  · by_cases c2 : (pool - kvPerTok * ctx - fixed) / blob = 0
    · simp [c1, c2] at h
    · simp [c1, c2] at h
      subst h
      simp <;> omega

-- § monotonicity in the context length

/-- KV bytes grow with the requested context. -/
theorem kv_mono (kvPerTok c1 c2 : Nat) (h : c1 ≤ c2) : kvPerTok * c1 ≤ kvPerTok * c2 :=
  Nat.mul_le_mul_left kvPerTok h

/-- A context that does not close stays unclosable at any larger context. -/
theorem overflow_persists (kvPerTok c1 c2 pool fixed : Nat) (h : c1 ≤ c2)
    (hov : kvPerTok * c1 + fixed > pool) : kvPerTok * c2 + fixed > pool := by
  have := kv_mono kvPerTok c1 c2 h
  omega

/-- The VRAM pool constant of the README table. -/
def vramPool : Nat := 5943000000

end Strata.Plan
