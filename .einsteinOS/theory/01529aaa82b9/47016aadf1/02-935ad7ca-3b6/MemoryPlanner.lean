/-!
# MODULE: memory_planner
The VRAM planner: derives KV, indexer-key and GDN-state byte costs from model geometry, subtracts fixed
costs from the shared pool, and either returns an expert-cache plan that fits or refuses.

## Source semantics
`state_bytes` is a fixed, non-evictable cost; `kv_bytes_per_token` is per token over the QSA layers only,
with one cached indexer key (128 wide) per 4-token block. A plan must satisfy
`kv + fixed + slots * blob <= pool` and must have at least one slot; otherwise it is refused.

## Form template identifiers
- `stateBytes`, `kvBytesPerToken` <- plan::state_bytes, plan::kv_bytes_per_token
- `makePlan` <- plan::make_plan (None = DoesNotClose)
- `kvT`, `ctx`, `fixed`, `blob`, `pool` <- kv_bytes_per_token(g), max_context, fixed sum, expert_blob, pool
-/

namespace Strata.MemoryPlanner

-- § geometry

structure Geometry where
  nLayers : Nat := 48
  nQsa : Nat := 12
  nKvHeads : Nat := 2
  headDim : Nat := 256
  kvGroup : Nat := 64
  qsaBlock : Nat := 4
  idxKeyHeads : Nat := 1
  idxKeyDim : Nat := 128
  ssmState : Nat := 128
  ssmVHeads : Nat := 48
  ssmConv : Nat := 4
  ssmChannels : Nat := 10240

def Geometry.nGdn (g : Geometry) : Nat := g.nLayers - g.nQsa

def stateBytes (g : Geometry) : Nat :=
  g.nGdn * (g.ssmState * g.ssmVHeads * g.ssmState * 4 + (g.ssmConv - 1) * g.ssmChannels * 4)

def kvBytesPerToken (g : Geometry) : Nat :=
  let kvElems := 2 * g.nKvHeads * g.headDim
  let kvScales := (kvElems / g.kvGroup) * 2
  let perLayer := kvElems + kvScales
  let idxLayer := g.idxKeyHeads * g.idxKeyDim / g.qsaBlock
  g.nQsa * (perLayer + idxLayer)

/-- 36 of the 48 layers are GDN. -/
theorem gdn_layer_count : ({} : Geometry).nGdn = 36 := by decide

/-- The GDN recurrence+conv state is 117,669,888 bytes (about 85 expert cache slots). -/
theorem state_bytes_value : stateBytes {} = 117669888 := by decide

/-- KV + indexer cost is 13,056 bytes per token (1056 KV + 32 indexer, over 12 QSA layers). -/
theorem kv_bytes_per_token_value : kvBytesPerToken {} = 13056 := by decide

/-- At a 32K context the KV store is 427,819,008 bytes. -/
theorem kv_bytes_at_32k : kvBytesPerToken {} * 32768 = 427819008 := by decide

/-- The expert blob (codes + scales) is 1,382,400 bytes. -/
theorem expert_blob_value : 1280 * 640 + 2560 * 160 + 1280 * 40 * 2 + 2560 * 10 * 2 = 1382400 := by decide

/-- KV memory grows monotonically with context length. -/
theorem kv_monotone (kvT a b : Nat) (h : a ≤ b) : kvT * a ≤ kvT * b := Nat.mul_le_mul_left kvT h

-- § planning

structure Plan where
  kv : Nat
  state : Nat
  cache : Nat
  slots : Nat
  used : Nat

/-- Planner: refuse when fixed costs do not fit, or no whole expert slot remains. -/
def makePlan (pool kvT ctx fixed blob : Nat) : Option Plan :=
  if kvT * ctx + fixed > pool then none
  else if (pool - kvT * ctx - fixed) / blob = 0 then none
  else some { kv := kvT * ctx, state := 0, cache := pool - kvT * ctx - fixed,
              slots := (pool - kvT * ctx - fixed) / blob,
              used := pool - (pool - kvT * ctx - fixed) % blob }

/-- A returned plan never overcommits the pool. -/
theorem plan_fits (pool kvT ctx fixed blob : Nat) (p : Plan)
    (h : makePlan pool kvT ctx fixed blob = some p) :
    kvT * ctx + fixed + p.slots * blob ≤ pool := by
  unfold makePlan at h
  split at h
  · simp at h
  · split at h
    · simp at h
    · rename_i h1 h2
      injection h with hp
      subst hp
      show kvT * ctx + fixed + ((pool - kvT * ctx - fixed) / blob) * blob ≤ pool
      have := Nat.div_mul_le_self (pool - kvT * ctx - fixed) blob
      omega

/-- A returned plan always has at least one expert cache slot. -/
theorem plan_has_slot (pool kvT ctx fixed blob : Nat) (p : Plan)
    (h : makePlan pool kvT ctx fixed blob = some p) : 0 < p.slots := by
  unfold makePlan at h
  split at h
  · simp at h
  · split at h
    · simp at h
    · rename_i h1 h2
      injection h with hp
      subst hp
      show 0 < (pool - kvT * ctx - fixed) / blob
      exact Nat.pos_of_ne_zero h2

/-- The reported VRAM use never exceeds the pool. -/
theorem plan_used_le (pool kvT ctx fixed blob : Nat) (p : Plan)
    (h : makePlan pool kvT ctx fixed blob = some p) : p.used ≤ pool := by
  unfold makePlan at h
  split at h
  · simp at h
  · split at h
    · simp at h
    · rename_i h1 h2
      injection h with hp
      subst hp
      show pool - (pool - kvT * ctx - fixed) % blob ≤ pool
      omega

/-- If the fixed costs alone exceed the pool the planner refuses (fails at startup, not at token 4000). -/
theorem plan_refuses_overcommit (pool kvT ctx fixed blob : Nat) (h : kvT * ctx + fixed > pool) :
    makePlan pool kvT ctx fixed blob = none := by
  unfold makePlan
  rw [if_pos h]

end Strata.MemoryPlanner
