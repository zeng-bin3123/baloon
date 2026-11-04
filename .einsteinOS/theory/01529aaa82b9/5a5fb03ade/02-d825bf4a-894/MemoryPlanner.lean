/-!
# MODULE: memory_planner
The VRAM memory planner: derives KV, indexer-key and recurrent-state costs from the model geometry and hands the
remainder of a fixed pool to expert-cache slots, refusing rather than overcommitting.

## Source semantics
* Recurrent state is `n_gdn * (S*HV*S*4 + (d_conv-1)*C*4)` bytes = 117,669,888 for the artifact (36 GDN layers).
* KV per token is `n_qsa * (INT8 K and V + fp16 group scales + one 128-wide indexer key per 4-token block)`
  = 13,056 bytes per token for the artifact.
* `make_plan` fails if `kv + fixed > pool` or if not even one slot fits; otherwise
  `kv + fixed + cache = pool`, `slots = cache / blob`, and `slots * blob <= cache`.
* A larger context never leaves more room for the expert cache.

## Form template identifiers
* `state_bytes` -> `stateBytes`, `kv_bytes_per_token` -> `kvBytesPerToken`, `make_plan` -> `makePlan`
* `Plan::cache_slots` -> `Plan.slots`, `Costs::expert_blob` -> `blob`, `DoesNotClose` -> `none`
-/

namespace Strata.MemoryPlanner

-- § geometry costs

/-- Recurrent state bytes per GDN layer: recurrence `128*48*128*4` plus conv history `3*10240*4`. -/
def stateBytesPerLayer : Nat := 128 * 48 * 128 * 4 + 3 * 10240 * 4

/-- Recurrent state across the 36 GDN layers of the artifact (48 layers minus 12 QSA). -/
def stateBytes : Nat := (48 - 12) * stateBytesPerLayer

/-- The planner's state term, measured in the source comments as 117.7 MB. -/
theorem stateBytes_value : stateBytesPerLayer = 3268608 ∧ stateBytes = 117669888 := by
  unfold stateBytes stateBytesPerLayer
  decide

/-- Slots lost to the state term at the 1,382,400-byte blob size: the recorded 85 slots. -/
theorem state_costs_85_slots : 117669888 / 1382400 = 85 := by decide

/-- KV bytes per token per QSA layer: INT8 K and V (`2*2*256`) plus fp16 scales (one per 64 elements). -/
def kvLayer : Nat := 2 * 2 * 256 + (2 * 2 * 256 / 64) * 2

/-- Indexer key bytes per token per layer: one 128-wide key head pooled per 4 tokens. -/
def idxLayer : Nat := 1 * 128 / 4

/-- KV plus indexer bytes per token over the 12 QSA layers. -/
def kvBytesPerToken : Nat := 12 * (kvLayer + idxLayer)

/-- The per-token cost of the artifact: 1056 + 32 bytes per layer, 13,056 over 12 layers. -/
theorem kvBytesPerToken_value : kvLayer = 1056 ∧ idxLayer = 32 ∧ kvBytesPerToken = 13056 := by
  unfold kvBytesPerToken kvLayer idxLayer
  decide

/-- At 32K tokens the KV and indexer cache is 427,819,008 bytes. -/
theorem kv_at_32k : kvBytesPerToken * 32768 = 427819008 := by
  unfold kvBytesPerToken kvLayer idxLayer
  decide

/-- KV cost is monotone in the context length. -/
theorem kv_monotone (a b : Nat) (h : a ≤ b) : kvBytesPerToken * a ≤ kvBytesPerToken * b :=
  Nat.mul_le_mul_left _ h

-- § the plan

/-- A computed plan. -/
structure Plan where
  kv : Nat
  cache : Nat
  slots : Nat
  used : Nat

/-- `make_plan`: `none` is the `DoesNotClose` exception. -/
def makePlan (pool kvPerTok ctx fixed blob : Nat) : Option Plan :=
  if pool < kvPerTok * ctx + fixed then none
  else if (pool - kvPerTok * ctx - fixed) / blob = 0 then none
  else some { kv := kvPerTok * ctx, cache := pool - kvPerTok * ctx - fixed,
              slots := (pool - kvPerTok * ctx - fixed) / blob,
              used := pool - (pool - kvPerTok * ctx - fixed) % blob }

/-- A successful plan accounts for the whole pool: KV, fixed costs and the cache add up exactly. -/
theorem makePlan_closes (pool kvPerTok ctx fixed blob : Nat) (p : Plan)
    (h : makePlan pool kvPerTok ctx fixed blob = some p) :
    p.kv + fixed + p.cache = pool := by
  unfold makePlan at h
  by_cases h1 : pool < kvPerTok * ctx + fixed
  · simp [h1] at h
  · by_cases h2 : (pool - kvPerTok * ctx - fixed) / blob = 0
    · simp [h1, h2] at h
    · simp [h1, h2] at h
      subst h
      try dsimp only
      omega

/-- A successful plan has at least one slot. -/
theorem makePlan_has_slot (pool kvPerTok ctx fixed blob : Nat) (p : Plan)
    (h : makePlan pool kvPerTok ctx fixed blob = some p) : 0 < p.slots := by
  unfold makePlan at h
  by_cases h1 : pool < kvPerTok * ctx + fixed
  · simp [h1] at h
  · by_cases h2 : (pool - kvPerTok * ctx - fixed) / blob = 0
    · simp [h1, h2] at h
    · simp [h1, h2] at h
      subst h
      try dsimp only
      exact Nat.pos_of_ne_zero h2

/-- The slots never take more bytes than the cache budget. -/
theorem makePlan_slots_fit (pool kvPerTok ctx fixed blob : Nat) (p : Plan)
    (h : makePlan pool kvPerTok ctx fixed blob = some p) : p.slots * blob ≤ p.cache := by
  unfold makePlan at h
  by_cases h1 : pool < kvPerTok * ctx + fixed
  · simp [h1] at h
  · by_cases h2 : (pool - kvPerTok * ctx - fixed) / blob = 0
    · simp [h1, h2] at h
    · simp [h1, h2] at h
      subst h
      exact Nat.div_mul_le_self _ _

/-- The plan's VRAM use never exceeds the pool. -/
theorem makePlan_used_le (pool kvPerTok ctx fixed blob : Nat) (p : Plan)
    (h : makePlan pool kvPerTok ctx fixed blob = some p) : p.used ≤ pool := by
  unfold makePlan at h
  by_cases h1 : pool < kvPerTok * ctx + fixed
  · simp [h1] at h
  · by_cases h2 : (pool - kvPerTok * ctx - fixed) / blob = 0
    · simp [h1, h2] at h
    · simp [h1, h2] at h
      subst h
      try dsimp only
      omega

/-- The plan refuses instead of overcommitting: if KV plus fixed costs exceed the pool there is no plan. -/
theorem makePlan_refuses (pool kvPerTok ctx fixed blob : Nat) (h : pool < kvPerTok * ctx + fixed) :
    makePlan pool kvPerTok ctx fixed blob = none := by
  simp [makePlan, h]

/-- A longer context leaves no more cache room than a shorter one. -/
theorem cache_room_antitone (pool kvPerTok fixed a b : Nat) (h : a ≤ b) :
    pool - kvPerTok * b - fixed ≤ pool - kvPerTok * a - fixed := by
  have := Nat.mul_le_mul_left kvPerTok h
  omega

end Strata.MemoryPlanner
