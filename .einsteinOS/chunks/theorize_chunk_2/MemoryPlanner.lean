/-!
# MODULE: memory_planner
VRAM planner: fixed costs (dense weights, embedding, MTP, workspace, recurrent state) plus KV and indexer
keys at the requested context must fit the pool; whatever remains becomes expert-cache slots.

## Source semantics
- INT8 KV per layer per token = 2*kv_heads*head_dim bytes + one fp16 scale per 64 elements.
- Indexer cache per layer per token = key_heads*key_dim/qsa_block bytes.
- GDN state per layer = S*Hv*S*4 + (d_conv-1)*C*4 bytes; fixed and not evictable.
- Planning refuses (DoesNotClose) if kv+fixed exceeds the pool or if no slot fits.

## Form template identifiers
kv_bytes_per_token -> kvBytesPerToken ; state_bytes -> stateBytes ; make_plan -> makePlan
-/

namespace Strata.Plan

-- § byte costs

/-- KV plus indexer bytes per token, from the geometry. -/
def kvBytesPerToken (nQsa nKv headDim kvGroup keyHeads keyDim qsaBlock : Nat) : Nat :=
  let kvElems := 2 * nKv * headDim
  let scales := (kvElems / kvGroup) * 2
  let perLayer := kvElems + scales
  let idx := keyHeads * keyDim / qsaBlock
  nQsa * (perLayer + idx)

/-- The artifact: 12 layers x (1056 + 32) = 13,056 bytes per token. -/
theorem kv_real : kvBytesPerToken 12 2 256 64 1 128 4 = 13056 := by decide

/-- Recurrent state bytes for `nGdn` layers. -/
def stateBytes (nGdn state vHeads dConv channels : Nat) : Nat :=
  nGdn * (state * vHeads * state * 4 + (dConv - 1) * channels * 4)

/-- The artifact's state is 117,669,888 bytes. -/
theorem state_real : stateBytes 36 128 48 4 10240 = 117669888 := by decide

/-- KV cost is linear in context. -/
theorem kv_total_add (per a b : Nat) : per * (a + b) = per * a + per * b := Nat.mul_add per a b

-- § planning

/-- Plan result: (cache bytes, slots) or refusal. -/
def makePlan (kvPerTok ctx fixed pool blob : Nat) : Option (Nat × Nat) :=
  if kvPerTok * ctx + fixed > pool then none
  else if (pool - kvPerTok * ctx - fixed) / blob = 0 then none
  else some (pool - kvPerTok * ctx - fixed, (pool - kvPerTok * ctx - fixed) / blob)

/-- A successful plan accounts for the whole pool exactly. -/
theorem makePlan_closes (kv ctx fixed pool blob c s : Nat)
    (h : makePlan kv ctx fixed pool blob = some (c, s)) : kv * ctx + fixed + c = pool := by
  unfold makePlan at h
  by_cases h1 : kv * ctx + fixed > pool
  · simp [h1] at h
  · by_cases h2 : (pool - kv * ctx - fixed) / blob = 0
    · simp [h1, h2] at h
    · simp [h1, h2] at h
      obtain ⟨hc, _⟩ := h
      omega

/-- The slots fit in the cache bytes. -/
theorem makePlan_slots_fit (kv ctx fixed pool blob c s : Nat)
    (h : makePlan kv ctx fixed pool blob = some (c, s)) : s * blob ≤ c := by
  unfold makePlan at h
  by_cases h1 : kv * ctx + fixed > pool
  · simp [h1] at h
  · by_cases h2 : (pool - kv * ctx - fixed) / blob = 0
    · simp [h1, h2] at h
    · simp [h1, h2] at h
      obtain ⟨hc, hs⟩ := h
      subst hc
      subst hs
      exact Nat.div_mul_le_self _ _

/-- A successful plan has at least one slot. -/
theorem makePlan_slots_pos (kv ctx fixed pool blob c s : Nat)
    (h : makePlan kv ctx fixed pool blob = some (c, s)) : 0 < s := by
  unfold makePlan at h
  by_cases h1 : kv * ctx + fixed > pool
  · simp [h1] at h
  · by_cases h2 : (pool - kv * ctx - fixed) / blob = 0
    · simp [h1, h2] at h
    · simp [h1, h2] at h
      obtain ⟨_, hs⟩ := h
      omega

/-- Overcommitted requests are refused. -/
theorem makePlan_refuses (kv ctx fixed pool blob : Nat) (h : kv * ctx + fixed > pool) :
    makePlan kv ctx fixed pool blob = none := by
  unfold makePlan
  simp [h]

end Strata.Plan
