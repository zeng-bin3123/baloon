/-!
# MODULE: memory_planner
The VRAM memory planner: KV/indexer bytes per token, the fixed GDN recurrent-state cost, and the division of the
VRAM pool into fixed costs and expert-cache slots. The planner refuses (throws `DoesNotClose`) when the fixed
costs do not fit, and wastes strictly less than one slot.

## Source semantics
- `kv_bytes_per_token = n_qsa * (kv_elems + 2 * kv_elems / group + key_heads * key_dim / block)`
  with `kv_elems = 2 * n_kv_heads * head_dim`.
- `state_bytes = n_gdn * (S * h_v * S * 4 + (d_conv - 1) * C * 4)`.
- `cache_bytes = pool - kv - fixed` when `kv + fixed <= pool`; `slots = cache_bytes / blob`.

## Form template identifiers
- `kvPerToken`, `stateBytes` <- `kv_bytes_per_token`, `state_bytes`
- `cacheBytes`, `cacheSlots` <- `p.cache_bytes`, `p.cache_slots`
- `vramUsed`                 <- `pool - cache_bytes % blob`
-/

namespace Strata.Plan

/-- KV + indexer bytes per token. -/
def kvPerToken (nKv hd grp nQsa keyHeads keyDim block : Nat) : Nat :=
  let kvElems := 2 * nKv * hd
  let perLayer := kvElems + kvElems / grp * 2
  let idx := keyHeads * keyDim / block
  nQsa * (perLayer + idx)

/-- GDN recurrent + conv state bytes per sequence (fp32). -/
def stateBytes (nGdn S hv dconv C : Nat) : Nat :=
  nGdn * (S * hv * S * 4 + (dconv - 1) * C * 4)

/-- Bytes left for the expert cache. -/
def cacheBytes (pool kv fixed : Nat) : Nat := pool - kv - fixed

/-- Number of whole expert slots. -/
def cacheSlots (pool kv fixed blob : Nat) : Nat := cacheBytes pool kv fixed / blob

/-- VRAM actually used (slot remainder is unusable). -/
def vramUsed (pool cache blob : Nat) : Nat := pool - cache % blob

/-- The real geometry costs 13,056 KV + indexer bytes per token. -/
theorem kvPerToken_real : kvPerToken 2 256 64 12 1 128 4 = 13056 := by
  decide

/-- The real geometry's GDN state is 117,669,888 bytes. -/
theorem stateBytes_real : stateBytes 36 128 48 4 10240 = 117669888 := by
  decide

/-- KV cost is linear in the number of QSA layers. -/
theorem kvPerToken_mul (nKv hd grp k keyHeads keyDim block : Nat) :
    kvPerToken nKv hd grp k keyHeads keyDim block =
      k * (2 * nKv * hd + 2 * nKv * hd / grp * 2 + keyHeads * keyDim / block) := by
  simp [kvPerToken]

/-- When the fixed costs fit, the three parts partition the pool exactly. -/
theorem pool_partition (pool kv fixed : Nat) (h : kv + fixed ≤ pool) :
    kv + fixed + cacheBytes pool kv fixed = pool := by
  unfold cacheBytes
  omega

/-- The slots never exceed the bytes left for the cache. -/
theorem slots_fit (pool kv fixed blob : Nat) :
    cacheSlots pool kv fixed blob * blob ≤ cacheBytes pool kv fixed := by
  unfold cacheSlots
  exact Nat.div_mul_le_self _ _

/-- The slots plus the unusable remainder account for every cache byte. -/
theorem slots_remainder (pool kv fixed blob : Nat) :
    cacheSlots pool kv fixed blob * blob + cacheBytes pool kv fixed % blob = cacheBytes pool kv fixed := by
  unfold cacheSlots
  have h := Nat.div_add_mod (cacheBytes pool kv fixed) blob
  rw [Nat.mul_comm] at h
  exact h

/-- The wasted remainder is strictly smaller than one slot. -/
theorem remainder_lt_blob (pool kv fixed blob : Nat) (h : 0 < blob) :
    cacheBytes pool kv fixed % blob < blob :=
  Nat.mod_lt _ h

/-- The reported VRAM use never exceeds the pool. -/
theorem vramUsed_le (pool cache blob : Nat) : vramUsed pool cache blob ≤ pool := by
  unfold vramUsed
  exact Nat.sub_le _ _

/-- A plan that closes always has at least the requested number of bytes for at least one slot when
`cacheSlots` is positive. -/
theorem slot_positive_bytes (pool kv fixed blob : Nat) (h : 0 < cacheSlots pool kv fixed blob) :
    blob ≤ cacheBytes pool kv fixed := by
  have h1 := slots_fit pool kv fixed blob
  have h2 : blob ≤ cacheSlots pool kv fixed blob * blob := by
    have := Nat.mul_le_mul_right blob (show 1 ≤ cacheSlots pool kv fixed blob from h)
    omega
  omega

/-- The planner refuses exactly when the KV and fixed costs exceed the pool. -/
def closes (pool kv fixed : Nat) : Bool := decide (kv + fixed ≤ pool)

/-- A closing plan never overcommits. -/
theorem closes_no_overcommit (pool kv fixed : Nat) (h : closes pool kv fixed = true) :
    kv + fixed ≤ pool := by
  simpa [closes] using h

end Strata.Plan
