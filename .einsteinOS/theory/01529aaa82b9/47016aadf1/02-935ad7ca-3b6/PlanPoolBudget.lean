/-!
# MODULE: plan_pool_budget
Arithmetic facts of `include/strata/plan/plan.hpp` beyond the planner's refusal logic: the indexer-key
derivation history, KV byte decomposition, the GDN state in expert slots, the unusable slot remainder and
monotonicity of the expert cache in the context length.

## Source semantics
The cached indexer key is ONE 128-wide key per 4-token block (32 bytes/token/layer); counting query heads
(4) or the wrong width (256) over-counts it. The KV scale term is exact because 1024 is a multiple of the
64-element group. `vram_used = pool - cache % blob`, i.e. the unusable remainder is excluded and the used
part is `kv + fixed + slots * blob`. A longer context never yields more expert cache slots.

## Form template identifiers
- `kv`, `fixed`, `pool`, `blob` <- p.kv_bytes, fixed sum, pool, c.expert_blob
- `slots` <- `(pool - kv - fixed) / blob`
-/

namespace Strata.PlanPoolBudget

-- § derivation checks

/-- The correct indexer term: 1 key head x 128 / 4-token blocks = 32 bytes per token per layer. -/
theorem indexer_correct : 1 * 128 / 4 = 32 := by decide

/-- Counting the 4 query heads over-counts the cached key 4x. -/
theorem indexer_round194_overcount : 4 * 128 / 4 = 4 * (1 * 128 / 4) := by decide

/-- The original 256-wide formula over-counted 2x. -/
theorem indexer_original_overcount : 1 * 256 / 4 = 2 * (1 * 128 / 4) := by decide

/-- The 1024 K+V elements per layer divide exactly into 64-element scale groups. -/
theorem kv_group_exact : 1024 % 64 = 0 := by decide

/-- One fp16 scale per group: 16 groups x 2 bytes = 32 bytes. -/
theorem kv_scale_bytes : 1024 / 64 * 2 = 32 := by decide

/-- Per layer: 1024 INT8 + 32 scale + 32 indexer = 1088 bytes; 12 QSA layers = 13056 per token. -/
theorem kv_layer_decomposition : 1024 + 32 + 32 = 1088 ∧ 12 * 1088 = 13056 := by decide

/-- The GDN state is worth 85 whole expert cache slots (117,669,888 / 1,382,400). -/
theorem gdn_state_in_slots : 117669888 / 1382400 = 85 := by decide

-- § scaling and monotonicity

/-- Doubling the context doubles the KV bytes. -/
theorem kv_doubles (kvT a : Nat) : kvT * (2 * a) = 2 * (kvT * a) := Nat.mul_left_comm kvT 2 a

/-- A longer context never yields more expert cache slots. -/
theorem slots_antitone (pool kvT fixed blob a b : Nat) (hab : a ≤ b) :
    (pool - kvT * b - fixed) / blob ≤ (pool - kvT * a - fixed) / blob := by
  have h := Nat.mul_le_mul_left kvT hab
  exact Nat.div_le_div_right (by omega)

-- § remainder accounting

/-- Reported use equals KV + fixed + whole slots: only the sub-slot remainder is left out. -/
theorem used_eq (pool kv fixed blob : Nat) (h : kv + fixed ≤ pool) :
    pool - (pool - kv - fixed) % blob = kv + fixed + (pool - kv - fixed) / blob * blob := by
  have h1 := Nat.div_add_mod (pool - kv - fixed) blob
  have h2 : blob * ((pool - kv - fixed) / blob) = (pool - kv - fixed) / blob * blob := Nat.mul_comm _ _
  omega

/-- The unusable remainder is strictly smaller than one expert slot. -/
theorem leftover_lt (pool kv fixed blob : Nat) (hb : 0 < blob) : (pool - kv - fixed) % blob < blob :=
  Nat.mod_lt _ hb

/-- A pool that exactly fits `kv + fixed` plus `n` slots yields at least `n` slots. -/
theorem slots_at_least (pool kv fixed blob n : Nat) (hb : 0 < blob) (h : kv + fixed + n * blob ≤ pool) :
    n ≤ (pool - kv - fixed) / blob := by
  apply (Nat.le_div_iff_mul_le hb).2
  omega

end Strata.PlanPoolBudget
